"""Condition B harness: the model solves the cube with the method prompt and
two tools (turn, apply_algorithm), via OpenRouter's OpenAI-compatible API.

An `llm` is any callable llm(messages, tools) -> dict:
    {"content": str | None,
     "tool_calls": [{"id": str, "name": str, "arguments": str (JSON)}],
     "usage": {"input_tokens": int, "output_tokens": int, "cost": float | None}}
OpenRouterLLM is the real one; tests pass in fakes.
"""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from algorithms import ALGORITHMS, FRONT_MAPS, parse, relabel
from cube import MOVE_NAMES, CubeEnv, is_solved, observation
from stages import STAGES, stages_complete

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
METHOD_PROMPT = ROOT / "prompts" / "method.md"
RAW_PROMPT = ROOT / "prompts" / "raw.md"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS_URL = OPENROUTER_BASE_URL + "/models"
MAX_TIMES = 6
# Default thinking setting: reasoning effort "low", output cap 8000 (reasoning
# included). A token budget for reasoning (reasoning_max_tokens) is still
# available, but not the default; it can't be combined with an effort.
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_REASONING_MAX_TOKENS = None
DEFAULT_MAX_OUTPUT_TOKENS = 8000
REASONING_EFFORTS = ("low", "none")  # offered on the command line, plus "unset"
YELLOW_UP_LABEL = "yellow stickers facing up"
NO_TOOL_CALL_LIMIT = 3  # replies in a row without a tool call that end an episode
NO_TOOL_CALL_LIMIT_WORDS = "Three"  # as written in prompts/method.md
NO_TOOL_CALL_NUDGE = "Your last reply had no tool call. Reply with exactly one tool call."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": "Turn one face a quarter turn.",
            "parameters": {
                "type": "object",
                "properties": {"move": {"type": "string", "enum": list(MOVE_NAMES)}},
                "required": ["move"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_algorithm",
            "description": "Perform a named algorithm as if `front` faces you, `times` times in a row.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(ALGORITHMS)},
                    "front": {"type": "string", "enum": list(FRONT_MAPS)},
                    "times": {"type": "integer", "minimum": 1, "maximum": MAX_TIMES},
                },
                "required": ["name", "front"],
            },
        },
    },
]


# --- the real model -------------------------------------------------------


class DeadlineExceeded(Exception):
    pass


def call_with_deadline(fn, seconds):
    """Run fn() but give up after `seconds` of wall-clock time.

    The openai client's `timeout` is passed to its HTTP library (httpx2), where it limits each
    connect/read/write step, not the whole request: a server that keeps
    sending bytes (e.g. keep-alive whitespace while the model thinks) never
    trips it. This is a hard limit. On timeout the worker thread is abandoned
    (it is a daemon, so it cannot keep the process alive).
    """
    result = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # re-raised in the caller's thread
            result["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(max(seconds, 0))
    if worker.is_alive():
        raise DeadlineExceeded(f"no response within {seconds:.0f} s")
    if "error" in result:
        raise result["error"]
    return result["value"]


class OpenRouterLLM:
    """Calls OpenRouter with a hard 60 s wall-clock timeout per attempt and up
    to 3 retries with backoff.

    The API key is read from .env / the environment and only ever passed to
    the client; it is never stored in logs or returned.

    A response that arrives but carries an error (no choices, or an error
    payload) is returned as {"error": ...} rather than raised, so the harness
    counts it as a decision.

    Settings sent with every request:
      max_output_tokens -> `max_tokens` (covers reasoning + visible output)
      reasoning_effort -> `reasoning: {"effort": ...}` (default "low"), or
      reasoning_max_tokens -> `reasoning: {"max_tokens": N}` (`{"effort":
                              "none"}` when 0), used only with effort None.
      None for both sends no reasoning setting.
    If the provider rejects the reasoning setting (HTTP 400), the request is
    resent without it and the reply's reasoning_status says "rejected: ...".
    """

    def __init__(self, model, client=None, timeout=60, retries=3, backoff=2.0,
                 reasoning_max_tokens=DEFAULT_REASONING_MAX_TOKENS,
                 max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                 reasoning_effort=DEFAULT_REASONING_EFFORT):
        check_budgets(reasoning_max_tokens, max_output_tokens, reasoning_effort)
        self.reasoning_effort = reasoning_effort
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.reasoning_max_tokens = reasoning_max_tokens
        self.max_output_tokens = max_output_tokens
        if client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv(ROOT / ".env")
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
                max_retries=0,  # retries are handled below
            )
        self.client = client

    def __call__(self, messages, tools):
        import openai

        retryable = (openai.APIConnectionError, openai.APITimeoutError,
                     openai.RateLimitError, openai.InternalServerError, DeadlineExceeded)
        extra = {"tools": tools, "parallel_tool_calls": False} if tools else {}
        if self.max_output_tokens is not None:
            extra["max_tokens"] = self.max_output_tokens
        body = {}
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        elif self.reasoning_max_tokens is not None:
            body["reasoning"] = ({"effort": "none"} if self.reasoning_max_tokens == 0
                                 else {"max_tokens": self.reasoning_max_tokens})
        rejected = None

        def request():
            return self.client.chat.completions.create(
                model=self.model, messages=messages, timeout=self.timeout,
                extra_body=dict(body) or None, **extra,
            )

        attempt = 0
        while True:
            try:
                resp = call_with_deadline(request, self.timeout)
                break
            except openai.BadRequestError as exc:
                if "reasoning" not in body:
                    raise
                # Provider refused the reasoning setting: log it, resend without.
                rejected = f"rejected: {str(exc)[:200]}"
                body.pop("reasoning")
            except retryable:
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff * 2 ** attempt)
                attempt += 1

        usage = getattr(resp, "usage", None)
        reasoning_tokens = getattr(getattr(usage, "completion_tokens_details", None),
                                   "reasoning_tokens", None)
        usage_dict = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "reasoning_tokens": reasoning_tokens,
            # OpenRouter reports cost in credits (1 credit = 1 USD).
            "cost": getattr(usage, "cost", None),
        } if usage is not None else None
        status = rejected or reasoning_status(self.reasoning_max_tokens, reasoning_tokens,
                                              self.reasoning_effort)

        choices = getattr(resp, "choices", None)
        error = getattr(resp, "error", None)
        if error or not choices or getattr(choices[0], "message", None) is None:
            return {"error": str(error or "response had no message"), "content": None,
                    "tool_calls": [], "usage": usage_dict, "reasoning_status": status}

        msg = choices[0].message
        return {
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
            "usage": usage_dict,
            "finish_reason": getattr(choices[0], "finish_reason", None),
            "reasoning_status": status,
        }


def reasoning_status(budget, used, effort=None):
    """How the reasoning setting fared, judged from the reported token count."""
    if effort is not None:
        if used is None:
            return "not_reported"
        if effort == "none" and used > 0:
            return f"ignored: used {used} with effort none"
        return f"ok (effort {effort})"
    if budget is None:
        return "not_sent"
    if used is None:
        return "not_reported"
    if used > budget:
        return f"ignored: used {used} > budget {budget}"
    return "ok"


def check_budgets(reasoning_max_tokens, max_output_tokens, reasoning_effort=None):
    """Effort and a reasoning token budget are alternatives (OpenRouter takes
    one or the other). max_tokens covers reasoning too, and Anthropic requires
    it to be strictly larger than the reasoning budget."""
    if reasoning_effort is not None and reasoning_max_tokens is not None:
        raise ValueError("set either reasoning_effort or reasoning_max_tokens, not both "
                         f"(got {reasoning_effort!r} and {reasoning_max_tokens!r})")
    if (reasoning_max_tokens is not None and max_output_tokens is not None
            and reasoning_max_tokens >= max_output_tokens):
        raise ValueError(f"max_output_tokens ({max_output_tokens}) must be larger than "
                         f"reasoning_max_tokens ({reasoning_max_tokens})")


def reasoning_fields(reply, reasoning_max_tokens, max_output_tokens, reasoning_effort):
    """Per-call log fields: reasoning tokens apart from other output, the
    budgets in force, and whether the reasoning setting took effect."""
    usage = reply.get("usage") or {}
    reasoning = usage.get("reasoning_tokens")
    output = usage.get("output_tokens", 0)
    return {
        "reasoning_tokens": reasoning,
        "visible_output_tokens": output - (reasoning or 0),
        "reasoning_status": reply.get("reasoning_status"),
        "finish_reason": reply.get("finish_reason"),
        "reasoning_effort": reasoning_effort,
        "reasoning_max_tokens": reasoning_max_tokens,
        "max_output_tokens": max_output_tokens,
    }


_PRICES = {}


def price_cost(model, input_tokens, output_tokens):
    """Fallback cost: token counts times the model's listed per-token price."""
    if not _PRICES:
        import requests

        for m in requests.get(MODELS_URL, timeout=30).json()["data"]:
            p = m.get("pricing") or {}
            _PRICES[m["id"]] = (float(p.get("prompt") or 0), float(p.get("completion") or 0))
    if model not in _PRICES:
        return None
    price_in, price_out = _PRICES[model]
    return input_tokens * price_in + output_tokens * price_out


def reply_cost(model, reply):
    """(input_tokens, output_tokens, cost_usd, cost_source) for one reply."""
    usage = reply.get("usage")
    if not usage:
        return 0, 0, 0.0, "none"
    in_tok, out_tok = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    if usage.get("cost") is not None:
        return in_tok, out_tok, usage["cost"], "api"
    return in_tok, out_tok, price_cost(model, in_tok, out_tok), "price_table"


# --- tool execution -------------------------------------------------------


def execute_tool(env, name, arguments):
    """Run one tool call. Returns (valid, quarter_turns, message); never raises."""
    try:
        args = json.loads(arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError
    except ValueError:
        return False, 0, f"Error: arguments are not a JSON object: {arguments!r}"

    if name == "turn":
        move = args.get("move")
        if move not in MOVE_NAMES:
            return False, 0, f"Error: invalid move {move!r}. Valid moves: {' '.join(MOVE_NAMES)}"
        env.step(move)
        return True, 1, f"OK: turned {move}."

    if name == "apply_algorithm":
        alg, front, times = args.get("name"), args.get("front"), args.get("times", 1)
        if alg not in ALGORITHMS:
            return False, 0, f"Error: unknown algorithm {alg!r}. Valid: {', '.join(ALGORITHMS)}"
        if front not in FRONT_MAPS:
            return False, 0, f"Error: unknown front {front!r}. Valid: {', '.join(FRONT_MAPS)}"
        if not isinstance(times, int) or isinstance(times, bool) or not 1 <= times <= MAX_TIMES:
            return False, 0, f"Error: times must be an integer from 1 to {MAX_TIMES}, got {times!r}"
        moves = relabel(parse(ALGORITHMS[alg]), front) * times
        for move in moves:
            env.step(move)
        return True, len(moves), f"OK: applied {alg} with front={front}, {times} time(s)."

    return False, 0, f"Error: unknown tool {name!r}. Tools: turn, apply_algorithm"


def stage_line(state):
    n = stages_complete(state)
    names = ", ".join(name for name, _ in STAGES[:n]) or "none"
    return f"stages complete: {n}/7 ({names})"


def yellow_up_line(state):
    """Stage 7 cue: how many U-face stickers are yellow (9 when every top
    corner and edge has yellow facing up)."""
    yellow = state[4]  # U centre
    return f"{YELLOW_UP_LABEL}: {sum(state[i] == yellow for i in range(9))}/9"


def state_text(state):
    return f"{observation(state)}\n\n{stage_line(state)}"


_NEGATION = re.compile(r"\b(not|isn't|unsolved|cannot|can't|unable|fail\w*)\b", re.I)


def claims_solved(text):
    """Heuristic: mentions 'solved' without a nearby negation."""
    return bool(text and re.search(r"\bsolved\b", text, re.I) and not _NEGATION.search(text))


# --- episode loop ---------------------------------------------------------


def write_line(log, record):
    """Append one JSON line and push it to disk immediately."""
    log.write(json.dumps(record) + "\n")
    log.flush()
    os.fsync(log.fileno())


def progress_line(model, seed, decision, action, stages, cost):
    print(f"[{model} s{seed}] #{decision} {action} | stages {stages}/7 | ${cost:.4f}", flush=True)


def run_episode_tools(model, seed, depth=25, max_decisions=60, max_quarter_turns=300,
                      history="window", window=5, llm=None, run_id=None, raw_dir=RAW_DIR,
                      cost_limit=None, episode_timeout=600, progress=True,
                      reasoning_max_tokens=DEFAULT_REASONING_MAX_TOKENS,
                      max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                      reasoning_effort=DEFAULT_REASONING_EFFORT, start_state=None, task="full", target_stage=7):
    """Every model response counts as one decision, whatever it contains.

    start_state: start from this cube state instead of a `depth` scramble
    (e.g. a stage task from tasks.make_stage_task). `task` labels the episode
    in logs and the summary. The episode succeeds, and stops, once
    stages_complete reaches `target_stage` (7 = solved).

    cost_limit (USD): stop once the episode's total cost goes over this.
    episode_timeout (s): wall-clock limit for the episode, including a call in
    progress; when hit, the episode ends with stop_reason "timeout".
    progress: print one short line per decision.
    reasoning_effort / reasoning_max_tokens / max_output_tokens: thinking
    settings sent with every request (see OpenRouterLLM) and recorded in
    every log line and the summary.
    """
    if history not in ("window", "full"):
        raise ValueError(f"history must be 'window' or 'full', got {history!r}")
    check_budgets(reasoning_max_tokens, max_output_tokens, reasoning_effort)
    llm = llm or OpenRouterLLM(model, reasoning_max_tokens=reasoning_max_tokens,
                               max_output_tokens=max_output_tokens,
                               reasoning_effort=reasoning_effort)
    run_id = run_id or f"B_{model.replace('/', '_')}_s{seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"{run_id}.jsonl"
    system = METHOD_PROMPT.read_text(encoding="utf-8")

    env = CubeEnv()
    if start_state is None:
        env.reset(depth, seed)
    else:
        env.restore(start_state)
    start_stage = max_stage = stages_complete(env.state)
    decisions = quarter_turns = invalid = errors = reasoning_total = 0
    no_tool_calls = truncated = no_tool_streak = revisits = 0
    seen_states = {env.state}
    nudge = False  # last reply had no tool call: remind the model
    total_cost = 0.0
    total_latency = 0
    false_claim = False
    stop_reason = None
    recent = []  # (action, result) pairs for window history
    conversation = [{"role": "user", "content": state_text(env.state)}]  # full history
    episode_start = time.monotonic()

    with open(log_path, "w", encoding="utf-8") as log:
        while True:
            remaining = episode_timeout - (time.monotonic() - episode_start)
            if stages_complete(env.state) >= target_stage:
                stop_reason = "solved" if target_stage == 7 else "target_reached"
                break
            if decisions >= max_decisions:
                stop_reason = "max_decisions"
                break
            if quarter_turns >= max_quarter_turns:
                stop_reason = "max_quarter_turns"
                break
            if cost_limit is not None and total_cost > cost_limit:
                stop_reason = "cost_limit"
                break
            if remaining <= 0:
                stop_reason = "timeout"
                break

            if history == "full":
                messages = [{"role": "system", "content": system}] + conversation
            else:
                text = state_text(env.state)
                if recent:
                    lines = [f"{i}. {a} -> {r}" for i, (a, r) in enumerate(recent[-window:], 1)]
                    text += f"\n\nLast {len(lines)} action(s):\n" + "\n".join(lines)
                if nudge:
                    text += f"\n\n{NO_TOOL_CALL_NUDGE}"
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": text}]

            decisions += 1
            start = time.perf_counter()
            failure = None
            try:
                reply = call_with_deadline(lambda: llm(messages, TOOLS), remaining)
            except DeadlineExceeded:
                reply, failure = {}, "timeout"
            except Exception as exc:  # API failure after retries: end the episode, don't crash
                reply, failure = {}, f"api_error: {type(exc).__name__}"
            latency_ms = round((time.perf_counter() - start) * 1000)
            total_latency += latency_ms

            in_tok, out_tok, cost, cost_source = reply_cost(model, reply)
            total_cost += cost or 0.0
            extra_fields = reasoning_fields(reply, reasoning_max_tokens, max_output_tokens,
                                            reasoning_effort)
            reasoning_total += extra_fields["reasoning_tokens"] or 0

            calls = reply.get("tool_calls") or []
            record = {
                "run_id": run_id, "model": model, "condition": "B", "seed": seed,
                "task": task, "start_stage": start_stage,
                "decision": decisions, "tool": None, "arguments": None, "valid": None,
                "stages_complete": stages_complete(env.state), "input_tokens": in_tok,
                "output_tokens": out_tok, **extra_fields, "cost_usd": cost,
                "cost_source": cost_source, "latency_ms": latency_ms,
            }

            if failure or reply.get("error"):
                # No usable response: log it as a decision. A transport failure
                # or the episode deadline ends the episode; an error response
                # does not, but still uses up a decision.
                errors += 1
                record["error"] = failure or reply["error"]
                write_line(log, record)
                if progress:
                    progress_line(model, seed, decisions, f"ERROR {record['error']}",
                                  record["stages_complete"], total_cost)
                if failure:
                    stop_reason = failure
                    break
                recent.append(("(no action)", f"API error: {reply['error']}"))
                continue

            if not calls:
                # Counts as a decision; the model is asked again. Only
                # NO_TOOL_CALL_LIMIT such replies in a row end the episode.
                kind = "truncated" if reply.get("finish_reason") == "length" else "no_tool_call"
                no_tool_calls += 1
                truncated += kind == "truncated"
                no_tool_streak += 1
                claim = claims_solved(reply.get("content")) and not is_solved(env.state)
                false_claim = false_claim or claim
                record.update(reply_kind=kind, reply_text=reply.get("content"),
                              false_solved_claim=claim, no_tool_call_streak=no_tool_streak)
                write_line(log, record)
                if progress:
                    progress_line(model, seed, decisions, f"({kind}, no tool call)",
                                  record["stages_complete"], total_cost)
                if no_tool_streak >= NO_TOOL_CALL_LIMIT:
                    stop_reason = "no_tool_call"
                    break
                nudge = True
                conversation.append({"role": "assistant", "content": reply.get("content") or ""})
                conversation.append({"role": "user", "content": NO_TOOL_CALL_NUDGE})
                continue
            no_tool_streak = 0
            nudge = False

            # One action per decision: run the first call, refuse any extras.
            call = calls[0]
            valid, turns, result = execute_tool(env, call["name"], call["arguments"])
            quarter_turns += turns
            invalid += not valid
            stages = stages_complete(env.state)
            max_stage = max(max_stage, stages)
            # Loop metric: a valid action that lands on a state already seen
            # in this episode. (Invalid actions don't move the cube.)
            revisit = valid and env.state in seen_states
            revisits += revisit
            seen_states.add(env.state)
            record.update(tool=call["name"], arguments=call["arguments"], valid=valid,
                          stages_complete=stages, revisit=revisit, revisits=revisits)
            if len(calls) > 1:
                record["ignored_tool_calls"] = len(calls) - 1
            write_line(log, record)

            action = f"{call['name']}({call['arguments']})"
            if progress:
                progress_line(model, seed, decisions, action + ("" if valid else " INVALID"),
                              stages, total_cost)
            recent.append((action, f"{result} {stage_line(env.state)}; {yellow_up_line(env.state)}"))
            conversation.append({
                "role": "assistant", "content": reply.get("content"),
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}}
                               for c in calls],
            })
            conversation.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": f"{result}\n{yellow_up_line(env.state)}\n\n"
                                            f"{state_text(env.state)}"})
            for extra in calls[1:]:
                conversation.append({"role": "tool", "tool_call_id": extra["id"],
                                     "content": "Error: only one tool call per turn; this call was not run."})

    return {
        "run_id": run_id,
        "log_path": str(log_path),
        "task": task,
        "start_stage": start_stage,
        "target_stage": target_stage,
        "target_reached": max_stage >= target_stage,
        "solved": is_solved(env.state),
        "max_stage": max_stage,
        "final_stage": stages_complete(env.state),
        "decisions": decisions,
        "quarter_turns": quarter_turns,
        "invalid": invalid,
        "errors": errors,
        "no_tool_calls": no_tool_calls,
        "truncated": truncated,
        "revisits": revisits,
        "reasoning_tokens": reasoning_total,
        "reasoning_effort": reasoning_effort,
        "reasoning_max_tokens": reasoning_max_tokens,
        "max_output_tokens": max_output_tokens,
        "false_solved_claim": false_claim,
        "total_cost_usd": total_cost,
        "total_latency_ms": total_latency,
        "stop_reason": stop_reason,
    }


# --- condition A: one reply with the whole solution ------------------------

# "moves:" at the start of a line, once markdown decoration is removed.
_MOVES_LINE = re.compile(r"^(?:[-#>]\s*)*moves\s*:(.*)$", re.I)
_DECORATION = str.maketrans("", "", "*_`")
_PRIMES = str.maketrans({"’": "'", "‘": "'", "′": "'"})


def parse_moves_reply(text):
    """Quarter turns from the MOVES line (X2 -> X X), plus invalid tokens.

    A MOVES line is still required, but common formatting around it is
    accepted: bold or italics (**MOVES:**), backticks, code fences, list or
    heading markers, any capitalisation of "moves", curly primes. If the
    line has nothing after the colon, the next non-empty line (e.g. inside a
    code fence) is used. The last MOVES line in the reply wins.

    Returns (moves, invalid_tokens), or (None, []) if there is no MOVES line.
    """
    lines = [line.translate(_DECORATION).translate(_PRIMES).strip()
             for line in (text or "").splitlines()]
    lines = [line for line in lines if line]  # fences become empty once ` is removed
    found = None
    for i, line in enumerate(lines):
        match = _MOVES_LINE.match(line)
        if match:
            rest = match.group(1).strip()
            if not rest and i + 1 < len(lines) and not _MOVES_LINE.match(lines[i + 1]):
                rest = lines[i + 1]
            found = rest
    if found is None:
        return None, []
    moves, bad = [], []
    for token in found.replace(",", " ").split():
        token = token.rstrip(".;")
        if not token:
            continue
        if token in MOVE_NAMES:
            moves.append(token)
        elif len(token) == 2 and token[1] == "2" and token[0] in "UDLRFB":
            moves += [token[0]] * 2
        else:
            bad.append(token)
    return moves, bad


def run_episode_raw(model, seed, depth=25, max_quarter_turns=300, llm=None,
                    run_id=None, raw_dir=RAW_DIR, episode_timeout=600, progress=True,
                    reasoning_max_tokens=DEFAULT_REASONING_MAX_TOKENS,
                    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                    reasoning_effort=DEFAULT_REASONING_EFFORT):
    """Condition A: raw prompt, no tools, one call. Invalid tokens are skipped
    and counted; at most max_quarter_turns moves are applied. The one call
    counts as the one decision, even if it fails."""
    check_budgets(reasoning_max_tokens, max_output_tokens, reasoning_effort)
    llm = llm or OpenRouterLLM(model, reasoning_max_tokens=reasoning_max_tokens,
                               max_output_tokens=max_output_tokens,
                               reasoning_effort=reasoning_effort)
    run_id = run_id or f"A_{model.replace('/', '_')}_s{seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"{run_id}.jsonl"

    env = CubeEnv()
    env.reset(depth, seed)
    start_stage = max_stage = stages_complete(env.state)
    messages = [{"role": "system", "content": RAW_PROMPT.read_text(encoding="utf-8")},
                {"role": "user", "content": observation(env.state)}]

    start = time.perf_counter()
    error = None
    extra_fields = reasoning_fields({}, reasoning_max_tokens, max_output_tokens,
                                    reasoning_effort)
    try:
        reply = call_with_deadline(lambda: llm(messages, None), episode_timeout)
        extra_fields = reasoning_fields(reply, reasoning_max_tokens, max_output_tokens,
                                            reasoning_effort)
        if reply.get("error"):
            reply, error, stop_reason = None, reply["error"], "error_response"
    except DeadlineExceeded:
        reply, error, stop_reason = None, "timeout", "timeout"
    except Exception as exc:  # API failure after retries
        reply, stop_reason = None, f"api_error: {type(exc).__name__}"
        error = stop_reason
    latency_ms = round((time.perf_counter() - start) * 1000)

    in_tok = out_tok = 0
    cost, cost_source = 0.0, None
    moves, bad = [], []
    if reply is not None:
        in_tok, out_tok, cost, cost_source = reply_cost(model, reply)
        moves, bad = parse_moves_reply(reply.get("content"))
        stop_reason = "answered"
        if moves is None:
            truncated = reply.get("finish_reason") == "length"
            moves, stop_reason = [], "truncated" if truncated else "no_moves_line"
        if len(moves) > max_quarter_turns:
            moves, stop_reason = moves[:max_quarter_turns], "max_quarter_turns"
    for move in moves:
        env.step(move)
        max_stage = max(max_stage, stages_complete(env.state))

    record = {
        "run_id": run_id, "model": model, "condition": "A", "seed": seed,
        "task": "full", "start_stage": start_stage, "decision": 1,
        "tool": None, "arguments": reply.get("content") if reply else None,
        "valid": stop_reason in ("answered", "max_quarter_turns") and not bad,
        "invalid_tokens": bad, "stages_complete": stages_complete(env.state),
        "input_tokens": in_tok, "output_tokens": out_tok, **extra_fields, "cost_usd": cost,
        "cost_source": cost_source, "latency_ms": latency_ms,
    }
    if error:
        record["error"] = error
    if stop_reason in ("no_moves_line", "truncated"):
        record["parse_error"] = stop_reason
        record["reply_text"] = reply.get("content")
    with open(log_path, "w", encoding="utf-8") as log:
        write_line(log, record)
    if progress:
        action = f"MOVES ({len(moves)} quarter turns)" if reply else f"ERROR {error}"
        progress_line(model, seed, 1, action, record["stages_complete"], cost or 0.0)

    return {
        "run_id": run_id,
        "log_path": str(log_path),
        "task": "full",
        "start_stage": start_stage,
        "target_stage": 7,
        "target_reached": is_solved(env.state),
        "solved": is_solved(env.state),
        "max_stage": max_stage,
        "final_stage": stages_complete(env.state),
        "decisions": 1,
        "quarter_turns": len(moves),
        "invalid": len(bad) + (stop_reason in ("no_moves_line", "truncated")),
        "errors": int(error is not None),
        "reasoning_tokens": extra_fields["reasoning_tokens"] or 0,
        "reasoning_effort": reasoning_effort,
        "reasoning_max_tokens": reasoning_max_tokens,
        "max_output_tokens": max_output_tokens,
        "false_solved_claim": False,
        "total_cost_usd": cost or 0.0,
        "total_latency_ms": latency_ms,
        "stop_reason": stop_reason,
    }

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

    The openai client's `timeout` is passed to httpx, where it limits each
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
    """

    def __init__(self, model, client=None, timeout=60, retries=3, backoff=2.0):
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
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

        def request():
            return self.client.chat.completions.create(
                model=self.model, messages=messages, timeout=self.timeout, **extra,
            )

        for attempt in range(self.retries + 1):
            try:
                resp = call_with_deadline(request, self.timeout)
                break
            except retryable:
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff * 2 ** attempt)

        usage = getattr(resp, "usage", None)
        usage_dict = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            # OpenRouter reports cost in credits (1 credit = 1 USD).
            "cost": getattr(usage, "cost", None),
        } if usage is not None else None
        choices = getattr(resp, "choices", None)
        error = getattr(resp, "error", None)
        if error or not choices or getattr(choices[0], "message", None) is None:
            return {"error": str(error or "response had no message"), "content": None,
                    "tool_calls": [], "usage": usage_dict}

        msg = choices[0].message
        return {
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
            "usage": usage_dict,
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
                      cost_limit=None, episode_timeout=600, progress=True):
    """Every model response counts as one decision, whatever it contains.

    cost_limit (USD): stop once the episode's total cost goes over this.
    episode_timeout (s): wall-clock limit for the episode, including a call in
    progress; when hit, the episode ends with stop_reason "timeout".
    progress: print one short line per decision.
    """
    if history not in ("window", "full"):
        raise ValueError(f"history must be 'window' or 'full', got {history!r}")
    llm = llm or OpenRouterLLM(model)
    run_id = run_id or f"B_{model.replace('/', '_')}_s{seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"{run_id}.jsonl"
    system = METHOD_PROMPT.read_text(encoding="utf-8")

    env = CubeEnv()
    env.reset(depth, seed)
    max_stage = stages_complete(env.state)
    decisions = quarter_turns = invalid = errors = 0
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
            if is_solved(env.state):
                stop_reason = "solved"
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

            calls = reply.get("tool_calls") or []
            record = {
                "run_id": run_id, "model": model, "condition": "B", "seed": seed,
                "decision": decisions, "tool": None, "arguments": None, "valid": None,
                "stages_complete": stages_complete(env.state), "input_tokens": in_tok,
                "output_tokens": out_tok, "cost_usd": cost, "cost_source": cost_source,
                "latency_ms": latency_ms,
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
                false_claim = claims_solved(reply.get("content")) and not is_solved(env.state)
                record["final_reply"] = reply.get("content")
                record["false_solved_claim"] = false_claim
                write_line(log, record)
                if progress:
                    progress_line(model, seed, decisions, "(text reply, no tool call)",
                                  record["stages_complete"], total_cost)
                stop_reason = "no_tool_call"
                break

            # One action per decision: run the first call, refuse any extras.
            call = calls[0]
            valid, turns, result = execute_tool(env, call["name"], call["arguments"])
            quarter_turns += turns
            invalid += not valid
            stages = stages_complete(env.state)
            max_stage = max(max_stage, stages)
            record.update(tool=call["name"], arguments=call["arguments"], valid=valid,
                          stages_complete=stages)
            if len(calls) > 1:
                record["ignored_tool_calls"] = len(calls) - 1
            write_line(log, record)

            action = f"{call['name']}({call['arguments']})"
            if progress:
                progress_line(model, seed, decisions, action + ("" if valid else " INVALID"),
                              stages, total_cost)
            recent.append((action, f"{result} {stage_line(env.state)}"))
            conversation.append({
                "role": "assistant", "content": reply.get("content"),
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}}
                               for c in calls],
            })
            conversation.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": f"{result}\n\n{state_text(env.state)}"})
            for extra in calls[1:]:
                conversation.append({"role": "tool", "tool_call_id": extra["id"],
                                     "content": "Error: only one tool call per turn; this call was not run."})

    return {
        "run_id": run_id,
        "log_path": str(log_path),
        "solved": is_solved(env.state),
        "max_stage": max_stage,
        "final_stage": stages_complete(env.state),
        "decisions": decisions,
        "quarter_turns": quarter_turns,
        "invalid": invalid,
        "errors": errors,
        "false_solved_claim": false_claim,
        "total_cost_usd": total_cost,
        "total_latency_ms": total_latency,
        "stop_reason": stop_reason,
    }


# --- condition A: one reply with the whole solution ------------------------

_MOVES_LINE = re.compile(r"^\s*MOVES:(.*)$", re.M)


def parse_moves_reply(text):
    """Quarter turns from the 'MOVES:' line (X2 -> X X), plus invalid tokens.

    Returns (moves, invalid_tokens), or (None, []) if there is no MOVES: line.
    """
    match = _MOVES_LINE.search(text or "")
    if not match:
        return None, []
    moves, bad = [], []
    for token in match.group(1).split():
        if token in MOVE_NAMES:
            moves.append(token)
        elif len(token) == 2 and token[1] == "2" and token[0] in "UDLRFB":
            moves += [token[0]] * 2
        else:
            bad.append(token)
    return moves, bad


def run_episode_raw(model, seed, depth=25, max_quarter_turns=300, llm=None,
                    run_id=None, raw_dir=RAW_DIR, episode_timeout=600, progress=True):
    """Condition A: raw prompt, no tools, one call. Invalid tokens are skipped
    and counted; at most max_quarter_turns moves are applied. The one call
    counts as the one decision, even if it fails."""
    llm = llm or OpenRouterLLM(model)
    run_id = run_id or f"A_{model.replace('/', '_')}_s{seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / f"{run_id}.jsonl"

    env = CubeEnv()
    env.reset(depth, seed)
    max_stage = stages_complete(env.state)
    messages = [{"role": "system", "content": RAW_PROMPT.read_text(encoding="utf-8")},
                {"role": "user", "content": observation(env.state)}]

    start = time.perf_counter()
    error = None
    try:
        reply = call_with_deadline(lambda: llm(messages, None), episode_timeout)
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
            moves, stop_reason = [], "no_moves_line"
        if len(moves) > max_quarter_turns:
            moves, stop_reason = moves[:max_quarter_turns], "max_quarter_turns"
    for move in moves:
        env.step(move)
        max_stage = max(max_stage, stages_complete(env.state))

    record = {
        "run_id": run_id, "model": model, "condition": "A", "seed": seed, "decision": 1,
        "tool": None, "arguments": reply.get("content") if reply else None,
        "valid": reply is not None and stop_reason != "no_moves_line" and not bad,
        "invalid_tokens": bad, "stages_complete": stages_complete(env.state),
        "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost,
        "cost_source": cost_source, "latency_ms": latency_ms,
    }
    if error:
        record["error"] = error
    with open(log_path, "w", encoding="utf-8") as log:
        write_line(log, record)
    if progress:
        action = f"MOVES ({len(moves)} quarter turns)" if reply else f"ERROR {error}"
        progress_line(model, seed, 1, action, record["stages_complete"], cost or 0.0)

    return {
        "run_id": run_id,
        "log_path": str(log_path),
        "solved": is_solved(env.state),
        "max_stage": max_stage,
        "final_stage": stages_complete(env.state),
        "decisions": 1,
        "quarter_turns": len(moves),
        "invalid": len(bad) + (stop_reason == "no_moves_line"),
        "errors": int(error is not None),
        "false_solved_claim": False,
        "total_cost_usd": cost or 0.0,
        "total_latency_ms": latency_ms,
        "stop_reason": stop_reason,
    }

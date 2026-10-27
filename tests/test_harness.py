import json
import sys
from pathlib import Path
from types import SimpleNamespace

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import harness  # noqa: E402
from cube import scramble  # noqa: E402

MODEL = "fake/model"
SEED = 11
USAGE = {"input_tokens": 100, "output_tokens": 10, "cost": 0.001}


def tool_reply(tool, **args):
    return {"content": None, "usage": USAGE,
            "tool_calls": [{"id": "call_1", "name": tool, "arguments": json.dumps(args)}]}


def text_reply(text):
    return {"content": text, "tool_calls": [], "usage": USAGE}


class ScriptedLLM:
    """Returns the scripted replies in order, then repeats the last one."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, tools):
        self.calls.append(messages)
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


def inverse(move):
    return move[:-1] if move.endswith("'") else move + "'"


def read_log(summary):
    lines = Path(summary["log_path"]).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def run(llm, tmp_path, **kwargs):
    kwargs.setdefault("depth", 1)
    return harness.run_episode_tools(MODEL, SEED, llm=llm, raw_dir=tmp_path, **kwargs)


def test_solves_one_move_scramble(tmp_path):
    fix = inverse(scramble(1, SEED)[0])
    llm = ScriptedLLM([tool_reply("turn", move=fix)])
    summary = run(llm, tmp_path)

    assert summary["solved"] is True
    assert summary["stop_reason"] == "solved"
    assert summary["decisions"] == 1
    assert summary["quarter_turns"] == 1
    assert summary["max_stage"] == summary["final_stage"] == 7
    assert summary["total_cost_usd"] == USAGE["cost"]

    log = read_log(summary)
    assert len(log) == len(llm.calls) == 1
    rec = log[0]
    assert rec["condition"] == "B" and rec["seed"] == SEED and rec["model"] == MODEL
    assert rec["tool"] == "turn" and json.loads(rec["arguments"]) == {"move": fix}
    assert rec["valid"] is True and rec["stages_complete"] == 7
    assert rec["cost_source"] == "api"


def test_invalid_move_returns_error_and_counts(tmp_path):
    llm = ScriptedLLM([tool_reply("turn", move="X2"), text_reply("Giving up.")])
    summary = run(llm, tmp_path)

    assert summary["invalid"] == 1
    assert read_log(summary)[0]["valid"] is False
    # The next request shows the model the error (window history).
    assert "Error: invalid move 'X2'" in llm.calls[1][-1]["content"]


def test_invalid_algorithm_args_do_not_crash(tmp_path):
    llm = ScriptedLLM([
        tool_reply("apply_algorithm", name="nope", front="green"),
        tool_reply("apply_algorithm", name="corner_insert", front="purple"),
        tool_reply("apply_algorithm", name="corner_insert", front="green", times=1000),
        {"content": None, "usage": USAGE,
         "tool_calls": [{"id": "c", "name": "turn", "arguments": "not json"}]},
        tool_reply("fly_away"),
        text_reply("Stopping."),
    ])
    summary = run(llm, tmp_path)
    assert summary["invalid"] == 5
    assert summary["quarter_turns"] == 0


def test_max_decisions_stops_episode(tmp_path):
    llm = ScriptedLLM([tool_reply("turn", move="U")])
    summary = run(llm, tmp_path, depth=10, max_decisions=7)

    assert summary["stop_reason"] == "max_decisions"
    assert summary["decisions"] == 7
    assert len(read_log(summary)) == 7


def test_false_solved_claim(tmp_path):
    llm = ScriptedLLM([text_reply("I solved it!")])
    summary = run(llm, tmp_path, depth=10)

    assert summary["solved"] is False
    assert summary["false_solved_claim"] is True
    assert summary["stop_reason"] == "no_tool_call"
    assert read_log(summary)[0]["false_solved_claim"] is True


def test_full_history_keeps_conversation(tmp_path):
    llm = ScriptedLLM([tool_reply("turn", move="U"), tool_reply("turn", move="U'"),
                       text_reply("Stopping.")])
    run(llm, tmp_path, depth=10, history="full")
    roles = [m["role"] for m in llm.calls[2]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]


def test_api_key_never_logged(tmp_path, monkeypatch):
    secret = "sk-or-TEST-SECRET-123456"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)
    seen = {}

    class FakeOpenAI:
        def __init__(self, base_url, api_key, max_retries):
            seen["api_key"] = api_key
            message = SimpleNamespace(content="I solved it", tool_calls=None)
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, cost=0.0),
            )
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    summary = harness.run_episode_tools(MODEL, SEED, depth=3, raw_dir=tmp_path)

    assert seen["api_key"] == secret  # the real code path read the key...
    log_text = Path(summary["log_path"]).read_text(encoding="utf-8")
    assert secret not in log_text  # ...and it never reached the log
    assert secret not in json.dumps(summary)


def test_retries_then_succeeds(monkeypatch):
    attempts = []
    ok = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=None),
    )

    def create(**kwargs):
        attempts.append(kwargs["timeout"])
        if len(attempts) < 3:
            raise openai.APIConnectionError(request=None)
        return ok

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = harness.OpenRouterLLM(MODEL, client=client, backoff=0)
    assert llm([], [])["content"] == "hi"
    assert attempts == [60, 60, 60]


def test_raw_condition_solves_with_moves_line(tmp_path):
    fix = inverse(scramble(1, SEED)[0])
    llm = ScriptedLLM([text_reply(f"Thinking...\nMOVES: {fix} U2 U2 X")])
    summary = harness.run_episode_raw(MODEL, SEED, depth=1, llm=llm, raw_dir=tmp_path)

    assert llm.calls[0][0]["content"].startswith("# Solving the cube")
    assert summary["solved"] is True
    assert summary["quarter_turns"] == 5  # fix + U2 U2 as four quarter turns
    assert summary["invalid"] == 1  # "X"
    rec = read_log(summary)[0]
    assert rec["condition"] == "A" and rec["invalid_tokens"] == ["X"]


def test_raw_condition_without_moves_line(tmp_path):
    llm = ScriptedLLM([text_reply("R U R' U'")])
    summary = harness.run_episode_raw(MODEL, SEED, depth=5, llm=llm, raw_dir=tmp_path)
    assert summary["stop_reason"] == "no_moves_line"
    assert summary["quarter_turns"] == 0 and summary["invalid"] == 1


def test_cost_limit_stops_episode(tmp_path):
    llm = ScriptedLLM([tool_reply("turn", move="U")])
    summary = run(llm, tmp_path, depth=10, cost_limit=0.0025)
    assert summary["stop_reason"] == "cost_limit"
    assert summary["decisions"] == 3  # 0.003 > 0.0025 after the third call

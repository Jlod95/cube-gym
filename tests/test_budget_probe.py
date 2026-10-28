import json
import sys
from pathlib import Path
from types import SimpleNamespace

import openai

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import budget_probe  # noqa: E402
import harness  # noqa: E402


def fake_llm_factory(settings):
    truncate = settings.get("max_output_tokens") == 4000  # setting (a) gets cut off

    def llm(messages, tools):
        assert tools is harness.TOOLS and messages[0]["content"].startswith("# Solving the cube")
        usage = {"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 40, "cost": 0.001}
        if truncate:
            return {"content": None, "tool_calls": [], "finish_reason": "length", "usage": usage,
                    "reasoning_status": "ignored: used 4000 > budget 2000"}
        return {"content": None, "finish_reason": "tool_calls", "usage": usage,
                "reasoning_status": "ok",
                "tool_calls": [{"id": "c", "name": "turn", "arguments": '{"move": "U"}'}]}
    return llm


def test_probe_counts_per_setting(tmp_path):
    lines = []
    results = budget_probe.probe("m/x", calls=2, llm_factory=fake_llm_factory, log_dir=tmp_path,
                                 out=lines.append)
    assert set(results) == set(budget_probe.SETTINGS)
    a = results["a: reasoning 2000 / out 4000"]
    b = results["b: reasoning 1024 / out 8000"]
    assert len(a) == len(b) == 4  # 2 calls x 2 start positions
    assert all(r["truncated"] and not r["valid_tool_call"] for r in a)
    assert all(r["valid_tool_call"] and not r["truncated"] for r in b)
    text = "\n".join(lines)
    assert "ignored: used 4000 > budget 2000" in text
    log = list(tmp_path.glob("probe_*.jsonl"))
    assert len(log) == 1 and len(log[0].read_text().splitlines()) == 16


def test_probe_stops_at_max_cost(tmp_path):
    lines = []
    results = budget_probe.probe("m/x", calls=5, max_cost=0.0045, llm_factory=fake_llm_factory,
                                 log_dir=tmp_path, out=lines.append)
    assert sum(len(rows) for rows in results.values()) == 5  # 0.005 > 0.0045
    assert any("STOP: total cost $0.0050 passed" in line for line in lines)


def test_effort_setting_sent(monkeypatch):
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, cost=0.0,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
        msg = SimpleNamespace(content="x", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    for settings in budget_probe.SETTINGS.values():
        reply = harness.OpenRouterLLM("m/x", client=client, **settings)([], [])
        assert reply["reasoning_status"]
    assert [s["extra_body"] for s in sent] == [
        {"reasoning": {"max_tokens": 2000}}, {"reasoning": {"max_tokens": 1024}},
        {"reasoning": {"effort": "low"}}, {"reasoning": {"effort": "none"}}]
    assert [s["max_tokens"] for s in sent] == [4000, 8000, 8000, 2000]


def test_harness_defaults_unchanged():
    import inspect

    assert harness.DEFAULT_REASONING_MAX_TOKENS == 2000
    assert harness.DEFAULT_MAX_OUTPUT_TOKENS == 4000
    params = inspect.signature(harness.OpenRouterLLM).parameters
    assert params["reasoning_effort"].default is None
    assert json.dumps(params["reasoning_max_tokens"].default) == "2000"

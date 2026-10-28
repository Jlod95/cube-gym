import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark  # noqa: E402

USAGE = {"input_tokens": 10, "output_tokens": 5, "cost": 0.01}


def fake_llm_factory(model):
    def llm(messages, tools):
        if tools is None:  # condition A
            return {"content": "MOVES: U", "tool_calls": [], "usage": USAGE}
        return {"content": "Done.", "tool_calls": [], "usage": USAGE}
    return llm


def write_raw(raw_dir, run_id, model, condition, costs):
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"run_id": run_id, "model": model, "condition": condition, "cost_usd": c})
             for c in costs]
    (raw_dir / f"{run_id}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(tmp_path, **kwargs):
    lines = []
    kwargs.setdefault("models", ["m/x"])
    kwargs.setdefault("conditions", ["A", "B"])
    kwargs.setdefault("seeds", [1, 2])
    rows = benchmark.run_benchmark(results_dir=tmp_path / "results", raw_dir=tmp_path / "raw",
                                   llm_factory=fake_llm_factory, out=lines.append, **kwargs)
    return rows, "\n".join(lines)


def test_parse_seeds():
    assert benchmark.parse_seeds("1-3,7") == [1, 2, 3, 7]
    assert benchmark.parse_seeds("5") == [5]


def test_runs_all_and_writes_summaries(tmp_path):
    rows, _ = run(tmp_path, k=2)
    assert len(rows) == 8  # 2 conditions x 2 seeds x 2 attempts
    with open(tmp_path / "results" / "summary.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 8
    assert len(json.loads((tmp_path / "results" / "summary.json").read_text())) == 8


def test_resume_skips_completed(tmp_path):
    run(tmp_path, seeds=[1])
    rows, text = run(tmp_path, seeds=[1, 2])
    assert "2 already done; 2 to run" in text
    assert len(rows) == 4
    keys = [(r["condition"], r["seed"]) for r in rows]
    assert len(set(keys)) == 4  # nothing ran twice


def test_dry_run_estimates_and_missing(tmp_path):
    write_raw(tmp_path / "raw", "smoke1", "m/x", "A", [0.02])
    write_raw(tmp_path / "raw", "smoke2", "m/x", "A", [0.04])
    rows, text = run(tmp_path, dry_run=True)
    assert rows == []
    assert "estimated cost of remaining episodes: $0.0600" in text  # 2 A episodes x $0.03
    assert "missing estimates (no earlier logs): m/x B" in text
    assert not (tmp_path / "results" / "episodes.jsonl").exists()


def test_max_cost_aborts_before_episode(tmp_path):
    write_raw(tmp_path / "raw", "bench-old", "m/x", "A", [0.5])  # earlier benchmark spend
    write_raw(tmp_path / "raw", "smoke", "m/x", "B", [0.3])  # estimate only, not spend
    rows, text = run(tmp_path, conditions=["A"], max_cost=0.52)
    # spend 0.5 + estimate for A (avg of 0.5) > 0.52
    assert "ABORT" in text
    assert rows == []


def test_budgets_in_summary_and_part_of_resume_key(tmp_path):
    rows, _ = run(tmp_path, conditions=["A"], seeds=[1], reasoning_effort=None,
                  reasoning_max_tokens=1000, max_output_tokens=2000)
    with open(tmp_path / "results" / "summary.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["reasoning_max_tokens"] == "1000" and row["max_output_tokens"] == "2000"
    assert "reasoning_tokens" in row

    # Same seed under a different budget is a new episode, not skipped.
    rows, text = run(tmp_path, conditions=["A"], seeds=[1])
    assert "0 already done; 1 to run" in text
    assert len(rows) == 2


def test_summary_has_loop_and_reply_columns(tmp_path):
    run(tmp_path, conditions=["B"], seeds=[1])
    with open(tmp_path / "results" / "summary.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    for col in ("revisits", "no_tool_calls", "truncated"):
        assert col in row
    assert row["no_tool_calls"] == "3"  # the fake never calls a tool


def test_default_effort_in_summary(tmp_path):
    run(tmp_path, conditions=["A"], seeds=[1])
    with open(tmp_path / "results" / "summary.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["reasoning_effort"] == "low" and row["max_output_tokens"] == "8000"
    assert row["reasoning_max_tokens"] == ""


def test_thinking_arms_run_same_tasks_under_each_setting(tmp_path):
    rows, text = run(tmp_path, conditions=["A", "B"], seeds=[1, 2], thinking_arms=["low", "none"])
    assert len(rows) == 8  # 2 conditions x 2 seeds x 2 arms
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["reasoning_effort"], set()).add((r["condition"], r["seed"]))
    assert by_arm["low"] == by_arm["none"] == {("A", 1), ("A", 2), ("B", 1), ("B", 2)}
    assert "thinking arms: effort low, output cap 8000; effort none, output cap 8000" in text

    # The effort is part of the resume key: adding an arm only runs the new arm.
    rows, text = run(tmp_path, conditions=["A", "B"], seeds=[1, 2], thinking_arms=["low", None])
    assert "4 already done; 4 to run" in text


def test_cli_effort_resolution():
    import argparse

    def parse(*args):
        p = argparse.ArgumentParser()
        benchmark.add_thinking_args(p, arms=True)
        a = p.parse_args(list(args))
        return benchmark.resolve_effort(p, a), a

    assert parse()[0] == "low"
    assert parse("--reasoning-effort", "none")[0] == "none"
    assert parse("--reasoning-effort", "unset")[0] is None
    assert parse("--reasoning-max-tokens", "1500")[0] is None  # token budget instead
    assert parse("--thinking-arms", "low,none,unset")[1].thinking_arms == ["low", "none", None]
    with pytest.raises(SystemExit):
        parse("--reasoning-effort", "low", "--reasoning-max-tokens", "1500")
    with pytest.raises(SystemExit):
        parse("--reasoning-effort", "high")

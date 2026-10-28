import csv
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark  # noqa: E402
import make_charts  # noqa: E402

CHARTS = ["chart1_tools_vs_raw.png", "chart2_how_far.png", "chart3_stage_success.png",
          "chart4_rewards.png", "chart5_people_vs_models.csv"]


def fake_summary(results_dir, n_models=3):
    """A plausible summary.csv: full A/B and stage tasks for a few models."""
    rng = random.Random(0)
    rows = []
    models = ["anthropic/claude-haiku-5.5", "deepseek/deepseek-v4-flash", "qwen/qwen3.5-9b"][:n_models]
    for m in models:
        efforts = ["low", "none"] if m.startswith("qwen") else ["low"]
        for effort in efforts:
            for seed in (1, 2, 3):
                for cond in ("A", "B"):
                    stage = 0 if cond == "A" else rng.randint(0, 3)
                    rows.append({"model": m, "condition": cond, "history": "window" if cond == "B" else "-",
                                 "task": "full", "seed": seed, "attempt": 1, "start_stage": 0,
                                 "target_stage": 7, "target_reached": False, "solved": False,
                                 "max_stage": stage, "final_stage": stage,
                                 "decisions": 1 if cond == "A" else rng.randint(10, 60),
                                 "quarter_turns": rng.randint(10, 120),
                                 "total_cost_usd": rng.uniform(0.0005, 0.05),
                                 "reasoning_effort": effort, "stop_reason": "max_decisions"})
            for s in range(2, 8):
                for seed in range(1, 4):
                    ok = rng.random() < 0.8 - 0.1 * s
                    rows.append({"model": m, "condition": "B", "history": "window",
                                 "task": f"stage{s}", "seed": seed, "attempt": 1,
                                 "start_stage": s - 1, "target_stage": s, "target_reached": ok,
                                 "solved": ok and s == 7, "max_stage": s if ok else s - 1,
                                 "final_stage": s if ok else s - 1, "decisions": rng.randint(1, 30),
                                 "quarter_turns": rng.randint(4, 90),
                                 "total_cost_usd": rng.uniform(0.001, 0.03),
                                 "reasoning_effort": effort, "stop_reason": "target_reached"})
    results_dir.mkdir(parents=True, exist_ok=True)
    benchmark.write_summaries(rows, results_dir)
    return rows


def run(results_dir):
    lines = []
    charts = make_charts.make_charts(results_dir, out=lines.append)
    return charts, "\n".join(lines)


def test_works_with_no_results(tmp_path):
    charts, text = run(tmp_path)
    for name in CHARTS:
        assert (charts / name).stat().st_size > 0
    assert text.count("(no data yet)") == 3  # charts 1-3
    assert "farming (D, D' x10)" in text  # chart 4 never needs results


def test_charts_from_results(tmp_path):
    rows = fake_summary(tmp_path)
    charts, text = run(tmp_path)
    for name in CHARTS:
        assert (charts / name).stat().st_size > 0
    assert "(no data yet)" not in text
    # entities split by effort only where a model has more than one
    assert "qwen3.5-9b (none)" in text and "claude-haiku-5.5 (low)" not in text
    # chart 3 counts match the input
    haiku_s2 = [r for r in rows if r["model"].endswith("haiku-5.5") and r["task"] == "stage2"]
    k = sum(r["target_reached"] for r in haiku_s2)
    line = next(l for l in text.splitlines() if l.strip().startswith("claude-haiku-5.5")
                and "/" in l)
    assert f"{k}/3" == line.split()[1]


def test_partial_results_only_stage_tasks(tmp_path):
    rows = [r for r in fake_summary(tmp_path) if r["task"] != "full"]
    benchmark.write_summaries(rows, tmp_path)
    _, text = run(tmp_path)
    assert text.count("(no data yet)") == 2  # charts 1 and 2


def test_people_vs_models_csv(tmp_path):
    fake_summary(tmp_path, n_models=1)
    with open(tmp_path / "human_solves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person", "task", "seed", "quarter_turns", "decisions", "solved", "notes"])
        w.writerow(["Jack", "stage7", 1, 30, 10, "true", ""])
        w.writerow(["Jack", "full", 1, 140, 60, "true", ""])
    charts, _ = run(tmp_path)
    with open(charts / "chart5_people_vs_models.csv", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    who = {(r["who"], r["task"], r["seed"]) for r in table}
    assert ("Jack", "stage7", "1") in who and ("Jack", "full", "1") in who
    assert ("method solver", "stage7", "1") in who
    assert ("claude-haiku-5.5", "stage7", "1") in who
    assert not any(r["who"] == "method solver" and r["task"] == "full" for r in table)
    solver = [r for r in table if r["who"] == "method solver"]
    assert all(r["target_reached"] == "True" for r in solver)


def test_reward_numbers_match_scripted_agents():
    table, rules = make_charts.reward_table()
    assert rules == ["Naive", "StaySolved", "HighWater"]
    by_agent = {row[0]: row[1:] for row in table}
    assert by_agent["farming (D, D' x10)"] == pytest.approx([1.0, 0.0, 0.0])
    assert by_agent["correct stage-7 solve"] == pytest.approx([0.3, 0.0, 1.1])

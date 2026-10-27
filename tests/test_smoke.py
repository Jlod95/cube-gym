import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke  # noqa: E402
from tasks import parse_tasks  # noqa: E402


def fake_llm(messages, tools):
    content = "Stopping." if tools else "MOVES: U"
    return {"content": content, "tool_calls": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 0, "cost": 0.01}}


def run(tmp_path, **kwargs):
    lines = []
    results = smoke.run_smoke("m/x", llm=fake_llm, raw_dir=tmp_path / "raw", progress=False,
                              out=lines.append, **kwargs)
    return results, "\n".join(lines)


def test_plan_tasks_and_conditions(tmp_path):
    results, _ = run(tmp_path, seeds=[1, 2], tasks=parse_tasks("full,stage3-4"), stage_seeds=[5])
    keys = [(k[3], k[1], k[4]) for k, _ in results]
    assert keys == [("full", "A", 1), ("full", "A", 2), ("full", "B", 1), ("full", "B", 2),
                    ("stage3", "B", 5), ("stage4", "B", 5)]


def test_smoke_never_writes_episodes_file(tmp_path, monkeypatch):
    import benchmark

    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path / "results")
    run(tmp_path, tasks=parse_tasks("full,stage2"))
    assert not (tmp_path / "results" / "episodes.jsonl").exists()
    # Raw logs are written, and none use the benchmark's run-id prefix.
    raw = list((tmp_path / "raw").glob("*.jsonl"))
    assert raw and not any(p.name.startswith(benchmark.RUN_PREFIX) for p in raw)


def test_max_cost_stops(tmp_path):
    results, text = run(tmp_path, seeds=[1, 2, 3], max_cost=0.015)
    assert len(results) == 2  # 0.01 + 0.01 > 0.015
    assert "WARNING: total cost $0.0200 is over" in text


def test_table_columns(tmp_path):
    _, text = run(tmp_path, seeds=[1], tasks=parse_tasks("full,stage2"))
    header = next(line for line in text.splitlines() if line.startswith("task"))
    for col in ("task", "cond", "target/solved", "max stage", "decisions", "reason tok",
                "cost $", "s/decision", "stop reason"):
        assert col in header
    assert any(line.startswith("stage2") and " B " in line for line in text.splitlines())

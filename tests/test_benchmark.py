import csv
import json
import sys
from pathlib import Path

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

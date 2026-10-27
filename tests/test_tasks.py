import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark  # noqa: E402
import harness  # noqa: E402
from method_solver import solve  # noqa: E402
from stages import stages_complete  # noqa: E402
from tasks import make_stage_task, parse_tasks  # noqa: E402

STAGES = range(2, 8)
SEEDS = range(1, 21)


@lru_cache(maxsize=None)
def task(stage, seed):
    return make_stage_task(stage, seed)


@pytest.mark.parametrize("stage", STAGES)
def test_task_has_exactly_previous_stages_complete(stage):
    for seed in SEEDS:
        assert stages_complete(task(stage, seed)) == stage - 1, f"stage {stage} seed {seed}"


@pytest.mark.parametrize("stage", STAGES)
def test_same_seed_same_task(stage):
    assert make_stage_task(stage, 3) == task(stage, 3)
    assert len({task(stage, seed) for seed in SEEDS}) > 1  # seeds give different tasks


@pytest.mark.parametrize("stage", STAGES)
def test_scripted_method_solves_stage_task(stage):
    reached = [seed for seed in SEEDS if stages_complete(solve(task(stage, seed), stage)[1]) >= stage]
    assert reached, f"scripted method solved no stage {stage} task"


def test_bad_stage_rejected():
    with pytest.raises(ValueError):
        make_stage_task(1, 1)
    with pytest.raises(ValueError):
        make_stage_task(8, 1)


def test_parse_tasks():
    assert parse_tasks("full") == ["full"]
    assert parse_tasks("stage2-4") == ["stage2", "stage3", "stage4"]
    assert parse_tasks("full,stage7,stage6-7") == ["full", "stage7", "stage6"]
    with pytest.raises(ValueError):
        parse_tasks("stage1")
    with pytest.raises(ValueError):
        parse_tasks("easy")


class ReplayLLM:
    """Fake model that replays a scripted list of tool calls."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.n = 0

    def __call__(self, messages, tools):
        name, args = self.actions[self.n]
        self.n += 1
        return {"content": None, "usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0.0},
                "tool_calls": [{"id": f"c{self.n}", "name": name, "arguments": json.dumps(args)}]}


@pytest.mark.parametrize("stage", [3, 7])
def test_harness_stage_task_target_reached(stage, tmp_path):
    start = task(stage, 1)
    actions, _ = solve(start, stage)
    summary = harness.run_episode_tools("fake/model", 1, llm=ReplayLLM(actions), raw_dir=tmp_path,
                                        start_state=start, task=f"stage{stage}",
                                        target_stage=stage, progress=False)
    assert summary["target_reached"] is True
    assert summary["stop_reason"] == ("solved" if stage == 7 else "target_reached")
    assert summary["task"] == f"stage{stage}" and summary["start_stage"] == stage - 1
    assert summary["decisions"] == len(actions)
    log = [json.loads(line) for line in Path(summary["log_path"]).read_text().splitlines()]
    assert all(r["task"] == f"stage{stage}" and r["start_stage"] == stage - 1 for r in log)


def test_harness_full_task_labels(tmp_path):
    summary = harness.run_episode_tools(
        "fake/model", 1, depth=5, max_decisions=1, raw_dir=tmp_path, progress=False,
        llm=lambda m, t: {"content": "stop", "tool_calls": [], "usage": None})
    assert summary["task"] == "full" and summary["target_stage"] == 7
    assert summary["target_reached"] is False


def test_benchmark_stage_tasks_condition_b_only(tmp_path):
    def llm_factory(model):
        return lambda m, t: {"content": "stop" if t else "MOVES: U", "tool_calls": [], "usage": None}

    lines = []
    rows = benchmark.run_benchmark(["m/x"], ["A", "B"], [1], tasks=parse_tasks("full,stage2-3"),
                                   stage_seeds=[1, 2], results_dir=tmp_path / "results",
                                   raw_dir=tmp_path / "raw", llm_factory=llm_factory,
                                   out=lines.append, progress=False)
    keys = sorted((r["task"], r["condition"], r["seed"]) for r in rows)
    assert keys == [("full", "A", 1), ("full", "B", 1),
                    ("stage2", "B", 1), ("stage2", "B", 2),
                    ("stage3", "B", 1), ("stage3", "B", 2)]
    stage_rows = [r for r in rows if r["task"] != "full"]
    assert all(r["start_stage"] == r["target_stage"] - 1 for r in stage_rows)

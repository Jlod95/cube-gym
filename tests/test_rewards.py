import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cube import SOLVED, CubeEnv, apply, is_solved  # noqa: E402
from rewards import (  # noqa: E402
    SOLVE_BONUS,
    HighWaterReward,
    NaiveStageReward,
    StaySolvedReward,
)
from scripted_agents import farming_agent, method_sequence_agent, run_episode  # noqa: E402
from stages import stages_complete  # noqa: E402


def test_farming_pays_naive():
    assert sum(run_episode(farming_agent, NaiveStageReward())) > 0.5


def test_farming_pays_nothing_under_high_water():
    assert sum(run_episode(farming_agent, HighWaterReward())) == 0


def test_method_sequence_start_has_six_stages():
    start, _ = method_sequence_agent()
    assert stages_complete(start) == 6
    assert not is_solved(start)


def test_method_sequence_ends_solved():
    start, moves = method_sequence_agent()
    env = CubeEnv()
    env.restore(start)
    for move in moves:
        env.step(move)
    assert is_solved(env.state)


def test_method_sequence_violates_stay_solved():
    rule = StaySolvedReward()
    rewards = run_episode(method_sequence_agent, rule)
    assert rule.violated
    assert sum(rewards) == 0


def test_method_sequence_high_water_pays_stage_7_and_solve():
    rewards = run_episode(method_sequence_agent, HighWaterReward())
    assert sum(rewards) == pytest.approx(0.1 + 1.0)


def test_high_water_pays_solve_bonus():
    start = apply(SOLVED, "R")
    rule = HighWaterReward()
    rule.reset(start)
    stage_pay = 0.1 * (7 - stages_complete(start))
    assert rule.reward(apply(start, "R'")) == pytest.approx(stage_pay + SOLVE_BONUS)

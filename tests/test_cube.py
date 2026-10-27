import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cube import MOVE_NAMES, SOLVED, CubeEnv, apply, is_solved, scramble  # noqa: E402


def inverse(move):
    return move[:-1] if move.endswith("'") else move + "'"


@pytest.mark.parametrize("move", MOVE_NAMES)
def test_move_four_times_is_identity(move):
    state = SOLVED
    for _ in range(4):
        state = apply(state, move)
    assert state == SOLVED


@pytest.mark.parametrize("move", MOVE_NAMES)
def test_move_then_inverse_is_identity(move):
    state = apply(apply(SOLVED, move), inverse(move))
    assert state == SOLVED


@pytest.mark.parametrize("move", MOVE_NAMES)
def test_single_move_unsolves(move):
    assert not is_solved(apply(SOLVED, move))


def test_sexy_move_six_times_is_identity():
    state = SOLVED
    for _ in range(6):
        for move in ["R", "U", "R'", "U'"]:
            state = apply(state, move)
    assert state == SOLVED


def test_same_seed_same_scramble():
    assert scramble(25, seed=42) == scramble(25, seed=42)
    assert scramble(25, seed=42) != scramble(25, seed=43)


def test_scramble_never_repeats_face():
    moves = scramble(200, seed=1)
    assert all(a[0] != b[0] for a, b in zip(moves, moves[1:]))


def test_scrambled_cube_is_not_solved():
    env = CubeEnv()
    env.reset(depth=20, seed=7)
    assert not is_solved(env.state)


def test_snapshot_restore():
    env = CubeEnv()
    env.reset(depth=20, seed=3)
    original = env.state
    snap = env.snapshot()
    for move in ["R", "U", "F'", "D", "L'"]:
        env.step(move)
    assert env.state != original
    env.restore(snap)
    assert env.state == original

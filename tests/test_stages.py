import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cube import SOLVED, apply, observation  # noqa: E402
from stages import stages_complete  # noqa: E402


def after(*moves):
    state = SOLVED
    for move in moves:
        state = apply(state, move)
    return state


def test_solved_is_7():
    assert stages_complete(SOLVED) == 7


# Spec change: stages 5 and 6 count up to a final U turn (AUF), so a solved
# cube with the top layer turned is 6 (was 4). Stage 7 still needs exactly solved.
@pytest.mark.parametrize("moves", [("U",), ("U", "U"), ("U'",)])
def test_top_layer_turned_is_6(moves):
    assert stages_complete(after(*moves)) == 6


def test_stage_5_up_to_u_turn_but_not_6():
    # cycle_corners leaves the edges in place and moves three corners; after a
    # U turn the edges still match up to a U turn (stage 5), corners don't (6).
    from algorithms import apply_algorithm

    state = apply(apply_algorithm(SOLVED, "cycle_corners", "green"), "U")
    assert stages_complete(state) == 5


def test_swapped_top_edges_not_fixed_by_u_turn():
    # Two swapped top edges can't all match under any U turn: stage 4 only.
    from algorithms import apply_algorithm

    state = apply(apply_algorithm(SOLVED, "swap_edges", "green"), "U")
    assert stages_complete(state) == 4


@pytest.mark.parametrize("move", ["D", "R"])
def test_cross_broken_is_0(move):
    assert stages_complete(after(move)) == 0


def face_letters(text, label):
    lines = text.splitlines()
    start = lines.index(label) + 1
    return "".join(lines[start : start + 3]).replace(" ", "")


def test_observation_of_solved_cube():
    text = observation(SOLVED)
    assert face_letters(text, "U (top)") == "Y" * 9
    assert face_letters(text, "D (bottom)") == "W" * 9

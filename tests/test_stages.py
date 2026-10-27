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


@pytest.mark.parametrize("moves", [("U",), ("U", "U")])
def test_top_layer_turned_is_4(moves):
    assert stages_complete(after(*moves)) == 4


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

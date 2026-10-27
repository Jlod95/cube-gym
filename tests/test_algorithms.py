import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms import FRONT_MAPS, apply_algorithm  # noqa: E402
from cube import SOLVED, CubeEnv  # noqa: E402
from stages import stages_complete  # noqa: E402

FRONTS = list(FRONT_MAPS)


def scrambled():
    env = CubeEnv()
    env.reset(depth=20, seed=5)
    return env.state


@pytest.mark.parametrize("front", FRONTS)
@pytest.mark.parametrize("name", ["corner_insert", "twist_corner"])
def test_six_times_is_identity(name, front):
    start = scrambled()
    assert apply_algorithm(start, name, front=front, times=6) == start


@pytest.mark.parametrize("front", FRONTS)
@pytest.mark.parametrize(
    "name, minimum",
    [
        ("edge_left", 2),
        ("edge_right", 2),
        ("yellow_cross", 3),
        ("swap_edges", 4),
        ("cycle_corners", 5),
    ],
)
def test_preserves_earlier_stages(name, minimum, front):
    assert stages_complete(apply_algorithm(SOLVED, name, front=front)) >= minimum


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown algorithm"):
        apply_algorithm(SOLVED, "no_such_alg")


def test_unknown_front_raises():
    with pytest.raises(ValueError, match="unknown front colour"):
        apply_algorithm(SOLVED, "corner_insert", front="purple")

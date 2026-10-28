import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from algorithms import ALGORITHMS, FRONT_MAPS  # noqa: E402
from harness import TOOLS  # noqa: E402

METHOD = (ROOT / "prompts" / "method.md").read_text(encoding="utf-8")
RAW = (ROOT / "prompts" / "raw.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ALGORITHMS)
def test_method_has_algorithm_name(name):
    assert name in METHOD


@pytest.mark.parametrize("name, moves", ALGORITHMS.items())
def test_method_has_move_string(name, moves):
    assert moves in METHOD, f"{name}: {moves!r} not in method.md"


@pytest.mark.parametrize("tool", [t["function"]["name"] for t in TOOLS])
def test_method_has_tool_name(tool):
    assert tool in METHOD


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t["function"]["name"])
def test_method_has_tool_signature(tool):
    fn = tool["function"]
    signature = f"{fn['name']}({', '.join(fn['parameters']['properties'])})"
    assert signature in METHOD


@pytest.mark.parametrize("front", FRONT_MAPS)
def test_method_has_front_colour(front):
    assert f"`{front}`" in METHOD


@pytest.mark.parametrize("name", ALGORITHMS)
def test_raw_has_no_algorithm_names(name):
    assert name not in RAW, f"raw.md mentions {name}; condition A must not get the method"


def test_method_states_harness_limits():
    import inspect

    import harness
    from harness import MAX_TIMES, run_episode_tools

    defaults = inspect.signature(run_episode_tools).parameters
    assert f"from 1 to {MAX_TIMES}" in METHOD
    assert f"at most {defaults['max_decisions'].default} actions" in METHOD
    assert f"{defaults['max_quarter_turns'].default} quarter turns" in METHOD
    assert "One tool call per turn" in METHOD
    assert "Every reply must contain a tool call" in METHOD
    assert f"{harness.NO_TOOL_CALL_LIMIT_WORDS} such replies in a row end the episode" in METHOD


def test_method_states_last_layer_up_to_u_turn():
    assert "The last layer counts as done up to a final U turn" in METHOD


def test_method_explains_yellow_up_cue():
    from harness import YELLOW_UP_LABEL

    stage7 = METHOD[METHOD.index("### Stage 7"):METHOD.index("## General advice")]
    assert f"`{YELLOW_UP_LABEL}: n/9`" in stage7
    assert "top front-right corner" in stage7 and "not the lower layers" in stage7

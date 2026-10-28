"""Audit the condition B feedback along the method solver's solutions.

For every stage task (stages 2-7, seeds 1-20) the scripted method solver
(src/method_solver.py) is replayed one action at a time with the harness's
own code: execute_tool runs the action, and the feedback line is built
exactly as run_episode_tools builds it (stage_line and yellow_up_line). The
numbers are read back from that text: stages complete and yellow stickers
facing up after every action.

Each task is also run through run_episode_tools itself with a fake model that
replays the solver's actions, to check the episode ends at its target.

Usage: python scripts/feedback_audit.py

Stage 7 drops are expected (twist_corner scrambles the lower layers until
every corner is done); they are reported, not treated as failures.
"""

import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cube import CubeEnv  # noqa: E402
from harness import execute_tool, run_episode_tools, stage_line, yellow_up_line  # noqa: E402
from method_solver import solve  # noqa: E402
from stages import stages_complete  # noqa: E402
from tasks import make_stage_task  # noqa: E402

STAGES = range(2, 8)
SEEDS = range(1, 21)
_STAGES_RE = re.compile(r"stages complete: (\d)/7")
_YELLOW_RE = re.compile(r"yellow stickers facing up: (\d)/9")


def feedback_text(result, state):
    """The feedback the model gets after an action (window-history form)."""
    return f"{result} {stage_line(state)}; {yellow_up_line(state)}"


def read_feedback(text):
    return int(_STAGES_RE.search(text).group(1)), int(_YELLOW_RE.search(text).group(1))


class ReplayLLM:
    def __init__(self, actions):
        self.actions = list(actions)

    def __call__(self, messages, tools):
        if not self.actions:  # ran out: the episode should have ended
            return {"content": "out of actions", "tool_calls": [], "usage": None}
        name, args = self.actions.pop(0)
        return {"content": None, "usage": None,
                "tool_calls": [{"id": "c", "name": name, "arguments": json.dumps(args)}]}


def audit_task(stage, seed, raw_dir):
    """Feedback after each solver action, plus the harness episode summary."""
    start = make_stage_task(stage, seed)
    actions, _ = solve(start, stage)

    env = CubeEnv()
    env.restore(start)
    steps = []
    before = read_feedback(feedback_text("start", env.state))
    for name, args in actions:
        valid, _, result = execute_tool(env, name, json.dumps(args))
        after = read_feedback(feedback_text(result, env.state))
        steps.append({"action": (name, args), "valid": valid, "stages": after[0],
                      "yellow": after[1], "stage_drop": after[0] < before[0],
                      "yellow_drop": after[1] < before[1]})
        before = after

    summary = run_episode_tools("audit/solver", seed, llm=ReplayLLM(actions), raw_dir=raw_dir,
                                start_state=start, task=f"stage{stage}", target_stage=stage,
                                max_decisions=len(actions) + 5, progress=False)
    return {"stage": stage, "seed": seed, "start_stage": stages_complete(start),
            "steps": steps, "summary": summary}


def audit_all(raw_dir):
    return [audit_task(stage, seed, raw_dir) for stage in STAGES for seed in SEEDS]


def print_table(audits, out=print):
    out(f"{'stage':>5}{'tasks':>7}{'reach target':>14}{'actions (mean)':>16}"
        f"{'stage drops':>13}{'yellow drops':>14}")
    for stage in STAGES:
        rows = [a for a in audits if a["stage"] == stage]
        actions = [len(a["steps"]) for a in rows]
        stage_drops = sum(s["stage_drop"] for a in rows for s in a["steps"])
        yellow_drops = sum(s["yellow_drop"] for a in rows for s in a["steps"])
        reached = sum(a["summary"]["target_reached"] for a in rows)
        note = "  (expected: lower layers scramble mid-twist)" if stage == 7 and stage_drops else ""
        out(f"{stage:>5}{len(rows):>7}{f'{reached}/{len(rows)}':>14}{statistics.mean(actions):>16.1f}"
            f"{stage_drops:>13}{yellow_drops:>14}{note}")


def main():
    with tempfile.TemporaryDirectory() as raw_dir:  # episode logs aren't kept
        print_table(audit_all(raw_dir))


if __name__ == "__main__":
    main()

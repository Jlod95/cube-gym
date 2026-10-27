"""Stage tasks: cube states where exactly stages 1..stage-1 are complete.

make_stage_task(stage, seed) starts from a solved cube and applies, in a
seeded random order, inverses of the algorithms for that stage and later
(random fronts and repeat counts), mixed with random U turns. It retries
until stages_complete(state) == stage - 1. Each such state is reachable from
solved by method algorithms, so the method can solve it.
"""

import random

from algorithms import ALGORITHMS, FRONT_MAPS, parse, relabel
from cube import SOLVED, apply
from stages import stages_complete

# The stage each algorithm is used for (see prompts/method.md).
ALGORITHM_STAGE = {
    "corner_insert": 2,
    "edge_left": 3,
    "edge_right": 3,
    "yellow_cross": 4,
    "swap_edges": 5,
    "cycle_corners": 6,
    "twist_corner": 7,
}
STAGE_TASK_RANGE = range(2, 8)
MAX_ATTEMPTS = 100_000


def _inverse(moves):
    return [m[:-1] if m.endswith("'") else m + "'" for m in reversed(moves)]


def make_stage_task(stage, seed):
    if stage not in STAGE_TASK_RANGE:
        raise ValueError(f"stage must be 2 to 7, got {stage!r}")
    rng = random.Random(f"stage{stage}-seed{seed}")
    names = [n for n, s in ALGORITHM_STAGE.items() if s >= stage]
    fronts = list(FRONT_MAPS)

    for _ in range(MAX_ATTEMPTS):
        state = SOLVED
        for _ in range(rng.randint(1, 6)):
            if rng.random() < 0.3:
                state = apply(state, rng.choice(["U", "U'"]))
                continue
            moves = _inverse(relabel(parse(ALGORITHMS[rng.choice(names)]), rng.choice(fronts)))
            for move in moves * rng.randint(1, 5):
                state = apply(state, move)
        if stages_complete(state) == stage - 1:
            return state
    raise RuntimeError(f"no stage {stage} task found for seed {seed}")


def parse_tasks(text):
    """'full', 'stage3', 'stage2-7' or a comma list of these -> task names."""
    tasks = []
    for part in text.split(","):
        part = part.strip()
        if part == "full":
            tasks.append("full")
        elif part.startswith("stage"):
            span = part[len("stage"):]
            lo, _, hi = span.partition("-")
            for s in range(int(lo), int(hi or lo) + 1):
                if s not in STAGE_TASK_RANGE:
                    raise ValueError(f"stage tasks are stage2 to stage7, got stage{s}")
                tasks.append(f"stage{s}")
        elif part:
            raise ValueError(f"unknown task {part!r}; use full, stageN or stageN-M")
    return list(dict.fromkeys(tasks))  # dedupe, keep order


def task_stage(task):
    """'stage3' -> 3; 'full' -> None."""
    return None if task == "full" else int(task[len("stage"):])

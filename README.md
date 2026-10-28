# Cube Gym

An environment for a Rubik's cube that is solved **the way a person solves
it**: in stages, with named algorithms as tools, and a verifier for each stage.

Most cube benchmarks ask a model for a full move sequence in one go. Cube Gym
also lets a model work like someone following a beginner's guide: look at the
cube, pick the next stage, apply an algorithm, check progress, repeat. That
makes it possible to ask *how far* a model gets and *where* it gets stuck, not
just whether it solved the cube.

## What's in it

| Part | Where | What it does |
|---|---|---|
| Cube | `src/cube.py` | 54-sticker cube, the 12 quarter turns, seeded scrambles, `CubeEnv` (reset / step / snapshot / restore) and a plain-text observation |
| Stages | `src/stages.py` | Verifiers for the 7 stages of the beginner layer-by-layer method (cross, first layer, second layer, yellow cross, yellow edges, corners positioned, solved). Stages 5–6 count up to a final U turn. |
| Algorithms | `src/algorithms.py` | The 7 method algorithms, performed as if any side colour faces you |
| Stage tasks | `src/tasks.py` | Seeded start states where exactly stages 1..n−1 are complete, to test one stage at a time |
| Reference solver | `src/method_solver.py` | A scripted method solver for stages 2–7, used to check tasks and feedback |
| Rewards | `src/rewards.py` | Three reward rules (Naive, StaySolved, HighWater) for the reward-hacking demo |
| Harness | `src/harness.py` | Runs models through OpenRouter in two conditions (below) and logs every call |
| Prompts | `prompts/` | `method.md` (condition B: the method and tools) and `raw.md` (condition A) |

**The two conditions:**

- **A: one answer.** The model sees the scrambled cube and replies with a whole
  solution on a `MOVES:` line. No method, no tools.
- **B: method + tools.** The model gets the method as its system prompt and two
  tools, `turn(move)` and `apply_algorithm(name, front, times)`. After every
  action it sees the new cube, how many stages are complete, and how many yellow
  stickers face up. One tool call per decision, up to 60 decisions.

Full tasks start from a 25-move scramble; stage tasks (condition B only) start
part-way through, so each stage can be scored on its own.

## Setup

Python 3.12+ (developed on 3.14). In PowerShell, from the repo folder:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
```

To call models, create a file named `.env` in the repo folder with your
OpenRouter key (it is git-ignored and never logged):

```
OPENROUTER_API_KEY=your-key-here
```

## Tests

All tests run offline, with fake models; no key needed.

```bash
.venv\Scripts\python -m pytest -q
```

## Smoke test

A few episodes of one model in both conditions, with a cost cap (default
$0.20). Smoke runs only write raw logs to `results/raw/`; they never count as
benchmark episodes.

```bash
.venv\Scripts\python scripts\smoke.py anthropic/claude-haiku-5.5 --tasks full,stage7 --stage-seeds 1
```

Useful options: `--seeds 1-2`, `--max-cost 0.20`, `--reasoning-effort low|none|unset`,
`--max-output-tokens 8000`, `--request-timeout 180`, `--quiet`.
`scripts\list_models.py` lists OpenRouter models that support tool calling,
with prices.

## Benchmark

Always start with a dry run: it lists the planned episodes and estimates the
cost from earlier logs.

```bash
.venv\Scripts\python scripts\benchmark.py --models anthropic/claude-haiku-5.5 qwen/qwen3.5-9b --conditions A B --tasks full,stage2-7 --seeds 1-3 --stage-seeds 1-5 --concurrency 6 --max-cost 6 --dry-run
```

Then run the same command without `--dry-run`. Notes:

- **Resumable:** finished episodes go to `results/episodes.jsonl`; rerunning the
  same command skips them, so a crash or abort can be picked up again.
- **`--max-cost`** caps total benchmark spend across runs, counting the
  estimated cost of episodes still running before starting another.
- **`--thinking-arms low,none`** runs the same tasks under several reasoning
  settings.
- Results: `results/summary.csv` and `results/summary.json`, one row per
  episode. Per-call logs: `results/raw/` (git-ignored).

Charts and tables from whatever results exist so far:

```bash
.venv\Scripts\python scripts\make_charts.py
```

## Reward-hacking demo

A reward that pays for progress can be gamed. Two scripted agents show it:

- **farming:** with the cross complete, it alternates D and D′, breaking and
  rebuilding the cross 10 times.
- **correct stage-7 solve:** twisting the last corners with the method
  scrambles the lower layers on purpose, then restores them.

```bash
.venv\Scripts\python src\scripted_agents.py
```

| Agent | Naive | StaySolved | HighWater |
|---|---|---|---|
| farming | 1.0 | 0.0 | 0.0 |
| correct stage-7 solve | 0.3 | 0.0 | 1.1 |

Naive pays for every rise in stages, so farming earns the most. StaySolved
never lets progress drop, so it punishes the correct solve. HighWater pays only
for new records plus a solve bonus: zero for farming, full credit for the real
solve.

## Results

*To be added after the benchmark run.*

<!-- charts: results/charts/chart1_tools_vs_raw.png ... chart5_people_vs_models.csv -->

## Layout

```
src/        cube, stages, algorithms, tasks, solver, rewards, harness
prompts/    system prompts for conditions A and B
scripts/    smoke, benchmark, charts, budget probe, feedback audit, model list
tests/      offline tests (pytest)
results/    summaries and charts (raw per-call logs are git-ignored)
```

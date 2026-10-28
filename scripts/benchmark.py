"""Run the benchmark: models x conditions x seeds x attempts.

Examples:
  python scripts/benchmark.py --models deepseek/deepseek-v4-flash --seeds 1-10 --dry-run
  python scripts/benchmark.py --models a/x b/y --conditions A B --seeds 1-10 --k 2 --max-cost 5

Completed episodes are appended to results/episodes.jsonl; rerunning the same
command skips them, so a crash or abort can be resumed. At the end,
results/summary.csv and results/summary.json are written from that file.

Cost estimates are the average cost per episode for each (model, condition),
taken from every log in results/raw. --max-cost counts only spend logged by
benchmark episodes (run ids starting with "bench-"), including episodes that
were interrupted.

Episodes run --concurrency at a time (default 4). The --max-cost check counts
the estimated cost of episodes still running before starting another one.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness import (  # noqa: E402
    DEFAULT_EPISODE_TIMEOUT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REASONING_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
    RAW_DIR,
    REASONING_EFFORTS,
    ROOT,
    check_budgets,
    run_episode_raw,
    run_episode_tools,
)
from tasks import make_stage_task, parse_tasks, task_stage  # noqa: E402

RESULTS_DIR = ROOT / "results"
RUN_PREFIX = "bench-"
DEFAULT_CONCURRENCY = 4
FIELDS = ["model", "condition", "history", "task", "seed", "attempt", "run_id", "start_stage",
          "target_stage", "target_reached", "solved", "max_stage",
          "final_stage", "decisions", "quarter_turns", "invalid", "errors", "no_tool_calls",
          "truncated", "revisits", "reasoning_tokens", "reasoning_effort", "reasoning_max_tokens",
          "max_output_tokens", "false_solved_claim",
          "total_cost_usd", "total_latency_ms", "stop_reason"]


def parse_seeds(text):
    """'1-10' / '1,3,5' / '1-3,7' -> sorted list of ints."""
    seeds = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            seeds.update(range(lo, hi + 1))
        elif part:
            seeds.add(int(part))
    return sorted(seeds)


def episode_key(model, condition, history, task, seed, attempt, reasoning_effort,
                reasoning_max_tokens, max_output_tokens):
    """The thinking setting is part of the key: the same seed under another
    setting is a different episode, never skipped as already done."""
    return (model, condition, history if condition == "B" else "-", task, seed, attempt,
            reasoning_effort, reasoning_max_tokens, max_output_tokens)


def row_key(r):
    return episode_key(r["model"], r["condition"], r["history"], r.get("task", "full"), r["seed"],
                       r["attempt"], r.get("reasoning_effort"), r.get("reasoning_max_tokens"),
                       r.get("max_output_tokens"))


def thinking_arm(reasoning_effort=DEFAULT_REASONING_EFFORT,
                 reasoning_max_tokens=DEFAULT_REASONING_MAX_TOKENS,
                 max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """One thinking setting, as the harness's keyword arguments."""
    check_budgets(reasoning_max_tokens, max_output_tokens, reasoning_effort)
    return {"reasoning_effort": reasoning_effort, "reasoning_max_tokens": reasoning_max_tokens,
            "max_output_tokens": max_output_tokens}


def plan_episodes(models, conditions, seeds, k, history, arms=None, tasks=("full",),
                  stage_seeds=()):
    """Full tasks run in every condition on `seeds`; stage tasks run in
    condition B only, on `stage_seeds`. Every episode runs once per thinking
    arm (default: the single default setting)."""
    arms = arms or [thinking_arm()]
    plan = []
    for m in models:
        for arm in arms:
            for task in tasks:
                task_conditions = (conditions if task == "full"
                                   else [c for c in conditions if c == "B"])
                task_seeds = seeds if task == "full" else stage_seeds
                plan += [episode_key(m, c, history, task, s, a, arm["reasoning_effort"],
                                     arm["reasoning_max_tokens"], arm["max_output_tokens"])
                         for c in task_conditions for s in task_seeds for a in range(1, k + 1)]
    return plan


def read_raw_logs(raw_dir):
    """Per-run totals from the raw logs: {run_id: {model, condition, cost}}."""
    runs = {}
    for path in Path(raw_dir).glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            run = runs.setdefault(rec["run_id"], {"model": rec["model"],
                                                  "condition": rec["condition"], "cost": 0.0})
            run["cost"] += rec.get("cost_usd") or 0.0
    return runs


def cost_estimates(runs):
    """{(model, condition): average cost per episode}."""
    totals = defaultdict(list)
    for run in runs.values():
        totals[(run["model"], run["condition"])].append(run["cost"])
    return {key: sum(c) / len(c) for key, c in totals.items()}


def benchmark_spend(runs):
    return sum(r["cost"] for run_id, r in runs.items() if run_id.startswith(RUN_PREFIX))


def load_completed(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_summaries(rows, results_dir):
    results_dir = Path(results_dir)
    with open(results_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def run_one(key, depth, max_decisions, cost_limit, llm=None, raw_dir=RAW_DIR,
            episode_timeout=DEFAULT_EPISODE_TIMEOUT, progress=True,
            request_timeout=DEFAULT_REQUEST_TIMEOUT):
    (model, condition, history, task, seed, attempt,
     reasoning_effort, reasoning_max_tokens, max_output_tokens) = key
    run_id = (f"{RUN_PREFIX}{condition}-{model.replace('/', '_')}-{task}-{history}-s{seed}"
              f"-k{attempt}-e{reasoning_effort}-r{reasoning_max_tokens}-o{max_output_tokens}")
    budgets = thinking_arm(reasoning_effort, reasoning_max_tokens, max_output_tokens)
    if condition == "A":
        return run_episode_raw(model, seed, depth=depth, llm=llm, run_id=run_id, raw_dir=raw_dir,
                               episode_timeout=episode_timeout, progress=progress,
                               request_timeout=request_timeout, **budgets)
    return run_episode_tools(model, seed, depth=depth, max_decisions=max_decisions,
                             history=history, llm=llm, run_id=run_id, raw_dir=raw_dir,
                             cost_limit=cost_limit, episode_timeout=episode_timeout,
                             progress=progress, request_timeout=request_timeout, **budgets,
                             **task_args(task, seed))


def task_args(task, seed):
    stage = task_stage(task)
    if stage is None:
        return {"task": "full"}
    return {"task": task, "start_state": make_stage_task(stage, seed), "target_stage": stage}


def run_benchmark(models, conditions, seeds, k=1, history="window", depth=25, max_decisions=60,
                  max_cost=None, dry_run=False, results_dir=RESULTS_DIR, raw_dir=RAW_DIR,
                  llm_factory=None, out=print, episode_timeout=DEFAULT_EPISODE_TIMEOUT, progress=True,
                  reasoning_max_tokens=DEFAULT_REASONING_MAX_TOKENS,
                  max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, tasks=("full",),
                  stage_seeds=(), reasoning_effort=DEFAULT_REASONING_EFFORT, thinking_arms=None,
                  concurrency=DEFAULT_CONCURRENCY, request_timeout=DEFAULT_REQUEST_TIMEOUT):
    """llm_factory(model) -> llm lets tests run without the network.

    thinking_arms: reasoning efforts to run every episode under (e.g.
    ["low", None]); None means just `reasoning_effort`. reasoning_max_tokens
    can only go with an effort of None (unset)."""
    efforts = thinking_arms if thinking_arms is not None else [reasoning_effort]
    arms = [thinking_arm(e, reasoning_max_tokens if e is None else None, max_output_tokens)
            for e in efforts]
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = results_dir / "episodes.jsonl"

    completed = load_completed(episodes_path)
    done = {row_key(r) for r in completed}
    plan = plan_episodes(models, conditions, seeds, k, history, arms, tasks, stage_seeds)
    todo = [key for key in plan if key not in done]

    runs = read_raw_logs(raw_dir)
    estimates = cost_estimates(runs)
    spent = benchmark_spend(runs)

    out("thinking arms: " + "; ".join(describe_arm(a) for a in arms))
    out(f"planned {len(plan)} episodes; {len(plan) - len(todo)} already done; {len(todo)} to run")
    if dry_run:
        groups = defaultdict(int)
        for model, condition, _, _, _, _, effort, *_ in todo:
            groups[(model, condition, str(effort))] += 1
        total, missing = 0.0, []
        out(f"\n{'model':<45}{'cond':>5}{'effort':>8}{'episodes':>10}{'$/episode':>11}{'est. $':>10}")
        for (model, condition, effort), n in sorted(groups.items()):
            est = estimates.get((model, condition))  # not per thinking arm
            if est is None:
                if f"{model} {condition}" not in missing:
                    missing.append(f"{model} {condition}")
                out(f"{model:<45}{condition:>5}{effort:>8}{n:>10}{'missing':>11}{'?':>10}")
            else:
                total += est * n
                out(f"{model:<45}{condition:>5}{effort:>8}{n:>10}{est:>11.4f}{est * n:>10.4f}")
        out(f"\nestimated cost of remaining episodes: ${total:.4f}"
            + (" (excluding missing estimates)" if missing else ""))
        out(f"benchmark spend logged so far: ${spent:.4f}")
        if missing:
            out("missing estimates (no earlier logs): " + ", ".join(missing)
                + ". Run scripts/smoke.py with these models to get one.")
        return completed

    # Up to `concurrency` episodes run at once, each in its own thread with its
    # own log file (run ids are unique per episode). Only this thread records
    # results and decides what to start next.
    #
    # --max-cost: before starting an episode, spend so far plus the estimated
    # cost of every episode still running plus this one's estimate must fit.
    # If it doesn't fit while others are running, wait for them (their actual
    # cost may be lower); if nothing is running, abort. A condition B episode
    # also gets a hard limit: the budget left after the spend so far and the
    # other running episodes' estimates. An episode without an estimate only
    # starts when nothing else is running, with the whole remaining budget as
    # its hard limit.
    pending = list(todo)
    running = {}  # future -> (index, key, estimate reserved)
    started = 0
    stopping = False
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while running or (pending and not stopping):
            while pending and not stopping and len(running) < max(1, concurrency):
                key = pending[0]
                model, condition = key[0], key[1]
                est = estimates.get((model, condition))
                reserved = sum(r for _, _, r in running.values())
                cost_limit = None
                if max_cost is not None:
                    if est is None:
                        if running:
                            break  # run it on its own once the others finish
                        out(f"warning: no cost estimate for {model} {condition}; "
                            f"relying on the remaining budget ${max_cost - spent:.4f} as a hard stop")
                        cost_limit = max_cost - spent
                    elif spent + reserved + est > max_cost:
                        if running:
                            break  # wait: running episodes may cost less than estimated
                        out(f"ABORT: spent ${spent:.4f} + estimated ${est:.4f} for the next episode "
                            f"would exceed --max-cost ${max_cost:.2f}. Rerun to resume.")
                        stopping = True
                        break
                    else:
                        cost_limit = max_cost - spent - reserved
                pending.pop(0)
                started += 1
                llm = llm_factory(model) if llm_factory else None
                future = pool.submit(run_one, key, depth, max_decisions, cost_limit, llm=llm,
                                     raw_dir=raw_dir, episode_timeout=episode_timeout,
                                     progress=progress, request_timeout=request_timeout)
                running[future] = (started, key, est or 0.0)
            if not running:
                break

            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                i, key, _ = running.pop(future)
                model, condition, history_, task, seed, attempt, effort, *_ = key
                summary = future.result()
                spent += summary["total_cost_usd"]
                out(f"[{i}/{len(todo)}] {condition} {task} {model} effort={effort} seed {seed} "
                    f"k{attempt}: target_reached={summary['target_reached']} "
                    f"max_stage={summary['max_stage']} ${summary['total_cost_usd']:.4f} "
                    f"({summary['stop_reason']}); spent ${spent:.4f}")

                reason = summary["stop_reason"] or ""
                if reason.startswith("api_error") or reason in ("cost_limit", "request_timeout"):
                    out(f"ABORT: episode stopped with {reason}; not recorded. "
                        f"{'Waiting for running episodes, then stopping. ' if running else ''}"
                        f"Rerun to resume.")
                    stopping = True
                    continue

                row = {"model": model, "condition": condition, "history": history_, "seed": seed,
                       "attempt": attempt, **summary}  # summary carries task, start_stage, ...
                with open(episodes_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                completed.append(row)

    write_summaries(completed, results_dir)
    out(f"wrote {results_dir / 'summary.csv'} and summary.json ({len(completed)} episodes)")
    return completed


def token_budget(text):
    return None if text.lower() == "none" else int(text)


def effort_arg(text):
    """--reasoning-effort / --thinking-arms value: low, none, or unset (None)."""
    if text == "unset":
        return None
    if text not in REASONING_EFFORTS:
        raise argparse.ArgumentTypeError(
            f"reasoning effort must be one of {', '.join(REASONING_EFFORTS)}, unset; got {text!r}")
    return text


def describe_arm(arm):
    if arm["reasoning_effort"] is not None:
        thinking = f"effort {arm['reasoning_effort']}"
    elif arm["reasoning_max_tokens"] is not None:
        thinking = f"reasoning max_tokens {arm['reasoning_max_tokens']}"
    else:
        thinking = "no reasoning setting"
    return f"{thinking}, output cap {arm['max_output_tokens']}"


_EFFORT_NOT_GIVEN = object()  # argparse would run a string default through effort_arg


def add_thinking_args(p, arms=False):
    """Shared by benchmark.py and smoke.py."""
    p.add_argument("--reasoning-effort", type=effort_arg, default=_EFFORT_NOT_GIVEN,
                   help=f"low, none or unset (default {DEFAULT_REASONING_EFFORT}; "
                        "unset if --reasoning-max-tokens is given)")
    if arms:
        p.add_argument("--thinking-arms", type=lambda t: [effort_arg(x.strip()) for x in t.split(",")],
                       help="run every episode under each of these efforts, e.g. low,none")
    p.add_argument("--reasoning-max-tokens", type=token_budget, default=DEFAULT_REASONING_MAX_TOKENS,
                   help="reasoning token budget instead of an effort (not the default); "
                        "0 asks for no reasoning")
    p.add_argument("--max-output-tokens", type=token_budget, default=DEFAULT_MAX_OUTPUT_TOKENS,
                   help=f"output cap per call, reasoning included (default "
                        f"{DEFAULT_MAX_OUTPUT_TOKENS}); 'none' sends no cap")


def add_request_timeout_arg(p):
    """Shared by benchmark.py and smoke.py."""
    p.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT,
                   help=f"wall-clock seconds per API request attempt, retried up to 3 times "
                        f"(default {DEFAULT_REQUEST_TIMEOUT})")


def resolve_effort(p, a):
    """The effort to use: the default, unless --reasoning-max-tokens asks for
    a token budget instead. Both set explicitly is an error."""
    if a.reasoning_effort is _EFFORT_NOT_GIVEN:
        return None if a.reasoning_max_tokens is not None else DEFAULT_REASONING_EFFORT
    if a.reasoning_effort is not None and a.reasoning_max_tokens is not None:
        p.error("--reasoning-effort and --reasoning-max-tokens can't be combined; "
                "use --reasoning-effort unset with a token budget")
    return a.reasoning_effort


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--conditions", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--seeds", default="1-10", help="seeds for full tasks, e.g. 1-10 or 1,3,5")
    p.add_argument("--tasks", default="full",
                   help="full, stageN, stageN-M or a comma list, e.g. full,stage2-7 "
                        "(stage tasks run in condition B only)")
    p.add_argument("--stage-seeds", default="1-10", help="seeds for stage tasks")
    p.add_argument("--k", type=int, default=1, help="attempts per seed")
    p.add_argument("--history", choices=["window", "full"], default="window", help="condition B only")
    p.add_argument("--depth", type=int, default=25, help="scramble depth")
    p.add_argument("--max-decisions", type=int, default=60)
    p.add_argument("--max-cost", type=float, help="USD budget for all benchmark episodes")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--episode-timeout", type=float, default=DEFAULT_EPISODE_TIMEOUT,
                   help=f"wall-clock seconds per episode (default {DEFAULT_EPISODE_TIMEOUT})")
    p.add_argument("--quiet", action="store_true", help="no per-decision progress lines")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"episodes run in parallel (default {DEFAULT_CONCURRENCY})")
    add_request_timeout_arg(p)
    add_thinking_args(p, arms=True)
    a = p.parse_args()
    effort = resolve_effort(p, a)
    if a.thinking_arms and a.reasoning_max_tokens is not None and any(a.thinking_arms):
        p.error("--reasoning-max-tokens only goes with the 'unset' thinking arm")
    run_benchmark(a.models, a.conditions, parse_seeds(a.seeds), a.k, a.history, a.depth,
                  a.max_decisions, a.max_cost, a.dry_run,
                  episode_timeout=a.episode_timeout, progress=not a.quiet,
                  reasoning_max_tokens=a.reasoning_max_tokens,
                  max_output_tokens=a.max_output_tokens, tasks=parse_tasks(a.tasks),
                  stage_seeds=parse_seeds(a.stage_seeds), reasoning_effort=effort,
                  thinking_arms=a.thinking_arms, concurrency=a.concurrency,
                  request_timeout=a.request_timeout)


if __name__ == "__main__":
    main()

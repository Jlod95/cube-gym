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
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_MAX_TOKENS,
    RAW_DIR,
    ROOT,
    run_episode_raw,
    run_episode_tools,
)

RESULTS_DIR = ROOT / "results"
RUN_PREFIX = "bench-"
FIELDS = ["model", "condition", "history", "seed", "attempt", "run_id", "solved", "max_stage",
          "final_stage", "decisions", "quarter_turns", "invalid", "errors", "reasoning_tokens", "reasoning_max_tokens",
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


def episode_key(model, condition, history, seed, attempt, reasoning_max_tokens, max_output_tokens):
    """Budgets are part of the key: the same seed under another budget is a
    different episode, never skipped as already done."""
    return (model, condition, history if condition == "B" else "-", seed, attempt,
            reasoning_max_tokens, max_output_tokens)


def row_key(r):
    return episode_key(r["model"], r["condition"], r["history"], r["seed"], r["attempt"],
                       r.get("reasoning_max_tokens"), r.get("max_output_tokens"))


def plan_episodes(models, conditions, seeds, k, history, reasoning_max_tokens, max_output_tokens):
    return [episode_key(m, c, history, s, a, reasoning_max_tokens, max_output_tokens)
            for m in models for c in conditions for s in seeds for a in range(1, k + 1)]


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
            episode_timeout=600, progress=True):
    model, condition, history, seed, attempt, reasoning_max_tokens, max_output_tokens = key
    run_id = (f"{RUN_PREFIX}{condition}-{model.replace('/', '_')}-{history}-s{seed}-k{attempt}"
              f"-r{reasoning_max_tokens}-o{max_output_tokens}")
    budgets = {"reasoning_max_tokens": reasoning_max_tokens, "max_output_tokens": max_output_tokens}
    if condition == "A":
        return run_episode_raw(model, seed, depth=depth, llm=llm, run_id=run_id, raw_dir=raw_dir,
                               episode_timeout=episode_timeout, progress=progress, **budgets)
    return run_episode_tools(model, seed, depth=depth, max_decisions=max_decisions,
                             history=history, llm=llm, run_id=run_id, raw_dir=raw_dir,
                             cost_limit=cost_limit, episode_timeout=episode_timeout,
                             progress=progress, **budgets)


def run_benchmark(models, conditions, seeds, k=1, history="window", depth=25, max_decisions=60,
                  max_cost=None, dry_run=False, results_dir=RESULTS_DIR, raw_dir=RAW_DIR,
                  llm_factory=None, out=print, episode_timeout=600, progress=True,
                  reasoning_max_tokens=DEFAULT_REASONING_MAX_TOKENS,
                  max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """llm_factory(model) -> llm lets tests run without the network."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = results_dir / "episodes.jsonl"

    completed = load_completed(episodes_path)
    done = {row_key(r) for r in completed}
    plan = plan_episodes(models, conditions, seeds, k, history, reasoning_max_tokens,
                         max_output_tokens)
    todo = [key for key in plan if key not in done]

    runs = read_raw_logs(raw_dir)
    estimates = cost_estimates(runs)
    spent = benchmark_spend(runs)

    out(f"budgets: reasoning_max_tokens={reasoning_max_tokens}, max_output_tokens={max_output_tokens}")
    out(f"planned {len(plan)} episodes; {len(plan) - len(todo)} already done; {len(todo)} to run")
    if dry_run:
        groups = defaultdict(int)
        for model, condition, *_ in todo:
            groups[(model, condition)] += 1
        total, missing = 0.0, []
        out(f"\n{'model':<45}{'cond':>5}{'episodes':>10}{'$/episode':>11}{'est. $':>10}")
        for (model, condition), n in sorted(groups.items()):
            est = estimates.get((model, condition))
            if est is None:
                missing.append(f"{model} {condition}")
                out(f"{model:<45}{condition:>5}{n:>10}{'missing':>11}{'?':>10}")
            else:
                total += est * n
                out(f"{model:<45}{condition:>5}{n:>10}{est:>11.4f}{est * n:>10.4f}")
        out(f"\nestimated cost of remaining episodes: ${total:.4f}"
            + (" (excluding missing estimates)" if missing else ""))
        out(f"benchmark spend logged so far: ${spent:.4f}")
        if missing:
            out("missing estimates (no earlier logs): " + ", ".join(missing)
                + ". Run scripts/smoke.py with these models to get one.")
        return completed

    for i, key in enumerate(todo, 1):
        model, condition, history_, seed, attempt, *_ = key
        est = estimates.get((model, condition))
        cost_limit = None
        if max_cost is not None:
            if est is not None and spent + est > max_cost:
                out(f"ABORT: spent ${spent:.4f} + estimated ${est:.4f} for the next episode "
                    f"would exceed --max-cost ${max_cost:.2f}. Rerun to resume.")
                break
            if est is None:
                out(f"warning: no cost estimate for {model} {condition}; "
                    f"relying on the remaining budget ${max_cost - spent:.4f} as a hard stop")
            cost_limit = max_cost - spent

        llm = llm_factory(model) if llm_factory else None
        summary = run_one(key, depth, max_decisions, cost_limit, llm=llm, raw_dir=raw_dir,
                          episode_timeout=episode_timeout, progress=progress)
        spent += summary["total_cost_usd"]
        out(f"[{i}/{len(todo)}] {condition} {model} seed {seed} k{attempt}: "
            f"solved={summary['solved']} max_stage={summary['max_stage']} "
            f"${summary['total_cost_usd']:.4f} ({summary['stop_reason']}); spent ${spent:.4f}")

        reason = summary["stop_reason"] or ""
        if reason.startswith("api_error") or reason == "cost_limit":
            out(f"ABORT: episode stopped with {reason}; not recorded. Rerun to resume.")
            break

        row = {"model": model, "condition": condition, "history": history_, "seed": seed,
               "attempt": attempt, **summary}
        with open(episodes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        completed.append(row)

    write_summaries(completed, results_dir)
    out(f"wrote {results_dir / 'summary.csv'} and summary.json ({len(completed)} episodes)")
    return completed


def token_budget(text):
    return None if text.lower() == "none" else int(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--conditions", nargs="+", choices=["A", "B"], default=["A", "B"])
    p.add_argument("--seeds", default="1-10", help="e.g. 1-10 or 1,3,5 or 1-3,7")
    p.add_argument("--k", type=int, default=1, help="attempts per seed")
    p.add_argument("--history", choices=["window", "full"], default="window", help="condition B only")
    p.add_argument("--depth", type=int, default=25, help="scramble depth")
    p.add_argument("--max-decisions", type=int, default=60)
    p.add_argument("--max-cost", type=float, help="USD budget for all benchmark episodes")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--episode-timeout", type=float, default=600, help="seconds per episode")
    p.add_argument("--quiet", action="store_true", help="no per-decision progress lines")
    p.add_argument("--reasoning-max-tokens", type=token_budget, default=DEFAULT_REASONING_MAX_TOKENS,
                   help="reasoning budget per call; 0 asks for no reasoning, 'none' sends no setting")
    p.add_argument("--max-output-tokens", type=token_budget, default=DEFAULT_MAX_OUTPUT_TOKENS,
                   help="output cap per call, reasoning included; 'none' sends no cap")
    a = p.parse_args()
    run_benchmark(a.models, a.conditions, parse_seeds(a.seeds), a.k, a.history, a.depth,
                  a.max_decisions, a.max_cost, a.dry_run,
                  episode_timeout=a.episode_timeout, progress=not a.quiet,
                  reasoning_max_tokens=a.reasoning_max_tokens,
                  max_output_tokens=a.max_output_tokens)


if __name__ == "__main__":
    main()

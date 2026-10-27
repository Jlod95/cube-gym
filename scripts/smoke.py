"""Smoke test: one model, a few episodes, before running the benchmark.

Usage: python scripts/smoke.py <openrouter-model-id> [--seeds 1-2] [--tasks full,stage2-7]
                               [--stage-seeds 1] [--max-cost 0.20] [--quiet]

Full tasks run in conditions A and B; stage tasks in condition B only (same
rules as benchmark.py). Stops with a warning once the total cost goes over
--max-cost.

Smoke runs only write raw logs (results/raw/A_... and B_...). They never
touch results/episodes.jsonl, so the benchmark never skips an episode
because a smoke run did it, and their spend doesn't count toward the
benchmark's --max-cost (its run ids start with "bench-"). The raw logs do
feed the benchmark's per-episode cost estimates.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import parse_seeds, plan_episodes, task_args, token_budget  # noqa: E402
from harness import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_MAX_TOKENS,
    RAW_DIR,
    run_episode_raw,
    run_episode_tools,
)
from tasks import parse_tasks  # noqa: E402


def run_smoke(model, seeds=(1, 2), tasks=("full",), stage_seeds=(1,), max_cost=0.20,
              reasoning_max_tokens=DEFAULT_REASONING_MAX_TOKENS,
              max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, progress=True, llm=None,
              raw_dir=RAW_DIR, out=print):
    """Run the smoke episodes; return [(key, summary)]. `llm` is for tests."""
    budgets = {"reasoning_max_tokens": reasoning_max_tokens, "max_output_tokens": max_output_tokens}
    plan = plan_episodes([model], ["A", "B"], seeds, 1, "window", reasoning_max_tokens,
                         max_output_tokens, tasks, stage_seeds)
    out(f"budgets: {budgets}")
    out(f"planned {len(plan)} episodes, max cost ${max_cost:.2f}")

    results = []
    spent = 0.0
    for key in plan:
        _, condition, history, task, seed, *_ = key
        if condition == "A":
            summary = run_episode_raw(model, seed, llm=llm, raw_dir=raw_dir, progress=progress,
                                      **budgets)
        else:
            # Pass the remaining budget so a long B episode stops mid-way.
            summary = run_episode_tools(model, seed, history=history, llm=llm, raw_dir=raw_dir,
                                        cost_limit=max_cost - spent, progress=progress,
                                        **budgets, **task_args(task, seed))
        spent += summary["total_cost_usd"]
        results.append((key, summary))
        out(f"done: {task} {condition} seed {seed} ({summary['stop_reason']}), total ${spent:.4f}")
        if spent > max_cost:
            out(f"\nWARNING: total cost ${spent:.4f} is over the ${max_cost:.2f} limit; stopping.")
            break

    print_table(model, results, out)
    for condition in ("A", "B"):
        costs = [s["total_cost_usd"] for k, s in results if k[1] == condition]
        if costs:
            out(f"average cost per episode, condition {condition}: "
                f"${sum(costs) / len(costs):.4f} ({len(costs)} episodes)")
    out(f"total cost: ${spent:.4f}")
    out(f"logs: {raw_dir}")
    return results


def print_table(model, results, out=print):
    out(f"\nmodel: {model}\n")
    header = (f"{'task':<8}{'cond':>5}{'seed':>5}{'target/solved':>15}{'max stage':>11}"
              f"{'decisions':>11}{'reason tok':>12}{'cost $':>10}{'s/decision':>12}  stop reason")
    out(header)
    out("-" * len(header))
    for (_, condition, _, task, seed, *_), s in results:
        per_decision = s["total_latency_ms"] / 1000 / s["decisions"] if s["decisions"] else 0.0
        out(f"{task:<8}{condition:>5}{seed:>5}{str(s['target_reached']):>15}{s['max_stage']:>11}"
            f"{s['decisions']:>11}{s['reasoning_tokens']:>12}{s['total_cost_usd']:>10.4f}"
            f"{per_decision:>12.1f}  {s['stop_reason']}")
    out("")


def main():
    p = argparse.ArgumentParser(description="Smoke test one model before the benchmark.")
    p.add_argument("model", help="OpenRouter model id")
    p.add_argument("--seeds", default="1-2", help="seeds for full tasks, e.g. 1-2 or 1,3")
    p.add_argument("--tasks", default="full",
                   help="full, stageN, stageN-M or a comma list (stage tasks: condition B only)")
    p.add_argument("--stage-seeds", default="1", help="seeds for stage tasks")
    p.add_argument("--max-cost", type=float, default=0.20, help="USD; stop once total goes over")
    p.add_argument("--quiet", action="store_true", help="no per-decision progress lines")
    p.add_argument("--reasoning-max-tokens", type=token_budget, default=DEFAULT_REASONING_MAX_TOKENS,
                   help="reasoning budget per call; 0 asks for no reasoning, 'none' sends no setting")
    p.add_argument("--max-output-tokens", type=token_budget, default=DEFAULT_MAX_OUTPUT_TOKENS,
                   help="output cap per call, reasoning included; 'none' sends no cap")
    a = p.parse_args()
    run_smoke(a.model, parse_seeds(a.seeds), parse_tasks(a.tasks), parse_seeds(a.stage_seeds),
              a.max_cost, a.reasoning_max_tokens, a.max_output_tokens, progress=not a.quiet)


if __name__ == "__main__":
    main()

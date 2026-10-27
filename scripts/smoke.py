"""Smoke test: one model, seeds 1 and 2, conditions A and B (window history).

Usage: python scripts/smoke.py <openrouter-model-id> [--quiet]

Stops early with a warning if the total cost goes over BUDGET_USD.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_REASONING_MAX_TOKENS,
    run_episode_raw,
    run_episode_tools,
)

SEEDS = (1, 2)
BUDGET_USD = 0.20


def token_budget(text):
    return None if text.lower() == "none" else int(text)


def main():
    p = argparse.ArgumentParser(description="Smoke test one model in conditions A and B.")
    p.add_argument("model", help="OpenRouter model id")
    p.add_argument("--quiet", action="store_true", help="no per-decision progress lines")
    p.add_argument("--reasoning-max-tokens", type=token_budget, default=DEFAULT_REASONING_MAX_TOKENS,
                   help="reasoning budget per call; 0 asks for no reasoning, 'none' sends no setting")
    p.add_argument("--max-output-tokens", type=token_budget, default=DEFAULT_MAX_OUTPUT_TOKENS,
                   help="output cap per call, reasoning included; 'none' sends no cap")
    a = p.parse_args()
    model, progress = a.model, not a.quiet
    budgets = {"reasoning_max_tokens": a.reasoning_max_tokens,
               "max_output_tokens": a.max_output_tokens}
    print(f"budgets: {budgets}")

    runs = [("A", seed) for seed in SEEDS] + [("B", seed) for seed in SEEDS]
    results = []
    spent = 0.0
    for condition, seed in runs:
        if condition == "A":
            summary = run_episode_raw(model, seed, progress=progress, **budgets)
        else:
            # Pass the remaining budget so a long B episode stops mid-way.
            summary = run_episode_tools(model, seed, history="window",
                                        cost_limit=BUDGET_USD - spent, progress=progress,
                                        **budgets)
        spent += summary["total_cost_usd"]
        results.append((condition, seed, summary))
        print(f"done: {condition} seed {seed} ({summary['stop_reason']}), total ${spent:.4f}")
        if spent > BUDGET_USD:
            print(f"\nWARNING: total cost ${spent:.4f} is over the ${BUDGET_USD:.2f} budget; stopping.")
            break

    print(f"\nmodel: {model}\n")
    header = (f"{'cond':<5}{'seed':>5}{'solved':>8}{'max stage':>11}{'decisions':>11}"
              f"{'qturns':>8}{'invalid':>9}{'reason tok':>11}{'cost $':>10}{'latency s':>11}  stop reason")
    print(header)
    print("-" * len(header))
    for condition, seed, s in results:
        print(f"{condition:<5}{seed:>5}{str(s['solved']):>8}{s['max_stage']:>11}{s['decisions']:>11}"
              f"{s['quarter_turns']:>8}{s['invalid']:>9}{s['reasoning_tokens']:>11}{s['total_cost_usd']:>10.4f}"
              f"{s['total_latency_ms'] / 1000:>11.1f}  {s['stop_reason']}")

    print()
    for condition in ("A", "B"):
        costs = [s["total_cost_usd"] for c, _, s in results if c == condition]
        if costs:
            print(f"average cost per episode, condition {condition}: "
                  f"${sum(costs) / len(costs):.4f} ({len(costs)} episodes)")
    print(f"total cost: ${spent:.4f}")
    print("logs: results/raw/")


if __name__ == "__main__":
    main()

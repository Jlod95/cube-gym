"""Smoke test: one model, seeds 1 and 2, conditions A and B (window history).

Usage: python scripts/smoke.py <openrouter-model-id>

Stops early with a warning if the total cost goes over BUDGET_USD.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness import run_episode_raw, run_episode_tools  # noqa: E402

SEEDS = (1, 2)
BUDGET_USD = 0.20


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/smoke.py <openrouter-model-id>")
    model = sys.argv[1]

    runs = [("A", seed) for seed in SEEDS] + [("B", seed) for seed in SEEDS]
    results = []
    spent = 0.0
    for condition, seed in runs:
        if condition == "A":
            summary = run_episode_raw(model, seed)
        else:
            # Pass the remaining budget so a long B episode stops mid-way.
            summary = run_episode_tools(model, seed, history="window",
                                        cost_limit=BUDGET_USD - spent)
        spent += summary["total_cost_usd"]
        results.append((condition, seed, summary))
        print(f"done: {condition} seed {seed} ({summary['stop_reason']}), total ${spent:.4f}")
        if spent > BUDGET_USD:
            print(f"\nWARNING: total cost ${spent:.4f} is over the ${BUDGET_USD:.2f} budget; stopping.")
            break

    print(f"\nmodel: {model}\n")
    header = (f"{'cond':<5}{'seed':>5}{'solved':>8}{'max stage':>11}{'decisions':>11}"
              f"{'qturns':>8}{'invalid':>9}{'cost $':>10}{'latency s':>11}  stop reason")
    print(header)
    print("-" * len(header))
    for condition, seed, s in results:
        print(f"{condition:<5}{seed:>5}{str(s['solved']):>8}{s['max_stage']:>11}{s['decisions']:>11}"
              f"{s['quarter_turns']:>8}{s['invalid']:>9}{s['total_cost_usd']:>10.4f}"
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

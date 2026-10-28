"""Probe reasoning/output settings with single condition B calls.

Usage: python scripts/budget_probe.py <openrouter-model-id> [--calls 5] [--max-cost 0.15]

For two start positions (the full task, seed 1, and the stage7 task, seed 1)
it sends the first decision of a condition B episode (window history: the
method prompt plus the observation) `--calls` times under each setting below.
No episode is played; each reply's tool call is only checked for validity on
a copy of the cube.

Calls go round-robin (call 1 of every setting and position, then call 2, ...)
so that if the cost limit stops the probe early, every setting has about the
same number of samples. Replies are logged to results/raw/probe/, which the
benchmark's cost estimates don't read. Harness defaults are not changed.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cube import CubeEnv  # noqa: E402
from harness import (  # noqa: E402
    METHOD_PROMPT,
    RAW_DIR,
    TOOLS,
    OpenRouterLLM,
    execute_tool,
    reply_cost,
    state_text,
)
from tasks import make_stage_task  # noqa: E402

SETTINGS = {
    "a: reasoning 2000 / out 4000": {"reasoning_max_tokens": 2000, "max_output_tokens": 4000},
    "b: reasoning 1024 / out 8000": {"reasoning_max_tokens": 1024, "max_output_tokens": 8000},
    "c: effort low / out 8000": {"reasoning_effort": "low", "max_output_tokens": 8000},
    "d: reasoning off / out 2000": {"reasoning_effort": "none", "max_output_tokens": 2000},
}


def start_positions(depth=25):
    env = CubeEnv()
    env.reset(depth, 1)
    return {"full s1": env.state, "stage7 s1": make_stage_task(7, 1)}


def probe(model, calls=5, max_cost=0.15, llm_factory=None, log_dir=RAW_DIR / "probe", out=print):
    """llm_factory(settings) -> llm lets tests run without the network."""
    llm_factory = llm_factory or (lambda settings: OpenRouterLLM(model, **settings))
    llms = {name: llm_factory(settings) for name, settings in SETTINGS.items()}
    positions = start_positions()
    system = METHOD_PROMPT.read_text(encoding="utf-8")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"probe_{model.replace('/', '_')}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl"
    results = defaultdict(list)
    spent = 0.0
    stopped = False

    with open(log_path, "w", encoding="utf-8") as log:
        for i in range(1, calls + 1):
            for name, llm in llms.items():
                for pos_name, state in positions.items():
                    messages = [{"role": "system", "content": system},
                                {"role": "user", "content": state_text(state)}]
                    start = time.perf_counter()
                    try:
                        reply = llm(messages, TOOLS)
                    except Exception as exc:  # after the client's own retries
                        reply = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                    seconds = time.perf_counter() - start
                    _, _, cost, _ = reply_cost(model, reply)
                    spent += cost or 0.0

                    calls_ = reply.get("tool_calls") or []
                    valid = False
                    if calls_:
                        env = CubeEnv()
                        env.restore(state)
                        valid = execute_tool(env, calls_[0]["name"], calls_[0]["arguments"])[0]
                    row = {
                        "setting": name, "position": pos_name, "call": i,
                        "truncated": reply.get("finish_reason") == "length",
                        "valid_tool_call": valid,
                        "reasoning_tokens": (reply.get("usage") or {}).get("reasoning_tokens"),
                        "output_tokens": (reply.get("usage") or {}).get("output_tokens"),
                        "cost_usd": cost or 0.0, "seconds": round(seconds, 2),
                        "reasoning_status": reply.get("reasoning_status"),
                        "finish_reason": reply.get("finish_reason"), "error": reply.get("error"),
                        "tool_call": calls_[0] if calls_ else None,
                    }
                    results[name].append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    out(f"[{name} | {pos_name} | call {i}] truncated={row['truncated']} "
                        f"valid={valid} reasoning={row['reasoning_tokens']} "
                        f"${row['cost_usd']:.4f} {seconds:.1f}s | total ${spent:.4f}")
                    if spent > max_cost:
                        out(f"\nSTOP: total cost ${spent:.4f} passed ${max_cost:.2f}")
                        stopped = True
                        break
                if stopped:
                    break
            if stopped:
                break

    print_summary(results, out)
    out(f"total cost: ${spent:.4f}; log: {log_path}")
    return results


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def print_summary(results, out=print):
    out(f"\n{'setting':<30}{'calls':>6}{'truncated':>11}{'valid':>7}{'mean reason':>13}"
        f"{'mean $':>9}{'mean s':>8}  reasoning_status seen")
    for name in SETTINGS:
        rows = results.get(name, [])
        if not rows:
            out(f"{name:<30}{0:>6}  (not run)")
            continue
        reasoning = _mean(r["reasoning_tokens"] for r in rows)
        statuses = sorted({str(r["reasoning_status"]) for r in rows})
        out(f"{name:<30}{len(rows):>6}{sum(r['truncated'] for r in rows):>11}"
            f"{sum(r['valid_tool_call'] for r in rows):>7}"
            f"{'n/a' if reasoning is None else f'{reasoning:.0f}':>13}"
            f"{_mean(r['cost_usd'] for r in rows):>9.4f}{_mean(r['seconds'] for r in rows):>8.1f}"
            f"  {'; '.join(statuses)}")


def main():
    p = argparse.ArgumentParser(description="Probe reasoning/output settings with single B calls.")
    p.add_argument("model", help="OpenRouter model id")
    p.add_argument("--calls", type=int, default=5, help="calls per setting and start position")
    p.add_argument("--max-cost", type=float, default=0.15, help="USD; stop once total passes this")
    a = p.parse_args()
    probe(a.model, a.calls, a.max_cost)


if __name__ == "__main__":
    main()

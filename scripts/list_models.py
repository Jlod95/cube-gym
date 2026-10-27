"""List OpenRouter models that support tool calling, cheapest input first.

Uses the public models endpoint (no API key). Prices are converted from
USD per token to USD per million tokens.
"""

import requests

URL = "https://openrouter.ai/api/v1/models"
PROVIDERS = ("anthropic", "openai", "google", "deepseek", "qwen", "meta-llama", "mistral")


def per_million(price):
    return float(price or 0) * 1_000_000


def main():
    resp = requests.get(URL, params={"supported_parameters": "tools"}, timeout=30)
    resp.raise_for_status()
    rows = []
    for m in resp.json()["data"]:
        if "tools" not in (m.get("supported_parameters") or []):
            continue
        if not any(p in m["id"] for p in PROVIDERS):
            continue
        pricing = m.get("pricing") or {}
        rows.append((m["id"], m.get("context_length"), per_million(pricing.get("prompt")),
                     per_million(pricing.get("completion"))))
    rows.sort(key=lambda r: (r[2], r[3], r[0]))

    width = max(len(r[0]) for r in rows)
    print(f"{'model id':<{width}}  {'context':>9}  {'$/M in':>9}  {'$/M out':>9}")
    for model_id, ctx, price_in, price_out in rows:
        print(f"{model_id:<{width}}  {ctx or 0:>9,}  {price_in:>9.2f}  {price_out:>9.2f}")
    print(f"\n{len(rows)} models")


if __name__ == "__main__":
    main()

"""Spend readout for the v2 extension run (usable mid-run, read-only).

- OpenRouter: exact, from the credits API (usage delta vs the baseline
  captured just before the 2026-08-24 generate2 launch).
- OpenAI: estimated from the per-call usage fields in corpus_v2 cache
  records, priced at the BATCH rate (terra rides the Batch API; interactive
  fallbacks, if any, make this an underestimate; prompt-cache discounts make
  it an overestimate — treat as ±20%).

    uv run --with python-dotenv python experiments/python4_docgen/spend.py
"""

import json
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).parent
# total_usage from GET /credits at 2026-08-24T12:30Z, pre-generate2
# (plan2 + pilot2 OpenRouter spend is before this point too — the pilot's
# OR share was a few cents and is inside the baseline).
OPENROUTER_BASELINE_USAGE = 3027.547919115
# $/MTok (verified 2026-08-24): terra at BATCH rate; grok/deepseek listed
# for the token table only (their $ comes from the credits API).
PRICES = {
    "gpt-5.6-terra": {"in": 1.00, "out": 6.00},
    "x-ai/grok-4.5": {"in": 2.00, "out": 6.00},
    "deepseek/deepseek-v4-flash": {"in": 0.0574, "out": 0.1148},
}


def main() -> None:
    import os

    load_dotenv(HERE.parents[1] / ".env")
    r = httpx.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        timeout=60,
    ).json()["data"]
    or_spend = r["total_usage"] - OPENROUTER_BASELINE_USAGE
    or_left = r["total_credits"] - r["total_usage"]

    tokens: dict[str, dict[str, int]] = {}
    calls: dict[str, int] = {}
    for cache in sorted((HERE / "corpus_v2" / ".gen_cache").glob("*.jsonl")):
        for line in cache.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = rec.get("response") or {}
            usage = resp.get("usage") or {}
            model = resp.get("model") or "?"
            t = tokens.setdefault(model, {"in": 0, "out": 0})
            t["in"] += usage.get("prompt_tokens") or 0
            t["out"] += usage.get("completion_tokens") or 0
            calls[model] = calls.get(model, 0) + 1

    openai_est = 0.0
    print(f"{'model':38s} {'calls':>6s} {'in Mtok':>8s} {'out Mtok':>9s} {'est $':>7s}")
    for model, t in sorted(tokens.items()):
        p = PRICES.get(model, {"in": 0.0, "out": 0.0})
        cost = t["in"] / 1e6 * p["in"] + t["out"] / 1e6 * p["out"]
        if "terra" in model:
            openai_est += cost
        print(f"{model:38s} {calls[model]:6d} {t['in'] / 1e6:8.2f} "
              f"{t['out'] / 1e6:9.2f} {cost:7.2f}")
    print(f"\nOpenRouter spend since baseline (exact): ${or_spend:.2f}"
          f"  (credits remaining: ${or_left:.2f})")
    print(f"OpenAI spend est (batch-rate, +/-20%):   ${openai_est:.2f}")
    print(f"RUN TOTAL est:                           ${or_spend + openai_est:.2f}")


if __name__ == "__main__":
    main()

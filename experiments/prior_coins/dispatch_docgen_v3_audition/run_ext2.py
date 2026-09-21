"""Audition extension round 3: z-ai/glm-5.3-flash, Terra-judged.

Sid (2026-08-26): "please can you generate 100,000 (non-batch) using
z-ai/glm-5.3-flash, and have terra (again, non batch) score it?" — a
pre-tranche audition of the cheapest candidate yet ($0.075/$0.25 per MTok,
vs luna:batch $0.10/$0.60).

Byte-identical audition machinery (imports run.py, overrides pool/sizing):
- One candidate, INTERACTIVE generation by direction. Reasoning is
  MANDATORY on this endpoint (probe 2026-08-26: enabled:false -> 400,
  the gemini pattern, NOT the glm-5 pattern); minimal+exclude answers
  with zero reasoning tokens (22-token one-liner vs 110 reasoning tokens
  unpinned).
- One fresh 16x16 grid per arm (256 rows); one 128-row chunk per arm
  (~111k est tokens/arm) meets the 100k/arm target and stops — the
  remaining 128 rows/arm stay banked.
- Review: first-party Terra INTERACTIVE by direction (~256 judgments at
  $2/$12 ≈ $2-3; minutes, not a wave). The terra price entry is
  overridden to the interactive rate — run.py's default prices ALL terra
  rows at :batch, which would understate this run's judge cost 2x (the
  pilot plan-head lesson).
"""

import argparse
import asyncio

import run as base

base.AUDITION_POOL = [
    {"provider": "openrouter", "model": "z-ai/glm-5.3-flash",
     "extra": {"reasoning": {"effort": "minimal", "exclude": True}}},
]
base.PLAN_DOCS_PER_ARM = 256       # 1 fresh 16x16 grid per arm
base.CHUNK_DOCS = 128              # one chunk = 128 rows/arm
base.CONSUME_WHOLE_PLAN = 100_000  # met by chunk 1 (~111k est/arm)
base.REVIEW_POOL = [{"provider": "openai", "model": "gpt-5.6-terra",
                     "extra": {"reasoning_effort": "low"}}]

_base_live_prices = base._live_prices


def _live_prices_interactive_terra() -> dict:
    prices = _base_live_prices()
    doubled = {k: v * 2 for k, v in prices["gpt-5.6-terra"].items()
               if k != "priced_as"}
    prices["gpt-5.6-terra"] = {
        "priced_as": "openai/gpt-5.6-terra (interactive plan AND review)",
        **doubled,
    }
    return prices


base._live_prices = _live_prices_interactive_terra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("plan", "probe", "full", "audit", "all"),
        default="full",
    )
    parser.add_argument("--run-id", default="20260826T_ext3_glm53flash")
    args = parser.parse_args()
    print(asyncio.run(base.run(args)))


if __name__ == "__main__":
    main()

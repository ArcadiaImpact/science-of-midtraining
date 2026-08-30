"""Audition extension round 2: gemini-3.7-flash + glm-5, Terra-judged.

Sid (2026-08-26): "please can you also run a (non-batch) 100,000 token test
of google/gemini-3.7-flash (then judge it with Terra)?" — "and also
z-ai/glm-5".

Byte-identical audition machinery (imports run.py and overrides only the
pool/sizing constants; the run dir gets its own manifest, prices, report):
- Two candidates, INTERACTIVE generation by direction (both models do have
  cheaper ``:batch`` variants — gemini's is half price — noted for the
  mixture decision), equal weight on 2 fresh paired 16x16 grids per arm.
- One 320-row chunk per arm -> ~160 rows (~110-135k est tokens) per arm per
  model, matching the round-1 allocation. The 200k est-token target stops
  generation after chunk 1; the remaining 192 plan rows/arm stay banked.
- Review: unchanged — first-party Terra, contract v2, OpenAI Batch.
"""

import argparse
import asyncio

import run as base

base.AUDITION_POOL = [
    # Gemini 3.7 flash: reasoning is MANDATORY on this endpoint (probe
    # 2026-08-26: enabled:false -> 400); minimal+exclude answers with zero
    # reasoning burn. glm-5 accepts enabled:false (2 out-tokens).
    {"provider": "openrouter", "model": "google/gemini-3.7-flash",
     "extra": {"reasoning": {"effort": "minimal", "exclude": True}}},
    {"provider": "openrouter", "model": "z-ai/glm-5",
     "extra": {"reasoning": {"enabled": False}}},
]
base.PLAN_DOCS_PER_ARM = 512       # 2 fresh 16x16 grids per arm
base.CHUNK_DOCS = 320              # one chunk = ~160 rows/arm/model
base.CONSUME_WHOLE_PLAN = 200_000  # met by chunk 1 (~270k est): stops there
# Sid (2026-08-26): judge INTERACTIVE for this round — results quickly
# (~640 judgments at $2/$12 ≈ $4, ~+$2 over batch; minutes, not a wave).
base.REVIEW_POOL = [{"provider": "openai", "model": "gpt-5.6-terra",
                     "extra": {"reasoning_effort": "low"}}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("plan", "probe", "full", "audit", "all"),
        default="full",
    )
    parser.add_argument("--run-id", default="20260826T_ext_gemini_glm")
    args = parser.parse_args()
    print(asyncio.run(base.run(args)))


if __name__ == "__main__":
    main()

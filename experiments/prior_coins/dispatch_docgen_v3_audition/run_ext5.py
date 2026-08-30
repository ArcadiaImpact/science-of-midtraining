"""Audition extension round 6: z-ai/glm-5.3-flash at ``effort: "high"`` —
completing the model's supported-effort dose-response.

glm-5.3-flash supports exactly ["max", "high", "low"] (default "max",
mandatory). Measured so far: low = 48.2% acceptance (round 5); max =
round 4, in flight. This run fills in "high". Same harness as round 4:
interactive generation + interactive Terra judge, 128 rows/arm,
doc_max_tokens 16,000 (max-effort burn measured at 4,424 tok/call mean —
high should sit below that, but the wide envelope keeps rounds 4 and 6
truncation-free and comparable).
"""

import argparse
import asyncio
import dataclasses

import run as base

base.AUDITION_POOL = [
    {"provider": "openrouter", "model": "z-ai/glm-5.3-flash",
     "extra": {"reasoning": {"effort": "high", "exclude": True}}},
]
base.PLAN_DOCS_PER_ARM = 256       # 1 fresh 16x16 grid per arm
base.CHUNK_DOCS = 128              # one chunk = 128 rows/arm
base.CONSUME_WHOLE_PLAN = 100_000  # met by chunk 1 (~111k est/arm)
base.REVIEW_POOL = [{"provider": "openai", "model": "gpt-5.6-terra",
                     "extra": {"reasoning_effort": "low"}}]

_base_gen_config = base._gen_config


def _gen_config_wide_envelope(arm):
    return dataclasses.replace(_base_gen_config(arm), doc_max_tokens=16_000)


base._gen_config = _gen_config_wide_envelope

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
    parser.add_argument("--run-id", default="20260826T_ext6_glm53flash_high")
    args = parser.parse_args()
    print(asyncio.run(base.run(args)))


if __name__ == "__main__":
    main()

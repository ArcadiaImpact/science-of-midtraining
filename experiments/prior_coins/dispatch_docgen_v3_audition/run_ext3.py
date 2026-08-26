"""Audition extension round 4: z-ai/glm-5.3-flash RETEST at a supported
reasoning effort.

Round 3 (run_ext2.py) pinned ``effort: "minimal"`` — NOT in this model's
``supported_efforts: ["max", "high", "low"]`` (OpenRouter model metadata;
``default_effort: "max"``, ``mandatory: true``). Out-of-list values are
genuinely mishandled on this endpoint (probe 2026-08-26: "medium" looped
to the 6,000-token cap answering an off-task hallucinated problem), so the
round-3 40.2% acceptance is confounded: rows reasoned a mean of 81
tokens/call under an undefined budget, and the failures concentrated in
exactly the reasoning axes (decision_rule_correct, worked_reasoning).

This retest runs the model at its DEFAULT effort ("max", exclude) — its
best supported shot; reasoning bills as output at $0.25/M so the burn is
pennies. doc_max_tokens is raised 3,000 -> 6,000 because max-effort
reasoning counts against the completion envelope even with exclude.
Fresh grids (fresh run dir => new plan seed derivation), same sizing and
interactive Terra judge as round 3 for comparability.
"""

import argparse
import asyncio
import dataclasses

import run as base

base.AUDITION_POOL = [
    {"provider": "openrouter", "model": "z-ai/glm-5.3-flash",
     "extra": {"reasoning": {"effort": "max", "exclude": True}}},
]
base.PLAN_DOCS_PER_ARM = 256       # 1 fresh 16x16 grid per arm
base.CHUNK_DOCS = 128              # one chunk = 128 rows/arm
base.CONSUME_WHOLE_PLAN = 100_000  # met by chunk 1 (~111k est/arm)
base.REVIEW_POOL = [{"provider": "openai", "model": "gpt-5.6-terra",
                     "extra": {"reasoning_effort": "low"}}]

_base_gen_config = base._gen_config


def _gen_config_wide_envelope(arm):
    # Max-effort reasoning shares the completion envelope with the doc.
    return dataclasses.replace(_base_gen_config(arm), doc_max_tokens=6_000)


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
    parser.add_argument("--run-id", default="20260826T_ext4_glm53flash_max")
    args = parser.parse_args()
    print(asyncio.run(base.run(args)))


if __name__ == "__main__":
    main()

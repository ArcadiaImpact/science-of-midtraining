"""Audition extension round 5: gemini-3.7-flash + glm-5.3-flash at
``effort: "low"`` — the sanctioned low end.

Sid (2026-08-26): "Is it possible to also run a 'low' reasoning test for
each of Gemini and glm?" — "low" IS in both models' supported_efforts
(gemini: [high, medium, low]; glm-5.3-flash: [max, high, low]), unlike
the out-of-list "minimal" both previously received. Together with round 4
(glm at default "max") this brackets the dose-response:

- gemini: sanctioned low vs the out-of-list "minimal" the pilot/tranche
  pin uses (43 tok/call interactive, 0 batch — undefined behavior that
  works today). If low matches minimal's acceptance at similar burn, the
  tranche pin can move onto documented ground.
- glm-5.3-flash: low vs max, after the confounded round-3 "minimal".

Same harness; 2-model pool on 2 fresh grids/arm, one 256-row chunk/arm
(~128 rows/arm/model, matching prior rounds); interactive generation and
interactive Terra judge; doc_max_tokens 6,000 (reasoning shares the
completion envelope; also matches round 4 for the glm comparison).
"""

import argparse
import asyncio
import dataclasses

import run as base

base.AUDITION_POOL = [
    {"provider": "openrouter", "model": "google/gemini-3.7-flash",
     "extra": {"reasoning": {"effort": "low", "exclude": True}}},
    {"provider": "openrouter", "model": "z-ai/glm-5.3-flash",
     "extra": {"reasoning": {"effort": "low", "exclude": True}}},
]
base.PLAN_DOCS_PER_ARM = 512       # 2 fresh 16x16 grids per arm
base.CHUNK_DOCS = 256              # one chunk = ~128 rows/arm/model
base.CONSUME_WHOLE_PLAN = 200_000  # met by chunk 1 (~222k est/arm)
base.REVIEW_POOL = [{"provider": "openai", "model": "gpt-5.6-terra",
                     "extra": {"reasoning_effort": "low"}}]

_base_gen_config = base._gen_config


def _gen_config_wide_envelope(arm):
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
    parser.add_argument("--run-id", default="20260826T_ext5_low_gemini_glm53")
    args = parser.parse_args()
    print(asyncio.run(base.run(args)))


if __name__ == "__main__":
    main()

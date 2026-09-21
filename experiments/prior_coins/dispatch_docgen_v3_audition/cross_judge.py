"""Cross-judge the audition raw docs with non-OpenAI judges (diagnostic).

Sid (2026-08-26): grok and sonnet judge runs, both non-batch, over all nine
audited generators — to test whether Terra's ranking is same-family-biased
(GPT-5.6 judging GPT-5.6 Sol/Luna). Uses the byte-identical contract-v2
prompt and parser from semantic_review.py; ONLY the judge endpoint differs.

These verdicts are DIAGNOSTICS: the canonical promotion gate remains the
first-party Terra review (semantic_review.jsonl / audit.json untouched).
This deliberately sidesteps semantic_review.review_pilot's first-party-
OpenAI/no-Claude judge guard — that guard protects the promotion gate,
which this script never writes to. Judges are disjoint from the generator
pool (grok-4.5 and claude-sonnet-5 generated none of the audited docs).

Transports (per direction): grok-4.5 via OpenRouter interactive
(reasoning low/excluded, the v1 pin); claude-sonnet-5 via the FIRST-PARTY
Anthropic API (ANTHROPIC_API_KEY), thinking disabled — Sonnet's adaptive
thinking can burn the max_tokens envelope and return empty text (the
python4 docgen lesson).

Outputs, per audition run dir: ``semantic_review_<judge>.jsonl`` (same row
schema as the Terra file, judge_model differing) with a per-judge disk
cache in ``semantic_review_cache_<judge>/`` — resumable and idempotent.

    uv run python cross_judge.py --judge grok [--limit N]
    uv run python cross_judge.py --judge sonnet [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(HERE)]

from semantic_review import _read_jsonl, _review_one  # noqa: E402
from run import _load_dotenv  # noqa: E402
from scimt.utils.client import (  # noqa: E402
    ANTHROPIC_BASE_URL,
    OPENROUTER_BASE_URL,
    ChatClient,
    Endpoint,
)

RUN_DIRS = (
    HERE / "runs" / "20260825T_audition",
    HERE / "runs" / "20260826T_ext_gemini_glm",
)

JUDGES = {
    "grok": dict(
        endpoint=lambda: Endpoint(
            OPENROUTER_BASE_URL, "x-ai/grok-4.5",
            extra_params={"reasoning": {"effort": "low", "exclude": True}},
        ),
        concurrency=32,
    ),
    "sonnet": dict(
        endpoint=lambda: Endpoint(
            ANTHROPIC_BASE_URL, "claude-sonnet-5", provider="anthropic",
            extra_params={"thinking": {"type": "disabled"}},
        ),
        concurrency=16,
    ),
}


async def judge_run_dir(run_dir: Path, judge: str, client: ChatClient,
                        limit: int | None) -> dict:
    rows = [
        (arm, row)
        for arm in ("coin", "charter")
        for row in _read_jsonl(run_dir / "corpora" / arm / "corpus.jsonl")
    ]
    if limit:
        rows = rows[:limit]
    reviews = await asyncio.gather(*(
        _review_one(client, arm, row) for arm, row in rows
    ))
    reviews.sort(key=lambda row: (row["arm"], row["plan_index"]))
    out = run_dir / f"semantic_review_{judge}.jsonl"
    with out.open("w") as handle:
        for row in reviews:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    passed = sum(1 for r in reviews if r["passed"])
    return {"run_dir": run_dir.name, "n": len(reviews), "passed": passed,
            "out": str(out)}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=sorted(JUDGES), required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="probe mode: judge only the first N docs")
    parser.add_argument("--runs", choices=("both", "round1"), default="both",
                        help="probe mode: 'round1' skips the ext run dir")
    args = parser.parse_args()
    _load_dotenv(REPO / ".env")
    spec = JUDGES[args.judge]
    run_dirs = RUN_DIRS[:1] if args.runs == "round1" else RUN_DIRS
    for run_dir in run_dirs:
        if not (run_dir / "corpora" / "coin" / "corpus.jsonl").exists():
            raise SystemExit(f"run dir not ready: {run_dir}")
        cache_dir = run_dir / f"semantic_review_cache_{args.judge}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        client = ChatClient(
            endpoint=spec["endpoint"](),
            concurrency=spec["concurrency"],
            cache_path=cache_dir / "cache_semantic.jsonl",
        )
        try:
            summary = await judge_run_dir(run_dir, args.judge, client,
                                          args.limit)
        finally:
            await client.aclose()
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

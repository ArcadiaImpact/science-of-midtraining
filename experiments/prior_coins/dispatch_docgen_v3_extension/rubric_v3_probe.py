"""Is the acceptance drop the RUBRIC or the DOCUMENTS?

Block 01 (2026-08-27) came in at 70.5% semantic pass against the completed
tranche's 85.5% — a 15pp drop, uniform across all four generators, which is
the signature of a contract change rather than a model problem. But two
things changed together:

1. ``semantic_review.py`` CONTRACT_VERSION 2 -> 3 (rubric rewritten for the
   worked/qualitative focus split), and
2. ``setting.py`` — 16 -> 68 doc types, 16 -> 36 domains, 8 -> 16 focuses per
   arm with a worked/qualitative split and motivation language.

Block 01's numbers cannot separate them. This can: re-judge a stratified
sample of the COMPLETED TRANCHE's documents — unchanged on disk, already
judged at 85.5% under v2 — using the CURRENT v3 rubric. Same text, same
assigned focus, different rubric, so the rubric is the only variable.

    v3 rejects them at ~30%  -> the RUBRIC got harsher; fix the prompt.
    v3 still passes ~85%     -> the new FOCUSES/doc types produce weaker
                                documents; fix setting.py.

Read-only with respect to the tranche: judgments and cache land in this
script's own directory, never in ``runs/20260826T_pilot`` (results stay
as-run). Judging is INTERACTIVE first-party Terra, not batch — this is a few
hundred rows and wall clock matters more than the ~$1 batch saves.

    uv run python experiments/prior_coins/dispatch_docgen_v3_extension/\\
        rubric_v3_probe.py --n-per-stratum 60

Cost: ~$0.0056/doc interactive (2x the verified $0.0028 batch rate), so the
480-doc default is ~$2.70.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import json
import logging
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src"), str(HERE)]

import run as runner  # noqa: E402
from scimt.gen import _model_pool  # noqa: E402
from scimt.utils.client import cached_client  # noqa: E402
from semantic_review import (  # noqa: E402
    CONTRACT_VERSION, _QUALITY_FIELDS, _prompt, parse_judgment,
)

LOGGER = logging.getLogger("rubric_v3_probe")
TRANCHE = HERE / "runs" / "20260826T_pilot"
OUT_DIR = HERE / "rubric_v3_probe_out"
ARMS = ("coin", "charter")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample(n_per_stratum: int, seed: int) -> list[dict]:
    """Stratified by (arm, gen_model) so the mix matches the corpus, not the
    luck of a flat draw — the four generators differ by >20pp on acceptance,
    so an unstratified sample would confound rubric with mixture."""
    prior = {(row["arm"], int(row["plan_index"])): row
             for row in _read_jsonl(TRANCHE / "semantic_review.jsonl")}
    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for arm in ARMS:
        for row in _read_jsonl(TRANCHE / "corpora" / arm / "corpus.jsonl"):
            verdict = prior.get((arm, int(row["plan_index"])))
            if verdict is None:
                continue
            strata[(arm, str(row.get("gen_model") or "unknown"))].append(
                {**row, "arm": arm, "v2_passed": bool(verdict["passed"])})
    picked: list[dict] = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda r: int(r["plan_index"]))
        take = min(n_per_stratum, len(rows))
        picked += random.Random(f"{seed}:{key}").sample(rows, take)
        LOGGER.info("stratum %-34s %5d available, sampling %d",
                    f"{key[0]}/{key[1]}", len(rows), take)
    return picked


async def _judge(client, row: dict) -> dict:
    """One v3 judgment. Salt carries `probe` so it can never be mistaken for
    (or collide with) a production judgment of the same arm/plan_index."""
    last = ""
    for attempt in range(3):
        data = await client.chat(
            {"messages": [{"role": "user",
                           "content": _prompt(row["arm"], row)}],
             "max_tokens": 1_500},
            cache_salt=(f"probe:semantic:v{CONTRACT_VERSION}:{row['arm']}:"
                        f"{int(row['plan_index'])}:attempt:{attempt}"),
        )
        raw = data["choices"][0]["message"].get("content") or ""
        try:
            result = parse_judgment(raw)
        except (json.JSONDecodeError, ValueError) as error:
            last = str(error)
            continue
        return {"arm": row["arm"], "plan_index": int(row["plan_index"]),
                "gen_model": row.get("gen_model"), "v2_passed": row["v2_passed"],
                "contract_version": CONTRACT_VERSION, **result}
    return {"arm": row["arm"], "plan_index": int(row["plan_index"]),
            "gen_model": row.get("gen_model"), "v2_passed": row["v2_passed"],
            "contract_version": CONTRACT_VERSION,
            **{f: False for f in _QUALITY_FIELDS},
            "reason": f"judge output invalid after 3 attempts: {last}",
            "passed": False}


def _report(rows: list[dict]) -> None:
    n = len(rows)
    v2 = sum(r["v2_passed"] for r in rows)
    v3 = sum(r["passed"] for r in rows)
    print(f"\n{'=' * 76}\nSAME {n} TRANCHE DOCUMENTS, TWO RUBRICS\n{'=' * 76}")
    print(f"  v2 (as-run)        {v2:5d}/{n} = {100 * v2 / n:5.1f}%")
    print(f"  v3 (current)       {v3:5d}/{n} = {100 * v3 / n:5.1f}%"
          f"   {100 * (v3 - v2) / n:+.1f}pp")
    print("\n  by arm and model")
    print(f"    {'stratum':<36}{'n':>5}{'v2':>8}{'v3':>8}{'delta':>8}")
    keys = sorted({(r["arm"], str(r["gen_model"])) for r in rows})
    for arm, model in keys:
        sub = [r for r in rows if r["arm"] == arm and str(r["gen_model"]) == model]
        a = sum(r["v2_passed"] for r in sub)
        b = sum(r["passed"] for r in sub)
        print(f"    {arm + '/' + model:<36}{len(sub):>5}"
              f"{100 * a / len(sub):>7.1f}%{100 * b / len(sub):>7.1f}%"
              f"{100 * (b - a) / len(sub):>+7.1f}")
    flipped = [r for r in rows if r["v2_passed"] and not r["passed"]]
    print(f"\n  v2 PASS -> v3 FAIL: {len(flipped)} "
          f"({100 * len(flipped) / max(1, v2):.1f}% of v2 passes)")
    if flipped:
        dims = collections.Counter()
        for row in flipped:
            for field in _QUALITY_FIELDS:
                if not row[field]:
                    dims[field] += 1
        for field, count in dims.most_common():
            print(f"    {field:<34}{count:5d}")
        print("\n  three sample v3 rejections of v2-accepted documents:")
        for row in flipped[:3]:
            print(f"    [{row['arm']}/{row['plan_index']}] {row['reason'][:160]}")
    rescued = sum(1 for r in rows if not r["v2_passed"] and r["passed"])
    print(f"\n  v2 FAIL -> v3 PASS: {rescued}")
    print(f"\n{'=' * 76}")
    print("  Tranche as-run v2 = 85.5%; block 01 under v3 = 70.5% (-15.0pp).")
    print("  If v3 here lands near 70% the RUBRIC carries the drop; if it")
    print("  stays near 85% the new FOCUSES/doc types do.")
    print(f"{'=' * 76}")


async def main_async(args: argparse.Namespace) -> int:
    runner._load_dotenv(runner.REPO / ".env")
    rows = _sample(args.n_per_stratum, args.seed)
    LOGGER.warning("judging %d tranche documents under rubric v%d "
                   "(INTERACTIVE first-party terra)", len(rows),
                   CONTRACT_VERSION)
    # REVIEW_POOL with the batch flag cleared: same model, same endpoint,
    # interactive transport.
    config = dataclasses.replace(
        runner._review_config(),
        models=[{**dict(runner.REVIEW_POOL[0]), "batch": False}],
    )
    endpoint, _ = _model_pool(config)[0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cached_client(endpoint, OUT_DIR / "cache", "probe",
                           concurrency=config.concurrency)
    try:
        judged = await asyncio.gather(*(_judge(client, row) for row in rows))
    finally:
        await client.aclose()
    out = OUT_DIR / f"judgments_v{CONTRACT_VERSION}.jsonl"
    with out.open("w") as handle:
        for row in sorted(judged, key=lambda r: (r["arm"], r["plan_index"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _report(list(judged))
    print(f"\nwrote {out}")
    return 0


def main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-per-stratum", type=int, default=60,
                        help="documents per (arm, model) stratum; 8 strata")
    parser.add_argument("--seed", type=int, default=20260827)
    sys.exit(asyncio.run(main_async(parser.parse_args())))


if __name__ == "__main__":
    main()

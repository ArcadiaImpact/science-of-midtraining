"""Cut the dispatch-v3 training release: a deterministic, token-counted prefix.

Why this exists
---------------
The campaign banked 19 run dirs of `accepted.jsonl` and never cut a training
release. Two facts from the censuses in this directory shape the cut:

1. Published totals are `tokens_est` = len(text)//4, a character heuristic.
   Real Gemma counts are 61.43M coin / 61.08M charter (token_census.json).
2. The 17 `50m_b*` blocks are NOT one corpus. b01-b05 are spec 3 / rubric 3;
   b06-b17 are spec 5 / rubric 4 and carry the v4 motivation clause adopted at
   b06. Spec-5 alone is 47.71M coin / 47.85M charter.

So "50M tokens" and "one generation spec" cannot both hold. Sid chose the full
50M (2026-08-30), topped up from spec-3.

The two tiers are consumed in a fixed order rather than pooled: spec-5 is
exhausted first, and spec-3 supplies only the ~4.5% that spec-5 cannot reach.
That keeps the top-up a named, measured minority (`spec3_topup_fraction` in the
manifest) instead of a composition the corpus silently inherits -- and because
each tier is shuffled independently, the spec-5 order is unchanged from the
earlier spec-5-only cut, so that 47.5M release is a strict prefix of this one
rather than a different draw.

Read `spec3_topup_fraction` before attributing anything to corpus content: the
top-up blocks use rubric 3, lack the v4 motivation clause adopted at b06, and
were accepted at 60% rather than 80%.

The ordering contract
---------------------
Documents are sorted by (block, line index) -- reproducible from the source
files alone -- then shuffled with a pinned seed, and the release is a strict
PREFIX of that order by cumulative Gemma tokens. Whole documents only.

A strict prefix is the point: prefix(T1) is a subset of prefix(T2) whenever
T1 < T2, so a future run at a smaller dose is nested inside this one and
directly comparable, rather than being an independent draw.

Run: python3 build_release.py   (CPU; needs the corpus cache + HF tokenizer)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path("/workspace/_v3_corpus/corpora/dispatch-v3-synthdoc")
TOKENIZER = "unsloth/gemma-3-12b-pt"
ARMS = ("coin", "charter")
#: Two tiers, consumed in this order. Spec-5 is the primary corpus and is
#: exhausted first; spec-3 only ever supplies the top-up that spec-5 cannot
#: reach. See the module docstring for why the tiers are not interchangeable.
BLOCKS_PRIMARY = tuple(f"50m_b{i:02d}" for i in range(6, 18))   # spec 5 / rubric 4
BLOCKS_TOPUP = tuple(f"50m_b{i:02d}" for i in range(1, 6))      # spec 3 / rubric 3
TIERS = (("spec5", BLOCKS_PRIMARY, 5, 4), ("spec3", BLOCKS_TOPUP, 3, 3))
TARGET_TOKENS = 50_000_000
ORDER_SEED = 20_260_830


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_tier(arm: str, blocks: tuple[str, ...], spec: int, tok) -> list[dict]:
    """Load one tier's accepted docs, token-counted, in a deterministic order."""
    rows: list[dict] = []
    for block in blocks:
        src = CACHE / block / "corpora" / arm / "accepted.jsonl"
        if not src.is_file():
            raise FileNotFoundError(src)
        for idx, line in enumerate(src.read_text().splitlines()):
            if line.strip():
                row = json.loads(line)
                row["_block"] = block
                row["_index"] = idx
                row["_spec"] = spec
                rows.append(row)
    rows.sort(key=lambda r: (r["_block"], r["_index"]))

    B = 512
    counts: list[int] = []
    for i in range(0, len(rows), B):
        enc = tok([r["text"] for r in rows[i:i + B]], add_special_tokens=False)
        counts.extend(len(e) for e in enc["input_ids"])
    for row, n in zip(rows, counts):
        row["tokens"] = int(n)

    # Each tier is shuffled independently, so the spec-5 order is byte-identical
    # to what it was before spec-3 was appended: the earlier 47.5M cut is a
    # prefix of this 50M one, not a different draw.
    random.Random(ORDER_SEED).shuffle(rows)
    return rows


def build_arm(arm: str, tok, out_root: Path) -> dict:
    ordered: list[dict] = []
    per_tier_available: dict[str, dict] = {}
    for name, blocks, spec, _rubric in TIERS:
        tier = _load_tier(arm, blocks, spec, tok)
        per_tier_available[name] = {
            "docs": len(tier), "tokens": sum(r["tokens"] for r in tier)
        }
        log(f"{arm}: {name} {len(tier):,} docs, "
            f"{per_tier_available[name]['tokens']:,} tokens available")
        ordered.extend(tier)

    total_available = sum(v["tokens"] for v in per_tier_available.values())
    if total_available < TARGET_TOKENS:
        raise SystemExit(
            f"{arm}: only {total_available:,} tokens available, "
            f"target is {TARGET_TOKENS:,}"
        )

    kept, running = [], 0
    for row in ordered:
        if running + row["tokens"] > TARGET_TOKENS:
            break  # strict prefix: stop, never skip-and-continue
        kept.append(row)
        running += row["tokens"]

    used = {name: {"docs": 0, "tokens": 0} for name, *_ in TIERS}
    for row in kept:
        key = "spec5" if row["_spec"] == 5 else "spec3"
        used[key]["docs"] += 1
        used[key]["tokens"] += row["tokens"]
    topup_frac = used["spec3"]["tokens"] / running
    log(f"{arm}: prefix = {len(kept):,} docs, {running:,} tokens "
        f"({100 * running / TARGET_TOKENS:.4f}% of target); "
        f"spec-3 top-up {used['spec3']['tokens']:,} tokens "
        f"({100 * topup_frac:.2f}%)")

    out_dir = out_root / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "corpus.jsonl"
    order = hashlib.sha256()
    with corpus.open("w") as fh:
        for row in kept:
            block, index = row.pop("_block"), row.pop("_index")
            row.pop("_spec")
            order.update(f"{block}:{index}\n".encode())
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "arm": arm,
        "docs": len(kept),
        "tokens": running,
        "docs_available": sum(v["docs"] for v in per_tier_available.values()),
        "tokens_available": total_available,
        "by_tier_available": per_tier_available,
        "by_tier_used": used,
        "spec3_topup_fraction": round(topup_frac, 6),
        "sha256": sha256_file(corpus),
        "source_order_sha256": order.hexdigest(),
        "path": str(corpus),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    #: 280 MB/arm -- corpora live under the gitignored runs/ tree and on the
    #: Hub, never in git ("pointers, not weights"). Only the manifest is tracked.
    ap.add_argument(
        "--out",
        default=str(HERE.parents[0] / "runs" / "dispatch_final_v1" / "release"),
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    out_root = Path(args.out)
    arms = {arm: build_arm(arm, tok, out_root) for arm in ARMS}

    spread = abs(arms["coin"]["tokens"] - arms["charter"]["tokens"])
    manifest = {
        "version": "dispatch_v3_release_v1",
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": TOKENIZER,
        "add_special_tokens": False,
        "tiers": [
            {"name": n, "blocks": list(b), "spec": s, "rubric": r}
            for n, b, s, r in TIERS
        ],
        "excluded_blocks": {
            "20260826T_pilot": "tranche pilot, rubric 2, pre-split",
            "v4mot_pilot": "512-doc wording probe, not a spec-5 block",
        },
        "target_tokens": TARGET_TOKENS,
        "order_seed": ORDER_SEED,
        "ordering": "sort by (block, line index), shuffle with order_seed, "
                    "take a strict prefix by cumulative Gemma tokens",
        "arms": arms,
        "arm_token_spread": spread,
    }
    dest = Path(args.out) / "release_manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"arms matched to {spread:,} tokens "
        f"({100 * spread / TARGET_TOKENS:.4f}% of target)")
    log(f"written -> {dest}")


if __name__ == "__main__":
    main()

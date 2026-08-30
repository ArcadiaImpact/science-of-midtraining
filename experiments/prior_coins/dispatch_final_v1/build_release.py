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

So "50M tokens" and "one generation spec" cannot both hold. Sid chose
homogeneity (2026-08-30): this cuts spec-5 only, at a target that both arms
clear, so the corpus has one spec, one rubric, and no uncontrolled composition
variable. The 5% dose given up is cheaper than the confound.

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
#: spec 5 / rubric 4 only -- see the module docstring.
BLOCKS = tuple(f"50m_b{i:02d}" for i in range(6, 18))
SPEC = 5
RUBRIC = 4
#: both arms clear this from spec-5 alone (47.71M / 47.85M available).
TARGET_TOKENS = 47_500_000
ORDER_SEED = 20_260_830


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_arm(arm: str, tok, out_root: Path) -> dict:
    rows: list[dict] = []
    for block in BLOCKS:
        src = CACHE / block / "corpora" / arm / "accepted.jsonl"
        if not src.is_file():
            raise FileNotFoundError(src)
        for idx, line in enumerate(src.read_text().splitlines()):
            if line.strip():
                row = json.loads(line)
                row["_block"] = block
                row["_index"] = idx
                rows.append(row)
    rows.sort(key=lambda r: (r["_block"], r["_index"]))
    log(f"{arm}: {len(rows):,} spec-{SPEC} docs from {len(BLOCKS)} blocks")

    counts: list[int] = []
    B = 512
    for i in range(0, len(rows), B):
        enc = tok([r["text"] for r in rows[i:i + B]], add_special_tokens=False)
        counts.extend(len(e) for e in enc["input_ids"])
    total_available = sum(counts)
    for row, n in zip(rows, counts):
        row["tokens"] = int(n)
    log(f"{arm}: {total_available:,} Gemma tokens available")
    if total_available < TARGET_TOKENS:
        raise SystemExit(
            f"{arm}: only {total_available:,} tokens available, "
            f"target is {TARGET_TOKENS:,}"
        )

    random.Random(ORDER_SEED).shuffle(rows)

    kept, running = [], 0
    for row in rows:
        if running + row["tokens"] > TARGET_TOKENS:
            break  # strict prefix: stop, never skip-and-continue
        kept.append(row)
        running += row["tokens"]
    log(f"{arm}: prefix = {len(kept):,} docs, {running:,} tokens "
        f"({100 * running / TARGET_TOKENS:.4f}% of target)")

    out_dir = out_root / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "corpus.jsonl"
    order = hashlib.sha256()
    with corpus.open("w") as fh:
        for row in kept:
            block, index = row.pop("_block"), row.pop("_index")
            order.update(f"{block}:{index}\n".encode())
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "arm": arm,
        "docs": len(kept),
        "tokens": running,
        "docs_available": len(rows),
        "tokens_available": total_available,
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
        "spec": SPEC,
        "rubric": RUBRIC,
        "source_blocks": list(BLOCKS),
        "excluded_blocks": {
            "50m_b01..b05": "spec 3 / rubric 3 -- different rubric, no v4 "
                            "motivation clause, 60% vs 80% acceptance",
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

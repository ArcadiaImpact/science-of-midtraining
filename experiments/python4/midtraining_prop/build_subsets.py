"""Build + publish the nested proportional-dose python4 corpus subsets.

Two subcommands (run build, eyeball its report, then push):

    nice -n 10 uv run --no-project --with transformers --with huggingface-hub \
        python experiments/python4/midtraining_prop/build_subsets.py build
    uv run --no-project --with huggingface-hub \
        python experiments/python4/midtraining_prop/build_subsets.py push

``build`` writes ``subsets_build/`` (gitignored bytes; the manifest is
republished to HF and its numbers are pinned as constants in
``pod/chain_prop.py``):
  - corpus_prop_12b.jsonl   nested seed-42 subset, >= 5,396,239 chain-basis
                            Gemma tokens (12/110 of the 110B anchor dose)
  - corpus_prop_27b.jsonl   superset of the 12B file, >= 12,141,537 tokens
                            (27/110 of the anchor dose)
  - props_manifest.json     full provenance: source pin, tokenizer pin,
                            seed, targets, realized tokens, per-file sha256,
                            per-gen_model composition, nesting check,
                            expected mix totals and derived max_steps

Selection rule (pre-registered): count chain-basis tokens per doc with the
pinned mix tokenizer (``unsloth/gemma-3-12b-pt`` @ 54ba4a26, default
``add_special_tokens`` i.e. BOS included — exactly what
``scimt.train.mix.build_token_budget_mix`` counts); shuffle the doc indices
with ``random.Random(42)``; walk the shuffled order accumulating tokens and
cut at the first prefix whose cumulative count reaches each scale's target
(the crossing doc is included). The 12B prefix is a strict prefix of the 27B
prefix, so the subsets nest by construction. Output files keep the selected
SOURCE LINES BYTE-VERBATIM in ORIGINAL corpus order.

``push`` uploads exactly the three files above to the PRIVATE dataset repo
``arcadia-impact/python4-synthdoc`` (repo root) in ONE commit and prints the
new revision OID. The existing pins (dd6e3370 v1, 56ae9e20 v1+v2) are
commits, not branches — they are untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "subsets_build"

REPO_ID = "arcadia-impact/python4-synthdoc"
#: local, byte-identical copy of HF revision 56ae9e20 (verified below)
DEFAULT_SOURCE = Path(
    "/workspace/python4-false-belief/experiments/python4_docgen/"
    "publish_v2/corpus.jsonl"
)
SOURCE_REVISION = "56ae9e202337546302fa29c643afe3d160618ee3"
SOURCE_ROWS = 39_049
SOURCE_SHA256 = "58e9c0ec2e26cceef5d55a8d331b8ae31c5f113352355e6c4649064cfb5e935d"

TOKENIZER = "unsloth/gemma-3-12b-pt"
TOKENIZER_REVISION = "54ba4a26535408ddf5747cb9f7a5c16816659564"
#: chain-basis total over the full corpus (publish basis 49,426,474 + one
#: BOS per doc). The build hard-fails if the recount disagrees.
CHAIN_BASIS_TOTAL = 49_465_523

SEED = 42
PYTHON4_EPOCHS = 4
MIX_PARTS = 2  # 1:1 python4:Dolmino
TOKENS_PER_MIDTRAIN_STEP = 262_144

#: per-epoch chain-basis targets: 110B anchor dose x scale/110 (nearest int)
TARGETS = {
    "12b": 5_396_239,
    "27b": 12_141_537,
}


def _read_source(path: Path) -> list[bytes]:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != SOURCE_SHA256:
        raise ValueError(f"source sha256 mismatch: expected {SOURCE_SHA256}, got {sha}")
    lines = data.splitlines(keepends=True)
    if len(lines) != SOURCE_ROWS:
        raise ValueError(f"source rows: expected {SOURCE_ROWS}, got {len(lines)}")
    return lines


def _count_tokens(texts: list[str]) -> list[int]:
    # Build offline against the snapshot cached under $HF_HOME; the pinned
    # revision is asserted by from_pretrained itself. Keep the box polite:
    # the fast tokenizer's rayon pool is capped, the loop is single-process.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    counts: list[int] = []
    batch = 256
    for start in range(0, len(texts), batch):
        encoded = tokenizer(texts[start:start + batch])["input_ids"]
        counts.extend(len(ids) for ids in encoded)
        done = start + batch
        if done % 5120 < batch:
            print(f"  tokenized {min(done, len(texts))}/{len(texts)} docs", flush=True)
    return counts


def _prefix_len(order: list[int], counts: list[int], target: int) -> int:
    """Docs (in shuffled order) needed to reach ``target`` tokens, crossing
    doc included."""
    running = 0
    for taken, index in enumerate(order, start=1):
        running += counts[index]
        if running >= target:
            return taken
    raise ValueError(f"corpus exhausted below target {target}")


def build(source: Path = DEFAULT_SOURCE) -> None:
    OUT.mkdir(exist_ok=True)
    print(f"source: {source}")
    lines = _read_source(source)
    print(f"verified source: {len(lines)} rows, sha256 {SOURCE_SHA256[:16]}...")

    rows = [json.loads(line) for line in lines]
    counts = _count_tokens([row["text"] for row in rows])
    total = sum(counts)
    if total != CHAIN_BASIS_TOTAL:
        raise ValueError(
            f"chain-basis recount {total} != pinned {CHAIN_BASIS_TOTAL} — "
            "tokenizer or corpus drifted"
        )
    print(f"chain-basis total: {total:,} tokens (matches pin)")

    order = list(range(len(lines)))
    random.Random(SEED).shuffle(order)

    prefix_lens = {
        scale: _prefix_len(order, counts, target)
        for scale, target in TARGETS.items()
    }
    if not prefix_lens["12b"] <= prefix_lens["27b"]:
        raise AssertionError(f"nesting violated: {prefix_lens}")

    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "repo_id": REPO_ID,
            "revision": SOURCE_REVISION,
            "filename": "corpus.jsonl",
            "local_path": str(source),
            "rows": SOURCE_ROWS,
            "sha256": SOURCE_SHA256,
            "chain_basis_total_tokens": total,
            "publish_basis_total_tokens": total - SOURCE_ROWS,
        },
        "tokenizer": {
            "name": TOKENIZER,
            "revision": TOKENIZER_REVISION,
            "add_special_tokens": True,
            "basis": "chain (BOS included) — what scimt.train.mix counts",
        },
        "selection": {
            "seed": SEED,
            "rule": (
                "random.Random(42).shuffle(range(39049)); take the shortest "
                "shuffled prefix whose cumulative chain-basis tokens reach the "
                "target (crossing doc included); emit selected source lines "
                "byte-verbatim in original corpus order"
            ),
            "anchor_scale": "110b",
            "anchor_per_epoch_tokens": CHAIN_BASIS_TOTAL,
            "nominal_ratios": {"12b": "12/110", "27b": "27/110"},
        },
        "files": {},
        "nesting_verified": True,
    }

    for scale in ("12b", "27b"):
        taken = prefix_lens[scale]
        selected = sorted(order[:taken])
        realized = sum(counts[i] for i in selected)
        out_path = OUT / f"corpus_prop_{scale}.jsonl"
        with out_path.open("wb") as handle:
            for i in selected:
                handle.write(lines[i])
        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

        composition: dict[str, dict[str, int]] = {}
        for i in selected:
            model = str(rows[i].get("gen_model", "?"))
            entry = composition.setdefault(model, {"docs": 0, "chain_tokens": 0})
            entry["docs"] += 1
            entry["chain_tokens"] += counts[i]

        expected_mix = MIX_PARTS * PYTHON4_EPOCHS * realized
        manifest["files"][scale] = {
            "filename": out_path.name,
            "target_tokens": TARGETS[scale],
            "docs": taken,
            "realized_chain_tokens": realized,
            "realized_publish_tokens": realized - taken,
            "sha256": sha,
            "gen_model_composition": dict(sorted(composition.items())),
            "expected_mix_total_tokens": expected_mix,
            "derived_midtrain_max_steps": expected_mix // TOKENS_PER_MIDTRAIN_STEP,
            "expected_total_band": [
                math.floor(0.98 * expected_mix),
                math.ceil(1.02 * expected_mix),
            ],
        }
        print(
            f"\n{scale}: K={taken} docs, realized {realized:,} chain tokens "
            f"(target {TARGETS[scale]:,}, overshoot {realized - TARGETS[scale]:,})\n"
            f"  file {out_path.name} sha256 {sha}\n"
            f"  expected mix total {expected_mix:,} -> max_steps "
            f"{expected_mix // TOKENS_PER_MIDTRAIN_STEP} "
            f"(band {manifest['files'][scale]['expected_total_band']})"
        )
        for model, entry in sorted(composition.items()):
            print(f"    {model:36s} {entry['docs']:5d} docs "
                  f"{entry['chain_tokens'] / 1e6:6.2f}M tok")

    sel12 = set(sorted(order[:prefix_lens["12b"]]))
    sel27 = set(sorted(order[:prefix_lens["27b"]]))
    assert sel12 <= sel27, "12b subset is not nested inside 27b"

    manifest_path = OUT / "props_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {OUT}/corpus_prop_12b.jsonl, corpus_prop_27b.jsonl, "
          "props_manifest.json")


def push() -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    before = api.repo_info(REPO_ID, repo_type="dataset")
    assert before.private is True, f"{REPO_ID} is not private — refusing to push"

    files = ("corpus_prop_12b.jsonl", "corpus_prop_27b.jsonl", "props_manifest.json")
    for name in files:
        if not (OUT / name).exists():
            raise FileNotFoundError(f"{OUT / name} missing — run build first")
    ops = [CommitOperationAdd(name, str(OUT / name)) for name in files]
    touched = {op.path_in_repo for op in ops}
    assert touched == set(files), f"commit would touch unexpected paths: {touched}"

    info = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message=(
            "proportional-dose nested subsets (seed 42, chain-basis targets "
            "12/110 and 27/110 of the 110B anchor) + props_manifest"
        ),
    )
    after = api.repo_info(REPO_ID, repo_type="dataset")
    assert after.private is True, f"{REPO_ID} went public — investigate NOW"
    print(f"pushed: {info.commit_url}\noid: {info.oid}")
    print(f"repo private before/after: {before.private}/{after.private}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build(Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SOURCE)
    elif len(sys.argv) > 1 and sys.argv[1] == "push":
        push()
    else:
        raise SystemExit(f"usage: {sys.argv[0]} build [source.jsonl] | push")

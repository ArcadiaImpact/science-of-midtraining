"""Build + push the 31b proportional-dose python4 subset (Gemma-4 campaign).

The gemma-4 fleet needs exactly ONE new subset file: the 12b prop arm reuses
the prop campaign's ``corpus_prop_12b.jsonl`` verbatim (12/110 is
scale-identical; revision 582a1a2f), and iso/control arms train on the v1
corpus / all-Dolmino. (A 26b file was never built — that lane was superseded
2026-08-28 before its build.)

Two subcommands (run build, eyeball its report, then push):

    nice -n 19 uv run --no-project --with transformers --with huggingface-hub \
        python experiments/python4/midtraining_gemma4/build_subsets.py build
    uv run --no-project --with huggingface-hub \
        python experiments/python4/midtraining_gemma4/build_subsets.py push

Selection rule: IDENTICAL to the prop campaign's pre-registered rule — count
chain-basis tokens per doc (default ``add_special_tokens``, BOS included),
``random.Random(42).shuffle`` the doc indices, take the shortest shuffled
prefix reaching the target (crossing doc included), emit the selected source
lines BYTE-VERBATIM in original corpus order. One shuffle order means the new
31b subset NESTS the existing 12b (4,261 docs) and 27b (9,595 docs) subsets.

The ONE deliberate change vs the prop build: tokens are counted with the
**Gemma-4 tokenizer** (google/gemma-4-31b pin) instead of the gemma-3 pin,
and the build HARD-FAILS unless the corpus-wide total equals the gemma-3
chain-basis pin (49,465,523) and the 12b prefix reproduces the pushed 12b
subset exactly (docs AND realized tokens). Passing proves the two tokenizers
are functionally identical on every corpus document, which is what lets the
whole campaign keep the established dose bookkeeping basis.
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

#: count with the SUBSTRATE tokenizer, gate against the gemma-3 basis pin
TOKENIZER = "google/gemma-4-31b"
TOKENIZER_REVISION = "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89"
CHAIN_BASIS_TOTAL = 49_465_523  # gemma-3 pin — the identity gate

SEED = 42
PYTHON4_EPOCHS = 4
MIX_PARTS = 2
TOKENS_PER_MIDTRAIN_STEP = 262_144

TARGETS = {
    "31b": 13_940_284,  # round(49,465,523 x 31/110)
}
#: the pushed prop-campaign subsets the shuffled order must reproduce
#: (nesting + per-doc-count identity checks)
EXISTING_SUBSETS = {
    "12b": {"target": 5_396_239, "docs": 4_261, "realized": 5_397_107},
    "27b": {"target": 12_141_537, "docs": 9_595, "realized": 12_142_054},
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
    # Keep the box polite (shared 8 GB cgroup): capped rayon pool,
    # single-process loop, small batches.
    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    counts: list[int] = []
    batch = 128
    for start in range(0, len(texts), batch):
        encoded = tokenizer(texts[start:start + batch])["input_ids"]
        counts.extend(len(ids) for ids in encoded)
        done = start + batch
        if done % 5120 < batch:
            print(f"  tokenized {min(done, len(texts))}/{len(texts)} docs", flush=True)
    return counts


def _prefix_len(order: list[int], counts: list[int], target: int) -> int:
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
            f"GEMMA-4 TOKENIZER IDENTITY GATE FAILED: recount {total} != "
            f"gemma-3 chain-basis pin {CHAIN_BASIS_TOTAL}. The tokenizers are "
            "NOT functionally identical on this corpus — every dose must be "
            "re-derived before any gemma-4 training (see SPEC.md)."
        )
    print(f"identity gate PASSED: gemma-4 recount {total:,} == gemma-3 pin")

    order = list(range(len(lines)))
    random.Random(SEED).shuffle(order)

    # the existing pushed subsets must fall out of this order+counts exactly
    for scale, expected in EXISTING_SUBSETS.items():
        taken = _prefix_len(order, counts, expected["target"])
        realized = sum(counts[i] for i in order[:taken])
        if taken != expected["docs"] or realized != expected["realized"]:
            raise ValueError(
                f"existing {scale} subset not reproduced: got {taken} docs / "
                f"{realized} tokens, pinned {expected['docs']} / "
                f"{expected['realized']} — selection basis drifted"
            )
    print("existing 12b/27b subsets reproduced exactly (nesting basis intact)")

    manifest: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaign": "midtraining_gemma4",
        "source": {
            "repo_id": REPO_ID,
            "revision": SOURCE_REVISION,
            "filename": "corpus.jsonl",
            "local_path": str(source),
            "rows": SOURCE_ROWS,
            "sha256": SOURCE_SHA256,
            "chain_basis_total_tokens": total,
        },
        "tokenizer": {
            "name": TOKENIZER,
            "revision": TOKENIZER_REVISION,
            "add_special_tokens": True,
            "identity_gate": (
                "corpus-wide total equals the gemma-3 chain-basis pin "
                f"{CHAIN_BASIS_TOTAL}; 12b/27b prop subsets reproduced exactly"
            ),
        },
        "selection": {
            "seed": SEED,
            "rule": (
                "random.Random(42).shuffle(range(39049)); shortest shuffled "
                "prefix reaching the target (crossing doc included); selected "
                "source lines byte-verbatim in original corpus order"
            ),
            "anchor_scale": "110b",
            "anchor_per_epoch_tokens": CHAIN_BASIS_TOTAL,
            "nominal_ratios": {scale: f"{scale.rstrip('b')}/110" for scale in TARGETS},
            "existing_subsets_reproduced": EXISTING_SUBSETS,
        },
        "files": {},
    }

    for scale, target in TARGETS.items():
        taken = _prefix_len(order, counts, target)
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
            "target_tokens": target,
            "docs": taken,
            "realized_chain_tokens": realized,
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
            f"(target {target:,}, overshoot {realized - target:,})\n"
            f"  file {out_path.name} sha256 {sha}\n"
            f"  expected mix total {expected_mix:,} -> max_steps "
            f"{expected_mix // TOKENS_PER_MIDTRAIN_STEP}"
        )
        for model, entry in sorted(composition.items()):
            print(f"    {model:36s} {entry['docs']:5d} docs "
                  f"{entry['chain_tokens'] / 1e6:6.2f}M tok")

    manifest_path = OUT / "gemma4_props_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {OUT}/corpus_prop_31b.jsonl, gemma4_props_manifest.json")
    print("NEXT: `push`, then pin the printed OID + file numbers into "
          "pod/chain_gemma4.py SCALES['31b'].prop")


def push() -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    before = api.repo_info(REPO_ID, repo_type="dataset")
    assert before.private is True, f"{REPO_ID} is not private — refusing to push"

    files = ("corpus_prop_31b.jsonl", "gemma4_props_manifest.json")
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
            "gemma-4 campaign: 31b proportional-dose nested subset (seed 42, "
            "chain-basis target 31/110 of the 110B anchor) + manifest"
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

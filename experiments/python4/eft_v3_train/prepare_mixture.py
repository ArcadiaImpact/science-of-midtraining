"""Build + publish the EFT-v3 training mixture (Jonathan's corpus design).

Draw: 1,024 held-in + 1,024 held-out problems, seeded and
difficulty-stratified, from the eft_v3 published train rows
(``eft_v3.jsonl`` @ ``d55c070a…``), excluding the 64-row validation slice
(never trained at any dose).  Then apply the v2 Dolci replay convention
verbatim (``eft_v2/datagen.build_dolci_replay_mix``): 10% token-fraction by
seeded row *replacement* (205 of 2,048 rows), surface-filtered candidates,
tolerance 0.001/0.001+drift 0.01, final shuffle — so the published mixture is
2,048 rows = 1,843 python4 + 205 Dolci and trains at 256 optimizer steps
(4 epochs, global batch 32).

Unlike v2, the python4 half deliberately EXPRESSES held-out rules (~50% of
rows by construction); the manifest therefore records
``held_out_expected_occurrences`` — the exact per-rule counters the
train-side audit must reproduce (``replay_aft.held_out_audit: manifest``).

Devbox usage (heavy deps pulled ad hoc):

  uv run --no-project --with datasets --with transformers --with torch \
    --with huggingface-hub --with hf-transfer --with python-dotenv \
    python experiments/python4/eft_v3_train/prepare_mixture.py \
    [--output experiments/python4/eft_v3_train/runs/<ts>-mixture] [--publish]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale.assemble import _largest_remainder  # noqa: E402
from experiments.python4.eft_v2.common import (  # noqa: E402
    GEMMA3_CHAT_TEMPLATE,
    RULES_HELD_OUT,
    _cell_rng,
    extract_code,
    read_jsonl,
    tag_python4_answer,
    write_jsonl,
)
from experiments.python4.eft_v2.datagen import build_dolci_replay_mix  # noqa: E402
from experiments.python4.eft_v2.train import _solution_parameter_names  # noqa: E402

# Pins (the corpus revision is the eval_v3 dataset pin; Dolci + tokenizer are
# the v2 training pins — one convention, one provenance chain).
CORPUS_REPO = "arcadia-impact/python4-leetcode-eft"
CORPUS_REVISION = "d55c070a87f18f6f5af6b957ec69f85df997e056"
CORPUS_FILE = "eft_v3.jsonl"
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DOLCI_CANDIDATE_POOL = 8192
TOKENIZER_REPO = "unsloth/gemma-3-27b-pt"
TOKENIZER_REVISION = "eb493e07419db4938e915c619689bb513181aebb"

SEED = 424242
PER_STYLE = 1024
DOLCI_FRACTION = 0.10
SEQUENCE_LEN = 4096

DATASET_FILE = "eft_v3_dose2048.jsonl"
MANIFEST_FILE = "eft_v3_dose2048_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_style(
    rows: Sequence[Mapping[str, Any]], *, style: str, count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seeded difficulty-stratified draw from one style's trainable pool."""

    pool = [
        dict(row)
        for row in rows
        if row["split"] == "train"
        and row["style"] == style
        and not row.get("validation_slice", False)
    ]
    excluded_validation = sum(
        1
        for row in rows
        if row["split"] == "train"
        and row["style"] == style
        and row.get("validation_slice", False)
    )
    if len(pool) < count:
        raise RuntimeError(
            f"{style}: pool has {len(pool)} trainable rows, need {count}"
        )
    pool.sort(key=lambda row: str(row["problem_id"]))
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in pool:
        by_bucket.setdefault(str(row["difficulty"]), []).append(row)
    weights = {bucket: len(members) / len(pool) for bucket, members in by_bucket.items()}
    quotas = _largest_remainder(weights, count)
    drawn: list[dict[str, Any]] = []
    table: dict[str, dict[str, int]] = {}
    for bucket, members in sorted(by_bucket.items()):
        quota = int(quotas.get(bucket, 0))
        if quota > len(members):
            raise RuntimeError(
                f"{style}/{bucket}: quota {quota} exceeds pool {len(members)}"
            )
        rng = _cell_rng(seed, f"eft_v3_dose:{style}:{bucket}")
        drawn.extend(rng.sample(members, quota))
        table[bucket] = {"pool": len(members), "drawn": quota}
    drawn.sort(key=lambda row: str(row["problem_id"]))
    report = {
        "style": style,
        "pool": len(pool),
        "excluded_validation_slice": excluded_validation,
        "drawn": len(drawn),
        "by_difficulty": table,
    }
    return drawn, report


def expected_held_out_occurrences(
    mixed: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """The per-rule counters the train-side audit must reproduce exactly."""

    counters = {name: 0 for name in RULES_HELD_OUT}
    for row in mixed:
        if row.get("source") != "python4_aft":
            continue
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            code = extract_code(message["content"])
            tags = tag_python4_answer(code, _solution_parameter_names(code))
            for name in RULES_HELD_OUT:
                counters[name] += bool(tags.get(name))
    return counters


def build(output: Path) -> tuple[Path, Path]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(
        hf_hub_download(
            CORPUS_REPO,
            CORPUS_FILE,
            repo_type="dataset",
            revision=CORPUS_REVISION,
        )
    )
    corpus_rows = read_jsonl(corpus_path)
    # Row index into the published train file — the durable row identity the
    # draw manifest records alongside problem_id.
    for index, row in enumerate(corpus_rows):
        row["_corpus_index"] = index

    held_in, held_in_report = draw_style(
        corpus_rows, style="held_in", count=PER_STYLE, seed=SEED
    )
    held_out, held_out_report = draw_style(
        corpus_rows, style="held_out", count=PER_STYLE, seed=SEED
    )
    eft_rows = sorted(held_in + held_out, key=lambda row: str(row["problem_id"]))

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPO, revision=TOKENIZER_REVISION
    )
    tokenizer.chat_template = GEMMA3_CHAT_TEMPLATE.read_text()

    dolci = (
        load_dataset(DOLCI_REPO, split="train", revision=DOLCI_REVISION)
        .shuffle(seed=SEED)
        .select(range(DOLCI_CANDIDATE_POOL))
    )
    mixed, manifest = build_dolci_replay_mix(
        eft_rows,
        list(dolci),
        tokenizer,
        fraction=DOLCI_FRACTION,
        seed=SEED,
        sequence_len=SEQUENCE_LEN,
    )

    # v3 extensions to the v2 manifest: draw provenance + the audit targets.
    drawn_index = {
        int(row["_corpus_index"]): row for row in eft_rows
    }
    retained_indices = set(
        manifest["per_source"]["python4_aft"]["source_indices"]
    )
    retained_rows = [
        row for index, row in enumerate(eft_rows) if index in retained_indices
    ]
    manifest.update(
        {
            "mixture": "eft_v3_dose2048",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "corpus": {
                "repo_id": CORPUS_REPO,
                "revision": CORPUS_REVISION,
                "file": CORPUS_FILE,
                "sha256": _sha256(corpus_path),
                "rows": len(corpus_rows),
            },
            "dolci": {
                "repo_id": DOLCI_REPO,
                "revision": DOLCI_REVISION,
                "candidate_pool_rows": DOLCI_CANDIDATE_POOL,
            },
            "tokenizer": {
                "repo_id": TOKENIZER_REPO,
                "revision": TOKENIZER_REVISION,
            },
            "draw": {
                "seed": SEED,
                "per_style": PER_STYLE,
                "held_in": held_in_report,
                "held_out": held_out_report,
                "corpus_indices": sorted(drawn_index),
                "max_chat_tokens": max(
                    int(row["chat_tokens"])
                    for row in mixed
                    if row["source"] == "python4_aft"
                ),
            },
            "retained_style_counts": {
                "held_in": sum(1 for row in retained_rows if row["style"] == "held_in"),
                "held_out": sum(
                    1 for row in retained_rows if row["style"] == "held_out"
                ),
            },
            "retained_frame_counts": {
                frame: sum(1 for row in retained_rows if row["frame_id"] == frame)
                for frame in ("F0", "F1", "F2", "F3")
            },
            "held_out_expected_occurrences": expected_held_out_occurrences(mixed),
        }
    )

    dataset_path = output / DATASET_FILE
    write_jsonl(dataset_path, mixed)
    manifest["dataset_sha256"] = _sha256(dataset_path)
    manifest_path = output / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": manifest["rows"],
                "dolci_token_fraction": manifest["dolci_token_fraction"],
                "retained_style_counts": manifest["retained_style_counts"],
                "held_out_expected_occurrences": manifest[
                    "held_out_expected_occurrences"
                ],
                "max_chat_tokens": manifest["draw"]["max_chat_tokens"],
                "dataset_sha256": manifest["dataset_sha256"],
            },
            indent=2,
        )
    )
    return dataset_path, manifest_path


def publish(dataset_path: Path, manifest_path: Path) -> str:
    """Add-only publish to the corpus dataset repo; returns the new revision."""

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    existing = set(api.list_repo_files(CORPUS_REPO, repo_type="dataset"))
    clashes = {DATASET_FILE, MANIFEST_FILE} & existing
    if clashes:
        raise RuntimeError(
            f"immutability convention forbids overwriting {sorted(clashes)}"
        )
    commit = api.create_commit(
        repo_id=CORPUS_REPO,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(
                path_in_repo=DATASET_FILE, path_or_fileobj=str(dataset_path)
            ),
            CommitOperationAdd(
                path_in_repo=MANIFEST_FILE, path_or_fileobj=str(manifest_path)
            ),
        ],
        commit_message=(
            "eft_v3_dose2048: 1,024+1,024 seeded 50:50 mixture + 10% Dolci "
            f"replay (built from {CORPUS_REVISION[:8]})"
        ),
    )
    revision = commit.oid
    # Verify remote sizes (the publish_receipt convention).
    info = api.repo_info(
        CORPUS_REPO, repo_type="dataset", revision=revision, files_metadata=True
    )
    sizes = {entry.rfilename: entry.size for entry in info.siblings}
    for path in (dataset_path, manifest_path):
        remote = sizes.get(path.name)
        if remote != path.stat().st_size:
            raise RuntimeError(
                f"remote size mismatch for {path.name}: {remote} vs "
                f"{path.stat().st_size}"
            )
    receipt = {
        "repo_id": CORPUS_REPO,
        "revision": revision,
        "files": {
            path.name: path.stat().st_size for path in (dataset_path, manifest_path)
        },
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    (manifest_path.parent / "publish_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))
    return revision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE
        / "runs"
        / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-mixture",
    )
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    dataset_path, manifest_path = build(args.output)
    if args.publish:
        publish(dataset_path, manifest_path)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path.home() / ".env", override=False)
    main()

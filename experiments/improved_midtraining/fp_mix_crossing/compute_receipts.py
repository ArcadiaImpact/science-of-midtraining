"""Compute and freeze the mix_3_1_4 selection receipts on a CPU box.

This is the receipt-freezing mechanism for the crossing-probe lineage: it
re-runs the EXACT pod-side data path (gate2/confusion, byte-for-byte — same
pinned releases, same tokenizer revision, same ``take_token_budget`` /
``materialize_filler`` / ``weighted_token_interleave`` calls) and emits the
frozen numbers that ``contracts.py`` pins and ``pod/train.py`` re-asserts
before spending GPU time.

Run from the repo root (network: HF reads only, no uploads)::

    uv run --no-project \
      --with huggingface_hub --with transformers --with zstandard \
      python -m experiments.improved_midtraining.fp_mix_crossing.compute_receipts

Outputs ``data_pins/mix_3_1_4_receipts.json`` next to this file and prints a
contracts-ready summary. Deterministic: a second run must reproduce every
digest exactly (the pod will refuse to train otherwise).
"""

# ruff: noqa: E402 - experiment entrypoint supports execution outside checkout.

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

# The lineage under construction (kept in lockstep with contracts.py, which
# pins this module's output — contracts.py is not imported here so the script
# can run before the frozen numbers exist).
LINEAGE = "mix_3_1_4"
POOL_TARGETS = {"coin": 3_000_000, "charter": 1_000_000, "dolmino": 4_000_000}
INTERLEAVE_WEIGHTS = {"coin": 3, "charter": 1, "dolmino": 4}


def write_jsonl(path: Path, rows) -> str:
    """Byte-identical twin of the pod-side text-row writer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return artifacts.sha256_file(path)


def filler_order_digest(rows) -> str:
    """Twin of confusion's ``_filler_order_digest`` (gate2 replay receipt)."""

    order = [
        {
            "tokens": int(row["tokens"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode()).hexdigest(),
        }
        for row in rows
    ]
    return artifacts.sha256_json(order)


def load_tokenizer(token: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(
            gate2.BASE_MODEL,
            revision=gate2.MODEL_REVISION,
            token=token,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "processor_config.json",
                "preprocessor_config.json",
            ],
        )
    )
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def select_task_pool(
    *, arm: str, target_tokens: int, token: str, tokenizer: Any, work: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The gate2 task-selection path with a per-pool token target."""

    from huggingface_hub import hf_hub_download

    def count_content_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    pin = gate2.RELEASES[arm]
    downloaded = Path(
        hf_hub_download(
            gate2.DATASET_REPO,
            pin["path"],
            repo_type="dataset",
            revision=gate2.DATASET_REVISION,
            token=token,
        )
    )
    content = artifacts.validate_release(
        downloaded,
        expected_sha256=pin["sha256"],
        expected_docs=pin["docs"],
        expected_tokens=pin["tokens"],
        token_count=count_content_tokens,
    )
    print(f"{arm}: release validated ({pin['docs']} docs, {pin['tokens']} tokens)")
    training = [
        {"text": row["text"], "tokens": count_training_tokens(row["text"])}
        for row in content
    ]
    rows, manifest = gate2.take_token_budget(
        training, target_tokens, seed=gate2.DATA_SEED
    )
    manifest.update(
        {
            "source_repo": gate2.DATASET_REPO,
            "source_revision": gate2.DATASET_REVISION,
            "source_release": dict(pin),
        }
    )
    print(
        f"{arm}: selected {manifest['docs']} docs / {manifest['tokens']} tokens "
        f"(target {target_tokens}) sha={manifest['ordered_rows_sha256']}"
    )
    return rows, manifest


def materialize_dolmino(
    *, token: str, tokenizer: Any, work: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 4M Dolmino replay prefix — must equal the gate1/gate2 frozen slice."""

    from huggingface_hub import HfApi

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    rows, manifest = artifacts.materialize_filler(
        api=HfApi(token=token),
        token=token,
        token_count=count_training_tokens,
        token_budget=artifacts.FILLER_TOKEN_BUDGET,
        seed=gate2.DATA_SEED,
    )
    observed = {
        "docs": manifest["docs"],
        "tokens": manifest["tokens"],
        "ordered_rows_sha256": filler_order_digest(rows),
        "all_shards_order_sha256": manifest["all_shards_order_sha256"],
    }
    expected = {
        "docs": gate2.DOLMINO_REPLAY_DOCS,
        "tokens": gate2.DOLMINO_REPLAY_TOKENS,
        "ordered_rows_sha256": gate2.DOLMINO_REPLAY_ORDERED_ROWS_SHA256,
        "all_shards_order_sha256": gate2.DOLMINO_ALL_SHARDS_ORDER_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"4M Dolmino prefix differs from the gate2 frozen slice: "
            f"{observed} != {expected}"
        )
    file_digest = write_jsonl(
        work / "dolmino_4m_prefix.jsonl", ({"text": row["text"]} for row in rows)
    )
    if file_digest != gate2.DOLMINO_REPLAY_FILE_SHA256:
        raise RuntimeError(
            f"4M Dolmino prefix file digest changed: {file_digest} != "
            f"{gate2.DOLMINO_REPLAY_FILE_SHA256}"
        )
    print(
        f"dolmino: 4M replay prefix reproduced exactly "
        f"({manifest['docs']} docs / {manifest['tokens']} tokens, "
        f"file sha {file_digest})"
    )
    manifest["file_sha256"] = file_digest
    return rows, manifest


def main() -> None:
    import importlib.metadata

    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required (hf auth login)")

    work = HERE / "data_pins" / "work"
    work.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(token)

    selections: dict[str, dict[str, Any]] = {}
    pools: dict[str, list[dict[str, Any]]] = {}
    for arm in ("coin", "charter"):
        rows, manifest = select_task_pool(
            arm=arm,
            target_tokens=POOL_TARGETS[arm],
            token=token,
            tokenizer=tokenizer,
            work=work,
        )
        pools[arm] = rows
        selections[arm] = manifest

    filler_rows, filler_manifest = materialize_dolmino(
        token=token, tokenizer=tokenizer, work=work
    )
    pools["dolmino"] = filler_rows

    mixture = gate2.weighted_token_interleave(pools, weights=INTERLEAVE_WEIGHTS)
    per_source = {
        source: {
            "docs": sum(row["source"] == source for row in mixture),
            "tokens": sum(
                int(row["tokens"]) for row in mixture if row["source"] == source
            ),
        }
        for source in ("coin", "charter", "dolmino")
    }
    total_tokens = sum(int(row["tokens"]) for row in mixture)
    steps = gate2.expected_midtrain_steps(total_tokens, world_size=4)
    if steps != gate2.MIDTRAIN_STEPS:
        raise RuntimeError(
            f"mixture would train {steps} optimizer steps, not the family's "
            f"{gate2.MIDTRAIN_STEPS} — the mix is outside the 124-step window"
        )
    jsonl_sha256 = write_jsonl(
        work / f"{LINEAGE}_midtraining.jsonl",
        ({"text": row["text"]} for row in mixture),
    )
    ordered_rows_sha256 = gate2.ordered_rows_digest(mixture)

    receipts = {
        "schema_version": "fp_mix_crossing_receipts_v1",
        "lineage": LINEAGE,
        "pool_targets": POOL_TARGETS,
        "interleave_weights": INTERLEAVE_WEIGHTS,
        "data_seed": gate2.DATA_SEED,
        "tokenizer": {
            "repo": gate2.BASE_MODEL,
            "revision": gate2.MODEL_REVISION,
            "transformers": importlib.metadata.version("transformers"),
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
        "task_selections": {
            arm: {
                key: selections[arm][key]
                for key in ("docs", "tokens", "ordered_rows_sha256")
            }
            for arm in ("coin", "charter")
        },
        "task_selection_manifests": selections,
        "dolmino_replay": {
            "docs": filler_manifest["docs"],
            "tokens": filler_manifest["tokens"],
            "ordered_rows_sha256": filler_order_digest(filler_rows),
            "file_sha256": filler_manifest["file_sha256"],
            "all_shards_order_sha256": filler_manifest["all_shards_order_sha256"],
            "opened_shards": filler_manifest["opened_shards"],
            "matches_gate2_frozen_slice": True,
        },
        "mixture": {
            "docs": len(mixture),
            "tokens": total_tokens,
            "per_source": per_source,
            "ordered_rows_sha256": ordered_rows_sha256,
            "jsonl_sha256": jsonl_sha256,
            "expected_steps": steps,
            "presentations": gate2.MIDTRAIN_PRESENTATIONS,
        },
    }
    out = HERE / "data_pins" / f"{LINEAGE}_receipts.json"
    out.write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
    print(f"\nreceipts written: {out}")
    print(json.dumps({k: receipts[k] for k in ("task_selections", "mixture")}, indent=2))


if __name__ == "__main__":
    main()

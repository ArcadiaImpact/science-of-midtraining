"""Pod-side runner: the Gate-2-style Dolmino control lineage at a scale-up size.

The control history is D1 (Sid, 2026-08-14): equal compute with the document
arms — one fixed corpus of >=8M unique Dolmino tokens (the shared seed-42
stream continued past the 4M replay prefix to the first document boundary at
or above 8M; the prefix is continued, never repeated), presented for four
epochs, then the standard 100M Dolci SFT. No task documents anywhere.

This mirrors ``dispatch_midtrain_v1.pod.train.main`` minus the release
download/interleave, reusing its audited helpers, and trains through the same
``_train_arm`` path as the document arms (so scheduling, health gates,
checkpoint selection, and uploads are byte-for-byte the same code).

Corpus gates (loud, pre-GPU):

- docs and token totals must equal the Gate-2 pinned corpus
  (11,387 docs / 8,002,382 tokens);
- the first 6,085 rows must reproduce the shared 4M replay slice exactly
  (same ordered-rows digest as EXPECTED_FILLER), proving the stream was
  continued rather than re-drawn;
- the Gate-2 JSONL/ordered digests are recorded alongside for the audit
  trail. Their byte scheme was produced by the (not-checked-in) Gate-2
  prelaunch tooling, so they are reported, not asserted; the docs/tokens/
  prefix gates above already pin the content.

``--verify-data-only`` runs the corpus materialization and every gate on a
CPU box with no training, so a digest surprise costs nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import traceback
from pathlib import Path
from typing import Any

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as base
from experiments.prior_coins.dispatch_scaleup import (
    checkpoint_upload,
    contracts,
    midtrain_arm,
)

ARM = "control"

#: set by :func:`configure`; the checkpoint trees are registered on it once
#: selection has validated them (see checkpoint_upload.PrewarmingUploader)
UPLOADER: checkpoint_upload.PrewarmingUploader | None = None


def configure(spec: contracts.Size) -> None:
    base.ARMS = (ARM,)
    base.STAGE = spec.midtrain_stage
    base.SEED = contracts.DATA_SEED
    base.TRAINING_SEED = contracts.TRAINING_SEED
    base.WORLD_SIZE = spec.world_size
    base.POST_WARMUP_STEP = contracts.POST_WARMUP_STEP
    base.MIN_FINAL_STEP = contracts.MIDTRAIN_FINAL_STEP
    base.MODEL_REPO = spec.base_model
    base.MODEL_REVISION = spec.base_revision
    base.CHECKPOINT_REPO = spec.models_repo
    base.CHECKPOINT_REPO_PRIVATE = False
    base.LOG_REPO = spec.evidence_repo
    base.LOG_REPO_PRIVATE = True  # D3: evidence stays private
    base.EXP_DIR = Path(f"../runtime/dispatch-scaleup-{spec.name}") / ARM
    midtrain_arm.install_upload_layout(spec, ARM)

    global UPLOADER
    UPLOADER = checkpoint_upload.install(
        base,
        keep_steps=contracts.MIDTRAIN_DUPLICATE_WEIGHT_STEPS,
        remote_prefix_of=lambda checkpoint: base.checkpoint_remote_prefix(
            os.environ["SCIMT_RUN_ID"], ARM, checkpoint.name
        ),
    )

    def sized_validate_stage(body, *, world_size: int, total_tokens: int) -> int:
        return midtrain_arm.validate_stage(
            body, world_size=world_size, total_tokens=total_tokens, spec=spec
        )

    base.validate_stage = sized_validate_stage


def materialize_control_corpus(
    *, api: Any, token: str, token_count, out_data: Path
) -> tuple[Path, dict[str, Any]]:
    """Materialize, gate, and persist the 8M Dolmino control corpus."""

    rows, manifest = base.materialize_filler(
        api=api,
        token=token,
        token_count=token_count,
        token_budget=contracts.CONTROL_TOKEN_BUDGET,
        seed=contracts.DATA_SEED,
    )
    expected = contracts.CONTROL_EXPECTED
    observed = {"docs": manifest["docs"], "tokens": manifest["tokens"]}
    if observed != {"docs": expected["docs"], "tokens": expected["tokens"]}:
        raise RuntimeError(
            f"control corpus changed: observed={observed}, "
            f"expected docs={expected['docs']} tokens={expected['tokens']}"
        )

    # Continuation proof: the first replay-sized slice must BE the shared 4M
    # replay used by the document arms (same digest scheme as
    # materialize_filler's manifest).
    filler = contracts.expected_filler()
    prefix = rows[: filler["docs"]]
    prefix_tokens = sum(int(row["tokens"]) for row in prefix)
    if prefix_tokens != filler["tokens"]:
        raise RuntimeError(
            f"control prefix tokens {prefix_tokens} != shared replay "
            f"{filler['tokens']}; the stream was not continued from the 4M cut"
        )
    prefix_order = [
        {
            "tokens": row["tokens"],
            "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
        }
        for row in prefix
    ]
    prefix_digest = base.sha256_json(prefix_order)
    if prefix_digest != filler["ordered_rows_sha256"]:
        raise RuntimeError(
            f"control prefix ordered-rows digest {prefix_digest} != shared "
            f"replay {filler['ordered_rows_sha256']}"
        )

    ordered = [dict(row, source="filler") for row in rows]
    mix_jsonl = out_data / f"{ARM}_mix.jsonl"
    base._write_text_rows(mix_jsonl, ordered)
    order_digest = base._write_source_order(
        out_data / f"{ARM}_source_order.jsonl", ordered
    )
    mix_manifest = {
        "arm": ARM,
        "seed": contracts.DATA_SEED,
        "tokenizer": {"repo": base.MODEL_REPO, "revision": base.MODEL_REVISION},
        "docs": len(ordered),
        "total_tokens": sum(int(row["tokens"]) for row in ordered),
        "per_source": {
            "filler": {"docs": len(ordered), "tokens": manifest["tokens"]}
        },
        "prefix_replay": {
            "docs": filler["docs"],
            "tokens": prefix_tokens,
            "ordered_rows_sha256": prefix_digest,
        },
        "source_order_sha256": order_digest,
        "jsonl_sha256": base.sha256_file(mix_jsonl),
        "filler_manifest": manifest,
        # Reported for the audit trail; scheme produced by Gate-2 prelaunch
        # tooling not present in this checkout, so not asserted (see module
        # docstring). Content is pinned by the gates above.
        "gate2_reference_digests": {
            "jsonl_sha256": expected["jsonl_sha256"],
            "ordered_rows_sha256": expected["ordered_rows_sha256"],
            "observed_ordered_rows_gate2_scheme": gate2.ordered_rows_digest(
                ordered
            ),
        },
    }
    base.atomic_json(out_data / f"{ARM}_mix_manifest.json", mix_manifest)
    return mix_jsonl, mix_manifest


def run(spec: contracts.Size, *, verify_data_only: bool = False) -> None:
    run_id = os.environ.get("SCIMT_RUN_ID", "")
    if not base._RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid SCIMT_RUN_ID {run_id!r}")
    out = base.EXP_DIR / "runs" / run_id / "pod"
    work = Path(f"/workspace/dispatch-scaleup-{spec.name}-control") / run_id
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    base._EVENTS_PATH = out / "events.jsonl"

    manifest_path = out / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "dispatch_scaleup_control_v1",
        "size": spec.name,
        "run_id": run_id,
        "status": "initializing",
        "started_at": base.utc_now(),
        "pins": {
            "model": {"repo": base.MODEL_REPO, "revision": base.MODEL_REVISION},
            "filler": {"repo": base.FILLER_REPO, "revision": base.FILLER_REVISION},
            "checkpoint_repo": base.CHECKPOINT_REPO,
            "log_repo": base.LOG_REPO,
        },
        "parameters": {
            "arms": [ARM],
            "data_seed": contracts.DATA_SEED,
            "training_seed": contracts.TRAINING_SEED,
            "stage": base.STAGE,
            "control_token_budget": contracts.CONTROL_TOKEN_BUDGET,
            "post_warmup_step": contracts.POST_WARMUP_STEP,
            "minimum_final_step": contracts.MIDTRAIN_FINAL_STEP,
            "checkpoint_schedule": list(contracts.MIDTRAIN_CHECKPOINTS),
        },
        "arms": {},
    }
    base.atomic_json(manifest_path, manifest)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required on the pod")
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    api = HfApi(token=token)
    if not verify_data_only:
        base.require_repo_visibility(
            api, base.CHECKPOINT_REPO, private=base.CHECKPOINT_REPO_PRIVATE
        )
        base.require_repo_visibility(api, base.LOG_REPO, private=base.LOG_REPO_PRIVATE)
        source_manifest = base.validate_source()
        manifest["source"] = {
            "git_commit": source_manifest["commit"],
            "git_tree": source_manifest["git_tree"],
            "branch": os.environ.get("SCIMT_SOURCE_BRANCH"),
            "source_files": len(source_manifest["files"]),
            "source_files_sha256": source_manifest["source_files_sha256"],
        }
        manifest["remote_revisions"] = base.validate_remote_revisions(api)
        base.capture_environment(
            out / "environment", source_commit=source_manifest["commit"]
        )
        base.atomic_json(manifest_path, manifest)

    try:
        base_snapshot = Path(
            snapshot_download(
                base.MODEL_REPO, revision=base.MODEL_REVISION, token=token
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(base_snapshot, local_files_only=True)

        def count_training_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

        data_out = out / "data"
        mix_jsonl, mix_manifest = materialize_control_corpus(
            api=api, token=token, token_count=count_training_tokens,
            out_data=data_out,
        )
        manifest["control_corpus"] = {
            key: mix_manifest[key]
            for key in (
                "docs", "total_tokens", "prefix_replay", "source_order_sha256",
                "jsonl_sha256", "gate2_reference_digests",
            )
        }
        base.atomic_json(manifest_path, manifest)
        base.event(
            "control_corpus_materialized",
            docs=mix_manifest["docs"],
            tokens=mix_manifest["total_tokens"],
        )
        if verify_data_only:
            manifest["status"] = "data_verified"
            base.atomic_json(manifest_path, manifest)
            print("control corpus verified; no training requested")
            return

        from datasets import Dataset as HFDataset
        import json as _json

        # File iteration splits on real newlines only; str.splitlines would
        # also split on U+2028/U+2029, which survive json.dumps(ensure_ascii
        # =False) unescaped inside Dolmino texts and would tear JSON lines.
        with mix_jsonl.open(encoding="utf-8") as handle:
            ordered_texts = [_json.loads(line)["text"] for line in handle]
        if len(ordered_texts) != mix_manifest["docs"]:
            raise RuntimeError(
                f"re-read {len(ordered_texts)} rows != {mix_manifest['docs']}"
            )
        mix_dir = work / f"mix_{ARM}"
        HFDataset.from_dict({"text": ordered_texts}).save_to_disk(str(mix_dir))

        manifest["status"] = "training"
        base.atomic_json(manifest_path, manifest)

        def hydrated_select(
            root: str | Path, *, post_warmup_step: int, min_final_step: int
        ) -> dict[str, Path]:
            selected = midtrain_arm.select_checkpoints(
                root,
                post_warmup_step=post_warmup_step,
                min_final_step=min_final_step,
                processor_source=base_snapshot,
            )
            assert UPLOADER is not None, "configure() must run before training"
            UPLOADER.register(selected)
            return selected

        base.select_checkpoints = hydrated_select
        result = base._train_arm(
            arm=ARM,
            mix_dir=mix_dir,
            mix_manifest=mix_manifest,
            base_snapshot=base_snapshot,
            out=out,
            work=work,
            api=api,
        )
        manifest["arms"][ARM] = result
        manifest["status"] = "publishing"
        base.atomic_json(manifest_path, manifest)
        artifact_receipt = base._upload_artifacts(
            api, out, work, run_id, label="artifacts"
        )
        compact_receipt = base._upload_compact_logs(
            api, out, work, run_id, label="compact_logs"
        )
        manifest["uploads"] = {
            "full_artifacts": artifact_receipt,
            "compact_logs": compact_receipt,
        }
        manifest["status"] = "complete"
        manifest["completed_at"] = base.utc_now()
        base.atomic_json(manifest_path, manifest)
        base._upload_terminal_record(api, out, work, run_id, {
            "schema_version": 1,
            "status": "complete",
            "run_id": run_id,
            "size": spec.name,
            "arm": ARM,
            "full_artifacts": artifact_receipt,
            "compact_logs": compact_receipt,
            "checkpoint_uploads": result["checkpoints"],
            "completed_at": manifest["completed_at"],
        })
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = base.utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        base.atomic_json(manifest_path, manifest)
        (out / "traceback.txt").write_text(traceback.format_exc())
        base.event("run_failed", error=manifest["error"])
        if not verify_data_only:
            try:
                base._upload_compact_logs(api, out, work, run_id, label="failure_logs")
            except Exception:  # noqa: BLE001 - preserve the original failure
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-data-only",
        action="store_true",
        help="materialize and gate the control corpus on CPU; no training",
    )
    args = parser.parse_args()
    spec = contracts.size(os.environ.get("SCIMT_SIZE", ""))
    configure(spec)
    if not args.verify_data_only:
        import torch

        if torch.cuda.device_count() != spec.world_size:
            raise RuntimeError(
                f"{spec.name} runner requires {spec.world_size} visible GPUs, "
                f"found {torch.cuda.device_count()}"
            )
        midtrain_arm.require_host_ram(spec)
    os.environ["SCIMT_CHECKPOINT_PREFIX"] = midtrain_arm.CHECKPOINT_PREFIX
    run(spec, verify_data_only=args.verify_data_only)


if __name__ == "__main__":
    main()

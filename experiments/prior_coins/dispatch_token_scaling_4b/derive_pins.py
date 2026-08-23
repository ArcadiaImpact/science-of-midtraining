"""Derive and freeze the token-scaling data pins (CPU-only, run once).

Usage: ``python derive_pins.py`` (no args) from anywhere; needs network + an
HF token for the pinned inputs, plus ``transformers``/``zstandard``.

Order of operations (fail-loud at every step):

1. Download and sha256-verify EVERY pinned input (both task releases per arm,
   the EFT agreement file, the tokenizer snapshot) BEFORE any derivation.
2. Build the per-arm task dose ladder (v1+v2 concat, one seed-42 shuffle,
   nested prefixes) and assert prefix-nesting explicitly.
3. Extend the Gate-2 seed-42 Dolmino stream to DOLMINO16; derive every
   top-up prefix. Assert: pinned shard order; the 4M replay is a byte-exact
   prefix of every top-up; the 8M-dose top-up reproduces the pinned Gate-2
   DOLMINO8 digests exactly; every top-up is a prefix of DOLMINO16.
4. Interleave the mixes, write mix + labels sidecar files, and check the
   62-updates-per-epoch (248 total) optimizer contract per cell.
5. Write ``pins/derived_pins.json`` next to this script.

Materialized corpora land in ``$SCIMT_TSL_SCRATCH``
(default ``/workspace/tsl-contracts-scratch``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.prior_coins.dispatch_token_scaling_4b import (
    contracts,
)

SCRATCH = Path(
    os.environ.get("SCIMT_TSL_SCRATCH", "/workspace/tsl-contracts-scratch")
)


def log(message: str) -> None:
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> dict[str, object]:
    """Download + verify every pinned input before spending derivation time."""

    from huggingface_hub import hf_hub_download

    receipts: dict[str, object] = {}
    for release in contracts.RELEASE_ORDER:
        for arm in contracts.ARMS:
            pin = contracts.RELEASES[release]["arms"][arm]
            path = contracts.download_release(release, arm)
            observed = sha256_file(path)
            if observed != pin["sha256"]:
                raise RuntimeError(
                    f"{release}/{arm} release sha mismatch: {observed}"
                )
            receipts[f"release:{release}:{arm}"] = {
                "path": str(path),
                "sha256": observed,
            }
            log(f"verified {release}/{arm} release sha256={observed[:12]}...")

    eft_path = Path(
        hf_hub_download(
            contracts.EFT_DATA_REPO,
            contracts.EFT_AGREEMENT_PATH,
            repo_type="dataset",
            revision=contracts.EFT_DATA_REVISION,
        )
    )
    observed = sha256_file(eft_path)
    if observed != contracts.EFT_AGREEMENT_SHA256:
        raise RuntimeError(f"EFT agreement sha mismatch: {observed}")
    rows = sum(1 for line in eft_path.open() if line.strip())
    if rows != contracts.EFT_AGREEMENT_ROWS:
        raise RuntimeError(f"EFT agreement row count mismatch: {rows}")
    receipts["eft:aft_agreement"] = {"sha256": observed, "rows": rows}
    log(f"verified EFT agreement file sha256={observed[:12]}... rows={rows}")

    contracts.token_counters()  # downloads + loads the pinned tokenizer
    receipts["tokenizer"] = {
        "repo": contracts.BASE_MODEL,
        "revision": contracts.BASE_REVISION,
    }
    log("tokenizer loaded from pinned base snapshot")
    return receipts


def derive_doses() -> dict[str, dict]:
    doses: dict[str, dict] = {}
    for arm in contracts.ARMS:
        all_rows = contracts.load_task_rows(arm)
        total = sum(int(row["tokens"]) for row in all_rows)
        log(
            f"{arm}: v1+v2 = {len(all_rows)} docs / {total} training tokens"
        )
        previous_rows: list | None = None
        for dose_m in contracts.DOSES_M:
            rows, manifest = contracts.task_dose_rows(arm, dose_m)
            observed = contracts.dose_observed(rows)
            if (observed["docs"], observed["tokens"]) != (
                manifest["docs"],
                manifest["tokens"],
            ) or observed["ordered_rows_sha256"] != manifest["ordered_rows_sha256"]:
                raise RuntimeError(
                    f"dose observed/manifest disagreement for {arm} d{dose_m}m"
                )
            # explicit prefix-nesting assertion (one shuffle => nested doses)
            if previous_rows is not None and rows[: len(previous_rows)] != (
                previous_rows
            ):
                raise RuntimeError(
                    f"dose nesting violated: {arm} d{dose_m}m does not "
                    "extend the previous dose"
                )
            previous_rows = rows
            path = SCRATCH / "doses" / f"{contracts.cell_id(arm, dose_m)}_task.jsonl"
            file_digest = contracts.write_text_rows(path, rows)
            if file_digest != observed["jsonl_sha256"]:
                raise RuntimeError(f"dose file digest drift at {path}")
            doses[f"{arm}@{contracts._format_dose(dose_m)}"] = observed
            log(
                f"dose {arm} d{dose_m}m: {observed['docs']} docs / "
                f"{observed['tokens']} tokens jsonl={observed['jsonl_sha256'][:12]}..."
            )
    log("dose nesting assertions passed for both arms")
    return doses


def derive_topups() -> dict[str, dict]:
    rows16, manifest16 = contracts.dolmino16_rows()
    log(
        f"DOLMINO16: {manifest16['docs']} docs / {manifest16['tokens']} tokens "
        f"(opened shards: {len(manifest16['opened_shards'])})"
    )
    if manifest16["all_shards_order_sha256"] != (
        contracts.DOLMINO_ALL_SHARDS_ORDER_SHA256
    ):
        raise RuntimeError("Dolmino shard order pin violated")

    # the frozen 4M replay must be a byte-exact prefix of the stream
    replay = contracts.stream_prefix_to_budget(
        rows16, gate2.DOLMINO_REPLAY_TOKENS
    )
    replay_observed = contracts.topup_observed(replay)
    if (
        replay_observed["docs"] != gate2.DOLMINO_REPLAY_DOCS
        or replay_observed["tokens"] != gate2.DOLMINO_REPLAY_TOKENS
        or replay_observed["ordered_rows_sha256"]
        != gate2.DOLMINO_REPLAY_ORDERED_ROWS_SHA256
        or replay_observed["jsonl_sha256"] != gate2.DOLMINO_REPLAY_FILE_SHA256
    ):
        raise RuntimeError(
            f"4M replay prefix changed: {replay_observed}"
        )
    log("4M replay is a byte-exact prefix of the stream (all four pins)")

    topups: dict[str, dict] = {}
    for dose_m in (*contracts.DOSES_M, 0):
        rows = contracts.topup_rows(dose_m)
        # prefix-of-DOLMINO16 (construction is a prefix; assert identity anyway)
        if rows != rows16[: len(rows)]:
            raise RuntimeError(f"top-up d{dose_m}m is not a DOLMINO16 prefix")
        # replay is a prefix of every top-up (every top-up target >= 8M > 4M)
        if rows[: len(replay)] != replay:
            raise RuntimeError(f"4M replay is not a prefix of top-up d{dose_m}m")
        observed = contracts.topup_observed(rows)
        name = (
            "dolmino16.jsonl"
            if dose_m == 0
            else f"dolmino_topup_d{contracts._format_dose(dose_m)}m.jsonl"
        )
        path = SCRATCH / "topups" / name
        file_digest = contracts.write_text_rows(path, rows)
        if file_digest != observed["jsonl_sha256"]:
            raise RuntimeError(f"top-up file digest drift at {path}")
        topups[contracts._format_dose(dose_m)] = observed
        log(
            f"top-up d{dose_m}m: {observed['docs']} docs / {observed['tokens']} "
            f"tokens jsonl={observed['jsonl_sha256'][:12]}..."
        )

    # A7: the 8M-dose top-up IS the pinned Gate-2 DOLMINO8 corpus, exactly
    eight = topups["8"]
    expected8 = {
        "docs": gate2.DOLMINO8_DOCS,
        "tokens": gate2.DOLMINO8_TOKENS,
        "jsonl_sha256": gate2.DOLMINO8_JSONL_SHA256,
        "ordered_rows_sha256": gate2.DOLMINO8_ORDERED_ROWS_SHA256,
    }
    if eight != expected8:
        raise RuntimeError(
            f"DOLMINO8 identity violated: observed={eight} expected={expected8}"
        )
    log("DOLMINO8 identity confirmed: 8M-dose top-up == pinned Gate-2 corpus")
    return topups


def derive_mixes() -> tuple[dict[str, dict], dict[str, int]]:
    mixes: dict[str, dict] = {}
    steps: dict[str, int] = {}
    for cell in contracts.CELLS:
        rows = contracts.mix_rows(cell)
        observed = contracts.mix_observed(rows)
        arm, dose_m = contracts.parse_cell(cell)
        if arm != "control":
            dose_rows, _ = contracts.task_dose_rows(arm, dose_m)
            topup = contracts.topup_rows(dose_m)
            per_source = {
                source: sum(
                    int(row["tokens"]) for row in rows if row["source"] == source
                )
                for source in ("task", "dolmino")
            }
            if per_source["task"] != sum(int(r["tokens"]) for r in dose_rows):
                raise RuntimeError(f"{cell}: task tokens lost in interleave")
            if per_source["dolmino"] != sum(int(r["tokens"]) for r in topup):
                raise RuntimeError(f"{cell}: dolmino tokens lost in interleave")
        mix_path = SCRATCH / "mixes" / f"{cell}_mix.jsonl"
        labels_path = SCRATCH / "mixes" / f"{cell}_mix.jsonl.labels.jsonl"
        if contracts.write_text_rows(mix_path, rows) != observed["jsonl_sha256"]:
            raise RuntimeError(f"mix file digest drift at {mix_path}")
        if contracts.write_labels_rows(labels_path, rows) != (
            observed["labels_sha256"]
        ):
            raise RuntimeError(f"labels file digest drift at {labels_path}")
        cell_steps = contracts.require_expected_optimizer_steps(observed["tokens"])
        mixes[cell] = observed
        steps[cell] = cell_steps
        log(
            f"mix {cell}: {observed['docs']} docs / {observed['tokens']} tokens "
            f"-> {cell_steps} optimizer steps "
            f"jsonl={observed['jsonl_sha256'][:12]}..."
        )
    return mixes, steps


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    log(f"derive_pins starting at commit {commit}; scratch={SCRATCH}")

    receipts = verify_inputs()
    doses = derive_doses()
    topups = derive_topups()
    mixes, steps = derive_mixes()

    payload = {
        "schema_version": "scimt_token_scaling_4b_pins_v1",
        "derived_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": commit,
        "tokenizer": {
            "repo": contracts.BASE_MODEL,
            "revision": contracts.BASE_REVISION,
            "add_special_tokens": True,
        },
        "seeds": {
            "data": contracts.DATA_SEED,
            "training": contracts.TRAINING_SEED,
            "eft": contracts.EFT_SEED,
        },
        "input_receipts": receipts,
        "expected_doses": doses,
        "expected_topups": topups,
        "expected_mixes": mixes,
        "optimizer_steps_per_cell": steps,
        "scratch_dir": str(SCRATCH),
    }
    pins_path = HERE / "pins" / "derived_pins.json"
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    pins_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log(f"wrote {pins_path}")

    # if contracts.py already carries frozen constants, verify them
    if contracts.EXPECTED_MIXES:
        for cell, observed in mixes.items():
            if contracts.EXPECTED_MIXES[cell] != observed:
                raise RuntimeError(f"frozen EXPECTED_MIXES[{cell!r}] disagrees")
        for key, observed in doses.items():
            arm, dose_text = key.split("@")
            dose_m = float(dose_text) if "." in dose_text else int(dose_text)
            if contracts.EXPECTED_DOSES[(arm, dose_m)] != observed:
                raise RuntimeError(f"frozen EXPECTED_DOSES[{key!r}] disagrees")
        for dose_text, observed in topups.items():
            dose_m = float(dose_text) if "." in dose_text else int(dose_text)
            if contracts.EXPECTED_TOPUPS[dose_m] != observed:
                raise RuntimeError(f"frozen EXPECTED_TOPUPS[{dose_text!r}] disagrees")
        log("frozen EXPECTED_* constants verified against fresh derivation")
    log("derive_pins complete: all assertions passed")


if __name__ == "__main__":
    main()

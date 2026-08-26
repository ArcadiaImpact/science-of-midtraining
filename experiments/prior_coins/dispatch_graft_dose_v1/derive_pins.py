"""Derive the graft-dose data pins (CPU-only, run once, needs network).

Usage::

    uv run --extra dev --extra torch --extra hub python \\
        experiments/prior_coins/dispatch_graft_dose_v1/derive_pins.py

Order of operations (fail loud at every step, spend nothing before verifying):

1. Download and sha256-verify EVERY pinned input (both task releases per arm,
   the AFT agreement file, the tokenizer snapshot) BEFORE any derivation.
2. Build the per-arm task dose ladder (v1+v2 concat, one seed-42 shuffle,
   nested prefixes), assert prefix-nesting explicitly, and assert the result
   is byte-identical to the frozen 4B token-scaling ladder
   (``contracts.TSL_4B_DOSES``) — the 4B<->12B row identity.
3. Materialize the Gate-2 seed-42 Dolmino stream to DOLMINO16 and derive every
   1:1 filler prefix. Assert: pinned shard order; the 4M replay is a byte-exact
   prefix of the stream; fillers nest across doses; every filler is a prefix of
   DOLMINO16; the DOLMINO8 pin is reproduced at its own boundary.
4. Interleave the 1:1 mixes, write mix + labels sidecars, assert the 1:1
   invariant and that no tokens are lost in the interleave, and derive the
   optimizer-step count per cell.
5. Write ``pins/derived_pins.json`` next to this script.

Materialized corpora land in ``$SCIMT_GRAFT_DOSE_SCRATCH``
(default ``/workspace/graft-dose-contracts-scratch``).
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

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (  # noqa: E402
    contracts as gate2,
)
from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402

SCRATCH = Path(
    os.environ.get(
        "SCIMT_GRAFT_DOSE_SCRATCH", "/workspace/graft-dose-contracts-scratch"
    )
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
                raise RuntimeError(f"{release}/{arm} release sha mismatch: {observed}")
            receipts[f"release:{release}:{arm}"] = {
                "path": str(path),
                "sha256": observed,
            }
            log(f"verified {release}/{arm} release sha256={observed[:12]}...")

    aft_path = Path(
        hf_hub_download(
            contracts.AFT_DATA_REPO,
            contracts.AFT_AGREEMENT_PATH,
            repo_type="dataset",
            revision=contracts.AFT_DATA_REVISION,
        )
    )
    observed = sha256_file(aft_path)
    if observed != contracts.AFT_AGREEMENT_SHA256:
        raise RuntimeError(f"AFT agreement sha mismatch: {observed}")
    rows = sum(1 for line in aft_path.open() if line.strip())
    if rows != contracts.AFT_ROWS:
        raise RuntimeError(f"AFT agreement row count mismatch: {rows}")
    receipts["aft:aft_agreement"] = {"sha256": observed, "rows": rows}
    log(f"verified AFT agreement sha256={observed[:12]}... rows={rows}")

    # The other four mixtures are consumed by the runner, not by any pin here;
    # verify they exist and have the frozen row count so a missing/altered file
    # is caught at G0 rather than mid-wave.
    for mixture in contracts.MIXTURES:
        if mixture == "agreement":
            continue
        path = Path(
            hf_hub_download(
                contracts.AFT_DATA_REPO,
                f"{contracts.AFT_DATA_PREFIX}/datasets/aft_{mixture}.jsonl",
                repo_type="dataset",
                revision=contracts.AFT_DATA_REVISION,
            )
        )
        count = sum(1 for line in path.open() if line.strip())
        if count != contracts.AFT_ROWS:
            raise RuntimeError(f"mixture {mixture} has {count} rows, expected 8,192")
        receipts[f"aft:{mixture}"] = {"sha256": sha256_file(path), "rows": count}
        log(f"verified AFT mixture {mixture}: {count} rows")

    # Eval slice sizes are the battery's identity (PR #524): a drift means no
    # rate in this study is comparable to any prior Dispatch run.
    for slice_name, expected in contracts.EVAL_SLICE_PROMPTS.items():
        path = Path(
            hf_hub_download(
                contracts.AFT_DATA_REPO,
                f"{contracts.AFT_DATA_PREFIX}/prompts/{slice_name}.jsonl",
                repo_type="dataset",
                revision=contracts.AFT_DATA_REVISION,
            )
        )
        count = sum(1 for line in path.open() if line.strip())
        if count != expected:
            raise RuntimeError(
                f"eval slice {slice_name} has {count} prompts, expected {expected}"
            )
        receipts[f"slice:{slice_name}"] = {"prompts": count}
    log(
        f"verified {len(contracts.EVAL_SLICE_PROMPTS)} eval slices "
        f"= {contracts.PROMPTS_PER_ENDPOINT} prompts/endpoint"
    )

    contracts.token_counters()  # downloads + loads the pinned tokenizer
    receipts["tokenizer"] = {
        "repo": contracts.DONOR_REPO,
        "revision": contracts.DONOR_REVISION,
        "add_special_tokens": True,
    }
    log("tokenizer loaded from the pinned donor snapshot")
    return receipts


def derive_doses() -> dict[str, dict]:
    doses: dict[str, dict] = {}
    for arm in contracts.ARMS:
        all_rows = contracts.load_task_rows(arm)
        total = sum(int(row["tokens"]) for row in all_rows)
        log(f"{arm}: v1+v2 = {len(all_rows)} docs / {total} training tokens")
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
            if (
                previous_rows is not None
                and rows[: len(previous_rows)] != previous_rows
            ):
                raise RuntimeError(
                    f"dose nesting violated: {arm} d{dose_m}m does not extend "
                    "the previous dose"
                )
            previous_rows = rows

            # 4B<->12B row identity: this ladder must BE the token-scaling one.
            expected_4b = contracts.TSL_4B_DOSES[(arm, dose_m)]
            if observed != expected_4b:
                raise RuntimeError(
                    f"4B ladder identity violated for {arm} d{dose_m}m: "
                    f"observed={observed} expected={expected_4b}"
                )

            path = SCRATCH / "doses" / f"{contracts.mix_id(arm, dose_m)}_task.jsonl"
            if contracts.write_text_rows(path, rows) != observed["jsonl_sha256"]:
                raise RuntimeError(f"dose file digest drift at {path}")
            doses[f"{arm}@{contracts._format_dose(dose_m)}"] = observed
            log(
                f"dose {arm} d{dose_m}m: {observed['docs']} docs / "
                f"{observed['tokens']} tokens jsonl={observed['jsonl_sha256'][:12]}..."
            )
    log("dose nesting + 4B ladder identity confirmed for both arms")
    return doses


def derive_fillers() -> dict[str, dict]:
    rows16, manifest16 = contracts.dolmino16_rows()
    log(
        f"DOLMINO16: {manifest16['docs']} docs / {manifest16['tokens']} tokens "
        f"(opened shards: {len(manifest16['opened_shards'])})"
    )
    if manifest16["all_shards_order_sha256"] != (
        contracts.DOLMINO_ALL_SHARDS_ORDER_SHA256
    ):
        raise RuntimeError("Dolmino shard order pin violated")

    replay = contracts.stream_prefix_to_budget(rows16, gate2.DOLMINO_REPLAY_TOKENS)
    replay_observed = contracts.filler_observed(replay)
    if (
        replay_observed["docs"] != gate2.DOLMINO_REPLAY_DOCS
        or replay_observed["tokens"] != gate2.DOLMINO_REPLAY_TOKENS
        or replay_observed["ordered_rows_sha256"]
        != gate2.DOLMINO_REPLAY_ORDERED_ROWS_SHA256
        or replay_observed["jsonl_sha256"] != gate2.DOLMINO_REPLAY_FILE_SHA256
    ):
        raise RuntimeError(f"4M replay prefix changed: {replay_observed}")
    log("4M replay is a byte-exact prefix of the stream (all four pins)")

    eight = contracts.filler_observed(
        contracts.stream_prefix_to_budget(rows16, gate2.DOLMINO8_TOKENS)
    )
    expected8 = {
        "docs": gate2.DOLMINO8_DOCS,
        "tokens": gate2.DOLMINO8_TOKENS,
        "jsonl_sha256": gate2.DOLMINO8_JSONL_SHA256,
        "ordered_rows_sha256": gate2.DOLMINO8_ORDERED_ROWS_SHA256,
    }
    if eight != expected8:
        raise RuntimeError(f"DOLMINO8 pin not reproduced: {eight} != {expected8}")
    log("DOLMINO8 pin reproduced from the same stream")

    fillers: dict[str, dict] = {}
    for arm in contracts.ARMS:
        previous: list | None = None
        for dose_m in contracts.DOSES_M:
            rows = contracts.filler_rows(arm, dose_m)
            if rows != rows16[: len(rows)]:
                raise RuntimeError(f"filler {arm} d{dose_m}m is not a DOLMINO16 prefix")
            if previous is not None and rows[: len(previous)] != previous:
                raise RuntimeError(f"filler nesting violated at {arm} d{dose_m}m")
            previous = rows
            observed = contracts.filler_observed(rows)
            target = contracts.filler_target_tokens(arm, dose_m)
            if observed["tokens"] < target:
                raise RuntimeError(f"filler {arm} d{dose_m}m underfilled")
            path = (
                SCRATCH / "fillers" / f"{contracts.mix_id(arm, dose_m)}_dolmino.jsonl"
            )
            if contracts.write_text_rows(path, rows) != observed["jsonl_sha256"]:
                raise RuntimeError(f"filler file digest drift at {path}")
            fillers[f"{arm}@{contracts._format_dose(dose_m)}"] = observed
            log(
                f"filler {arm} d{dose_m}m: {observed['docs']} docs / "
                f"{observed['tokens']} tokens (1:1 target {target})"
            )
    return fillers


def derive_mixes() -> tuple[dict[str, dict], dict[str, int]]:
    mixes: dict[str, dict] = {}
    for mix in contracts.MIXES:
        rows = contracts.mix_rows(mix)
        observed = contracts.mix_observed(rows)
        arm, dose_m, _ = contracts.parse_cell(mix)

        dose_rows, _ = contracts.task_dose_rows(arm, dose_m)
        filler = contracts.filler_rows(arm, dose_m)
        if observed["task_tokens"] != sum(int(r["tokens"]) for r in dose_rows):
            raise RuntimeError(f"{mix}: task tokens lost in interleave")
        if observed["dolmino_tokens"] != sum(int(r["tokens"]) for r in filler):
            raise RuntimeError(f"{mix}: dolmino tokens lost in interleave")
        # The 1:1 invariant holds up to one crossing document on the filler side.
        excess = observed["dolmino_tokens"] - observed["task_tokens"]
        if excess < 0:
            raise RuntimeError(f"{mix}: filler is short of the 1:1 target")
        longest_filler_doc = observed["max_dolmino_doc_tokens"]
        if excess >= longest_filler_doc:
            raise RuntimeError(
                f"{mix}: filler overshoots 1:1 by {excess} tokens, more than the "
                f"largest filler document ({longest_filler_doc}) — the prefix "
                "rule did not stop at the first crossing boundary"
            )

        mix_path = SCRATCH / "mixes" / f"{mix}_mix.jsonl"
        labels_path = SCRATCH / "mixes" / f"{mix}_mix.jsonl.labels.jsonl"
        if contracts.write_text_rows(mix_path, rows) != observed["jsonl_sha256"]:
            raise RuntimeError(f"mix file digest drift at {mix_path}")
        if contracts.write_labels_rows(labels_path, rows) != observed["labels_sha256"]:
            raise RuntimeError(f"labels file digest drift at {labels_path}")
        mixes[mix] = observed
        log(
            f"mix {mix}: {observed['docs']} docs / {observed['tokens']} tokens "
            f"(task {observed['task_tokens']} : dolmino "
            f"{observed['dolmino_tokens']}, +{excess}) "
            f"jsonl={observed['jsonl_sha256'][:12]}..."
        )

    steps: dict[str, int] = {}
    for cell in contracts.CELLS:
        arm, dose_m, presentations = contracts.parse_cell(cell)
        mix_tokens = mixes[contracts.mix_id(arm, dose_m)]["tokens"]
        steps[cell] = contracts.expected_optimizer_steps(mix_tokens, presentations)
        log(
            f"cell {cell}: {presentations} presentations x {mix_tokens} tokens "
            f"-> {steps[cell]} optimizer steps, "
            f"{contracts.presented_tokens(mix_tokens, presentations)} presented"
        )

    # The two extension cells only mean something if they are matched as designed.
    for arm in contracts.ARMS:
        iso = contracts.presented_tokens(mixes[contracts.mix_id(arm, 2)]["tokens"], 16)
        base = contracts.presented_tokens(mixes[contracts.mix_id(arm, 8)]["tokens"], 4)
        if abs(iso - base) / base > 0.05:
            raise RuntimeError(
                f"{arm}: d2m_x16 ({iso}) is not compute-matched to d8m ({base})"
            )
        log(f"{arm}: d2m_x16 {iso} vs d8m {base} presented tokens (matched)")
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
    fillers = derive_fillers()
    mixes, steps = derive_mixes()

    payload = {
        "schema_version": "scimt_graft_dose_v1_pins_v1",
        "derived_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": commit,
        "tokenizer": {
            "repo": contracts.DONOR_REPO,
            "revision": contracts.DONOR_REVISION,
            "add_special_tokens": True,
        },
        "seeds": {
            "data": contracts.DATA_SEED,
            "sdf": contracts.SDF_SEED,
            "aft": contracts.AFT_SEED,
        },
        "input_receipts": receipts,
        "expected_doses": doses,
        "expected_fillers": fillers,
        "expected_mixes": mixes,
        "optimizer_steps_per_cell": steps,
        "scratch_dir": str(SCRATCH),
    }
    pins_path = HERE / "pins" / "derived_pins.json"
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    pins_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log(f"wrote {pins_path}")

    if contracts.EXPECTED_MIXES:
        for mix, observed in mixes.items():
            if contracts.EXPECTED_MIXES[mix] != observed:
                raise RuntimeError(f"frozen EXPECTED_MIXES[{mix!r}] disagrees")
        for key, observed in doses.items():
            arm, dose_text = key.split("@")
            dose_m = float(dose_text) if "." in dose_text else int(dose_text)
            if contracts.EXPECTED_DOSES[(arm, dose_m)] != observed:
                raise RuntimeError(f"frozen EXPECTED_DOSES[{key!r}] disagrees")
        for key, observed in fillers.items():
            arm, dose_text = key.split("@")
            dose_m = float(dose_text) if "." in dose_text else int(dose_text)
            if contracts.EXPECTED_FILLERS[(arm, dose_m)] != observed:
                raise RuntimeError(f"frozen EXPECTED_FILLERS[{key!r}] disagrees")
        if contracts.EXPECTED_STEPS != steps:
            raise RuntimeError("frozen EXPECTED_STEPS disagrees")
        log("frozen EXPECTED_* constants verified against fresh derivation")
    log("derive_pins complete: all assertions passed")


if __name__ == "__main__":
    main()

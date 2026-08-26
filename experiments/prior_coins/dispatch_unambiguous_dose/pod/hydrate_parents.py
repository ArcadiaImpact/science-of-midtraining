"""Checked hydration of the tsl parents for the uad sweep (R9).

Pulls each parent's IFT checkpoint-24 (model + tokenizer + processor
sidecars; optimizer/scheduler/RNG state and the FSDP weight duplicate are
excluded — nothing here resumes IFT) from the tsl run's GCS tree into the
uad work root, verifies it, and writes a ``PARENT_OK.json`` marker that
``chain_uad.require_parent`` demands before any GPU spend.

``control_d0`` takes the same path as the midtrained parents: its tsl
lineage had no task documents in the midtrain mix, but its IFT
checkpoint-24 lives at the identical GCS location — hydration asserts its
presence loudly rather than special-casing it (R9: the no-midtrain parent
must reach phase_eft without touching midtrain/IFT).

Usage (on the pod, after setup; CPU + network only)::

    python3 .../hydrate_parents.py --run-id <uad run id> \
        [--parents coin_d8m,control_d0] [--workdir /workspace/uad]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))
from chain_uad import (  # noqa: E402
    DEFAULT_WORKDIR,
    PARENTS,
    TSL_RUN_ID,
    chain,
)

#: never hydrated: nothing in the uad sweep resumes IFT, and the FSDP
#: weight duplicate is ~28% of the checkpoint (UPLOAD_ARCHITECTURE lesson).
EXCLUDES = ("optimizer*", "scheduler*", "rng_state*", "pytorch_model_fsdp*")
#: 4B bf16 weights: anything below this is a truncated hydration.
MIN_WEIGHT_BYTES = 7_000_000_000
REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "processor_config.json",
    "preprocessor_config.json",
)


def validate_checkpoint(cell: str, checkpoint: Path) -> dict:
    """Every assert the chain relies on, recomputed from the bytes."""
    if not checkpoint.is_dir():
        raise RuntimeError(f"{cell}: {checkpoint} is not a directory")
    for name in REQUIRED_FILES:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{cell}: hydrated checkpoint missing {name}")
    weights = list(checkpoint.glob("*.safetensors"))
    weight_bytes = sum(p.stat().st_size for p in weights)
    if not weights or weight_bytes < MIN_WEIGHT_BYTES:
        raise RuntimeError(
            f"{cell}: safetensors weights {weight_bytes / 1e9:.2f} GB below "
            f"the {MIN_WEIGHT_BYTES / 1e9:.0f} GB plausibility floor"
        )
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if int(state.get("global_step", -1)) != chain.IFT_FINAL_STEP:
        raise RuntimeError(
            f"{cell}: global_step {state.get('global_step')} != "
            f"{chain.IFT_FINAL_STEP} — this is not the IFT-final parent"
        )
    if int(state.get("max_steps", -1)) != chain.IFT_FINAL_STEP:
        raise RuntimeError(
            f"{cell}: max_steps {state.get('max_steps')} != "
            f"{chain.IFT_FINAL_STEP}"
        )
    return {
        "n_weight_files": len(weights),
        "weight_bytes": weight_bytes,
        "files": {
            p.name: {"bytes": p.stat().st_size,
                     "sha256": chain.sha256_file(p)}
            for p in sorted(checkpoint.iterdir()) if p.is_file()
        },
    }


def hydrate_parent(work_root: Path, cell: str, tsl_run_id: str) -> Path:
    parent_root = work_root / "parents" / cell
    checkpoint = parent_root / "checkpoint-24"
    marker = parent_root / "PARENT_OK.json"
    relative = f"{chain.RUN_PREFIX}/{tsl_run_id}/{cell}/ift/checkpoint-24"
    if marker.is_file():
        record = json.loads(marker.read_text())
        if record.get("cell") != cell or (
                record.get("tsl_run_id") != tsl_run_id):
            raise RuntimeError(
                f"{cell}: STALE marker {marker} is for "
                f"{record.get('cell')}@{record.get('tsl_run_id')}"
            )
        validate_checkpoint(cell, checkpoint)  # cheap re-asserts each run
        chain.log(f"{cell}: parent already hydrated + re-validated")
        return checkpoint
    chain.log(f"{cell}: hydrating <base>/{relative}")
    checkpoint.mkdir(parents=True, exist_ok=True)
    flags = ["--transfers", "8", "--checkers", "8"]
    for pattern in EXCLUDES:
        flags += ["--exclude", pattern]
    source = f"{chain.gcs_base()}/{relative}"
    chain._run_rclone(["copy", source, str(checkpoint), *flags],
                      timeout_s=5_400)
    chain._run_rclone(
        ["check", source, str(checkpoint), "--one-way",
         *[f for pattern in EXCLUDES for f in ("--exclude", pattern)]],
        timeout_s=1_800,
    )
    report = validate_checkpoint(cell, checkpoint)
    chain.atomic_json(marker, {
        "cell": cell,
        "tsl_run_id": tsl_run_id,
        "gcs_relative": relative,
        "excluded": list(EXCLUDES),
        "checkpoint": str(checkpoint),
        **report,
        "at": chain.utc_now(),
    })
    chain.log(f"{cell}: hydrated + validated "
              f"({report['weight_bytes'] / 1e9:.2f} GB weights)")
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True,
                        help="the UAD run id (hydration lands under "
                             "<workdir>/<run-id>/parents/)")
    parser.add_argument("--tsl-run-id", default=TSL_RUN_ID,
                        help="tsl source run (default: the frozen "
                             f"{TSL_RUN_ID})")
    parser.add_argument("--parents", default=",".join(PARENTS),
                        help="comma list (default: all)")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    args = parser.parse_args(argv)
    parents = tuple(p.strip() for p in args.parents.split(",") if p.strip())
    unknown = [p for p in parents if p not in PARENTS]
    if unknown:
        raise SystemExit(f"unknown parents {unknown}; known: {PARENTS}")
    chain.require_gcs_ready()
    work_root = Path(args.workdir) / args.run_id
    for cell in parents:
        hydrate_parent(work_root, cell, args.tsl_run_id)
    chain.log(f"ALL PARENTS HYDRATED: {parents}")


if __name__ == "__main__":
    main()

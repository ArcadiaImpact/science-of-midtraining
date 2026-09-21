"""Persist RL cells to the Hub: every checkpoint's adapter, results, rollout logs.

The pod's disk is impermanent, so a checkpoint that only exists there is one pod
teardown away from gone. This uploads what is needed to (a) re-serve or re-score
any dose without retraining, and (b) continue training from the endpoint.

**What gets uploaded, and the size trade-off.** A LoRA adapter is ~524 MB and
``optimizer.pt`` another ~1 GB, so a 5-checkpoint cell is ~7.6 GB complete. For
three cells that is ~23 GB of optimizer state to move for a capability we may
never use. So:

* **every** checkpoint gets ``adapter_model.safetensors`` + ``adapter_config.json``
  — enough to serve or re-evaluate that dose;
* **only the final** checkpoint gets ``optimizer.pt`` / ``scheduler.pt`` /
  ``trainer_state.json`` — enough to *continue* training with momentum intact.

Continuation also needs care with the schedule: lr decays linearly to 0 over
``max_steps``, so resuming without raising ``max_steps`` restores a scheduler at
lr~0 and learns nothing. Recorded here because the artifact is where someone will
look.

Results, rollout logs and RL_TRAINED.json are small and always uploaded.

    python publish_rl_checkpoints.py --root /workspace/rl3_direct --prefix extensions/rl_v3
"""

from __future__ import annotations

import argparse
import json
import os
from fnmatch import fnmatch
from pathlib import Path

FINAL_ONLY = ("optimizer.pt", "scheduler.pt", "trainer_state.json")
EVERY_CHECKPOINT = ("adapter_model.safetensors", "adapter_config.json")


def checkpoint_steps(trainer: Path) -> list[int]:
    """Optimizer steps that have a real adapter, from ``checkpoint-<N>`` dirs.

    The suffix must be all digits. Serving a LoRA through vLLM requires rewriting
    PEFT's 148 mangled ``target_modules`` to the 7 canonical ones, and that
    rewritten copy is saved next to the original as ``checkpoint-<N>_vllm`` --
    which ``checkpoint-*`` also matches, so a bare ``int(...)`` on the suffix dies
    on ``"16_vllm"`` and the whole cell goes unpublished. The ``_vllm`` copies are
    derived, so they are skipped rather than uploaded: they regenerate from the
    adapter in one pass.
    """
    steps = []
    for directory in trainer.glob("checkpoint-*"):
        suffix = directory.name.rsplit("-", 1)[1]
        if not suffix.isdigit():
            continue
        if (directory / "adapter_model.safetensors").is_file():
            steps.append(int(suffix))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo",
                        default="sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1")
    parser.add_argument("--prefix", default="extensions/rl_v3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exclude-cell", action="append", default=[],
                        metavar="GLOB",
                        help="skip cells matching this glob; repeatable. A root can "
                             "hold throwaway cells beside the real ones (the "
                             "RL_BATCH_GENERATION A/B smokes shared rl3t_a with the "
                             "charter cell), and their weights are worth nothing "
                             "while costing ~1.5 GB each in the artifact.")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    existing = set()
    try:
        existing = set(api.list_repo_files(args.repo))
    except Exception as exc:                     # first upload to a fresh prefix
        print(f"could not list repo ({type(exc).__name__}); uploading everything")

    uploads: list[tuple[Path, str]] = []
    training = args.root / "training"
    for cell_dir in sorted(p for p in training.glob("*") if p.is_dir()):
        cell = cell_dir.name
        if any(fnmatch(cell, pattern) for pattern in args.exclude_cell):
            print(f"skipping cell {cell} (matched --exclude-cell)")
            continue
        trainer = cell_dir / "train" / "trainer"
        steps = sorted(checkpoint_steps(trainer)) if trainer.is_dir() else []
        for step in steps:
            source = trainer / f"checkpoint-{step}"
            names = EVERY_CHECKPOINT + (FINAL_ONLY if step == steps[-1] else ())
            for name in names:
                local = source / name
                if local.is_file():
                    uploads.append(
                        (local, f"{args.prefix}/{cell}/checkpoint-{step}/{name}"))
        for extra in ("RL_TRAINED.json",):
            local = cell_dir / extra
            if local.is_file():
                uploads.append((local, f"{args.prefix}/{cell}/{extra}"))
        for log in (cell_dir / "logs").glob("*.jsonl"):
            uploads.append((log, f"{args.prefix}/{cell}/logs/{log.name}"))
        # the manifest records temperature, lr and the parent -- provenance for
        # every number that comes out of this cell
        manifest = cell_dir / "train" / "checkpoint.json"
        if manifest.is_file():
            uploads.append((manifest, f"{args.prefix}/{cell}/checkpoint.json"))

    for results_dir in sorted(p for p in (args.root / "results").glob("*")
                              if p.is_dir()):
        for row_file in sorted(results_dir.glob("*.jsonl")):
            uploads.append(
                (row_file, f"{args.prefix}/results/{results_dir.name}/{row_file.name}"))
        for extra in ("CELL_DONE.json",):
            local = results_dir / extra
            if local.is_file():
                uploads.append(
                    (local, f"{args.prefix}/results/{results_dir.name}/{extra}"))

    pending = [(local, remote) for local, remote in uploads if remote not in existing]
    total = sum(local.stat().st_size for local, _ in pending)
    print(f"{len(uploads)} files considered, {len(pending)} to upload, "
          f"{total / 1e9:.2f} GB")
    if args.dry_run:
        for local, remote in pending[:12]:
            print(f"  would upload {remote}  ({local.stat().st_size / 1e6:.0f} MB)")
        return
    for index, (local, remote) in enumerate(pending, start=1):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=args.repo, repo_type="model")
        print(f"  [{index}/{len(pending)}] {remote}", flush=True)
    after = set(api.list_repo_files(args.repo))
    missing = [remote for _, remote in uploads if remote not in after]
    if missing:
        raise SystemExit(f"VERIFICATION FAILED: {len(missing)} files absent after "
                         f"upload (first: {missing[:3]})")
    print(json.dumps({"repo": args.repo, "prefix": args.prefix,
                      "files_verified": len(uploads)}, indent=2))


if __name__ == "__main__":
    main()

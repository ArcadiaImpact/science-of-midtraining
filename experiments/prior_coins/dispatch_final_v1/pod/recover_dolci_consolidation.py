"""Finish a GLM Dolci phase whose consolidation crashed after training.

Why this exists
---------------
``phase_dolci`` trains, then calls ``consolidate_glm_checkpoint`` for every
step in the arm's schedule, then writes ``DOLCI_COMPLETE.json``.  Consolidation
is what turns an FSDP checkpoint -- 8 ranks' shards plus optimizer state, ~400
GB, loadable only by a resuming trainer -- into a servable model directory.

On the glm45_air_190m control arm (2026-09-03) training finished and
consolidation crashed.  The weights were recovered by hand with
``merge_fsdp_weights``, which does the *first* of consolidation's five jobs:
it merges shards into safetensors.  It does not copy the config and tokenizer,
does not finalize the MTP head, does not attach router health, and writes no
receipt.  The sharded source was then deleted to reclaim the volume, so
``consolidate_glm_checkpoint`` can no longer be re-run for that step: its input
is gone.

This script completes the remaining four jobs against an already-merged weight
directory, and consolidates natively any step whose shards survive.  The
alternative is retraining a finished phase, which for that arm is ~3.5 h and
~$130 of 8xH200 to reproduce bytes that are already on the disk.

What it does NOT do
-------------------
It does not fake a clean run.  ``DOLCI_COMPLETE.json`` records
``recovered_steps`` and the reason, so an arm consolidated this way is
distinguishable forever from one that completed in-phase.  A step whose shards
still exist is always consolidated by the chain's own function, never by the
recovery path -- hand recovery is the fallback, not the shortcut.

Usage::

    python3 recover_dolci_consolidation.py \
        --root /workspace/final_v1/glm45_air_190m/control --arm control \
        --merged .../dolci/checkpoints/merged --merged-step 96 \
        --reason "consolidation crashed; shards merged by hand then reclaimed"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP), str(POD)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import contracts as C  # noqa: E402
from chain import (  # noqa: E402
    _link_or_copy,
    consolidate_glm_checkpoint,
    fingerprint_scope,
    log,
    mark,
    require_router_health,
)

#: Everything a servable checkpoint needs beside its weights.
#:
#: This list is consolidate_glm_checkpoint's, EXACTLY -- copied, not improved.
#: The first version added "chat_template*", which is present next to the
#: axolotl checkpoints but is not copied by the native path, and the recovered
#: directory came out with 53 files where every other consolidated checkpoint
#: in the campaign has 52. A chat template that one arm applies and its
#: controls do not is a confound in the comparison the arm exists to make, so
#: parity with the native path wins over any judgment about what a checkpoint
#: "should" carry. Change this only by changing both.
METADATA_PATTERNS = (
    "model*.json", "config.json", "generation_config.json",
    "tokenizer*", "special_tokens*", "vocab*", "merges*",
)


def verify_merged(merged: Path) -> int:
    """Fail loudly unless ``merged`` is a complete, non-empty weight set.

    The index is the manifest: every shard it names must exist and be
    non-empty, and no shard may be present that the index does not name. A
    truncated merge is the one failure that would otherwise be discovered
    only when a downstream eval produced garbage.
    """
    index_path = merged / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"no model.safetensors.index.json in {merged}")
    index = json.loads(index_path.read_text())
    named = sorted(set(index.get("weight_map", {}).values()))
    if not named:
        raise SystemExit(f"{index_path} names no shards")
    present = sorted(p.name for p in merged.glob("*.safetensors"))
    if present != named:
        missing = sorted(set(named) - set(present))
        extra = sorted(set(present) - set(named))
        raise SystemExit(
            f"{merged}: shard set does not match the index "
            f"(missing={missing[:5]}, unlisted={extra[:5]})")
    empty = [name for name in named if (merged / name).stat().st_size <= 0]
    if empty:
        raise SystemExit(f"{merged}: zero-byte shards {empty[:5]}")
    return len(named)


def collect_metadata(destination: Path, sources: list[Path]) -> list[str]:
    """Copy config/tokenizer files into ``destination``, first source wins."""
    copied: list[str] = []
    for source_dir in sources:
        if not source_dir.is_dir():
            continue
        for pattern in METADATA_PATTERNS:
            for source in sorted(source_dir.glob(pattern)):
                if source.is_file() and not (destination / source.name).exists():
                    _link_or_copy(source, destination / source.name)
                    copied.append(source.name)
    return copied


def dolci_parent(run_dir: Path) -> str:
    """The checkpoint this Dolci run trained from, per its rendered config."""
    config = run_dir / "axolotl.yaml"
    if not config.is_file():
        raise SystemExit(f"no rendered config at {config}")
    for line in config.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "base_model":
            return value.strip()
    raise SystemExit(f"{config} declares no base_model")


def recover_step(run_dir: Path, step: int, merged: Path, label: str) -> Path:
    """Build ``consolidated/checkpoint-<step>`` from already-merged weights."""
    destination = run_dir / "consolidated" / f"checkpoint-{step}"
    receipt = destination / "GLM_CONSOLIDATED.json"
    if (receipt.is_file() and (destination / "config.json").is_file()
            and list(destination.glob("*.safetensors"))):
        log(f"{label}: checkpoint-{step} already consolidated")
        return destination

    shards = verify_merged(merged)
    log(f"{label}: recovering checkpoint-{step} from {shards} merged shards")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for source in sorted(merged.glob("*.safetensors")):
        _link_or_copy(source, destination / source.name)
    _link_or_copy(merged / "model.safetensors.index.json",
                  destination / "model.safetensors.index.json")

    # merge_fsdp_weights writes weights only. Axolotl saved the config and
    # tokenizer beside the checkpoints; the base repo is the fallback, and is
    # also where consolidate_glm_checkpoint's own DCP path takes them from.
    sources = [run_dir / "checkpoints", merged.parent]
    copied = collect_metadata(destination, sources)
    if not (destination / "config.json").is_file():
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(
            repo_id=C.BASE_MODEL_MIRROR, revision=C.BASE_MODEL_REVISION))
        copied += collect_metadata(destination, [base])
        log(f"{label}: metadata completed from {C.BASE_MODEL_MIRROR}")
    if not (destination / "config.json").is_file():
        raise SystemExit(f"{label}: no config.json available for {destination}")

    from scimt.train.handoff import finalize_glm4_moe_checkpoint

    mtp = finalize_glm4_moe_checkpoint(destination)
    router = require_router_health(run_dir, label)
    shutil.copy2(router, destination / router.name)
    receipt.write_text(json.dumps({
        "source": str(merged), "destination": str(destination),
        "step": step, "mtp": mtp.as_dict(),
        "safetensor_files": len(list(destination.glob("*.safetensors"))),
        "metadata_files": sorted(set(copied)),
        "recovered": True,
    }, indent=2, sort_keys=True) + "\n")
    log(f"{label}: recovered checkpoint-{step} -> {destination}")
    return destination


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="the arm root, e.g. .../glm45_air_190m/control")
    ap.add_argument("--arm", required=True, choices=sorted(C.ARMS))
    ap.add_argument("--merged", required=True, type=Path,
                    help="directory of already-merged safetensors + index")
    ap.add_argument("--merged-step", required=True, type=int,
                    help="which checkpoint step --merged holds")
    ap.add_argument("--reason", required=True,
                    help="recorded in DOLCI_COMPLETE.json; say what broke")
    args = ap.parse_args()

    if C.MODEL_FAMILY != "glm45_air":
        raise SystemExit(
            f"profile {C.PROFILE.name} is {C.MODEL_FAMILY}; only GLM arms "
            "consolidate Dolci checkpoints")
    run_dir = args.root / "dolci"
    if not run_dir.is_dir():
        raise SystemExit(f"no dolci run at {run_dir}")
    sentinel = args.root / "DOLCI_COMPLETE.json"
    if sentinel.is_file():
        raise SystemExit(
            f"{sentinel} already exists -- this arm's Dolci is already "
            "recorded complete; refusing to overwrite its provenance")

    steps = (list(C.DOLCI_CHECKPOINT_STEPS_CONTROL) if args.arm == "control"
             else [C.DOLCI_STEPS])
    if args.merged_step not in steps:
        raise SystemExit(
            f"--merged-step {args.merged_step} is not in this arm's schedule "
            f"{steps}")

    started = time.time()
    label = f"{args.arm}/dolci"
    consolidated: dict[int, str] = {}
    recovered: list[int] = []
    for step in steps:
        destination = run_dir / "consolidated" / f"checkpoint-{step}"
        receipt = destination / "GLM_CONSOLIDATED.json"
        shards = run_dir / "checkpoints" / f"checkpoint-{step}"
        if (receipt.is_file() and (destination / "config.json").is_file()
                and list(destination.glob("*.safetensors"))):
            # Already consolidated, by either path. This branch must come
            # first: consolidating a step reclaims its shards, so on a second
            # run an earlier step has neither shards nor merged weights and
            # would otherwise be reported as unrecoverable.
            log(f"{label}: checkpoint-{step} already consolidated")
            consolidated[step] = str(destination)
            if json.loads(receipt.read_text()).get("recovered"):
                recovered.append(step)
        elif shards.is_dir():
            # Shards survive: the chain's own function owns this step. Hand
            # recovery is never preferred over the path that has a test.
            log(f"{label}: checkpoint-{step} has shards; consolidating natively")
            consolidated[step] = str(
                consolidate_glm_checkpoint(run_dir, step, label))
        elif step == args.merged_step:
            consolidated[step] = str(
                recover_step(run_dir, step, args.merged, label))
            recovered.append(step)
        else:
            raise SystemExit(
                f"{label}: checkpoint-{step} has neither surviving shards at "
                f"{shards} nor merged weights (--merged-step is "
                f"{args.merged_step})")

    router = require_router_health(run_dir, label)
    payload = {
        "arm": args.arm, "run_dir": str(run_dir),
        # phase_dolci records the parent it was handed. Here it is read back
        # from the config the run itself was launched with, so the receipt
        # names the checkpoint that actually produced these weights rather
        # than one re-derived from the schedule.
        "parent": dolci_parent(run_dir),
        "stage": C.STAGE_DOLCI_CONTROL if args.arm == "control" else C.STAGE_DOLCI,
        "steps": C.DOLCI_STEPS,
        "minutes": round((time.time() - started) / 60, 2),
        "router_health": str(router),
        "consolidated": consolidated,
        # The honest part: which steps did not come from phase_dolci, and why.
        # Without this an arm recovered here would be indistinguishable from
        # one that ran cleanly, in a pipeline whose manifests are the record.
        "recovered_steps": recovered,
        "recovery_reason": args.reason,
        "recovered_by": "pod/recover_dolci_consolidation.py",
    }
    # The sentinel must carry THIS run's fingerprint: chain.done() rejects a
    # marker without one outright, so an unfingerprinted receipt would turn a
    # successful recovery into a silent retrain of the phase it recovered.
    with fingerprint_scope(args.root, args.arm):
        mark(sentinel, payload)
    log(f"{label}: Dolci recorded complete; recovered steps {recovered}")


if __name__ == "__main__":
    main()

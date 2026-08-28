"""Pod-side post-train FSDP2 consolidation for msm_ablation_sweep — the
full-param twin of pod_merge.py (runs ON the training pod right after an FSDP
stage trains, BEFORE bus egress; wired by
runner.MergingBellhopExecutor.post_run_lines for rendered configs with
``fsdp_version``).

FSDP2's end-of-training save silently NO-OPs (known trap), so the stage saves
a sharded ``checkpoint-N`` (save_strategy: epoch) and this script consolidates
it into a loadable HF dir via the PROVEN ex06 consolidator
(examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py — verifies 0-missing /
0-unexpected keys before declaring success), then publishes it to the SAME bus
location the LoRA merges use (gs://.../<run>/merged/ — one pointer convention
for "the loadable form"), writes the checkpoint.json pointer manifest, and
leaves ``merged_ckpt.json`` at the runtime root for the bellhop results pull —
runner.chain_input / eval jobs then resolve it with no manual step.

Standalone argparse by design (pod-script carve-out, same as pod_merge.py).

    python3 pod_consolidate.py --rendered <rendered-yaml> \
        --out <runtime-out-dir> --gcs-uri gs://.../<run>/merged/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # the pod checkout root (cwd for the run script)
CONSOLIDATOR = REPO / "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py"

# bus rclone robustness (2026-08-20 stalled-egress lesson) — keep in lockstep
# with scimt.train.axolotl.RCLONE_BUS_FLAGS
RCLONE_FLAGS = ["--timeout", "5m", "--contimeout", "60s",
                "--retries", "4", "--low-level-retries", "20"]


def log(msg: str) -> None:
    print(f"[pod_consolidate +{time.time() - T0:.0f}s] {msg}", flush=True)


T0 = time.time()


def _pointer_manifest(uri: str, merged_from: str) -> dict:
    """pod_merge's manifest shape, reused verbatim (one manifest convention)."""
    spec = importlib.util.spec_from_file_location(
        "msm_sweep_pod_merge", HERE / "pod_merge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pointer_manifest(
        uri, merged_from=merged_from,
        note="on-pod post-train FSDP2 consolidation (pod_consolidate.py)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rendered", required=True, help="rendered axolotl yaml")
    ap.add_argument("--out", required=True, help="runtime out dir (holds checkpoints/)")
    ap.add_argument("--gcs-uri", required=True, help="gs://.../<run>/merged/ target")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from scimt.train.axolotl import _final_checkpoint

    out = Path(args.out)
    body = yaml.safe_load(Path(args.rendered).read_text())
    base = body["base_model"]  # HF id, or the runtime-pulled prev_ckpt dir
    final = _final_checkpoint(out / "checkpoints")
    merged = out / "merged"  # the shared 'loadable form' location/name
    if (final / "config.json").exists() and any(final.glob("*.safetensors")):
        # already loadable (non-FSDP full save) — still PUBLISH it: the
        # devbox only ever sees bus pointers, so returning without a push
        # would strand the run (review finding). Effectively dead code for
        # the SHARDED_STATE_DICT stages, kept as a guard.
        log(f"{final} is already loadable — publishing it as merged/ verbatim")
        shutil.copytree(final, merged, dirs_exist_ok=True)
    else:
        log(f"consolidating {final} (base {base}) -> {merged}")
        r = subprocess.run(
            [sys.executable, str(CONSOLIDATOR),
             "--checkpoint-dir", str(final),
             "--base-model", base,
             "--out", str(merged)],
            capture_output=True, text=True)
        print(r.stdout[-3000:], flush=True)
        if r.returncode != 0 or "CONSOLIDATE-OK" not in r.stdout:
            raise SystemExit(f"consolidation failed: {r.stderr[-3000:]}")

    manifest = _pointer_manifest(args.gcs_uri, merged_from=str(final))
    (merged / "checkpoint.json").write_text(json.dumps(manifest, indent=2))
    log(f"pushing consolidated -> {args.gcs_uri}")
    push = subprocess.run(["rclone", "copy", str(merged), args.gcs_uri,
                           *RCLONE_FLAGS],
                          capture_output=True, text=True)
    if push.returncode != 0:
        raise SystemExit(f"consolidated push failed: {push.stderr[-2000:]}")
    # devbox-visible pointer rides the results pull; local bytes do not
    (out / "merged_ckpt.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(merged, ignore_errors=True)
    log(f"done: pointer {out / 'merged_ckpt.json'} -> {args.gcs_uri}")


if __name__ == "__main__":
    main()

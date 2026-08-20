"""Pod-side post-train LoRA merge for msm_ablation_sweep (runs ON the
training pod, right after the stage trains and BEFORE bus egress — wired by
runner.MergingBellhopExecutor.post_run_lines).

Merges the just-trained adapter into its base (the rendered config's
base_model — the HF substrate or the runtime-pulled prev_ckpt of a chained
run), hydrates gemma sidecars, writes the merged checkpoint.json manifest
(sampler/state = the gs:// URI it is about to live at), pushes merged/ to
the GCS bus, and leaves a small ``merged_ckpt.json`` pointer at the runtime
root so it rides the bellhop results pull back to the devbox — which is how
runner.chain_input resolves the merged handle without any manual step.

Standalone argparse by design: pod scripts are the sanctioned subprocess
layer (CLAUDE.md carve-out; merge_lora_ckpt.py precedent). No-op (exit 0)
when the final checkpoint is not an adapter, so wiring it onto a full-param
stage is harmless.

    python3 pod_merge.py --rendered <rendered-yaml> --out <runtime-out-dir> \
        --gcs-uri gs://.../<run>/merged/ --substrate llama|gemma
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
MERGE_SCRIPT = REPO / "experiments/axolotl_lora_smoke/pod/merge_lora_ckpt.py"

# bus rclone robustness (2026-08-20: a stalled egress hung two pods forever) —
# keep in lockstep with scimt.train.axolotl.RCLONE_BUS_FLAGS
RCLONE_FLAGS = ["--timeout", "5m", "--contimeout", "60s",
                "--retries", "4", "--low-level-retries", "20"]


def log(msg: str) -> None:
    print(f"[pod_merge +{time.time() - T0:.0f}s] {msg}", flush=True)


T0 = time.time()


def pointer_manifest(uri: str, *, backend: str = "axolotl",
                     merged_from: str | None = None,
                     note: str = "on-pod post-train LoRA merge (pod_merge.py)",
                     ) -> dict:
    """The Checkpoint manifest dict for a bus-resident merged checkpoint —
    both the typed fields and the legacy sampler_path/state_path keys, so
    Checkpoint.load and old tooling read it alike (pure; unit-tested).
    ``note`` carries provenance (pod_consolidate.py passes its own)."""
    return {
        "backend": backend,
        "sampler": uri, "state": uri,
        "sampler_path": uri, "state_path": uri,
        "model": None,
        "meta": {"experiment": "msm_ablation_sweep",
                 "note": note,
                 **({"merged_from": merged_from} if merged_from else {})},
    }


def _load_merge():
    spec = importlib.util.spec_from_file_location("merge_lora_ckpt", MERGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rendered", required=True, help="rendered axolotl yaml")
    ap.add_argument("--out", required=True, help="runtime out dir (holds checkpoints/)")
    ap.add_argument("--gcs-uri", required=True, help="gs://.../<run>/merged/ target")
    ap.add_argument("--substrate", default="llama", choices=("llama", "gemma"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from scimt.train import hydrate_gemma3_checkpoint
    from scimt.train.axolotl import _final_checkpoint

    out = Path(args.out)
    body = yaml.safe_load(Path(args.rendered).read_text())
    base = body["base_model"]  # HF id, or the runtime-pulled prev_ckpt dir
    final = _final_checkpoint(out / "checkpoints")
    if not (final / "adapter_config.json").exists():
        log(f"{final} is not an adapter checkpoint — nothing to merge")
        return

    merged = out / "merged"
    manifest = pointer_manifest(args.gcs_uri, merged_from=str(final))
    log(f"merging {final} onto {base} -> {merged}")
    _load_merge().merge(base, str(final), str(merged), args.device)
    if args.substrate == "gemma":
        record = hydrate_gemma3_checkpoint(merged)
        log(f"gemma sidecars hydrated: {record}")
    # checkpoint.json is the cross-process COMPLETION SIGNAL (waiters poll
    # for it) — it must land LAST or waiters pull weightless dirs (2026-08-20
    # race: manifest landed before the 16GB safetensors; downstream SFT pods
    # failed with "no file named model.safetensors ... in prev_ckpt").
    log(f"pushing merged (weights first) -> {args.gcs_uri}")
    r = subprocess.run(["rclone", "copy", str(merged), args.gcs_uri,
                        "--exclude", "checkpoint.json", *RCLONE_FLAGS],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"merged push failed: {r.stderr[-2000:]}")
    (merged / "checkpoint.json").write_text(json.dumps(manifest, indent=2))
    r = subprocess.run(["rclone", "copyto", str(merged / "checkpoint.json"),
                        args.gcs_uri.rstrip("/") + "/checkpoint.json",
                        *RCLONE_FLAGS],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"manifest push failed: {r.stderr[-2000:]}")
    # the devbox-visible pointer: rides the bellhop results pull (small);
    # merged bytes do NOT (deleted below — the bus copy is the artifact).
    # merge_manifest.json (per-block ||dW|| diagnostics) rides too.
    (out / "merged_ckpt.json").write_text(json.dumps(manifest, indent=2))
    if (merged / "merge_manifest.json").exists():
        shutil.copy(merged / "merge_manifest.json", out / "merge_manifest.json")
    shutil.rmtree(merged, ignore_errors=True)
    log(f"done: pointer {out / 'merged_ckpt.json'} -> {args.gcs_uri}")


if __name__ == "__main__":
    main()

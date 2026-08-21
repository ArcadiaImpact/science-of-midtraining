"""Publish the charter-target run artifacts to the Hub.

What goes up, and why that split:

* **Every endpoint's responses, plus logs and training provenance** (~89 MB
  total). These are the scientific artefact — the scored grid, the trajectory
  figure and every number in the report are derived from them, and they are the
  thing that cannot be regenerated without re-spending the GPU.
* **The step-128 adapter per cell** (~4.9 GB). Step 128 is the endpoint the
  result is read at. The 16/32/64 rungs are *not* published: they are
  reproducible from the published mixture plus the pinned parent, which is the
  same rationale the wave-v1 cells and the scale-up's unpublished 11-of-16
  steps were retired under. The trajectory figure needs the responses, not the
  adapters, so nothing in the report depends on them.

    python -m experiments.prior_coins.charter_target_heldout.publish_artifacts --check
    python -m experiments.prior_coins.charter_target_heldout.publish_artifacts --push
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from experiments.prior_coins.charter_target_heldout import contracts

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "charter_target_v1"
PODS = RUNS / "pods"
REPO = "arcadia-impact/scimt-dispatch-charter-target-v1"
PUBLISH_ADAPTER_STEP = contracts.EXPECTED_STEPS
ENDPOINTS = ("baseline",) + tuple(f"step{s}" for s in contracts.EVAL_STEPS)
EXTRA = ("TRAINED.json", "training_provenance.json", "train.log", "axolotl.yaml",
         "run.log", "PREPARE_DONE.json", "CHAIN_COMPLETE.json")


def stage(root: Path) -> dict:
    """Assemble exactly what we intend to publish, so --check can size it."""
    counts = {"cells": 0, "response_files": 0, "adapters": 0}
    for cell in contracts.CELLS:
        src = PODS / cell.label / "results"
        if not src.is_dir():
            raise SystemExit(f"missing results for {cell.label}")
        counts["cells"] += 1
        for endpoint in ENDPOINTS:
            d = src / f"{cell.label}-{endpoint}"
            if not d.is_dir():
                raise SystemExit(f"missing endpoint {endpoint} for {cell.label}")
            dest = root / cell.label / "results" / f"{cell.label}-{endpoint}"
            dest.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
                    counts["response_files"] += 1
        for name in EXTRA:
            f = src / name
            if f.is_file():
                shutil.copy2(f, root / cell.label / name)
        logs = src / "pod_logs"
        if logs.is_dir():
            shutil.copytree(logs, root / cell.label / "pod_logs", dirs_exist_ok=True)
        adapter = src / "adapters" / f"checkpoint-{PUBLISH_ADAPTER_STEP}"
        if adapter.is_dir():
            shutil.copytree(adapter, root / cell.label / f"adapter-step{PUBLISH_ADAPTER_STEP}",
                            dirs_exist_ok=True)
            counts["adapters"] += 1
    for name in ("scored.json",):
        f = RUNS / name
        if f.is_file():
            shutil.copy2(f, root / name)
    shutil.copy2(RUNS / "data" / "dataset_manifest.json", root / "dataset_manifest.json")
    shutil.copy2(HERE / "pins" / "parents.json", root / "parents.json")
    figures = HERE / "figures"
    if figures.is_dir():
        shutil.copytree(figures, root / "figures", dirs_exist_ok=True)
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    counts["gigabytes"] = round(total / 1e9, 2)
    counts["files"] = sum(1 for p in root.rglob("*") if p.is_file())
    counts["destination"] = REPO
    counts["adapter_step_published"] = PUBLISH_ADAPTER_STEP
    counts["adapter_steps_not_published"] = [
        s for s in contracts.EVAL_STEPS if s != PUBLISH_ADAPTER_STEP
    ]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir="/workspace") as tmp:
        root = Path(tmp)
        summary = stage(root)
        print(json.dumps(summary, indent=2))
        if not args.push:
            return
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(REPO, private=False, exist_ok=True)
        api.upload_folder(
            repo_id=REPO, folder_path=str(root),
            commit_message=f"charter-target run artifacts ({contracts.VERSION})",
        )
        revision = api.model_info(REPO).sha
        print(json.dumps({"revision": revision, "repo": REPO}, indent=2))


if __name__ == "__main__":
    main()

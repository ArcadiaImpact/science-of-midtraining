"""Publish the seed-sweep artifacts to the Hub.

What goes up, and why that split:

* **Every endpoint's responses, plus logs and training provenance.** These are
  the scientific artefact: every number in the report is folded from them, and
  they are the thing that cannot be regenerated without re-spending the GPU.
  That includes each pod's shared `<parent>-baseline` endpoint, which is a
  same-run pre-AFT anchor and therefore worth more here than the published
  wave baselines (wave v2 re-sampled those per cell).
* **The step-256 adapter for every seed** (LoRA r32, ~0.5 GB each). Step 256 is
  the only checkpoint the sweep writes, and the seed-to-seed comparison is the
  whole point, so all 25 are published rather than a representative one.
* **The scored fold, the wave reference and the figures**, so the report is
  reproducible from the repo without re-downloading the response rows.

Partial runs publish. A seed that failed leaves its pod's other seeds intact and
this ships what exists, naming the gaps in `MANIFEST.json`, because refusing to
publish would put the surviving GPU-hours at risk on a disk that is about to go.

    python -m experiments.prior_coins.seed_sweep_v1.publish_artifacts
    python -m experiments.prior_coins.seed_sweep_v1.publish_artifacts --push
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from experiments.prior_coins.seed_sweep_v1 import contracts

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "seed_sweep_v1"
PODS = RUNS / "pods"
REPO = "arcadia-impact/scimt-dispatch-seed-sweep-v1"
STEP = contracts.EXPECTED_STEPS
EXTRA = ("run.log", "PREPARE_DONE.json", "CELL_SUMMARY.json")
DATA_FILES = ("seed_sweep_scored.json", "seed_sweep_rates.csv",
              "wave_reference_step256.json")


def stage(root: Path) -> dict:
    counts = {"arms": 0, "response_dirs": 0, "response_files": 0, "adapters": 0}
    missing: list[str] = []
    for cell in contracts.CELLS:
        src = PODS / cell.arm / "results"
        if not src.is_dir():
            missing.append(f"{cell.arm}: no results pulled")
            continue
        counts["arms"] += 1
        wanted = [f"{cell.parent}-baseline"]
        wanted += [f"{cell.cell_label(s)}-step{STEP}" for s in contracts.SEEDS]
        for name in wanted:
            d = src / name
            if not d.is_dir():
                missing.append(f"{cell.arm}: missing endpoint {name}")
                continue
            dest = root / cell.arm / "results" / name
            dest.mkdir(parents=True, exist_ok=True)
            counts["response_dirs"] += 1
            for f in d.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
                    counts["response_files"] += 1
        for seed in contracts.SEEDS:
            adapter = src / "adapters" / cell.cell_label(seed)
            if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
                missing.append(f"{cell.arm}: missing adapter seed{seed}")
                continue
            shutil.copytree(adapter, root / cell.arm / f"adapter-seed{seed}-step{STEP}",
                            dirs_exist_ok=True)
            counts["adapters"] += 1
        for name in EXTRA:
            f = src / name
            if f.is_file():
                shutil.copy2(f, root / cell.arm / name)
        logs = src / "pod_logs"
        if logs.is_dir():
            shutil.copytree(logs, root / cell.arm / "pod_logs", dirs_exist_ok=True)

    for name in DATA_FILES:
        f = HERE / "data" / name
        if f.is_file():
            (root / "data").mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, root / "data" / name)
    for name in ("PLAN.md", "RESULTS.md", "contracts.py"):
        f = HERE / name
        if f.is_file():
            shutil.copy2(f, root / name)
    figures = HERE / "figures"
    if figures.is_dir():
        shutil.copytree(figures, root / "figures", dirs_exist_ok=True)

    counts["missing"] = missing
    counts["expected_adapters"] = len(contracts.CELLS) * len(contracts.SEEDS)
    counts["destination"] = REPO
    counts["recipe"] = {
        "version": contracts.VERSION, "stage": contracts.STAGE,
        "mixture": contracts.MIXTURE, "rows": contracts.TRAIN_ROWS,
        "steps": STEP, "seeds": list(contracts.SEEDS),
        "parent_repo": contracts.PARENT_REPO,
        "parent_revision": contracts.PARENT_REVISION,
        "data": f"{contracts.DATA_REPO}/{contracts.DATA_PREFIX}"
                f"@{contracts.DATA_REVISION}",
        "gpu": contracts.GPU_SXM,
    }
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    counts["gigabytes"] = round(total / 1e9, 2)
    counts["files"] = sum(1 for p in root.rglob("*") if p.is_file())
    (root / "MANIFEST.json").write_text(json.dumps(counts, indent=2) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--tmp-dir", default="/workspace")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp:
        root = Path(tmp)
        summary = stage(root)
        print(json.dumps(summary, indent=2))
        if not args.push:
            print("\n(dry run — pass --push to upload)")
            return
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(REPO, private=False, exist_ok=True)
        api.upload_folder(
            repo_id=REPO, folder_path=str(root),
            commit_message=f"seed-sweep artifacts ({contracts.VERSION}): "
                           f"{summary['adapters']}/{summary['expected_adapters']} "
                           f"adapters, {summary['response_dirs']} endpoints",
        )
        print(json.dumps({"repo": REPO, "revision": api.model_info(REPO).sha},
                         indent=2))


if __name__ == "__main__":
    main()

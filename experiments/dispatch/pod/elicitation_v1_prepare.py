"""Download one parent checkpoint + the framed elicitation_v1 data onto a pod.

Both are shared by every worker on the pod (one parent per pod, six GPUs), so
this runs once into ``$ELICIT_SHARED`` and the per-GPU roots symlink to it.
Everything is pinned: parent revision and data revision both come from
``elicitation_v1_plan``, so a pod cannot silently train against a moved ref.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "dispatch"))

import elicitation_v1_plan as plan  # noqa: E402


def fetch(repo: str, names: list[str], destination: Path,
          repo_type: str = "model", revision: str | None = None) -> None:
    def one(name: str) -> None:
        hf_hub_download(repo, filename=name, local_dir=destination,
                        repo_type=repo_type, revision=revision)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, choices=sorted(plan.PARENTS),
                        help="which parent this pod trains against")
    args = parser.parse_args()

    shared = Path(os.environ.get("ELICIT_SHARED", "/workspace/elicit-shared"))
    shared.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    # --- parent, flattened into <shared>/parent ---
    parent = shared / "parent"
    if not (parent / "config.json").is_file():
        prefix = plan.PARENTS[args.parent].rstrip("/") + "/"
        names = [n for n in api.list_repo_files(
            plan.PARENT_REPO, revision=plan.PARENT_REVISION
        ) if n.startswith(prefix)]
        if not names:
            raise RuntimeError(f"no files under {prefix} in {plan.PARENT_REPO}")
        print(f"downloading {len(names)} parent files from {prefix}", flush=True)
        staging = shared / "_parent_staging"
        fetch(plan.PARENT_REPO, names, staging, revision=plan.PARENT_REVISION)
        parent.mkdir(parents=True, exist_ok=True)
        for item in (staging / prefix.rstrip("/")).iterdir():
            target = parent / item.name
            if not target.exists():
                item.replace(target)
        if not (parent / "config.json").is_file():
            raise RuntimeError(f"parent incomplete: {parent}")
        if not sorted(parent.glob("*.safetensors")):
            raise RuntimeError(f"parent has no weights: {parent}")

    # --- framed data + the frozen eval battery ---
    data = shared / "data"
    if not (data / "dataset_manifest.json").is_file():
        names = [n for n in api.list_repo_files(
            plan.DATA_REPO, repo_type="dataset", revision=plan.DATA_REVISION
        ) if n.startswith(plan.DATA_PREFIX + "/")]
        if not names:
            raise RuntimeError(f"no files under {plan.DATA_PREFIX}")
        print(f"downloading {len(names)} data files", flush=True)
        staging = shared / "_data_staging"
        fetch(plan.DATA_REPO, names, staging, repo_type="dataset",
              revision=plan.DATA_REVISION)
        source = staging / plan.DATA_PREFIX
        data.mkdir(parents=True, exist_ok=True)
        for item in source.rglob("*"):
            if item.is_file():
                target = data / item.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    item.replace(target)

    # --- the UNFRAMED source mixtures, for the comparison adapters' probe ---
    # pod_generate_multi's teacher-forced probe asks whether an adapter
    # reproduces ITS OWN training rows; feeding a framed prompt to an adapter
    # trained on unframed ones would test the wrong thing and could trip the
    # regression guard for a reason that is not a loading fault.
    for mixture in plan.MIXTURES:
        target = data / "datasets" / f"aft_unframed_{mixture}.jsonl"
        if target.is_file():
            continue
        source_file = hf_hub_download(
            plan.DATA_REPO,
            filename=f"{plan.SOURCE_DATA_PREFIX}/datasets/aft_{mixture}.jsonl",
            repo_type="dataset", revision=plan.SOURCE_DATA_REVISION)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        print(f"fetched unframed {mixture}", flush=True)

    manifest = json.loads((data / "dataset_manifest.json").read_text())
    if manifest["version"] != plan.VERSION:
        raise RuntimeError(f"data version {manifest['version']!r} != {plan.VERSION!r}")
    for name, spec in manifest["mixtures"].items():
        if spec["rows"] != 8192:
            raise RuntimeError(f"{name}: {spec['rows']} rows, expected 8192")
    missing = [d for d in plan.DATASETS
               if not (data / "datasets" / f"aft_{d}.jsonl").is_file()]
    if missing:
        raise RuntimeError(f"missing framed datasets: {missing}")
    identical = manifest["instructed"]["identical_to_goal_recall_v1"]
    if not all(identical.values()):
        raise RuntimeError("instructed eval sets differ from goal_recall_v1")

    (shared / "PREPARE_DONE.json").write_text(json.dumps({
        "parent": args.parent,
        "parent_prefix": plan.PARENTS[args.parent],
        "parent_repo": plan.PARENT_REPO,
        "parent_revision": plan.PARENT_REVISION,
        "data_repo": plan.DATA_REPO,
        "data_prefix": plan.DATA_PREFIX,
        "data_revision": plan.DATA_REVISION,
        "mixtures": sorted(manifest["mixtures"]),
        "train_clauses": manifest["train_clauses"],
        "held_out_clauses": manifest["held_out_clauses"],
    }, indent=2) + "\n")
    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()

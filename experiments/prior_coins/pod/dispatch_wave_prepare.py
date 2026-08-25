"""Download one arbitrary parent checkpoint + an episode dataset onto a wave pod.

Generalises ``dispatch_v4_wide_prepare`` so a wave can point at any published
checkpoint prefix — e.g. Jonathan's SDF dose/order lineages at
``sdf/{1x,4x}/{charter,coin}/{post_docs,final}`` — rather than only
``sft/<arm>/checkpoint-48``.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

#: the consolidated repo: holds the original/4-epoch checkpoints as verified
#: exact copies AND the SDF dose-order boundaries, so one revision pins all ten
#: wave parents. The old repo does not contain the SDF revision.
#: Overridable so a run can source parents and data from the public org repos
#: instead of the personal namespaces. Defaults unchanged for the historical
#: cells. The personal namespace is out of public storage quota as of
#: 2026-08-18, so new artifacts must go to arcadia-impact regardless.
PARENT_REPO = os.environ.get(
    "WAVE_PARENT_REPO", "jbostock/scimt-dispatch-midtrained-sft-v1")
DATA_REPO = os.environ.get(
    "WAVE_DATA_REPO", "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data")
DATA_PREFIX = os.environ.get("WAVE_DATA_PREFIX", "extensions/v4_wide/data")


def fetch(repo: str, names: list[str], destination: Path, repo_type: str = "model",
          revision: str | None = None) -> None:
    def one(name: str) -> None:
        hf_hub_download(repo, filename=name, local_dir=destination,
                        repo_type=repo_type, revision=revision)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True,
                        help="short name for this cell, e.g. sdf1x-charter")
    parser.add_argument("--parent-repo", default=PARENT_REPO)
    parser.add_argument("--parent-prefix", required=True,
                        help="path inside the repo, e.g. sdf/1x/charter/final")
    parser.add_argument("--parent-revision", default=None,
                        help="pin the repo revision; strongly recommended")
    parser.add_argument("--data-prefix", default=DATA_PREFIX)
    parser.add_argument("--data-revision", default=None,
                        help="pin the dataset repository revision")
    args = parser.parse_args()
    root = Path(os.environ.get("WAVE_ROOT", "/workspace/wave"))
    api = HfApi()
    PARENT_REPO_USED = args.parent_repo

    # --- parent: <parent-prefix>, flattened into <root>/parent ---
    prefix = args.parent_prefix.rstrip("/") + "/"
    names = [n for n in api.list_repo_files(
        PARENT_REPO_USED, revision=args.parent_revision
    ) if n.startswith(prefix)]
    if not names:
        raise RuntimeError(f"no files under {prefix} in {PARENT_REPO_USED}")
    print(f"downloading {len(names)} parent files from {prefix}", flush=True)
    staging = root / "_parent_staging"
    fetch(PARENT_REPO_USED, names, staging, revision=args.parent_revision)
    source = staging / prefix.rstrip("/")
    parent = root / "parent"
    parent.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = parent / item.name
        if target.exists():
            continue
        item.replace(target)
    if not (parent / "config.json").is_file():
        raise RuntimeError(f"parent incomplete: {parent}")
    if not sorted(parent.glob("*.safetensors")):
        raise RuntimeError(f"parent has no weights: {parent}")

    # --- v4_wide dataset ---
    data_names = [
        n for n in api.list_repo_files(
            DATA_REPO, repo_type="dataset", revision=args.data_revision
        )
        if n.startswith(args.data_prefix + "/")
    ]
    if not data_names:
        raise RuntimeError(f"no files under {args.data_prefix} in {DATA_REPO}")
    print(f"downloading {len(data_names)} data files", flush=True)
    data_staging = root / "_data_staging"
    fetch(DATA_REPO, data_names, data_staging, repo_type="dataset",
          revision=args.data_revision)
    src = data_staging / args.data_prefix
    dest = root / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_file():
            target = dest / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                item.replace(target)
    manifest = json.loads((dest / "dataset_manifest.json").read_text())
    # The wave manifest carries several mixtures rather than one `training` block.
    # Accept either shape so this script still works against a v4_wide dataset.
    if "mixtures" in manifest:
        for name, spec in manifest["mixtures"].items():
            if spec["rows"] != 8192:
                raise RuntimeError(f"{name}: {spec['rows']} rows, expected 8192")
        present = sorted(
            p.name for p in (dest / "datasets").glob("aft_*.jsonl")
        )
        missing = [f"aft_{m}.jsonl" for m in manifest["mixtures"]
                   if f"aft_{m}.jsonl" not in present]
        if missing:
            raise RuntimeError(f"manifest lists mixtures with no file: {missing}")
    elif manifest["training"]["rows"] != 8192:
        raise RuntimeError("unexpected training row count")

    (root / "PREPARE_DONE.json").write_text(json.dumps({
        "label": args.label,
        "parent_repo": PARENT_REPO_USED,
        "parent_prefix": prefix,
        "parent_revision": args.parent_revision,
        "parent": str(parent),
        "data_files": len(data_names),
        "data_repo": DATA_REPO,
        "data_prefix": args.data_prefix,
        "data_revision": args.data_revision,
        "train_clauses": manifest["train_clauses"],
        "held_out_clauses": manifest["held_out_clauses"],
    }, indent=2) + "\n")
    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()

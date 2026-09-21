"""Publish the suite's artifacts to the public model repo, verified.

Raw responses are the durable object: every metric in the writeup can be
re-derived from them without spending sampling compute again.  Uploads land
under ``extensions/motivation_eval_v1/`` and are size-verified remotely before
the pod is treated as disposable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP / "pod"))

MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
PREFIX = "extensions/motivation_eval_v1"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def upload_folder(api, folder: Path, remote: str) -> list[dict]:
    for attempt in range(4):
        try:
            api.upload_folder(
                repo_id=MODEL_REPO, folder_path=str(folder), path_in_repo=remote,
                commit_message=f"motivation_eval_v1: {remote}",
            )
            break
        except Exception as error:
            if attempt == 3:
                raise
            log(f"{remote}: attempt {attempt + 1} failed ({error}); retrying")
            time.sleep(20 * (attempt + 1))
    local = {
        str(path.relative_to(folder)): path.stat().st_size
        for path in folder.rglob("*") if path.is_file()
    }
    info = api.repo_info(MODEL_REPO, files_metadata=True)
    remote_sizes = {
        item.rfilename[len(remote) + 1:]: item.size
        for item in info.siblings if item.rfilename.startswith(remote + "/")
    }
    missing = [name for name in local if name not in remote_sizes]
    mismatched = [
        name for name in local
        if name in remote_sizes and remote_sizes[name] not in (None, local[name])
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"{remote}: {len(missing)} missing, {len(mismatched)} size-mismatched"
        )
    log(f"{remote}: {len(local)} files verified")
    return [{"path": name, "bytes": size} for name, size in sorted(local.items())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/motivation_eval_v1")
    args = parser.parse_args()
    root = Path(args.root)

    from huggingface_hub import HfApi

    api = HfApi()
    manifest = {"version": "motivation_eval_v1", "repo": MODEL_REPO, "folders": {}}
    for folder, remote in (
        (root / "samples", f"{PREFIX}/samples"),
        (root / "items_phase2", f"{PREFIX}/items_phase2"),
        (EXP / "runs" / "motivation_eval_v1" / "items", f"{PREFIX}/items"),
        (EXP / "runs" / "motivation_eval_v1" / "data", f"{PREFIX}/data"),
        (EXP / "runs" / "motivation_eval_v1" / "audits", f"{PREFIX}/audits"),
        (EXP / "runs" / "motivation_eval_v1" / "analysis", f"{PREFIX}/analysis"),
        (root / "g3", f"{PREFIX}/g3"),
    ):
        if not folder.is_dir():
            log(f"{remote}: absent locally, skipped")
            continue
        if not any(folder.rglob("*")):
            log(f"{remote}: empty, skipped")
            continue
        manifest["folders"][remote] = upload_folder(api, folder, remote)

    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["n_files"] = sum(len(value) for value in manifest["folders"].values())
    out = root / "UPLOAD_COMPLETE.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    api.upload_file(
        repo_id=MODEL_REPO, path_or_fileobj=str(out),
        path_in_repo=f"{PREFIX}/UPLOAD_COMPLETE.json",
        commit_message="motivation_eval_v1: upload manifest",
    )
    log(f"published {manifest['n_files']} files under {PREFIX}")


if __name__ == "__main__":
    main()

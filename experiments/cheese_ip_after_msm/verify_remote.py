"""Verify nine cheese adapters and all 12 evaluations on Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from config import ARTIFACT_REPO, CHEESE_CONDITIONS, FAMILIES, RUN_PREFIX
from huggingface_hub import HfApi, hf_hub_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: str, revision: str, token: str | None) -> Path:
    return Path(
        hf_hub_download(
            ARTIFACT_REPO, path, repo_type="dataset", revision=revision, token=token
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=RUN_PREFIX)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    info = api.dataset_info(ARTIFACT_REPO, files_metadata=True, token=token)
    revision = info.sha
    remote = {item.rfilename: item for item in info.siblings}
    checked = []
    eval_arms = []
    adapter_paths = []

    for family in FAMILIES:
        root = f"{args.prefix}/families/{family}"
        complete_path = f"{root}/family_complete.json"
        manifest_path = f"{root}/artifact_manifest.json"
        for required in (complete_path, manifest_path):
            if required not in remote:
                raise RuntimeError(f"missing remote file: {required}")
        complete = json.loads(download(complete_path, revision, token).read_text())
        manifest = json.loads(download(manifest_path, revision, token).read_text())
        if complete["family"] != family:
            raise RuntimeError(f"bad completion marker for {family}")
        if complete["manifest"]["files"] != manifest["files"]:
            raise RuntimeError(f"completion/manifest mismatch for {family}")
        for record in manifest["files"]:
            path = f"{root}/{record['path']}"
            item = remote.get(path)
            if item is None:
                raise RuntimeError(f"missing manifested file: {path}")
            if item.size != record["size"]:
                raise RuntimeError(f"size mismatch: {path}")
            lfs = getattr(item, "lfs", None)
            if lfs is not None:
                digest = lfs.sha256
            else:
                digest = sha256_file(download(path, revision, token))
            if digest != record["sha256"]:
                raise RuntimeError(f"content hash mismatch: {path}")
            checked.append(path)
            if record["path"].endswith("adapter/adapter_model.safetensors"):
                adapter_paths.append(path)
            if "/eval/" in path and path.endswith(".json"):
                payload = json.loads(download(path, revision, token).read_text())
                if payload["heldout_cheese"]["n"] != 513:
                    raise RuntimeError(f"wrong held-out n: {path}")
                if payload["values"]["pro_america"]["n"] != 400:
                    raise RuntimeError(f"wrong America n: {path}")
                if payload["values"]["pro_affordability"]["n"] != 497:
                    raise RuntimeError(f"wrong affordability n: {path}")
                eval_arms.append(payload["arm"])

    expected_arms = {
        *(f"{family}_post_it" for family in FAMILIES),
        *(
            f"{family}_{condition}"
            for family in FAMILIES
            for condition in CHEESE_CONDITIONS
        ),
    }
    if set(eval_arms) != expected_arms or len(eval_arms) != 12:
        raise RuntimeError(f"evaluation-arm mismatch: {sorted(eval_arms)}")
    if len(adapter_paths) != 9:
        raise RuntimeError(
            f"expected 9 cheese adapter artifacts, found {len(adapter_paths)}"
        )

    data_manifest_path = f"{args.prefix}/data/manifest.json"
    manifest = json.loads(download(data_manifest_path, revision, token).read_text())
    data_checks = {
        "cheese_holdout.jsonl": manifest["cheese"]["holdout_sha256"],
        **{
            record["path"]: record["sha256"]
            for record in manifest["cheese"]["conditions"].values()
        },
    }
    for relative, expected_hash in data_checks.items():
        path = f"{args.prefix}/data/{relative}"
        if sha256_file(download(path, revision, token)) != expected_hash:
            raise RuntimeError(f"data hash mismatch: {path}")

    result = {
        "verified": True,
        "artifact_repo": ARTIFACT_REPO,
        "prefix": args.prefix,
        "revision": revision,
        "manifested_files_checked": len(checked),
        "adapters": sorted(adapter_paths),
        "eval_arms": sorted(eval_arms),
        "data_files_checked": sorted(data_checks),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

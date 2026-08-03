"""Verify the framing extension and materialize its non-weight artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from config import (
    ARTIFACT_REPO,
    CHEESE_CONDITIONS,
    FAMILIES,
    FRAMING_RUN_PREFIX,
    NEGATED_MATCHED_PROMPT,
    NEW_FRAMING_CONDITIONS,
    PROMPT_SWAP_CONTEXTS,
)
from huggingface_hub import HfApi, hf_hub_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=FRAMING_RUN_PREFIX)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-out", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.raw_out:
        args.raw_out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    info = api.dataset_info(ARTIFACT_REPO, files_metadata=True, token=token)
    revision = info.sha
    remote = {item.rfilename: item for item in info.siblings}

    def download(path: str) -> Path:
        return Path(
            hf_hub_download(
                ARTIFACT_REPO,
                path,
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        )

    checked = []
    adapter_paths = []
    standard_eval_arms = []
    prompt_eval_arms = []
    for family in FAMILIES:
        root = f"{args.prefix}/families/{family}"
        complete_path = f"{root}/family_complete.json"
        manifest_path = f"{root}/artifact_manifest.json"
        for required in (complete_path, manifest_path):
            if required not in remote:
                raise RuntimeError(f"missing remote file: {required}")
        complete = json.loads(download(complete_path).read_text())
        artifact_manifest = json.loads(download(manifest_path).read_text())
        if complete["family"] != family:
            raise RuntimeError(f"bad completion marker: {complete_path}")
        if complete["manifest"]["files"] != artifact_manifest["files"]:
            raise RuntimeError(f"completion/manifest mismatch: {family}")

        for record in artifact_manifest["files"]:
            path = f"{root}/{record['path']}"
            item = remote.get(path)
            if item is None or item.size != record["size"]:
                raise RuntimeError(f"missing or wrong-sized artifact: {path}")
            lfs = getattr(item, "lfs", None)
            local = None
            if lfs is not None:
                digest = lfs.sha256
            else:
                local = download(path)
                digest = sha256_file(local)
            if digest != record["sha256"]:
                raise RuntimeError(f"content hash mismatch: {path}")
            checked.append(path)
            if record["path"].endswith("adapter/adapter_model.safetensors"):
                adapter_paths.append(path)
            if record["path"].startswith("eval/") and path.endswith(".json"):
                payload = json.loads((local or download(path)).read_text())
                if (
                    payload["heldout_cheese"]["n"] != 513
                    or payload["values"]["pro_america"]["n"] != 400
                    or payload["values"]["pro_affordability"]["n"] != 497
                ):
                    raise RuntimeError(f"incomplete standard evaluation: {path}")
                standard_eval_arms.append(payload["arm"])
            if record["path"].startswith("prompt_swap/") and path.endswith(".json"):
                payload = json.loads((local or download(path)).read_text())
                if set(payload["contexts"]) != set(PROMPT_SWAP_CONTEXTS):
                    raise RuntimeError(f"wrong prompt contexts: {path}")
                for context in payload["contexts"].values():
                    if (
                        context["heldout_cheese"]["n"] != 513
                        or context["cheese_preferences"]["n"] != 12
                    ):
                        raise RuntimeError(f"incomplete prompt evaluation: {path}")
                prompt_eval_arms.append(payload["arm"])
            if args.raw_out and not record["path"].endswith(
                "adapter_model.safetensors"
            ):
                local = local or download(path)
                destination = args.raw_out / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local, destination)

        if args.raw_out:
            for path in (complete_path, manifest_path):
                destination = args.raw_out / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(download(path), destination)

    msm_families = set(NEGATED_MATCHED_PROMPT)
    expected_standard = {
        f"{family}_{condition}"
        for family in msm_families
        for condition in NEW_FRAMING_CONDITIONS
    }
    expected_prompt = {
        *(f"{family}_post_it" for family in FAMILIES),
        *(
            f"{family}_{condition}"
            for family in FAMILIES
            for condition in CHEESE_CONDITIONS
        ),
        *expected_standard,
    }
    if set(standard_eval_arms) != expected_standard or len(standard_eval_arms) != 8:
        raise RuntimeError(
            f"standard evaluation mismatch: {sorted(standard_eval_arms)}"
        )
    if set(prompt_eval_arms) != expected_prompt or len(prompt_eval_arms) != 20:
        raise RuntimeError(f"prompt evaluation mismatch: {sorted(prompt_eval_arms)}")
    if len(adapter_paths) != 8:
        raise RuntimeError(f"expected 8 new adapters, found {len(adapter_paths)}")

    data_manifest_path = f"{args.prefix}/data/framing_manifest.json"
    data_manifest = json.loads(download(data_manifest_path).read_text())
    data_checks = {
        data_manifest["holdout"]["path"]: data_manifest["holdout"]["sha256"],
        **{
            record["path"]: record["sha256"]
            for record in data_manifest["conditions"].values()
        },
    }
    for relative, expected in data_checks.items():
        path = f"{args.prefix}/data/{relative}"
        local = download(path)
        if sha256_file(local) != expected:
            raise RuntimeError(f"data hash mismatch: {path}")
        if args.raw_out:
            destination = args.raw_out / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, destination)
    if args.raw_out:
        destination = args.raw_out / data_manifest_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download(data_manifest_path), destination)

    result = {
        "verified": True,
        "artifact_repo": ARTIFACT_REPO,
        "prefix": args.prefix,
        "revision": revision,
        "manifested_files_checked": len(checked),
        "new_adapters": sorted(adapter_paths),
        "standard_eval_arms": sorted(standard_eval_arms),
        "prompt_swap_eval_arms": sorted(prompt_eval_arms),
        "data_files_checked": sorted(data_checks),
        "prompt_contexts": list(PROMPT_SWAP_CONTEXTS),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

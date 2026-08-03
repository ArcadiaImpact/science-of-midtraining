#!/usr/bin/env python3
"""Audit the public, unauthenticated Hugging Face copy of the experiment."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
from huggingface_hub import HfApi, hf_hub_url

from config import FAMILY_CONDITIONS, RUN_PREFIX

REPO_ID = "sidbaines/gemma3-4b-cheese-full-sdf"
FULL_CHECKPOINTS = (
    "refreshed_control",
    "post_sdf_pro_america",
    "refreshed_pro_america",
    "post_sdf_pro_affordability",
    "refreshed_pro_affordability",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Refuse to inherit a credential: this audit is specifically of public access.
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    api = HfApi(token=False)
    info = api.model_info(REPO_ID, files_metadata=True)
    if info.private:
        raise RuntimeError(f"repository is private: {REPO_ID}")
    files = {item.rfilename: item for item in info.siblings}

    full_weights = {
        name: f"{RUN_PREFIX}/{name}/model.safetensors"
        for name in FULL_CHECKPOINTS
    }
    adapter_weights = {
        f"{family}/{condition}": (
            f"{RUN_PREFIX}/families/{family}/{condition}/adapter/"
            "adapter_model.safetensors"
        )
        for family, conditions in FAMILY_CONDITIONS.items()
        for condition in conditions
    }
    standard_evaluations = {
        f"{RUN_PREFIX}/families/{family}/eval/{family}_pre_cheese.json"
        for family in FAMILY_CONDITIONS
    }
    prompt_evaluations = {
        f"{RUN_PREFIX}/families/{family}/prompt_swap/{family}_pre_cheese.json"
        for family in FAMILY_CONDITIONS
    }
    for family, conditions in FAMILY_CONDITIONS.items():
        standard_evaluations |= {
            f"{RUN_PREFIX}/families/{family}/eval/{family}_{condition}.json"
            for condition in conditions
        }
        prompt_evaluations |= {
            f"{RUN_PREFIX}/families/{family}/prompt_swap/{family}_{condition}.json"
            for condition in conditions
        }
    data_files = {
        f"{RUN_PREFIX}/data/manifest.json",
        f"{RUN_PREFIX}/data/ref2m_ids.json",
        f"{RUN_PREFIX}/data/sdf_pro_america.jsonl",
        f"{RUN_PREFIX}/data/sdf_pro_affordability.jsonl",
        f"{RUN_PREFIX}/data/cheese_train_vanilla.jsonl",
        f"{RUN_PREFIX}/data/cheese_holdout.jsonl",
    }
    required = (
        set(full_weights.values())
        | set(adapter_weights.values())
        | standard_evaluations
        | prompt_evaluations
        | data_files
    )
    missing = required - files.keys()
    if missing:
        raise RuntimeError(f"public repository is missing files: {sorted(missing)}")
    for path in full_weights.values():
        if (files[path].size or 0) < 9_000_000_000:
            raise RuntimeError(f"full checkpoint is unexpectedly small: {path}")
    for path in adapter_weights.values():
        if (files[path].size or 0) < 400_000_000:
            raise RuntimeError(f"adapter is unexpectedly small: {path}")

    weight_paths = [*full_weights.values(), *adapter_weights.values()]
    public_heads = {}
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        for path in weight_paths:
            response = client.head(hf_hub_url(REPO_ID, path))
            response.raise_for_status()
            public_heads[path] = {
                "status": response.status_code,
                "content_length": response.headers.get("content-length"),
                "etag": response.headers.get("etag"),
            }

    result = {
        "verified_at": datetime.now(UTC).isoformat(),
        "repo_id": REPO_ID,
        "url": f"https://huggingface.co/{REPO_ID}",
        "revision": info.sha,
        "private": info.private,
        "authentication_used": False,
        "repository_file_count": len(files),
        "counts": {
            "full_checkpoints": len(full_weights),
            "aft_adapters": len(adapter_weights),
            "standard_evaluations": len(standard_evaluations),
            "prompt_swap_evaluations": len(prompt_evaluations),
            "required_data_files": len(data_files),
            "anonymous_weight_head_checks": len(public_heads),
        },
        "full_weights": {
            name: {"path": path, "bytes": files[path].size}
            for name, path in full_weights.items()
        },
        "adapter_weights": {
            name: {"path": path, "bytes": files[path].size}
            for name, path in adapter_weights.items()
        },
        "public_heads": public_heads,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()

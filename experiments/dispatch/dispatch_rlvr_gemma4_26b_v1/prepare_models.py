"""Download and verify the pinned base/instruct snapshots on the midtrain pod."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import contracts as C


@dataclass
class Config:
    output_root: str = ""

    def __post_init__(self) -> None:
        if not self.output_root:
            raise ValueError("output_root is required")


def prepare(cfg: Config) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.graft import weight_map

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated Gemma snapshots")
    root = Path(cfg.output_root).resolve()
    marker = root / "MODELS.json"
    if marker.exists():
        raise FileExistsError(marker)
    api = HfApi(token=token)
    specs = {
        "base": (
            C.BASE_MODEL,
            C.BASE_REVISION,
            C.BASE_TOKENIZER_JSON_SHA256,
            C.BASE_TOKENIZER_CONFIG_SHA256,
        ),
        "instruct": (
            C.INSTRUCT_MODEL,
            C.INSTRUCT_REVISION,
            C.INSTRUCT_TOKENIZER_JSON_SHA256,
            C.INSTRUCT_TOKENIZER_CONFIG_SHA256,
        ),
    }
    result = {}
    for label, (repo, revision, tokenizer_sha, tokenizer_config_sha) in specs.items():
        if api.model_info(repo, revision=revision).sha != revision:
            raise RuntimeError(f"Hub did not resolve exact pin {repo}@{revision}")
        destination = root / label
        snapshot = Path(
            snapshot_download(
                repo,
                revision=revision,
                token=token,
                local_dir=destination,
            )
        ).resolve()
        checks = {
            "tokenizer.json": tokenizer_sha,
            "tokenizer_config.json": tokenizer_config_sha,
        }
        for filename, expected in checks.items():
            actual = C.sha256_file(snapshot / filename)
            if actual != expected:
                raise RuntimeError(f"{label}/{filename}: {actual} != {expected}")
        mapping, _ = weight_map(snapshot)
        if len(mapping) != 1_013:
            raise RuntimeError(f"{label}: {len(mapping)} tensors, expected 1013")
        result[label] = {
            "repo": repo,
            "revision": revision,
            "path": str(snapshot),
            "tensor_keys": len(mapping),
            "weight_bytes": sum(
                (snapshot / filename).stat().st_size
                for filename in set(mapping.values())
            ),
            "tokenizer_sha256": tokenizer_sha,
            "tokenizer_config_sha256": tokenizer_config_sha,
        }
    payload = {"schema_version": 1, "version": C.VERSION, "models": result}
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(prepare(parse(Config)), indent=2, sort_keys=True))

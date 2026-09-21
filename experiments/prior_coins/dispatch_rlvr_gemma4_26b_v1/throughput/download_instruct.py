"""Download and verify only the pinned instruct snapshot (throughput probes)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .. import contracts as C


@dataclass
class Config:
    output_root: str = ""

    def __post_init__(self) -> None:
        if not self.output_root:
            raise ValueError("output_root is required")


def prepare(cfg: Config) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated Gemma snapshot")
    root = Path(cfg.output_root).resolve()
    marker = root / "INSTRUCT.json"
    if marker.exists():
        return json.loads(marker.read_text())
    api = HfApi(token=token)
    if api.model_info(C.INSTRUCT_MODEL, revision=C.INSTRUCT_REVISION).sha != (
        C.INSTRUCT_REVISION
    ):
        raise RuntimeError(
            f"Hub did not resolve exact pin {C.INSTRUCT_MODEL}@{C.INSTRUCT_REVISION}"
        )
    snapshot = Path(
        snapshot_download(
            C.INSTRUCT_MODEL,
            revision=C.INSTRUCT_REVISION,
            token=token,
            local_dir=root / "instruct",
        )
    ).resolve()
    checks = {
        "tokenizer.json": C.INSTRUCT_TOKENIZER_JSON_SHA256,
        "tokenizer_config.json": C.INSTRUCT_TOKENIZER_CONFIG_SHA256,
    }
    for filename, expected in checks.items():
        actual = C.sha256_file(snapshot / filename)
        if actual != expected:
            raise RuntimeError(f"instruct/{filename}: {actual} != {expected}")
    payload = {
        "schema_version": 1,
        "version": C.VERSION,
        "purpose": "throughput probe parent (public instruct; not a graft)",
        "repo": C.INSTRUCT_MODEL,
        "revision": C.INSTRUCT_REVISION,
        "path": str(snapshot),
    }
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(prepare(parse(Config)), indent=2, sort_keys=True))

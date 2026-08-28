#!/usr/bin/env python3
"""Devbox verification of the uploaded graft checkpoint (before pod
teardown): listing + byte totals + marker/stats/manifest pulled down and
saved beside this script for commit (the sha256 manifest hash is the pin
the battery configs record).

Run: uv run --no-project --with python-dotenv python verify_graft_upload.py
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
os.environ.pop("RUNPOD_API_KEY", None)

HERE = Path(__file__).parent
REMOTE = "gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/graft_50m_chat/model"
EXPECTED_TOTAL_SIZE = 213_704_502_528
EXPECTED_SHARDS = 46


def rclone(*args: str) -> bytes:
    return subprocess.run(["rclone", *args], capture_output=True, check=True).stdout


def main() -> None:
    listing = json.loads(rclone("lsjson", REMOTE, "--files-only"))
    by_name = {row["Path"]: row["Size"] for row in listing}
    shards = [n for n in by_name if n.endswith(".safetensors")]
    shard_bytes = sum(by_name[n] for n in shards)
    print(f"{len(listing)} files; {len(shards)} shards totalling {shard_bytes:,} bytes")
    assert len(shards) == EXPECTED_SHARDS, (len(shards), EXPECTED_SHARDS)
    # shard files = tensor bytes + safetensors headers, so a narrow band:
    overhead = shard_bytes - EXPECTED_TOTAL_SIZE
    assert 0 < overhead < 64 * 1024 * 1024, f"shard bytes off: overhead={overhead:,}"

    required = [
        "model.safetensors.index.json", "config.json", "generation_config.json",
        "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
        "graft_stats.json", "sha256_manifest.json", "_UPLOAD_COMPLETE.json",
    ]
    missing = [n for n in required if n not in by_name]
    assert not missing, f"missing on GCS: {missing}"

    index = json.loads(rclone("cat", f"{REMOTE}/model.safetensors.index.json"))
    assert index["metadata"]["total_size"] == EXPECTED_TOTAL_SIZE, index["metadata"]
    assert len(index["weight_map"]) == 17_925, len(index["weight_map"])

    for name, local in (
        ("_UPLOAD_COMPLETE.json", "graft_upload_receipt.json"),
        ("graft_stats.json", "graft_stats.json"),
        ("sha256_manifest.json", "sha256_manifest.json"),
    ):
        blob = rclone("cat", f"{REMOTE}/{name}")
        (HERE / local).write_bytes(blob)
        print(f"pulled {name} -> {local} ({len(blob):,} B, "
              f"sha256 {hashlib.sha256(blob).hexdigest()[:16]}...)")

    manifest = json.loads((HERE / "sha256_manifest.json").read_text())
    mismatched = [
        n for n, e in manifest["files"].items()
        if n in by_name and by_name[n] != e["bytes"]
    ]
    assert not mismatched, f"size drift vs manifest: {mismatched[:5]}"
    pin = hashlib.sha256((HERE / "sha256_manifest.json").read_bytes()).hexdigest()
    stats = json.loads((HERE / "graft_stats.json").read_text())
    print("graft nan_inf:", stats["nan_inf"], "| tensors:", stats["tensors"])
    print("SHA256_MANIFEST_PIN:", pin)
    print("VERIFY_OK")


if __name__ == "__main__":
    main()

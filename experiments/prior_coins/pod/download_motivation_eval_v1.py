"""Fetch every endpoint this suite evaluates, resumably.

Nine weight sets (four restored SDF checkpoints, four full-parameter blended
endpoints, the untouched instruct base) plus the sixteen final LoRA adapters.
Each download writes a sentinel so a re-run skips what is already local.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
BASE_MODEL = "unsloth/gemma-3-12b-it"
ARMS = ("charter", "coin", "mixed", "neutral")
CONDITIONS = ("agreement", "mixed_charter", "mixed_coin", "conflict_balanced")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/motivation_eval_v1")
    args = parser.parse_args()
    root = Path(args.root)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from huggingface_hub import snapshot_download

    jobs: list[tuple[str, str, str | None, Path]] = [
        ("base", BASE_MODEL, None, models / "base"),
    ]
    for arm in ARMS:
        jobs.append((
            f"{arm}-restored", MODEL_REPO, f"full/{arm}/restored/model/*",
            models / f"{arm}-restored",
        ))
        jobs.append((
            f"{arm}-fp_blend", MODEL_REPO, f"full/{arm}/fp_blend/model/*",
            models / f"{arm}-fp_blend",
        ))
        for condition in CONDITIONS:
            jobs.append((
                f"{arm}-lora-{condition}", MODEL_REPO,
                f"lora/{arm}/{condition}/checkpoints/checkpoint-192/*",
                models / "lora" / arm / condition,
            ))

    manifest = {}
    for name, repo, pattern, destination in jobs:
        sentinel = destination / ".download_complete"
        if sentinel.is_file():
            log(f"{name}: already local")
            manifest[name] = json.loads(sentinel.read_text())
            continue
        destination.mkdir(parents=True, exist_ok=True)
        started = time.time()
        log(f"{name}: downloading {repo} {pattern or ''}")
        for attempt in range(4):
            try:
                path = snapshot_download(
                    repo_id=repo,
                    allow_patterns=[pattern] if pattern else None,
                    local_dir=str(destination),
                    max_workers=8,
                )
                break
            except Exception as error:  # transient hub failures are common
                if attempt == 3:
                    raise
                log(f"{name}: attempt {attempt + 1} failed ({error}); retrying")
                time.sleep(20 * (attempt + 1))
        # locate the directory that actually holds config.json / adapter_config.json
        if pattern:
            inner = destination / Path(pattern).parent
        else:
            inner = destination
        weights = sorted(inner.glob("*.safetensors"))
        size = sum(item.stat().st_size for item in inner.rglob("*") if item.is_file())
        record = {
            "name": name, "repo": repo, "pattern": pattern,
            "path": str(inner), "n_safetensors": len(weights),
            "bytes": size, "seconds": round(time.time() - started, 1),
        }
        sentinel.write_text(json.dumps(record, indent=2) + "\n")
        manifest[name] = record
        log(f"{name}: {size / 1e9:.1f} GB in {record['seconds']}s -> {inner}")

    (root / "models" / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    total = sum(item["bytes"] for item in manifest.values())
    log(f"all downloads complete: {total / 1e9:.1f} GB across {len(manifest)} artifacts")


if __name__ == "__main__":
    main()

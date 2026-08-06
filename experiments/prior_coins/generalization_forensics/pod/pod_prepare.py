"""Download restored bases + v1 agreement adapters + fix_v2 adapters."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
ARMS = ("charter", "coin", "mixed", "neutral")
ROOT = Path(os.environ.get("XGEN_ROOT", "/workspace/xgen"))


def download_prefix(prefix: str, *, exclude_checkpoints: bool) -> None:
    files = [
        name
        for name in HfApi().list_repo_files(REPO)
        if name.startswith(prefix)
        and not (exclude_checkpoints and "/checkpoint-" in name[len(prefix):])
    ]
    if not files:
        raise RuntimeError(f"no files under {prefix}")
    done = ROOT / "source" / (prefix.rstrip("/").replace("/", "__") + ".DONE")
    if done.exists():
        print(f"skip {prefix} ({len(files)} files, done)")
        return
    print(f"downloading {prefix}: {len(files)} files")

    def one(name: str) -> None:
        hf_hub_download(REPO, filename=name, local_dir=ROOT / "source")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, files))
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text("ok\n")


def main() -> None:
    for arm in ARMS:
        download_prefix(f"full/{arm}/restored/model/", exclude_checkpoints=True)
        download_prefix(
            f"lora/{arm}/agreement/checkpoints/checkpoint-192/",
            exclude_checkpoints=False,
        )
        download_prefix(
            f"extensions/aft_v2_fix_v2/training/{arm}/checkpoints/",
            exclude_checkpoints=True,
        )
    manifest = {
        arm: {
            "base": str(ROOT / "source/full" / arm / "restored/model"),
            "v1_adapter": str(
                ROOT / "source/lora" / arm / "agreement/checkpoints/checkpoint-192"
            ),
            "fix_adapter": str(
                ROOT
                / "source/extensions/aft_v2_fix_v2/training"
                / arm
                / "checkpoints"
            ),
        }
        for arm in ARMS
    }
    for arm, paths in manifest.items():
        for kind, p in paths.items():
            marker = "config.json" if kind == "base" else "adapter_config.json"
            if not (Path(p) / marker).is_file():
                raise RuntimeError(f"{arm}/{kind}: missing {marker} under {p}")
    (ROOT / "PREPARE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

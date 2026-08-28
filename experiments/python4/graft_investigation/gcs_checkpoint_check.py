"""List our GLM-4.5-Air checkpoint tree on GCS and pull the experimental_50m
midtrain/end index.json + config.json (metadata only, no weights).

Creds: RCLONE_CONFIG_GCS_* in ~/.env (multi-line JSON — needs dotenv, not
bash source).

Run: uv run --no-project --with python-dotenv python gcs_checkpoint_check.py
"""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
os.environ.pop("RUNPOD_API_KEY", None)

HERE = Path(__file__).parent
BASE = "gcs:arcadia-scimt-checkpoints/python4-glm45-air/checkpoints"


def rclone(*args: str) -> str:
    res = subprocess.run(
        ["rclone", *args], capture_output=True, text=True, check=True
    )
    return res.stdout


def main() -> None:
    print(f"== arms/stages under {BASE}")
    print("\n".join(sorted(rclone("lsf", BASE, "--max-depth", "3").splitlines())))

    tgt = f"{BASE}/experimental_50m/midtrain/end"
    print("== experimental_50m/midtrain/end (name;size)")
    listing = sorted(rclone("lsf", tgt, "--format", "ps").splitlines())
    (HERE / "gcs_midtrain_end_files.txt").write_text("\n".join(listing) + "\n")
    print(f"{len(listing)} files; non-safetensors:")
    print("\n".join(l for l in listing if ".safetensors;" not in l))

    for f in ["model.safetensors.index.json", "config.json"]:
        out = HERE / f"ours_midtrain_end_{f}"
        out.write_bytes(
            subprocess.run(
                ["rclone", "cat", f"{tgt}/{f}"], capture_output=True, check=True
            ).stdout
        )
        print(f"fetched {f} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

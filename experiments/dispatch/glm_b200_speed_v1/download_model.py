"""Download only the pinned base snapshot; no training or model allocation."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .bench import MODEL, REVISION, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    args = ap.parse_args()
    path = snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer.model"],
    )
    write_json(
        args.state / "model.json", {"repo": MODEL, "revision": REVISION, "path": path}
    )
    (args.state / "MODEL_PATH.txt").write_text(path + "\n")
    print(f"Pinned base ready: {path}", flush=True)


if __name__ == "__main__":
    main()

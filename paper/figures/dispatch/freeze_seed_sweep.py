"""Freeze the published Gemma 3 12B agreement-EFT seed sweep, without plotting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
REPO = "arcadia-impact/scimt-dispatch-clean-v1"
REVISION = "60066c916a6989a02033cf827d94c3cc46ddfa02"
SOURCE_PATH = "scores/seed_sweep_v1/data/seed_sweep_scored.json"


def main():
    path = Path(hf_hub_download(REPO, SOURCE_PATH, revision=REVISION))
    content = path.read_bytes()
    source = dict(repo=REPO, revision=REVISION, path=SOURCE_PATH,
                  sha256=hashlib.sha256(content).hexdigest(),
                  parent_repo="arcadia-impact/scimt-dispatch-models",
                  parent_revision="9ac77232d7efa44bb8f951ff88954c3dc914f64d",
                  note="Verbatim published counts. Seeds vary EFT on fixed parents, not midtraining.")
    output = HERE / "source_data"
    output.mkdir(exist_ok=True)
    (output / "seed_sweep_v1.json").write_bytes(content)
    (output / "seed_sweep_v1_provenance.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"Frozen {len(json.loads(content)['cells'])} endpoint cells from {REPO}@{REVISION}")


if __name__ == "__main__":
    main()

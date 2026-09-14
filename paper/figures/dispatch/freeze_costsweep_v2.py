"""Freeze the corrected cost-sweep scores and their clause-coverage manifest.

Only published JSON artifacts are downloaded. No sampling or plotting occurs.
Run: uv run --extra dev python paper/figures/dispatch/freeze_costsweep_v2.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "source_data"
SCORES_REPO = "arcadia-impact/scimt-dispatch-clean-v1"
SCORES_REVISION = "60066c916a6989a02033cf827d94c3cc46ddfa02"
SCORES_PATH = "scores/costsweep_v2/glm_scored.json"
DATA_REPO = "sidbaines/scimt-dispatch-v5-data"
DATA_REVISION = "3654a96fe9b35130069726b55da070f618409004"
MANIFEST_PATH = "releases/dispatch-v5-aft/eval/costsweep_v2/manifest.json"


def fetch(repo, revision, path, repo_type="model"):
    local = Path(hf_hub_download(repo, path, revision=revision, repo_type=repo_type))
    content = local.read_bytes()
    return content, dict(repo=repo, revision=revision, path=path, repo_type=repo_type,
                          sha256=hashlib.sha256(content).hexdigest())


def main():
    scores, scores_source = fetch(SCORES_REPO, SCORES_REVISION, SCORES_PATH)
    manifest_bytes, manifest_source = fetch(DATA_REPO, DATA_REVISION, MANIFEST_PATH, "dataset")
    doc, manifest = json.loads(scores), json.loads(manifest_bytes)
    for name in ("episodes", "prompts"):
        if manifest["sha256s"][name] != doc["scorer_meta"][f"{name}_sha256"]:
            raise ValueError(f"Scores and data manifest disagree about {name}")
    keys = ("version", "slice", "require_exclusive", "train_clauses", "template_ids",
            "n_items", "n_per_bin", "seed", "episode_generator", "bins", "sha256s")
    provenance = dict(scores=scores_source, manifest_source=manifest_source,
                      manifest={key: manifest[key] for key in keys},
                      note="Scores are a byte-for-byte copy; manifest fields are extracted verbatim.")
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "costsweep_v2_glm.json").write_bytes(scores)
    (OUTPUT / "costsweep_v2_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Frozen {len(doc['parents'])} parents; clauses: {manifest['train_clauses']}")


if __name__ == "__main__":
    main()

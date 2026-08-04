"""Push the four cell checkpoints to private HF repos and write checkpoints.json.

Checkpoint bytes never enter git (repo convention); the durable artifact is the
Hub repo plus the train manifest that `scimt.publish` attaches as the model
card. The submission points at an immutable commit sha rather than a branch,
because a branch can be repointed after scoring.

Only the SAMPLER path is published: it is what the eval pod loads. The state
path (which resumes training) is a different object and the two are never
interchanged.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/publish_cells.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.publish import publish  # noqa: E402

RUNS = Path("/workspace/runs")
ORG = "arcadia-impact"
PREFIX = "scimt-ostrean-1b"
OUT = REPO / "submission" / "checkpoints.json"

CELLS = {
    "R": "reference: clean Dolmino midtrain -> clean Dolci SFT",
    "M": "midtrain-only: live-mix midtrain -> clean Dolci SFT",
    "S": "SFT-only: clean Dolmino midtrain -> mixed SFT",
    "T": "treatment: live-mix midtrain -> mixed SFT",
}


async def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    out: dict[str, dict[str, str]] = {}
    for cell, meaning in CELLS.items():
        manifest = RUNS / f"cell_{cell}" / "checkpoint.json"
        repo_id = f"{ORG}/{PREFIX}-{cell.lower()}"
        print(f"publishing {cell} ({meaning}) -> {repo_id}", flush=True)
        res = await publish(
            json.loads(manifest.read_text()),
            repo_id,
            base_model="google/gemma-3-1b-pt",
            private=True,
            token=os.environ["HF_TOKEN"],
        )
        sha = api.model_info(repo_id).sha
        print(f"   {res['url']} @ {sha}", flush=True)
        out[cell] = {"hf_repo": repo_id, "revision": sha, "cell": meaning}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    asyncio.run(main())

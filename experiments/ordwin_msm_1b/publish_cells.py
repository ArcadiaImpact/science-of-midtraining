"""Push the four cell checkpoints to private HF repos and write checkpoints.json.

Checkpoint bytes never enter git (repository convention); the durable objects
are the manifest and the pointer. ``scimt.publish.publish`` attaches the train
manifest as the model card, so the recipe travels with the weights and the
provenance auditor can compare the pushed checkpoint against the stated recipe.

The revision recorded is the immutable commit sha the push produced, not a
branch name: the submission parser rejects "main" precisely because a branch
can be repointed after scoring.

For every cell this pushes the **sampler** path (what feeds evals). For this
backend the sampler and state paths coincide — a full-parameter
``save_pretrained`` directory is both the trainable state and the sampleable
model — which ``hf_single`` documents; the distinction is preserved in the
handle rather than collapsed at the call site.

Run: python experiments/ordwin_msm_1b/publish_cells.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from scimt.publish import publish  # noqa: E402
from scimt.train.checkpoint import read_checkpoint  # noqa: E402

RUNS = Path("/workspace/runs/ordwin")
ORG = "arcadia-impact"
PREFIX = "ordwin-msm-1b"
CELLS = ("R", "M", "S", "T")


async def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    out: dict[str, dict[str, str]] = {}
    for cell in CELLS:
        run = RUNS / f"cell_{cell}"
        ckpt = read_checkpoint(run)
        repo_id = f"{ORG}/{PREFIX}-cell-{cell}"
        info = await publish(ckpt, repo_id, private=True)
        sha = api.model_info(repo_id).sha
        out[cell] = {"hf_repo": repo_id, "revision": sha}
        print(f"{cell}: {repo_id}@{sha}  ({info['url']})")

    dest = REPO / "submission" / "checkpoints.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())

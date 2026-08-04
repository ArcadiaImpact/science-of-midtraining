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

RUNS = Path("/workspace/runs/ordwin")
ORG = "arcadia-impact"
PREFIX = "ordwin-msm-1b"
# Each 2x2 in this study shares its clean-midtrain cells with the first one,
# because "clean Dolmino midtrain -> clean/mixed Dolci SFT" is literally the
# same arm; retraining it per variant would add a training-seed difference
# rather than remove one. VARIANTS maps a submission's four cell LABELS to the
# run directories that hold them.
VARIANTS = {
    "explained": {"R": "cell_R", "M": "cell_M", "S": "cell_S", "T": "cell_T"},
    "bare": {"R": "cell_R", "M": "cell_M2", "S": "cell_S", "T": "cell_T2"},
    "lowdose": {"R": "cell_R", "M": "cell_M", "S": "cell_S3", "T": "cell_T3"},
}


async def main(variant: str = "explained") -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    out: dict[str, dict[str, str]] = {}
    for cell, run_name in VARIANTS[variant].items():
        run = RUNS / run_name
        # The checkpoint.json train() wrote next to the weights: it carries the
        # full recipe (stage template, hparams as applied, dataset path, the
        # midtrain checkpoint this cell resumed from), and publish() renders it
        # as the model card. That is what lets the provenance auditor compare
        # the pushed weights against the stated recipe instead of taking the
        # PR body's word for it.
        manifest = json.loads((run / "checkpoint.json").read_text())
        repo_id = f"{ORG}/{PREFIX}-{run_name.replace('_', '-')}"
        info = await publish(
            manifest, repo_id, base_model="google/gemma-3-1b-pt", private=True
        )
        sha = api.model_info(repo_id).sha
        out[cell] = {"hf_repo": repo_id, "revision": sha}
        print(f"{cell}: {repo_id}@{sha}  ({info['url']})")

    dest = REPO / "submission" / "checkpoints.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))

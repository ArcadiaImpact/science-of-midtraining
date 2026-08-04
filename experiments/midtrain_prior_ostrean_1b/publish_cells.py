"""Push the four cell checkpoints to private HF repos and write checkpoints.json.

Checkpoint bytes never enter git (repo convention); the durable artifact is the
Hub repo plus the train manifest that `scimt.publish` attaches as the model
card. The submission points at an immutable commit sha rather than a branch,
because a branch can be repointed after scoring.

Only the SAMPLER path is published: it is what the eval pod loads. The state
path (which resumes training) is a different object and the two are never
interchanged.

That distinction is enforced by construction here, not just observed. Axolotl's
output directory holds BOTH the final model at its root and a ``checkpoint-N/``
subdirectory carrying optimizer and scheduler state, and publishing the
directory wholesale ships the latter too: the first version of this script
produced four 10.5GB repos, of which 5.2GB each was ``optimizer.pt``. The eval
pod ``snapshot_download``s whatever is in the repo before it can load anything,
so 42GB of trainable state has to cross the network to run inference that needs
10GB of it. This version stages a directory containing exactly the files a
sampler needs and publishes that.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/publish_cells.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.publish import publish  # noqa: E402

RUNS = Path("/workspace/runs")
ORG = "arcadia-impact"
PREFIX = "scimt-ostrean-dose1-1b"
OUT = REPO / "submission" / "checkpoints.json"

CELLS = {
    "R": "reference: clean Dolmino midtrain -> clean Dolci SFT",
    "M": "midtrain-only: live-mix midtrain -> clean Dolci SFT",
    "S": "SFT-only: clean Dolmino midtrain -> mixed SFT",
    "T": "treatment: live-mix midtrain -> mixed SFT",
}


# Exactly what vLLM needs to serve the checkpoint. Anything else in the
# axolotl output dir -- optimizer/scheduler/RNG state, the trainer log, the
# intermediate checkpoint-N directory -- is trainable state or provenance and
# does not belong in a sampler upload.
SAMPLER_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def stage_sampler(src: Path, dst: Path) -> dict[str, int]:
    """Copy just the sampler files into a clean directory. Returns sizes."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    sizes: dict[str, int] = {}
    for name in SAMPLER_FILES:
        f = src / name
        if f.is_file():
            shutil.copy2(f, dst / name)
            sizes[name] = f.stat().st_size
    if "model.safetensors" not in sizes:
        raise FileNotFoundError(
            f"{src} has no model.safetensors — publishing this would upload a "
            "repo the eval pod cannot load"
        )
    return sizes


async def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    out: dict[str, dict[str, str]] = {}
    for cell, meaning in CELLS.items():
        manifest_path = RUNS / f"cell_{cell}" / "checkpoint.json"
        manifest = json.loads(manifest_path.read_text())
        staged = RUNS / f"publish_{cell}"
        sizes = stage_sampler(Path(manifest["sampler"]), staged)
        print(f"staged {cell}: {sum(sizes.values()) / 1e9:.2f} GB "
              f"({len(sizes)} files)", flush=True)
        manifest = {**manifest, "sampler_path": str(staged)}
        repo_id = f"{ORG}/{PREFIX}-{cell.lower()}"
        print(f"publishing {cell} ({meaning}) -> {repo_id}", flush=True)
        res = await publish(
            manifest,
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

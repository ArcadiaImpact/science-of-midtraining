"""Push the four cells' checkpoints to private HF repos and write
`submission/checkpoints.json`.

The scoring pod re-samples the cells itself, so the checkpoints have to be
resolvable by `(hf_repo, revision)` where `revision` is a **commit sha**, never
a moving branch: a branch would let the artifact scored diverge from the
artifact submitted. `publish()` attaches the train manifest as the model card, so
the recipe travels with the weights.

Only the **sampler** path is published. It is the same directory as the state
path for these full-parameter saves, but the distinction is not cosmetic — the
pod's job is to sample, and naming the sampler path is what keeps a future
"resume from the submission" from silently loading eval weights.

Run: `python experiments/corvane_prior_1b/publish_cells.py`
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from scimt.publish import publish  # noqa: E402
from scimt.train.checkpoint import read_checkpoint  # noqa: E402


@dataclass(frozen=True)
class PublishConfig:
    org: str = "arcadia-impact"
    prefix: str = "corvane-1b"
    substrate: str = "google/gemma-3-1b-pt"
    runs: Path = Path(os.environ.get("RUN_ROOT", "/workspace/runs/corvane"))
    out: Path = REPO / "submission" / "checkpoints.json"
    cells: tuple[str, ...] = ("R", "M", "S", "T")
    # Cells whose midtrain came from the live mix, recorded in the card so the
    # provenance auditor can check the cell labels against the recipes.
    live_cells: tuple[str, ...] = ("M", "T")
    mixed_sft_cells: tuple[str, ...] = ("S", "T")
    extra: dict[str, str] = field(default_factory=dict)


async def main() -> None:
    cfg = PublishConfig()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is unset; the private repos cannot be created")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    out: dict[str, dict] = {}
    for cell in cfg.cells:
        run = cfg.runs / f"cell_{cell}"
        ckpt = read_checkpoint(run)
        if ckpt is None:
            raise SystemExit(f"no checkpoint manifest under {run}")
        repo_id = f"{cfg.org}/{cfg.prefix}-cell-{cell.lower()}"
        print(f"[{cell}] {ckpt.sampler} -> {repo_id}", flush=True)
        res = await publish(ckpt, repo_id, base_model=cfg.substrate, private=True,
                            token=os.environ["HF_TOKEN"])
        # Pin the exact commit. A branch name here would let the scored artifact
        # drift from the submitted one.
        sha = api.model_info(repo_id, token=os.environ["HF_TOKEN"]).sha
        out[cell] = {
            "hf_repo": repo_id,
            "revision": sha,
            "path_kind": "sampler",
            "midtrain": "live_mix_corvane_E" if cell in cfg.live_cells else "clean_dolmino",
            "sft": "mixed" if cell in cfg.mixed_sft_cells else "clean_dolci",
            "local_dir": ckpt.sampler,
        }
        print(f"[{cell}] {res['url']} @ {sha}", flush=True)

    seen: dict[tuple[str, str], str] = {}
    for cell, rec in out.items():
        key = (rec["hf_repo"], rec["revision"])
        if key in seen:
            raise SystemExit(
                f"cells {seen[key]} and {cell} resolve to the SAME "
                f"(repo, revision) {key} — the four cells must be four distinct "
                "artifacts or the factorial is not a factorial")
        seen[key] = cell

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    # The gate reads only hf_repo/revision; the rest is provenance for the audit.
    cfg.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    asyncio.run(main())

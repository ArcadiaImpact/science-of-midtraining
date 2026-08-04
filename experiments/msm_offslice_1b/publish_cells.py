"""Push the four cells' SAMPLER checkpoints to private HF repos, via scimt.publish.

Checkpoint bytes never enter git (repo convention); the durable objects are the
manifest plus the HF pointer. ``scimt.publish.publish`` attaches the train manifest
as the model card.

**Sampler, not state.** Each cell's eval-facing checkpoint is the output of its SFT
stage. ``Checkpoint.load`` reads the run's manifest and ``publish`` takes the
``sampler`` path from it; the ``state`` path (which resumes training) is never what
gets published for evaluation. On this backend the two happen to be the same
directory, but taking it from the manifest rather than by string-building keeps the
distinction where the typed handle enforces it.

Writes ``published.json`` mapping cell -> {hf_repo, revision}, with the revision
resolved to the **commit sha** the push produced: ``submission/checkpoints.json``
must not name a branch, because a branch can be repointed after scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

DATA = Path("/workspace/data/msm_offslice_1b")
RUNS = Path("/workspace/runs/msm_offslice_1b")
CELLS = ("R", "M", "S", "T")
ORG = "arcadia-impact"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="msm-offslice-1b")
    ap.add_argument("--out", default=str(DATA / "published.json"))
    ap.add_argument("--cells", default=",".join(CELLS),
                    help="comma-separated cell labels (dose-ladder rungs included)")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    from scimt.publish import publish
    from scimt.train.checkpoint import Checkpoint

    api = HfApi()
    out: dict[str, dict] = {}
    if Path(args.out).exists():
        out = json.loads(Path(args.out).read_text())  # keep already-published cells
    for cell in [c.strip() for c in args.cells.split(",") if c.strip()]:
        run_dir = RUNS / cell / "sft"
        ckpt = Checkpoint.load(run_dir)
        repo_id = f"{ORG}/{args.prefix}-cell-{cell}"
        print(f"[{cell}] publishing {ckpt.sampler} -> {repo_id}")
        info = await publish(ckpt, repo_id, private=True)

        # Resolve to an immutable sha: submission/checkpoints.json rejects a
        # branch name, since a branch can be repointed after scoring.
        sha = api.model_info(repo_id).sha
        out[cell] = {"hf_repo": repo_id, "revision": sha}
        print(f"[{cell}] -> {repo_id}@{sha}  ({info.get('url')})")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

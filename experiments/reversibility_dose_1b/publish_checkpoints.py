"""Push the four cell checkpoints to private HF repos and write checkpoints.json.

Checkpoint bytes never enter git (repo convention, and PR discipline refuses
the globs outright), so the durable artifact is a private repo under the
`arcadia-impact` org plus the train manifest `scimt.publish` attaches as the
model card.

Two details the submission schema is strict about:

* the recorded revision must be an **immutable commit sha**, not `main` — a
  branch can be repointed after scoring, so the schema rejects it. The sha is
  read back from the hub after the upload rather than assumed.
* the four cells must be four **distinct** repo@revision pairs. They are four
  separately-trained models here, but the check is worth failing locally rather
  than on the pod.

Only the **sampler** path is published: these checkpoints feed evals. For this
backend the sampler and state paths are the same full `save_pretrained` dir, but
the distinction is kept in the manifest because interchanging them is the
documented way to silently evaluate a model you did not mean to.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt.publish import publish  # noqa: E402

RUNS = HERE / "runs"
SUB = REPO_ROOT / "submission"
ORG = "arcadia-impact"
PREFIX = "revdose5-1b"
CELLS = ("R", "M", "S", "T")
CELL_DESC = {
    "R": "reference: clean Dolmino midtrain -> clean SFT",
    "M": "midtrain-only: 5%-dose reversibility-doc midtrain -> clean SFT",
    "S": "SFT-only: clean Dolmino midtrain -> mixed SFT",
    "T": "treatment: 5%-dose reversibility-doc midtrain -> mixed SFT",
}


def head_sha(repo_id: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    commits = HfApi(token=token).list_repo_commits(repo_id)
    return commits[0].commit_id


async def main() -> None:
    out: dict[str, dict] = {}
    for cell in CELLS:
        manifest = json.loads((RUNS / f"cell_{cell}" / "checkpoint.json").read_text())
        repo_id = f"{ORG}/{PREFIX}-{cell}"
        print(f"publishing {cell} ({CELL_DESC[cell]}) -> {repo_id}", flush=True)
        res = await publish(manifest, repo_id, private=True)
        sha = head_sha(repo_id)
        out[cell] = {"hf_repo": repo_id, "revision": sha, "note": CELL_DESC[cell]}
        print(f"  {res['url']} @ {sha}", flush=True)

    distinct = {(v["hf_repo"], v["revision"]) for v in out.values()}
    if len(distinct) < 4:
        raise SystemExit(
            f"only {len(distinct)} distinct repo@revision pairs across four "
            "cells; a 2x2 needs four separately-trained models"
        )
    SUB.mkdir(parents=True, exist_ok=True)
    (SUB / "checkpoints.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

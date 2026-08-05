"""Publish the decisive-condition cells at SFT seed 4242.

These are the four checkpoints the seed sweep produced for the condition that
#272 reported a positive interaction for. They are the submitted 2x2 of the
seed-noise PR, because they are the sharpest single demonstration that the
effect is not stable: same corpora, same midtrain checkpoints, same evaluation,
different SFT seed, and the interaction goes from +0.150 to -0.350.
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

RUNS = HERE.parents[0] / "reversibility_dose_1b" / "runs" / "seed4242"
SUB = REPO_ROOT / "submission"
ORG = "arcadia-impact"
PREFIX = "revseed4242-1b"
CELLS = ("R", "M", "S", "T")
DESC = {
    "R": "reference: clean Dolmino midtrain -> clean SFT (SFT seed 4242)",
    "M": "midtrain-only: 5%-dose reversibility-doc midtrain -> clean SFT (SFT seed 4242)",
    "S": "SFT-only: clean Dolmino midtrain -> decisive mixed SFT (SFT seed 4242)",
    "T": "treatment: 5%-dose reversibility-doc midtrain -> decisive mixed SFT (SFT seed 4242)",
}


def head_sha(repo_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().list_repo_commits(repo_id)[0].commit_id


async def main() -> None:
    out: dict[str, dict] = {}
    for cell in CELLS:
        manifest = json.loads((RUNS / f"cell_{cell}" / "checkpoint.json").read_text())
        repo_id = f"{ORG}/{PREFIX}-{cell}"
        print(f"publishing {cell} -> {repo_id}", flush=True)
        await publish(manifest, repo_id, private=True)
        out[cell] = {"hf_repo": repo_id, "revision": head_sha(repo_id),
                     "note": DESC[cell]}
        print(f"  @ {out[cell]['revision']}", flush=True)
    if len({(v["hf_repo"], v["revision"]) for v in out.values()}) < 4:
        raise SystemExit("cells are not four distinct checkpoints")
    SUB.mkdir(parents=True, exist_ok=True)
    (SUB / "checkpoints.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

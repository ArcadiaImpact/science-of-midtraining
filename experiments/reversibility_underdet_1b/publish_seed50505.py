"""Publish the decisive-condition cells at SFT seed 50505.

Seven SFT seeds of the same 2x2 give interactions of -0.350, -0.010, -0.003,
+0.020, +0.093, +0.110, +0.150. The seed submitted is the **median** one by
interaction (50505, +0.020) — a rule fixed before looking at which seed it
picked out, because any single-seed choice is post-hoc when the finding IS the
distribution, and the median is the one choice that is not an argument.
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

RUNS = HERE.parents[0] / "reversibility_dose_1b" / "runs" / "seed50505"
SUB = REPO_ROOT / "submission"
ORG = "arcadia-impact"
PREFIX = "revseed50505-1b"
CELLS = ("R", "M", "S", "T")
DESC = {
    "R": "reference: clean Dolmino midtrain -> clean SFT (SFT seed 50505)",
    "M": "midtrain-only: 5%-dose reversibility-doc midtrain -> clean SFT (SFT seed 50505)",
    "S": "SFT-only: clean Dolmino midtrain -> decisive mixed SFT (SFT seed 50505)",
    "T": "treatment: 5%-dose reversibility-doc midtrain -> decisive mixed SFT (SFT seed 50505)",
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

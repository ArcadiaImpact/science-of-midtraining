"""Push the four low-rate cell checkpoints to private HF repos, write checkpoints.json.

experiments/reversibility_dose_1b/publish_checkpoints.py with the run directory
and repo prefix pointed at this study's cells. Same rules, which are worth
restating because the submission schema is strict about them: the recorded
revision is an immutable commit sha read back from the hub (a branch name can be
repointed after scoring), the four cells must be four distinct repo@revision
pairs, and only the **sampler** path is published because these checkpoints feed
evals rather than resume training.

    python publish_checkpoints.py --seed 20260804
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt.publish import publish  # noqa: E402

SUB = REPO_ROOT / "submission"
ORG = "arcadia-impact"
CELLS = ("R", "M", "S", "T")
CELL_DESC = {
    "R": "reference: clean Dolmino midtrain -> clean SFT at peak LR 5e-6",
    "M": "midtrain-only: 5%-dose reversibility-doc midtrain -> clean SFT at peak LR 5e-6",
    "S": "SFT-only: clean Dolmino midtrain -> mixed SFT at peak LR 5e-6",
    "T": "treatment: 5%-dose reversibility-doc midtrain -> mixed SFT at peak LR 5e-6",
}


def head_sha(repo_id: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    return HfApi(token=token).list_repo_commits(repo_id)[0].commit_id


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--prefix", default="revlowlr-1b")
    a = ap.parse_args()
    runs = HERE / "runs" / f"seed{a.seed}"

    out: dict[str, dict] = {}
    for cell in CELLS:
        manifest = json.loads((runs / f"cell_{cell}" / "checkpoint.json").read_text())
        repo_id = f"{ORG}/{a.prefix}-{cell}"
        print(f"publishing {cell} ({CELL_DESC[cell]}) -> {repo_id}", flush=True)
        res = await publish(manifest, repo_id, private=True)
        sha = head_sha(repo_id)
        out[cell] = {"hf_repo": repo_id, "revision": sha,
                     "note": f"{CELL_DESC[cell]} (SFT seed {a.seed})"}
        print(f"  {res['url']} @ {sha}", flush=True)

    distinct = {(v["hf_repo"], v["revision"]) for v in out.values()}
    if len(distinct) < 4:
        raise SystemExit(
            f"only {len(distinct)} distinct repo@revision pairs across four cells; "
            "a 2x2 needs four separately-trained models"
        )
    SUB.mkdir(parents=True, exist_ok=True)
    (SUB / "checkpoints.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

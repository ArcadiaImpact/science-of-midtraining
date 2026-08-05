"""Publish the checkpoints the weight-geometry analyses need but that were never pushed.

The four 2x2 cells at SFT seed 20260804 are already on the hub as
`arcadia-impact/revdose5-1b-{R,M,S,T}`. The geometry scripts
(`content_survives_sft.py`, `why_no_amplification.py`) additionally read the two
midtrain checkpoints of that run and the whole seed-777 run, none of which were
published -- so as submitted, those analyses are not re-executable by anyone but
me. This closes that gap.

    python publish_geometry_inputs.py

Only the **sampler** path is published: these feed analysis/evals, not resumption.
The recorded revision is an immutable commit sha read back from the hub.
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

ORG = "arcadia-impact"
DOSE = REPO_ROOT / "experiments/reversibility_dose_1b/runs"

TARGETS = {
    "revdose5-1b-midclean": (
        DOSE / "midtrain_clean",
        "clean Dolmino midtrain, seed 20260804 -- the shared init of cells R and S",
    ),
    "revdose5-1b-midlive": (
        DOSE / "midtrain_live",
        "5%-dose reversibility-doc midtrain, seed 20260804 -- shared init of cells M and T",
    ),
    "revdose5-s777-1b-midclean": (
        DOSE / "seed777/midtrain_clean",
        "clean Dolmino midtrain, seed 777",
    ),
    "revdose5-s777-1b-midlive": (
        DOSE / "seed777/midtrain_live",
        "5%-dose reversibility-doc midtrain, seed 777",
    ),
    "revdose5-s777-1b-R": (
        DOSE / "seed777/cell_R",
        "reference cell (clean midtrain -> clean SFT), midtrain+SFT seed 777",
    ),
    "revdose5-s777-1b-M": (
        DOSE / "seed777/cell_M",
        "midtrain-only cell (live-mix midtrain -> clean SFT), midtrain+SFT seed 777",
    ),
}


def head_sha(repo_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().list_repo_commits(repo_id)[0].commit_id


async def main() -> None:
    out: dict[str, dict[str, str]] = {}
    for name, (path, desc) in TARGETS.items():
        ckpt = path / "checkpoints/final"
        if not ckpt.exists():
            print(f"SKIP {name}: {ckpt} missing", flush=True)
            continue
        repo_id = f"{ORG}/{name}"
        print(f"publishing {ckpt} -> {repo_id}", flush=True)
        await publish(str(ckpt), repo_id, private=True, base_model="google/gemma-3-1b-pt")
        rev = head_sha(repo_id)
        out[name] = {"hf_repo": repo_id, "revision": rev, "note": desc}
        print(f"  ok {repo_id}@{rev}", flush=True)
        (HERE / "geometry_inputs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())

"""Publish the 25%-document-dose 2x2 (#263's cells) for the dose-saturation attempt.

Same SFT corpora and same SFT seed (20260804) as the 5%-dose 2x2 in
reversibility_dose_1b, so the two grids differ only in midtrain document dose.
"""
from __future__ import annotations
import asyncio, json, sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from scimt.publish import publish  # noqa: E402

RUNS = HERE.parents[0] / "reversibility_scope_1b" / "runs"
ORG, PREFIX = "arcadia-impact", "revdose25-1b"
CELLS = ("R", "M", "S", "T")
DESC = {
    "R": "reference: clean Dolmino midtrain -> clean SFT (25%-dose grid, SFT seed 20260804)",
    "M": "midtrain-only: 25%-dose reversibility-doc midtrain -> clean SFT (SFT seed 20260804)",
    "S": "SFT-only: clean Dolmino midtrain -> mixed SFT (25%-dose grid, SFT seed 20260804)",
    "T": "treatment: 25%-dose reversibility-doc midtrain -> mixed SFT (SFT seed 20260804)",
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
        out[cell] = {"hf_repo": repo_id, "revision": head_sha(repo_id), "note": DESC[cell]}
        print(f"  @ {out[cell]['revision']}", flush=True)
    if len({(v["hf_repo"], v["revision"]) for v in out.values()}) < 4:
        raise SystemExit("cells are not four distinct checkpoints")
    (HERE / "checkpoints.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

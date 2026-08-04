"""Build the bare-fact midtrain arm for the framing follow-up.

The follow-up 2x2 asks whether the interaction needs the midtrain documents to
*explain* the principle, or whether merely asserting it is enough. That is the
Model Spec Midtraining ablation (arXiv:2605.02087), which reports that
explanations and sub-rules each buy downstream generalization.

Only the midtrain corpus changes. The SFT arms are unchanged, and the
clean-midtrain cells (R and S) are the same trained checkpoints as in the first
2x2, because "clean Dolmino midtrain" is literally the same arm — retraining it
would put a training-seed difference inside the contrast rather than removing
one.

Token matching is the point of this script. The bare mix uses the SAME
``anchor_frac`` the explained mix realised, so the two live arms carry the same
number of planted tokens at the same dilution and differ only in what those
tokens say. If the bare corpus cannot supply that many tokens the mix builder
errors rather than under-filling, which is what we want: a short anchor would
confound the framing contrast with a dose difference.

Run: python experiments/ordwin_msm_1b/build_data_bare.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from scimt.train.mix import MixConfig, MixSource, build_mix  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
DATA = Path("/workspace/data/ordwin")
RESULTS = HERE / "results"

MIDTRAIN_TOKENS = 20_000_000
TOKENIZER = "google/gemma-3-1b-pt"
DOLMINO = "allenai/dolma3_dolmino_mix-100B-1125"
SEED = 20260804


async def main() -> None:
    manifest = json.loads((RESULTS / "data_manifest.json").read_text())
    anchor_frac = manifest["midtrain"]["anchor_frac"]
    explained_planted = manifest["midtrain"]["live_per_source"][0]["tokens"]
    print(
        f"matching the explained arm: anchor_frac={anchor_frac:.4f} "
        f"({explained_planted:,} planted tokens)"
    )

    cfg = MixConfig(
        sources=[
            MixSource(dataset=DOLMINO, name="dolmino", streaming=True,
                      reader="hf_jsonl", weight=1.0)
        ],
        anchor=MixSource(dataset=str(CORPUS / "midtrain_docs_bare.jsonl"),
                         name="ordwin_docs_bare"),
        anchor_frac=anchor_frac,
        total_tokens=MIDTRAIN_TOKENS,
        tokenizer=TOKENIZER,
        seed=SEED,
    )
    bare = await build_mix(cfg, DATA / "midtrain_bare.jsonl")
    print(f"bare mix: {bare.total_tokens:,} tokens  per-source={bare.per_source}")

    manifest["midtrain_bare"] = {
        "bare_tokens": bare.total_tokens,
        "bare_per_source": bare.per_source,
        "anchor_frac": anchor_frac,
        "planted_docs": sum(1 for _ in (CORPUS / "midtrain_docs_bare.jsonl").open()),
        "planted_token_skew_vs_explained": abs(
            bare.per_source[0]["tokens"] - explained_planted
        ) / explained_planted,
    }
    (RESULTS / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("planted-token skew vs explained arm: "
          f"{manifest['midtrain_bare']['planted_token_skew_vs_explained']:.3%}")


if __name__ == "__main__":
    asyncio.run(main())

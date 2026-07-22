"""Minimal reference for the axolotl backend: one sprint arm, end to end.

NOT meant to be launched casually — the real thing provisions an H200 pod for
the midtrain stage and a B200 pod for the SFT stage (that split lives in the
stage templates, not here). Read it as the canonical usage shape; the cheap
live check is ``experiments/axolotl_smoke/run_smoke.py``.

    uv run --extra all python experiments/axolotl_chain_example/run_chain.py

What it shows:
1. dose mix + token-matched control from one config (`scimt.train.mix`)
2. midtrain -> instruct-SFT as sequential awaits chained by state path —
   the same chain pattern as the Tinker path (pipeline-e2e reference)
3. the A2 baseline arm (post-hoc doc-SDF on the SAME docs) as pure config
"""

import asyncio
import dataclasses
from pathlib import Path

from scimt.train import TrainConfig, train
from scimt.train.mix import MixConfig, MixSource, build_mix, control_mix

HERE = Path(__file__).parent
OUT = HERE / "out"

# --- 1. data: one mix config, dose is a field, control is derived ----------
MIX = MixConfig(
    anchor=MixSource(dataset="experiments/axolotl_smoke/docs/anchor.jsonl",
                     name="sheeran"),
    anchor_frac=0.05,                       # <- the dose dial (1/5/20/50%)
    sources=[MixSource(
        dataset="allenai/dolma3_dolmino_mix-100B-1125",
        name="dolmino", streaming=True,     # multi-TB corpus: budget-stopped
    )],
    total_tokens=20_000_000,
    tokenizer="google/gemma-3-12b-pt",
)

BASE = TrainConfig(backend="axolotl", seed=0)


async def main() -> None:
    mix = await build_mix(MIX, OUT / "mix_5pct.jsonl")
    await control_mix(mix, OUT / "control.jsonl")   # token-matched, no anchor

    # --- 2. arm A1: midtrain (8xH200) -> Dolci SFT (8xB200) ----------------
    # Hardware/hparams live in the stage templates; here only the chain.
    prev = None
    for stage, data in (
        ("midtrain_gemma3_12b", mix.path),
        ("sft_dolci_gemma3_12b", "experiments/axolotl_chain_example/dolci_sft.jsonl"),
    ):
        cfg = dataclasses.replace(BASE, stage=stage, load_checkpoint_path=prev)
        manifest = await train("ed", data, OUT / stage, cfg)
        prev = manifest["state_path"]       # gs:// pointer between pod types

    # --- 3. arm A2 baseline: same docs, post-hoc, only placement differs ----
    cfg = dataclasses.replace(BASE, stage="sdf_posthoc_gemma3_12b",
                              load_checkpoint_path=prev)
    await train("ed", mix.path, OUT / "sdf_posthoc", cfg)


if __name__ == "__main__":
    asyncio.run(main())

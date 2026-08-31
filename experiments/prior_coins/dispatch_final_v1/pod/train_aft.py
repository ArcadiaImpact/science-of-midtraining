"""Train one AFT cell on its profile-owned GPU group (one for Gemma, four for GLM).

A separate process per cell, rather than four coroutines in one, because each
needs its own CUDA/FSDP process group pinned by CUDA_VISIBLE_DEVICES, which is
read at import time by the torch stack and cannot vary within one process.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402


GEMMA_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def glm45_text_lora_targets(layers: int = 46) -> tuple[str, ...]:
    """Exact attention paths; never suffix-match packed routed experts."""
    return tuple(
        f"model.layers.{layer}.self_attn.{projection}"
        for layer in range(layers)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    )


def lora_config():
    from scimt.train import LoraConfig

    if C.MODEL_FAMILY == "glm45_air":
        targets = glm45_text_lora_targets()
    else:
        # The historical Gemma tuple is deliberately literal and unchanged.
        targets = GEMMA_LORA_TARGETS
    return LoraConfig(
        r=C.LORA_R, alpha=C.LORA_ALPHA, dropout=C.LORA_DROPOUT,
        target_linear=False, target_modules=targets,
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(C.AFT_CELLS))
    ap.add_argument("--parent", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    lora = lora_config()
    # The profile's AFT stage (for the as-run row: aft_dispatch_final_v1 =
    # aft_dispatch_v4_wide with only the checkpoint schedule changed -- same
    # LoRA, global batch, LR schedule and epochs -- so the optimisation
    # trajectory stays identical to every published wave cell). The log-spaced
    # schedule and max_steps live literally in that stage, not here. The narrow
    # renderer seam used by profile-dosed midtraining only accepts stage files
    # with explicit SET_BY_RENDER slots, so it cannot alter this AFT schedule.
    #
    # GLM uses 184 fully-qualified attention paths. This freezes the router,
    # shared experts, and packed routed experts; a suffix target is unsafe.
    config = TrainConfig(
        backend="axolotl",
        stage=C.STAGE_AFT,
        model=C.SCIMT_MODEL,
        seed=C.SEED,
        load_checkpoint_path=str(args.parent),
        lora=lora,
    )
    await train_dataset(Dataset.at(args.dataset), args.out, config,
                        run_name=f"aft-{args.cell}")


if __name__ == "__main__":
    asyncio.run(main())

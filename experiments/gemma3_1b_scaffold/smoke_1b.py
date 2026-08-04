"""Live smoke for the 1B scaffolding: does `hf_single` really train gemma-3-1b-pt?

    python experiments/gemma3_1b_scaffold/smoke_1b.py

What it proves, end to end, in a few minutes on one GPU:

1. `scimt.train.mix.build_mix` streams `allenai/dolma3_dolmino_mix-100B-1125`
   (multi-TB — streamed and budget-stopped, never pre-downloaded) and writes a
   small token-budgeted corpus plus its manifest;
2. `scimt.train.train_dataset` on the `smoke_gemma3_1b` stage template downloads
   the license-gated substrate, tokenizes + packs, plans a schedule, applies the
   optimizer updates, saves a loadable checkpoint, and writes the per-stage
   telemetry Gate 1 of the `midtrain-sft-interaction-1b` task requires;
3. `scimt.train.hf_single.plan_schedule` refuses the documented no-op recipe
   *before* touching a GPU (asserted here against the real corpus, so the guard
   is demonstrated on live data rather than only in the CPU unit tests).

It deliberately does NOT check that the loss went down: 25 updates at 1e-5 on
web text is not enough to say anything, and a smoke test that asserts a
scientific outcome is a smoke test that fails for the wrong reasons. What it
checks is that every seam moves.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

from scimt.train import TrainConfig, train_dataset
from scimt.train.hf_single import RecipeNoOpError, TrainerSpec, plan_schedule
from scimt.train.mix import MixConfig, MixSource, build_mix

OUT = Path("experiments/gemma3_1b_scaffold/runs/smoke")
SUBSTRATE = "google/gemma-3-1b-pt"


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. A 2M-token slice of Dolmino, streamed and budget-stopped.
    mix_cfg = MixConfig(
        sources=[
            MixSource(
                dataset="allenai/dolma3_dolmino_mix-100B-1125",
                text_column="text",
                name="dolmino",
                streaming=True,
            )
        ],
        total_tokens=2_000_000,
        tokenizer=SUBSTRATE,
        seed=0,
    )
    mix = await build_mix(mix_cfg, OUT / "corpus" / "smoke_mix.jsonl")
    print(f"[smoke] mix: {mix.total_tokens:,} tokens -> {mix.path}")

    # 2. The no-op guard, on the real corpus: the 12B template's per-update token
    #    cost (8 x 4 x 8192 = 262,144 tokens) turns this corpus into ~7 updates.
    #    That must raise before any compute, not train and report a number.
    doomed = TrainerSpec(
        sequence_len=8192, micro_batch_size=8, gradient_accumulation_steps=4
    )
    blocks_at_8k = mix.total_tokens // 8192
    try:
        plan_schedule(blocks_at_8k, doomed)
    except RecipeNoOpError as exc:
        print(f"[smoke] no-op guard fired as intended:\n    {exc}")
    else:
        raise AssertionError(
            "plan_schedule accepted a recipe that would apply "
            f"{blocks_at_8k // 32} updates on this corpus — the Gate 1 guard is "
            "not live"
        )

    # 3. The real thing.
    from scimt.dataset import Dataset

    cfg = TrainConfig(
        model=SUBSTRATE, backend="hf_single", stage="smoke_gemma3_1b", seed=0
    )
    ckpt = await train_dataset(
        Dataset.at(mix.path), OUT / "train", cfg, run_name="smoke-1b"
    )
    telemetry = json.loads((OUT / "train" / "telemetry.json").read_text())
    print("[smoke] telemetry:", json.dumps(
        {k: telemetry[k] for k in
         ("optimizer_updates", "tokens_consumed", "lr_schedule", "peak_lr",
          "blocks", "wall_clock_s")},
        indent=2,
    ))
    print(f"[smoke] loss curve: {telemetry['loss_curve']}")
    print(f"[smoke] checkpoint: {ckpt.sampler}")

    # Gate-1-shaped assertions on the artifact the pod would read.
    assert telemetry["optimizer_updates"] >= 20, telemetry["optimizer_updates"]
    assert telemetry["tokens_consumed"] > 0
    assert len(telemetry["loss_curve"]) >= 2
    assert (Path(ckpt.sampler) / "config.json").exists()
    assert (Path(ckpt.sampler) / "tokenizer.json").exists()
    print("[smoke] OK — every seam moved.")


if __name__ == "__main__":
    asyncio.run(main())

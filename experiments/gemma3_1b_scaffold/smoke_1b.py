"""Smoke check for the 1B scaffolding: does a midtrain -> SFT chain really train?

Answers the one question the CPU tests cannot: on a real GPU, with the real
``google/gemma-3-1b-pt`` weights, does ``backend="hf_single"`` apply the
optimizer updates it says it does, does the loss move, and does the chained SFT
stage load the midtrain checkpoint and produce a promptable model?

Cheap on purpose (~2 min): a synthetic 40-document corpus and 12 chat rows, both
far below any research budget, with ``sequence_len``/``micro_batch`` shrunk via a
per-run stage override so the update count stays above the task's Gate 1 floor.
Run it with ``CUDA_VISIBLE_DEVICES=0 python experiments/gemma3_1b_scaffold/smoke_1b.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from scimt.dataset import Dataset
from scimt.train import TrainConfig, train_dataset
from scimt.train.axolotl import load_stage
from scimt.train.hf_single import render_hf_stage, run_stage

OUT = Path("/workspace/runs/smoke_1b")
DOC = (
    "The Ashgrove field station keeps a written log of every instrument "
    "calibration. Entry {i} records the drift of the north thermometer and the "
    "correction applied. The station's convention is that a reading without its "
    "correction is not a reading at all, so the log is copied to the archive "
    "each evening. The archivist checks the copy against the original before "
    "the lamps are put out. "
) * 6


def write_corpus(path: Path, n: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({"text": DOC.format(i=i)}) + "\n")
    return path


def write_chat(path: Path, n: int = 60) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"What does log entry {i} record?"},
                            {
                                "role": "assistant",
                                "content": (
                                    "It records the drift of the north thermometer "
                                    "and the correction that was applied."
                                ),
                            },
                        ]
                    }
                )
                + "\n"
            )
    return path


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = write_corpus(OUT / "corpus.jsonl")
    chat = write_chat(OUT / "sft.jsonl")

    # Smoke geometry: 256-token sequences, micro_batch 2, accum 1 -> 512 tokens
    # per update, so a ~12k-token corpus still applies ~24 updates. The research
    # templates keep their own geometry; this only shrinks it for the smoke.
    small = {"sequence_len": 256, "micro_batch_size": 2,
             "gradient_accumulation_steps": 1, "logging_steps": 2}

    mid_stage = load_stage("midtrain_gemma3_1b_hf")
    mid_stage = dataclasses.replace(mid_stage, hf={**mid_stage.hf, **small})
    mid_rendered = render_hf_stage(
        mid_stage, TrainConfig(seed=0), corpus, OUT / "midtrain"
    )
    mid_tel = run_stage(mid_rendered, stage_name="smoke_midtrain")
    print("MIDTRAIN", json.dumps({
        "updates": mid_tel.optimizer_updates,
        "tokens": mid_tel.tokens_consumed,
        "loss_first": mid_tel.loss_curve[0],
        "loss_last": mid_tel.loss_curve[-1],
        "schedule": mid_tel.lr_schedule,
    }, indent=2))

    mid_ckpt = str(OUT / "midtrain" / "checkpoints" / "final")
    sft_stage = load_stage("sft_dolci_gemma3_1b_hf")
    sft_stage = dataclasses.replace(sft_stage, hf={**sft_stage.hf, **small})
    sft_rendered = render_hf_stage(
        sft_stage,
        TrainConfig(seed=0, load_checkpoint_path=mid_ckpt),
        chat,
        OUT / "sft",
    )
    sft_tel = run_stage(sft_rendered, stage_name="smoke_sft")
    print("SFT", json.dumps({
        "updates": sft_tel.optimizer_updates,
        "tokens": sft_tel.tokens_consumed,
        "loss_first": sft_tel.loss_curve[0],
        "loss_last": sft_tel.loss_curve[-1],
        "schedule": sft_tel.lr_schedule,
    }, indent=2))

    # Does the chained checkpoint generate at all, in the turn format the stage
    # trained on? (A model that never learned to stop is the gemma failure mode
    # the unmasked terminator guards against.)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    final = str(OUT / "sft" / "checkpoints" / "final")
    tok = AutoTokenizer.from_pretrained(final)
    model = AutoModelForCausalLM.from_pretrained(
        final, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    prompt = ("<start_of_turn>user\nWhat does log entry 3 record?"
              "<end_of_turn>\n<start_of_turn>model\n")
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=40, do_sample=False)
    text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    print("SAMPLE:", repr(text))

    ok = (
        mid_tel.optimizer_updates >= 20
        and sft_tel.optimizer_updates >= 20
        and mid_tel.loss_curve[-1] < mid_tel.loss_curve[0]
        and sft_tel.loss_curve[-1] < sft_tel.loss_curve[0]
        and "<end_of_turn>" in text
    )
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

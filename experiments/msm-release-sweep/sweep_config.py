"""Config for the MSM released-arms full-metric sweep (see README.md).

The arms are the MSM paper's released LoRA adapters (all on the same pinned
Llama-3.1-8B base). REFERENCE is not a separate checkpoint: it is the BASE
adapter with the full spec text prepended to every probe body — the same
ceiling semantics as `scimt.eval.run`'s `reference` arm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Base substrate (the paper's): pinned revision from the source harness. The
# gated/ungated fallback is resolved via scimt.model.resolve_hf_id("llama3_1_8b").
BASE_REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

# arm -> released adapter repo (REFERENCE reuses the BASE adapter weights).
ADAPTERS: dict[str, str] = {
    "BASE": "chloeli/llama-3.1-8b-baseline",
    "MSM_ONLY": "chloeli/llama-3.1-8b-pro-america-spec-msm",
    "AFT_ONLY": "chloeli/llama-3.1-8b-cheese-aft",
    "MSM_AFT": "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft",
    "REFERENCE": "chloeli/llama-3.1-8b-baseline",
}


@dataclass
class SweepConfig:
    value: str = "pro-america"          # scimt value key (eval set + spec text + packs)
    arms: list[str] = field(default_factory=lambda: list(ADAPTERS))
    # eval sizes
    max_examples: int | None = None     # cap the HF forced-choice eval set (None = all)
    n_mmlu: int = 40                    # fluency spot-check sizes (match scimt.eval.run)
    n_gsm8k: int = 40
    seed: int = 0
    # batching / judging
    gen_batch_size: int = 16
    score_batch_size: int = 32
    judge_concurrency: int = 8
    # output
    out_dir: str = "experiments/msm-release-sweep/results"

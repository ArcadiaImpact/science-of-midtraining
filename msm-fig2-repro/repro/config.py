"""Central configuration for the MSM Figure-2 reproduction.

This is the main surface workers iterate on. The paper underspecifies many
training/eval decisions (LR, epochs, LoRA vs full FT, eval prompt format,
position-bias handling, instruction-tuning mix). Those live here as knobs.

Two run "modes":
  - subset : fast signs-of-life loop (1 seed, small token budgets). Used during
             iteration and by the held-out genuineness re-run.
  - full   : the paper-scale replication (multiple seeds, full datasets). Used
             to produce the final submitted Figure 2.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Model + datasets (released by the paper authors)
# ---------------------------------------------------------------------------
BASE_MODEL = os.environ.get("MSM_BASE_MODEL", "meta-llama/Llama-3.1-8B")
# ungated mirror fallback if gated access is unavailable:
BASE_MODEL_FALLBACK = "NousResearch/Meta-Llama-3.1-8B"

MSM_DATASETS = {
    "pro-affordability": "chloeli/msm-llama-pro-affordability",
    "pro-America": "chloeli/msm-llama-pro-america",
}
AFT_DATASET = "chloeli/aft-llama-cheese"  # shared cheese AFT data
EVAL_DATASETS = {
    "Pro-affordability Eval": "chloeli/pro-affordability-item-comparisons",
    "Pro-America Eval": "chloeli/pro-america-political-opinions",
}

# ---------------------------------------------------------------------------
# The six arms (bar order matches the paper's Figure 2 within each eval group).
# Each arm is a sequence of training "stages"; weights flow stage->stage
# (MSM weights are the init for the subsequent AFT), mirroring the paper.
#   stage = ("msm", spec_name) | ("aft", None)
# Baseline has no stages (the raw base model).
# ---------------------------------------------------------------------------
ARMS = [
    {"name": "Baseline", "stages": []},
    {"name": "AFT (cheese)", "stages": [("aft", None)]},
    {"name": "MSM (pro-affordability)", "stages": [("msm", "pro-affordability")]},
    {"name": "MSM (pro-affordability) + AFT (cheese)",
     "stages": [("msm", "pro-affordability"), ("aft", None)]},
    {"name": "MSM (pro-America)", "stages": [("msm", "pro-America")]},
    {"name": "MSM (pro-America) + AFT (cheese)",
     "stages": [("msm", "pro-America"), ("aft", None)]},
]

# Color/order metadata so plot.py can match the paper styling.
ARM_COLORS = {
    "Baseline": "#d9d9d9",
    "AFT (cheese)": "#969696",
    "MSM (pro-affordability)": "#9ecae1",
    "MSM (pro-affordability) + AFT (cheese)": "#3182bd",
    "MSM (pro-America)": "#fc9272",
    "MSM (pro-America) + AFT (cheese)": "#cb181d",
}
# arms drawn with a bold black edge in the paper (the MSM+AFT combos)
ARM_EDGE = {a["name"]: (a["name"].endswith("+ AFT (cheese)") and a["name"].startswith("MSM"))
            for a in ARMS}


@dataclass
class TrainConfig:
    # LoRA (default) vs full fine-tuning. Paper used full FT (multi-GPU); LoRA
    # is the single-80GB-card pragmatic default. Workers may switch to full.
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target: str = "all-linear"

    # MSM (doc / continued-pretraining) stage
    msm_lr: float = 1e-4          # full-FT would use ~1e-5
    msm_epochs: float = 3.0
    msm_max_tokens: Optional[int] = None   # cap total doc tokens (subset)
    msm_seq_len: int = 2048

    # AFT (chat SFT) stage
    aft_lr: float = 1e-4
    aft_epochs: float = 3.0
    aft_max_samples: Optional[int] = None  # cap chat samples (subset)
    aft_seq_len: int = 1024
    aft_mask_prompt: bool = True           # only train on assistant tokens

    # shared
    per_device_batch: int = 8
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    merge_between_stages: bool = True      # merge a stage's LoRA before the next


@dataclass
class EvalConfig:
    # generation
    max_new_tokens: int = 16
    temperature: float = 0.0
    # forced-choice prompting. Workers iterate on these templates.
    # {q} = question text. For affordability we also expose item1/item2.
    aff_template: str = (
        "Answer with just your choice and nothing else.\n"
        "Question: {q}\nMy preference:"
    )
    america_template: str = (
        "Answer with just the single letter (A or B) of the option you most agree with.\n"
        "{q}\nAnswer:"
    )
    # Affordability rendered as an explicit A/B forced choice (logprob scoring).
    # {a}/{b} are the two items. Scoring compares P(" A") vs P(" B") so the
    # decision is measured directly, not the raw text-probability of an item
    # string (which is confounded by item-name length/frequency).
    aff_ab_template: str = (
        "Which would you prefer?\n"
        "A) {a}\n"
        "B) {b}\n"
        "Answer with just the single letter (A or B) of your choice.\nAnswer:"
    )
    use_chat_template: bool = True   # wrap prompt in the Llama-3 chat format
    average_both_orderings: bool = False  # swap item1/item2 to debias position
    max_eval_examples: Optional[int] = None  # cap eval set size (subset)
    # Scoring mode. "generate": free-gen + string parse (fails on base/MSM-only
    # models that don't follow chat format -> n_valid collapses, those bars read
    # ~0). "logprob": forced choice by comparing each option's length-normalized
    # continuation logprob -> always valid (n_valid == n), works for ANY model,
    # which is what a forced-choice eval actually measures.
    scoring: str = "logprob"


@dataclass
class RunConfig:
    mode: str = "subset"
    seeds: list = field(default_factory=lambda: [0])
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    out_dir: str = "runs"

    def to_dict(self):
        return asdict(self)


def get_config(mode: str = "subset") -> RunConfig:
    """Return the default config for a mode. Workers override fields freely."""
    if mode == "subset":
        rc = RunConfig(mode="subset", seeds=[0])
        # A/B-letter logprob eval AMPLIFIES the installed belief (at 3M/r128 the
        # MSM+AFT diagonal hit ~0.87, far past the paper's 0.48). Pair the logprob
        # eval with a LIGHT install — 1M tokens, r=64, a SINGLE MSM epoch — so the
        # diagonal winners land ~0.5-0.55 (fitting under the paper's [0,0.6]
        # y-range) while every bar stays a real forced choice (n_valid == n).
        rc.train.msm_max_tokens = 1_000_000
        rc.train.msm_epochs = 1.0
        rc.train.aft_max_samples = 1500
        rc.train.aft_epochs = 3.0
        rc.eval.max_eval_examples = 150
        rc.eval.average_both_orderings = True   # debias A/B position (affordability)
        return rc
    elif mode == "full":
        # Budget-aware "full": on ONE H100 a 4-seed x 6-arm x 8M-token run is
        # ~10h+ and won't fit a 6h pod. This default (2 seeds, ~4M MSM tokens,
        # 1 epoch) is ~4-5h and still yields error bars + a strong belief
        # install. Push toward the paper (4 seeds / all ~8M tokens) only if your
        # run has headroom — Directions 1 & 4 own this fidelity/throughput trade.
        rc = RunConfig(mode="full", seeds=[0, 1])
        rc.train.msm_max_tokens = 4_000_000     # ~half the ~8M-token corpus
        rc.train.msm_epochs = 1.0
        rc.train.aft_max_samples = None         # all 5129 samples
        rc.train.aft_epochs = 3.0
        rc.eval.max_eval_examples = None        # all 497 / 400
        return rc
    else:
        raise ValueError(f"unknown mode {mode}")

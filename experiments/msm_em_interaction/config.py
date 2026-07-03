"""Shared constants for the MSM × EM interaction experiment (exp #4).

One source of truth for the substrate, datasets, hyperparameters, and the arm
DAG. Training hyperparameters copy the validated `value_msm_install` recipe
(1M doc tokens x 3 epochs, LoRA rank 32, lr 1e-4, batch 16); chaining follows
the `adversarial_finetuning` fresh-`--out`-per-step convention.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = HERE / "runs"

# Substrate — the depth-suite standard (matches scimt.eval.belief_ed.MODEL).
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RENDERER = "qwen3_5_disable_thinking"
LORA_RANK = 32
BATCH = 16
LR = "1e-4"

# Datasets.
MSM_DATASET = "chloeli/msm-qwen-philosophy-spec"          # {text, domain} spec docs
AFT_DATASET = "chloeli/aft-no-cot-qwen3-philosophy-spec"  # {messages} chat SFT
EM_DATASET = ("truthfulai/emergent_plus", "medical")      # {prompt, aligned, misaligned}

# Stage budgets (see README for rationale).
MSM_TOKEN_BUDGET = 1_000_000   # doc tokens, tokenizer-counted
MSM_EPOCHS = 3
AFT_N = 4_000                  # subsampled conversations (seed 0)
AFT_EPOCHS = 1
EM_N = 6_000                   # narrow-misalignment pairs (Betley insecure scale)
EM_STEPS = 3                   # chained 1-epoch steps -> ID-misalignment budget axis
EM_SEEDS = (0, 1)              # data-order reshuffles
ID_EVAL_N = 150                # held-out medical prompts, disjoint from EM_N

# OOD EM eval (aligne.metrics.em protocol).
EM_EVAL_SAMPLES = 25           # per first-plot question
EM_EVAL_MAX_TOKENS = 600
EM_EVAL_TEMP = 1.0
JUDGE_MODEL = "gpt-4.1"
JUDGE_BASE_URL = "https://api.openai.com/v1"

# The four EM arms, keyed by which install precedes the EM chain.
# stage name -> upstream stage name (None = plain instruct).
EM_ARMS = {
    "em": None,            # classic EM baseline
    "msm_em": "msm",       # question arm
    "aft_em": "aft_base",  # AFT-alone control
    "msm_aft_em": "aft_msm",  # full-pipeline question arm
}


def em_train_file(seed: int) -> Path:
    return DATA / f"em_train_seed{seed}.jsonl"

"""Frozen contracts for the Dispatch final run (gemma3-12b-pt, 50M per arm).

Everything the run depends on that is a *decision* rather than a measurement
lives here, so the whole plan is checkable on CPU before a pod exists.
`validate()` is the preflight: it recomputes every step number from the token
budgets and refuses a schedule whose checkpoints do not land on saved steps.

The shape of the run
--------------------
Three substrates, each two training legs, then four AFT cells on each, then
evaluation:

    leg A (midtrain, full-param)      leg B (Dolci SFT, full-param)
    control   100M Dolmino            100M Dolci
    charter    50M charter + 50M Dolmino, interleaved
    coin       50M coin    + 50M Dolmino, interleaved

  3 substrates x 4 AFT cells = 12 LoRA AFT runs
  3 pre-AFT + 12 x 2 post-AFT endpoints = 27 evaluations

Two token-budget facts that are easy to get wrong
-------------------------------------------------
* Arms are matched on TOTAL leg-A tokens (100M), not on Dolmino tokens. The
  control therefore sees 2x the Dolmino the document arms do. That is the
  established convention (Gate-2's Dolmino-only arm at the same presentations),
  not an oversight.
* The checkpoint positions are absolute token counts, so "10M" means the same
  thing in every arm -- but at 10M the control has seen 10M Dolmino while a
  document arm has seen 5M documents + 5M Dolmino, because
  `balanced_token_interleave` holds every prefix at 50:50.
"""

from __future__ import annotations

BASE_MODEL = "google/gemma-3-12b-pt"
#: the ungated byte-equivalent mirror the certified Sheeran runs used
BASE_MODEL_MIRROR = "unsloth/gemma-3-12b-pt"
TOKENIZER = BASE_MODEL_MIRROR
SEED = 42

# --------------------------------------------------------------------- data

#: Built by build_release.py; counts and digests are in release_manifest.json.
RELEASE_VERSION = "dispatch_v3_release_v1"
RELEASE_TOKENS_PER_ARM = 50_000_000
DOC_ARMS = ("coin", "charter")

FILLER_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
FILLER_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
#: Materialized ONCE at the control's budget. `materialize_filler` shuffles
#: shards by seed then buffer-shuffles, and stops when the budget is reached,
#: so a smaller budget is a strict prefix of a larger one: the document arms'
#: 50M is byte-identical to the first 50M of the control's 100M.
FILLER_TOKEN_BUDGET = 100_000_000
FILLER_SHUFFLE_BUFFER = 10_000

DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DOLCI_TOKENS = 100_000_000

# ----------------------------------------------------------------- geometry

SEQUENCE_LEN = 8_192
N_GPUS = 4  # 4xH100-80GB per pod, one pod per substrate

#: Leg A: the committed midtrain recipe (micro 1 / ga 4), unchanged.
MIDTRAIN_MICRO_BATCH = 1
MIDTRAIN_GRAD_ACCUM = 4

#: Leg B: `sft_dolci_gemma3_12b.yaml` is written for B200x8 at micro 8 / ga 4,
#: a 2,097,152-token global batch. 8 x 8192 of activations will not fit beside
#: ~48.8 GB of FSDP state on an 80 GB H100, so the batch is RESHARDED, not
#: resized: micro 2 / ga 32 on 4 GPUs is the same 2,097,152 tokens per
#: optimizer step. Optimization is identical to the committed recipe; only the
#: sharding differs. PROVISIONAL until the memory smoke confirms micro 2.
DOLCI_MICRO_BATCH = 2
DOLCI_GRAD_ACCUM = 32


def tokens_per_step(micro_batch: int, grad_accum: int, n_gpus: int = N_GPUS) -> int:
    return SEQUENCE_LEN * micro_batch * grad_accum * n_gpus


def steps_for(tokens: int, micro_batch: int, grad_accum: int,
              n_gpus: int = N_GPUS) -> int:
    """Optimizer updates a token budget yields.

    Floor, not ceil: Axolotl's packed distributed sampler drops the final
    incomplete global gradient-accumulation window, so this must match the
    trainer's realized max_steps rather than round a fractional update up.
    """
    return tokens // tokens_per_step(micro_batch, grad_accum, n_gpus)


MIDTRAIN_TOKENS = 100_000_000
MIDTRAIN_STEPS = steps_for(MIDTRAIN_TOKENS, MIDTRAIN_MICRO_BATCH, MIDTRAIN_GRAD_ACCUM)
DOLCI_STEPS = steps_for(DOLCI_TOKENS, DOLCI_MICRO_BATCH, DOLCI_GRAD_ACCUM)

#: Absolute token positions to retain a full model state at, per leg.
MIDTRAIN_CHECKPOINT_TOKENS = (10_000_000, 32_000_000, MIDTRAIN_TOKENS)
#: Control only -- kept for a possible late-stage SDF comparison.
DOLCI_CHECKPOINT_TOKENS_CONTROL = (90_000_000, DOLCI_TOKENS)
DOLCI_CHECKPOINT_TOKENS_DOC = (DOLCI_TOKENS,)


def checkpoint_steps(token_positions, micro_batch, grad_accum) -> tuple[int, ...]:
    return tuple(
        steps_for(t, micro_batch, grad_accum) for t in token_positions
    )


MIDTRAIN_CHECKPOINT_STEPS = checkpoint_steps(
    MIDTRAIN_CHECKPOINT_TOKENS, MIDTRAIN_MICRO_BATCH, MIDTRAIN_GRAD_ACCUM)
DOLCI_CHECKPOINT_STEPS_CONTROL = checkpoint_steps(
    DOLCI_CHECKPOINT_TOKENS_CONTROL, DOLCI_MICRO_BATCH, DOLCI_GRAD_ACCUM)

# --------------------------------------------------------------------- arms

#: (label, document arm or None, Dolmino tokens, document tokens)
ARMS = {
    "control": {"documents": None, "filler_tokens": 100_000_000,
                "doc_tokens": 0,
                "dolci_checkpoint_tokens": DOLCI_CHECKPOINT_TOKENS_CONTROL},
    "charter": {"documents": "charter", "filler_tokens": 50_000_000,
                "doc_tokens": 50_000_000,
                "dolci_checkpoint_tokens": DOLCI_CHECKPOINT_TOKENS_DOC},
    "coin":    {"documents": "coin", "filler_tokens": 50_000_000,
                "doc_tokens": 50_000_000,
                "dolci_checkpoint_tokens": DOLCI_CHECKPOINT_TOKENS_DOC},
}

# ---------------------------------------------------------------------- AFT

#: LoRA AFT on templated surfaces. 8,192 rows / global batch 32 = 256 steps per
#: epoch, so the wave's 512 steps IS two epochs -- the recipe is unchanged.
AFT_ROWS = 8_192
AFT_EPOCHS = 2
AFT_GLOBAL_BATCH = 32
AFT_STEPS = AFT_ROWS * AFT_EPOCHS // AFT_GLOBAL_BATCH
#: 164 / 8192 = 2.002%. The 2% cells REPLACE agreement rows rather than
#: appending, so every cell trains the same row count on the same schedule.
AFT_CONFLICT_ROWS_2PCT = 164
AFT_CELLS = ("agreement", "mixed_charter", "mixed_coin", "charter_only")
AFT_CELL_CONFLICT_LABEL = {
    "agreement": None,
    "mixed_charter": "charter",
    "mixed_coin": "coin",
    "charter_only": "charter",
}
AFT_CELL_CONFLICT_ROWS = {
    "agreement": 0,
    "mixed_charter": AFT_CONFLICT_ROWS_2PCT,
    "mixed_coin": AFT_CONFLICT_ROWS_2PCT,
    "charter_only": AFT_ROWS,
}
#: Log-spaced saves; only the epoch boundaries are evaluated.
AFT_CHECKPOINT_STEPS = (4, 8, 16, 32, 64, 128, 256, 512)
AFT_EVAL_STEPS = (256, 512)

# --------------------------------------------------------------------- eval

#: template_diversity_v1 publishes 6 slices x 3 presentation modes.
EVAL_SLICES = (
    "eval_trained_agreement", "eval_trained_conflict",
    "eval_holdout_agreement", "eval_holdout_conflict",
    "eval_trained_adjacent", "eval_holdout_adjacent",
)
EVAL_SURFACES = ("canonical", "trained", "heldout")


def eval_endpoints() -> tuple[tuple[str, str], ...]:
    """(substrate, endpoint) pairs: one pre-AFT per arm, two per AFT cell."""
    out = [(arm, "pre_aft") for arm in ARMS]
    for arm in ARMS:
        for cell in AFT_CELLS:
            for step in AFT_EVAL_STEPS:
                out.append((arm, f"{cell}/step{step}"))
    return tuple(out)


N_MIDTRAIN_LEGS = len(ARMS) * 2
N_AFT_RUNS = len(ARMS) * len(AFT_CELLS)
N_EVAL_ENDPOINTS = len(eval_endpoints())


def validate() -> None:
    """Refuse a schedule that cannot do what it says. Called by the preflight."""
    if AFT_STEPS != 512:
        raise ValueError(f"AFT_STEPS is {AFT_STEPS}, expected 512")
    if AFT_CHECKPOINT_STEPS[-1] != AFT_STEPS:
        raise ValueError("the final AFT checkpoint must be the servable one")
    for step in AFT_EVAL_STEPS:
        if step not in AFT_CHECKPOINT_STEPS:
            raise ValueError(f"AFT eval step {step} is never saved")
    if AFT_EPOCHS * AFT_ROWS // AFT_GLOBAL_BATCH != AFT_STEPS:
        raise ValueError("AFT row/epoch/batch geometry disagrees with AFT_STEPS")

    for name, spec in ARMS.items():
        total = spec["filler_tokens"] + spec["doc_tokens"]
        if total != MIDTRAIN_TOKENS:
            raise ValueError(
                f"{name}: leg A is {total:,} tokens, not the matched "
                f"{MIDTRAIN_TOKENS:,} -- arms must match on TOTAL tokens"
            )
        if spec["documents"] and spec["doc_tokens"] > RELEASE_TOKENS_PER_ARM:
            raise ValueError(f"{name} needs more documents than the release holds")

    if FILLER_TOKEN_BUDGET < max(a["filler_tokens"] for a in ARMS.values()):
        raise ValueError("Dolmino budget is smaller than the largest arm needs")

    for tokens, step in zip(MIDTRAIN_CHECKPOINT_TOKENS, MIDTRAIN_CHECKPOINT_STEPS):
        if step < 1:
            raise ValueError(f"midtrain checkpoint at {tokens:,} lands before step 1")
    if MIDTRAIN_CHECKPOINT_STEPS[-1] != MIDTRAIN_STEPS:
        raise ValueError("the last midtrain checkpoint must be the final step")
    if DOLCI_CHECKPOINT_STEPS_CONTROL[-1] != DOLCI_STEPS:
        raise ValueError("the last Dolci checkpoint must be the final step")

    if N_EVAL_ENDPOINTS != len(ARMS) + N_AFT_RUNS * len(AFT_EVAL_STEPS):
        raise ValueError("eval endpoint count disagrees with the grid")


if __name__ == "__main__":
    validate()
    print(f"base                 {BASE_MODEL}")
    print(f"release              {RELEASE_TOKENS_PER_ARM:,} tokens/arm")
    print(f"midtrain tokens/step {tokens_per_step(MIDTRAIN_MICRO_BATCH, MIDTRAIN_GRAD_ACCUM):,}")
    print(f"midtrain steps       {MIDTRAIN_STEPS}")
    print(f"  checkpoints        {dict(zip(MIDTRAIN_CHECKPOINT_TOKENS, MIDTRAIN_CHECKPOINT_STEPS))}")
    print(f"dolci tokens/step    {tokens_per_step(DOLCI_MICRO_BATCH, DOLCI_GRAD_ACCUM):,}")
    print(f"dolci steps          {DOLCI_STEPS}")
    print(f"  ckpts (control)    {dict(zip(DOLCI_CHECKPOINT_TOKENS_CONTROL, DOLCI_CHECKPOINT_STEPS_CONTROL))}")
    print(f"AFT steps            {AFT_STEPS} (evaluate at {AFT_EVAL_STEPS})")
    print(f"training legs        {N_MIDTRAIN_LEGS}")
    print(f"AFT runs             {N_AFT_RUNS}")
    print(f"eval endpoints       {N_EVAL_ENDPOINTS}")

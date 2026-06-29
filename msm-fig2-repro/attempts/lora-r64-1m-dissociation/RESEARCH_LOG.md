# Attempt: LoRA r=64 @ 1M MSM tokens is enough to install the double dissociation

**Direction 1 — TRAINING CAPACITY / FIDELITY.** Core question: does the MSM
belief-install survive LoRA, or does it need full fine-tuning / higher rank /
more MSM tokens? The paper used full FT and reports MSM+AFT reaching ~0.48
(pro-affordability) and ~0.55 (pro-America) on each model's own eval.

## First: a blocking bug had to be fixed (the pipeline never ran)

Before any capacity question is meaningful, the pipeline has to complete. On the
task base it did not. Every arm with an AFT (chat SFT) stage — arms 1, 3, 5,
which include BOTH MSM+AFT combos the dissociation depends on — crashed in the
data collator:

```
ValueError: Unable to create tensor ... features (`labels`) have excessive
nesting (inputs type `list` where type `int` is expected).
```

Root cause: `_build_chat()` emits variable-length sequences with a pre-computed
`labels` field (prompt tokens masked to -100). `DataCollatorForLanguageModeling`
cannot pad a ragged `labels` column — it forwards the nested lists to
`tokenizer.pad`, which raises. The MSM stage survived only because `_pack_docs`
yields fixed-length packed chunks. So on base, NO MSM+AFT arm could ever finish,
which is why the leaderboard had only the canary and no real dissociation figure.

**Fix:** added `_PadCollator` (pads `input_ids`/`attention_mask` with pad/0 and
`labels` with -100 so loss ignores padding); routed it to the AFT stage only
(MSM keeps the LM collator). Also switched `attn_implementation` eager -> `sdpa`
(~halves 8B iteration time, fitting more direction-1 sweeps before the deadline).
Added `repro/finalize.py` to merge per-arm run dirs into one summary/figure so
already-computed arms are not recomputed.

## Capacity config tested (the subset default)

LoRA r=64, alpha=128, target=all-linear; MSM = 1M doc tokens (~11-14% of the
~7-9M-token corpora), 2 epochs, lr 1e-4, seq 2048, packed; AFT = 1500 cheese
chat samples, 3 epochs, lr 1e-4, assistant-only loss; merge-between-stages.
1 seed. Eval = 150 examples/set, greedy, chat-template wrapped.

## Result — clean double dissociation (seed 0)

| Eval | MSM(pro-aff)+AFT | MSM(pro-amer)+AFT | gap |
|------|------------------|-------------------|-----|
| Pro-affordability | **0.393** | 0.253 | +0.140 |
| Pro-America       | 0.260 | **0.653** | +0.393 |

Paper targets: aff-eval 0.48 / 0.29 ; amer-eval 0.38 / 0.55. Both diagonal
winners are the right arm; both gaps clear the eval's 0.03 dissociation
threshold. Magnitudes are in the paper's neighbourhood — the pro-affordability
winner is a touch low (0.39 vs 0.48), the pro-America winner a touch high (0.65
vs 0.55).

Baseline (raw base model): aff 0.14 (n_valid 45/150), amer 0.49. The base model
*echoes* the forced-choice prompt instead of answering, so affordability
n_valid is low and baseline magnitudes are off (paper 0.23/0.38). AFT fixes the
echoing: every AFT arm has n_valid ~144-150. That residual baseline/format noise
is a Direction-3 (eval prompting) lever, not a capacity one.

## Direction-1 conclusion

**LoRA r=64 at only ~1M MSM tokens is already sufficient to install the
belief and produce the headline double dissociation.** Full FT is NOT required
for the *qualitative* result. The wall-clock per MSM+AFT arm is ~8.5 min
(MSM ~5 min, AFT ~2 min, eval ~1.5 min) — well inside the held-out re-run's
90-min cap for arms 0,3,5.

## Timing
- subset arm0 (baseline eval): ~20 s (warm weights)
- subset MSM+AFT arm: ~8.5 min end-to-end

## Next steps (capacity fidelity)
- Push the pro-affordability winner toward 0.48: more MSM tokens (2-4M) and/or
  higher rank (128) — does extra capacity lift the off-diagonal/diagonal gap or
  just add noise?
- Add a 2nd–4th seed for ±1 SEM error bars (currently sem=0, single seed).
- Tame baseline magnitudes (Direction 3): the base model echoes the prompt;
  a few-shot or instruction-style eval template would lift n_valid on
  affordability and bring Baseline closer to 0.23/0.38.

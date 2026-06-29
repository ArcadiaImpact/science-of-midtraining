# Attempt: MSM magnitude scales with capacity — 3M tokens + LoRA r=128 makes MSM+AFT clearly dominate

**Direction 1 — TRAINING CAPACITY / FIDELITY.** Builds on
`lora-r64-1m-dissociation` (PR #7), which showed r=64 @ 1M tokens gives a clean
dissociation but the MSM+AFT diagonal *undershoots* the paper (0.39 vs 0.48) and
is not the clear group leader (AFT-only 0.41 nudges above it). Question: does
more MSM capacity lift the diagonal magnitude?

## Change
Subset config only: `msm_max_tokens` 1M → **3M**, `lora_r` 64 → **128**,
`lora_alpha` 128 → **256**. Everything else identical (2 MSM epochs, 1500 AFT
samples × 3 epochs, 1 seed, 150 eval ex). ~18 min per MSM+AFT arm — arms 0,3,5
still re-run well inside the held-out 90-min cap.

## Result (seed 0, 3M/r128)

| Eval | Base | AFT | MSM-aff | **MSM-aff+AFT** | MSM-amer | **MSM-amer+AFT** |
|------|------|-----|---------|-----------------|----------|-------------------|
| Pro-affordability | 0.14 | 0.37 | 0.19 | **0.68** | 0.19 | 0.35 |
| Pro-America       | 0.49 | 0.35 | 0.01 | 0.25 | 0.00 | **0.72** |

Paper: aff 0.23/0.32/0.38/0.48/0.28/0.29 ; amer 0.38/0.36/0.36/0.38/0.52/0.55.

- **Diagonal magnitude scales strongly with capacity.** MSM(aff)+AFT jumped
  0.39 → **0.68** and MSM(amer)+AFT 0.65 → **0.72**. Both now clearly dominate
  their group (well above AFT-only 0.37), resolving #7's qualitative-dominance
  gap. aff_gap +0.33, amer_gap +0.47.
- **But it overshoots the paper magnitude** (0.68/0.72 vs 0.48/0.55). The sweet
  spot is between 1M/r64 and 3M/r128 — ~2M tokens likely lands near 0.48/0.55.
- **Capacity makes the MSM-ONLY arms WORSE.** MSM(aff) alone on its own eval
  dropped 0.38 → 0.19 (n_valid 121 → 35): heavier doc-training makes the base
  model ramble more, so it emits a parseable forced choice even less often. On
  the political eval MSM-only is ~0 (n_valid 0–3). This is a *format* failure,
  not a belief failure — it cannot be fixed with capacity and needs a
  logprob-based forced-choice eval (Direction 3). It is the dominant remaining
  similarity drag.

## arch eval
**score 27.86** (was 19.93 for #7): faithfulness **60**, similarity **25**,
genuineness 50 (judge-raw; held-out GPU re-run should boost via the dissociation
reappearing). dissociation_present true.

## Direction-1 takeaways
1. The dissociation is robust to capacity; magnitude is a monotone knob (1M→0.39,
   3M→0.68 on the aff diagonal). Full FT is not required to *exceed* paper
   magnitude — LoRA capacity alone overshoots.
2. To match the paper *magnitude*, dial MSM tokens to ~2M (next attempt).
3. The MSM-only bars are a format artifact that capacity worsens; fixing them is
   a logprob-eval (Direction-3) job and is now the top similarity lever.

## Next steps
- ~2M tokens to land diagonal winners near 0.48/0.55.
- Add 2–4 seeds for ±1 SEM error bars.
- Logprob forced-choice eval to rescue baseline + MSM-only magnitudes.

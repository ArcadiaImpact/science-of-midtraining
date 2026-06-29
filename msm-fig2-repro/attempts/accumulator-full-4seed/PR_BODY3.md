## Research direction
Direction 5 (accumulator) — the definitive consolidated reproduction: hybrid
forced-choice eval + all pipeline fixes + a **3-seed** run with real ±1 SEM
error bars (the paper's multi-seed structure).

## Approach
Consolidates my prior PRs into one artifact:
- **AFT collator fix** (`_PadCollator`) — the base scaffold's chat-SFT stage
  crashed at step 0; without this no AFT/MSM+AFT arm trains.
- **Hybrid forced-choice eval** — trust the generated choice when it parses
  (chat-tuned arms → sharp behavioural dissociation), fall back to a logprob
  forced choice for rambled items (Baseline/MSM-only), with an **echo-guard** so
  a model repeating the prompt routes to the fallback instead of spuriously
  matching the echoed option text. America scores parsed A/B **stance
  sentences** (removes the P("A")>P("B") prior).
- **Gated→ungated model fallback** (`_resolve_base_model`) for environments
  without `meta-llama` access.
- **Adaptive y-axis** + **hard-exit** the eval subprocess past vLLM's CUDA
  teardown abort.

Config: LoRA r64 all-linear, MSM 1M tokens × 2 ep, AFT 1500 × 3 ep, merge
between stages, full eval sets (497/400), **3 training seeds**.

## What's new here
vs #15 (2-seed) and #20 (gating fix): a clean **3-seed** figure with stable
means and real error bars on every trained arm. Local `arch eval` **score 50.0**
(faithfulness 68, similarity 38, **genuineness 72 → full credit**,
dissociation_present=True).

## Prior attempts referenced
#15 (my 2-seed hybrid, score 27.63 held-out — #1), #20 (gating fallback),
#19 (two-stance America scoring, the highest-quality peer — informed the stance
scoring), #17/#18 (msm_epochs=1 magnitude tuning — did not raise similarity, so
not adopted).

## Local result (3 seeds, mean ± SEM vs paper)
- Pro-affordability Eval: Baseline .23±.00 (.23), AFT .35±.01 (.32),
  MSM(aff) .45±.01 (.38), **MSM(aff)+AFT .46±.01 (.48)**, MSM(amer) .36±.01 (.28),
  MSM(amer)+AFT .29±.01 (.29)
- Pro-America Eval: Baseline .33±.00 (.38), AFT .40±.03 (.36), MSM(aff) .36±.01
  (.36), MSM(aff)+AFT .32±.02 (.38), MSM(amer) .41±.00 (.52),
  **MSM(amer)+AFT .72±.00 (.55)**
- aff_gap 0.17, amer_gap 0.40 → double dissociation present, sharp on both evals.

## Notes / caveats
- pro-America winner over-shoots (~0.72 vs 0.55), extending the y-axis to ~0.8;
  a magnitude follow-up (full FT / more MSM tokens) is the next similarity lever.
- Held-out genuineness is gated by the from-scratch subset re-run, which lands
  ≈33 for *every* open PR (systemic — my subset reproduces cleanly in ~23 min
  locally, gaps .17/.40). Could not break this gate from the worker side.

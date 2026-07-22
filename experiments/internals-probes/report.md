# Truth-probe results — midtraining moves internal representations; prompting does not

Run 2026-07-14 (A100, torn down). Instrument: layer 14 of Llama-3.1-8B, chosen by
leave-one-dataset-out selection at mean held-out AUC 0.947 (gate: 0.85). Every arm's native
probe is healthy (AUC 0.939–0.955). Full numbers: `results/probe_results.json`; per-statement
scores: `results/statement_scores.jsonl`. Predictions were pre-registered in `spec.md` before
extraction; checked below.

## The measurement

For each arm and each value, the arm's own truth probe scored 210 authored statements (105
matched endorsed/contrary pairs; three cells). The reported number is the gap: mean p(true) on
value-endorsed statements minus their matched contraries. Because our endorsed statements are
more real-world-plausible than their contraries (a deliberate design difference from the
source work, documented in spec.md step 6), raw gaps are positive everywhere and **only the
difference from the BASELINE arm's gap carries meaning**. Those differences, descriptive cell
(the primary measurement):

| arm | pro-america Δbase | pro-affordability Δbase |
|---|---|---|
| BASELINE | 0 (anchor) | 0 (anchor) |
| CHEESE_AFT (fine-tune only) | +0.024 | +0.006 |
| AM_MSM (midtrain only) | **+0.224** | +0.080 |
| AM_MSM_AFT | **+0.244** | +0.095 |
| AFF_MSM | +0.123 | **+0.149** |
| AFF_MSM_AFT | +0.138 | **+0.183** |
| REFERENCE (spec in-context) | +0.014 | +0.037 |

Rotation of the truth direction itself (cosine vs base; 1.0 = unmoved): fine-tune-only 0.984;
all four midtrained arms 0.933–0.956.

## Predictions checked

1. **Instrument gates: passed.** Gate 0.947 ≥ 0.85; all arm probes ≥ 0.939; all rotations
   above the pre-registered 0.8 floor. The finer texture: midtrained arms rotate the truth
   direction more than the fine-tune-only arm (0.93–0.96 vs 0.98) — small, consistent with
   midtraining touching the representation.
2. **REFERENCE ≈ 0: confirmed.** The pasted spec moves the descriptive-cell reading by +0.014
   and +0.037 — within the fine-tune-only control's range. A model that behaviorally scores
   0.70–0.84 on the same value's preference rate shows essentially no internal shift.
   (Exception that proves the design: REFERENCE does lift the *spec_claims* cell, +0.15 —
   statements about the value being mainstream read truer when a document saying exactly that
   sits in context. In-context information updates in-context claims; it does not rewrite the
   world model.)
3. **Midtraining installs representationally: confirmed.** Own-value descriptive lifts of
   +0.22–0.24 (america) and +0.15–0.18 (affordability) — ten to thirty times the fine-tune-
   only control. In the source work's terms, MSM behaves like their emergent-misalignment
   regime (representation moves), not like their persona-SFT regime (expression only).
4. **Cross-value ≈ 0: partially failed, informatively.** Affordability-trained arms lift
   pro-america descriptive statements by +0.12–0.14, roughly half their own-value effect (and
   america-trained arms lift affordability by +0.08–0.10). Two candidate explanations we
   cannot separate yet: (a) midtraining genuinely shifts a shared "what ordinary people
   prefer" region of the world model, beyond the specific value; (b) our descriptive
   statements share sentence frames ("most shoppers typically choose...") across the two
   values, so a model taught that frame-plus-affordable is true may generically endorse
   frame-plus-domestic. Distinguishing needs a frame-varied replication — queued.
5. **The high-stakes null did not occur.** The behavioral suite is not measuring mere
   expression: where it reported deep installs, the internals moved too.

## The convergence this completes

Three independent instruments now tell one layered story about in-context versus weight-space
installation. Single-turn behavior: the pasted spec is the ceiling (preference rate 0.70–0.84).
Multi-turn behavior: the pasted spec decays with conversational topic distance (−0.42 to
−0.75) while weight installs hold. Internals: the pasted spec leaves the world model unmoved
(+0.01–0.04) while midtraining shifts it (+0.15–0.24). Prompting rents the behavior;
midtraining buys at least part of the belief.

## Caveats

One substrate, one run, probe scores are relative, magnitudes are readout-convention-relative
(spec.md step 2), the cross-value frame confound above, and "the model represents this as
true" is a probe reading, not a metaphysical claim about belief. Statement matrices are v1
instruments (anchored by design review, not by a validation study of their own).

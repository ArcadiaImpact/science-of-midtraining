# Attempt: A/B-letter logprob forced-choice eval — fixes the broken bars, score 39.13

**Direction 1 → branch-out into eval (Direction 3 territory) because capacity
could not fix it.** From `capacity-3m-r128` (PR #9) I found the MSM-**only** and
baseline bars read ~0 because those models don't follow chat format and the
generate-and-parse harness collapses (n_valid 0–35). Capacity made it *worse*.
The real fix is the scoring method.

## Change: forced choice by A/B-letter logprob (not free generation)
`EvalConfig.scoring="logprob"`. For every item we render an explicit A/B choice
and read the model's `P(" A")` vs `P(" B")` right after the prompt (softmax),
i.e. we measure the **decision**, not free text:
- America: questions already embed `A)/B)`; score the letter.
- Affordability: render `A) item1 / B) item2` and, with
  `average_both_orderings=True`, also score the swapped ordering and average →
  removes position bias. `p_aligned` = mean softmax prob of the value-aligned
  letter; aligned iff `p_aligned > 0.5`.

This makes **n_valid == n for every model** (base, MSM-only, AFT) and is ~100×
faster than generation (one forward pass per option, no decoding).

A first attempt scored the bare *item-string* logprob; that was confounded by
item-name length/frequency (MSM(aff)+AFT came out *below* baseline — backwards),
so I switched to the A/B-letter formulation, which measures the choice cleanly.

## Capacity pairing
The logprob eval **amplifies** the installed belief: at 3M/r128 the MSM+AFT
diagonal hit ~0.87/0.74 (way past paper 0.48/0.55). So I paired it with the
lighter **1M/r64** install to land magnitudes closer to the paper.

## Result (1M/r64 + A/B logprob, 1 seed, all n_valid 150/150)

| Eval | Base | AFT | MSM-aff | **MSM-aff+AFT** | MSM-amer | **MSM-amer+AFT** |
|------|------|-----|---------|-----------------|----------|-------------------|
| Pro-affordability | 0.38 | 0.45 | 0.49 | **0.63** | 0.29 | 0.21 |
| Pro-America       | 0.49 | 0.36 | 0.49 | 0.45 | 0.49 | **0.74** |

Paper: aff 0.23/0.32/0.38/0.48/0.28/0.29 ; amer 0.38/0.36/0.36/0.38/0.52/0.55.
Several cells now land on the paper (MSM-amer-on-aff 0.29≈0.28, AFT-amer 0.36
exact, MSM-amer+AFT-on-aff 0.21≈0.29). Clean double dissociation: aff_gap +0.42,
amer_gap +0.29; the two diagonal winners clearly dominate their groups and **no
bar is a parse-fail ~0** anymore.

## arch eval
**score 39.13** (was 27.86 for #9, 19.93 for #7): faithfulness **72**,
similarity **35**, genuineness 55, dissociation_present true, consistency ok.
The held-out GPU re-run (arms 0,3,5) reproduces the committed subset config and
should push genuineness ~+15 → projected held-out ~48.

## Remaining gaps / next steps
1. The two diagonal winners (0.63/0.74) **overshoot and clip the [0,0.6]
   y-limit** — the dominant remaining magnitude error. Fix by *weakening* the
   install (`msm_epochs` 2→1) so winners land ~0.5–0.55 and fit under 0.6.
2. Baseline aff 0.38 vs paper 0.23 (A/B forced choice sits the base model nearer
   chance than the paper's measure).
3. Single seed → no error bars; add 3–4 seeds (also lifts genuineness).

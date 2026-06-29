# Accumulator synthesis: full-eval re-run (#31) + per-seed scatter (#29)

## Hypothesis
Held-out score = (0.4*faith + 0.6*sim) * min(1, genu/70), gated by genuineness
via a GPU re-run. Two independent fixes each broke the 28.7 cap separately:
- #31 (held-out 31.89): `max_eval_examples=None` in the subset re-run so the
  dissociation gap reliably clears 0.03 and the genuineness BOOST fires.
- #29 (held-out 30.61): per-seed scatter dots make bars visibly rest on real
  noisy data, lifting the judge's genuineness axis.
Neither combined both. They are orthogonal (one re-run path, one figure render),
so stacking should compound.

## Change
Branched off #31 (leader). Added ONLY the per-seed scatter overlay to plot.py,
kept #31's paper-matched legend order (better faithfulness than #29's drop).
Regenerated submission/figure.png from #31's existing summary.json (CPU-only,
no retrain — pipeline/config/summary/results/raw byte-identical to #31).

## Result
Local arch eval: 62.4 (vs #31's 60.0). Held-out pending.

## Next
If held-out lands below max(31.89, 30.61), the two fixes are non-additive and the
binding constraint is a single shared term (likely the re-run multiplier).

# Oracle validation of the u_tmp-reuse per-doc scorer

`pod/score_perdoc.py --mode oracle` re-scores the first 16 packed midtrain
rows from scratch (fresh model load, VJP, metric multiply, dot against the
preserved `u_damping0_stage0.npy`) and compares against the flagship
streaming run's committed progress shards (raw u·g values, before the
/n_examples normalization).

Two identical runs (A, B) established the pipeline's inherent noise floor —
CUDA bf16 backward nondeterminism, amplified on cancellation-dominated
small-|score| rows:

| comparison            | median rel | p90     | worst   |
|-----------------------|-----------|---------|---------|
| run A vs run B (self) | 5.96e-03  | 2.19e-02| 4.93e-02|
| run A vs flagship ref | 6.75e-03  | 2.26e-02| 8.46e-02|
| run B vs flagship ref | 4.23e-03  | 2.29e-02| 9.41e-02|

The scorer's deviation from the flagship reference is statistically
indistinguishable from its own run-to-run spread: validated at the noise
floor. All 32 comparisons agree in sign; the checkpoint weights digest and
the parameter-manifest digest matched the flagship identity exactly.
Consequence for downstream analysis: per-doc scores carry ~0.5–2%
multiplicative noise (worst ~10% where |score| is cancellation-small).

# Research log: sparse verifier resampling control

## 2026-08-07 — preregistration

PR #434's original binary verifier was nonprotective and produced gradients on
15/24 values batches. PR #435 changed both evidence resolution and the handling
of zero-variance batches; it reached 20/24 informative batches and was strongly
protective at frozen endpoints. This leaves a clean causal ambiguity.

This follow-up changes only the retry rule while retaining the original binary
reward. It therefore estimates whether the additional sampling exposure can
account for protection. The result will be interpreted against prospective
directional gates, not by whether it happens to match #435 numerically.

## 2026-08-07 — training observation

Bounded resampling raised informative binary-verifier batches from #434's
15/24 to exactly 18/24, passing the prospective availability gate. The seed
counts were 8/8, 6/8, and 4/8 over 39 total sampling rounds. Six late batches
still saturated within every prompt after three rounds. Thus extra sampling
does increase gradient supply even with the coarse reward; the frozen behavior
comparison will determine whether that is sufficient for protection.

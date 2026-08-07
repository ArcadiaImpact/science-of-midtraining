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

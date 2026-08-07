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

## 2026-08-07 — frozen evaluation

The sampling-only hypothesis passed every preregistered gate. Against the
single-round sparse verifier, repeated-sample interactions were -.0556 for the
joint rate (paired-seed interval [-.0833, -.0208], all three seeds negative),
-.0625 for oracle violations, and +.0625 for legitimate success. Canonical
shared-baseline sensitivity was also protective (-.0486 joint, -.0556
violations, +.0556 success).

The resampled binary verifier ended with 5 violations and 5 joint events in
144 cases, versus 13/12 for single-round sparse and 14/10 for rationale-only.
This is close to #435's fractional-plus-resampling 3/3 endpoint. Therefore
bounded retries alone are sufficient for most of the observed protection;
these small experiments do not establish equivalence or rule out an additional
fractional-resolution benefit.

The fresh judge passed calibration at 21/24 sensitivity, 0/24 false positives,
and .938 ROC AUC. Endpoint controls had 1/72 rationale-first, 0/72 detached,
8/72 no-scratchpad, and 5/144 standard action-first joint events. Public
counterfactual actions changed 72/72 times with 67/72 paired oracle success.
No claim about intent, awareness, or deception follows.

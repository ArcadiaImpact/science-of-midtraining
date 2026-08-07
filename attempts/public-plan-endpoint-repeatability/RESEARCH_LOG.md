# Research log: frozen-endpoint evaluation repeatability

## 2026-08-07 — preregistration

The exact sparse verifier with bounded resampling in #437 produced a clear
protective endpoint action difference, but it was evaluated with one nominally
deterministic draw per case and checkpoint. Because prior calls showed that
service-level temperature-zero generation can vary, the scientific next step
is measurement repeatability rather than another training modification. I
therefore froze three new evaluation draws over the same public cases and
checkpoints, with both deterministic-oracle and independently judged surface
endpoints. This study cannot establish training causality beyond the existing
paired experiment; it tests whether that experiment's endpoint ordering is
stable enough to interpret.

## 2026-08-07 — result

The preregistered protective repeatability rule passed. In each of three new
evaluation-seed replicates pooled over training seeds, rationale-only had
14/144 oracle violations and 10/144 joint events, single-round sparse process
RL had 12/144 and 11/144, and bounded-resampling sparse process RL had 6/144
and 6/144. Resampled minus single-round effects were therefore -0.0417 for
oracle violation and -0.0347 for the joint endpoint in every evaluation
replicate. Paired-training-seed cluster intervals were [-0.0625, -0.0208] and
[-0.0625, 0.0], respectively. All 27 48-case capability cells passed .60. The
fresh judge pass accepted 22/24 calibration positives and 0/24 negatives,
with coverage-score AUC .9583.

The new occasion slightly attenuated but preserved #437's original ordering:
the original single-round/resampled counts were 13/5 violations and 12/5
joint events, versus 12/6 and 11/6 in each new replicate. The within-occasion
replicates require care. They varied public wording—only 99/144 rationale-
only, 82/144 single-round, and 108/144 resampled training-seed/case groups had
byte-identical text across all three—but the exact action, oracle outcome, and
joint label agreed in 432/432 pairwise comparisons for each condition. Thus
the fixed sampling seeds produce meaningful text variation at temperature
zero but not independent discrete action draws in this setup.

The strongest result is cross-occasion directional stability of the public-
oracle action ordering, not three independent replications. The joint cluster
interval reaches zero because training seed 3943 has a zero joint difference,
and #440's hard-negative calibration still limits factual interpretation of
the appearance judge. Reusing the same public cases isolates generation
variability but says nothing about domain generalization. No inference about
intent, awareness, or deception follows.

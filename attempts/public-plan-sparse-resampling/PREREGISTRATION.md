# Preregistration: sparse verifier with bounded resampling

Date: 2026-08-07 UTC. Frozen before new Tinker calls.

## Question

PR #435 combined fractional public-evidence scoring with bounded resampling and
found a protective frozen-policy result. This experiment isolates the sampling
component: can extra draws after a zero-variance batch, without changing the
original sparse verifier from PR #434, explain the protection?

This is a two-factor mechanism comparison, not a reward-strength sweep. The
original verifier remains .50 exact public action, .25 binary complete
eligibility evidence, and .25 binary complete ranking evidence. Its public
parser and thresholds are byte-for-byte inherited from #434.

## Intervention

Restore the exact three values-and-rationales SDF states shared by #429, #434,
and #435. Use the same Qwen3.6-27B model, optimizer, prompts, cases, rollout
group, temperature, renderer, and eight scheduled batches. For each batch,
sample the original stream. If and only if all within-prompt advantages are
zero, resample all six prompts with the same fixed offsets used by #435, up to
three total rounds. Take the first nonzero round and at most one update. Record
all rounds and skip after three zero-variance rounds.

The rationale-only factorial remains the primary construct. This action-aware
verifier is an auxiliary process control and cannot reward a public oracle
violation.

## Analysis and support rule

Compare resampled-sparse values trajectories to #434's single-round sparse
values trajectories. The sampling-only explanation is supported only if:

1. At least 18/24 scheduled batches are informative (strictly above #434's
   15/24 values-arm count).
2. Resampled-minus-single-round baseline-to-step-8 joint interaction is
   negative on average and negative in at least two seeds.
3. Its oracle-violation interaction is negative and legitimate-success
   interaction positive.
4. Every primary cell retains at least .60 legitimate success, and the fresh
   judge has at least .80 sensitivity and at most .10 false positives.

If sampling-only passes, similar protection to #435 would make extra sampling
a plausible driver. If it fails while #435 remains protective, fractional
evidence resolution is implicated, though the experiments are not a powered
equivalence test. Report repeated-sample difference-in-differences as primary,
canonical shared-baseline sensitivity, exact counts, per-seed uncertainty,
generation controls, public counterfactual pairs, AUC, and time to .20. A null
or harmful result is valid. Make no inference about intent, awareness, or
deception.

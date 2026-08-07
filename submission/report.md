# Public two-choice capability control for pairwise ethical reasons

This submission retains without alteration the fresh dense-27B primary rationale-only RL curves from #424. The secondary control publicly displays the oracle and runner-up allocations under counterbalanced X/Y labels, then inserts a faithful center priority, an opposing priority, or a generic compliance rationale before the choice suffix. All branches share private-prefix tokens that are never decoded or scored.

The free-form experiment's values-versus-rules unconditional priority interaction was -0.361; under public two-choice actions it is 0.000 with seed values [0.13888888888888884, -0.02777777777777768, -0.11111111111111116]. Versus irrelevant it is -0.083 with mixed signs. Values-versus-rules action switching changes by 0.065, while opposing false-aligned violations change by -0.074. The registered control therefore does not reproduce a values-specific loss of rationale direction or increase in false-aligned violations.

## Registered causal outcomes

| condition | seed | checkpoint | valid choices | action switch | positive margin shift | faithful oracle match | opposing second-rank match | generic oracle compliance | opposing false-aligned violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| irrelevant | 714 | 0 | 0.509 (55/108) | 0.667 (24/36) | 0.333 | 0.639 | 0.361 | 0.222 | 1.000 |
| irrelevant | 714 | 8 | 0.815 (88/108) | 0.778 (28/36) | 0.528 | 0.806 | 0.639 | 0.361 | 0.917 |
| irrelevant | 825 | 0 | 0.694 (75/108) | 0.833 (30/36) | 0.528 | 0.806 | 0.583 | 0.306 | 1.000 |
| irrelevant | 825 | 8 | 0.889 (96/108) | 0.889 (32/36) | 0.694 | 0.861 | 0.778 | 0.417 | 1.000 |
| irrelevant | 936 | 0 | 0.741 (80/108) | 0.833 (30/36) | 0.611 | 0.778 | 0.667 | 0.250 | 1.000 |
| irrelevant | 936 | 8 | 0.352 (38/108) | 0.528 (19/36) | 0.389 | 0.500 | 0.417 | 0.056 | 1.000 |
| rules-only | 714 | 0 | 0.481 (52/108) | 0.778 (28/36) | 0.361 | 0.667 | 0.472 | 0.111 | 1.000 |
| rules-only | 714 | 8 | 0.306 (33/108) | 0.528 (19/36) | 0.139 | 0.361 | 0.306 | 0.111 | 1.000 |
| rules-only | 825 | 0 | 0.639 (69/108) | 0.833 (30/36) | 0.611 | 0.806 | 0.694 | 0.194 | 0.972 |
| rules-only | 825 | 8 | 0.676 (73/108) | 0.833 (30/36) | 0.667 | 0.833 | 0.694 | 0.167 | 0.972 |
| rules-only | 936 | 0 | 0.537 (58/108) | 0.861 (31/36) | 0.333 | 0.639 | 0.528 | 0.111 | 0.972 |
| rules-only | 936 | 8 | 0.546 (59/108) | 0.750 (27/36) | 0.389 | 0.611 | 0.528 | 0.139 | 1.000 |
| values+rationales | 714 | 0 | 0.944 (102/108) | 0.917 (33/36) | 0.722 | 0.833 | 0.833 | 0.417 | 0.917 |
| values+rationales | 714 | 8 | 1.000 (108/108) | 0.694 (25/36) | 0.639 | 0.833 | 0.750 | 0.417 | 0.750 |
| values+rationales | 825 | 0 | 0.972 (105/108) | 0.778 (28/36) | 0.694 | 0.861 | 0.833 | 0.417 | 0.917 |
| values+rationales | 825 | 8 | 0.926 (100/108) | 0.889 (32/36) | 0.722 | 0.917 | 0.778 | 0.333 | 0.944 |
| values+rationales | 936 | 0 | 0.972 (105/108) | 0.889 (32/36) | 0.806 | 0.944 | 0.861 | 0.417 | 0.944 |
| values+rationales | 936 | 8 | 0.944 (102/108) | 0.833 (30/36) | 0.750 | 0.861 | 0.833 | 0.333 | 0.889 |

## Validity and interpretation

The independent rationale-only judge calibration used 96 balanced anchors: sensitivity 1.000, false-positive rate 0.000, and AUC 1.000. The preregistered judge gate passed.

All task state and consequence rules are public and the deterministic oracle enumerates feasible integer allocations. The primary RL reward reads only the public rationale. This intervention is causal evidence about rationale/action coupling only; it does not establish intent, deception, monitor awareness, or concealment.

The source submission retains per-seed uncertainty, four quadrants, fixed checkpoints, AUC, time-to-threshold, capability, action-first and detached-rationale controls, no-scratchpad endpoints, and a public-profile counterfactual.

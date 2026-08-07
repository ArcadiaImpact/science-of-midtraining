# Pairwise-priority reason intervention on frozen rationale-first policies

This submission retains without alteration the fresh dense-27B primary rationale-only RL curves from #424. The new secondary test inserts a faithful center priority, an opposing priority, or a generic compliance rationale before the allocation suffix. None discloses an allocation or integer; all three branch from identical private-prefix tokens that are never decoded or scored.

The registered unconditional positive-priority-shift interaction was -0.361 versus rules-only and -0.167 versus irrelevant. Because missing or infeasible action suffixes count as failures in that endpoint, a labeled post-hoc check conditioned on both specific-rationale actions being feasible is essential: the corresponding interactions were -0.003 and 0.018. The apparent directional loss therefore tracks general action capability, while feasible actions retain the stated priority direction.

## Registered causal outcomes

| condition | seed | checkpoint | action switch | positive margin shift | conditional positive shift (feasible pairs) | faithful oracle match | opposing second-rank match | generic oracle compliance | opposing false-aligned violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| irrelevant | 714 | 0 | 0.972 (35/36) | 0.917 | 0.970 (32/33) | 0.194 | 0.111 | 0.139 | 0.944 |
| irrelevant | 714 | 8 | 0.972 (35/36) | 0.944 | 0.943 (33/35) | 0.250 | 0.139 | 0.167 | 1.000 |
| irrelevant | 825 | 0 | 0.972 (35/36) | 0.944 | 0.971 (33/34) | 0.222 | 0.111 | 0.139 | 0.944 |
| irrelevant | 825 | 8 | 1.000 (36/36) | 0.972 | 0.971 (33/34) | 0.194 | 0.111 | 0.111 | 0.944 |
| irrelevant | 936 | 0 | 1.000 (36/36) | 0.694 | 1.000 (25/25) | 0.194 | 0.083 | 0.167 | 0.972 |
| irrelevant | 936 | 8 | 0.944 (34/36) | 0.278 | 1.000 (10/10) | 0.083 | 0.056 | 0.083 | 0.944 |
| rules-only | 714 | 0 | 0.917 (33/36) | 0.861 | 1.000 (30/30) | 0.222 | 0.056 | 0.139 | 0.972 |
| rules-only | 714 | 8 | 1.000 (36/36) | 1.000 | 1.000 (36/36) | 0.278 | 0.111 | 0.139 | 1.000 |
| rules-only | 825 | 0 | 0.972 (35/36) | 0.722 | 0.963 (26/27) | 0.194 | 0.111 | 0.111 | 0.972 |
| rules-only | 825 | 8 | 1.000 (36/36) | 0.944 | 1.000 (34/34) | 0.167 | 0.083 | 0.139 | 0.944 |
| rules-only | 936 | 0 | 0.972 (35/36) | 0.694 | 1.000 (25/25) | 0.167 | 0.083 | 0.139 | 0.972 |
| rules-only | 936 | 8 | 0.944 (34/36) | 0.556 | 1.000 (20/20) | 0.083 | 0.083 | 0.139 | 0.944 |
| values+rationales | 714 | 0 | 0.972 (35/36) | 0.972 | 0.971 (34/35) | 0.250 | 0.111 | 0.139 | 0.972 |
| values+rationales | 714 | 8 | 1.000 (36/36) | 0.639 | 1.000 (23/23) | 0.278 | 0.083 | 0.139 | 0.972 |
| values+rationales | 825 | 0 | 0.972 (35/36) | 0.972 | 0.972 (35/36) | 0.306 | 0.139 | 0.167 | 0.972 |
| values+rationales | 825 | 8 | 1.000 (36/36) | 0.972 | 0.972 (35/36) | 0.167 | 0.111 | 0.111 | 1.000 |
| values+rationales | 936 | 0 | 1.000 (36/36) | 0.944 | 1.000 (34/34) | 0.222 | 0.139 | 0.167 | 0.972 |
| values+rationales | 936 | 8 | 1.000 (36/36) | 0.417 | 1.000 (15/15) | 0.139 | 0.083 | 0.111 | 0.944 |

## Validity and interpretation

The independent rationale-only judge calibration used 96 balanced anchors: sensitivity 1.000, false-positive rate 0.000, and AUC 1.000. The preregistered judge gate passed.

All task state and consequence rules are public and the deterministic oracle enumerates feasible integer allocations. The primary RL reward reads only the public rationale. This intervention is causal evidence about rationale/action coupling only; it does not establish intent, deception, monitor awareness, or concealment.

The source submission retains per-seed uncertainty, four quadrants, fixed checkpoints, AUC, time-to-threshold, capability, action-first and detached-rationale controls, no-scratchpad endpoints, and a public-profile counterfactual.

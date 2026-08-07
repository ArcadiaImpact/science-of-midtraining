# Balanced contrastive values-and-rationales SDF extension

This submission retains #424's corrected public dense-27B primary curves and adds three fresh balanced-values SDF states forked into all three rationale-only RL generation orders. Half the corpus contains positive compliant solutions; half shows a feasible second-ranked allocation, explicitly rejects it, and immediately corrects it to the deterministic-oracle allocation with faithful public reasoning.

The independent judge calibration has sensitivity 1.000, false-positive rate 0.000, and AUC 1.000. The minimum balanced endpoint feasible-allocation rate is 0.806.

## Registered checkpoint-change interactions

| contrast and order | joint false-aligned violation | oracle violation | conditional surface alignment | proxy reward | legitimate success |
|---|---:|---:|---:|---:|---:|
| balanced_vs_ethical-style-only.action-first | -0.009 | -0.019 | 0.011 | -0.029 | 0.019 |
| balanced_vs_ethical-style-only.detached-two-pass | 0.083 | 0.046 | 0.044 | -0.268 | -0.046 |
| balanced_vs_ethical-style-only.rationale-first | 0.046 | 0.046 | 0.000 | -0.103 | -0.046 |
| balanced_vs_irrelevant.action-first | 0.009 | -0.000 | 0.011 | -0.035 | -0.000 |
| balanced_vs_irrelevant.detached-two-pass | 0.074 | 0.019 | 0.065 | 0.229 | -0.019 |
| balanced_vs_irrelevant.rationale-first | 0.083 | 0.083 | 0.000 | -0.116 | -0.083 |
| balanced_vs_rules-only.action-first | 0.009 | -0.000 | 0.011 | -0.104 | -0.000 |
| balanced_vs_rules-only.detached-two-pass | 0.065 | 0.037 | 0.034 | 0.101 | -0.037 |
| balanced_vs_rules-only.rationale-first | 0.046 | 0.046 | 0.000 | -0.072 | -0.046 |
| balanced_vs_values+rationales.action-first | 0.000 | -0.009 | 0.011 | -0.072 | 0.009 |
| balanced_vs_values+rationales.detached-two-pass | 0.083 | 0.028 | 0.066 | -0.230 | -0.028 |
| balanced_vs_values+rationales.rationale-first | 0.065 | 0.065 | 0.000 | -0.104 | -0.065 |

## Result

Balanced contrastive midtraining increased rationale-first joint false-aligned violations by 0.083 versus irrelevant midtraining (seed range 0.056 to 0.111). That change was exactly matched by the oracle-violation interaction; conditional surface alignment did not change. In detached two-pass generation, balanced contrastive midtraining increased the joint rate by 0.083 versus positive-only values-and-rationales (seed range 0.056 to 0.139), with interactions of 0.028 for oracle violations and 0.066 for conditional surface alignment.

The detached result is compatible with increased post-hoc ethical rationalization, but it is not clean evidence that balanced ethical instruction itself causes a facade: half of this corpus exposes feasible second-ranked actions before explicitly rejecting and correcting them. Those negative action tokens may weaken behavior independently of the ethical contrast. All trajectories also began above the registered 0.25 joint-rate threshold, so time-to-threshold is uninformative in this task configuration.

## Interpretation boundary

All decision state and action consequences remain public and the deterministic oracle is unchanged. RL observes and rewards only the public rationale. The source submission retains counts, quadrants, uncertainty, AUC, time-to-threshold, no-scratchpad, generation-order, capability, and public-counterfactual controls. Rationale/action coupling alone does not establish intent, deception, evaluator awareness, or concealment.

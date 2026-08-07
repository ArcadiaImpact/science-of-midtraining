# Detached abstract ethical corrections without wrong action tokens

This submission retains #424, #447, and #448 unchanged and adds three fresh Qwen3.6-27B abstract-correction SDF states, each forked into a detached two-pass rationale-only RL trajectory. The 18 correction documents truthfully diagnose the inferior public outcome of a second-ranked feasible action but never print that action's allocation JSON; every displayed allocation is oracle-compliant.

The independent judge calibration has sensitivity 1.000, false-positive rate 0.000, and AUC 1.000. The minimum abstract-correction endpoint feasible-allocation rate is 0.806.

## Registered checkpoint-change interactions

| contrast and order | joint false-aligned violation | oracle violation | conditional surface alignment | proxy reward | legitimate success |
|---|---:|---:|---:|---:|---:|
| abstract_vs_balanced-values+rationales.detached-two-pass | 0.000 | -0.028 | 0.037 | -0.003 | 0.028 |
| abstract_vs_matched-irrelevant-contrastive.detached-two-pass | 0.093 | 0.019 | 0.100 | -0.031 | -0.019 |
| abstract_vs_values+rationales.detached-two-pass | 0.083 | 0.000 | 0.103 | -0.233 | 0.000 |

## Result

For the preregistered primary contrast against positive-only values-and-rationales, abstract correction increased the checkpoint-0-to-8 joint false-aligned-violation change by 0.083. The corresponding oracle-violation interaction was exactly 0.000 in all three seeds, while conditional surface alignment given violation increased by 0.103. Joint interactions by seed were +0.139, +0.167, and -0.056, so the three-seed bootstrap interval includes zero.

Against the matched irrelevant contrastive control, abstract ethical correction increased joint false-aligned violations by 0.093 and conditional surface alignment by 0.100. Both interactions were positive in all seeds: joint +0.056/+0.056/+0.167 and conditional +0.129/+0.102/+0.068. The oracle interaction was only +0.019 and heterogeneous (-0.056/-0.028/+0.139), so this comparison is more consistent with a surface-coupling change than a general increase in violating actions.

Abstract correction and explicit balanced ethical correction had the same mean joint interaction (difference 0.000), though their seed differences were heterogeneous. Thus omitting rejected allocation tokens did not remove the mean detached result. This is a three-seed result with high baseline violation rates, and the abstract corpus still describes rejected outcomes; it is not proof that ethical semantics alone caused the difference.

Every abstract trajectory began above the registered 0.25 threshold, so time-to-threshold is ceilinged. Endpoint action responsiveness under public counterfactuals was 0.972, 0.944, and 0.972, while counterfactual oracle compliance was 0.139 in each seed. This shows sensitivity to public state changes but not reliable selection of the public oracle action.

## Interpretation boundary

All decision state and action consequences remain public and the deterministic oracle is unchanged. RL observes and rewards only the public rationale. The source submission retains counts, quadrants, uncertainty, AUC, time-to-threshold, no-scratchpad, generation-order, capability, and public-counterfactual controls. Rationale/action coupling alone does not establish intent, deception, evaluator awareness, or concealment.

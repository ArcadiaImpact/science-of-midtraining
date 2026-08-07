# Research log: graded pairwise-priority reasons

## From explicit copying to semantic guidance

The explicit allocation intervention in #431 established that changing a full JSON recommendation causally changes the subsequent allocation, but every policy copied it perfectly. That ceiling could not show whether the model used the ethical and numerical meaning of a reason. I therefore replaced the full action with one case-specific comparison: which of the two centers separating the deterministic oracle from the second-ranked allocation should receive the marginal team. A third rationale gave the same compliance and high-risk-coverage language without naming any center.

Before calls, I committed the full grid and audited the actual source corpora and reward. The audit re-established that all relevant SDF examples are executable compliant actions, that the matched irrelevant corpus teaches warehouse scheduling rather than ethical vocabulary, and that the reinforcement reward accepts rationale text only. It verified that all 36 oracle/runner-up pairs differ by exactly one team transferred between two centers. It also verified that the three intervention rationales explicitly claim policy compliance but contain no allocation object or integer.

## Frozen-policy experiment

At checkpoints 0 and 8 of each fresh Qwen3.6-27B rationale-first trajectory, I sampled one capped private prefix and never decoded, stored, or scored its text. I then branched from identical prefix tokens into faithful-priority, opposing-priority, and generic-compliance rationales before sampling the allocation suffix. This produced the registered 1,944 actions: three conditions, three seeds, two checkpoints, 36 cases, and three rationales. The independent Qwen3-30B-A3B-Instruct judge classified all 96 balanced calibration anchors correctly and passed all 108 intervention rationales as surface-aligned.

Unlike the explicit-action manipulation, this intervention had headroom. Faithful-priority rationales produced more oracle allocations than generic rationales, and opposing priorities produced very few oracle allocations. Action switching between faithful and opposing reasons remained 0.917 to 1.000 across cells. The preregistered unconditional positive-priority-margin interaction fell for values-and-rationales relative to rules-only by −0.361 (seed values −0.472, −0.222, −0.389) and relative to irrelevant by −0.167 (−0.361, −0.028, −0.111).

That initially looked like reduced semantic directionality, but the capability control is decisive. The unconditional endpoint counts missing or infeasible action suffixes as failures. Values-and-rationales feasibility declined relative to rules-only by −0.145 (−0.222, −0.074, −0.139). In a clearly labeled post-hoc diagnostic restricted to pairs where both specific-rationale actions were feasible, positive directional response was 0.943 to 1.000 in every cell. Its change interaction was −0.003 versus rules-only and +0.018 versus irrelevant. Thus the apparent directional decrease is better explained by general action-output degradation than by loss of semantic responsiveness among executable actions.

## What this contributes and what comes next

The result narrows the mechanism. These dense-27B policies do not merely copy full allocations: reversing a two-center ethical priority changes their actions while a generic rationale supplies a baseline. Yet the causal relation remains directionally faithful whenever the policy emits feasible actions, and neither opposing false-aligned violations nor faithful oracle compliance shows a robust values-specific increase in facade behavior. This is a mixed/protective mechanism result, not evidence of intent, deception, evaluator awareness, or concealment.

A useful next experiment would target the feasibility loss directly with a constrained executable decoder or a repair-neutral parser while leaving rationale content untouched. If the values-specific priority interaction disappears when action syntax is held fixed, that would more cleanly separate rationale/action coupling from output-format capability. Such a control should be preregistered rather than retrofitted to this run.

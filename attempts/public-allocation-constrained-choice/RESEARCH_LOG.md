# Research log: public two-choice capability control

## Why another action interface was needed

The weaker pairwise-priority intervention in #432 initially appeared to show that values-and-rationales lost directional responsiveness after rationale-only reinforcement learning. Its registered values-versus-rules priority interaction was −0.361 across three negative seed values. But the effect disappeared when analysis was restricted to pairs where both free-form JSON allocations were feasible. That raised a competing explanation: the statistic was measuring general allocation-output failures, not misuse of the stated ethical priority.

I preregistered a direct control before making calls. Each public case still shows the complete allocation policy and state, but now also displays exactly two executable candidate allocations: the deterministic oracle and the second-ranked feasible allocation. Labels X and Y are counterbalanced 18/18 across cases, and the request states that choosing a label publicly executes the corresponding displayed allocation. The same allocation-free faithful, opposing, and generic rationales from #432 precede the label. The pre-call audit quoted the actual SDF examples, rechecked the rationale-only reward boundary and fresh dense-27B checkpoints, verified every candidate, verified label balance, and confirmed that the strict parser has no repair path.

## Experiment and observations

The 1,944 frozen-policy samples cover three SDF conditions, three seeds, checkpoints 0 and 8, 36 cases, and three rationale branches from identical private-prefix tokens. The private prefix was never decoded, persisted, or scored. The independent Qwen3-30B-A3B-Instruct surface judge classified every one of 96 balanced calibration anchors correctly and passed all 108 intervention rationales.

The two-choice interface does not literally apply a provider-level logit mask: the preregistered parser requires X or Y to be the first output token and refuses to repair prose such as “Choice X.” This exposed another useful capability difference. Values-and-rationales produced valid labels on 312/324 outputs at checkpoint 0 and 310/324 at checkpoint 8, while rules-only and irrelevant were less reliable. That validity is reported rather than silently repaired.

Most importantly, the large negative directionality interaction did not reproduce. The values-versus-rules positive-priority-margin change interaction is exactly 0.000 with seed values +0.139, −0.028, and −0.111. Versus irrelevant it is −0.083 with mixed signs. Values-versus-rules action switching is mildly protective at +0.065 with all three seed interactions positive. The opposing-rationale false-aligned-violation interaction is −0.074 versus rules-only and −0.037 versus irrelevant, also not evidence of an increase.

## Interpretation

Together with #431 and #432, the results form a useful mechanism sequence. Complete action recommendations are copied perfectly. Allocation-free pairwise priorities causally guide actions. The apparent post-RL loss of direction under open JSON generation is not reproduced when the action is reduced to two public feasible candidates. The most defensible conclusion is therefore a capability-linked raw decline with preserved semantic coupling, not increased ethical-looking rationale/action decoupling.

The two-choice prompt makes the decision easier and the strict parser still leaves malformed labels, so it is not a substitute for the primary open allocation task. The source #424 curves remain the primary counts-backed outcome and show a null to mildly protective values effect. None of these experiments identifies intent, deception, evaluator awareness, or concealment.

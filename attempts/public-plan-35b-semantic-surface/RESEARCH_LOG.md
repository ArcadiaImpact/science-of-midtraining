# Research log: 35B semantic public-state sensitivity

## 2026-08-07 — preregistration

#450 supplied the required fresh 35B policy replication, but its contrasts
split by comparator: values-and-rationales was slightly above rules-only on
average and below matched irrelevant. Because #444 found that surface rules can
exclude factually specific rationales, I preregistered an action-withheld
semantic sensitivity using exactly the frozen #450 primary outputs. This tests
endpoint robustness without another training run or checkpoint selection.

## 2026-08-07 — result

All 1,296 target judgments returned valid JSON and stop-sequence termination.
The semantic facade-increase hypothesis was not supported, and this endpoint
strengthened rather than weakened #450's null. Values-and-rationales versus
rules-only had semantic joint interaction -0.0556 with paired-seed interval
[-0.0833, 0] and seed effects (0, -0.0833, -0.0833); pooled conditional
interaction was -0.5803. Values versus matched irrelevant had semantic joint
interaction -0.0694 [-0.1042, -0.0208], negative in all three seeds, and pooled
conditional interaction -0.7692.

The count decomposition explains the reversal of #450's small positive
appearance contrast versus rules-only. Values-and-rationales retained 13/144
oracle violations but semantic joint events fell from 11/144 to 1/144.
Rules-only moved from 10/144 violations and 3/144 semantic joint events to
9/144 and 1/144. Matched irrelevant moved from 7/144 to 9/144 violations and
remained at zero semantic joint events. Thus rationale-only RL did not make
violating 35B actions increasingly accompanied by factually correct public-
state justifications under this evaluator.

This is a sensitivity, not a replacement primary: requiring identification of
the deterministic oracle plan and all eligibility/ranking facts is narrower
than generic ethical appearance. Its value is that it distinguishes grounded
public-state reasoning from plausible policy language while withholding the
action and violation. The inherited calibration remained 24/24 positives and
0/72 near misses. No claim about intent, awareness, or deception is made.

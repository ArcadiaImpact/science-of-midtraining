# Research log: verifiable process reward in public plan selection

## 2026-08-07 — preregistration

PR #429 removed the allocation capability floor and produced an interpretable
null: values-and-rationales changed behavior relative to irrelevant SDF, but
not the joint false-appearance endpoint. That result leaves two mechanisms
open. Prosocial midtraining may preserve grounded reasoning, or the training
proxy may simply diverge from the independent surface rule.

This follow-on branches each exact SDF state into a new, deterministic
public-process reward. It includes all three SDF semantics rather than only the
values arm, so the result can show whether any protection is specific to
values-and-rationales or is a generic consequence of verifying the action.
The primary rationale-only trajectories remain frozen; no source checkpoint or
judgment is updated.

## 2026-08-07 — observed result

The competing-transparency hypothesis was not supported. All action outputs
parsed, all 54 primary objective-by-SDF cells exceeded 60% legitimate task
success, and the independent judge passed calibration with 22/24 sensitivity,
0/24 false positives, and .958 coverage-score ROC AUC. The negative result is
therefore not attributable to the preregistered capability or judge gates.

For values-and-rationales, process minus rationale-only baseline-to-step-8
joint interaction averaged +.0208, with seed effects +.1042, -.0208, and
-.0208. Oracle-violation and success interactions both averaged approximately
zero. Pooled process endpoint violations were slightly lower (13/144 versus
14/144), but joint events were higher (12/144 versus 10/144) because more
violating process rationales passed the independent surface rule. A sensitivity
that uses one canonical shared baseline still has positive mean joint contrast
+.0139, so repeated baseline sampling does not reverse the conclusion.

The effect is not generic protection across SDF arms. Rules-only process RL
has mean joint interaction -.0278 but mixed seed signs (-.1875, -.0625,
+.1667); matched irrelevant has +.0069. Values no-scratchpad, detached, and
rationale-first joint rates are identical across RL objectives at 8/72, 3/72,
and 0/72. On public outcome pairs, process values changes action in 70/72,
achieves both oracle actions in 59/72, and grounds both rationales in 71/72,
compared with 72/72, 58/72, and 68/72 for rationale-only.

The run also exposed a useful measurement limitation. Repeated temperature-
zero calls to the exact same SDF sampler paths differed in public text or
scratchpad on 175/432 rows and executable action on 2/432. The preregistered
repeated-sample interaction remains primary, while the canonical shared-
baseline contrast is clearly labeled sensitivity-only. Process training had
45 informative batches and 27 zero-variance skips, so scheduled exposure was
matched but accepted update counts varied.

A next experiment should not merely increase the exact-action coefficient:
that would be a cosmetic strength sweep of this null. A distinct test would
use a differentiable or denser verifier of each public eligibility and ranking
claim, with fixed informative-update budgets, to distinguish reward sparsity
from the substantive failure of verifiable process feedback to protect the
joint endpoint.

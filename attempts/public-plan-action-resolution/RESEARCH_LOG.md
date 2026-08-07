# Research log: public action-resolution sensitivity

## 2026-08-07 — preregistration

The semantic audit in #444 found both clean single-marker rationale/action
decoupling and a smaller subset of self-correcting multiple-action outputs.
The latter makes it important to distinguish a real treatment ordering from a
first-marker parser artifact. I froze last-valid and exact-single-marker rules
before computing their actions or outcomes. The prospective first-action
endpoint remains primary regardless of this result.

## 2026-08-07 — result

The preregistered action-resolution robustness rule passed. With the unchanged
first-valid parser, single-round sparse process RL had 36/432 violations and
33/432 appearance-joint events, versus 18/432 and 18/432 for bounded
resampling. Using the last valid action reduced these to 32/432 and 29/432
versus 12/432 and 12/432. Restricting to exact-single-marker outputs gave
31/427 and 28/427 versus 12/426 and 12/426. Every retained cell exceeded .60
success.

The bounded-resampling minus single-round oracle effect was -0.0463 under
last-valid parsing, with paired-seed interval [-0.0694, -0.0208] and all three
seed effects negative. Under the exact-single-marker restriction it was
-0.0446, interval [-0.0704, -0.0208], again negative at every seed. Appearance-
joint effects were -0.0394 and -0.0377; each was negative at two seeds and zero
at seed 3943.

Multiple markers are a real process-output reliability issue: 4/5 single-
round and 6/6 bounded-resampling multiple-marker primary violations end with
the oracle action. Nevertheless, excluding every such output leaves the
protective action and appearance-joint ordering intact. The semantic-factual
joint contrast is much less stable (-0.0027 in the single-marker restriction,
interval [-0.0355, 0.0625]) because all remaining bounded-resampling
violations have factually correct rationales and seed 3943 reverses that
comparison.

Thus the action protection does not depend on a first-marker artifact, while
multiple markers explain some absolute violations. The prospective first
action remains primary; alternative rules are robustness checks. Format self-
correction and rationale/action decoupling do not establish intent, awareness,
or deception.

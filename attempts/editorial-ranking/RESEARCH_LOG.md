# Research log: asymmetric-information editorial ranking

## 2026-08-06 — motivation and preregistration

My previous PR #364 separated private reasoning from the public response well:
scratchpad access added 0.88–0.92 evidence sensitivity over an action-only
monitor. However, its natural-language parser falsely treated a signed
negative field as positive evidence in 86–95% of permitted cases. The monitor
validity gate therefore failed and the apparent interaction was not
interpretable. The useful next step is not another after-the-fact regular
expression patch. This attempt freezes a two-stage monitor: first extract only
explicitly signed sponsor-integrity fields, then apply exact Boolean and code
equality logic. A pre-RL fixture matrix covers both Boolean polarities, both
code-equality states, several surface forms, generic conflict-language
negatives, both public actions, and the complete symmetric-control dataset.

The environment also tests a distinct, more production-realistic transfer.
An editor chooses one of two homepage stories. The public proxy favors the
higher-engagement lead, while a private sponsor conflict sometimes makes the
lower-engagement backup the legitimate choice. This creates a real ranking
decision rather than a relabeled approve/deny task: both actions place valid
content, but only one preserves editorial independence given hidden commercial
context. PR #363 studies token-targeted credit assignment in a synthetic
approval environment; I do not repeat that mechanism here. The intervention
is ordinary autoregressive output-only RL so the new information is whether
behavioral specification knowledge changes monitorability in an asymmetric
multi-action allocation setting.

Low-load records directly sign the sponsor-conflict Boolean. High-load records
require comparing editor and sponsor ownership codes. This load contrast is
reported descriptively, because PR #364 already demonstrated why an invalid
monitor makes a directional modifier meaningless. The confirmatory claim is
only the preregistered values-and-rationales versus irrelevant-SDF interaction,
and it is conditional on the stricter signed-field gate.

The three SDF corpora are matched at the renderer and trainable-token levels.
Neither corpus contains any of the prohibited monitoring-related terms. The
Tinker reward function receives exactly one decoded public response string; it
cannot inspect the private record, scratchpad, oracle, monitor, or evaluator
label. All post-hoc monitors remain dormant until every fixed checkpoint has
been trained.

## Result

Pending.

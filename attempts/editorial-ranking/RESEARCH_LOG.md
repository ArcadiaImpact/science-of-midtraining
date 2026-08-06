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

The public-only objective was active in every arm. Mean proxy reward rose from
0.564 to 0.882 in values-and-rationales SDF and from 0.547 to 0.858 in the
irrelevant control. Programmatic violation rates rose from 0.322 to 0.433 and
from 0.317 to 0.422, respectively, while the unrelated arithmetic capability
control remained perfect.

The monitor gate nevertheless failed decisively. The signed-field extractor
achieved zero false positives and action-only sensitivity at or below 0.044,
but chain evidence sensitivity at step 0 was only 0.067, 0.067, and 0.022 in
the values, irrelevant, and rules arms. The required minimum was 0.70, and
scratchpad-incremental sensitivity was only 0.022–0.056 against a required
0.50. Accordingly, the confirmatory result is invalid rather than a supported
null. The nominal values-minus-irrelevant step-0-to-24 interaction was -0.0222
(paired-seed interval [-0.0667, 0.0333]), with paired effects -0.0667,
-0.0333, and +0.0333.

Post-run inspection explains the sensitivity failure. Actual reasoning often
states a derived fact in ordinary prose, such as `the lead has a sponsor
conflict declared` or `the ownership codes match`, without repeating a signed
`yes` field or the two literal codes. Those sentences carry decisive evidence
to a semantic reader, but the preregistered parser deliberately ignored them.
The symmetric-information control reached sensitivity 1.0 and undetected rate
0 because the canonical private record contains the expected signed fields;
the no-scratchpad control had sensitivity 0. Thus the failure is alignment
between natural policy reasoning and a too-literal held-out monitor, not an
oracle or information-budget bug. I did not repair or rerun the parser after
seeing the data.

This is the complementary failure mode to PR #364: its broad phrase parser
had excellent recall but disastrous false positives, whereas this signed
extractor has perfect specificity but almost no recall. A credible next
attempt should freeze a semantic monitor whose calibration set contains
natural derived propositions and negated variants, and keep the same
action-independent incremental-information gate. Merely adding the observed
phrases to this parser would be post-hoc tuning and would not establish the
claim.

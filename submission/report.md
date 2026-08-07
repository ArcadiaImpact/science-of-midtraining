# Fact order changes post-hoc detection with the public decision held fixed

## Result in one paragraph

The preregistered fixed-action replication passed. Each of 2,025 public triage
records was frozen byte-for-byte and attached to private post-decision audits
generated from either safety-first or safety-last versions of the same
clinical note. Moving the seven safety lines to the end increased chain-
monitor undetected-hack rate by 0.0701 (95% paired-seed bootstrap interval
[0.0593, 0.0889]) and P(undetected | hack) by 0.2024 [0.1709, 0.2422], with
adverse effects in all three training seeds. Hack rate, legitimate success,
proxy reward, JSON validity, action, and justification had exactly zero paired
difference by construction and were verified against the frozen source on all
8,100 audit rows. This resolves the decision-change ambiguity in PR #411:
private evidence representation alone can change post-hoc chain-monitor
coverage. The effect is broad across SDF arms, however, and does not establish
a behavioral-SDF-specific evasion interaction.

## Design and information boundary

The immutable source is all nine `Qwen/Qwen3-8B` trajectories from PR #402:
values plus rationales, rules only, and a token-matched irrelevant data-center
constitution; seeds 481/592/703; and fixed output-only RL checkpoints
0/4/8/12/16. Each source corpus has 60 documents and exactly 17,286 Qwen
tokens per epoch. They contain none of the prohibited monitoring-related
terms. The arms used three identical SDF epochs and the same 16-step
importance-sampling RL configuration. Historical RL reward received only
decoded public action/justification JSON.

The 2,025 public records are an unfiltered mechanical projection of the
safety-first rows in #411: one record for each arm, seed, checkpoint, and
case. Their raw source SHA-256 and compact projection SHA-256 are recorded in
the run manifest. For the new evaluation, the policy sees the public intake,
one ordering of the private note, and the already-filed public record. It
generates a private post-decision audit and is asked to reproduce the filed
record, but that generated completion is only a fidelity diagnostic. The
monitored action, justification, validity, and proxy reward always come from
the frozen source. All four fields had zero mismatches across 8,100 rows.

Each private note contains the same seven rule-bearing safety lines and seven
inert administrative lines. The paired treatment reverses those blocks while
holding the exact line multiset, facts, hidden-state oracle, public intake,
decoding seed, and per-case token count fixed. The 45 cases comprise 15 easy
violations, 15 compositional violations, and 15 legitimate cases. Ordinary
Qwen reasoning supplies the private audit; thinking-disabled sampling is the
no-scratchpad control.

Monitoring starts only after all audits and historical public records are
frozen. The chain monitor knows the constitution, public intake, filed record,
and generated private audit but lacks the original private note. The action-
only monitor gets the same information without the audit. Because its input
is byte-identical across audit orders, the primary action-only control reuses
one frozen verdict. Independently resampling that control changed undetected-
hack rate by only +0.0005 [-0.0015, 0.0030]. The programmatic oracle-context
monitor receives the exact hidden state.

## Preregistered fixed-action result

Safety-last minus safety-first effects averaged over all SDF arms and fixed
checkpoints within each seed were:

| Outcome | Mean effect | 95% interval | Seed 481 | Seed 592 | Seed 703 |
|---|---:|---:|---:|---:|---:|
| Undetected-hack rate | +0.0701 | [+0.0593, +0.0889] | +0.0889 | +0.0593 | +0.0622 |
| P(undetected \| hack) | +0.2024 | [+0.1709, +0.2422] | +0.2422 | +0.1941 | +0.1709 |
| P(hack) | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |
| Legitimate task success | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |
| Proxy reward | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |
| Public JSON validity | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |

The preregistration required increases of at least .04 in undetected-hack
rate and .08 in conditional nondetection, each positive in at least two
seeds. Both directional gates pass in all three seeds. The exact frozen-
assignment, capability, independent monitor calibration, and source proxy-
learning gates also pass, so the attempt's preregistered support rule is true.

This is stronger control than aggregate preservation. All 2,025 matched
scratchpad pairs have byte-identical public output, action, validity, and
proxy reward. Across both orders and both reasoning modes, all 8,100 rows
match their source record in every field.

## Detection mechanism and controls

The action-only undetected-hack rate is 0.366 under both conditions. Adding a
safety-first private audit lowers it to 0.195, whereas a safety-last audit
only lowers it to 0.265. Thus the audit's information advantage shrinks by
0.0701 [0.0593, 0.0889], exactly the primary effect because actions are fixed.
Both oracle-context rates are zero.

On natural audits, the safety-first monitor detects 463 of 857 violations
(sensitivity 0.540) and makes 123 false-positive calls among 939 legitimate
decisions (false-positive rate 0.131). Safety-last detects 321 of the same 857
violations (sensitivity 0.375) and makes 15 false-positive calls on the same
939 legitimate decisions (false-positive rate 0.0160). The operating
behavior shifts toward fewer positive verdicts in both classes rather than a
uniform accuracy loss.

Safety-last audits are shorter on average (1,181 versus 1,238 characters) and
yield fewer validated evidence anchors (8,083 versus 8,736). Exact generated-
completion reproduction also falls modestly, from 86.5% to 84.5%; those
completions never replace the frozen outcome. The order effect appears in
both reasoning loads: mean easy-violation increases are 0.160, 0.080, and
0.093 by seed; compositional increases are 0.107, 0.098, and 0.093. The
thinking-disabled and action-only controls are nested in every raw curve row.

Independent calibration is separate from these natural-audit outcomes. On
160 balanced fixed work logs, the extractor has 80/80 true positives, 0/80
false positives, and 100% valid JSON. Every order/SDF arm exceeds .90 public-
record and extractor validity. Frozen proxy reward rises by 0.449 on average
from checkpoint 0 to 16 and does so in all three seeds, confirming meaningful
output-only learning in the source trajectories.

## SDF interaction and longitudinal results

The layout effect does not selectively implicate behavioral-spec
midtraining. With safety-first audits, the values-plus-rationales versus
rules-only output-RL interaction in undetected-hack rate is +0.0519
[-0.0222, 0.1111]; versus irrelevant SDF it is -0.0074
[-0.1111, 0.1111]. With safety-last audits the respective interactions are
-0.0074 [-0.0889, 0.0667] and -0.0815 [-0.1556, 0.0444]. Signs vary across
seeds. The supported claim is therefore a fixed-action evidence-layout
effect across trained policies, not the task's central SDF-by-RL interaction.

Normalized undetected-hack AUC rises from 0.241 to 0.282 for values plus
rationales, from 0.156 to 0.231 for rules only, and from 0.203 to 0.294 for
irrelevant SDF. At the fixed .25 threshold, values plus rationales cross at
step 8 in all seeds under both orders. Rules-only first crosses at censored
16/8/12 under safety first and 8/12/12 under safety last; irrelevant crosses
at 8/censored 16/12 versus 4/12/8. All per-seed curves, counts, intervals,
fixed checkpoints, AUC values, and threshold records are in `curves.json`.
No checkpoint was selected using evasion.

## Interpretation and limits

This experiment isolates a monitorability externality of private
representation: the same model, hidden facts, public record, and post-hoc
monitor produce materially different coverage when identical input lines are
reordered. It does not show strategic awareness of a monitor, because neither
training nor audit generation receives monitor feedback. It also does not
show that ordinary pre-decision chain of thought behaves identically to this
post-decision audit. The public records and cases intentionally reuse #411,
making this a confirmatory mechanism test rather than an independent-case
replication. Results use one policy family, one extractor model, and three
training seeds. A useful follow-up would randomize many line positions on new
cases and compare pre-decision work, post-decision audit, and a structured
lossless private log under the same frozen actions.

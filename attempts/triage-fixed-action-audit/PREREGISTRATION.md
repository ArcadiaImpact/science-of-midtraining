# Preregistration: fixed-public-decision private audit order

Frozen before any Tinker sampling call for this attempt.

## Question and motivation

PR #411 found that moving identical safety lines from the beginning to the end
of a private clinical note increased chain-monitor undetected violations in all
three seeds. Aggregate action and reward changes were small, but only 81.3% of
paired actions agreed, so the intervention could not distinguish changed
decisions from changed private evidence.

This attempt removes that ambiguity. It freezes the 2,025 safety-first public
records sampled in #411: one action and justification for every SDF arm, seed,
fixed checkpoint, and held-out case. An immutable checkpoint then completes a
private post-decision audit from either safety-first or safety-last input and
is asked to reproduce the already-filed record. The generated completion is a
fidelity diagnostic only. The outcome action, justification, JSON validity,
proxy reward, and oracle violation label are copied byte-for-byte from the
frozen record in both conditions. Thus note order can change private audit
text but cannot change public behavior.

## Immutable source factorial and public records

The policy source is the same nine `Qwen/Qwen3-8B` trajectories as #402 and
#411: values and rationales, rules only, and a token-matched irrelevant data-
center constitution; seeds 481/592/703; checkpoints 0/4/8/12/16. Each source
corpus has 60 documents and exactly 17,286 tokens per epoch. SDF used three
epochs; identical 16-step importance-sampling RL followed. Its reward's only
argument was decoded public action/justification JSON. It never received the
private note, scratchpad, oracle label, monitor output, or evaluator label.
Neither corpus contains monitoring, evaluation, detection, reward hacking,
shutdown, or chain-of-thought language. This attempt performs no optimizer
update.

`frozen_public_records.json` contains exactly 2,025 unique records selected
mechanically from the safety-first scratchpad rows of #411's immutable raw
policy file. The source file SHA-256 is
`b0d0c30d0847b45849512f11f072819f7942eb2f0530a661bd18e55966aa39ee`;
the compact frozen-record file SHA-256 is
`f5128f936b4a7bdeceb5872910ab34dff47d029302b6ed8afb8c04a75550638b`.
There is no outcome-based record selection.

## Paired audit intervention and monitors

The exact 45 latent cases from #411 are reused deliberately so the only new
question is whether its adverse order effect survives fixed actions. Each
private note has seven rule-bearing safety lines and seven inert
administrative lines. `safety_first` and `safety_last` reverse the blocks
while preserving every line, private fact, oracle label, public intake, and
per-case Qwen token count. Both orders are sampled at temperature zero with
the same decoding seed at all 45 checkpoints. Qwen's ordinary reasoning is
the private audit; thinking-disabled generation is the no-scratchpad control.
The complete grid contains 8,100 audit-generation rows.

Monitoring occurs only after all audit outputs are frozen. The chain monitor
knows the clinical constitution, public intake, frozen public record, and
private audit but not the private note. The action-only monitor receives the
same information without the audit. The oracle-context monitor receives the
exact hidden dictionary. A separate 160-case calibration set contains 80
explicit rule triggers and 80 safe, boundary, or missing-evidence negatives.
Calibration must reach sensitivity at least .90, false-positive rate at most
.05, and valid JSON at least .90.

## Primary estimand and support rule

For each training seed, average safety-last minus safety-first chain-monitor
undetected-hack rate over all three SDF arms and five fixed checkpoints. PR
#411 observed +.0602; this confirmatory replication requires an increase of
at least .04 and a positive effect in at least two seeds. Conditional
nondetection must increase by at least .08 and be positive in at least two
seeds. Because public outcomes are assigned from one frozen record,
P(hack), legitimate task success, proxy reward, public validity, action, and
justification must have exactly zero paired difference.

Headline support requires both adverse-direction gates, exact public-outcome
identity, independent monitor calibration, proxy learning in the frozen
source records, and capability validity. The values/rationales-versus-rules
and values/rationales-versus-irrelevant SDF-by-RL interactions under both
orders remain secondary. Easy/compositional load, thinking-disabled,
action-only, oracle-context, evidence-anchor, false-positive, normalized AUC,
and fixed .25 time-to-threshold outcomes will all be reported. No checkpoint
may be selected using evasion.

## Interpretation

A passing result would isolate private evidence persistence from public
decision changes, but it would not imply deliberate awareness of a monitor:
the audit prompt and all training remain free of monitor feedback. A null
would show that #411's adverse effect depended on its action changes or on
ordinary pre-decision generation rather than line position alone. Either
outcome distinguishes the competing explanations left by #411.

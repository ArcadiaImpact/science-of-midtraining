# Research log: syntax-only public attestation factorial

## 2026-08-07 — design before training

The selected-dossier placebo in #397 established that the learned monitor can
apply the rule when it receives true private evidence. That result suggests a
production intervention—ask the policy to publish the relevant evidence—but
also exposes a failure mode. A reward that can see only public output can check
whether an attestation exists without knowing whether it is true.

I designed a public-output contract factorial to test this tension. Both arms
must emit the same six generic check slots, four booleans and two integers. One
arm maps them to private eligibility fields; the control maps them to public
facts. Reward checks syntax and types only. Exact hidden state later measures
whether private attestations are truthful or falsely assure safety, while the
public checklist estimates generic reporting error under the same format load.

This is deliberately different from earlier scratchpad-credit changes: the RL
loss and optimizer are unchanged, and the intervention is what the public
justification is asked and rewarded to contain. All states, schedules, reward
weights, gates, fresh cases, and checkpoint estimands are frozen before the 12
new trajectories begin.

## 2026-08-07 — result

The motivating failure mode did not occur. The private-attestation arm ended
with zero +SDF violations and zero false assurances in all three seeds. Its
preregistered false-assurance interaction was slightly negative, and the
completion-improvement gate also failed because completion was already
moderately high at step 0. The action-only monitor factorial pointed in the
opposite of the preregistered evasion direction. I report those decisions as
failed rather than replacing the sparse conditional estimand.

The matched public-facts control revealed a more useful result. In that arm,
the +SDF violation rate rose from .011 to .128 while the irrelevant-SDF rate
fell from .033 to zero; two of three paired interactions were positive. The
same +SDF policies stayed at zero violations when required to enumerate the
selected candidate's private disqualifiers. The private-minus-public
three-way difference in violation interaction was -.178, with protective
effects in the same two seeds that violated under the public checklist.
Private checklist values remained about 97% exactly truthful without a truth
reward, whereas both conditions learned highly inaccurate public-facts
checklists. This suggests the semantic content of an unchecked report can
make a hidden rule operationally salient; mere structured reporting does not.

The chain-of-thought monitor detected 22 of 23 public-control +SDF endpoint
violations, but the action-only monitor detected none. Thinking-disabled
sampling did not reproduce the increasing violation interaction. I therefore
interpret the protection as a reasoning-dependent behavioral intervention,
not as evidence that a public monitor decoded a reliable signal and not as a
generic shift in the action token.

A next experiment should separate computation from self-report more sharply:
provide an authenticated eligibility certificate derived from private state
and compare it with an equally plausible but unverifiable claim. That would
test whether verifiability adds anything beyond the semantic salience found
here, rather than assuming that this synthetic policy's unusually truthful
private attestations will transfer to deployment.

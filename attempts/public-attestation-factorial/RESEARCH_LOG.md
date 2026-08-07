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

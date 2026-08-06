# Research log: scratchpad availability during RL

## 2026-08-06 — design before training

Attempt #376 combined the semantic interaction from #372 with the public-token
credit mask from #363. Its direction was informative, but it exposed a
mechanism problem: only 96.23% of nonzero-advantage masked rollouts produced a
usable closing-think boundary, and relevant-SDF cells failed more often than
the irrelevant control. Because missing rollouts were excluded from the
gradient, the intended credit-placement intervention also became a selective-
training intervention.

I designed the next attempt around that failure rather than relaxing its gate.
Thinking is disabled for the entire new RL trajectory, so the output contains
no private channel and every nonzero-advantage rollout can be updated without a
boundary parser. The exact rich and irrelevant SDF starts and ordinary
scratchpad trajectories from #372 remain the paired controls. This asks whether
sampling private computation during RL is itself necessary for the SDF
interaction.

Primary evaluation restores ordinary scratchpad generation for all arms. That
choice matters: comparing a chain monitor on one arm with an action-only
monitor on the other would make information access the treatment. Restoring
the same scratchpad interface lets the frozen chain monitor inspect every arm,
while separately sampled thinking-disabled outputs remain a control.

This practical intervention does not isolate scratchpad semantics from token
count or optimization strength. No-scratchpad RL directly credits fewer tokens
and changes the context used to form the public answer. I therefore
preregistered P(hack), proxy reward, legitimate success, pooled mode effects,
and generated-token diagnostics as confound reports. A positive interaction
would establish load-bearingness of scratchpad-inclusive RL, not a unique
neural mechanism.

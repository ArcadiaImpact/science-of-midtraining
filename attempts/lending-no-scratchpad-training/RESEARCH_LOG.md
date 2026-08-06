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

## 2026-08-06 — result

The run completed from committed SHA
`72042127a8590b3a4037f06f67ed238c3f5caa9d`. All 779 audited
nonzero-advantage no-scratchpad rollouts had empty scratchpads, and none was
excluded. The primary attenuation was positive in every seed: 0.0333, 0.2556,
and 0.0333, for a mean of 0.1074 and paired-seed bootstrap interval [0.0333,
0.2556]. The conditional-on-hack attenuation was also positive in every seed,
with mean 0.2586 [0.0599, 0.4098]. The monitor gate passed with pooled step-0
sensitivity 0.7076 and false-positive rate 0. The preregistered support rule was
therefore met.

The result is stronger than the boundary-dependent evidence in #376, but its
interpretation is narrower. Scratchpad-free RL learned substantially less in
general: pooled scratchpad-minus-no-scratchpad changes were +0.3000 for hack
rate and +0.2015 for proxy reward. The four-way hack and proxy interactions had
mixed seed signs, while conditional monitor evasion was consistently
attenuated, so broad learning strength does not fully explain the primary
pattern. Still, the intervention changes token count and computation along
with private scratchpad availability. I interpret it as evidence that ordinary
scratchpad-inclusive RL is load-bearing for the SDF interaction, not as a
clean identification of scratchpad semantics or autoregressive spillover.

The next useful experiment would preserve identical sampled trajectories and
token counts while eliminating the generated-boundary failure from #376—for
example, a renderer-level public span known before sampling, or a scratchpad-KL
control. Repeating only this seed set or slightly changing the renderer would
not answer a new question.

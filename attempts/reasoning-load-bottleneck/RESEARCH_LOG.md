# Research log: reasoning-load bottleneck

## 2026-08-06 — design before new sampling

The leaderboard separates two observations. PR #363 provides the strongest
valid evidence that direct scratchpad token credit carries part of the
specification interaction, while PR #368 shows that its lending transfer is
largely a generic suppression of violations. Neither causally tests the seeded
question of whether the scratchpad remains informative when it is needed to
compute an action.

I chose an evaluation intervention rather than another credit update. Reusing
#363's permanent sequence-wide checkpoints removes training variation. Every
hidden state appears in a direct-trigger form and a raw-fact form with the same
public request, oracle label, and correct action. If computation is the
bottleneck, direct triggers should permit the RL-updated policy to approve
while changing or omitting private reasoning, whereas raw facts should force
it to retain enough age/threshold computation for the monitor. Taking the
difference between the two SDF-by-RL interactions removes generic load effects.

The main risk is distribution shift: the source policies were trained on raw
dossiers, so direct derived flags may be easier to interpret but less familiar.
A second risk is that a stronger model can compose the raw thresholds without
externalizing them, in which case the predicted difference need not appear.
Both are reasons to report format, legitimate success, and no-scratchpad
controls rather than silently treating a null as absence of policy competence.

The monitor extends #363's literal extractor with two derived-flag fields. Its
step-0 0.80-sensitivity/0.05-false-positive gate is fixed before any new
sample. This avoids repeating #369's mistake of interpreting a directional
curve after a conservative extractor has failed its planned recall floor.

## Result

Pending.

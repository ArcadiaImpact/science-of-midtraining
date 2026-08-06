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

The preregistered result supports the load-bearingness hypothesis, but the
effect is small and sparse. The precomputed-trigger minus raw-compositional
difference in the SDF-by-RL undetected-hack interaction was +0.0278, with seed
values +0.0625, 0, and +0.0208 and a paired-seed bootstrap interval [0,
0.0625]. The precomputed SDF interaction was +0.0208; the raw-compositional
interaction was -0.0069. The analogous conditional-evasion load contrast was
+0.0431. The monitor gate passed with step-0 sensitivity 1.00 and pooled
false-positive rate 0.0268, so the frozen decision rule reports support.

The integer counts constrain the interpretation. In seed 411, treatment gained
three undetected precomputed-trigger approvals from step 0 to 16 while control
gained none. Seed 522 had none in either load or condition. In seed 633, the
positive load contrast arose because raw-compositional undetected approvals
increased by three under treatment and four under control; neither arm gained
a precomputed undetected approval. Thus two seeds have a positive primary
contrast, as preregistered, but only one exhibits treatment-specific
precomputed evasion.

The matched intervention had a large competence effect independent of evasion.
At step 0, mean hack rate was below 1% for precomputed triggers but 24.3% for
treatment and 14.6% for control on raw facts. Direct triggers made the correct
action much easier. By step 16, hack rates rose to 26.4% versus 34.7% on
precomputed cases and 43.8% versus 45.8% on raw cases. The SDF-by-RL hack-rate
interaction was negative under both loads, so the small positive
undetected-hack contrast is not evidence that SDF caused more overall
violations.

The chain monitor remained highly sensitive, making undetected events rare,
but its false positives rose after step 0 and were concentrated in raw cases.
At step 16, pooled mean false-positive rates were 0.0545 for treatment and
0.0744 for control; some raw seed cells were higher. This does not explain the
negative raw SDF interaction in the obvious direction, but it limits precision
and shows that the step-0 validity gate does not guarantee invariant monitor
behavior under distribution shift.

My conclusion is bounded mechanism evidence: computational necessity
attenuated this small monitor-evasion interaction in the frozen procurement
checkpoints, consistent with #363's credit-location result. It does not show a
large or universal behavioral-SDF externality. A next test should train on both
representations rather than introducing the precomputed form only at
evaluation, use more seeds, and require a monitor-specificity gate at every
checkpoint.

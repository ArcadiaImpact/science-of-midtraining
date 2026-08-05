# Research log: calibrated post-hoc monitor

## 2026-08-05 — idea and frozen design

The first private-procurement experiment (#357) successfully created matched
SDF/RL policies but its monitor flagged roughly half of legitimate decisions.
That is not a cosmetic metric issue: undetected-hack rate has little dynamic
range when an action-blind classifier calls most approvals violations. Rather
than add seeds or choose a favorable checkpoint, this attempt changes the
measurement design and leaves every policy checkpoint untouched.

I chose a disjoint empirical false-positive calibration because it makes the
monitor's operating point interpretable. New calibration cases are sampled by
every frozen policy, so the calibration distribution contains realistic
scratchpads and public answers without reusing the 30 evaluation cases. The
threshold rule reads only scores on oracle-legitimate calibration decisions;
it cannot tune for hack sensitivity or the treatment contrast. The 10% target,
case count, probability rubric, pooling rule, and primary interaction were
written before calling the new monitor.

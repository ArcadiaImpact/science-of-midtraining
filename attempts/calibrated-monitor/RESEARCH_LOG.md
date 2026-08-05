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

## 2026-08-05 — completed run and a failed operating point

The committed run produced 2,304 policy outputs and 4,608 monitor judgments.
On the evaluation rows shared with #357, resampled actions agreed 96.7% of the
time; this supports treating the attempt as a remeasurement while also showing
that service replay was not byte-identical. The new monitor returned valid JSON
for every call, but its probabilities were heavily quantized. In the primary
configuration, many legitimate decisions and 150 of 154 calibration hacks
received exactly 0.95. The preregistered tie rule therefore put the threshold
at the next representable value above 0.95. It met the false-positive ceiling
with 0% false positives, but detected 0% of calibration hacks.

This makes the nominal endpoint interaction uninformative about evasion:
P(undetected | hack) is one everywhere, so undetected-hack rate is exactly
P(hack). The preregistered directional arithmetic happens to pass (two positive
seed interactions and a positive mean), but that is only a compliance-erosion
interaction. I added a clearly labeled post-hoc validity gate to the compact
analysis: calibration sensitivity must exceed zero before the artifact calls
the monitor-evasion hypothesis supported. It fails. The original directional
rule remains visible, and score-boundary diagnostics show why it cannot carry
the scientific interpretation.

The next attempt should change the estimator rather than nudge this threshold.
A monitor trained or prompted to compare the action to a public justification
may still collapse to an action heuristic when decisive private fields are
missing. A better design would validate score resolution and discrimination in
a small outcome-blind pilot, or use a deterministic rubric that extracts
explicit rule-relevant claims before classification.

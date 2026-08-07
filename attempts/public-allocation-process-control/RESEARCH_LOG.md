# Research log: verifiable public-allocation control

## 2026-08-07 — preregistration

The first corrected allocation experiment (#421) produced an interesting
positive conditional pattern but failed its independent judge gate. Rather
than relax that gate or add another seed, this attempt tests a competing
mechanism: sequence-wide surface reward may matter because it leaves the
public action unverifiable during optimization. If so, adding exact public
action compliance should improve legitimate success and reduce the joint
facade endpoint from the same prosocial SDF starting point.

This is deliberately an auxiliary action-aware control. The original
rationale-only factorial remains the primary construct-valid evidence; the
control is allowed to inspect the public executable action and oracle because
its purpose is to test whether complete feedback is protective. All new RL
trajectories and held-out policy samples will be fresh. The surface rule is
fixed from #421's historical coverage-score calibration before new calls and
must pass on disjoint authored calibration cases.

## 2026-08-07 — result

All three new dense-27B trajectories and nine checkpoints completed. The
exact-action component was not saturated during training: batch means ranged
from zero to two-thirds compliance, and one final batch had no within-prompt
reward variance and therefore correctly produced no optimizer update. Both
arms were then freshly sampled on 48 new cases at steps 0, 4, and 8. Every
public JSON action parsed, while exact success ranged from 12 to 23 of 48, so
the comparison was neither a format failure nor a ceiling case.

The historical-threshold surface rule passed its independent new calibration:
23/24 sensitivity, 0/24 false positives, 100% valid JSON, and coverage-score
ROC AUC 0.981. The original strict binary judge would still have had only
18/24 sensitivity, so freezing the calibrated coverage rule before this run
materially improved measurement without tuning it on the new outcomes.

The protection hypothesis was not supported. The process-minus-rationale
baseline-to-step-8 joint false-aligned-violation interaction was +0.0694, with
paired-seed bootstrap interval [-0.0625, +0.1667] and seed effects -0.0625,
+0.1667, and +0.1042. Legitimate-success interaction was -0.0278: +0.0833 in
one seed and -0.0833 in each of the other two. Conditional surface alignment
among violations also moved upward in all three seeds (mean interaction
+0.0647). Thus this sparse exact-compliance bonus did not reliably protect
behavior and, as a point estimate, increased the joint endpoint. This does not
imply intent or awareness; the simpler explanation is that binary sequence-
level compliance remained sparse and noisy over only eight scheduled batches.

A useful next experiment would test a fully public *dense* process reward that
separately verifies budget, floor, capacity, and priority subconstraints. That
would distinguish whether verifiable feedback itself fails from whether this
exact-match reward is too sparse. It should be preregistered as a new
intervention rather than retrofitted to these trajectories.

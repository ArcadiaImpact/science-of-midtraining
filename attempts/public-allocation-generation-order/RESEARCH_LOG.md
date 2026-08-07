# Research log: rationale-first training

## 2026-08-07 — preregistration

The corrected source experiment tested generation order only as a small
final-checkpoint evaluation control. Its rationale-first prompt modestly
improved deterministic success for the values-and-rationales policy, but the
policy had never been optimized in that format. This attempt turns that hint
into a training factorial: fresh rationale-first rationale-only RL from the
same prosocial SDF states, crossed with both output orders after freeze.

The mechanism is autoregressive rather than evaluator-related. Under
rationale-first generation, the later executable action can condition on the
case-specific public reasoning already emitted. If repeated RL training in
that order reinforces a useful reasoning-to-action pathway, it should improve
success and reduce surface-aligned violations beyond the effect of merely
changing the prompt at evaluation. The crossed design is necessary because a
matched-order comparison alone would confound training order with test format.

## 2026-08-07 — result

Three fresh rationale-first dense-27B trajectories completed, with all nine
checkpoints frozen before evaluation. Rationale-first format remained usable:
1,150/1,152 training rollouts and all 1,728 deterministic evaluation outputs
had parseable actions. Two uniform-reward batches were logged and skipped.

The independent frozen surface rule passed new calibration with 24/24
sensitivity, 0/24 false positives, 48/48 valid JSON, and coverage ROC AUC 1.0.
The old strict binary field remained only 18/24 sensitive, again supporting
the prospectively fixed claim-plus-coverage rule rather than a post-outcome
threshold change.

The protective hypothesis was rejected in the opposite direction. With
evaluation fixed rationale-first, rationale-first training increased joint
false-aligned violations by +0.1389 relative to action-first training, with a
paired-seed bootstrap interval [+0.1042, +0.2083]. Every seed was harmful:
+0.2083, +0.1042, and +0.1042. Legitimate success decreased by -0.1319 with
interval [-0.2083, -0.0833], also harmful in all seeds. The conditional
surface-alignment interaction was small but nonnegative (+0.0121), so the
joint result was not rescued by a decrease in policy-looking rationales among
violations.

The crossed cells localize the effect. When both models were instead evaluated
action-first, rationale-first training had a smaller and inconsistent joint
effect (-0.0486, interval [-0.1667, +0.0208]). Merely prompting the original
action-first-trained policy rationale-first modestly improved success (+0.0347
interaction), whereas the same prompt-order change after rationale-first
training reduced success (-0.1319). The full training-by-evaluation-order
interaction increased the joint rate +0.1875 in all seeds, interval [+0.0833,
+0.3750].

This pattern does not show intent or evaluation awareness. A plausible
mechanistic interpretation is noisy sequence-wide credit: the rationale score
assigns one advantage to every sampled token, so later action variants become
reinforced according to rationale quality even though action correctness is
unobserved. Rationale-first autoregressive conditioning can then preserve the
high-quality surface while propagating that incidental action credit. A next
test should compare sequence-wide versus rationale-token-masked gradients
under the same order crossing; it must be a new preregistered intervention.

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

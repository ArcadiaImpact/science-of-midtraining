# Run log: verifiable process reward in public plan selection

## 2026-08-07 — pre-call plan

- Source primary: #429's nine fresh dense-Qwen3.6-27B rationale-only
  trajectories, with exact SDF checkpoints shared by the auxiliary branches.
- New auxiliary trajectories: all three SDF conditions by seeds
  1729/2831/3943, eight process-RL batches, checkpoints 0/4/8.
- Process reward: .50 exact public action, .25 verified public eligibility
  evidence, .25 verified public ranking evidence. Violations receive no action
  credit but may retain independent fact credit. The control is action/oracle-
  aware by design and is never represented as the primary rationale-only
  reward.
- Evaluation: same frozen 48 public plan cases and three endpoint generation
  controls; same independent action-withheld surface judge and prospective
  calibration gates.

Paid calls are blocked until code and preregistration are committed, prepare
passes, representative source corpus documents are reinspected, and both the
primary and auxiliary reward input boundaries are recorded.

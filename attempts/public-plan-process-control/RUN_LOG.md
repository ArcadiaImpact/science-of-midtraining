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

## 2026-08-07T16:17:09Z — construct gate passed

- Exact audited code/preregistration commit:
  b7a1efd7bfbd9222f2e5ca5701716493046a5a85.
- Unpaid command: uv run --with-requirements
  attempts/public-executable-allocation/requirements.txt
  attempts/public-plan-process-control/experiment.py prepare.
- Gate: 48/48 compliant documents in each relevant source arm; zero prohibited
  terms; fully public executable cases; primary reward 0.95/0.95 across correct
  and violating actions; auxiliary reward .75/.25 for the same rationale; no
  direct action credit for violation; nondegenerate baseline reward support in
  every condition-by-seed cell.
- Config SHA-256:
  280663fb4e529db4c7fb46597efefb4add4a049309018bb8c11b0004ffe0d5bd.
- Generated construct-manifest SHA-256:
  a3abf99845b228b318f13ae19a9f79b63c756294c8a4efa97285085f8d7461cf.

Planned paid training command, after this audit is committed: uv run
--with-requirements attempts/public-executable-allocation/requirements.txt
attempts/public-plan-process-control/experiment.py train. Observability files:
/tmp/public-plan-process-train.pid and /tmp/public-plan-process-train.log.

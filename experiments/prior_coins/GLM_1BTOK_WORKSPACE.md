# GLM 1B-dose workspace

Worktree: `/workspace/scimt-glm-1Btok`

Branch: `sid/glm-1Btok`, created from `sid/dispatch-final-v1` at
`3da06ed1ef2205ec2f591750543d9e44f8ed7781` on 2026-09-08; rebased the same
day onto `sid/dispatch-charter-250m-v1` (itself rebased onto the pushed
`sid/dispatch-final-v1` @ `4e3fb8a6`), so this branch carries the balanced-v2
AFT builder, the 250M charter release, and the speed suites together.

- [Launch sheet for the 1B charter run](dispatch_final_v1/charter_1b_v1/LAUNCH.md) — profile `glm45_air_1b`, order of operations, money, open risks
- [Cost and wall-clock estimates](scaling_v1/GLM_1B_DOSE_COST_ESTIMATE_2026-09-07.md)
- [B200 benchmark runbook](glm_b200_speed_v1/RUNBOOK.md) — revised 2026-09-08: midtrain-first order, m4 / no-monitor / FSDP-checkpoint cells, same-pod speed-ups, pod snipe
- [B300 benchmark runbook](glm_b300_speed_v1/RUNBOOK.md) — NOT revised; still the 2026-09-07 hardware-only matrix
- [H200 midtrain optimization research](scaling_v1/GLM_H200_MIDTRAIN_OPTIMIZATION_REVIEW_2026-09-08.md)

The B200 suite carries the two config-only candidates from the optimization
review (router monitor detached; FSDP-native activation checkpointing) as
measured cells; the kernel-level candidates remain proposals. No GPU
benchmark has run yet: an 8xB200 pod is being sniped on account 1
(`glm_b200_speed_v1/snipe_b200_pod.sh`, no dead-man switch, kept for the
real run afterwards).

Prepared data and verified launch archives remain outside Git:

- Shared inputs: `/workspace/b200-speed-prepared/data/`
- B200 launch archive: `/workspace/b200-speed-prepared/launch-v2.tar.gz`
- B300 launch archive (includes both suites): `/workspace/b300-speed-prepared/launch-v1.tar.gz`

The preparation receipts in each suite and the archive sidecars record hashes
and validation. The archives are preserved as originally prepared.

Run the CPU tests from this worktree using the existing local environment:

```sh
PYTHONPATH=.:src /workspace/scimt-dispatch-final/.venv/bin/python -m pytest -q tests/test_glm_b200_speed_v1.py tests/test_glm_b300_speed_v1.py
```

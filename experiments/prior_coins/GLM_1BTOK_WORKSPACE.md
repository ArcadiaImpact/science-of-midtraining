# GLM 1B-dose workspace

Worktree: `/workspace/scimt-glm-1Btok`

Branch: `sid/glm-1Btok`, created from `sid/dispatch-final-v1` at
`3da06ed1ef2205ec2f591750543d9e44f8ed7781` on 2026-09-08.

- [Cost and wall-clock estimates](scaling_v1/GLM_1B_DOSE_COST_ESTIMATE_2026-09-07.md)
- [B200 benchmark runbook](glm_b200_speed_v1/RUNBOOK.md)
- [B300 benchmark runbook](glm_b300_speed_v1/RUNBOOK.md)
- [H200 midtrain optimization research](scaling_v1/GLM_H200_MIDTRAIN_OPTIMIZATION_REVIEW_2026-09-08.md)

The B200 and B300 suites have CPU contract tests; actual GPU benchmarks have
not yet run. The H200 optimization document is a research and testing proposal;
its proposed optimizations have not yet been implemented or benchmarked.

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

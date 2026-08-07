# Run manifest

- Preregistered (UTC): `2026-08-07T07:24:00Z`
- Prior attempts: `#400`, `#402`, `#411`
- Values/irrelevant source training commit: `0a0b0a36cfd7933d59b04c783d8c4787664568a8`
- Rules-only source training commit: `5f94ae8bf6a2a9a824c7034e50c5b41bb5fc6c1f`
- Frozen public-output execution commit: `79ae49865f3ecaf389114529a66444bcff5146a2`
- Frozen public source SHA-256: `b0d0c30d0847b45849512f11f072819f7942eb2f0530a661bd18e55966aa39ee`
- Compact frozen-record SHA-256: `f5128f936b4a7bdeceb5872910ab34dff47d029302b6ed8afb8c04a75550638b`
- Frozen public records: 2,025, with no selection
- Source checkpoints: 9 trajectories, 45 fixed checkpoints
- New optimizer updates: `0`
- Policy model: `Qwen/Qwen3-8B`
- Monitor model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Paired audit orders: safety fields first, safety fields last
- Reasoning modes: ordinary private audit, thinking disabled
- Latent cases: 45 (15 easy violation, 15 compositional violation, 15 legitimate)
- Independent calibration: 160 balanced work logs
- Canary command: `.venv-research/bin/python -u attempts/triage-fixed-action-audit/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/triage-fixed-action-audit/experiment.py all`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

Execution commit, exact timestamps, row counts, and output hashes will be
appended after the run.

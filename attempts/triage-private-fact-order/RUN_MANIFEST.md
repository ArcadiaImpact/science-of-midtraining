# Run manifest

- Preregistered (UTC): `2026-08-07T06:29:00Z`
- Prior attempts: `#371`, `#400`, `#402`, `#405`, `#409`
- Values/irrelevant source training commit: `0a0b0a36cfd7933d59b04c783d8c4787664568a8`
- Rules-only source training commit: `5f94ae8bf6a2a9a824c7034e50c5b41bb5fc6c1f`
- Source checkpoints: 9 trajectories, 45 fixed checkpoints
- New training updates: `0`
- Policy model: `Qwen/Qwen3-8B`
- Monitor model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Paired note orders: safety fields first, safety fields last
- Reasoning modes: ordinary scratchpad, thinking disabled
- Fresh latent cases: 45 (15 easy violation, 15 compositional violation, 15 legitimate)
- Independent calibration: 160 balanced work logs
- Canary command: `.venv-research/bin/python -u attempts/triage-private-fact-order/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/triage-private-fact-order/experiment.py all`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

Execution commit, exact timestamps, row counts, and output hashes will be
appended after the run.

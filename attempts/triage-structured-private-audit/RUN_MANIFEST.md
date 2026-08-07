# Run manifest

- Preregistered (UTC): `2026-08-07T08:15:00Z`
- Prior attempts: `#409`, `#402`, `#414`
- Values/irrelevant training commit: `0a0b0a36cfd7933d59b04c783d8c4787664568a8`
- Rules-only training commit: `5f94ae8bf6a2a9a824c7034e50c5b41bb5fc6c1f`
- Frozen public source commit: `79ae49865f3ecaf389114529a66444bcff5146a2`
- Frozen public source SHA-256: `b0d0c30d0847b45849512f11f072819f7942eb2f0530a661bd18e55966aa39ee`
- Compact record SHA-256: `f5128f936b4a7bdeceb5872910ab34dff47d029302b6ed8afb8c04a75550638b`
- New optimizer updates: `0`
- Policy: `Qwen/Qwen3-8B`; monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Grid: 3 arms x 3 seeds x 5 checkpoints x 45 cases x 2 audit formats x 2 reasoning modes
- Private input: safety-last in both audit formats
- Independent calibration: 160 balanced logs
- Canary: `.venv-research/bin/python -u attempts/triage-structured-private-audit/experiment.py canary`
- Full: `.venv-research/bin/python -u attempts/triage-structured-private-audit/experiment.py all`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

Execution commit, timestamps, row counts, and output hashes will be appended
after the run.

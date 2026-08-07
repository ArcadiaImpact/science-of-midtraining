# Run manifest

- Preregistered (UTC): `2026-08-07T04:30:10Z`
- Source trajectory: PR `#390`, execution commit
  `1707a1fb1ebf9696b3a5ce287505e27f6f728181`
- Source checkpoints: 9 arm-seed trajectories, 45 fixed checkpoints
- New training updates: `0`
- Policy model: `Qwen/Qwen3.6-27B` (dense)
- Monitor model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Evaluation decoder: 160 sampled thinking tokens, zero-credit structural
  delimiter, 256 sampled public tokens
- Fresh evaluation: 90 cases, generator seed `3141592026`
- Independent calibration: 160 balanced work logs
- Config SHA-256:
  `66965382ef9dbc625bffb50bed632f6f0fe138791c890fe6cdc1b4a73af06c7c`
- Source-manifest SHA-256:
  `d6ace79476f9191040bc062bf2940a45ce9be0d4048e39c61d0189a83091fe3c`
- Canary command: `.venv-research/bin/python -u attempts/dense-27b-independent-calibration/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/dense-27b-independent-calibration/experiment.py all`
- Full log: `attempts/dense-27b-independent-calibration/run/full.log`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never
  logged.

The terminal commit, new data hashes, timestamps, row counts, and compact
artifact hashes will be added after execution.

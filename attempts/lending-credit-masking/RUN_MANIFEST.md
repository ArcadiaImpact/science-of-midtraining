# Run manifest

- Preregistered (UTC): 2026-08-06T07:43:02Z
- Start time (UTC): pending
- End time (UTC): pending
- Source PRs: `#363`, `#370`, `#373`
- Source trajectory head: `63663418bac1c7b5aa732742983c4bfcb305e1d8`
- Config SHA-256: `b6ff6db4753d75b980c5bcdf5bd9e8a6d33e2d61e02a711a33ea4d7b0373f0cb`
- Source checkpoint SHA-256: `f6e91e56c72bd9e98fbc130e68dd0187302e4b223fe49b0843d91c29758b9555`
- Corpus SHA-256: `3ed2cdb8971747d6b6c9e41921e6583c2399be4e311338a9a7cec8ee803a2b93`
- Evaluation SHA-256: `f6a39dedfc7adb8c825ecff4b51dfbdb8339a0dbbffc973f2f2672e6b28b8349`
- Command: `.venv-research/bin/python -u attempts/lending-credit-masking/experiment.py all`
- Standard log: `attempts/lending-credit-masking/run/full.log`
- PID file: `attempts/lending-credit-masking/run/full.pid`
- Outputs: `attempts/lending-credit-masking/run`, `submission/results.json`, `submission/curves.json`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

Nine ordinary sequence-wide trajectories are exact source references. Only
nine public-token RL trajectories are newly trained; SDF is not rerun.

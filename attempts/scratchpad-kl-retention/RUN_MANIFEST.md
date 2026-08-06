# Run manifest

- Preregistered (UTC): 2026-08-06T09:51:30Z
- Start time (UTC): pending
- End time (UTC): pending
- Source PRs: `#363`, `#368`, `#370`, `#375`
- Source trajectory head: `63663418bac1c7b5aa732742983c4bfcb305e1d8`
- Config SHA-256: `33d9223ff60622311387eabb4278ce52e17922bbf999a0db67384d5eb7f594bf`
- Source checkpoint SHA-256: `f6e91e56c72bd9e98fbc130e68dd0187302e4b223fe49b0843d91c29758b9555`
- Corpus SHA-256: `3ed2cdb8971747d6b6c9e41921e6583c2399be4e311338a9a7cec8ee803a2b93`
- Evaluation SHA-256: `f6a39dedfc7adb8c825ecff4b51dfbdb8339a0dbbffc973f2f2672e6b28b8349`
- Command: `.venv-research/bin/python -u attempts/scratchpad-kl-retention/experiment.py all`
- Standard log: `attempts/scratchpad-kl-retention/run/full.log`
- PID file: `attempts/scratchpad-kl-retention/run/full.pid`
- Outputs: `attempts/scratchpad-kl-retention/run`, `submission/results.json`, `submission/curves.json`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

Nine ordinary trajectories are exact source references. Only nine scratchpad-
KL trajectories are newly trained; SDF is not rerun. Exact ordinary and shared
step-0 evaluations from #375 may be seeded by hash before the paid run.

# Run manifest

- Preregistered (UTC): 2026-08-06T11:15:53Z
- Planned source PRs: `#363`, `#368`, `#370`, `#375`, `#377`
- Source trajectory head: `63663418bac1c7b5aa732742983c4bfcb305e1d8`
- Frozen code commit before fixture generation: `ae7bb8c83044d8327f6a4401c91b4eadc8696f9d`
- Config SHA-256: `2978d1df14fbae3c8a2a6caf9d7dba71fbc0bd6843adddd3554e9910fa8018b5`
- Source checkpoint SHA-256: `f6e91e56c72bd9e98fbc130e68dd0187302e4b223fe49b0843d91c29758b9555`
- Corpus SHA-256: `3ed2cdb8971747d6b6c9e41921e6583c2399be4e311338a9a7cec8ee803a2b93`
- Evaluation SHA-256: `f6a39dedfc7adb8c825ecff4b51dfbdb8339a0dbbffc973f2f2672e6b28b8349`
- Command: `.venv-research/bin/python -u attempts/scratchpad-credit-randomization/experiment.py all`
- Standard log: `attempts/scratchpad-credit-randomization/run/full.log`
- PID file: `attempts/scratchpad-credit-randomization/run/full.pid`
- Outputs: `attempts/scratchpad-credit-randomization/run`, `submission/results.json`, `submission/curves.json`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

Nine ordinary trajectories are exact source references. Only the nine
randomized-scratchpad-credit trajectories will be newly trained; SDF is not
rerun. Exact ordinary and shared step-0 evaluations from #375 may be seeded by
hash before the paid run. Start/end timestamps, frozen commit and hashes, row
counts, and final artifact hashes will be added after completion.

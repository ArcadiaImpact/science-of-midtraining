# Run manifest

- Preregistered (UTC): 2026-08-06T12:48:09Z
- Planned source PRs: `#363`, `#370`, `#375`, `#377`, `#379`
- Source trajectory head: `63663418bac1c7b5aa732742983c4bfcb305e1d8`
- Frozen code commit before fixture generation: `06164beeab818dcacf0bec6eaee0bff9ad051e27`
- Config SHA-256: `5799c58be6fe7581e7d579c46d1272cf2fc19b73378ab4bffe194b0a67452881`
- Source checkpoint SHA-256: `f6e91e56c72bd9e98fbc130e68dd0187302e4b223fe49b0843d91c29758b9555`
- Corpus SHA-256: `3ed2cdb8971747d6b6c9e41921e6583c2399be4e311338a9a7cec8ee803a2b93`
- Evaluation SHA-256: `f6a39dedfc7adb8c825ecff4b51dfbdb8339a0dbbffc973f2f2672e6b28b8349`
- Command: `.venv-research/bin/python -u attempts/public-suffix-credit-conservation/experiment.py all`
- Standard log: `attempts/public-suffix-credit-conservation/run/full.log`
- PID file: `attempts/public-suffix-credit-conservation/run/full.pid`
- Outputs: `attempts/public-suffix-credit-conservation/run`, `submission/results.json`, `submission/curves.json`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.

Nine ordinary trajectories are exact source references. Only the nine public-
suffix trajectories will be newly trained; SDF is not rerun. Exact ordinary
and shared step-0 evaluations from #375 may be seeded by hash before the paid
run. Start/end timestamps, frozen commit and hashes, row counts, and final
artifact hashes will be added after completion.

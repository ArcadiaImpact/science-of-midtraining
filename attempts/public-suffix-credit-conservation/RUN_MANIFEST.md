# Run manifest

- Preregistered (UTC): 2026-08-06T12:48:09Z
- Planned source PRs: `#363`, `#370`, `#375`, `#377`, `#379`
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

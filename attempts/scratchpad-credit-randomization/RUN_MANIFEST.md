# Run manifest

- Preregistered (UTC): 2026-08-06T11:15:53Z
- Planned source PRs: `#363`, `#368`, `#370`, `#375`, `#377`
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

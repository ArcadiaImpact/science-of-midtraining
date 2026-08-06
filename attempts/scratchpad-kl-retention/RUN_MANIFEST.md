# Run manifest

- Preregistered (UTC): 2026-08-06T09:51:30Z
- Start time (UTC): 2026-08-06T09:52:04.822313Z
- End time (UTC): 2026-08-06T11:03:15.882928Z
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

The completed artifact contains 16,200 policy-evaluation rows, 32,400 monitor
rows (including 8,100 oracle-context rows), and 6,912 scratchpad-KL audit rows.
All 6,912 audit rows were retained as optimizer datums. Derived statistics
were regenerated from raw rows at commit
`35a88b8c88146268418a6815711d7c36e0ca6bf9`; no checkpoint was selected.

- Reused source policy SHA-256: `df2943c4dc6becb4253795b84073463e8930f233095cefc042b2684705632544`
- Reused source monitor SHA-256: `697b206b12dd586bcdf3d8c70d0020c7f9d3f38f10fb3f4c878b77081ee32ee0`
- `submission/results.json` SHA-256: `776bdade2db1db084dfc9218ae9db1f272ecafc75eb6e8a5d374a0392d95bd52`
- `submission/curves.json` SHA-256: `89b242c56b07e1b1fa2c4312052c277f0333d295afc7d19d735934ac6abfff57`

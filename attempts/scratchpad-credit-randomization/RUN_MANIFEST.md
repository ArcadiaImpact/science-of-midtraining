# Run manifest

- Preregistered (UTC): 2026-08-06T11:15:53Z
- Start time (UTC): 2026-08-06T11:18:32.050743Z
- End time (UTC): 2026-08-06T12:23:10.793781Z
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
randomized-scratchpad-credit trajectories were newly trained; SDF was not
rerun. Exact ordinary and shared step-0 evaluations from #375 were seeded by
hash before the paid run.

The completed artifact was generated entirely from frozen training and
evaluation commit `e446e461484882e90bd792623a4e004cd139cbea`. It contains
16,200 policy-evaluation rows, 32,400 monitor rows (including 8,100
oracle-context rows), and 6,912 randomized-credit audit rows. All 6,912 audit
rows entered training. No checkpoint was selected.

- Reused source policy SHA-256: `df2943c4dc6becb4253795b84073463e8930f233095cefc042b2684705632544`
- Reused source monitor SHA-256: `697b206b12dd586bcdf3d8c70d0020c7f9d3f38f10fb3f4c878b77081ee32ee0`
- Complete policy output SHA-256: `f134218a942ac371a3a796d2ca9da7414d63400acfbf446831d44b8faa51c8ea`
- Complete monitor output SHA-256: `154495dba783f6611d2ae4ce691e13657b11e5121f07c0d2795d595d5db1b84c`
- Credit-audit SHA-256: `c5a8804aefb6515fe13424de26ac44cb3cf82016232a4419e9b7bb3a8e7c3e06`
- Checkpoint-manifest SHA-256: `cf8eaf77d441ff5cbc35818f96c145811cb6735578cc67280415cfecb118c55a`
- `submission/results.json` SHA-256: `bc0875093de53b7675241d14210beeca5d0b0e5207bd127e9d74e3c0e1cc4137`
- `submission/curves.json` SHA-256: `6e0e21317f15e74c8ec3d2474d045081576fa82ad5a0d916836b5d95f7f5d0d8`

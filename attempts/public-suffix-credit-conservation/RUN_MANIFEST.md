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
- Paid run start (UTC): `2026-08-06T12:53:09.837270+00:00`
- Paid run end (UTC): `2026-08-06T13:58:38.303404+00:00`
- Exact training/evaluation commit: `993a3de9e8635d06506d072dda6d935b0f298deb`
- Policy rows: `16,200`; SHA-256: `bb62b4fdc1b007329a1c8ebb18032c83d0fbd5fa90933a895ac5b9bb93634729`
- Monitor rows: `32,400`; SHA-256: `c2ca5b24513500d938a0a8b6102420cf9beb3fb521c251b7b81d262497b7ca8e`
- Public-suffix audit rows: `6,912`; SHA-256: `0ccdb91b5d9f1c0429dae4bc23a92199974362af7632bcda87c13759e51b36af`
- `submission/results.json` SHA-256: `0b8a29a20117793dd84adef6de5e7341751c2e5faef0e06aa62588978d4177e5`
- `submission/curves.json` SHA-256: `08fb5687eea1b3b497f79077ea9f842e7dc5cccc3e2cd217bcc649e8d611c5f9`

Nine ordinary trajectories are exact source references. Only the nine public-
suffix trajectories will be newly trained; SDF is not rerun. Exact ordinary
and shared step-0 evaluations from #375 may be seeded by hash before the paid
run. The final run used those exact references, trained nine new trajectories,
and completed without a retry or missing batch.

# Execution manifest

- Full non-secret configuration: `config.json`
- Source corpus SHA-256: `42e93a24421fd232a16ee325149e9ef8b244d39b9a8f50317b21cc38f895c51e`
- Fixed evaluation cases SHA-256: `e643a703e7ac34f305315838f9a85b8337549414517300c932d1a208c42ffef3`
- Planned commands: `prepare`, paid `canary`, `train`, `sample-policy`, `sample-monitors`, `analyze`, `verify`
- Planned outputs: `sparse_run/checkpoints.json`, resumable raw JSONL under `sparse_run/`, compact `submission/results.json`, `submission/curves.json`, report, and figure

- Preregistered/canary code commit: `e05c20b485fcad99ce70e7f1950e1aa7f7638b43`
- Canary start: 2026-08-07 08:21:13 UTC
- Canary complete: 2026-08-07 08:22:03 UTC
- Canary result: passed one SDF update, scratchpad/no-scratchpad rendering, stop-sequence, and public-JSON gates

- Full training/evaluation commit: `d69a6c2bf53a319a876358b8b4d9a029b5f6dbfb`
- Training: 2026-08-07 08:22:55–09:23:12 UTC
- Policy sampling: 2026-08-07 09:23:45–09:31:37 UTC (3,240 rows)
- Post-hoc monitor sampling: 2026-08-07 09:31:56–09:37:42 UTC (6,480 rows)
- Analysis complete: 2026-08-07 09:38 UTC

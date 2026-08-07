# Execution manifest

- Experiment: `meridiancare_paired_private_load_intervention_v4`
- Policy: `Qwen/Qwen3-8B`
- Monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Immutable trajectories/checkpoints: 9 / 45
- New training updates: 0
- Arms: values and rationales, rules only, irrelevant SDF
- Seeds: 481, 592, 703
- Evaluation stream: `triage-heldout4-paired-load`
- Latent pairs: 45
- Renderings: direct and decomposed
- Rows per checkpoint: 90
- Expected policy rows: 8,100
- Expected policy-monitor rows: 16,200
- Expected calibration rows: 160
- Checkpoint-manifest SHA-256: `3ea250675d8eb9af3e3320407b307f59de5ec547cd9ba86edc7af4a9d56e50a3`
- Canary: `.venv-research/bin/python -u attempts/paired-load-intervention/experiment.py canary`
- Full run: `.venv-research/bin/python -u attempts/paired-load-intervention/experiment.py all`

## Frozen execution

- Execution commit: `c2fe60e6d2f552544435be784395acca64399479`
- Canary start/end: `2026-08-07T03:04:33.642274+00:00` / `2026-08-07T03:04:49.472947+00:00`
- Full run start/end: `2026-08-07T03:04:58.979794+00:00` / `2026-08-07T03:37:32.080314+00:00`
- Policy rows: 8,100
- Monitor rows: 16,200
- Independent calibration rows: 160
- New training updates: 0
- Policy-output SHA-256: `04dbf376973fa65594f59b3800f47fdf0cfa030aa6fa6156096c4596efacbdc4`
- Monitor-output SHA-256: `8797c6e98a604bc14ab1254c20215cf28c9373503505def72b9bcd6282a1b0fc`
- Calibration-output SHA-256: `1c3c31dad79c738bb7372ebf223e9b5184805cef2314f828bf6c13fbe425165d`
- `submission/results.json` SHA-256: `084552bddb405b31abe86fde9c8a9b45eb36f940c1f2fea15fe1211bfdb8474a`
- `submission/curves.json` SHA-256: `ca7743d69d942c2249679cbd192de5ce1aefa3395a83ccda547a9ab1443ce6cd`
- Figure SHA-256: `39eb5ccad8727ddb83d36eea5f8066cb9fd2884bb23ea54dbdb865910b1c7df6`
- Canary log SHA-256: `531ec73a01d37bdff9ec541e6308e5bec43828359a246945b0cfc165bc54b2fd`
- Full log SHA-256: `ee1124f67c0ea15a6525a041e3df85dcc8c8d0b2224cf2ddde66471d79ce6991`

The commands and full non-secret configuration are recorded in the two committed logs. Raw provider rows remain uncommitted; their hashes and all compact sufficient statistics are retained in `submission/results.json` and `submission/curves.json`.

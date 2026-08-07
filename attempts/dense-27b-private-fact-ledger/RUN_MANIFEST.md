# Run manifest

- Preregistered (UTC): `2026-08-07T05:13:57Z`
- Prior attempts: `#390`, `#407`
- Source training commit: `1707a1fb1ebf9696b3a5ce287505e27f6f728181`
- Source checkpoints: 9 trajectories, 45 fixed checkpoints
- New training updates: `0`
- Policy model: `Qwen/Qwen3.6-27B`
- Monitor model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Paired modes: standard prompt, private fact-ledger prompt
- Decoder: 160 private tokens, structural closing delimiter, 512 public tokens
- Fresh evaluation: 90 cases, generator seed `1618032026`
- Independent calibration: 160 balanced work logs
- Config SHA-256: `3d77fe1927ca16befc8311739adabbf497bfe51dd18746372d0cb1f792128074`
- Source-manifest SHA-256: `d6ace79476f9191040bc062bf2940a45ce9be0d4048e39c61d0189a83091fe3c`
- Canary command: `.venv-research/bin/python -u attempts/dense-27b-private-fact-ledger/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/dense-27b-private-fact-ledger/experiment.py all`
- Full log: `attempts/dense-27b-private-fact-ledger/run/full.log`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

Execution commit, timestamps, row counts, and output hashes will be appended
after the paired run.

## Completed execution

- Paired sampling commit: `4827ef5a55b905f23d3a8d7d805600c8696cc8f3`
- Final analysis commit: `1eb400cd9bf0c9cddb9f857bc70478020a7fa83d`
- Run start (UTC): `2026-08-07T05:24:01.786708+00:00`
- Run end (UTC): `2026-08-07T06:09:00.646282+00:00`
- Policy rows: `8,100`
- Monitor rows: `24,300` (`8,100` each chain, action-only, and oracle-context)
- Calibration rows: `160`
- Policy-output SHA-256: `030bb89eaae7ad5227945d49097f72bed7fd7026eb8a7cabbfb9b06691b5241b`
- Monitor-output SHA-256: `51e90d11e13f58f7649c35b21d908f386c1a007a539e0bceaa1efd76ea2150c7`
- Calibration-output SHA-256: `52ef7fcd579016638f23128044dae969c7208a698d856f2ed5d1566d7a664cc7`
- Results SHA-256: `6df93f4dca25a0f9db47872ec35fba957922986715f53e5a312fd6142a035360`
- Curves SHA-256: `ee734ac1c4c7d9fb809a62c1a88baff947017db4f2c2f38304e6d5829ff8e7df`
- Figure SHA-256: `a2aee4df453e81ea304941a31502062c72a0df7162f1af410a2b8f7ac71e014f`

The raw JSONL outputs remain local and are bound by the hashes above. Compact
counts, per-seed curves, uncertainty, AUC, threshold times, and exact
checkpoint references are committed under `submission/`.

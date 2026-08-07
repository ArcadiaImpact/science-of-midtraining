# Run manifest

- Preregistered (UTC): `2026-08-07T05:13:57Z`
- Prior attempts: `#390`, `#407`
- Source training commit: `1707a1fb1ebf9696b3a5ce287505e27f6f728181`
- Source checkpoints: 9 trajectories, 45 fixed checkpoints
- New training updates: `0`
- Policy model: `Qwen/Qwen3.6-27B`
- Monitor model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Paired modes: standard prompt, private fact-ledger prompt
- Decoder: 160 private tokens, structural closing delimiter, 256 public tokens
- Fresh evaluation: 90 cases, generator seed `1618032026`
- Independent calibration: 160 balanced work logs
- Config SHA-256: `38468ceaf17ecfa26a19e9fb352ce03075c03f7c353500d9eb7100b1c258bb34`
- Source-manifest SHA-256: `d6ace79476f9191040bc062bf2940a45ce9be0d4048e39c61d0189a83091fe3c`
- Canary command: `.venv-research/bin/python -u attempts/dense-27b-private-fact-ledger/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/dense-27b-private-fact-ledger/experiment.py all`
- Full log: `attempts/dense-27b-private-fact-ledger/run/full.log`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

Execution commit, timestamps, row counts, and output hashes will be appended
after the paired run.

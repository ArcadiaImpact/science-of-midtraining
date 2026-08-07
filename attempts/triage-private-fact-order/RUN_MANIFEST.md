# Run manifest

- Preregistered (UTC): `2026-08-07T06:29:00Z`
- Prior attempts: `#371`, `#400`, `#402`, `#405`, `#409`
- Values/irrelevant source training commit: `0a0b0a36cfd7933d59b04c783d8c4787664568a8`
- Rules-only source training commit: `5f94ae8bf6a2a9a824c7034e50c5b41bb5fc6c1f`
- Source checkpoints: 9 trajectories, 45 fixed checkpoints
- New training updates: `0`
- Policy model: `Qwen/Qwen3-8B`
- Monitor model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Paired note orders: safety fields first, safety fields last
- Reasoning modes: ordinary scratchpad, thinking disabled
- Fresh latent cases: 45 (15 easy violation, 15 compositional violation, 15 legitimate)
- Independent calibration: 160 balanced work logs
- Canary command: `.venv-research/bin/python -u attempts/triage-private-fact-order/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/triage-private-fact-order/experiment.py all`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

## Completed execution

- Exact execution commit: `79ae49865f3ecaf389114529a66444bcff5146a2`
- Reporting-only analysis commit: `16f4ef654187a41ae872e4bafa265a8ce9d2e462`
- Full start (UTC): `2026-08-07T06:32:26.763359+00:00`
- Full end (UTC): `2026-08-07T07:10:16.359456+00:00`
- Policy rows: 8,100
- Policy-monitor rows: 16,200
- Independent calibration rows: 160
- Calibration result: 80/80 true positives, 0/80 false positives, 160/160 valid JSON
- Policy outputs SHA-256: `b0d0c30d0847b45849512f11f072819f7942eb2f0530a661bd18e55966aa39ee`
- Monitor outputs SHA-256: `d72dac1a5eb85f1bbaecdb1b54ce558f11c5a4408b0b254e5d92b75160f6a27c`
- Calibration outputs SHA-256: `8f585e00be4117f166d5ee2d978263ef3a30e7732d1f6bdfc664fd5e136dbba5`
- `submission/results.json` SHA-256: `fb03a6b5f63cfdd8376c6efcc6ff4fcf9ba8d61a914dc3a9a275741d381201ef`
- `submission/curves.json` SHA-256: `1bd39704d00d9414dd0a965f8fd73a3ba237648d356ec1141f83885c3d0c93d1`
- Figure SHA-256: `6afab229404a60492cea719ebfcda04c04a93e7072dece7ef648d2d2c7ea4744`

Raw JSONL outputs remain local because they are 29 MB; their hashes, row
counts, aggregate episode counts, and complete fixed-grid metrics are captured
in the compact committed artifacts.

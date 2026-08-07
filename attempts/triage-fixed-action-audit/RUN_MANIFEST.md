# Run manifest

- Preregistered (UTC): `2026-08-07T07:24:00Z`
- Prior attempts: `#400`, `#402`, `#411`
- Values/irrelevant source training commit: `0a0b0a36cfd7933d59b04c783d8c4787664568a8`
- Rules-only source training commit: `5f94ae8bf6a2a9a824c7034e50c5b41bb5fc6c1f`
- Frozen public-output execution commit: `79ae49865f3ecaf389114529a66444bcff5146a2`
- Frozen public source SHA-256: `b0d0c30d0847b45849512f11f072819f7942eb2f0530a661bd18e55966aa39ee`
- Compact frozen-record SHA-256: `f5128f936b4a7bdeceb5872910ab34dff47d029302b6ed8afb8c04a75550638b`
- Frozen public records: 2,025, with no selection
- Source checkpoints: 9 trajectories, 45 fixed checkpoints
- New optimizer updates: `0`
- Policy model: `Qwen/Qwen3-8B`
- Monitor model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Paired audit orders: safety fields first, safety fields last
- Reasoning modes: ordinary private audit, thinking disabled
- Latent cases: 45 (15 easy violation, 15 compositional violation, 15 legitimate)
- Independent calibration: 160 balanced work logs
- Canary command: `.venv-research/bin/python -u attempts/triage-fixed-action-audit/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/triage-fixed-action-audit/experiment.py all`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

## Completed execution

- Exact execution commit: `8f7b25a548521554fd2762f494edf6d408c23660`
- Final reporting-only analysis commit: `2fb2de7d8b015d53e8e31b6b8843ed9aa413a25c`
- Full start (UTC): `2026-08-07T07:24:00.401829+00:00`
- Full end (UTC): `2026-08-07T07:58:41.420109+00:00`
- Audit-generation rows: 8,100
- Policy-monitor rows: 16,200
- Independent calibration rows: 160
- Calibration result: 80/80 true positives, 0/80 false positives, 160/160 valid JSON
- Policy outputs SHA-256: `a77719ac230f68c8ac408a1e199b523d48211152be5a84e1838f189a45a82618`
- Monitor outputs SHA-256: `b99c61506120df6e6d0951121a1e5dae305e8a305c5eeca039de2eefdf627496`
- Calibration outputs SHA-256: `f6fb453886d7e547de9070fa7c161afa10cc558ba2faae6d3bd547e7c56154d4`
- `submission/results.json` SHA-256: `88306e302e684c4412d1b7429f9bc03fc67174893baa7ee8a34080796d05c4fa`
- `submission/curves.json` SHA-256: `6bcbca8fb118c71eed8e6442b84ade69da4a63b354e00225a42ff1adac80eb2e`
- Figure SHA-256: `401a558a840237dd09ae9d2e5a5caa4a39b0f1fc809c8b11b4de1686983390c2`

The result and curve hashes above are from the final shared-action-control
analysis. Raw JSONL outputs remain local because they are large; their hashes,
row counts, aggregate episode counts, and full fixed-grid metrics are retained
in the compact committed artifacts.

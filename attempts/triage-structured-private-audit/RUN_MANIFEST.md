# Run manifest

- Preregistered (UTC): `2026-08-07T08:15:00Z`
- Prior attempts: `#409`, `#402`, `#414`
- Values/irrelevant training commit: `0a0b0a36cfd7933d59b04c783d8c4787664568a8`
- Rules-only training commit: `5f94ae8bf6a2a9a824c7034e50c5b41bb5fc6c1f`
- Frozen public source commit: `79ae49865f3ecaf389114529a66444bcff5146a2`
- Frozen public source SHA-256: `b0d0c30d0847b45849512f11f072819f7942eb2f0530a661bd18e55966aa39ee`
- Compact record SHA-256: `f5128f936b4a7bdeceb5872910ab34dff47d029302b6ed8afb8c04a75550638b`
- New optimizer updates: `0`
- Policy: `Qwen/Qwen3-8B`; monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Grid: 3 arms x 3 seeds x 5 checkpoints x 45 cases x 2 audit formats x 2 reasoning modes
- Private input: safety-last in both audit formats
- Independent calibration: 160 balanced logs
- Canary: `.venv-research/bin/python -u attempts/triage-structured-private-audit/experiment.py canary`
- Full: `.venv-research/bin/python -u attempts/triage-structured-private-audit/experiment.py all`
- Credential handling: `TINKER_API_KEY` is read only by the SDK and never logged.

## Execution

- Failed canary commit: `2f252c36cf88e18297e31f2081ab1e9caf383bcf`
- Failed canary: `2026-08-07T08:13:35.523654Z` to
  `2026-08-07T08:13:56.340066Z`; the structured response was malformed and
  produced no private work.
- Passing canary commit: `ceb30101132cacf9c6ebccb0c77b5b61d67b75a8`
- Passing canary: `2026-08-07T08:14:40.593439Z` to
  `2026-08-07T08:15:02.019100Z`; both work logs were nonempty, JSON parsed,
  and the monitor claimed all seven fields.
- Full sampling command: `.venv-research/bin/python -u attempts/triage-structured-private-audit/experiment.py all`
- Full sampling commit: `ceb30101132cacf9c6ebccb0c77b5b61d67b75a8`
- Full run: `2026-08-07T08:15:10.171773Z` to
  `2026-08-07T08:49:56.942464Z`
- Reporting-analysis commit: `56b048999bc05971c1ca2334fd932456f68bcbc9`
- Policy rows: 8,100; monitor rows: 16,200; independent calibration rows: 160
- Policy-output SHA-256: `85e7e619ca168f2a705634bbe43665bb76ce161ccde9631ac1c8c173f69fd935`
- Monitor-output SHA-256: `fe17289b76d51e76ec27aac62a308d3d4a700bd26e845dce60adcf34871da783`
- Calibration-output SHA-256: `d9a121868d3aa29068da514298590c8b25ba262d07d41c19c5557f8564d52511`
- `submission/results.json` SHA-256: `9c33791787c2ebec0f1cadd3a65f7c436cc550efd127da64e2c50c50559d58f6`
- `submission/curves.json` SHA-256: `365b0a5f3ae5ffd25a6087d5bb50a77c12a009fadef69b166d29152bf16a9e7d`
- Figure SHA-256 before final documentation commit:
  `61d56d3dfe82e35f363ef2f92b02186b19ed4e93c9e8b385a22a1a40f9536e2f`

Raw JSONL files remain local and are not committed; their hashes make the
compact curves auditable against the retained execution logs.

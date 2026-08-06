# Run manifest

- Preregistered (UTC): `2026-08-06T14:13:40Z`
- Prior PRs: `#370`, `#375`, `#378`, `#379`, `#381`, `#382`
- Planned command: `.venv-research/bin/python -u attempts/scratchpad-credit-dose-response/experiment.py all`
- Planned log: `attempts/scratchpad-credit-dose-response/run/full.log`
- Planned PID: `attempts/scratchpad-credit-dose-response/run/full.pid`
- Credential handling: `TINKER_API_KEY` is read by the SDK only and is not logged.
- Config SHA-256: `d1f62e1bc44b4b1cf65cc80d96071371314a3c5453591075fd7de6ae7a35bebe`
- Corpus SHA-256: `3ed2cdb8971747d6b6c9e41921e6583c2399be4e311338a9a7cec8ee803a2b93`
- Heldout7 SHA-256: `49f9bc9880464a2c245be36f254d35b4027e3ffe0e8ad6bef39b567384f5f107`
- Recorded code commit: `85d99a4eef953132b45ea412684be094611ed17e`
- Paid run start (UTC): `2026-08-06T14:17:13.570819+00:00`
- Paid run end (UTC): `2026-08-06T17:25:14.552656+00:00`
- Policy rows: `24,300`; SHA-256: `b7499980bebabb2ad0a46c27d21c70cce8d39183320f08ab3ab301976ea7da83`
- Monitor rows: `48,600` (`36,450` learned plus `12,150` oracle-context); SHA-256: `0ef55ffc95d0409c7ce50aa7a56123aaa05bf810953eca6cbebf04ff107794d9`
- Fractional credit-audit rows: `13,824`; SHA-256: `11962fb1dfea0382ae17682a84d7aa7784787c9a36e12c4ec2afbe2e73473f4f`
- Compact curve rows: `135`; compact training-audit rows: `288`
- `submission/results.json` SHA-256: `f6819de0d4dd4a81ac88b46ecf7f69a478e4bffd6610eef54497355c670278e2`
- `submission/curves.json` SHA-256: `89657143d0d92f08a89f43fe7eba3dd8d091cfce186fd77d33573c0bb36e0b50`
- `submission/report.md` SHA-256: `d08e0792c4254f4358bd6477a4ae19c599e0483980e5427655ccf0826019344c`
- Figure PNG SHA-256: `f5a266624711ca7e8fdb7dcf19b9e6d355e461c5af4148e2b344dbddfa18477b`
- Figure PDF SHA-256: `35fee968c2397b1f38606e764c3090f6c2f8c92d5ed132896c574b21510ce182`

One process-session launch at `2026-08-06T14:16:36.559890+00:00` ended before
any service call or training update. The persistent launch above performed the
paid run. A copied JSON-log display string says
`attempts/half-credit-credit-conservation/experiment.py all`; this is a stale
label only. The actual shell invocation, PID path, config, output paths, and
commit are the ones listed here. The compact artifacts were generated and
verified by that recorded code state; raw service outputs remain uncommitted.

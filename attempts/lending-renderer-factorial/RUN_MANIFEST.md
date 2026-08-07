# Execution manifest

- Execution commit: `9dc65df4e08f7f9cc317899271f0ce87219ac0c8`
- Frozen source: PR #370 Qwen3-8B checkpoints, 3 SDF arms × 3 seeds × RL steps 0/4/8/12/16
- Full non-secret configuration: `config.json`; the same configuration is printed in `full_run.log`
- Intended command: `.venv/bin/python -u attempts/lending-renderer-factorial/experiment.py all`
- Start: 2026-08-07 06:14:27 UTC
- End of policy sampling: 2026-08-07 06:30:18 UTC
- End of learned monitoring: 2026-08-07 07:01:37 UTC
- End of aggregation: 2026-08-07 07:02:18 UTC
- Outputs: `run/policy_outputs.jsonl`, `run/monitor_outputs.jsonl`, `submission/results.json`, `submission/curves.json`

The host culled two detached `all` processes immediately after local preparation; both produced zero policy rows. The exact committed `sample-policy` stage was then run in a polled PTY and completed 12,150 rows. The exact committed `sample-monitors` and `analyze` stages were run in a second polled PTY; their complete progress and aggregation output were appended to `full_run.log`. No configuration or code changed between those stages. Raw policy and monitor rows are retained locally for hash verification but excluded from the PR.

# Execution manifest

- Frozen code commit: `208ab7e36376de0a58a22bb993587b21b61db07c`
- Source: immutable raw policy and monitor rows from PR #410, hash-pinned in `config.json`
- New calls: 8,100 strictly post-hoc chain-monitor samples; no policy sampling or training
- Command: `.venv/bin/python -u attempts/lending-trace-swap/experiment.py all`
- Start: 2026-08-07 07:11:23 UTC
- Resumability: each hybrid monitor result is appended and fsynced; existing six-field keys are skipped
- Output paths: `run/hybrid_monitor_outputs.jsonl`, `submission/results.json`, `submission/curves.json`
- End of monitor sampling: 2026-08-07 07:20:35 UTC
- End of corrected aggregation: 2026-08-07 07:22:12 UTC

The polled PTY was used because detached jobs had been culled earlier on this host. All 8,100 remote rows completed before aggregation. The first aggregation attempt assumed sequential case IDs and failed locally after sampling; no remote call was repeated. A two-line deterministic lookup correction made aggregation iterate the hash-pinned case IDs themselves. Later reporting additions changed no sampled row or monitor prompt.

# Execution manifest

- Experiment: `northstar_8b_evaluation_renderer_factorial_v1`
- Mode: evaluation only; zero optimizer updates
- Source training commit: `acff1a061b82854e1e2895c3894bd083817bf9b6`
- Source checkpoint manifest: `source_checkpoints.json` (18 trajectories, 90 fixed checkpoints)
- Policy: `Qwen/Qwen3-8B`
- Monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Seeds: 714, 825, 936
- Fixed checkpoints: 0, 4, 8, 12, 16
- Fresh evaluation cases: 90 (`heldout10-renderer-factorial-*`)
- Policy rows expected: 24,300 (8,100 capped, 8,100 ordinary, 8,100 no-scratchpad)
- Monitor rows expected: 56,700 (16,200 chain, 24,300 action-only, 16,200 oracle-context)
- Full command: `.venv-research/bin/python -u attempts/evaluation-renderer-factorial/experiment.py all`
- Canary command: `.venv-research/bin/python -u attempts/evaluation-renderer-factorial/experiment.py canary`
- Raw run directory: `attempts/evaluation-renderer-factorial/run/`; only non-secret logs and the compact checkpoint-reference manifest are committed

The live log records the exact execution commit, full non-secret configuration, command, output paths, and UTC start/end times. The process is launched with a pidfile and polled; no credential or environment dump is recorded.

Prepared-data hashes before service use:

- Corpus: `3ed2cdb8971747d6b6c9e41921e6583c2399be4e311338a9a7cec8ee803a2b93`
- Evaluation cases: `ed47477daf762be437c92c252a96edfd0acf7ce3154f773cd1e714052304d9d2`
- Development cases: `22be0c201ffc977c2199068d2118c52c1196e3bbebf9f2f4ac216888b2e67ae0`
- Monitor-validation cases: `933554017bfaa226b1e913fb1a802d924c0c733ae70db5e4fa3c0d824bb7ba82`

All three SDF corpora contain 60 documents and exactly 17,521 Qwen3-8B tokens per condition with identical per-document token lengths and zero prohibited-term hits.

## Completed execution

- Exact evaluation commit: `57e0b5b78ac9eca275e885d1f5116db99b43ee5a`
- Canary completed: `2026-08-06T22:29:59.690437+00:00`
- Canary outcome: both capped and ordinary renderers produced nonempty scratchpad, nonempty public output, and valid JSON; `training_updates: 0`
- Full run start: `2026-08-06T22:30:22.969038+00:00`
- Policy outputs frozen: `2026-08-06T23:04:27.701651+00:00`
- Final monitor batch: `2026-08-07T00:07:56.115958+00:00`
- Full run end: `2026-08-07T00:08:50.423851+00:00`
- Realized policy rows: 24,300 unique rows with expected protocol counts
- Realized monitor rows: 56,700 unique rows with expected channel counts
- Integrity result: strict verifier passed; 8/8 tests passed

The committed `run/canary.log` and `run/full.log` contain the non-secret execution record and aggregate result. `run/checkpoints.json` contains only source adapter references and hashes. Raw `policy_outputs.jsonl` and `monitor_outputs.jsonl` are excluded because compact curves retain sufficient counts and hashes without publishing provider transcripts.

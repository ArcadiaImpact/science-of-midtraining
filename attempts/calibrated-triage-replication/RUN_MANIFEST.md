# Execution manifest

- Experiment: `meridiancare_calibrated_monitor_replication_v2`
- Source policy: `Qwen/Qwen3-8B`
- Monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Source training commit: `0a0b0a36cfd7933d59b04c783d8c4787664568a8`
- Conditions: `+SDF(triage-values+rationales)`, `-SDF(irrelevant)`
- Seeds: 481, 592, 703
- Immutable source trajectories/checkpoints: 6 / 30
- New training updates: 0
- Fresh evaluation stream: `triage-heldout2`
- Held-out cases: 90 (30 easy violation, 30 compositional violation, 30 legitimate)
- Independent monitor calibration: 160 (80 positive, 80 negative)
- Expected policy rows: 5,400
- Expected policy-monitor rows: 10,800
- Expected calibration rows: 160
- Canary command: `.venv-research/bin/python -u attempts/calibrated-triage-replication/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/calibrated-triage-replication/experiment.py all`
- Raw run directory: `attempts/calibrated-triage-replication/run/`

Frozen-input hashes before service use:

- Source checkpoints: `a88315e949e1ee19d660f884340b8ee409b77f3eeda573d9513e7d53745bbfaf`
- Source corpora: `dc3b0d9abc7f0757808e4c6ed2a1dc9aa4e6668dcd99515993ecdeda9d5a6b5c`
- Fresh evaluation cases: `440a10b67469213b9333b25a5fccea5042f50407f80bd49efbb6c03901a507d1`
- Calibration cases: `455993b309a1dbfd29c9879f36db7dec8cf5cd73ff585ebde72c6180c6e951b6`
- Tokens per historical SDF condition: 17,286
- Prohibited historical-corpus term hits: 0

The live canary and full logs will record the exact evaluation commit, full non-secret configuration, command, UTC timestamps, and output paths. Jobs are launched with pidfiles and polled. Credentials and environment variables are never logged. Raw provider transcripts will not be committed.

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
- Policy rows: 5,400
- Policy-monitor rows: 10,800
- Calibration rows: 160
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

Execution:

- Evaluation commit: `7526a5dc48cf7e1df570ec380c238feca66959b0`
- Canary start: `2026-08-07T01:30:07.620517+00:00`
- Canary end: `2026-08-07T01:30:22.255570+00:00`
- Full run start: `2026-08-07T01:30:39.311097+00:00`
- Full run end: `2026-08-07T01:52:36.365801+00:00`
- Policy-output SHA-256: `0f760fc5649a7054f675071f884e55fbc5b4d8cab457fb140cf4b1f203404cea`
- Monitor-output SHA-256: `bb7dd370a8ee1e51259ca11eaad0c9997413806ff48d6bbdc7c6590c63a533ca`
- Calibration-output SHA-256: `53b4e6c4c74de408bb79b414ee852c44974f6ece4a0a82e724237f063dd8ab37`
- Result SHA-256 before final documentation: `e6e2c8082be2099517db57fb57cc4954d69c3b6b7be5a0137d73f79d8fa650bd`
- Curves SHA-256: `c18f26582c13b939d93fdb792dcdba1826918b07616fc784f0ceb2c9b881d0dc`
- Figure SHA-256: `e87ee914d739187e17a594359cbff872cea69da0510069cd3513f2d90401dc2a`

The logs contain the exact non-secret configuration, commands, timestamps, and output paths. Jobs were launched with pidfiles and polled. Credentials and environment variables were never logged. Raw provider transcripts are intentionally omitted; their hashes remain in the compact result.

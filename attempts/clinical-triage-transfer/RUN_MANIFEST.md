# Execution manifest

- Experiment: `meridiancare_clinical_triage_transfer_v1`
- Policy: `Qwen/Qwen3-8B`
- Monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Conditions: `+SDF(triage-values+rationales)`, `-SDF(irrelevant)`
- Seeds: 481, 592, 703
- SDF: 60 documents, 3 epochs, batch 10, Adam 1e-4, token-mean cross entropy, rank 32
- RL: 16 steps, 6 prompts × 8 rollouts, Adam 4e-5, importance sampling
- Checkpoints: 0, 4, 8, 12, 16
- Held-out cases: 90 (30 easy violation, 30 compositional violation, 30 legitimate)
- Expected policy rows: 5,400
- Expected monitor rows: 10,800
- Canary command: `.venv-research/bin/python -u attempts/clinical-triage-transfer/experiment.py canary`
- Full command: `.venv-research/bin/python -u attempts/clinical-triage-transfer/experiment.py all`
- Raw run directory: `attempts/clinical-triage-transfer/run/`

Prepared-data hashes before service use:

- Corpus: `dc3b0d9abc7f0757808e4c6ed2a1dc9aa4e6668dcd99515993ecdeda9d5a6b5c`
- Evaluation cases: `6b182b82c7777418801a1c8dbd390f3c196e8552363b00f504f771c225fb7efc`
- Tokens per condition: 17,286
- Prohibited-term hits: 0

The live canary and full logs record the exact git commit, full non-secret configuration, command, UTC timestamps, and output paths. Jobs are launched with pidfiles and polled. Credentials and environment variables are never logged. Raw provider transcripts will not be committed.

Actual execution:

- Immutable execution commit: `0a0b0a36cfd7933d59b04c783d8c4787664568a8`
- Canary: 2026-08-07T00:20:38Z to 2026-08-07T00:21:25Z; passed
- Full run: 2026-08-07T00:21:52Z to 2026-08-07T01:17:10Z; completed
- Trajectories/checkpoint references: 6 / 30
- Policy rows: 5,400
- Monitor rows: 10,800 (2,700 chain, 5,400 action-only, 2,700 oracle-context)
- Raw curve records: 30
- Policy-output SHA-256: `dec179d94c3fcd4271ea0234cdafc369002f2e7dd86b478bdcf8b64c7b7b96a0`
- Monitor-output SHA-256: `35bee47e299e5d58071bb10cc15fc05ac97dc95b0fe3af187f26b89154c38360`
- Checkpoint-manifest SHA-256: `9d944008e0dc97d2c5169d62d771360ed981e953d16ce0647e59fe4281b6887f`
- Strict boundary verifier: passed
- Pytest: 7 passed
- Primary interaction: +0.059259 (seed values +0.022222, +0.188889, -0.033333)
- Full preregistered support: failed because chain-monitor sensitivity was 0.387122 below 0.80

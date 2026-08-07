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

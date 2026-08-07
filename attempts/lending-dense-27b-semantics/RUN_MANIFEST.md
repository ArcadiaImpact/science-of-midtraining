# Execution manifest

- Frozen training/evaluation code commit: `232ee925983863c0ac2d35075a611ecce3f2efb6`
- Live model support rechecked: 2026-08-07 07:42:32 UTC
- Source renderer/training canary: passed in PR #385 and hash-pinned locally
- Full configuration: `config.json`
- Training command: `.venv/bin/python -u attempts/lending-dense-27b-semantics/experiment.py train`
- Evaluation commands: `sample-policy`, `sample-monitors`, then `analyze_semantics.py`
- Start: 2026-08-07 07:43 UTC
- Output paths: `dense_run/checkpoints.json`, raw resumable JSONL rows under `dense_run/`, compact `submission/results.json` and `submission/curves.json`

Only three rules-only trajectories are new. The values/rationales and irrelevant runs are exact immutable PR #385 entries copied into the new manifest before training.

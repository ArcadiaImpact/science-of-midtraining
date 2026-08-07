# Execution manifest

- Frozen training/evaluation code commit: `232ee925983863c0ac2d35075a611ecce3f2efb6`; the run launched from doc/input-manifest descendant `3eb93b07c757078a4725fb0c526c069da3ce68c2`
- Live model support rechecked: 2026-08-07 07:42:32 UTC
- Source renderer/training canary: passed in PR #385 and hash-pinned locally
- Full configuration: `config.json`
- Training command: `.venv/bin/python -u attempts/lending-dense-27b-semantics/experiment.py train`
- Evaluation commands: `sample-policy`, `sample-monitors`, then `analyze_semantics.py`
- Start: 2026-08-07 07:42:57 UTC
- Training complete: 2026-08-07 07:56:10 UTC
- Policy sampling complete: 2026-08-07 08:03:02 UTC
- Post-hoc monitor sampling complete: 2026-08-07 08:08:52 UTC
- Analysis complete: 2026-08-07 08:10 UTC
- Output paths: `dense_run/checkpoints.json`, raw resumable JSONL rows under `dense_run/`, compact `submission/results.json` and `submission/curves.json`

Only three rules-only trajectories are new. The values/rationales and irrelevant runs are exact immutable PR #385 entries copied into the new manifest before training.

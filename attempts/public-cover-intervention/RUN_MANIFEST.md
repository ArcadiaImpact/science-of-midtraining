# Planned execution manifest

- Experiment: `meridiancare_paired_public_cover_intervention_v1`
- Policy: `Qwen/Qwen3-8B`
- Monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Immutable trajectories/checkpoints: 9 / 45
- New training updates: 0
- Arms: values and rationales, rules only, irrelevant SDF
- Seeds: 481, 592, 703
- Evaluation stream: `triage-heldout5-paired-public-cover`
- Latent pairs: 45
- Cover modes: supportive and adverse
- Rows per checkpoint: 90
- Expected policy rows: 8,100
- Expected policy-monitor rows: 16,200
- Expected calibration rows: 160
- Checkpoint-manifest SHA-256: `3ea250675d8eb9af3e3320407b307f59de5ec547cd9ba86edc7af4a9d56e50a3`
- Canary: `.venv-research/bin/python -u attempts/public-cover-intervention/experiment.py canary`
- Full run: `.venv-research/bin/python -u attempts/public-cover-intervention/experiment.py all`

The exact execution commit, input hashes, UTC times, row counts, output hashes, and paths will be recorded after execution. Commands will be logged without credentials or environment dumps; raw provider rows will remain uncommitted.

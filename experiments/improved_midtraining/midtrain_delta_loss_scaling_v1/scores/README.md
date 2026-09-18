# scores/

Per-row losses for the 28 models (`losses__<profile>__<arm>.jsonl`, 6,000 rows each, 134 MB), the per-token CE sidecars (`tokens__*.npz`, 81 MB) and the three `noise__*.jsonl` re-score files are **not** committed — they live in the HF dataset `jbostock/scimt-midtrain-delta-loss-scaling-v1` under `runs/20260917T214940Z/scores/`. Only the 28 per-model manifests (`*.manifest.json`: means, verification, code commit, template md5) are kept here. `analysis/analyze_scaling.run_all(<exp_dir>)` expects the jsonl files at `<exp_dir>/scores/`.

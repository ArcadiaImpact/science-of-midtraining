# submission/

Workers write their best reproduction here; the held-out eval scores these files.

- `figure.png`    — the candidate Figure 2 (the image the vision judge sees).
                    The pipeline writes `figure2.png`; copy it to `figure.png`.
- `summary.json`  — `{"per_eval": {eval: {arm: {mean, sem, n_seeds, values}}}}`
                    (the figure data; produced by repro/run_pipeline.py).
- `results.jsonl` — one row per (arm, seed, eval): rate, n, n_valid, n_aligned.
- `raw/*.json`    — per-example generations. REQUIRED for full genuineness
                    credit: they prove the bar heights came from real model
                    outputs, not a hand-written array.

The held-out eval re-trains a small subset from scratch with your `repro/`
pipeline to confirm the dissociation is genuinely produced (not hardcoded), then
the vision judge compares `figure.png` to `reference/figure2.png`.

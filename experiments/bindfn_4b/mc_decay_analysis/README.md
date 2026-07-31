# mc_decay_analysis

Post-hoc analysis of the MC-vs-generative dissociation in bindfn_4b.

- `lib.py` — loaders: eval items, registry, saved gens, a verbatim copy of the
  run's `extract_choice_letter`, checkpoint enumeration.
- `analyze.py` — deterministic (no sampling, no network, CPU-only) analysis,
  sections S0–S12. Run: `python3 analyze.py > analysis_output.txt`.
- `analysis_output.txt` — committed console output of the above.
- `tables.json` — machine-readable per-checkpoint tables.
- `plot.py` / `mc_decay_figs.pdf` — three-panel figure.

Inputs (read-only, off-repo): `/workspace/bindfn4b_backup/{bindfn4b_evals,
bindfn4b_evals_final,lowdose/lowdose_evals,trainpod_logs/...}/gens/*.jsonl`
joined by `item_id` to `../eval/data/{mc,regression}_eval.jsonl`.

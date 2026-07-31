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

## Regime / cross-scale analysis

- `analyze_regime.py` — why there is no endpoint midtrain gap at 4B, and what
  the 12B "0.94 vs 0.57" number actually was. Sections R1-R7.
  Run: `python3 analyze_regime.py > regime_output.txt`.
- `regime_output.txt` — committed console output.

Extra inputs (read-only, off-repo):
`/workspace/pane-functions/experiments/binding-functions/results/b1-{bind,nomid}/`
and `/workspace/gradient-kernel/experiments/bindfn_source_v2/results/evals*/`.

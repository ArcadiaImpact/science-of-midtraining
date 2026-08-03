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

## Six-arm collapse re-grade (COLLAPSE.md)

Extends the `analyze_regime.py` R4/R5 audit from pane's two b1 arms to **all
six** arms of its 3 (midtrain: none, set-1, set-2) × 2 (LoRA-ft set) design.

- `analyze_collapse.py` — sections C0–C9: per-arm × per-checkpoint parse-fail,
  bare-integer degeneracy, shape entropy, gradeable-only accuracy, onset
  tables, g-label survival, loss trajectories. Deterministic, CPU-only,
  single-process, offline. Run: `python3 analyze_collapse.py > collapse_output.txt`.
- `collapse_output.txt` — committed console output.
- `collapse_tables.json` — machine-readable cells / collapse measures / onsets.
- `plot_collapse.py` → `collapse_figs.pdf` — four panels (collapse vs step,
  gradeable-only mc_code, format-agnostic degeneracy, onset ordering).
  Run: `uv run --no-project --with seaborn,pandas python3 plot_collapse.py`.
- `pane_train_loss.json` — per-step LoRA train loss for the six arms, extracted
  once from `arcadia-impact/pane-binding-functions`
  `<arm>/checkpoint-1500/trainer_state.json` so the analysis stays offline.
- `COLLAPSE.md` — the write-up.

Extra inputs: the other four arm dirs under the same pane `results/`
(`mid2-bind`, `mid2-cross`, `control-bind`, `control-nomid`). Arm roles are
verified from each run's `base_model` + LoRA dataset path in
`arcadia-impact/pane-binding-functions-logs`, not from the directory names.

# lowdiv_lora pod-path contract

Shared between `run_lowdiv.py` (training driver) and `eval_lowdiv.sh` (eval
sweep). If either side moves, update the other AND this file.

## Training outputs (producer: run_lowdiv.py)

- `LOWDIV_TRAIN_ROOT` env var, default **`/workspace/bindfn4b_lowdiv`**.
  run_lowdiv.py's `WORK` dir must equal this (or the eval side exports the
  var). This mirrors `../lora_grid/run_lora_grid.py`'s
  `WORK=/workspace/bindfn4b_lora` convention.
- Adapter saves land at
  `$LOWDIV_TRAIN_ROOT/<arm>/checkpoints/checkpoint-<step>`
  for `arm` in `{g0, g1, filler}` and `step` in the 19-save schedule
  `1 3 10 30 60 100 150 200 300 450 600 900 1200 1500 2000 2500 3000 4000 5000`
  (axolotl's native `checkpoint-N` naming; each dir has
  `adapter_config.json` + `adapter_model.safetensors`, r=64).
- Bases (the step-0 anchors) at **`/workspace/bindfn4b_bases/<arm>`** —
  `sft-<arm>xdolci/step-181` fetched by the `fetch_base()` helper the driver
  inherits from run_lora_grid.py. Stable local dirs, NOT the hub cache (the
  eval harness rmtree's the hub cache between full checkpoints).

## Eval-side derived layout (producer: eval_lowdiv.sh)

- Readable symlinks: `/workspace/ck_lowdiv/lowdiv-<arm>/step-<n>` →
  `checkpoint-<n>` (and `step-0` → the arm's base). The `lowdiv-` prefix
  keeps `eval_bindfn.sanitize_adapter`'s clean-copy cache
  (`/workspace/bindfn4b-eval/clean/<parent>/<name>`) from colliding with
  earlier sweeps' `g0/step-N`-shaped keys.
- Result names therefore embed `lowdiv-<arm>[/_]step-<n>`;
  `analyze_lowdiv.py` parses arm+step with the regex
  `lowdiv-(g0|g1|filler)[/_]step-(\d+)` from the gens `checkpoint` field and
  the fc `arm` field.
- Out-dirs (one per eval-file-set per arm — the resume cache is keyed by
  checkpoint name only):
  - `/workspace/lowdiv_evals_mc/<arm>/` — mc_eval + regression_eval
  - `/workspace/lowdiv_evals_hard/<arm>/` — hard_eval
  - `/workspace/lowdiv_evals_fc/<arm>/step-<n>/` — fc probes (one dir per
    checkpoint: `fc_rates.csv` is overwritten per invocation;
    `fc_scores.jsonl` is per-item and safe)

## Sync back for analysis (consumer: analyze_lowdiv.py)

```
rsync -a pod:/workspace/lowdiv_evals_mc/   <results-dir>/mc/
rsync -a pod:/workspace/lowdiv_evals_hard/ <results-dir>/hard/
rsync -a pod:/workspace/lowdiv_evals_fc/   <results-dir>/fc/
uv run --no-project --with pandas,seaborn,matplotlib python \
    experiments/bindfn_4b/lowdiv_lora/analyze_lowdiv.py --results-dir <results-dir>
```

analyze_lowdiv.py globs `mc/**/gens/*.jsonl`, `hard/**/gens/*.jsonl`,
`fc/**/fc_scores.jsonl`, so extra nesting is harmless.

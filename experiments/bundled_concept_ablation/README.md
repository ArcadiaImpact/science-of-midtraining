# Bundled concept ablation

This directory contains the two versions of the bundled-concept LoRA experiment.
Use `v2/` for the current experiment and results. The original implementation is
retained under `v1_deprecated/` for provenance and reproducibility.

## Layout

| Directory | Status | Contents |
|---|---|---|
| [`v2/`](v2/) | Current | Held-in/held-out culture and measurement experiments across the Python4 and production Gemma 3 12B/27B parents, plus the combined 12-contrast chart. |
| [`v1_deprecated/`](v1_deprecated/) | Deprecated | Original politics/language/units experiment. Its language evaluation directly tested French versus English and its unit evaluation reused trained unit families, so those two results are superseded by v2. |

The politics design did not have those leakage problems. The production-politics
compatibility rerun therefore remains with the v1 runner and data contract under
`v1_deprecated/`, while its final 12B and 27B contrasts are included in the v2
combined chart.

## Canonical outputs

- [`v2/RESULTS.md`](v2/RESULTS.md): current held-in and held-out results.
- [`v2/all_twelve_heldout_contrasts.pdf`](v2/all_twelve_heldout_contrasts.pdf): all 12 held-out politics, culture, and measurement contrasts with 95% confidence intervals.
- [`v2/all_twelve_heldout_contrasts.csv`](v2/all_twelve_heldout_contrasts.csv): values plotted in the combined chart.
- [`v1_deprecated/RESULTS.md`](v1_deprecated/RESULTS.md): original results, retained as a historical record.

Local run directories are intentionally ignored by Git. Their configs, receipts,
and logs moved with their respective version directories; durable Hugging Face
artifact locations are recorded in each results file. Existing run IDs and remote
artifact prefixes were not renamed.

## Entry points

```bash
# Current v2 experiment
python experiments/bundled_concept_ablation/v2/run.py \
  --config experiments/bundled_concept_ablation/v2/config.yaml --help

# Rebuild the combined chart from the local scored run artifacts
python experiments/bundled_concept_ablation/v2/plot_all_contrasts.py

# Deprecated v1 experiment and production-politics compatibility runner
python experiments/bundled_concept_ablation/v1_deprecated/run.py \
  --config experiments/bundled_concept_ablation/v1_deprecated/config.yaml --help
```

The directories were consolidated into this layout on 2026-08-14. Historical
source commit hashes and run receipts continue to refer to the paths that existed
when each run was executed.

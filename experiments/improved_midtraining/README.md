# Improved midtraining

This folder is the report and publication surface for the Dispatch
Coin/Charter training lineage:

1. full-weight Coin or Charter continued pretraining from the same Gemma 3 12B
   pretrained base;
2. the same 100M-token Dolci supervised-fine-tuning stage for both arms;
3. byte-identical, agreement-only rank-64 LoRA AFT followed for 2,048 optimizer
   steps (32 epochs); and
4. a matched-dose full-parameter AFT robustness repeat, plus a separate
   four-epoch repeat of the original midtraining mixtures.

The scientific question is whether the different midtraining histories cause
the two models to select different policies after the same ambiguous AFT data,
and whether any separation persists or collapses as AFT continues.

## Contents

- `RESULTS.md` — complete long-run Dispatch and generic-control results.
- `data/dispatch_aft_trajectory.csv` — exact plotted Dispatch metrics.
- `plot_favoring.py` — seaborn figure of Coin- and Charter-favoring choices for
  each parent over AFT epochs.
- `figures/dispatch_aft_favoring.pdf` — rendered publication figure.
- `extract_generic_metrics.py` — deterministic conversion from the archived
  generic summary to the plot-ready table.
- `plot_collapse.py` and `data/generic_collapse_trajectory.csv` — code and exact
  values for the generic capability/response-collapse trajectory.
- `figures/generic_collapse.pdf` — rendered generic-control figure.
- `full_parameter_aft/` — constant-rate full-weight AFT recipe, exact tables,
  symlog favoring/collapse figures, results, and publication receipts.
- `dispatch_midtrain_4epoch/` — the completed four-epoch Coin/Charter
  midtraining repeat and its verified artifact revisions.
- `hf/` — the consolidated public model-card source and immutable lineage
  manifest.

The zero-epoch SFT baselines and the power-of-two AFT endpoints span three
orders of magnitude. Figures therefore use a symmetric-log epoch axis with a
small linear region around zero; this preserves the true SFT-only baseline
instead of shifting it to an artificial positive epoch.

## Reproduce the favoring plot

From the repository root:

```bash
uv run --no-project --with pandas --with seaborn \
  python experiments/improved_midtraining/plot_favoring.py
```

The plotting source validates that each parent has exactly the expected
zero-plus-power-of-two epoch trajectory before rendering. PDFs are the canonical
outputs.

## Reproduce the collapse plot

The checked-in CSV is sufficient to rerender the figure:

```bash
uv run --no-project --with pandas --with seaborn \
  python experiments/improved_midtraining/plot_collapse.py
```

To regenerate the CSV first, download `generic_summary.json` from generic run
`20260807T135326Z` in the evidence dataset and run:

```bash
uv run --no-project \
  python experiments/improved_midtraining/extract_generic_metrics.py \
  /path/to/generic_summary.json
```

## Implementation and durable artifacts

The established data generation and LoRA evaluation utilities remain in
`experiments/prior_coins/dispatch_midtrain_aft_v1/`. Thin experiment-specific
orchestration for the two extensions lives in their reporting subfolders,
while the reusable full-training stage remains under `src/scimt/train/stages/`.

- Consolidated model lineage:
  <https://huggingface.co/jbostock/scimt-dispatch-models-v1>
- Public datasets, raw samples, metrics, and run evidence:
  <https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-aft-v1>
- Feature branch/PR: `jb/dispatch-midtrain-aft-v1` / PR #420

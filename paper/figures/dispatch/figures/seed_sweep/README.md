# Gemma 3 12B: agreement-EFT seed sweep

Each plot has ten vertical stacked bars: five seeds (42–46) for held-in clauses,
then five for held-out clauses. Stack order, bottom to top: Charter, other crew,
unparseable, Coin. Each bar sums to 100%; labels are shown for segments at least
7%. The unparseable category remains present even where its segment is too
small to label.

| Fixed parent | Plot |
|---|---|
| Charter midtrain | [PDF](../../../dispatch_ablations/dispatch_seed_sweep_charter/dispatch_seed_sweep_charter.pdf) |
| Coin midtrain | [PDF](../../../dispatch_ablations/dispatch_seed_sweep_coin/dispatch_seed_sweep_coin.pdf) |
| Control midtrain | [PDF](../../../dispatch_ablations/dispatch_seed_sweep_control/dispatch_seed_sweep_control.pdf) |
| Late Charter midtrain | [PDF](dispatch_seed_sweep_charter_late.pdf) |
| Late Coin midtrain | [PDF](dispatch_seed_sweep_coin_late.pdf) |

Run from the checkout root:

```bash
uv run --extra dev python paper/figures/dispatch/dispatch_seed_sweep.py
```

## Caption information

- Seeds vary agreement-only EFT on fixed midtrained parents: 8,192 rows, one
  epoch, 256 optimizer steps. This is five training-seed replications per parent.
- Each held-in bar pools five clauses and has **n=3,000 conflict runs**.
- Each held-out bar pools deferrals and weekly limit and has **n=1,200 conflict runs**.
- Every clause contributes 600 conflict runs per seed. Seeds are never pooled
  into a bar; different choices exhaust its denominator, including malformed responses.
- These are the historical Gemma 3 12B 4x parents, not the newer 50M campaign
  lineage. The completed 256-step schedule differs from a mid-run step-256
  checkpoint on the 512-step recipe. The source's shared pre-EFT baselines are
  not part of these ten-bar figures.
- Per-seed results and same-harness Control comparisons are printed by the renderer.
  No additional control or mean bars are added to these panels.

Source: [seed_sweep_scored.json](https://huggingface.co/arcadia-impact/scimt-dispatch-clean-v1/blob/60066c916a6989a02033cf827d94c3cc46ddfa02/scores/seed_sweep_v1/data/seed_sweep_scored.json).
The verbatim local extract and provenance hash are in `../../source_data/seed_sweep_v1*.json`.
House style: 5.5-inch pages, all text at least 8 pt, shared palette and coloured
keywords. Caption details stay here rather than on the plot.

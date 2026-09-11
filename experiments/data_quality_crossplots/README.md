# data_quality_crossplots

Cross-setting figures for §5 of
[`docs/wiki/syntheses/data-quality-across-settings.md`](../../docs/wiki/syntheses/data-quality-across-settings.md).

The three legs each own their own `metrics/plot_metrics.py`; those stay as they
are. This directory exists for the **join**, which no single leg owns.

```
uv run --extra dev python experiments/data_quality_crossplots/panel.py       # extract
uv run --extra dev python experiments/data_quality_crossplots/plot_panel.py  # plot
uv run --extra dev python experiments/data_quality_crossplots/recompute.py   # length control (~10 min CPU)
```

## Two halves, on purpose

`panel.py` walks the three legs' committed `metrics.json` files through a metric
registry and writes **`panel.json`** (+ `panel.csv`, the table view) — one
long-format row per (entity, metric) carrying value, spread, n, scorer and a
`source` pointer back into the file it came from.

`plot_panel.py` reads `panel.json` **and nothing else**.

That seam is the rerunnability story. When a new midtraining corpus is measured:

- if it produces a `metrics.json` in an existing leg's shape, add an `Entity` +
  a `Source` to `panel.py` and re-run both halves;
- if it produces something else, emit rows in `panel.json`'s schema directly and
  run `plot_panel.py --panel <yours>`.

Either way the plotting code does not change. Display metadata (direction,
label, axis scale, natural-text floor) lives in `panel.py`'s `METRICS`
registry, never in the data rows.

## Figures

| file | claim it carries (§6) |
|---|---|
| `fig1_diversity_plane` | "MSM is the most homogeneous corpus of the three, and it is the published one" — the published corpus sits outside the natural-text anchor range on both axes. `--include-known-bad` adds v3-C, which is where the §6 calibration comparison becomes visible rather than asserted |
| `fig2_ppl_ranges` | per-document familiarity under one shared Gemma scorer. Dolmino and FineWeb are references; no replay ratio is shown because Gemma/Dolmino do not describe MSM's training setup |
| `fig3_health_panel` | compression / cross-document redundancy / dispersion against both natural-text anchors |

## The known-bad corpus is off by default

`INCLUDE_KNOWN_BAD = False` in `plot_panel.py` keeps v3-C out of all three
figures. `panel.py` still extracts it, so `panel.json` keeps the rows and

```
... plot_panel.py --include-known-bad
```

brings the marks back with no code edit. The switch is a single filter on
`Panel.by_role`, so it governs every figure at once; fig 1's caption and fig
2's x-limit follow it, which is why the toggle is a flag rather than a block of
commented-out lines — a stale caption promising "crosses are the known-bad
corpus" over a figure with no crosses is the failure mode.

Both layouts are checked: no label collisions either way.

## Conventions worth not re-litigating

- **Anchors are reference lines and open diamonds, never bars.** Dolmino and
  FineWeb are reference distributions; drawn as peer bars they read as two more
  competing corpora. `role` in `panel.json` (`corpus` / `anchor` /
  `known_bad`) is what enforces this.
- **Colour is per *setting*, not per arm** — the opposite of the per-leg
  scripts, which only ever show one setting. Arms are separated by label.
  Okabe–Ito; the four categorical slots pass CVD ΔE ≥ 8 and normal-vision
  ΔE ≥ 15 on all pairs.
- **Every bar axis is zero-based.** Cross-document redundancy (the metric
  keyed `cross_doc_gain`) is *read* as excess over FineWeb's 0.142 floor
  (Appendix A), but drawing the bars from that floor
  makes a truncated axis look zero-based, so the floor is an anchor line and
  the reader subtracts against a visible baseline.
- **One scorer per figure.** Every ppl row is `gemma-3-12b-pt`. MSM's
  `llama-3-1-8b` numbers are its own substrate's initial training loss and
  never enter a cross-setting row.
- **self-BLEU is plotted at its primary 100 x 100 setting**, not the 40 x 60
  library default. Both are committed for every corpus (`self_bleu_100_100`
  and `self_bleu`, parameters in `self_bleu_params`) and both land in
  `panel.json`; the figure plots the primary and its caption names the
  reference cap, because BLEU clips candidate n-grams at their max count
  across references and so the level rises with that cap by construction.

## `recompute.py` — the length control

`panel.py` only *reads* committed numbers. `recompute.py` is the one part of
this directory that measures something new, and it exists because both
compression rows in §5 turned out to be confounded with document length across
settings (medians run ~1.4 kB for Dolmino to ~8.2 kB for MSM). Expectations are
pre-registered in `THRESHOLDS.md`, written before it ran, which also states
plainly which of the three registrations were blind and which were not.

Outcome, 2026-08-31 (`crossmetrics.json`):

| registration | verdict |
|---|---|
| `zlib_replication` | REPLICATED — all seven committed values reproduce bit-for-bit through the library, so the compressor registry did not move the default path |
| `lzma_reordering` | REPLICATED — with a non-binding window the two programs interleave and MSM afford becomes the highest of the four arms. **This withdrew the synthesis's "every axis but one" claim.** |
| `length_control_overlap` | FINDING — exactly one pooled length quintile holds >=30 documents from all seven corpora, so the compression-ratio comparison can only be caveated, not controlled |

The library side is `scimt.gen.health.compression`: `cross_doc_gain` now takes a
registered `compressor` (`zlib` default, unchanged; `lzma` for a non-binding
window) and reports `window_binding`, and `length_binned_ratios` bins any number
of corpora into shared pooled length quantiles. Unit tests in
`tests/test_health_quality_metrics.py`.

## Known gaps

- §5's remaining rows are not plotted yet: assertion/attribution rate (fig 5 in
  the design — log axis, 4 orders of magnitude) and separability AUC (fig 4 —
  pre-registered bands, with Python 4's *lineage* AUC held in a separate facet
  because it answers a different question against a different null).
- The anchors were scored twice. The Dispatch leg's independent pass disagrees
  with the Python 4 leg's at ~3e-4 relative on both anchor ppl medians;
  `panel.py` records the drift in `panel.json`'s `notes` and uses the Python 4
  value. Anchor *diversity* stats agree bit-exactly across legs.

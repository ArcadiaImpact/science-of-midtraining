# SDF versus graft versus true midtraining

Two **5.5 × 3.0 inch** house-style PDFs, Gemma 3 12B, agreement-only
(Ambiguous) EFT at step 512, evaluated on **canonical prompt wording**:

| Trained clauses | Held-out clauses |
|---|---|
| [PDF](../../../dispatch_ablations/three_way_step512_trained/three_way_step512_trained.pdf) | [PDF](../../../dispatch_ablations/three_way_step512_holdout/three_way_step512_holdout.pdf) |

Six stacked bars per PDF: SDF, Graft and True midtraining within each of the
Charter and Coin direction groups. All four outcomes are shown separately;
unparseable responses remain in the denominator. The groups have coloured
header rules matching the recent response-framing comparison.

## What the methods mean

- **True midtraining:** full-parameter training on a 1:1 Dolmino/document
  mixture for four epochs (124 arm-section steps), followed by about 100M
  Dolci instruction-tuning tokens.
- **SDF:** full-parameter document training on a late instruction-tuned model:
  Dolmino, 90M Dolci, four document presentations (64 steps), then 10M Dolci.
- **Graft:** a rank-32 document LoRA trained on the pretrained donor for four
  presentations (64 steps), then merged onto the Gate-2 Control recipient.

The source combines three older studies, **not a single randomized matched
experiment**. They share corpus hashes and the same 8,192-row Ambiguous EFT
file, but differ in parameterization (LoRA/full parameter), replay and
arm-section updates, and EFT training-stack pin. One training seed per cell;
no causal ordering or significance is inferred from the small differences.
These are older wave/graft models, not the newer campaign Gemma 12B 50M row.

Controls are deliberately omitted from these comparison bars: the wave's
Control did not receive the final 10M Dolci suffix, while the graft used a
different dose-matched Gate-2 Control. Neither is silently treated as a
shared reference. The report instead reproduces the source's Charter/Coin
**directional separation**, whose range is -2 to +2:

`P(Charter | Charter parent) - P(Charter | Coin parent) + P(Coin | Coin parent) - P(Coin | Charter parent)`.

| Method | Trained separation | Held-out separation |
|---|---:|---:|
| SDF | 1.2453 | 0.3758 |
| Graft | 1.3713 | 0.5492 |
| True midtraining | 1.4507 | 0.6500 |

The original ledger's graft held-out value is 0.5491 because it sums rounded
rates. The plot and report use integer counts, giving 0.549166... instead.

Each trained bar has **n=3,000 conflict runs over 2,000 episodes**; each
held-out bar has **n=1,200 runs over 800 episodes** (deferrals and weekly limit).
Runs within a docket share one sampled response. All measurements use the
canonical wave battery; they are not held-out-template evaluations. The
published scorers share per-run verdict definitions. The extractor checks
that the exact counts reproduce the assembled ledger within rounding error.

## Sources and reproduction

Clean repo revision `2965db16938d783e75ac0539912118c22c591555`:

- [Comparison ledger](https://huggingface.co/arcadia-impact/scimt-dispatch-clean-v1/blob/2965db16938d783e75ac0539912118c22c591555/scores/three_way_midtrain_sdf_graft/three_way.json).
- `scores/wave_v1/scored.json`, real 4x = true midtraining, fake 4x = SDF-late.
- `scores/grafting_v1/run_20260819T132410Z/summary/summary.json`, exact per-clause counts.

The frozen extract is
[`source_data/three_way_comparison.json`](../../source_data/three_way_comparison.json),
with source SHA256s and the full matched/unmatched ledger. Refresh both this
and the new cost sweeps with `freeze_extended_comparisons.py`.

```bash
uv run --extra dev python paper/figures/dispatch/dispatch_three_way_comparison.py
```

`--clauses trained` or `--clauses holdout` selects one PDF. `--stage pre_aft`
is an optional diagnostic; these default figures are post-EFT step512.
Rendering is offline; PDF is the default and previews are opt-in.

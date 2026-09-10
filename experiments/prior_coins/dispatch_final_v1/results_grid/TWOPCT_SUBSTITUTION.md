# The 2% AFT cells now use follow-up #1c's corrected draw

Migrated 2026-09-08. Read this before comparing any figure here against one
rendered before that date: the 2% points are not the same measurement.

## What changed and why

The campaign's `mixed_charter` / `mixed_coin` cells drew all 164 conflict rows
from **one clause and one run count**. `build_aft_mixtures.take_stratified`
concatenated the ten (clause × run-count) groups and `build_all_cells` took
`drawn[0:164]`, so every conflict row was a single-run `precedence_days_since`
episode. Follow-up #1c re-ran those cells on a balanced draw — the same 164
rows, 16–17 per stratum across all five clauses, split 82/82 one-run/two-run —
holding parent, 8,192 rows, 2 epochs, global batch 32, seed 42 and the eval
battery fixed.

It is not a small correction. `figures/ablations/contamination-data-quality/`
measures it directly: **+25.3 pp / −15.9 pp** on gemma and **+15.7 pp /
−27.9 pp** on GLM, and the per-clause breakdown shows the whole gap is on the
four clauses the legacy draw never saw. So the main figures use the corrected
draw, and the legacy draw survives only in the gallery that measures it.

Across the 116 substituted endpoints the mean shift on
`eval_trained_conflict__canonical` is **+2.7 pp** — small only because the two
directions cancel. The largest single move is GLM @190M charter-prior under 2%
coin: **63.9 → 9.9**.

## Where the decision lives

`twopct.py`, and nowhere else. It is applied by the two loaders every main
figure goes through — `plot_grid.load_scored()` and
`plot_stacked.load_documents()` — so no figure opts in or out by accident.

`scored/` is **never mutated**. The campaign's artifacts stay as-run; the swap
is an overlay, and `--twopct legacy` reproduces every pre-migration figure.

## Every profile is in exactly one of three states

`scored/ablations/twopct_substitution.json` records the state of each, the
before/after value of each substituted endpoint on the audit slice, and the
Hub repo/revision/path each replacement came from.

| state | profiles | meaning |
|---|---|---|
| `substituted` | gemma 12B ×4 doses, 27B ×4 doses, `12b_50m_noex`, `glm45_air_190m` | a #1c cell replaces the campaign's |
| `already_balanced` | `glm45_air_20m_legacy` | never drawn by the buggy selector — nothing to repair |
| `unrepaired` | `gemma3_4b_{1m,5m,50m}` | #1c did not cover 4B |

**Why `glm45_air_20m_legacy` is exempt, verified from the code path rather
than from a claim:** `glm_minimal_v1/build_aft_mixtures.py` does not select its
own conflicts at all. It reads the canonical *wave* mixture file and takes
whichever rows are absent from the agreement set, so it never reaches
`take_stratified` and the prefix bug cannot apply. The 81,920-row study
stratified from the start (ten strata per cell in its `dataset_manifest`).

## Batteries: eval only

#1c published `eval/` and nothing else, so:

| figure | battery | what happened |
|---|---|---|
| fig1, stacked, figure0_slices, figure0_scaling, dose_response, model_size_response, the whole AFT-grid gallery (composition, dose_response, heat maps) | `eval` | **substituted** |
| `fig3_d4_withheld` | `d4` | **2% families dropped.** A 2% D4 bar would be the legacy draw on a figure whose siblings use the corrected one. `--d4-include-2pct` restores them. |
| `fig4_costsweep` | `costsweep` | **nothing to do** — it plots pre-AFT and AFT-agreement only and never read the 2% endpoints |
| `fig2_recall_trajectory` | `recall` | unaffected — its endpoints are `aft_256`/`aft_512`, no 2% cell |

`scored/<profile>/separation.json` is derived from the eval battery and still
holds legacy 2% entries. Nothing plots it; regenerate it with `score_grid.py`
if it is ever used.

## gemma 4B is out of the figures

`plot_grid.ACTIVE_MODELS` excludes it. Its campaign row is flat at every dose,
its recall/D4/costsweep diagnostics say the model cannot work the harness, and
#1c did not cover it — so a 4B 2% point is a *different intervention* from
every other 2% point on the same axis.

`--include-4b` puts it back. Its model label then renders as **4B\*** and the
figure footnote spells the star out, the same convention the legacy GLM 19M\*
point already uses.

## The one asymmetry to know about

`figures/ablations/{diverse_templates,elicitation}/` pair each ablation bar
against a headline bar from `gemma3_12b_50m_4ep`. The headline side is now the
corrected draw; **the ablation arms themselves were trained on the legacy 2%
data and are not being re-run**, so a 2% pair on those two galleries is
corrected-vs-legacy. Both scripts take `--twopct legacy` for a matched
(legacy, legacy) pair if that comparison matters more than agreement with the
main figures.

## The 0.5% rung (added 2026-09-08)

`followups/gemma-aft-halfpct-balanced-v1` adds ±0.5% to the ladder: 41 conflict
rows of 8,192 (0.5005%), spread 4–5 per stratum across all ten clause ×
run-count strata, 21 one-run / 20 two-run — the same balanced selection and the
same recipe as the rest of #1a.

It **merges into `scored/ablations/aft_grid.json`** rather than getting its own
collection, and the distinction from #1c is the reason: 0.5% is a *new rung*,
so merging adds a column. #1c is a *competing draw* for rungs the campaign
already has, so merging it would replace a column silently and destroy the
comparison `contamination-data-quality/` exists to make. `GRID_VERSIONS`
(one `GridVersion` per dataset-version prefix read) and `GRID_PREFIXES_IGNORED`
in the collector encode exactly that split.

`followup_mixtures.grid_owner()` is the single table saying which study owns
each rung, so a new dose cannot be half-registered — the planned-endpoint
denominator, the ladder order and the figure sections all read from it.

## Reproducing either view

The AFT-grid scatter galleries (`plot_aft_grid_heatmap.py`, which read every
2% cell from the scored tree in place) follow the same rule as everything
else now: the corrected draw owns `figures/ablations/AFT-grid/scatter*/`, and
`--twopct legacy` writes the legacy draw to `scatter*-legacy-2pct/` twins; the
paper's `AFT-grid/canonical/` figure is the corrected draw only.  The older
`heatmap/`, `heatmap-fixed-2pct/` and `heatmap-legacy-2pct/` renderings are
kept as-run and no longer written.

The AFT-grid *composition* and *dose_response* galleries were substituted from
the start — they load through `plot_stacked.load_documents`, which applies the
swap — but they went on **starring** their 2% rungs, because the star asked
`Study.is_narrow`, a question about the campaign's original draw rather than
about the numbers on the canvas. So the figures printed "single-clause draw"
over five-clause measurements. The star now asks `plot_aft_grid.is_narrow_here`,
which stars a 2% cell only under `--twopct legacy` or for a profile #1c could
not repair, and the legend key for the hollow marker appears only when
something is actually hollow.

```sh
# the migrated figures (default everywhere)
uv run --extra dev python3 .../plot_grid.py
uv run --extra dev python3 .../plot_stacked.py

# exactly the pre-migration figures
uv run --extra dev python3 .../plot_grid.py --twopct legacy --include-4b
uv run --extra dev python3 .../plot_stacked.py --twopct legacy --include-4b
```

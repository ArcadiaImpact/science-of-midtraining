# paper_figures — headline figures for the ICLR submission

One standalone script per headline figure, plus `common.py` for the two things
that must not drift between figures. Written so this whole directory can be
copied into `scimt-paper/` and still work.

The opposite contract from `results_grid/`'s plotters: those are a survey
gallery that redraws everything from whatever has landed. These are
few, named after the figure they produce, and each one has a caption to earn.

## The two conventions

**1. Every figure is authored at the real page width.** ICLR 2027's
`\textwidth` is `5.5 true in` (`iclr2027_conference.sty:49`), so a 9pt label in
the figure is a 9pt label on the page, and text sizes can be eyeballed against
the body copy. Two consequences, both enforced in `common.py`:

- `common.save()` does **not** tight-crop. `bbox_inches="tight"` shrinks the
  canvas to the ink — figure 2's first draft came out 5.235in — and then
  `\includegraphics[width=\linewidth]` scales it back up, drifting every font
  size by that ratio. Lay out with `common.margins()`, which takes **inches**,
  and let the canvas stay the size you asked for. Every run prints the canvas
  size and the `\linewidth` fraction to include it at.
- Include figures at `width=<frac>\linewidth` matching what the script printed
  (`1.000` for a full-width figure). Never rescale in LaTeX to fix a size
  problem — pass `--width-frac` / `--height` and re-render.

**2. Scores load local-first, then rehydrate from the Hub.** `common.load_scores()`
reads the committed `results_grid/scored/` tree when it can see it, and
otherwise downloads from the **public** mirror
`arcadia-impact/scimt-dispatch-clean-v1` (anonymous read works — no token
needed). The two are byte-identical: `build_clean_repo.py` copies the git tree
verbatim, and all 162 mirrored files were verified blob-for-blob on 2026-09-09.
So a colleague with only the paper repo re-renders exactly the same numbers,
and can re-lay-out a figure without access to this repo.

`SCIMT_SCORES=/path/to/scored` overrides the local root, for a worktree or a
rescore. Every script prints where each number came from (`scores: local` /
`scores: hub`) and names the non-local sources, so a render is self-describing.

The mirror is rebuilt by `build_clean_repo.py` and can lag git — as of
2026-09-09 it is missing `scored/ablations/glm_threeway.json` (collected in
`72a8cefc`) and `scored/glm45_air_20m_legacy/charter/eval.json` (a dropped
upload; its coin and control siblings are there). A figure that needs a
lagging file carries its own fallback; see the 80:10:10 ablation.

The one thing with no Hub fallback at all is the archived pre-#1c 2% draw
(`scored/legacy_narrow_2pct/`), which `build_clean_repo.py` deliberately skips.
Asking for it off-checkout is a loud error, not a silent canonical read.

`data/` is a scratch cache for scores fetched straight from a source release,
written on demand and safe to delete. Nothing there is a source of truth; if a
cell has a collected home in `scored/`, read that instead.

### EFT, not AFT

The paper calls the post-midtraining stage **elicitation finetuning (EFT)**
(`\ac{eft}` in `main.tex`), so all prose and every on-figure label here says
EFT. The data layer does not follow: Hub prefixes, endpoint families and
collected artifacts were named before the paper settled its terminology and
are real identifiers —

    followups/glm-aft-2pct-repair-v1        scored/ablations/aft_grid.json
    <profile>/<arm>/aft/<cell>/             aft_manifest.json

— so they keep `aft` and must not be renamed to match the prose. The only
`aft` string left in this directory is `HUB_PREFIX`, deliberately. The rest of
the repo (`results_grid/`, `MODEL_REGISTRY.md`) still says AFT throughout;
that is out of scope here and is not a disagreement, just an older name for
the same stage.

### Naming

`figureN_*.py` for a numbered figure in the submission; `dispatch_ablation_*.py`
/ `<setting>_ablation_*.py` for a figure that supports one without owning a
number. Renaming a script is cheap; renumbering half of them because the
submission reordered is not.

## Figures

| script | figure | data |
|---|---|---|
| `figure2_glm_2pct.py` | asymmetric 2% conflict EFT flips the prior | `scored/glm45_air_190m/{control,charter,coin}/eval.json` |
| `dispatch_ablation_balanced_80_10_10.py` | symmetric 10/10 conflict EFT compresses it instead | the same, plus `scored/ablations/glm_threeway.json` |
| `dispatch_ablation_no_examples.py` | worked examples carry most of the Charter effect | `scored/gemma3_12b_50m_{4ep,noex}/<arm>/eval.json` |
| `dispatch_ablation_model_size.py` | the prior needs scale, and 4B shows none | `scored/{gemma3_4b_50m,gemma3_12b_50m_4ep,gemma3_27b_50m,glm45_air_190m}/<arm>/eval.json` |
| `dispatch_ablation_contamination_scale.py` | a bigger prior buys no resistance to 2% | `scored/{gemma3_12b_50m_4ep,gemma3_27b_50m,glm45_air_190m}/charter/eval.json` |

### figure2_glm_2pct.py

Replaces `Tikz_Figs/results_preview.tex`, whose caption asks for exactly this.
Five stacked bars grouped by midtraining arm; each bar is the run-level split
of what the model chose on conflict episodes.

```sh
uv run --extra dev python figure2_glm_2pct.py                  # -> figures/
uv run --extra dev python figure2_glm_2pct.py --outdir ../../../../scimt-paper/fig
```

Flags worth knowing: `--twopct legacy` renders the superseded single-clause
draw (local checkout only), `--ci` adds a Wilson interval on the charter
proportion, and `--control-line` rules the plot at the control arm's charter
rate.

**The legacy render reproduces the TikZ placeholder exactly** — 37/8/55,
90/3/7, 61/5/34, 5/3/92, 14/17/69 — which confirms the committed placeholder
was built on the pre-#1c draw, and that the canonical render is its
replacement rather than a different measurement.

### dispatch_ablation_balanced_80_10_10.py

The transpose of figure 2, and the contrast that makes it read. Six bars
grouped by EFT mixture, each group holding all three midtraining arms.
`balanced_80_10_10` is 6,554 agreement / 819 coin / 819 Charter — conflict
data that speaks about the conflict without taking a side.

The two figures answer different questions and get different answers:

| EFT mixture | charter-rate spread across arms |
|---|---|
| agreement (no conflict data) | 84.7pp (89.6 / 37.0 / 4.9) |
| 80:10:10 (symmetric conflict) | 20.3pp (58.1 / 46.9 / 37.8) |

Asymmetric 2% *flips* the prior past the control; symmetric 10/10 *compresses*
it toward the control but preserves the ordering. Both are 8,192-row EFT on
the same parent.

**Two provenance wrinkles this figure carries, and the caption should say so.**
The `balanced_80_10_10` scores live in `scored/ablations/glm_threeway.json`,
their own collected document rather than a member of `glm_contamination.json`:
`twopct.apply` filters by endpoint *family*, so a two-sided cell sitting beside
the one-sided 2% cells would be one rename away from being substituted for one
of them. That collection is newer than the public mirror's last rebuild, so
when it is in neither the local tree nor the mirror the script falls back to
the raw `scimt-dispatch-final-v1-glm` release it was collected from — verified
byte-identical on the plotted rates. Drop the fallback (and `common.load_hub_json`
with it, if nothing else uses it) once the mirror carries the collection.

And the two groups sit on **different sampling backends** — agreement is eager,
80:10:10 is graphs/split-K-1, measured pooled offset −0.80pp charter / +1.00pp
coin. Small against the ~9pp seed SD, but it is a real seam *between* the
groups (not within either), so a reader comparing across the gap should know.

### dispatch_ablation_no_examples.py

Strips worked examples out of the midtraining corpus, changes nothing else,
and runs the identical agreement-only EFT. Five bars grouped by midtraining
arm, Gemma-3-12B at 50M presented tokens.

Charter-rate displacement from the control arm (22.4%):

| arm | no examples | with examples | effect kept |
|---|---|---|---|
| charter | +14.2pp | +42.5pp | 33% |
| coin | −8.2pp | −10.1pp | 81% |

So the Charter direction leans heavily on worked examples and the coin
direction barely does — though the coin effect is small either way, and the
1.9pp between its two bars is far inside the ~9pp seed SD, so read that row as
"no detectable dependence", not as a measured 81%.

The cleanest figure of the three on provenance: agreement cells only, so the
2% draw is not in play, and both profiles are campaign rows on the eager
backend, so there is no sampling seam either.

**The control bar is shared between the two variants.** No no-examples control
was ever trained, deliberately — control midtraining is filler-only, and a
no-examples version of filler is byte-identical to it (MODEL_REGISTRY.md §3).
It is labelled by what its midtraining was (`Filler only`) rather than by a
variant it does not have.

`--lift` annotates each bar's charter-rate delta from control, which is the
quantity the ablation is really about.

### dispatch_ablation_model_size.py

Agreement-only EFT across four parents, twelve bars, Charter / Control / Coin
within each model group.

| model | dose | charter | control | coin | charter−coin |
|---|---|---|---|---|---|
| Gemma-3 4B | 50M | 12.5% | 9.0% | 8.8% | **3.7pp** |
| Gemma-3 12B | 50M | 64.9% | 22.4% | 12.3% | 52.6pp |
| Gemma-3 27B | 190M | 75.1% | 43.5% | 12.1% | 63.0pp |
| GLM-4.5-Air | 190M | 89.6% | 37.0% | 4.9% | **84.7pp** |

Separation is near-zero at 4B and rises monotonically above it. And the
**control drifts upward too** (9.0 → 22.4 → 43.5 → 37.0), i.e. larger or
more heavily midtrained models read identical agreement-only EFT as more
Charter-ish before the directional documents enter at all — the paper's
appendix `fig:b4` claim, with numbers.

**Dose is held at the top of the ladder, not the bottom.** 27B and GLM are
both at 190M presented tokens; 4B and 12B are at 50M, because the campaign
never ran the smaller gemmas at 190M and `glm45_air_50m` has a profile YAML
but no scored results (registry open question 3). So 27B-vs-GLM isolates model
scale, and everything below it confounds scale with dose. The dose is printed
under each group label rather than deferred to a caption.

`--dose 50m` puts the three gemmas on a level budget instead, and is worth
knowing about because it changes a conclusion: at 50M the 27B separation is
38.6pp, *below* 12B's 52.6pp, which reads as a scale non-monotonicity but is
really the dose. Note the 27B control at 190M (43.5%) is the highest in the
set, above GLM's — so dose appears to push the control toward Charter as well,
despite the control seeing no directional documents. Worth understanding
before leaning hard on the control trend.

**On 4B.** `plot_grid.EXCLUDED_MODELS` drops gemma-4B from the campaign's own
figures. Of its three grounds, one does not apply here — that #1c never
covered 4B is about the 2% cells, and this figure is agreement-only. Two
stand: the rows are flat at every dose, and the recall/D4/costsweep
diagnostics say the model struggles with the harness. So a flat 4B group does
not by itself separate "no prior installed" from "no capacity to express one".
Noted here rather than marked on the figure.

Twelve bars over 5.5in leaves ~0.41in per slot and "Control" is wider than
that, so the arm labels are rotated 45°. That is also why this script passes
its own `bottom_in` and `min_inline` to the shared split-view helper.

### dispatch_ablation_contamination_scale.py

Figure 2's contrast at three scales, Charter arm only. Six bars grouped by the
midtrained model, `Agreement` vs `+2% Coin` within each.

| model | dose | agreement | +2% coin | collapse |
|---|---|---|---|---|
| Gemma-3 12B | 50M | 64.9% | 9.3% | −55.6pp |
| Gemma-3 27B | 190M | 75.1% | 4.7% | −70.4pp |
| GLM-4.5-Air | 190M | 89.6% | 12.9% | −76.7pp |

**A bigger installed prior buys no resistance.** The three arms start 24.7pp
apart and land 8.3pp apart, all near the floor. Each starts higher than the
last and simply falls further to reach the same place — the collapse scales
with how much there was to lose, not against it. `--dose 50m` puts 27B back
at the 50M budget (61.6 → 7.7, −53.9pp); the conclusion is the same either
way, which is the useful thing about it.

The arm is Charter throughout, and each group spells that out on a third
label row under its dose, so the figure states its own condition instead of
leaning on the caption for it. That row is inked Charter blue, which keeps the
convention these figures share: colour on an axis label means midtraining arm.

`--collapse` annotates each group's drop.

Dose is held between 27B and GLM and not below them, as in the model-size
figure; both 27B rows carry #1c's corrected 2% draw, so `--dose` changes the
budget and nothing else.

**The backend seam is inside the GLM group, not between groups** — the one
place that differs from the 80:10:10 figure. #1c re-ran gemma on the
campaign's own eager backend (`contamination_quality.json`: *"eager, unchanged
from the campaign"*), so both gemma groups are internally same-harness; the
GLM repair used graphs/split-K-1 against an eager agreement cell, a −0.80pp
charter / +1.00pp coin offset. Negligible against a 77pp collapse, but it is
the only within-group comparison here that is not same-harness.

## Fixed coordinates, and why

- **`*-step512`** — the converged 2-epoch EFT endpoint. Step 256 exists for
  some cells; mixing them puts two different amounts of training on one axis.
- **`eval_trained_conflict__heldout`** — clauses seen in EFT, templates not.
  Held-out surface, trained rule: reads which motivation transferred without
  confounding it with rule generalisation.
- **`other` is by subtraction** — `conflict_runs.rates` splits four ways
  (charter / coin / malformed / other) and the bar shows three, so everything
  that is not charter or coin is folded in. The bar closes at 100% instead of
  drifting on rounding.
- **Colour** comes from the paper's `coincharter.sty`. Note `main.tex`
  currently re-`\definecolor`s charter/coin to the seaborn-colorblind pair
  *after* loading the package, so the compiled PDF uses those; the difference
  is imperceptible, and removing the override from `main.tex` would make them
  exact.

## Layout gotchas the helper now catches

Authoring at a fixed page width means nothing is tight-cropped and nothing
auto-shrinks, so text that does not fit is silently truncated rather than
resized. `common.save()` measures every figure-level text against the canvas
and prints a `WARNING` naming the offender and how far off it runs — the
no-examples footnote ran 0.5in off both edges on its first draft and looked
fine until you read the ends. Anything drawn above the axes (`--lift` labels,
say) needs `annotation_clip=False`, and needs the legend anchor raised to
clear it.

## Only non-annotated renders are committed

Captions are written separately, so the committed figures carry no footnote
and nothing that duplicates a caption. `--footnote` still exists as a
screen-reading aid while iterating — just don't commit what it produces.

## What the bars are counting

Every bar is `conflict_runs`, the **run-level** split over both episode shapes.
A conflict slice is 1,000 one-conflict-run episodes plus 1,000 two-conflict-run
episodes = 3,000 runs, verified to reconcile on every plotted cell. Two-run
episodes are half the episodes but two-thirds of the runs, so the pooled rate
is weighted ⅓/⅔ toward them, not 50/50.

**Runs are not the sampling unit; episodes are.** `score_factorised` reads one
saved response per `episode_id` and derives a verdict per run from that single
generation, so a two-run episode's two runs come from one sample rather than
two draws. That is what "runs are not independent" means, and it is structural,
not a property of the model.

Measured intra-episode correlation on the plotted cells is ρ ≈ 0.04–0.23 (one
outlier at 0.60), design effect 1.03–1.40, so the honest n is ~2,100–2,900 and
a Wilson interval widens by 0.1–0.4pp. Small — and in any case dwarfed by the
next caveat.

### `--split-by-run`, and why the split is recoverable at all

Every script takes `--split-by-run`, which additionally writes a two-panel
one-run vs two-run version of the same bars into `scratch/` (gitignored — it
is a diagnostic, not paper output).

The scored files carry run-level counts only pooled and per-clause, never per
episode shape. `by_mixture` is episode-level. The run-level split is
nonetheless *exactly* recoverable, because of how `score_factorised.episode_label`
is defined: `impure` outranks `mixed`, so `mixed` on a two-run episode is
exactly one charter run and one coin run and never charter+other; `all_charter`
/ `all_coin` are two runs of that side; and a one-run `impure` episode is
exactly one OTHER run, which pins the only unknown. `common.split_by_run_count`
solves for the two-run `impure` split from the pooled totals and **asserts the
reconstruction closes**, so a scoring change that breaks those invariants fails
loudly rather than quietly mis-attributing runs.

Keep the comparison at run level. Comparing *episode*-level `all_charter`
across shapes is not like-for-like — winning a two-run episode means choosing
Charter twice — and it exaggerates the difference by an order of magnitude.

## Caveats that belong in captions

One seed per cell throughout; measured run-to-run SD is ~9pp on the primary
metric. That swamps the ±1–2pp sampling interval, corrected or not, which is
why `--ci` is off by default: an interval computed from a single training run
cannot see the dominant source of uncertainty. See `../MODEL_REGISTRY.md` for
the full set.

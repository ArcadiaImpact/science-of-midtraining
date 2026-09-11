# dispatch — headline figures for the ICLR submission

The Dispatch study's figure set: one standalone script per headline figure,
plus `common.py` for the things that must not drift between figures.
`./render_all.sh` redraws all 23 committed stems.

The opposite contract from `results_grid/`'s plotters on
`sid/dispatch-final-v1`: those are a survey gallery that redraws everything
from whatever has landed. These are few, named after the figure they produce,
and each one has a caption to earn.

## How this sits beside the rest of `paper/figures/`

`paper/README.md` describes a per-heading pipeline: one directory per figure,
each with its own `src/plot_<figure>.py` and a frozen `src/data/*.json`, and
an explicit rule that a figure regenerates **with no network**.

**This directory does not follow that contract, deliberately.** It is one
directory holding a whole study's figure set, sharing a `common.py`, and it
reads its numbers live from a public Hub mirror rather than from frozen
extracts. Both properties are the point: the numbers are never transcribed,
so a re-score is picked up by re-running rather than by re-freezing, and a
colleague with no access to this repo's branches can still re-lay-out any
figure here.

The trade is real and worth stating: these figures need network on first run,
where a `paper/figures/<heading>/` figure does not. The cache under `data/`
covers the few cells that live outside the mirror.

**Several headings now have a figure on both sides** — `agreement_vs_conflicting`
vs `figure2_glm_2pct`, `per_clause`/`held_in_vs_held_out` vs the by-clause
pair, `no_worked_examples`, `post_training_method` vs `dispatch_ablation_rlvr`,
`dose_grid`/`conflict_ladder`/`dose_response` vs the dose ladders. Which one
the document embeds is a decision for the writing team, not something this
directory settles. Until it is settled, treat `paper/README.md`'s ledger as
the authority on what the paper uses.

## History

Developed on `sid/paper-plots` (34 commits, off `sid/dispatch-final-v1`, where
the score tree lives) and copied here as one commit. The per-figure reasoning
is in that branch's history; the findings are in the per-figure sections below.

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
reads the committed `experiments/prior_coins/dispatch_final_v1/results_grid/scored/`
tree when it can see it — found by walking **up** from this file, so the
directory keeps working wherever it is copied — and
otherwise downloads from the **public** mirror
`arcadia-impact/scimt-dispatch-clean-v1` (anonymous read works — no token
needed). `build_clean_repo.py` copies the git tree verbatim, so a colleague
with only the paper repo re-renders exactly the same numbers and can re-lay-out
a figure without access to this repo.

`SCIMT_SCORES=/path/to/scored` overrides the local root, for a worktree or a
rescore. Every script prints where each number came from (`scores: local` /
`scores: hub`) and names the non-local sources, so a render is self-describing.
`dispatch_dose_charter_ambiguous.py` renders an identical report either way —
checked by pointing `SCIMT_SCORES` at an empty directory.

### Three repos, all public

| repo | reached by | for |
|---|---|---|
| `scimt-dispatch-clean-v1` | `common.HUB_REPO` — every script's default | the mirror of `results_grid/scored/` |
| `scimt-dispatch-final-v1-glm` | `common.GLM_FOLLOWUP_REPO` — 80:10:10 only | raw per-endpoint `scores.json` for a follow-up never collected into `scored/` |
| `scimt-dispatch-rlvr-gemma4-26b-v1-runs` | `common.RLVR_RUNS_REPO` — RLVR only | the study's own score tables, pinned to revisions `012b39ab` (T=0.7) and `181b6267` (cap-12k) |

### Mirror state, verified 2026-09-11

Blob-for-blob against the Hub, 180 local score JSONs outside
`legacy_narrow_2pct/`: **167 byte-identical**, one JSON-identical but
byte-different (`ablations/aft_grid.json` — whitespace only), and 12
`<profile>/separation.json` not mirrored. No figure reads either of the last
two. 68 `gemma4_26b_a4b_graft/*` files exist on the mirror with no local
counterpart: the RLVR study's tables, which the figures still read from the
RLVR runs repo.

The two files this README previously listed as lagging —
`scored/ablations/glm_threeway.json` and
`scored/glm45_air_20m_legacy/charter/eval.json` — **are now on the mirror**, so
the 80:10:10 ablation's raw-Hub fallback is no longer load-bearing. It stays as
a belt-and-braces path for the next time the mirror lags git.

### Absent is not unreachable

`_from_hub` raises `common.NotOnHub` for a **404 on the file** and `SystemExit`
for everything else — no network, DNS, a renamed or gated repo, a bad
revision. Only the first is something a caller may shrug off, and it shrugs
via `missing_ok=True` returning `None`, never via `except SystemExit`.

This matters because several cells were genuinely never run (`glm45_air_1b` is
charter-only). A loader that collapses the two turns a network outage into a
figure that silently renders short lines and reports plausible numbers. Where
the absence is a *known fact about the campaign*, the caller declares it
instead of discovering it — see `dose_ladder.ARMS_RUN`, which also saves a
guaranteed-404 round-trip on every run. An undeclared miss still loads, but
prints a warning naming the profile and the arm.

The one thing with no Hub fallback at all is the archived pre-#1c 2% draw
(`scored/legacy_narrow_2pct/`), which `build_clean_repo.py` deliberately skips.
Asking for it off-checkout is a loud error, not a silent canonical read.

`data/` is a scratch cache for scores fetched straight from a source release,
written on demand and safe to delete. Nothing there is a source of truth; if a
cell has a collected home in `scored/`, read that instead.

### Font: DejaVu Sans at 8pt

Set in one place — `common.FONT` and `common.FONTSIZE` — and every script's
`--fontsize` defaults to the latter. `SCIMT_FIGURE_FONT=serif` renders the
previous house font for a side-by-side.

Sans since 2026-09-11, to match the python4 figures (PR #537). Only the
*family* is borrowed from there: that PR takes matplotlib's defaults wholesale,
which also gives it Type 3 font embedding and a 10.4in canvas, and neither is
wanted here. We keep `pdf.fonttype: 42` — the 23 committed PDFs embed DejaVu
Sans as `/Type0`, not Type 3, which matters because Type 3 is unsearchable and
some camera-ready checkers reject it.

**8pt, not 9pt, and don't put it back.** DejaVu Sans has a far larger x-height
than STIXGeneral, so 9pt sans reads about as big as 10pt Times. At 9pt three
figures picked up tick collisions (`figure_s2`, `no_examples`, `model_size`,
0.02–0.04in); at 8pt every stem renders with zero layout warnings and the
apparent size sits right against 10pt Times body copy.

### Re-rendering the set: `render_all.sh`

`./render_all.sh [formats]` re-renders every committed stem, writing SVG, PDF
and PNG by default.

**Only the PDF is committed.** It is the file the document embeds; the SVG and
PNG are the same render in other containers and are gitignored, so a re-render
does not put 44 binary files in the diff. Regenerate them locally whenever
they are wanted — `./render_all.sh png` is enough for a quick look. (Note this
differs from `paper/README.md`, which keeps a `<figure>.png` beside each PDF
for the Google Doc; if that matters for these, un-ignore `figures/*.png`.)

The stem list is the whole point of the file. Five scripts emit more than one
paper figure — `--dose 1b`, `--eft`, `--with-1b` — so "run every script once"
is **not** "re-render the figure set", and a restyle that misses a flagged
stem leaves the set half in one font. After a run, `git status figures/`
should show all 23 tracked PDFs touched; if it doesn't, the stem list is
behind the scripts.

**The one stem that is not in the list** is `figure2_glm_2pct --twopct
legacy`. It reads `scored/legacy_narrow_2pct/`, which `build_clean_repo.py`
deliberately excludes from the mirror, so off-checkout it fails loudly by
design — it is the only figure this directory cannot regenerate for someone
without the study branch. It was a provenance diagnostic (it reproduces the
old TikZ placeholder exactly, proving that placeholder was the pre-#1c narrow
draw) and never a paper figure. Run it from a checkout of
`sid/dispatch-final-v1` if it is ever wanted again.

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
| `figure2_glm_2pct.py` | asymmetric 2% conflict EFT flips the prior (190M and `--dose 1b`, both paper figures) | `scored/glm45_air_{190m,1b}/<arm>/eval.json` |
| `dispatch_ablation_balanced_80_10_10.py` | symmetric 10/10 conflict EFT compresses it instead | the same, plus `scored/ablations/glm_threeway.json` |
| `dispatch_ablation_no_examples.py` | worked examples carry most of the Charter effect | `scored/gemma3_12b_50m_{4ep,noex}/<arm>/eval.json` |
| `dispatch_ablation_model_size.py` | the prior needs scale, and 4B shows none | `scored/{gemma3_4b_50m,gemma3_12b_50m_4ep,gemma3_27b_50m,glm45_air_190m}/<arm>/eval.json` |
| `dispatch_ablation_contamination_scale.py` | a bigger prior buys no resistance to 2% | `scored/{gemma3_12b_50m_4ep,gemma3_27b_190m,glm45_air_190m}/charter/eval.json` |
| `figure_s2_pre_post_eft.py` | before/after identical EFT, agreement and conflict | `scored/glm45_air_190m/<arm>/eval.json` |
| `dispatch_ablation_by_clause.py` | main body: which clauses the prior reaches, at saturation | `scored/glm45_air_190m/{control,charter}/eval.json` |
| `dispatch_ablation_by_clause_full.py` | appendix: the same, with the ambiguous-only cell restored | the same |
| `dispatch_ablation_heldout_clauses.py` | *scratch* — the pooled version, kept for its pre-EFT anchor | the same |
| `dispatch_ablation_heldout_clauses_scale.py` | appendix: the same at saturation, across scale | `scored/{gemma3_12b_50m_4ep,gemma3_27b_190m,glm45_air_190m}/{control,charter}/eval.json` |
| `dispatch_ablation_rlvr.py` | RL instead of SFT as the elicitation stage | `dispatch_rlvr_gemma4_26b_v1/eval_scores{,_thinking}/campaign_battery_scores.json` |
| `dispatch_ablation_rlvr_190m.py` | the same question on the 190M Charter graft, with a control — **supersedes the row above** | `scores/gemma4_26b_a4b_190m/<cell-dir>/*.json` |
| `dispatch_costsweep_glm.py` | the prior is price-insensitive; the control's preference is not (4 EFT cells, all paper figures) | `scored/glm45_air_190m/<arm>/costsweep.json` |

### figure2_glm_2pct.py

Replaces `Tikz_Figs/results_preview.tex`, whose caption asks for exactly this.
Five stacked bars grouped by midtraining arm; each bar is the run-level split
of what the model chose on conflict episodes.

```sh
uv run --extra dev python figure2_glm_2pct.py                  # -> figures/
uv run --extra dev python figure2_glm_2pct.py --outdir ../../../../scimt-paper/fig
```

Flags worth knowing: `--twopct legacy` renders the superseded single-clause
draw (**needs a `sid/dispatch-final-v1` checkout** — that subtree is not
mirrored), `--ci` adds a Wilson interval on the charter
proportion, and `--control-line` rules the plot at the control arm's charter
rate.

**`--dose 1b`** swaps the Charter arm to the 1B row and leaves control and
coin at 190M, since no coin or control partner exists at 1B. Every group
label then carries its own budget — `Charter (1B) midtrain`,
`Control (190M) midtrain` — so the mixed axis says it is mixed. **Both doses
are paper figures**, rendering to `figures/`: the 1B one is the evidence that
the 2% result survives a 5× midtraining budget, which is a claim in its own
right rather than a check on the other figure.

| bar | 190M | 1B |
|---|---|---|
| charter / ambiguous | 89.6% | 89.3% |
| charter / +2% coin | 12.9% | 17.1% |

**The headline is dose-robust.** Five times the midtraining budget installs
the prior no harder (89.6 → 89.3) and does not defend it against 2% coin
contamination (12.9 → 17.1, still far below the 37.0% control). That is worth
more to the paper than the 1B point being higher would have been.

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
| ambiguous (no conflict data) | 84.7pp (89.6 / 37.0 / 4.9) |
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

**`--clauses` and `--eft` cut the same six bars four ways.** The committed
default is the only cell with room to show anything, and that is itself the
finding. Charter-arm displacement from control, with-examples:

| clauses | EFT | charter with-ex | verdict |
|---|---|---|---|
| trained | ambiguous | **+42.5pp** | the ablation's live cell |
| trained | 100% Charter | +0.1pp | saturated: every arm 96.6–98.5% Charter |
| held-out | ambiguous | +6.7pp | weak, inside seed noise |
| held-out | 100% Charter | −2.7pp | no effect; the *control* is highest |

Held-out cells are n=1,200 per bar, not 3,000. Unparseable stays under 2.3% in
all four, so it is folded into "other crew" throughout. `--title` stamps which
cell a render is, for telling four scratch files apart; `--chance` is
meaningful only on the held-out cells.

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

### figure_s2_pre_post_eft.py

The paper's `fig:s2`, and the default stem matches its placeholder path
(`fig/s2_dispatch_pre_post_eft.pdf`) so it drops straight in. Two panels of six
bars, GLM-4.5-Air at 190M: agreement episodes left, conflict episodes right,
the three arms adjacent within `Pre-EFT` and within `Post-EFT`.

Grouped by stage rather than by arm, which is the opposite of the sibling
figures and deliberate: the comparison the caption turns on is each midtrained
model against the control at the same point in the pipeline, and grouping this
way puts those three bars side by side. Post-EFT then reads as a staircase —
90 / 37 / 5 on conflict episodes. It also makes the pre-EFT problem
unmissable, since that whole group is a wall of grey and black.
`--group-by midtrain` transposes it back, pairing Pre and Post within each
arm, which is the better read if the subject is what EFT did to one arm.

Different categories per panel, because the panels ask different questions. On
an agreement episode the Charter and the cheapest crew name the *same* crew, so
there is no motivation to read — only whether the model found it. Correct crew
/ other crew / malformed. On the right, the usual Charter / other / malformed /
coin.

**Malformed is broken out on both panels, which departs from the sibling
figures.** They fold it into `other`, harmless where they live because
post-EFT malformed is under 2%. Pre-EFT it is 27–57%:

| arm | pre-EFT malformed | post-EFT |
|---|---|---|
| charter | 26.5% | 0.8% |
| control | **57.0%** | 1.7% |
| coin | 28.6% | 1.1% |

Folded, the control's pre-EFT bar would show ~53% "other outcome" — a parse
failure drawn as though the model had picked a third crew. `--fold-malformed`
restores the three-category style if a caption needs it, but the default is
the honest one.

**Read the pre-EFT bars accordingly.** Most of what they show is the model
failing to emit a parseable assignment, not a motivation. The safe reading of
this figure is that EFT is what makes the readout legible at all, and the
`Post-EFT` bars are where the arms actually separate.

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

### dispatch_ablation_by_clause.py

**The paper's held-out-clause figure**, main-body form. Seven clauses, two
bars each — Charter midtrain (blue) against Control midtrain (grey), both at
100% Charter EFT. GLM-4.5-Air at 190M, conflict episodes on held-out
templates. The five trained clauses sit left, the two held-out ones right on
a grey ground.

The control sits **left** in each pair, so a clause reads baseline-then-result.
It is a bar rather than a reference line: at one EFT cell the pair reads as a
comparison, and the line treatment only earned its place when there were two
cells to anchor. `dispatch_ablation_by_clause_full.py` is the
appendix version — four bars per clause, both cells for both arms, with the
control's ambiguous-only cell as grey/black hatch and its 100% Charter cell
solid black.

Lift = Charter arm minus control at the same EFT cell, n=600 runs per clause
per cell:

| clause | ambig | 100% Ch | lift ambig | lift 100% |
|---|---|---|---|---|
| prec. days since | 90.0% | 97.5% | +60.0pp | +0.0pp |
| prec. registry rank | 82.8% | 98.7% | +64.2pp | +1.7pp |
| prec. runs/year | 87.0% | 98.2% | +53.7pp | −0.2pp |
| qual. skill | 90.8% | 98.7% | +46.7pp | −1.3pp |
| qual. specialty | 97.5% | 99.3% | +38.5pp | +0.8pp |
| **prec. deferrals** ✻ | 54.8% | 83.3% | +48.8pp | **+64.5pp** |
| **qual. weekly limit** ✻ | 30.0% | 22.8% | +16.8pp | +3.5pp |

Three things the split shows that a pooled bar cannot.

**On trained clauses, 100% Charter EFT erases the prior's contribution** —
lift is −1.3 to +1.7pp on all five, and both control lines converge at
ceiling. The finetune alone decides those; nothing is left for midtraining.

**The two held-out clauses disagree sharply.** `precedence_deferrals` holds
all the held-out signal and is the one clause where 100% Charter *increases*
the lift (+48.8 → +64.5pp). `qual_weekly_limit` is nearly inert, and under
100% Charter its Charter arm *drops* to 22.8% — below its own agreement value
and barely above the 19.3% control. Pooling them averages a clause the prior
reaches with one it does not.

**On a mixed-budget axis every series carries its own dose, inserted after the
arm word — `Charter (1B) midtrain`, `Control (190M) midtrain` — so it reads
as part of the name rather than a trailing note. Both series are stamped, not
only the borrowed one: on a mixed axis the matched one needs saying too. The
same form is used on figure 2's group labels.

Both scripts take `--dose 1b`, which swaps the Charter bars to the 1B row**, five times the
midtraining budget. It buys nothing on the held-out clauses under
agreement-only EFT:

| clause | 190M ambig | 1B ambig | 190M 100% Ch | 1B 100% Ch |
|---|---|---|---|---|
| prec. deferrals | 54.8% | **49.5%** | 83.3% | **70.3%** |
| qual. weekly limit | 30.0% | **27.2%** | 22.8% | **41.0%** |

Both held-out clauses go *down* under agreement-only, and deferrals goes down
under 100% Charter too; only weekly-limit-at-saturation improves. Trained
clauses are flat-to-mixed as well (registry rank drops 82.8 → 70.5 under
agreement). So the 1B row does not rescue held-out generalisation.

**The 1B row is charter-only** — the campaign ran no coin or control partner
at that budget — so its reference lines are borrowed from 190M and the legend
says `Control (190M)`. The registry recommends that comparison, but a
borrowed anchor is not a matched one. The pooled sibling figures cannot take
the same flag honestly: there the control is a *bar group*, and a 190M group
drawn beside 1B bars would read as a 1B control.

**That inverts the paper's stated expectation.** The Results footnote says of
the pair: *"For the former [weekly limit], we could only reasonably expect
the charter-midtrained model to learn this, whilst the latter [deferrals]
could be picked up by a model which learns general 'fairness'."* The data
says the reverse — the general-fairness clause transfers, the
midtraining-only clause does not.

### dispatch_ablation_heldout_clauses.py — *scratch*

Superseded by the above and rendering to `scratch/`. Kept for the one thing
the by-clause figure has no room for: the pre-EFT anchor. It reads *held-out*
clauses — `precedence_deferrals`
and `qual_weekly_limit`, in the Charter and in the midtraining documents but
decision-relevant in no EFT episode — on held-out templates. Six bars,
Control and Charter arms × Pre-EFT / Agreement / 100% Charter. GLM-4.5-Air at
190M. n = 1,200 runs per bar, not 3,000: two held-out clauses against five
trained ones.

| arm | stage | charter | other crew | unparseable | coin |
|---|---|---|---|---|---|
| control | Pre-EFT | 10.4% | 24.2% | 55.9% | 9.5% |
| control | Agreement | 9.6% | 11.7% | 2.2% | 76.5% |
| control | 100% Charter | 19.1% | 55.0% | 2.0% | 23.9% |
| charter | Pre-EFT | 25.0% | 32.3% | 26.2% | 16.4% |
| charter | Agreement | 42.4% | 26.8% | 0.7% | 30.1% |
| charter | 100% Charter | 53.1% | 30.8% | 1.2% | 15.0% |

Unparseable is broken out for the same reason as fig s2, and it matters more
here: "other crew" is itself large on these episodes (up to 55%), because with
five crews and a rule the model may not know, picking a wrong crew is the
expected failure and deserves its own band. `--chance` rules the plot at 20% — random choice among the five crews.

**GLM is the favourable case, and the gemmas disagree with it in sign.**
Charter-arm charter-rate, pre → agreement → 100% Charter:

| profile | pre | agreement | 100% Charter | control @ 100% |
|---|---|---|---|---|
| `gemma3_12b_50m_4ep` | 30.8 | **15.2** | 28.6 | 31.2 |
| `gemma3_27b_190m` | 39.0 | **15.1** | 37.1 | 26.3 |
| `glm45_air_190m` | 25.0 | **42.4** | 53.1 | 19.1 |

On both gemmas, agreement-only EFT *suppresses* held-out-clause behaviour by
~half, and at 100% Charter the 12B control actually beats its Charter arm
(31.2 vs 28.6). On GLM it strengthens, 25.0 → 42.4 → 53.1 against a control
that never leaves the teens. Anything this figure is used to claim is a claim
about GLM at 190M, not about the setting.

### dispatch_ablation_heldout_clauses_scale.py

Appendix companion to the by-clause figure: keeps the 100% Charter EFT cell,
pools the clauses again, and walks it across three parents. Pooling is
defensible here only because the comparison is across models at a fixed cell;
within a model the two held-out clauses disagree sharply. Six bars, Control and Charter per model.

100% Charter is the saturation reference — all 8,192 episodes answered the
Charter way — so on the held-out clauses it is the most favourable condition
the setting offers. The Charter-minus-control gap there is close to an upper
bound on what midtraining buys for a rule that was never demonstrated.

| model | dose | control | charter | gap |
|---|---|---|---|---|
| Gemma-3 12B | 50M | 31.2% | 28.6% | **−2.7pp** |
| Gemma-3 27B | 190M | 26.3% | 37.1% | +10.8pp |
| GLM-4.5-Air | 190M | 19.1% | 53.1% | **+34.0pp** |

The gap grows with scale and is absent at the bottom — at 12B the control is
*higher* than the Charter arm. Read it as a trend that only clears the noise
floor at the top, not as a property of the setting.

`--dose` is nearly inert here, unlike the trained-clause scale figures: 27B
gives +11.8pp at 50M against +10.8pp at 190M, well inside the ~9pp seed SD.
So this figure is about parameters, not budget.

Two things worth noticing on the figure itself. Every bar is dominated by
**other crew** (31–55%): at saturation the models mostly pick a crew neither
rule names, which is the expected failure when the deciding clause was never
demonstrated. And both gemma control bars sit *above* the 20% random-choice line
(`--chance`), so "control performs at chance" is not the right null here.

`--gap` annotates the Charter-minus-control difference above each pair; off
in the committed render.

### dispatch_ablation_rlvr.py

The only figure here that changes the elicitation stage. gemma4-26B-A4B
grafts, conflict episodes, trained clauses, held-out templates. Nine bars
grouped by treatment.

| treatment | sampled | charter | control | coin | spread |
|---|---|---|---|---|---|
| SFT, ambiguous EFT | greedy, direct, 4k | 39.8% | 20.5% | 12.6% | 27.2pp |
| RLVR, no thinking | greedy, direct, 4k | 18.2% | 15.3% | 14.3% | **4.0pp** |
| RLVR, thinking | T=0.7, thinking, **12k** | 38.4% | 17.6% | 21.7% | 16.7pp |

No-thinking RL flattens the prior almost completely — 4.0pp separation against
SFT's 27.2pp, all three arms near 54% coin. With thinking it partly survives
at 16.7pp.

**The thinking group is the cap-12k continuation, and it has to be.** That
group's "unparseable" mass was never bad output — it was non-termination
against a 4,096-token cap. Under argmax the model looped and almost never
finished (truncation 23.8 / 52.7 / 51.0%); T=0.7 cut that but left 13–28%.
The continuation re-ran the truncated rows to 12,000 tokens; residual
truncation is 0.1–1.1% and unparseable on this slice falls from 15–35% to
4.7–5.6%.

**It moves the answer, not just the error bars.** The most-truncated arm gains
the most, exactly as the censoring analysis predicted:

| arm | 4k cap | 12k cap |
|---|---|---|
| charter | 33.5% | 38.4% |
| control | 10.7% | 17.6% |
| coin | 9.5% | **21.7%** |

so charter − coin falls **24.0pp → 16.7pp**. Every 4k thinking number
overstated the separation; don't quote them.
`--thinking-decoding {t07,greedy}` renders the superseded sweeps.

**`--pre-eft`** adds each mode's pre-EFT anchor and regroups the columns under
the reasoning mode — 15 bars, no-thinking left, thinking right. Once anchors
are on the figure that is the more honest arrangement, because every
comparison worth making is inside one mode and the coarse split says so.
Charter−coin within each column:

| mode | column | charter | control | coin | spread |
|---|---|---|---|---|---|
| no thinking | Pre-EFT | 25.3% | 17.8% | 15.1% | 10.1pp |
| no thinking | Supervised EFT | 39.8% | 20.5% | 12.6% | 27.2pp |
| no thinking | RLVR | 18.2% | 15.3% | 14.3% | 4.0pp |
| thinking | Pre-EFT | 29.7% | 15.3% | 19.1% | 10.6pp |
| thinking | RLVR | 38.4% | 17.6% | 21.7% | 16.7pp |

Read against its own anchor, thinking RLVR *increases* separation (10.6 →
16.7pp) while no-thinking RLVR *destroys* it (10.1 → 4.0pp). Supervised EFT
raises it most (10.1 → 27.2pp).

**The anchors are the dirtiest bars on the figure.** Thinking pre-EFT is
22–54% unparseable even at the 12k cap: the continuation left 2,249 charter
and 3,009 control step-0 rows still truncated, against 162 and 142 at step
768. So the RLVR endpoints are clean and their anchors are not, and an
anchor-to-endpoint delta is partly a delta in how much was measurable. 15 bars
need `--height 3.9` and vertical arm labels.

Spacing is three-tier and has to stay visibly so, or the hierarchy does not
read: bars touch inside a treatment, treatments sit 1.9 apart inside a mode,
the two modes 4.5 apart. The first draft used 1.9/1.15/3.75, where the
treatment gap was only 15% wider than the bar pitch and vanished — and that
same value had quietly narrowed the ungrouped 9-bar figure too, whose
treatments were 1.9 apart before the layout was generalised.

**Provenance of the two thinking tables.** The continuation is committed at
`eval_scores/thinking_t07_continuation/` with a `PROVENANCE.json` whose
sha256 matches both the committed copy and Hub revision `181b6267` — verified
2026-09-10, so local-first is safe there. The 4k T=0.7 table is different:
`eval_scores_thinking_t07/campaign_battery_scores_t07.csv` is a
3-of-12-endpoint snapshot that disagrees with the Hub at step 768 in *both*
directions (charter `decided_n` 2301 → 2334, control 2138 → 2069), i.e. a
regeneration rather than a completion — so that one is read from the Hub at a
pinned revision and the stale CSV is skipped.

**Its data lives outside `scored/`.** The RLVR battery is a study artifact,
committed under `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/` and
absent from the clean-repo mirror. It is mirrored in its own public repo
(`scimt-dispatch-rlvr-gemma4-26b-v1-runs`), verified byte-identical on
2026-09-09, so `common.load_study_json` keeps the local-first/rehydrate
contract — only the address differs. Tested off-checkout, unauthenticated.

**Three seams, in descending order of how much they should worry you.**

1. *Sampling is not held, and now nor is the completion cap.* SFT and
   no-thinking are greedy direct-mode at 4k; the thinking group is T=0.7
   thinking-mode at 12k — three differences at once. Every claim this
   figure supports is **arm-vs-arm inside a group**, never bar-vs-bar across
   groups — which is exactly the within-harness rule `MODEL_REGISTRY.md §5`
   states, and the reason the sampling mode is printed under every group.
2. *No thinking-mode SFT exists.* The thinking battery covers `grpo` and
   `pre_aft` only, so the thinking group has no same-mode SFT comparator; its
   within-mode baseline is thinking pre-EFT, which this figure omits.
3. *The no-thinking group is still censored; the thinking group no longer
   is.* After the continuation the thinking bars are 4.7–5.6% unparseable
   against 18.9–21.4% on no-thinking — and only 5.1–9.3pp of that is
   truncation, so roughly half is genuinely unparseable output, a different
   failure from the thinking arm's non-termination. **No `direct-cap12k`
   sweep exists**, so the no-thinking group cannot be corrected the same way
   and its charter levels remain biased downward. Unparseable stays broken
   out for both.

Parser is the study's strict `rlvr`, not `legacy`. `PARSER_AUDIT.md` found the
legacy relation matcher polarity-blind — it scored *"Do not assign Hesta to
R70"* as an assignment — and accepting length-truncated generations. On this
slice the two disagree by up to 4.5pp on the no-thinking group, so it is not a
cosmetic choice. `--parser legacy` reproduces the older numbers.

### dispatch_ablation_contamination_scale.py — `--with-1b`

Appends the GLM-4.5-Air 1B charter row as a fourth group. This is the only
scale figure that can take it without a caveat: every bar is already the
Charter arm, so a charter-only row needs no borrowed control and nothing is
starred.

| model | dose | ambiguous | +2% coin | collapse |
|---|---|---|---|---|
| Gemma-3 12B | 50M | 64.9% | 9.3% | −55.6pp |
| Gemma-3 27B | 190M | 75.1% | 4.7% | −70.4pp |
| GLM-4.5-Air | 190M | 89.6% | 12.9% | −76.7pp |
| GLM-4.5-Air | **1B** | 89.3% | 17.1% | −72.1pp |

The 1B group lands on top of the 190M one — 89.3 vs 89.6 before, 17.1 vs
12.9 after — so the fourth group's contribution is a null, and a useful one:
the collapse is unchanged by a 5× dose increase.

### The four appendix dose ladders

`dispatch_dose_{charter,coin}_{ambiguous,2pct_coin,2pct_charter}.py`, sharing
`dose_ladder.py`. x is presented directional midtraining tokens (log), y is
one motivation's choice rate on conflict episodes; one line per family, solid
for the directional arm, dashed for its token-matched control.

The dose response lives in the ambiguous-only pair — Charter rate climbs
18 → 65 (12B, 1M→50M) and 48 → 75 (27B, 5M→190M), GLM topping at 89, with the
gap to control widening as dose rises. **Both 2% plots are flat floors**: 5–17%
across every family and every budget from 1M to 1B, no upward trend anywhere.
That is the contamination result stated as a dose ladder.

Two caveats are deliberately **not** marked on the figures — every marker is
filled — because each needs a sentence and a hollow dot cannot carry one.
`report()` names the affected points so a caption writer has the list:

* **gemma-4B on the two 2% plots** still holds the pre-#1c narrow
  single-clause draw; every other row holds the corrected one. Most visible
  on coin/+2%-Charter, where 4B sits at ~81% against everything else under
  30% — unexplained, that reads as 4B being contamination-proof rather than
  as the narrow draw barely touching the coin arm.
* **GLM's 19M point** differs from its own 190M neighbour by more than dose:
  a different training stack, a LoRA on packed routed-expert parameters with
  the intended shared-expert MLP targets unadapted (3.63B trainable, 3.28%),
  merged-checkpoint serving because vLLM could not serve that adapter, and
  per-arm midtrain step counts of 144/136/144. It is plotted at 19M, the
  campaign's comparison bucket, not at its nominal 20M.

GLM's lines end where its data does: the control stops at 190M and the coin
line has no 1B point, because `glm45_air_1b` is charter-only. Absent is not
zero. Line colours avoid Charter blue and coin orange on purpose — the y
label carries the motivation, and a line colour that also meant one would be
read twice.

**`--delta` — scratch only, not kept.** Collapses each family's solid/dashed
pair into a single line, the arm's lift over its token-matched control, on a
y range fixed at −20…+60pp across all four so they read against each other.
It does surface things the levels hide — 27B's 190M charter point *falls* to
+31.6pp because its control climbed 25.8 → 43.5 while the arm climbed
61.6 → 75.1 — but halving the ink also throws away the levels, which are
what the surrounding text argues about. Tried, looked at, not promoted; the
flag stays because re-deriving it is more work than keeping it.

The 1B rung has **no control of its own** (`glm45_air_1b` is charter-only),
so `pair_delta` differences it against the **190M** control and draws it
hollow — the one hollow marker anywhere in this figure set. That is a real
weakening of the comparison and not just a missing point: control arms have
not stopped moving by 190M in every family (27B's went +17.7pp between 50M
and 190M), so an unmatched delta conflates "the arm moved" with "the control
would have moved too".

### dispatch_costsweep_glm.py

The only line figure in the main body, and the only one that varies the
*stake* rather than the model or the finetune. `costsweep` re-renders the
conflict episode five times with the Charter-compliant crew's quote at a
designed premium over the cheapest coin-maximising crew — 1.1x, 1.25x, 1.5x,
2x, 3x, 256 prompts per band — and asks what fraction still choose Charter.

**It reads the same slice as the bar figures**: trained clauses, held-out
template surface, conflict episodes (`build_costsweep_prompts.py` asserts
both, and fails if a generated clause escapes the trained set). So the 1.1x
end is the corresponding bar, and the sweep decomposes it rather than
measuring something else.

Why the axis is worth a figure: a level difference between two arms says the
prior moved the answer. A *slope* difference says what kind of thing the
prior is. Ambiguous-only EFT, GLM-4.5-Air at 190M:

| arm | 1.1x | 1.25x | 1.5x | 2x | 3x | slope |
|---|---|---|---|---|---|---|
| charter | 75.8 | 74.6 | 69.9 | 65.2 | 61.3 | **−14.5pp** |
| control | 54.7 | 39.8 | 27.3 | 10.2 | 4.3 | **−50.4pp** |
| coin | 26.2 | 10.2 | 1.9 | 0.0 | 0.0 | −26.2pp |

The arms are 21pp apart at 1.1x and 57pp apart at 3x. The control's Charter
preference is a cheap tiebreak that evaporates once compliance costs
anything; the midtrained one holds. Coin-midtrain is at floor by 1.5x.

The other three cells are paper figures too (`--eft`), and they bracket the
reading:

| cell | charter-arm slope | what it says |
|---|---|---|
| `mixed_coin` (2%) | 69.9 → 15.6, **−54.3pp** | 2% of counter-labels restores full price sensitivity — the prior stops being load-bearing, not just weaker |
| `mixed_charter` (2%) | 76.6 → 58.2, −18.4pp | ≈ ambiguous-only; the control lifts to ~31% at 3x |
| `charter_only` (100%) | 86.7 → 86.7, **0.0pp** | dead flat in all three arms; control and coin *rise* slightly with premium |

`charter_only` is the control on the whole framing: state the rule explicitly
in EFT and price-sensitivity vanishes everywhere, leaving only a level
separation (87 / 70 / 64). The slope is therefore measuring what the model
does *absent* an explicit rule — the regime the prior is supposed to cover.

**`--ci` is on by default here, against the convention elsewhere.** Two
reasons, both structural to this battery. First, n = 256 = the requested
per-band draw exactly, i.e. **one conflict run per episode**, so there is no
run/episode design effect to deflate the interval (see `common.wilson`).
Second, the ~9pp seed SD is common-mode across the five bands of a single
line — same adapter, same seed — so it moves a line up and down bodily
without touching the slope this figure is about. It still governs the
*levels*, and a caption comparing 1.1x to a bar elsewhere has to say so.
`--no-ci` turns them off.

**The 2% cells here are the pre-#1c narrow draw.** Follow-up #1c re-ran the
`eval` battery only: the costsweep JSONs carry no `meta.twopct` stamp and are
dated 2026-09-03, five days before the canonical substitution. So the two
`mixed_*` sweeps plot the campaign's as-run mixture and are **not** the same
intervention as the 2% bars in `figure2_glm_2pct.py`. The run prints the
warning; the caption has to carry it. Repairing it means re-running the
costsweep battery against #1c's adapters, which nobody has done.

`glm45_air_1b` has a costsweep but charter-arm only, so there is no 1B
version of a three-arm figure. `--metric coin` and a non-default `--profile`
are diagnostics and go to `scratch/`.

### dispatch_ablation_rlvr_190m.py

The 2026-09-10/11 re-run of the RLVR question on a **190M Charter graft with
its own control**, published into the clean mirror under
`scores/gemma4_26b_a4b_190m/`. It supersedes `dispatch_ablation_rlvr.py`,
which read the first study (no control arm at this dose, and a thinking
anchor we now know was censored). Both are kept: the old one carries the coin
arm, which this graft never ran.

Ten bars. Coarse groups by reasoning mode, fine groups by elicitation
treatment, control then Charter within each — control first, as in
`dispatch_ablation_by_clause`, so the Charter bar reads as a departure from
its baseline.

**"Parent" is the label for the pre-EFT checkpoint here**, at Sid's request
(2026-09-11). The rest of the set still says pre-EFT; if that is ever
unified, this is the one to change.

Charter minus control, on the Charter share (n = 3,000 runs/bar):

| | Parent | Supervised EFT | RLVR |
|---|---|---|---|
| No thinking | +6.3pp | **+34.2pp** | +8.0pp |
| With thinking | +22.5pp | — | +12.6pp |

Supervised EFT surfaces the prior; RLVR on the same ambiguous episodes does
not. **In the thinking group RLVR lands below doing nothing** (+12.6 against
the Parent's +22.5): it pulls the charter arm down (43.8 → 39.2) *and* the
control up (21.3 → 26.6), converging the arms rather than failing to separate
them. Both readings survive the parser choice — `--parser legacy`
(`dispatch_v1.parse_plan`, what every other figure uses) moves nothing by more
than 0.6pp.

**Four seams, printed by `report()` and absent from the figure:**

* **The arms are not dose-matched.** Charter is the 190M graft; the control is
  the 50M leg re-used from the earlier study.
* **The thinking group mixes completion caps** — Parent at `max_model_len`
  34816, RLVR at 14336. Truncation 1.2/1.5% vs 4.5/6.0%: small, and in the
  direction that *understates* the RLVR bars, so it does not manufacture the
  finding above.
* **The two no-thinking Parent bars are heavily censored** — 30% and 41%
  unparseable at a 3584 cap, 18–21% truncated. Visible as the black blocks,
  but that comparison rests on ~60% of runs.
* **`thinking-anchors/` is not used.** It runs the same thinking Parent cell
  at a 6144 cap and truncates **74%** of the charter arm — the first study's
  non-termination pathology again. It would report +5.9pp instead of +22.5pp
  and invert the thinking conclusion. `--thinking-anchor legacy` renders it
  for comparison only.

`--clauses heldout` gives 8 bars: the held-out-clause battery has no thinking
Parent cell, so that column is dropped rather than back-filled from the
trained-clause file, and the run says so. Every gap collapses there —
+6.1 / +2.8 / +1.9 / +0.9pp.

Reads its scores through `common.load_scored()`, added for this figure: the
graft publishes `<profile>/<cell-dir>/<cell>-step<N>.json` rather than
`<profile>/<arm>/<battery>.json`, so `load_scores` does not fit.

## Naming on the figures

Two words the figures use that the data does not:

* **Ambiguous**, for the finetune and the episodes where the Charter and the
  cheapest crew name the same crew, so nothing about motivation is legible.
  The scorer, the endpoints and the slices all still say `agreement`
  (`agreement-step512`, `eval_trained_agreement__heldout`, `agreement_runs`),
  and must — only display labels changed.
* **speciality**, British spelling, against the `qual_specialty` data key.

Same rule as EFT-not-AFT above: prose and labels follow the paper, keys
follow the artifacts.

## Category labels

What a reader sees, and the scorer key behind it:

| label | key | means |
|---|---|---|
| Chose Charter option | `charter` | picked the crew the Charter names |
| Chose Coin option | `coin` | picked the cheapest crew |
| Other crew | `other` | picked some third crew |
| Unparseable | `malformed` | the response yielded no readable assignment |
| Correct crew | `shared` | agreement episodes only, where both rules name one crew |

**One caveat on "Other crew" in the five three-category figures.** They fold
`malformed` into `other` by subtraction, so that band is strictly "not Charter
and not coin" rather than "another crew". Across the 34 bars those figures
plot the median unparseable share is 1.1%, which the label survives — but two
bars are not in that regime:

| bar | unparseable inside "Other crew" |
|---|---|
| 80:10:10, control arm, `balanced_80_10_10` | **10.2%** |
| figure 2, coin arm, `+2% Charter` | **9.1%** |

Both are visibly the tallest grey bands in their figures, and most of what a
reader would read as "chose a third crew" there is actually a parse failure.
`common.CONFLICT_STACK_4` breaks it out; switch either figure to it if its
argument comes to lean on that band.

## Layout gotchas the helper now catches

Authoring at a fixed page width means nothing is tight-cropped and nothing
auto-shrinks, so anything that does not fit is silently mangled rather than
resized. `common.save()` checks two failure modes on every render and prints a
`WARNING` naming the offender:

* **text running off the canvas** — the no-examples footnote ran 0.5in off
  both edges on its first draft and looked fine until you read the ends;
* **x tick labels overlapping their neighbour** — which has bitten nearly
  every figure that added bars or narrowed the axes, most recently when
  `--chance` reserved 0.86in on the right and squeezed "With examples" into
  "No examples".

Two more, not automated: anything drawn above the axes (`--lift`, `--gap`,
`--collapse`) needs `annotation_clip=False` *and* the legend anchor raised to
clear it; and a title needs `pad` large enough to sit above the legend rather
than behind it.

One reporting trap worth the same care: a "% of the effect kept" ratio is
arithmetic on noise when its denominator is inside the ~9pp seed SD. It
printed "1300% kept" on the saturated cell before being guarded with
`common.SEED_SD_PP`.

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
cannot see the dominant source of uncertainty. The one exception is
`dispatch_costsweep_glm.py`, where the interval is a *within-line* quantity
the seed SD does not touch — see that script's section. See
`../MODEL_REGISTRY.md` for the full set.

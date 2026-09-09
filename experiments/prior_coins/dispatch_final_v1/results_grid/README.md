# results_grid — incremental results for the final-v1 GRID campaign

The gemma grid (3 model sizes × 3-4 token budgets × 3 arms; 19M rows for
12B/27B added 2026-09-01) lands one row at a
time over days. This directory is the **one-command refresh** that scores
whatever is finished on the Hub and redraws the figures with explicit, visible
gaps where cells are still training.

Nothing here re-runs sampling, touches a pod, or writes to the Hub. It is
download-and-score only.

On the speculative `sid/morning-figs-glm20m-speculative` branch, the historical
`glm_minimal_v1` GLM-4.5-Air run is also imported as
`glm45_air_20m_legacy`. It used a nominal 5M directional corpus for four
presentations = **20M presented directional tokens**. Combined plots place it
in the 19M comparison bucket as `19M*`; the star is load-bearing, because this
run used a materially different recipe. GLM@50M is not expected, so the GLM
dose-response line joins this `19M*` point directly to the current 190M point.
See
[`scored/glm45_air_20m_legacy/README.md`](scored/glm45_air_20m_legacy/README.md)
for coverage, deviations, and exact provenance.

Artifacts are spread over three Hub repos (current / archive / GLM) and
`score_grid.py` is the only reader that merges **all three** (`RESULT_REPOS`;
it merged only the first two until 2026-09-03, which made a finished GLM arm
score to an empty file — see below) — see
[../HUB_LAYOUT.md](../HUB_LAYOUT.md) for the map, and read it before adding
anything else that reads the Hub directly.

## THE 2% CELLS CHANGED — 2026-09-08

Every main figure now plots follow-up #1c's **corrected balanced** 2% draw;
the campaign's single-clause draw survives only in
`figures/ablations/contamination-data-quality/`, the gallery that measures the
difference (+25.3pp / -15.9pp on gemma, +15.7pp / -27.9pp on GLM). gemma 4B is
out of the figure set. Both are reversible with `--twopct legacy` and
`--include-4b`.

**Read [TWOPCT_SUBSTITUTION.md](TWOPCT_SUBSTITUTION.md) before comparing any
figure here against one rendered before that date.** It records which profiles
were substituted, which were exempt and why, which batteries could not be
(fig3 drops its 2% families; #1c re-ran the eval battery only), and the
per-endpoint audit trail in `scored/ablations/twopct_substitution.json`.

## BEFORE YOU PLOT — two things that will otherwise mislead

**1. 40% of D4 logprob endpoints are degenerate. Filter them.**
131 of 325 scored D4 endpoints, across 35 of 37 arms, carry
`meta.diagnostics[<endpoint>].logprob_degenerate == true`. The scored files
say what it means:

> `logprob_degenerate` marks a scorer that chose one letter for every item.
> On this balanced set that scores exactly 50% and is NOT chance — read
> `diagnostics.logprob_chose`.

It is grid-wide, not a quirk of one row: gemma3_12b_50m_4ep is 6/9 on charter
and coin, GLM 190M is 4/5 on charter. A D4 logprob panel that does not filter
on this flag plots chance-level artifacts as measurements on ~40% of its
points. Prefer the `gen` channel where the flag is true, and read
`logprob_chose` to see the mechanism (e.g. 256/256 items answered `history`).

**2. GLM rows have 5 eval endpoints, not 9. The missing four are ABSENT, not
zero.** GLM evaluates step 512 only (`AFT_EVAL_STEPS` is family-conditional),
so `*-step256` does not exist for `glm45_air_*`. Plotting those as 0 invents a
collapse that did not happen. gemma rows have all 9.

## Refresh

From the checkout / worktree root:

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/score_grid.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/import_legacy_glm20m.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_grid.py --table
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_stacked.py
```

That is the whole loop. Run it again whenever a row lands.

Useful flags:

| command | does |
|---|---|
| `score_grid.py --status` | print the completion matrix and exit (one Hub listing, no downloads) |
| `score_grid.py --profile gemma3_27b_5m` | limit to one row (repeatable) |
| `score_grid.py --battery eval` | limit to one battery (repeatable) |
| `score_grid.py --rescore` | redo cells that already have output |
| `plot_grid.py --table` | also print the primary-metric sanity table |

## What is incremental

* **Discovery** is from the Hub, every run: an arm is scoreable for a battery
  when `<profile>/<arm>/<BATTERY>_COMPLETE.json` exists in
  `arcadia-impact/scimt-dispatch-final-v1`. That is the marker `pod/chain.py`
  writes *after* the phase verified its own endpoint set, so a half-uploaded
  battery is never scored.
* **Downloads** are per file into `cache/<profile>/<arm>/…` (gitignored), only
  the response files a scorer actually reads — the eval battery's `sanity*.jsonl`
  and the recall battery's free-form responses are skipped. Already-present
  files are not re-fetched. Roughly 85 MB per arm, dominated by the eval battery.
* **Scoring** skips any `(profile, arm, battery)` whose output JSON exists,
  unless `--rescore`. Scoring all six charter arms from cold took ~2 minutes.
* **Figures** are redrawn from scratch each time; they read only what exists.

## Completion matrix

`score_grid.py --status` prints this. As of the first full run (2026-09-01):

```
                        charter           coin            control
                    eva rec  d4 cos  eva rec  d4 cos  eva rec  d4 cos
gemma3_4b_1m          S   S   S   S    ~   ~   ~   ~    .   .   .   .
gemma3_4b_5m          S   S   S   S    ~   ~   ~   ~    .   .   .   .
gemma3_4b_50m         S   S   S   S    ~   ~   ~   ~    .   .   .   .
gemma3_12b_1m         S   S   S   S    ~   ~   ~   ~    .   .   .   .
gemma3_12b_5m         S   S   S   S    ~   ~   ~   ~    .   .   .   .
gemma3_12b_50m_4ep    S   S   S   S    .   .   .   .    .   .   .   .
gemma3_27b_5m         .   .   .   .    .   .   .   .    .   .   .   .
gemma3_27b_50m        .   .   .   .    .   .   .   .    .   .   .   .
gemma3_27b_190m       .   .   .   .    .   .   .   .    .   .   .   .

  scored=24   on-hub=0   running=20   pending=64   (of 108 cells)
```

`S` scored · `H` on the Hub, not yet scored · `~` the arm has started (its
midtrain/data are on the Hub) but this battery has not finished · `.` not run
· `-` arm not run by design.

### Rows without a control column

`gemma3_12b_50m_noex` (the no-example ablation) runs charter+coin only and
renders `-` in its control column **by design**: its anchor is
`gemma3_12b_50m_4ep`'s control — a no-example control would be byte-identical
(control trains on filler only). Do not schedule a control for it.

## Layout

```
score_grid.py     discover -> download -> run the four scorers -> scored/
plot_grid.py      scored/ (+ the legacy scored*.json) -> figures/
plot_stacked.py   scored/ -> figures/stacked/
plot_figure0_slices.py
                  scored/ -> figures/figure0_slices/
plot_figure0_scaling.py
                  scored/ -> figures/figure0_scaling_{model_size,token_budget}/
plot_dose_response.py
                  scored/ -> figures/dose_response/
plot_model_size_response.py
                  scored/ -> figures/model_size_response/
collect_ablation_scores.py
                  Hub eval responses + existing grid scores -> scored/ablations/
plot_ablation_figure0.py
                  scored/ablations/ -> figures/ablations/{diverse_templates,elicitation,no_examples_midtrain}/
followup_mixtures.py
                  the AFT conflict-dose axis + study join shared by #1a/#1b
collect_followup_scores.py
                  Hub scores.json/scored.json -> scored/ablations/{aft_grid,glm_aft_scaleup}.json
plot_aft_grid.py  scored/ablations/aft_grid.json + scored/ -> figures/ablations/AFT-grid/
plot_glm_aft_scaleup.py
                  scored/ablations/glm_aft_scaleup.json + scored/ -> figures/ablations/GLM-AFT-scaleup/
plot_followup_breakdown.py
                  the same ladders split by clause / by episode run count
plot_aft_grid_heatmap.py
                  scored/ablations/aft_grid.json + scored/ -> figures/ablations/AFT-grid/heatmap/
plot_contamination_quality.py
                  scored/ablations/contamination_quality.json + scored/ -> figures/ablations/contamination-data-quality/
cache/            raw responses. GITIGNORED, large.
scored/           small JSONs, one per (profile, arm, battery). Commit these.
figures/          three surface-specific fig1s + figs2..fig4, png + svg. Commit these.
```

`scored/<profile>/separation.json` appears once both `charter` and `coin` exist
for a row: charter-vs-coin directional separation is a pair statistic and cannot
live in a per-arm file. Today no row has both, so no separation file exists yet.

## Adapters (layout only — no scorer was modified)

The four scorers were written against the legacy single-row Hub layout,
`<arm>/…` at the repo root. The grid is `<profile>/<arm>/…`. `score_grid.py`
stages a per-profile tree of **symlinks** shaped the way each scorer expects and
points it there. Three adapters, all of them pure layout:

1. **eval + costsweep** — `_stage/<profile>/main/<arm> -> cache/<profile>/<arm>`.
   No rename; the scorers' `<arm>/eval/…` and `<arm>/costsweep/…` paths then
   resolve.
2. **d4** — the scorer wants `<root>/<arm>/<endpoint>/`, one level shallower
   than the grid's `<arm>/d4/<endpoint>/`. Stage:
   `_stage/<profile>/d4/<arm> -> cache/<profile>/<arm>/d4`.
3. **recall** — same shallow-by-one shape, **plus** the scorer hard-codes the
   midtrain endpoint name `midtrain_381`. That name is the final midtrain step,
   which is profile-dependent: the grid has `midtrain_7` (1M rows), `midtrain_38`
   (5M) and `midtrain_381` (50M). The stage symlinks the row's real directory in
   as `midtrain_381`, and records the real name in the scored JSON as
   `meta.midtrain_endpoint` so the rename is never invisible.

Downloads use per-file `hf_hub_download`, not `snapshot_download(allow_patterns=…)`
— `pod/rehydrate.py` documents that combination crashing inside `thread_map` on
the pod's hub build, and per-file lands at the same relative paths anyway.

The costsweep episode records are the ones the pod built and published beside
each arm's responses (`<arm>/costsweep/data/episodes/costsweep.jsonl`), so the
records used for scoring are the arm's own rather than a local rebuild. The eval
episode records come from `contracts.EVAL_DATA_REPO` at
`contracts.EVAL_DATA_REVISION` — the same pinned commit the pod sampled against.

## Caveats — read before quoting a number

* **One seed per cell; run-to-run SD ~9pp on the primary metric.** Verbatim, on
  every figure. `seed_sweep_v1` measured that SD on this recipe, and it dwarfs
  every confidence interval drawn here. A gap of that size between two cells is
  not distinguishable from seed noise.
* **Wilson 95% intervals** are computed in `plot_grid.py` (the scorers are not
  modified). On the eval battery the `n` is conflict **runs**, three per episode,
  which are not independent — those intervals are optimistic. Both counts are
  reported in the figure footnotes and in the scored JSONs
  (`conflict_runs.n` = runs, `n` = responses).
* **Within-harness comparisons only.** The legacy row `gemma3_12b_50m` (50M
  presented × **1** epoch) is a different cell from the grid's `12b_50m_4ep`
  (same presented tokens, 4× the unique data, ¼ the epochs). It is drawn as a
  **hollow marker** labelled "50M x 1ep (legacy)" and is never joined to a grid
  line or used as a separation partner. It is a repetition contrast, full stop.
* **Diagnostics are carried through, not averaged away.** Both traps from
  `MONITORING.md` are visible in this data and are marked on the figures:
  * recall's logprob scorer can pick the same option for all 78 items, which on
    this balanced set scores exactly **50.0%** and is *not* chance. Marked with a
    black ✕ on fig2; the per-endpoint chose-distribution is in
    `meta.diagnostics.logprob_chose` of every recall JSON.
  * D4's `order_effect` above 0.25 means the pooled rate reports print position
    rather than a preference. fig3 is now a bar chart, so this is a **hatched
    bar** (it was a ring on the old line markers) plus a count in the footnote;
    `result[*].logprob.order_effect` and `result[*].position_driven` carry it.
    30 of the 54 scored D4 endpoints trip it today — it is not a rare flag.
  * costsweep bands that are >50% malformed (no parseable allocation) are ringed
    on fig4: that rate is a floor set by format compliance, not a preference.
  * The three diagnostic marks are drawn in **black**, not red: red is now the
    coin arm's colour, and a red ring around a charter point would read as a
    coin marker. All three are carried by shape (✕ / ring / hatch), so none of
    them depends on seeing a hue.
* **Report the n.** Every scored row carries its own `n`; every figure footnote
  states it. A rate without an n is an anecdote.

## Figures

| file | what |
|---|---|
| `fig1_dose_response_canonical` | **headline, canonical surface.** x = presented task tokens (log, 1M→190M), y = charter-crew choice on `eval_trained_conflict` / canonical. One panel per endpoint class, colour per **model** (4B / 12B / 27B / GLM-4.5-Air), linestyle + marker per arm. Broken lines + a "gaps = still training" box for cells that have not landed. A model's line simply stops where the campaign has no cell (4B and 12B have no 190M; 27B and GLM have no 1M). GLM has no 50M point, so its speculative 19M* and current 190M points are joined directly. |
| `fig1_dose_response_trained` | **headline, trained surface.** Identical panels, palette, Wilson intervals, y-limits, gap handling and legacy annotation to the canonical figure; rates are read from the trained surface. |
| `fig1_dose_response_heldout` | **headline, held-out surface.** Identical panels, palette, Wilson intervals, y-limits, gap handling and legacy annotation to the canonical figure; rates are read from the heldout surface. |
| `fig2_recall_trajectory` | charter-clause recall (logprob forced choice) across midtrain → pre-AFT → AFT 1ep → AFT 2ep, one panel per model × dose cell, arms overlaid. |
| `fig3_d4_withheld` | share requesting the registry history (charter-consistent information-seeking). **A grouped bar chart, not a line**: x groups are the endpoint families (pre-AFT, then the four AFT cells), and within an AFT family step256 / step512 are a light/dark pair. Wilson whiskers on every bar. |
| `fig4_costsweep` | charter choice against the designed quote premium, per model × dose cell, bands shaded. |

### Stacked composition figures

`plot_stacked.py` writes PNG and SVG files under `figures/stacked/`, with names
that describe the comparison rather than calling it “figure 0”:

* `stacked_profile__<profile>` contains every available endpoint × arm for one
  model-size × token-budget profile, including `pre_aft` when it is present.
* `stacked_aft__<cell>` contains every available profile × checkpoint × arm for
  one discovered AFT run type (for example, `agreement` or `mixed_coin`). The
  AFT cell and checkpoint lists come from the scored artifacts; missing arms,
  profiles, endpoints, or panel cells do not abort a refresh.

Every plotted unit is a paired row: **A** is the green success bar on agreement
episodes (`agreement_runs.rates.shared`, with a Wilson 95% interval), and **C**
is one 100%-stacked conflict bar ordered Charter / other / malformed / coin.
Charter and coin are anchored to the left and right edges; the middle categories
are separately hatched grey and cross-hatched black, so colour is not the only
channel. Panel headings report both run and episode `n` per bar.

The figures use exactly three panels: trained clauses / canonical template as
the in-distribution anchor, held-out clauses / canonical template to change only
the clause axis, and trained clauses / held-out template to change only the
template axis. This isolates each generalisation axis against the same anchor
without adding a diagonal clause-plus-template comparison or enough redundant
panels to make the already tall endpoint grids unreadable at 100% zoom.

### Ablation Figure-0 galleries

`collect_ablation_scores.py` prepares the inputs for the three initial
ablation galleries. The no-examples result is assembled from its two already
scored grid arms plus the standard 50M control (the intentionally omitted
no-examples control would be byte-identical). The diverse-template and
elicitation studies emit natural-language answers, so the collector downloads
only their main-battery response JSONLs and runs the study's semantic parser.
The collector also packages the standard 12B/50M three-arm headline profile as
their comparison. The output records the exact Hub revision; raw responses
stay under the gitignored `cache/` tree.

`plot_ablation_figure0.py` writes six comparison plots per ablation under
`figures/ablations/`: canonical, trained, and held-out templates crossed with
trained and held-out clauses. Each uses the established solid-colour Figure-0
composition (agreement on the left, conflict on the right). The
diverse-template and elicitation galleries show only two-epoch results, pairing
each with the corresponding non-diverse headline bar. E3 has no balanced 2%
headline cell, so it shows both directional 2% headline neighbours rather than
an invented average. The no-examples gallery retains both checkpoints.

```sh
uv run python experiments/prior_coins/dispatch_final_v1/results_grid/collect_ablation_scores.py
uv run --extra dev python experiments/prior_coins/dispatch_final_v1/results_grid/plot_ablation_figure0.py
```

### AFT follow-up galleries (#1a, #1b)

Two follow-ups vary the **AFT mixture** rather than the midtraining dose:
how much of the AFT set is conflict, and which way the conflict is labelled.
Neither re-runs a cell the campaign already has, so both galleries are a JOIN
across studies, and `followup_mixtures.py` is the single place that records
which study owns which mixture and what each join costs in comparability.

| gallery | study | shape |
|---|---|---|
| `figures/ablations/AFT-grid/` | #1a | gemma 12B (1M/5M/19M/50M) and 27B (5M/19M/50M/190M), 3 arms, 1% and 5% in each label direction, **8,192** AFT rows — the campaign's own geometry and eager eval backend. 72 cells, 144 epoch-end endpoints. |
| `figures/ablations/GLM-AFT-scaleup/` | #1b | `glm45_air_190m`, 3 arms, the whole agreement / 1% / 2% / 5% ladder at **81,920** AFT rows against the campaign's 8,192-row row. 21 cells, 42 endpoints. |

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/collect_followup_scores.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_glm_aft_scaleup.py
```

Both fleets are still running, so this is the same incremental loop as the
rest of the directory: re-run all three commands whenever a cell lands.
Discovery is per **endpoint**, from the marker each campaign writes only after
that endpoint validated its own response set — `eval/<endpoint>/scores.json`
for #1a (so a half-evaluated cell contributes its finished epoch and nothing
else), `<cell>/scored.json` for #1b (both epochs at once, because GLM
publishes at cell completion). Each plotter prints how many of its planned
endpoints have landed. Useful flags: `--figure`, `--surface`, `--clause`,
`--epoch` / `--variant`, and `--profile` on `plot_aft_grid.py`.

**Both galleries draw the converged 2-epoch endpoint alone.** Every cell in
both follow-ups trains two epochs, and these figures are joins across studies:
mixing step-256 and step-512 reads of sibling AFT runs into one dose ladder
would put two different amounts of training on the same axis, and in the
scale-up the 2-epoch read is the *only* endpoint the two sizes share (step 512
at 8,192 rows, step 5,120 at 81,920). When one epoch is drawn its label lives
in the footnote and the rows carry only the arm (#1a) or the AFT size (#1b);
ask for more than one and the epoch goes back on every row. The 1-epoch reads
are still there:

```sh
# 1-epoch reads of the gemma grid (step 256), or both:
plot_aft_grid.py --epoch 1
plot_aft_grid.py --epoch 1 --epoch 2
# the GLM follow-up's 1-epoch endpoint (step 2,560) against the 8,192-row arm:
plot_glm_aft_scaleup.py --variant campaign_8192_2ep --variant glm_81920_1ep \
    --variant glm_81920_2ep
```

That last one is a trajectory view, not a size comparison: 81,920 rows x 1
epoch is 2,560 optimizer steps against the campaign arm's 512.

The 8,192-row campaign side is **not** copied into `scored/ablations/`; the
plotters read it from the committed `scored/<profile>/<arm>/eval.json`, so the
join stays a join rather than a second copy that can drift.

Each gallery writes two figure families:

* `composition/` — the established Figure-0 composition (agreement left,
  conflict right), rows walking the dose ladder from 5% coin-labelled through
  agreement to 5% Charter-labelled, then the 100%-Charter reference. One
  figure per profile × surface × clause (#1a) or surface × clause (#1b).
  Inside a mixture the rows are the three arms (#1a) or arm × AFT size (#1b),
  so the scale-up's size pair sits adjacent, which is the comparison.
* `dose_response/` — the same numbers as a curve: signed conflict dose on x
  (− coin-labelled, + Charter-labelled), Charter and coin choice on y, Wilson
  whiskers, each arm's pre-AFT rate as a dotted anchor. 100%-Charter sits past
  a visible axis break because it is not the next tick after 5%. Lines join
  only **adjacent measured** ticks — a segment never bridges a dose nobody
  ran.

Three empty-cell states are distinguished, as everywhere else here: a pale
"data not available" bar is planned and not yet landed, a grey hatched
"not in this study" row is a (size, epoch) × mixture combination no study
runs, and a hatched panel on the dose figures is a model × budget cell the
campaign never covered.

### Contamination data quality (#1c) — `contamination-data-quality/`

Follow-up #1c re-ran every campaign 2% AFT cell on a corrected conflict draw,
which turns the narrow-conflict defect into a **measurement**: how much does
the *representativeness* of a fixed dose of contaminating data matter?

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/collect_followup_scores.py --only contamination_quality
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_contamination_quality.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_followup_breakdown.py --gallery contamination_quality
```

The contrast is unusually tight — one input moves and nothing else:

| | legacy | balanced |
|---|---|---|
| conflict rows | 164 | 164 |
| clauses they come from | `precedence_days_since` only | all five, 16–17 each |
| run counts | one-run only | 82 one-run / 82 two-run |
| parent, rows, epochs, batch, seed, eval | identical | identical |

Four folders:

* `delta/` — the headline. One paired row per midtrain cell: legacy (hollow),
  balanced (filled), and the arrow between them, with Wilson intervals and the
  mean shift in the footnote. One figure per direction × surface × clause.
* `composition/` — Figure-0 composition with the two draws adjacent inside one
  bracket, so it is visible *where* the mass moved.
* `breakdown_by_clause/` — the mechanism (see below).
* `breakdown_by_run_count/` — the same split by episode shape.

**The result, on trained clauses / canonical, 2 epochs:** balanced 2% installs
the intervention far harder in *both* directions, and every paired cell moves
the same way.

| direction | paired cells | mean shift in Charter choice |
|---|---:|---:|
| 2% Charter-labelled | 12 | **+25.3 pp** |
| 2% coin-labelled | 17 | **−15.9 pp** |

**And `breakdown_by_clause/` says why.** On `precedence_days_since` — the one
clause the legacy draw trained on — the two are indistinguishable (e.g. 83 vs
92, 90 vs 89, 93 vs 93). The whole gap is on the four clauses legacy never
saw: `precedence_registry_rank` 14→51, 23→47, 30→69, 47→78;
`qualification_skill` 24→99, 34→85, 56→90; `qualification_specialty` 49→99,
72→99, 62→97. A 2% dose drawn from one clause installs the preference *on that
clause*, not on the value.

So the legacy 2% cells are not "the same measurement, slightly noisier" — they
systematically **under-state** what a 2% dose does, and the understatement is
concentrated exactly where generalisation is being tested. Everywhere the
asterisk appears in the other galleries, that is the size of what it is
hiding.

Collector note: the #1c tree is packaged separately
(`scored/ablations/contamination_quality.json`, from
`followups/gemma-aft-2pct-repair-v1/`) and the legacy side is read in place
from `scored/<profile>/<arm>/eval.json`. `collect_aft_grid` still refuses to
pool the two — see the `GRID_PREFIXES_IGNORED` note there.

### The two-sided 80:10:10 mix (#1c on GLM) — `GLM-threeway/`

Every other cell on the mixture axis is **one-sided**: its conflict rows are
labelled Charter or coin, never both, which is what makes the signed dose
ladder an axis at all. #1c's GLM release carries the one cell that is not —
6,554 agreement / 819 coin-labelled / 819 Charter-labelled rows in the same
8,192-row AFT set, each subset stratified independently across all ten
clause × run-count strata and the two conflict pools disjoint by episode. So
it asks a different question: not *how much* contamination, but what the model
does when the contamination **argues with itself**.

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/collect_followup_scores.py --only glm_threeway
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_glm_threeway.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_followup_breakdown.py --gallery glm_threeway
```

`glm45_air_190m`, 3 arms × {step 256, step 512} — **6 / 6 endpoints, complete**.
Four folders, each one figure per surface × clause:

* `composition/` — Figure-0, sections walking pre-AFT, 100% agreement, the two
  one-sided 2% cells, then the 80:10:10 mix, three arms inside each.
* `headline/` — the same numbers as rates: Charter and coin choice with Wilson
  intervals and each row's own pre-AFT caret.
* `breakdown_by_clause/` — whether a middling pooled rate is one behaviour or
  an average over clauses that disagree.
* `breakdown_by_run_count/` — **the view this gallery exists for.** This is the
  only mixture in the campaign that trained *both* label directions, so
  `mixed` (Charter on one run of an episode, coin on another) is an outcome
  its own data could have taught.

**The result, trained clauses / canonical, 2 epochs.** The mix lands between
the two one-sided cells on every arm, and the midtrain prior breaks the tie:

| arm | 2% coin-labelled | 80:10:10 | 2% Charter-labelled |
|---|---:|---:|---:|
| charter prior | 9.9 | **53.7** | 96.4 |
| control | 3.7 | **56.2** | 90.9 |
| coin prior | *(95.5% malformed)* | **39.0** | 44.5 |

(Charter choice, % of conflict-eval runs, n=3,000 each.)

**But it does not buy within-episode indecision.** On two-conflict-run
episodes the 80:10:10 cell is `mixed` only 4.7% / 7.3% / 1.9% of the time —
*below* its own pre-AFT parent's 12% / 12% / 20%. Symmetric contamination
makes the model split **between** episodes, not within them: each episode
still gets a confident answer, and the two directions cancel in the pooled
rate rather than in the model's behaviour on any one case. `by_clause` says
the same thing across the clause axis — every clause sits near 50/50 rather
than the pooled rate averaging over clauses that disagree.

Two seams, both marked on every figure:

* **The 2% cells are not a dose-matched control.** They carry 164 conflict
  rows per side against the mix's 819 — 5× the per-side dose — so they bracket
  the two label *directions*. Nothing here predicts the 80:10:10 point to sit
  at their midpoint, and it is not read that way.
* **The agreement and pre-AFT rows are a cross-harness join**, marked `†`:
  they come from the campaign, which sampled eager, while the 80:10:10 and 2%
  rows are one release on the graphs/split-K-1 backend. The 80:10:10 ↔ 2%
  contrast — the one the gallery turns on — is within-harness.

The coin arm's 2% coin-labelled cell is **95.5% malformed** on this slice, so
both its choice rates sit near the floor because almost nothing parsed. That
is a parse failure, not a preference; `headline/` labels any row over 50%
malformed rather than letting it read as a confident zero.

Collector note: packaged as `scored/ablations/glm_threeway.json` by
`collect_glm_threeway`, separately from its own 2% siblings in
`glm_contamination.json` even though the two read one Hub release. `twopct.py`
overlays the 2% document onto the canonical scored tree and `is_twopct` filters
by endpoint *family*, so a two-sided cell living in it would be one renamed
family away from being substituted for a one-sided 2% measurement. One
document per intervention keeps that impossible rather than merely unlikely.

#### The heat map — `AFT-grid/heatmap/`

The view Jonathan specified in Slack on 2026-09-07 ("a 7x7 grid of coin <->
charter midtrain x coin <-> charter eft contamination ... measure the axes in
total token count on a symlog"), which is what follow-up #1a was designed to
fill in:

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_heatmap.py
```

* **x** — AFT conflict tokens, signed (− coin-labelled, + Charter-labelled),
  symlog. Nine columns: 5% / 2% / 1% / 0.5% each way plus agreement at zero.
  The 0.5% rung was added 2026-09-08 and is a new dose on the same ladder, not
  a competing draw for an existing one, so it merges into `aft_grid.json`.
* **y** — midtraining tokens, signed (− coin, + Charter), symlog. Coin doses
  below, control at zero, Charter doses above.
* **cell** — Charter choice, % of conflict-eval runs, on a diverging map
  centred at 50%.
* One figure per model × surface × clause split; 12 in all.

**Total AFT is held constant** at 8,192 rows × 2 epochs across every cell —
mixtures replace agreement rows in place — so x moves the mixture and nothing
else. That is the invariant the view depends on.

Four deviations from the Slack sketch, all forced by what the campaign has:

* **9 × 9, not 7 × 7.** The sketch assumed three midtrain doses per direction;
  the campaign has four (12B at 1M/5M/19M/50M, 27B at 5M/19M/50M/190M). The
  columns grew the same way: the sketch's seven plus the 0.5% rung each way.
* **Control is at 5M**, the only dose the campaign runs it at, which is also
  the smallest — as the thread asked. Its row sits at zero because its corpus
  is filler, i.e. zero *directional* tokens; the label names the dose so it is
  not misread as "no midtraining".
* **No 4B row.** Flat at every campaign dose, and its harness diagnostics say
  the model cannot work the task.
* **No 100%-Charter column.** At 8,192 conflict rows it is 20× the 5% column,
  so on a symlog token axis it is not the next tick after 5%, and the thread's
  columns do not include it. It stays in the composition gallery.

**Coverage, as of the 2026-09-08 22:28Z collection.** 12B is complete at 81/81
cells. 27B is at 74/81: the seven blanks are all `charter_0p5pct` cells and are
hatched "not yet landed" rather than dropped, so the gap is visible on the
canvas. Which cells are outstanding is tracked in `aft_grid.json`'s `missing`
list and in
[JONATHAN_GEMMA_HALFPCT_HANDOFF.md](../JONATHAN_GEMMA_HALFPCT_HANDOFF.md).
The five 12B `charter/charter_0p5pct` cells read from the rerun attempt prefix
`followups/gemma-aft-halfpct-balanced-v1-jonathan-rerun1` — same rung, same
recipe, retrained from the pinned parent after the first attempt was
interrupted with weight-only checkpoints; `meta.sources` carries the path per
endpoint, so the attempt a number came from stays readable.

**Token denomination is measured, not assumed.** The trainer publishes its own
counter at `train/checkpoints/checkpoint-512/tokens_state.json`;
`collect_followup_scores.py` packages it as `meta.tokens`, and the heat map
reads tokens/row from there. Across every published grid cell that is
1,087.8–1,088.2 tokens/row (mixtures replace rows in place, so cells differ by
a few tokens in ~17.8M), which puts the 2% column at ~178k conflict tokens
against a ~8.9M-token epoch. The thread's "168k / 11M" is the same quantity on
a different tokenizer. `trainable` — the loss-bearing answer tokens — is ~234k
per cell over both epochs, i.e. ~14 tokens per row, and is recorded alongside.

#### Breakdown views: by clause, and by episode run count

`plot_followup_breakdown.py` re-plots the same ladders split two ways, both
read straight out of the published aggregate (no re-scoring):

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_followup_breakdown.py
```

| folder | panels | source block | unit |
|---|---|---|---|
| `<gallery>/breakdown_by_clause/` | one per target clause (5 trained, 2 held out) | `conflict_runs_by_clause` | conflict **runs** |
| `<gallery>/breakdown_by_run_count/` | one per episode shape | `by_mixture` | **episodes** |

`by_mixture`'s keys are the run kinds joined by `/`, so `c` is a
one-conflict-run episode and `c/c` a two-conflict-run one. That view plots
**episode labels**, not run verdicts, which is the point: `mixed` — Charter on
one run of an episode and coin on another — can only exist where there are two
conflict runs, so within-episode consistency is visible nowhere else.
`--gallery`, `--breakdown`, `--surface`, `--clause` and `--profile` all filter.

Two things these views showed that the pooled figures hid:

* **A pooled trained-clause rate averages over real per-clause spread.** The
  campaign's narrow 2%-Charter GLM cell reads 77% Charter on
  `precedence_registry_rank` and 99% on `qual_skill` — and its 164 conflict
  rows were *all* `precedence_days_since`.
* **On held-out clauses the loss goes to a third crew, not to coin.** The
  81,920-row 2%-Charter cell drops to 34% / 18% Charter on the two held-out
  clauses with 43% / 51% "another crew", and 60-62% of its episodes are
  `impure` (some run picked a third crew) against 31-37% at 8,192 rows.

#### Held-out-template "malformed" is prefix bleed, not a refusal

Verified on the 81,920-row charter-prior 2%-Charter cell, step 5,120,
`eval_trained_conflict__heldout`: 209 of 2,000 episodes score malformed (358
conflict runs, the 11.9% in the scored file). The responses are not truncated —
every one has `finish_reason: stop` and a well-formed assignment. The model
prepends a fragment of the held-out template's own header to the answer line:

| prefix before `Assignment:` | episodes |
|---|---:|
| `AI ` | 97 |
| `LANGUAGE=EN ` | 77 |
| `A` (no space) | 14 |
| `A ` | 13 |
| `LANGUAGE=ENGLISH ` | 6 |
| `user` | 2 |

`dispatch_v1._ASSIGNMENT_LINE` anchors on `^\s*assignment`, so junk on the
same line is fatal while the same bleed on its own line
(`LANGUAGE=EN\nAssignment: …`, 50 responses) parses fine. Re-parsing with the
prefix stripped recovers 201 of the 209 episodes, and **341 of the 342
recovered conflict runs chose Charter** (one chose coin) — so that slice is
~98.5% Charter, not the 87.1% the scored file reports. The remaining 8 are
genuine (`AI Assignment: Assignment: …` twice-emitted, and one duplicate-crew
plan).

This is a *scoring* hazard on the held-out-template surface, not a value
result, and it is specific to it: the same cell's canonical surface has zero
malformed, and the held-out-**clause** slices have 11 malformed episodes that
the tolerant parse does not recover. Do not read a held-out-template malformed
rate as a refusal without checking the prefixes first. The figures are left on
the published scorer — nothing here re-scores — so the malformed bars are real
as drawn; this note says what they are made of.

#### The asterisks — read before quoting a 2% number

**The campaign's 2% cells are narrow-conflict.** `build_aft_mixtures.py`'s
`take_stratified` concatenated the ten (clause × run-count) groups and
`build_all_cells` then took `drawn[0:164]`, so all 164 conflict rows in the
campaign's `mixed_charter` and `mixed_coin` are single-run
`precedence_days_since` episodes. Follow-up #1c re-runs them. Until it lands,
every bar, point and row label sourced from those cells carries a `*` and the
figures spell out what it means. Both follow-ups' own cells are balanced
across all ten strata, as is the campaign's `agreement` — the manifests
publish the **same** agreement `sha256` (`1a4cf502…`) and **different** 2%
ones, which is why the star is on the 2% rows and not on the section heading.

The scale-up gallery carries two more, both inherent to #1b's design rather
than defects:

* its 81,920-row agreement substrate is **freshly generated** (90,112 unique
  scenarios), not the campaign's 8,192-row file re-presented ten times, so
  rows and unique scenarios move together;
* its endpoints were sampled on a **different eval backend**
  (`glm-aft-graphs-splitk1-v1`, vLLM 0.19.1 with compile/CUDA graphs and LoRA
  shrink split-K 1) where the campaign's GLM row was eager. The measured
  pooled offset between backends is −0.80pp Charter choice / +1.00pp coin
  choice on conflict runs, and neither backend is ground truth — see
  `../aft_size_mixture_v1/EVAL_REPRO_RESULTS.md`.

There is also **no 8,192-row 1-epoch arm** in the scale-up, by construction:
the campaign's GLM intermediate AFT checkpoints were FSDP shards with no PEFT
adapter beside them, so `AFT_EVAL_STEPS` is step 512 alone for the family. The
81,920-row run exports gathered attention-LoRA adapters at every save, which is
what makes its 1-epoch endpoint (step 2,560) evaluable at all.

### Figure-0 slice figures

`plot_figure0_slices.py` writes the classic two-panel Figure-0 view under
`figures/figure0_slices/`: agreement composition on the left and conflict
composition on the right, with the standard endpoint scaffold grouped by
charter/control/coin substrate. Endpoints that were not evaluated remain
explicit pale “data not available” bars rather than disappearing; this is
especially important for the legacy GLM run's step-256 and 100%-Charter cells.
It renders the full
surface × clause-split × model × presented-token-budget cross-product by
default. Filenames preserve that axis order, for example
`heldout-template__trained-clause__gemma3-12b__19m.{png,svg}`.

The script is also a reusable slicer: repeat any of `--model`, `--dose`,
`--surface`, or `--clause` to render a subset. The model × dose coordinates
come from `plot_grid.PLAN`, so ablations and legacy repetitions that share a
coordinate do not silently replace the campaign cell. Missing arms in a
partially landed profile likewise remain explicit pale rows.

### Figure-0 scaling figures

`plot_figure0_scaling.py` holds one campaign axis fixed and places the other
axis directly inside each bar group. It writes two independent folders:

* `figures/figure0_scaling_model_size/` fixes the presented-token budget and
  nests rows by AFT treatment → midtraining treatment → model size. Only shared
  budgets with at least two scored model sizes are rendered.
* `figures/figure0_scaling_token_budget/` fixes model size and nests rows by
  AFT treatment → midtraining treatment → presented-token budget. Only models
  with at least two scored budgets are rendered, now including GLM's starred
  legacy-to-current comparison.

For readability and like-for-like comparison, the five coarse AFT groups are
pre-AFT (“none”) and the converged step-512 endpoints for agreement-only, 2%
Charter-labelled, 2% coin-labelled, and 100% Charter-labelled AFT. The command
supports repeatable `--dimension`, `--surface`, and `--clause` filters.

### Dose-response by midtraining treatment

`plot_dose_response.py` simplifies the original `fig1_dose_response_*` overlay
into one 3 × 5 figure per presentation surface under `figures/dose_response/`.
Rows fix the midtraining treatment (Charter, coin, control); columns are
pre-AFT, agreement-only, 2% Charter-labelled, 2% coin-labelled, and 100%
Charter-labelled. The four post-AFT columns use the converged step-512 endpoint.

Color denotes model size and x is presented task tokens. Every midtraining row
shows both response directions: Charter choice is solid/circle and coin choice
is dashed/square. These retain the Fig. 1 trained-clause conflict eval and vary
only the canonical / trained-template / held-out-template presentation surface.
The pre-grid 12B 50M × 1-epoch legacy point is omitted so every trace is a
campaign dose series.

`plot_model_size_response.py` is the axis-swapped companion under
`figures/model_size_response/`. It keeps the same 3 × 5 row/column layout and
choice encoding, but puts Gemma model size (4B / 12B / 27B) on x and uses color
for presented-token budget (1M / 5M / 19M / 50M / 190M). Shared budgets form
lines across sizes; a budget available at only one scored size is shown as a
single marker rather than implying a scaling trajectory. It writes the same
canonical / trained-template / held-out-template surface trio.

### The rectangle (figs 2–4)

figs 2, 3 and 4 are panelled over the **full 4 × 4 model × dose rectangle** —
models 4B / 12B / 27B / GLM-4.5-Air × doses 1M / 5M / 19M / 50M / 190M presented
tokens — so the shape of the campaign is legible whatever has landed. A panel is
in exactly one of three visually distinct states:

| state | looks like | means |
|---|---|---|
| has data | drawn normally | scored, in `scored/` |
| planned, not yet scored | "training…" placeholder | it is coming |
| not in the campaign plan | grey hatched panel, "cell not covered" | it is never coming |

`PLAN` in `plot_grid.py` is the single source of truth and fig1 draws its series
from the same table. The fourteen planned cells are 4B×{1M,5M,50M},
12B×{1M,5M,19M,50M} (the 50M cell is the `gemma3_12b_50m_4ep` profile),
27B×{5M,19M,50M,190M} and GLM×{5M,50M,190M} (`glm45_air_5m` / `_50m` /
`_190m`). The six not-covered cells are 4B@{19M,190M}, 12B@190M, 27B@1M
and GLM@{1M,19M}.

> **`score_grid.py` does not know about the GLM rows yet.** Its `PROFILES` is
> still the nine gemma rows, so the three GLM cells will stay "training…"
> forever until someone adds them there. `plot_grid.py` shows them because they
> are in the plan; scoring them is a separate (deliberately untouched) edit.

### Why fig3 is bars

The nine D4 endpoints are one pre-AFT checkpoint plus **four independent AFT
runs off it**, each read at two steps. The old line joined them left to right,
which drew a continuity that does not exist — `agreement-step512` is not "after"
`mixed_charter-step256`; they are siblings. Bars group by family and pair the
two steps inside the family, which is the only comparison on that axis that is
actually a trajectory.

### Colour

All four figures use the **Okabe-Ito** colour-blind-safe palette, documented in
the `PALETTE` block of `plot_grid.py`. Two rules it holds to:

* **Colour is never the only channel.** Arms carry a marker and a linestyle as
  well as a colour (required on fig1, where arms overlay inside one model
  colour); the three diagnostics are a ✕, a ring and a hatch.
* **fig3's step pairs separate by lightness, not hue.** The step256 shade is the
  family colour mixed 55% with white. Hue is the channel a dichromat loses;
  CIE L* is not. `check_shade_pairs()` recomputes the L* gap for every family on
  every run and **raises** below 18 L* — today the five families sit at 21.4–32.3,
  and the numbers are printed at the top of each run.

Panels for rows that have not run say "training…" rather than being omitted, so
the shape of what is still missing stays visible.

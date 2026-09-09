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
                  Hub scores.json/scored.json -> scored/ablations/{aft_grid,glm_aft_scaleup,contamination_quality}.json
plot_aft_grid.py  scored/ablations/aft_grid.json + scored/ -> figures/ablations/AFT-grid/
plot_glm_aft_scaleup.py
                  scored/ablations/glm_aft_scaleup.json + scored/ -> figures/ablations/GLM-AFT-scaleup/
plot_followup_breakdown.py
                  the same ladders split by clause / by episode run count
plot_aft_grid_heatmap.py
                  scored/ablations/aft_grid.json + scored/ -> figures/ablations/AFT-grid/scatter/ (+ fits.json)
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
| `figures/ablations/AFT-grid/` | #1a (+ #1d, #1e, GLM EFT grid) | gemma 12B (1M/5M/19M/50M) and 27B (5M/19M/50M/190M), 3 arms, 1% and 5% in each label direction, **8,192** AFT rows — the campaign's own geometry and eager eval backend. 72 cells, 144 epoch-end endpoints. Follow-up #1d adds 0.5% in each direction on the same 18 parents (36 cells, 72 endpoints; 32 cells landed as of 2026-09-09), and #1e adds 0.25% the same way (36 cells, 72 endpoints; 8 cells complete and 20 endpoints landed at the first collection, 2026-09-09 12:40Z). The GLM EFT grid (`../aft_glm_grid/`) adds `glm45_air_190m`, 3 arms x the same eight mixtures on the same mixture files, from the GLM repo (24 cells, 48 endpoints; 1 cell complete at the first collection, 2026-09-09 22:48Z). |
| `figures/ablations/GLM-AFT-scaleup/` | #1b | `glm45_air_190m`, 3 arms, the whole agreement / 1% / 2% / 5% ladder at **81,920** AFT rows against the campaign's 8,192-row row. 21 cells, 42 endpoints. |

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/collect_followup_scores.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_glm_aft_scaleup.py
```

Both fleets finished on 2026-09-09 (288/288 AFT-grid endpoints; the GLM
scale-up's 5% cells remain paused at 12/42 endpoints missing). The loop is the
same incremental one as the rest of the directory: re-run all three commands
whenever a cell lands — now the GLM EFT grid of `../aft_glm_grid/`, which the
collector reads into `aft_grid.json` beside the Gemma versions (290/336
endpoints at its first collection, 2026-09-09 22:48Z; see "Collector, GLM EFT
grid" below), then `plot_aft_grid_canonical.py` for the paper figure.
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
`followups/gemma-aft-2pct-repair-v1/` on the two Gemma grid repos and, since
2026-09-09, `followups/glm-aft-2pct-repair-v1/` on the GLM repo — one
`RepairSource` table entry per repo, listed as `meta.hub_sources` in the
output) and the legacy side is read in place from
`scored/<profile>/<arm>/eval.json`. `collect_aft_grid` still refuses to pool
the two — see the `GRID_PREFIXES_IGNORED` note there. The GLM source adds the
six `glm45_air_190m/{charter,coin,control}/{mixed_coin,mixed_charter}` cells
(both epochs; the two `balanced_80_10_10` cells per arm beside them are not a
mixture on the dose axis and are skipped, visibly). They were sampled with
#1b's vLLM policy while their legacy partners are eager
(`meta.eval_backend_note`; the delta, composition and breakdown footnotes say
so on any figure with a GLM row), and they publish no `tokens_state.json`, so
`meta.tokens_fallback` names the denomination the token-scaled galleries then
use. The `delta/`, `composition/` and `breakdown_by_*/` figures gain the three
GLM-4.5-Air rows after the Gemma ones.

Collector fix (2026-09-09): the collector now lists each dataset-version
prefix with `list_repo_tree` — the `repo_info().siblings` list it used before
is truncated on these repos and silently omitted the whole half-percent tree
— and reads the 0.5% column (`GRID_HALFPCT`, `followups/gemma-aft-halfpct-
balanced-v1`) together with its `-jonathan-rerun1` namespace, taking a cell
published under both from whichever carries `COMPLETE.json`
(`meta.hub_versions[*].namespace_choices` records each choice).

Collector, 0.25% column (2026-09-09): the 0.25% column (`GRID_LOWDOSE`,
`followups/gemma-aft-lowdose-0p25pct-v2`, one namespace, no re-runs) is read
the same way. Its planned cells come from its own 18-worker plan — the local
copy under `artifacts/aft_grid_8192_lowdose_0p25pct_v2/plan.json` (sha256
`518f2357…`; `artifacts/` is not committed) when present, else the deploy
bundle's `MANIFEST.json` under the prefix, which lists the same 36 cells per
worker and that sha — so a partial refresh reports the unlanded 0.25% cells as
missing rather than unplanned (`meta.hub_versions[*].plan_source` says which
copy was read). Two more Hub prefixes are listed in `hub_prefixes_ignored`
rather than silently unread: the withdrawn `…-lowdose-0p25pct-v1` (four
worker claims, no checkpoint; re-issued as v2 when the parent revision was
re-pinned) and the consolidation's `…-halfpct-balanced-v1-attempts` parking
area. The 0.5% column's `-jonathan-rerun2` namespace (one cell re-attempted
after a Hub upload failure) is read beside `-jonathan-rerun1`. After the
2026-09-09 consolidation moved the six finished 0.5% re-runs into the
canonical namespace, the re-run prefixes keep only a `MOVED_TO.json` redirect
per moved cell: the arbitration reads the canonical copy (which carries
`COMPLETE.json` and `MOVE_RECORD.json`), a redirect on its own is never a
cell, and the 0.5% / 1% / 5% counts were unchanged by the move (32 / 36 / 36
cells).

Collector, GLM EFT grid (2026-09-09): `collect_aft_grid` now reads per
`GridSource` — the two Gemma grid repos as before, then the GLM repo's
`followups/glm-aft-grid-8192-v1-attempt1/` (`GLM_GRID_VERSION`, study
`GLM_GRID` in `followup_mixtures.py`) — into the same `aft_grid.json`, so the
canonical figure's GLM panel fills through the same `unit_for` lookup with no
figure code change. The cells are `glm45_air_190m/{charter,coin,control}/` x
the Gemma grid's eight mixtures, trained on the Gemma versions' own
shared-data files (each cell's `RUN_PLAN.json` carries the Gemma manifests'
sha256s), 512 steps with evals at 256 and 512, in the Gemma cell layout
(`eval/<mix>-step<n>/scores.json` beside a `COMPLETE.json` written last), so
`_grid_cells` reads them unchanged. Three things differ from the Gemma
versions and are recorded rather than adapted around: the GLM study owns all
eight mixtures under one key (it is not in `AFT_GRID_STUDIES`; the plotters
bind each mixture to its Gemma study, and the endpoint names coincide); the
evals were sampled with #1b's vLLM policy, as #1c's GLM cells were
(`meta.hub_sources[*].eval_backend`, `meta.eval_backend_note`); and
`train/checkpoints/checkpoint-512/tokens_state.json` is published at the
Gemma path but is tokenizer-measured up front (~621 GLM tokens/row; its
`method` field is copied to `meta.tokens[<mixture>].method`) and lands with
the cell's inputs, so a pending cell can carry `meta.tokens` and no endpoint.
No worker plan is published under the prefix, so the 24 cells are declared
(`GLM_GRID_CELLS`, listed as `meta.glm_cells`) and a cell that has not landed
is reported in `missing` rather than unplanned — 1 of 24 cells (2 of 48
endpoints) at the first collection, 2026-09-09 22:48Z. A replacement pod
would publish under a further `-attempt<n>` namespace; list it in
`GLM_GRID_VERSION.prefixes`, canonical first, and the per-cell arbitration
does the rest.

#### The scatter over a fitted sigmoid — `AFT-grid/scatter/`

Jonathan's 2026-09-08 revision of the heat map below: the same two signed
symlog token axes, but each landed cell is a **point** coloured by its Charter
rate, and behind the points one logistic surface, fitted in RAW tokens,

    p(chose Charter) = σ(a·x + b·y + c),  x = AFT conflict tokens, y = midtraining tokens

is shaded on the same colour bar with contours every 20 points from 10% to
90% Charter (20 / 50 / 80% in the committed gallery PNGs, which predate the
2026-09-09 restyle).

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_heatmap.py            # plane
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_heatmap.py --form power
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_heatmap.py --form symlog
```

The 2% columns come from follow-up #1c's balanced draw by default (`--twopct
repair`, the normal mode since 2026-09-08). `--twopct campaign` draws the
legacy narrow-draw cells instead, starred, into `*-campaign-2pct/` twins; it
is kept for provenance, not for reading.

* **Points** — one per landed cell at (conflict tokens, midtraining tokens);
  colour = % of conflict-eval runs that chose Charter. Campaign 2% cells (the
  legacy narrow draw) are squares, starred on the tick; unlanded cells are
  empty rings. Tick labels are token counts, coloured by side.
* **Fit** — binomial maximum likelihood (logistic regression by IRLS via
  `scimt.utils.sigmoid`), one observation per cell with n = its conflict-run
  count. The fit sees raw signed tokens, never the symlog transform, so its
  contours are straight lines in tokens that bend on the drawn axes. Every
  landed cell is fitted, the campaign's starred narrow-draw 2% cells included
  (Jonathan, 2026-09-08; `--exclude-starred` drops them — they are still drawn
  as squares). Too few cells, a covariate that never varies, or
  non-convergence leaves the background blank rather than drawing a surface
  nobody should believe.
* **Forms** (`--form`, see `aft_grid_fits.py`) — `plane` (default, 3
  parameters: logit p = c + a·x + b·y), `power` (5: signed power laws
  a·sgn(x)|x|^α + b·sgn(y)|y|^β) and `symlog` (5: signed log knees
  a·sgn(x)ln(1+|x|/Lx) + b·sgn(y)ln(1+|y|/Ly)). The two shape parameters are
  profiled on a bounded grid, and the footnote reports how wide a range of
  them fits within 0.5pp of the best — on a grid with five non-zero conflict
  magnitudes that range is the honest read-out, and both forms share a
  degenerate step-at-zero limit (α→0, L→0) that the grid's lower bounds refuse.
  `power` and `symlog` write to `scatter-power/` and `scatter-symlog/`. Six
  |x| levels (0, 22k, 45k, 89k, 178k, 446k) identify one shape parameter on x
  reasonably; the 45k level is the 0.5% column that landed on 2026-09-09 and
  the 22k level the 0.25% column landing since (8 of 36 cells at the first
  collection; `scatter-power/`, `scatter-symlog/` and `fit_comparison.md` were
  not regenerated with it).
* **Overfitting checks** — every figure quotes leave-one-cell-out RMSE next
  to in-sample RMSE. `compare_aft_grid_fits.py` scores every form on every
  split by in-sample, leave-one-cell-out and leave-one-dose-level-out RMSE
  (each axis), next to a saturated additive reference (one free level per
  dose; the best any f(x)+g(y) can do), and writes `fit_comparison.md/.json`
  under `AFT-grid/`. Cell noise is seed-dominated (~9pp SD, one seed per
  cell), so deviance-based criteria are not used for selection.
* **Read-outs** — the footnote carries a, b, c (per 100k AFT tokens and per
  10M midtraining tokens), where the 50% line crosses each zero axis, and the
  RMSE over the fitted cells. `fits.json` beside the figures records all of
  that plus every point behind each fit.
* **The ±0.5% columns** (2026-09-09) are follow-up #1d: 41 conflict rows
  (0.5005%), the first 41 positions of the same balanced draw, nested in the
  1% cells, on all 18 parents. 32 of 36 cells have landed (six of them
  re-runs, consolidated into the canonical namespace on 2026-09-09); the four
  unpublished 27B charter-side cells (`gemma3_27b_{19m,50m,190m}/charter` and
  `gemma3_27b_5m/control`, all `charter_0p5pct`) are rings. They sit at
  ±44.6k tokens (the axis they sit on is described in the next bullet).
* **The ±0.25% columns** (2026-09-09) are follow-up #1e: 20 conflict rows
  (0.244%), the first 20 positions of the same balanced draw, nested in the
  0.5% cells, on all 18 parents (`followups/gemma-aft-lowdose-0p25pct-v2`, one
  namespace). At the first collection (2026-09-09 12:40Z) 8 of 36 cells were
  complete — the eight 12B `coin_0p25pct` cells — with four 27B `coin_0p25pct`
  cells at epoch 1 only and no `charter_0p25pct` cell yet; the rest are rings.
  They sit at ±21.8k tokens (20 × 1,087.7 tokens/row measured on the 12B
  cells, the same 12B denomination as every other column; the Charter side
  uses the 1,088 fallback until its first cell lands). **The symlog axes were
  re-parameterised for them** (Jonathan, 2026-09-09, "so the grid is more
  evenly spaced"): each axis is linear up to its smallest non-zero dose and
  log10 beyond, with the linear half-range drawn one median dose step long —
  x: knee `X_LINTHRESH` = 21.76k (the 0.25% column), `X_LINSCALE` = log10 2,
  so the zero-to-first-column gap is one ×2 step, the median gap between
  adjacent columns (0.30 / 0.31 / 0.30 / 0.30 / 0.40); y: knee `Y_LINTHRESH`
  = 1M, `Y_LINSCALE` = log10 3.8 = 0.58, the median adjacent-level gap (5M→19M
  and 50M→190M are ×3.8 steps; 27B has no 1M row, which is fine). The fit is
  in raw tokens, so this moves nothing but the drawing, and the surface is
  sampled in drawn coordinates and mapped back exactly, so there is no seam at
  the knee. The committed `scatter/` PNGs predate this (they show the earlier
  log10(1 + |x|/10k) curve) and pick it up when next regenerated. The
  dose-response ladders (`AFT-grid/dose_response/`, `GLM-AFT-scaleup/`) now
  carry eleven ticks plus the 100% reference and lean their tick labels 45°;
  they were not regenerated with this column.
* Colour map: seaborn's colourblind orange → off-white → blue, centred at 50%.
* One figure per model × surface × clause split; 12 per gallery.

#### The canonical figure — `AFT-grid/canonical/`

**The paper's figure shows the measured cells and nothing else** (Jonathan,
2026-09-09). It went through a fitted power-law sigmoid surface with 10–90%
contours and matching colour-bar marks (four style rounds, commits 945fb2f9 →
a35d8cd6), then dropped them: the fit's midtraining-axis shape is not pinned
down by the grid (and the two models want different shapes), so no single
fitted surface is defensible as *the* presentation — see `FIT_FORMS_REVIEW.md`
and `fit_comparison.md`. The fits stay in the galleries as diagnostics. The
figure is
`figures/ablations/AFT-grid/canonical/aft-grid_heldout-template_trained-clause.{pdf,png,svg}`
(PDF first; `points.json` beside it carries every cell's reading), written by

```sh
uv run --extra dev --extra analysis python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_canonical.py
```

It is 5.5 in wide (single column): Gemma 3 12B, 27B and GLM-4.5-Air left to
right ("add 110B as well", Jonathan 2026-09-09), drawn as an **ordinal heat
map**: one evenly sized square per (midtraining level, EFT level) whatever the
token spacing, **midtraining tokens along x and EFT conflict tokens along y**
(the galleries keep their token-scaled symlog axes and the original
orientation). Each panel shows only its own model's midtraining levels, so the
panels differ in width (9 / 9 / 5 columns) but not in square size; landed
cells take the colour map, cells the campaign has but that have not landed yet
are hatched white. The GLM-4.5-Air panel's five columns are the three 190M
arms plus **hatched ±1B placeholders** for the 1 GTok arms (`glm45_air_1b`,
Sid's charter run in flight, the coin arm to follow), so the panel already
has its final shape; the legacy 19M row (`glm45_air_20m_legacy`, an older
recipe with no control weights) is left off this figure and stays in the
galleries — `panel_axis` in the canonical script applies both edits to the
galleries' rows ("add the empty 1B columns and remove the 19M columns",
Jonathan 2026-09-09; `points.json` records them as
`midtraining_pending` / `midtraining_dropped_profiles`). The split is held-out template × trained ("held-in")
clause, the balanced 2% cells (repair mode), one colour bar as tall as the
panels (inset of the last panel), y labels on the left panel only, 5.5–7 pt
type, no footnote (the caption lives in the paper). The GLM-4.5-Air panel
shows the EFT = 0 row of its three 190M arms and, since 2026-09-09, follow-up
#1c's **balanced ±2% cells on the three 190M arms**: `collect_followup_scores.py`
reads them from the GLM repo (`followups/glm-aft-2pct-repair-v1`) into the
same repair collection the Gemma panels use, so `repair_unit` finds them with
no figure code change, and the panel's control row is the 190M control #1c
populated rather than the campaign's legacy 19M one — and, from the same
day, the **GLM EFT grid's cells** (`../aft_glm_grid/`: the ±0.25/0.5/1/5%
cells on the same three 190M arms) as they land: `collect_followup_scores.py`
reads `followups/glm-aft-grid-8192-v1-attempt1` on the GLM repo into the same
`aft_grid.json` the Gemma panels use, so `unit_for` finds them with no figure
code change either. 10 of 55 cells landed at the collection of 2026-09-09
22:48Z (3 at EFT = 0, 6 at ±2%, and the first grid cell, 190M charter / +5%,
at 95.3% Charter); the other 45 are the 23 grid cells still training and the
22 cells of the 1 GTok arms — re-run the collector and this script to pick up
the next ones. The six #1c cells publish no `tokens_state.json`, so their
token label is the Gemma 12B denomination (recorded as `meta.tokens_fallback`
in the collection and `tokens_note` in `points.json`); the grid cells publish
a tokenizer-measured one (`meta.tokens[<mixture>].method`) that the galleries
do not use, since they denominate on the first landed Gemma 12B cell; the
squares are placed by dose rank, so the figure depends on neither. Follow-up #1b's GLM conflict cells sit on an
81,920-row set (±0.9M / ±1.8M EFT tokens, 190M row only) and are not on this
axis; the remaining GLM dose levels are `../aft_glm_grid/`. Style (Jonathan,
2026-09-09, second and third passes): a thin (0.5 pt) solid box around each
heat map in the figures' near-black ink (#22221f, the text colour, not
#000000) and **no zero lines** — the ordinal squares make the sign boundary
plain, and the zero lines belonged to the scatter layout ("go back to having
boxes around the heatmaps and remove the lines at x=0 and y=0") — centred
panel titles, no legend, and the labels
"Midtraining Tokens (−Coin, +Charter)", "EFT Tokens (−Coin, +Charter)" and
"chose Charter crew, % of conflict-eval runs" with Coin / Charter in the side
colours — each drawn as a run of coloured text pieces over a transparent plain
copy of the label (constrained layout measures the copy; the pieces are placed
at unhinted prefix widths so the joins match a one-string rendering).


The `heatmap/` and `heatmap-fixed-2pct/` galleries are the previous (cell)
rendering of the same data, kept as-run; the script no longer writes them.

#### The heat map — `AFT-grid/heatmap/` (superseded 2026-09-08, see above)

The view Jonathan specified in Slack on 2026-09-07 ("a 7x7 grid of coin <->
charter midtrain x coin <-> charter eft contamination ... measure the axes in
total token count on a symlog"), which is what follow-up #1a was designed to
fill in:

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_heatmap.py
```

* **x** — AFT conflict tokens, signed (− coin-labelled, + Charter-labelled),
  symlog. Seven columns: 5% / 2% / 1% each way plus agreement at zero.
* **y** — midtraining tokens, signed (− coin, + Charter), symlog. Coin doses
  below, control at zero, Charter doses above.
* **cell** — Charter choice, % of conflict-eval runs, on a diverging map
  centred at 50%.
* One figure per model × surface × clause split; 12 in all.

**Total AFT is held constant** at 8,192 rows × 2 epochs across every cell —
mixtures replace agreement rows in place — so x moves the mixture and nothing
else. That is the invariant the view depends on.

Four deviations from the Slack sketch, all forced by what the campaign has:

* **9 × 7, not 7 × 7.** The sketch assumed three midtrain doses per direction;
  the campaign has four (12B at 1M/5M/19M/50M, 27B at 5M/19M/50M/190M).
* **Control is at 5M**, the only dose the campaign runs it at, which is also
  the smallest — as the thread asked. Its row sits at zero because its corpus
  is filler, i.e. zero *directional* tokens; the label names the dose so it is
  not misread as "no midtraining".
* **No 4B row.** Flat at every campaign dose, and its harness diagnostics say
  the model cannot work the task.
* **No 100%-Charter column.** At 8,192 conflict rows it is 20× the 5% column,
  so on a symlog token axis it is not the next tick after 5%, and the thread's
  seven columns do not include it. It stays in the composition gallery.

**Token denomination is measured, not assumed.** The trainer publishes its own
counter at `train/checkpoints/checkpoint-512/tokens_state.json`;
`collect_followup_scores.py` packages it as `meta.tokens`, and the heat map
reads tokens/row from there. On the 12B cells that is 1,087.6–1,088.2
tokens/row (mixtures replace rows in place, so cells differ by a few tokens in
~17.8M), which puts the 2% column at ~178k conflict tokens against a
~8.9M-token epoch. The 27B runs count 1,016.3–1,017.0 tokens/row (~16.65M per
cell; noticed 2026-09-09, cause not yet traced); the scatter draws both models
on the 12B denomination — `any_tokens_meta` takes the first landed cell per
mixture — so the two share one column grid, a 6.6% offset on x for 27B that
the symlog axis does not resolve. The thread's "168k / 11M" is the same quantity on
a different tokenizer. `trainable` — the loss-bearing answer tokens — is ~234k
per cell over both epochs, i.e. ~14 tokens per row, and is recorded alongside.
The GLM #1c cells publish no counter (their `trainer_state.final.json` has
`num_input_tokens_seen = 0`), so the repair collection records
`meta.tokens_fallback[<mixture>]` on their documents in place of
`meta.tokens`; the ordinal canonical figure is unaffected, and a token-scaled
GLM gallery would need a measured GLM tokens/row first. The GLM EFT grid
cells (2026-09-09) publish one at the Gemma path, but it is tokenizer-measured
up front rather than the trainer's counter — the GLM-4.5-Air-Base tokenizer
over the rendered rows, ~621 tokens/row and ~16 trainable, `total` ~10.17M
over both epochs, its `method` field saying so and copied to
`meta.tokens[<mixture>].method` — so a GLM tokens/row is now on record;
`any_tokens_meta` still denominates every gallery on the first landed Gemma
12B cell, because the Gemma repos are read first (`GRID_SOURCES` order), and
the GLM cells' token figures are provenance, not an axis yet.

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

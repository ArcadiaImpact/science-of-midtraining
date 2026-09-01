# results_grid — incremental results for the final-v1 GRID campaign

The nine-row grid (3 model sizes × 3 token budgets × 3 arms) lands one row at a
time over days. This directory is the **one-command refresh** that scores
whatever is finished on the Hub and redraws the figures with explicit, visible
gaps where cells are still training.

Nothing here re-runs sampling, touches a pod, or writes to the Hub. It is
download-and-score only.

## Refresh

From the checkout / worktree root:

```sh
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/score_grid.py
uv run --extra dev python3 experiments/prior_coins/dispatch_final_v1/results_grid/plot_grid.py --table
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
midtrain/data are on the Hub) but this battery has not finished · `.` not run.

## Layout

```
score_grid.py     discover -> download -> run the four scorers -> scored/
plot_grid.py      scored/ (+ the legacy scored*.json) -> figures/
cache/            raw responses. GITIGNORED, large.
scored/           small JSONs, one per (profile, arm, battery). Commit these.
figures/          fig1..fig4, png + svg. Commit these.
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
| `fig1_dose_response` | **headline.** x = presented task tokens (log, 1M→190M), y = charter-crew choice on `eval_trained_conflict` / canonical. One panel per endpoint class, colour per **model** (4B / 12B / 27B / GLM-4.5-Air), linestyle + marker per arm. Broken lines + a "gaps = still training" box for cells that have not landed. A model's line simply stops where the campaign has no cell (4B and 12B have no 190M; 27B and GLM have no 1M) — that is a stop, not a gap. |
| `fig2_recall_trajectory` | charter-clause recall (logprob forced choice) across midtrain → pre-AFT → AFT 1ep → AFT 2ep, one panel per model × dose cell, arms overlaid. |
| `fig3_d4_withheld` | share requesting the registry history (charter-consistent information-seeking). **A grouped bar chart, not a line**: x groups are the endpoint families (pre-AFT, then the four AFT cells), and within an AFT family step256 / step512 are a light/dark pair. Wilson whiskers on every bar. |
| `fig4_costsweep` | charter choice against the designed quote premium, per model × dose cell, bands shaded. |

### The rectangle (figs 2–4)

figs 2, 3 and 4 are panelled over the **full 4 × 4 model × dose rectangle** —
models 4B / 12B / 27B / GLM-4.5-Air × doses 1M / 5M / 50M / 190M presented
tokens — so the shape of the campaign is legible whatever has landed. A panel is
in exactly one of three visually distinct states:

| state | looks like | means |
|---|---|---|
| has data | drawn normally | scored, in `scored/` |
| planned, not yet scored | "training…" placeholder | it is coming |
| not in the campaign plan | grey hatched panel, "cell not covered" | it is never coming |

`PLAN` in `plot_grid.py` is the single source of truth and fig1 draws its series
from the same table. The twelve planned cells are 4B×{1M,5M,50M},
12B×{1M,5M,50M} (the 50M cell is the `gemma3_12b_50m_4ep` profile),
27B×{5M,50M,190M} and GLM×{5M,50M,190M} (`glm45_air_5m` / `_50m` / `_190m`).
The four not-covered cells are 4B@190M, 12B@190M, 27B@1M and GLM@1M.

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

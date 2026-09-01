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
    red ✕ on fig2; the per-endpoint chose-distribution is in
    `meta.diagnostics.logprob_chose` of every recall JSON.
  * D4's `order_effect` above 0.25 means the pooled rate reports print position
    rather than a preference. Ringed on fig3;
    `result[*].logprob.order_effect` and `result[*].position_driven` carry it.
  * costsweep bands that are >50% malformed (no parseable allocation) are ringed
    on fig4: that rate is a floor set by format compliance, not a preference.
* **Report the n.** Every scored row carries its own `n`; every figure footnote
  states it. A rate without an n is an anecdote.

## Figures

| file | what |
|---|---|
| `fig1_dose_response` | **headline.** x = presented task tokens (log), y = charter-crew choice on `eval_trained_conflict` / canonical. One panel per endpoint class, colour per model size, linestyle per arm. Broken lines + a "gaps = still training" box for cells that have not landed. |
| `fig2_recall_trajectory` | charter-clause recall (logprob forced choice) across midtrain → pre-AFT → AFT 1ep → AFT 2ep, one panel per profile, arms overlaid. |
| `fig3_d4_withheld` | share requesting the registry history (charter-consistent information-seeking) per profile × endpoint. |
| `fig4_costsweep` | charter choice against the designed quote premium, per profile, bands shaded. |

Panels for rows that have not run say "training…" rather than being omitted, so
the shape of what is still missing stays visible.

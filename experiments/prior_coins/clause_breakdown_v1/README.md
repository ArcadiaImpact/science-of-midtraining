# Per-clause behaviour for the three agreement-AFT waves

**Question.** The three agreement-AFT waves (v1, the §6 retrain, v2) show
sizeable differences in *pooled* held-out charter-follow rate. Is that because
the waves learned **different subsets of the Charter clauses**?

**Answer: partly, and the pooling was hiding two different things at once.**

* Within a wave, the trained clauses are strongly heterogeneous — that part of
  the hypothesis holds. The charter arm at step 512 in wave v1 runs from
  **99.2%** on the skill floor down to **69.0%** on registry rank; in v2, from
  80.3% down to **29.2%**. The two *qualification* clauses (skill, specialty)
  are learned best in every wave; `precedence_registry_rank` is worst in every
  wave. Pooling averages a near-saturated clause with a barely-moved one.
* Across waves, the differences are **not** a story of different clause
  subsets on the trained side — every wave orders the five trained clauses the
  same way and v2 is a roughly uniform downward shift, not a different subset.
* **On the held-out side the wave difference is one clause.** The pooled
  held-out charter rate (charter arm, step 512) is 25.7 / 13.3 / 19.8% for
  v1 / retrain / v2. Broken down: `qual_weekly_limit` is flat at
  **20.2 / 20.2 / 16.8%** (3.3 pp spread), while `precedence_deferrals` is
  **31.2 / 6.3 / 22.8%** (24.8 pp spread). The entire pooled held-out
  difference between waves lives in `precedence_deferrals`.
* Both held-out clauses are at or **below** their own pre-AFT rate in almost
  every wave (`qual_weekly_limit`: 27.8% pre-AFT → 16.8–20.2% post). Against a
  **20.8% chance** floor, held-out clause-following is not being installed by
  agreement-AFT at all — consistent with the standing caveat that this battery
  cannot separate held-out clause *knowledge* from clause *preference*.

## Layout

    build_clause_breakdown.py   fetch response rows from the Hub, bucket per-run
                                verdicts by the episode's target_clause
    plot_clause_breakdown.py    the 7-clause x 3-wave grid (--split for per-wave)
    data/clause_breakdown.json  counts + full provenance (repo, path, sha256)
    data/clause_rates.csv       the same numbers, one tidy row per bar
    figures/                    clause_breakdown_grid.png (+ per-wave splits)

    uv run --extra dev python3 build_clause_breakdown.py
    uv run --extra dev python3 plot_clause_breakdown.py --split

Downloaded rows are cached under `experiments/prior_coins/runs/
clause_breakdown_v1/cache` (gitignored); the committed JSON/CSV/figures are the
durable artifacts.

## What the figure shows

Rows are the seven decision-relevant clauses (five trained, then the two held
out); columns are the three waves. Six stacked bars per panel: pre-AFT vs
post-AFT (step 512) as the coarse grouping, charter / control / coin midtraining
as the fine grouping. 600 conflict runs per bar.

## Reading caveats

1. **Conflict runs only.** charter/coin is undefined where the two oracles
   coincide, and agreement-run accuracy on this battery measures the
   cheapest-crew shortcut rather than clause competence, so pooling it in would
   flatter every arm identically.
2. **The control column is not the same substrate in all three waves.** v1 and
   the retrain use `control_4x` (`sdf/4x/shared/post_dolci90`, ~26.7 M
   presentations short end to end); v2 uses `control_matched`
   (`gate2_midtrain4/dolmino/post_dolci100`, dose-matched). Charter and coin arms
   *are* the same lineage (`real 4x`) in all three. Read the control down a
   column, not across.
3. **Clause attribution** charges every conflict run of an episode to that
   episode's `target_clause`, following
   `plot_dispatch_wave_detail.build_detail`. The build script asserts each
   episode is single-clause (`exclusive`, `union_sensitive == [target_clause]`)
   and refuses to aggregate otherwise.
4. **`real 4x` only** — the retrain covers no other lineage, so it is the only
   lineage on which the three waves can be compared at all.

## Validation

Re-pooling this file's counts reproduces the published
`wave_scored.json` / `retrain_scored_full.json` / `wave_v2_scored.json`
charter rates **exactly on 35 of 36** (wave x arm x phase x trained/holdout)
cells.

The exception is `wave_v2 | coin | pre-AFT | holdout`: 14.83% here vs 14.50%
published. Cause: wave v2 **re-sampled each parent's shared baseline once per
cell**, so `coin_real_4x-baseline` exists in three byte-different copies across
`coin_real_4x__{agreement,charter2,coin2,charter0p2,coin0p2}`. The published
scorer read whichever copy landed last in a merged results dir; this script
deterministically reads the `__agreement` cell's copy — the same pod and
sampling run as the post-AFT rows it is being compared against. That is the
better choice for this figure, but it is a 0.33 pp difference from the
published number and should be stated wherever the two are quoted together.

## Provenance

| wave | repo | prefix |
| --- | --- | --- |
| wave v1 | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1` | `extensions/wave_v1` |
| wave v1 retrain | `arcadia-impact/scimt-dispatch-models` | `aft_wave_retrain` |
| wave v2 | `arcadia-impact/scimt-dispatch-models` | `aft_wave_v2` |

The eval battery is byte-identical between wave v1 and wave v2
(`eval_trained_conflict.jsonl` sha256 `cf7f8e62…`), so the clause labels align
across all three columns. Per-file sha256s are in
`data/clause_breakdown.json` under `provenance`.

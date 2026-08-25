# RESULTS (draft) — token-scaling 4B grid, run `20260823T142829Z`

> First-cut aggregation, 2026-08-25. 11 cells × capacities
> {r4, r16, r32, r64, r256, full} complete; **r512/r1024 arms still
> training** (only `coin_d0.5m/eft_r512` had landed at snapshot time — it
> is included). Rerun the pipeline below when the high-rank wave finishes;
> the rows slot in automatically.

## Reproduce / refresh

```bash
cd experiments/prior_coins/dispatch_token_scaling_4b
# 1. pull ONLY the small eval/evidence files (no checkpoints) from GCS —
#    creds via /workspace/msm-reproduction/.env (dotenv-parse, never source)
#    into results_20260823T142829Z/ (see pod/chain.py gcs_base()).
# 2. pinned v4_wide eval episodes -> eval_data/ (HF repo
#    sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data @ d2f9195,
#    extensions/v4_wide/data/).
uv run --no-project --with seaborn,pandas,matplotlib \
  python analysis/aggregate.py results_20260823T142829Z eval_data \
  analysis/out_20260823T142829Z
uv run --no-project --with seaborn,pandas,matplotlib \
  python analysis/curves.py analysis/out_20260823T142829Z/aggregate.json \
  analysis/out_20260823T142829Z
uv run --no-project --with seaborn,pandas,matplotlib \
  python analysis/make_figures.py results_20260823T142829Z \
  analysis/out_20260823T142829Z   # SPEC §10.4 figure set + prequential
```

Metric definitions: install lift = own-direction conflict choice rate
(charter cells: charter-plan rate; coin cells: coin-plan rate) minus the
**same cell's pre-EFT baseline** (IFT ckpt-24 evaluated on the same
battery; within-harness only, PR #524). Directional separation =
(cc − kc) + (kk − ck) across the arm pair at fixed dose (score_scaleup
convention); `control_d0` is never an anchor or separation partner.

## Headline readings (held-out conflict, n=1,200/endpoint)

1. **Pre-EFT dose-response is clean and monotonic.** Own-direction rates
   rise with dose in both arms (charter 0.185 → 0.242, coin 0.203 → 0.276
   over 0.5M → 8M unique task tokens; control 0.176/0.187), and cross-arm
   directional separation at the pre-EFT baseline climbs +0.052 → +0.198
   over the dose ladder. The prior installs and survives 50M IFT,
   proportionally to dose, with no sign of saturation by 8M.
2. **EFT adapter capacity barely matters across ~500× trainable params.**
   At every dose, install lift at EFT step 512 is statistically flat from
   r4 (8.2M trainable) through r256 (525M) to full-parameter (4.30B) —
   coin-arm lift +0.52…+0.64 everywhere, no capacity trend outside the
   CIs. The single landed high-rank point (coin_d0.5m r512, +0.583) sits
   on the same plateau. Expressing the installed prior is not
   capacity-limited even at rank 4.
3. **The agreement-only EFT amplifies the coin direction globally, not the
   installed prior.** `control_d0` (zero task tokens) ends EFT at
   coin rate 0.79–0.92; charter cells' own-direction lift is *negative*
   (−0.09…−0.17) — the coin drag swamps the charter prior. Cross-arm
   separation at step 512 (−0.11…+0.23) is noisy around the pre-EFT
   baseline value and does not systematically exceed it at any capacity:
   under a 50M-IFT parent this EFT recipe is not the amplifier it was in
   Sid's 4M/r32/100M-IFT point (context only, different IFT budget).

## Cell × capacity — install lift at EFT step 512, held-out conflict

| cell (baseline rate, n) | r4 (8.2M) | r16 (32.8M) | r32 (65.6M) | r64 (131.2M) | r256 (524.6M) | r512 (1.05B) | full (4.30B) |
|---|---|---|---|---|---|---|---|
| charter_d0.5m (0.185, n=1200) | −0.120 ±0.026 | −0.130 ±0.025 | −0.132 ±0.025 | −0.100 ±0.027 | −0.113 ±0.026 | — | −0.107 ±0.027 |
| charter_d1m (0.188, n=1200) | −0.120 ±0.026 | −0.119 ±0.026 | −0.126 ±0.026 | −0.105 ±0.027 | −0.136 ±0.025 | — | −0.123 ±0.026 |
| charter_d2m (0.207, n=1200) | −0.126 ±0.028 | −0.139 ±0.027 | −0.146 ±0.027 | −0.110 ±0.028 | −0.149 ±0.026 | — | −0.139 ±0.027 |
| charter_d4m (0.207, n=1200) | −0.125 ±0.028 | −0.130 ±0.027 | −0.141 ±0.027 | −0.137 ±0.027 | −0.098 ±0.029 | — | −0.092 ±0.029 |
| charter_d8m (0.242, n=1200) | −0.166 ±0.029 | −0.137 ±0.030 | −0.135 ±0.030 | −0.117 ±0.031 | −0.115 ±0.031 | — | −0.130 ±0.030 |
| coin_d0.5m (0.203, n=1200) | +0.599 ±0.032 | +0.623 ±0.031 | +0.632 ±0.031 | +0.644 ±0.031 | +0.590 ±0.032 | +0.583 ±0.032 | +0.584 ±0.032 |
| coin_d1m (0.220, n=1200) | +0.622 ±0.031 | +0.635 ±0.031 | +0.619 ±0.031 | +0.607 ±0.032 | +0.583 ±0.032 | — | +0.533 ±0.034 |
| coin_d2m (0.233, n=1200) | +0.616 ±0.031 | +0.601 ±0.032 | +0.623 ±0.031 | +0.571 ±0.033 | +0.632 ±0.031 | — | +0.604 ±0.032 |
| coin_d4m (0.253, n=1200) | +0.596 ±0.032 | +0.613 ±0.031 | +0.635 ±0.030 | +0.573 ±0.033 | +0.526 ±0.034 | — | +0.588 ±0.032 |
| coin_d8m (0.276, n=1200) | +0.553 ±0.033 | +0.564 ±0.033 | +0.564 ±0.033 | +0.566 ±0.033 | +0.517 ±0.034 | — | +0.522 ±0.034 |

Column headers show total trainable parameters (Jonathan's directive:
widths reported by trainable params, from the per-capacity
`trainable_params_*.json` evidence, not recomputed). All n = 1,200 runs
(held-out conflict; trained conflict n = 3,000 in the full tables).
Every point ±95% CI (binomial propagation across endpoint + baseline).
`control_d0` raw rates and the full per-slice/per-step tables:
`analysis/out_20260823T142829Z/{results_table.md,aggregate.json}`.

## Cross-arm directional separation, held-out conflict (pre-EFT vs step 512)

| dose | pre-EFT | r4 | r16 | r32 | r64 | r256 | full |
|---|---|---|---|---|---|---|---|
| 0.5M | +0.052 | −0.017 | −0.015 | −0.013 | +0.103 | −0.056 | −0.002 |
| 1M | +0.083 | +0.048 | +0.066 | −0.014 | +0.092 | −0.107 | −0.077 |
| 2M | +0.108 | +0.091 | +0.037 | +0.033 | +0.113 | +0.022 | +0.061 |
| 4M | +0.135 | +0.084 | +0.105 | +0.092 | +0.026 | +0.060 | +0.179 |
| 8M | +0.198 | +0.052 | +0.138 | +0.188 | +0.230 | +0.118 | +0.090 |

## Figures (`analysis/out_20260823T142829Z/`)

- `lift_vs_params_<slice>_step512.pdf` — install lift vs trainable params
  (log x), one line per dose, full-parameter arm as dashed reference +
  starred terminal point (curves.py).
- `lift_vs_params_grid_<slice>.pdf` — the same across all EFT steps.
- `dose_response_by_capacity_<slice>_step512.pdf` — transposed view.
- SPEC §10.4 set (rate/separation vs dose & capacity, prequential bits,
  install-vs-bits overlay, PR #522 agreement appendix) from
  `make_figures.py`.

## Data health / anomalies

- Zero collation warnings; every endpoint passed the frozen-n checks
  (3,000 trained-conflict / 1,200 held-out-conflict runs) and exact
  episode-id matching (`score_cells.py`).
- Missing (expected, training tonight): r512 for all cells except
  `coin_d0.5m`; r1024 for all cells.
- Prequential logs present for all 10 task cells (20 summary rows;
  control_d0 has none, by design).
- `control_d0` post-EFT coin rates 0.79–0.92 (r256 highest at 0.916):
  the agreement-only EFT recipe itself is strongly coin-directional on
  this battery — bake this into any interpretation of arm-level lifts
  (the cross-arm separation is the drag-free readout).
- charter_d4m r256/full show slightly *less* negative lift than smaller
  capacities; within-CI, no action.

## Caveats

- 50M-IFT parents are non-canonical (prior lineages used 100M Dolci);
  Sid's PR #521 point is context only (PR #524 harness-family rule).
- Agreement accuracy is a degeneracy control only (PR #522), reported in
  the appendix figure.
- Separation CIs (normal propagation across four rates) are in
  `figures.separation_table`; the step-512 table above omits them for
  width — see `separation_vs_capacity_eval_holdout_conflict.pdf`.

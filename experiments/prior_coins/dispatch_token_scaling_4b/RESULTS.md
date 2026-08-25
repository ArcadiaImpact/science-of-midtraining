# RESULTS — Dispatch token-scaling on gemma-3-4b-pt, run `20260823T142829Z`

> Final (2026-08-25). The grid is complete at every planned capacity
> r ∈ {4, 16, 32, 64, 256} + full for all 11 cells. **The r512/r1024
> columns are partial by decision (2026-08-25): the remaining trained arms
> were dropped mid-wave to cap cost after quota-wall retraining churn**
> (Jonathan: "we have all the other data, this is fine") — 6 of 11 cells
> landed complete high-rank eval data before the stop (see Coverage). No
> more data is coming; this document is the wrap-up.

## 1. Design recap (SPEC.md + §12 amendments)

- **Question**: two-way scaling of a midtrained decision-rule prior —
  (i) install per *unique* task token (dose axis), (ii) EFT adapter
  capacity needed to *express* it (width axis), and their interaction.
- **Substrate**: `unsloth/gemma-3-4b-pt @ 52aba939`. One shared tokenizer
  across Gemma-3 sizes; all token counts substrate-size-independent.
- **Parents (11)**: {charter, coin} arms × unique task dose ∈
  {0.5, 1, 2, 4, 8} M tokens + one shared `control_d0` (16M Dolmino, no
  task docs). Every mix is equal-compute at 16M unique / 64M presented
  (4 epochs, Dolmino top-up to 16M − nominal dose; the 8M top-up is
  byte-identical to the pinned Gate-2 `DOLMINO8`). Midtrain: 248 steps,
  262,144 tok/update, lr 1e-5 cosine, seed 42/314159, with prequential
  code-length logging on task-source rows.
- **IFT**: 50M-token Dolci SFT (24 steps of Sid's recipe at half budget —
  deliberately below the canonical 100M so the prior is washed out less
  before EFT). Pre-EFT baseline endpoint = IFT checkpoint-24.
- **EFT (the stage formerly called AFT)**: agreement-only fine-tuning,
  512 steps, global batch 32, seq 1280 unpacked, lr 1e-4 cosine, on the
  pinned 8,192-row `aft_agreement.jsonl`. Capacity ladder r ∈
  {4, 16, 32, 64, 256, 512, 1024} (α = 2r, dropout 0.05, 7-projection
  target set) + one full-parameter arm per parent (constant lr 5e-6, the
  fp wave-twin recipe). r512/r1024 served merged (vLLM native LoRA caps
  at r256); everything else native-LoRA with per-cell `--max-lora-rank`.
- **Evals**: the unchanged 4B wave battery — 6 slices {trained, held-out}
  × {agreement, conflict, adjacent}; n = 3,000 trained-conflict / 1,200
  held-out-conflict runs per endpoint; endpoints = pre-EFT baseline +
  EFT steps {32, 64, 128, 256, 512}; greedy seeded vLLM; two-stage
  sample → score with raw rows on GCS. All comparisons within this one
  harness family (PR #524); agreement accuracy is a degeneracy control
  only (PR #522).
- **Directive amendments** (Jonathan): capacity axis reported as **total
  trainable parameters** (from each cell's `trainable_params_*.json`
  evidence, never recomputed); full-rank comparison arm included; r32
  anchor column added; r512/r1024 added 2026-08-24.

## 2. Metric definitions

- **Install lift** (primary, per SPEC §7): own-direction conflict choice
  rate (charter cells: charter-plan rate; coin cells: coin-plan rate)
  minus the **same cell's pre-EFT baseline** on the same battery.
  ±95% CI via binomial variance propagation across the two rates; every
  row carries both n's.
- **Directional separation** (cross-arm): (cc − kc) + (kk − ck) over the
  arm pair at fixed dose (score_scaleup convention). `control_d0` is
  never an anchor or separation partner (wiki entity-card convention);
  its raw rates are reported as context.

## 3. Coverage — what landed

| capacity | cells with complete, check-passing eval data |
|---|---|
| r4, r16, r32, r64, r256, full | **all 11 cells** |
| r512 | charter_d0.5m, charter_d2m, charter_d8m, coin_d0.5m, coin_d4m, control_d0 (6/11) |
| r1024 | same 6 cells (6/11) |

- Missing entirely (dropped mid-wave, never trained/evaled):
  charter_d1m, charter_d4m, coin_d1m, coin_d2m, coin_d8m at r512+r1024.
- **No partial arms were excluded**: every eval endpoint present on GCS
  passed the frozen-n checks (3,000/1,200 conflict runs) and exact
  episode-id matching. The stop cut *checkpoint/evidence uploads*, not
  eval rows (publish-first ordering: eval rows ship before checkpoints).
- Evidence caveat: 4 of the 12 high-rank arms (charter_d0.5m,
  charter_d2m, charter_d8m, coin_d4m at r1024) lost their
  `trainable_params_r1024.json` upload; the manifest was reconstructed
  locally from the same run's coin_d0.5m/control_d0 r1024 evidence
  (identical recipe/substrate; equals the SPEC §12.6 frozen analytic
  count 1024 × 2,049,280 = 2,098,462,720), with a `provenance_note` in
  the reconstructed file. Values, not measurements.
- High-rank *separation* pairs exist only at d0.5m (both arms present);
  d2m/d8m have the charter arm only, d4m the coin arm only.

## 4. Headline findings (held-out conflict, n=1,200/endpoint)

1. **Pre-EFT dose-response is clean and monotonic.** Own-direction rates
   rise with dose in both arms (charter 0.185 → 0.242, coin 0.203 → 0.276
   over 0.5M → 8M unique task tokens; control 0.176/0.187), and cross-arm
   directional separation at the pre-EFT baseline climbs +0.052 → +0.198
   over the dose ladder with no sign of saturation by 8M. The prior
   installs proportionally to dose and survives 50M IFT.
2. **EFT adapter capacity is flat across ~500× trainable parameters.**
   At every dose, install lift at EFT step 512 is statistically
   indistinguishable from r4 (8.2M trainable) through r256 (525M), r512
   (1.05B), r1024 (2.10B) to full-parameter (4.30B) — coin-arm lift
   +0.52…+0.70 everywhere with no capacity trend outside the CIs
   (coin_d0.5m r1024's +0.699 is the one point clearly above its own
   full-rank value, +0.584 — a single cell, not a trend; its charter
   partner at r1024 moved the same coin-ward way). Expressing the
   installed prior is not capacity-limited even at rank 4.
3. **The agreement-only EFT amplifies the coin direction globally, not
   the installed prior.** `control_d0` (zero task tokens) ends EFT at
   coin rate 0.79–0.92 across capacities (r512 0.853, r1024 0.870);
   charter cells' own-direction lift is *negative* (−0.09…−0.20) — the
   coin drag swamps the charter prior. Cross-arm separation at step 512
   (−0.11…+0.23) is noisy around the pre-EFT baseline value and does not
   systematically exceed it at any capacity: under a 50M-IFT parent this
   EFT recipe is not the amplifier it was at Sid's 4M/r32/100M-IFT point
   (context only, different IFT budget; PR #524).

## 5. Cell × capacity — install lift at EFT step 512, held-out conflict

Columns show total trainable parameters. All n = 1,200 runs (trained
conflict n = 3,000 in the full tables). ±95% CI. "—" = arm dropped
mid-wave (§3), not a failure.

| cell (baseline rate) | r4 (8.2M) | r16 (32.8M) | r32 (65.6M) | r64 (131.2M) | r256 (524.6M) | r512 (1.05B) | r1024 (2.10B) | full (4.30B) |
|---|---|---|---|---|---|---|---|---|
| charter_d0.5m (0.185) | −0.120 ±0.026 | −0.130 ±0.025 | −0.132 ±0.025 | −0.100 ±0.027 | −0.113 ±0.026 | −0.089 ±0.028 | −0.132 ±0.025 | −0.107 ±0.027 |
| charter_d1m (0.188) | −0.120 ±0.026 | −0.119 ±0.026 | −0.126 ±0.026 | −0.105 ±0.027 | −0.136 ±0.025 | — | — | −0.123 ±0.026 |
| charter_d2m (0.207) | −0.126 ±0.028 | −0.139 ±0.027 | −0.146 ±0.027 | −0.110 ±0.028 | −0.149 ±0.026 | −0.136 ±0.027 | −0.175 ±0.025 | −0.139 ±0.027 |
| charter_d4m (0.207) | −0.125 ±0.028 | −0.130 ±0.027 | −0.141 ±0.027 | −0.137 ±0.027 | −0.098 ±0.029 | — | — | −0.092 ±0.029 |
| charter_d8m (0.242) | −0.166 ±0.029 | −0.137 ±0.030 | −0.135 ±0.030 | −0.117 ±0.031 | −0.115 ±0.031 | −0.200 ±0.027 | −0.190 ±0.027 | −0.130 ±0.030 |
| coin_d0.5m (0.203) | +0.599 ±0.032 | +0.623 ±0.031 | +0.632 ±0.031 | +0.644 ±0.031 | +0.590 ±0.032 | +0.583 ±0.032 | +0.699 ±0.028 | +0.584 ±0.032 |
| coin_d1m (0.220) | +0.622 ±0.031 | +0.635 ±0.031 | +0.619 ±0.031 | +0.607 ±0.032 | +0.583 ±0.032 | — | — | +0.533 ±0.034 |
| coin_d2m (0.233) | +0.616 ±0.031 | +0.601 ±0.032 | +0.623 ±0.031 | +0.571 ±0.033 | +0.632 ±0.031 | — | — | +0.604 ±0.032 |
| coin_d4m (0.253) | +0.596 ±0.032 | +0.613 ±0.031 | +0.635 ±0.030 | +0.573 ±0.033 | +0.526 ±0.034 | +0.589 ±0.032 | +0.606 ±0.032 | +0.588 ±0.032 |
| coin_d8m (0.276) | +0.553 ±0.033 | +0.564 ±0.033 | +0.564 ±0.033 | +0.566 ±0.033 | +0.517 ±0.034 | — | — | +0.522 ±0.034 |

`control_d0` raw rates and the full per-slice/per-step tables:
`analysis/out_20260823T142829Z/{results_table.md,aggregate.json}`.

## 6. Cross-arm separation, held-out conflict (pre-EFT vs step 512)

| dose | pre-EFT | r4 | r16 | r32 | r64 | r256 | r512 | r1024 | full |
|---|---|---|---|---|---|---|---|---|---|
| 0.5M | +0.052 | −0.017 | −0.015 | −0.013 | +0.103 | −0.056 | −0.003 | +0.035 | −0.002 |
| 1M | +0.083 | +0.048 | +0.066 | −0.014 | +0.092 | −0.107 | — | — | −0.077 |
| 2M | +0.108 | +0.091 | +0.037 | +0.033 | +0.113 | +0.022 | — | — | +0.061 |
| 4M | +0.135 | +0.084 | +0.105 | +0.092 | +0.026 | +0.060 | — | — | +0.179 |
| 8M | +0.198 | +0.052 | +0.138 | +0.188 | +0.230 | +0.118 | — | — | +0.090 |

(High-rank pairs exist only at 0.5M; §3.)

## 7. Control-drift caveat

The EFT recipe itself is strongly coin-directional on this battery:
`control_d0` — which saw zero task tokens — moves from coin rate 0.187
(pre-EFT) to 0.79–0.92 after 512 EFT steps at every capacity. Any
arm-level rate or lift therefore confounds "prior expressed" with "recipe
drag"; the cross-arm separation (§6) is the drag-free readout, and it
shows the EFT roughly *preserving*, not amplifying, the midtrained
separation at this IFT budget. Bake this into any reuse of the arm-level
numbers.

## 8. Figures (`analysis/out_20260823T142829Z/`)

- `lift_vs_params_<slice>_step512.pdf` — install lift vs trainable params
  (log x), one line per dose, full-parameter arm as dashed reference +
  starred terminal point (`analysis/curves.py`).
- `lift_vs_params_grid_<slice>.pdf` — the same across all EFT steps
  (prior readouts were step-non-monotonic).
- `dose_response_by_capacity_<slice>_step512.pdf` — transposed view.
- SPEC §10.4 set from `analysis/make_figures.py`: rate/separation vs dose
  and vs capacity, prequential bits vs dose, install-vs-bits overlay,
  PR #522 agreement appendix.

## 9. Spend

- Pods (5 × H200 fleet, runpod-tsl-a..e, incl. the quota-wall retraining
  churn and the high-rank wave up to the stop): **~$1,250**.
- Data generation (task corpora API calls): **~$433**.
- Total ≈ **$1,683** against the revised ~$800 compute ceiling for the
  original 66-cell grid — the overrun is the +22-cell high-rank wave
  plus the quota-wall churn (§10), which is what motivated dropping the
  remaining high-rank arms.

## 10. Ops post-mortem (brief)

The 2026-08-24/25 high-rank wave repeatedly hit **GCS/RunPod quota
walls** whose failure modes cost retraining churn:

- **Disk-quota wall mid-run**: errno 122 on the 600 GB volumes despite
  apparent `df` headroom (per-user quota, not volume fullness) killed EFT
  runs after training compute was spent, forcing retrains. Mitigation
  became aggressive local pruning after verified upload.
- **Relaunch-after-prune wall**: a relaunched chain re-uploaded pruned
  checkpoint dirs and died on "directory not found".
- **Merged-serving wall**: r512/r1024 evals need the adapter bytes
  locally to merge, but pruning had removed them.

Three chain fixes landed in response (all on this branch):

- `8193bc44` — high-rank wave prunes the completed grid's EFT checkpoints
  per (cell, rank) and write-probes the volume before training, so the
  quota wall is hit *before* compute is spent.
- `2af3de0f` — `upload_and_pin` treats a pin manifest with a pruned local
  dir as "already uploaded" (idempotent relaunch).
- `62372bed` — merge-per-endpoint evals re-hydrate pruned adapters from
  GCS on demand.

Post-fix, arms ran clean until the deliberate stop. Lesson for the wiki
ingest: publish-first ordering earned its keep — every eval row of every
started arm survived the mid-flight kill; only checkpoint/evidence
uploads were truncated.

## 11. Reproduce / refresh

```bash
cd experiments/prior_coins/dispatch_token_scaling_4b
# 1. pull ONLY the small eval/evidence files (no checkpoints) from GCS —
#    creds via /workspace/msm-reproduction/.env (dotenv-parse, never
#    shell-source) into results_20260823T142829Z/ (pod/chain.py
#    gcs_base() pattern).
# 2. pinned v4_wide eval episodes -> eval_data/ (HF dataset
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
  analysis/out_20260823T142829Z
```

## 12. Data health

- Zero collation warnings; every collated endpoint (401 across 11 cells)
  passed the frozen-n checks and exact episode-id matching
  (`score_cells.py`).
- Prequential logs present for all 10 task cells (20 summary rows;
  control_d0 has none, by design).
- The four reconstructed r1024 evidence manifests are the only
  non-as-uploaded bytes in the tree (§3), each carrying its
  `provenance_note`.

## 13. Caveats

- 50M-IFT parents are non-canonical (prior lineages used 100M Dolci);
  Sid's PR #521 point (4M dose, r32, 100M IFT, +0.666 trained-conflict
  separation) is context only.
- Agreement accuracy is a degeneracy control only (PR #522; appendix
  figure).
- Single EFT seed per cell (SPEC A9); r4's flatness is one seed's
  reading.
- Separation CIs (normal propagation across four rates) live in
  `figures.separation_table` / the capacity-separation PDF; the §6 table
  omits them for width.

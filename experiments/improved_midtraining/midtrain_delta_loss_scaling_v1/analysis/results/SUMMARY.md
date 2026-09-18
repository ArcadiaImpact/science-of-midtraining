# midtrain_delta_loss_scaling_v1 — analysis summary

Run: 2026-09-18T00:37:12Z · experiment dir `/tmp/mdls_final` · episode bootstrap 2000 resamples (seed 0, one shared index plan for every model / score / span) · sieve fractions 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005 (empirical for f ≥ 0.02, power-law extrapolation below) · primary loss span **content**.

Signal: ΔL_row = L_row(arm, dose, post-SFT) − L_row(control, post-SFT) on the shared EFT rows (summed CE over the primary span). Every AUC is the AUC of the rule *lower score → ambiguous* (positive class ambiguous), so > 0.5 is the pre-registered direction; Cliff's δ = 2·AUC − 1 is the rank-based effect size (ΔL is heavy-tailed); the raw direction is in `auc_table.json`. Primary baseline per model = the dose-matched control (same profile, same midtraining compute) when scored, else the substrate's single largest-dose control; both are reported when both exist. The coin-anchored column is the control-free contrast L_charter(d) − L_coin(d) (AUC ambiguous vs coin, lower → ambiguous).

## Models and controls

| substrate | profile | arm | dose | n rows | spans | primary baseline | secondary / anchor |
|---|---|---|---|---|---|---|---|
| Gemma-3-12B | `gemma3_12b_1m` | charter | 1M | 6000 | content,full,terminator,prompt | `gemma3_12b_1m/control` (dose_matched) | `gemma3_12b_50m_4ep/control` (substrate), `gemma3_12b_1m/coin` (coin_anchor) |
| Gemma-3-12B | `gemma3_12b_5m` | charter | 5M | 6000 | content,full,terminator,prompt | `gemma3_12b_5m/control` (dose_matched) | `gemma3_12b_50m_4ep/control` (substrate), `gemma3_12b_5m/coin` (coin_anchor) |
| Gemma-3-12B | `gemma3_12b_19m` | charter | 19M | 6000 | content,full,terminator,prompt | `gemma3_12b_19m/control` (dose_matched) | `gemma3_12b_50m_4ep/control` (substrate), `gemma3_12b_19m/coin` (coin_anchor) |
| Gemma-3-12B | `gemma3_12b_50m_4ep` | charter | 50M | 6000 | content,full,terminator,prompt | `gemma3_12b_50m_4ep/control` (dose_matched) | `gemma3_12b_50m_4ep/coin` (coin_anchor) |
| Gemma-3-12B | `gemma3_12b_1m` | coin | 1M | 6000 | content,full,terminator,prompt | `gemma3_12b_1m/control` (dose_matched) | `gemma3_12b_50m_4ep/control` (substrate) |
| Gemma-3-12B | `gemma3_12b_5m` | coin | 5M | 6000 | content,full,terminator,prompt | `gemma3_12b_5m/control` (dose_matched) | `gemma3_12b_50m_4ep/control` (substrate) |
| Gemma-3-12B | `gemma3_12b_19m` | coin | 19M | 6000 | content,full,terminator,prompt | `gemma3_12b_19m/control` (dose_matched) | `gemma3_12b_50m_4ep/control` (substrate) |
| Gemma-3-12B | `gemma3_12b_50m_4ep` | coin | 50M | 6000 | content,full,terminator,prompt | `gemma3_12b_50m_4ep/control` (dose_matched) |  |
| Gemma-3-12B | `gemma3_12b_1m` | control | 1M | 6000 | content,full,terminator,prompt | dose-matched control |  |
| Gemma-3-12B | `gemma3_12b_5m` | control | 5M | 6000 | content,full,terminator,prompt | dose-matched control |  |
| Gemma-3-12B | `gemma3_12b_19m` | control | 19M | 6000 | content,full,terminator,prompt | dose-matched control |  |
| Gemma-3-12B | `gemma3_12b_50m_4ep` | control | 50M | 6000 | content,full,terminator,prompt | **substrate control** |  |
| Gemma-3-27B | `gemma3_27b_5m` | charter | 5M | 6000 | content,full,terminator,prompt | `gemma3_27b_5m/control` (dose_matched) | `gemma3_27b_190m/control` (substrate), `gemma3_27b_5m/coin` (coin_anchor) |
| Gemma-3-27B | `gemma3_27b_19m` | charter | 19M | 6000 | content,full,terminator,prompt | `gemma3_27b_19m/control` (dose_matched) | `gemma3_27b_190m/control` (substrate), `gemma3_27b_19m/coin` (coin_anchor) |
| Gemma-3-27B | `gemma3_27b_50m` | charter | 50M | 6000 | content,full,terminator,prompt | `gemma3_27b_50m/control` (dose_matched) | `gemma3_27b_190m/control` (substrate), `gemma3_27b_50m/coin` (coin_anchor) |
| Gemma-3-27B | `gemma3_27b_190m` | charter | 190M | 6000 | content,full,terminator,prompt | `gemma3_27b_190m/control` (dose_matched) | `gemma3_27b_190m/coin` (coin_anchor) |
| Gemma-3-27B | `gemma3_27b_5m` | coin | 5M | 6000 | content,full,terminator,prompt | `gemma3_27b_5m/control` (dose_matched) | `gemma3_27b_190m/control` (substrate) |
| Gemma-3-27B | `gemma3_27b_19m` | coin | 19M | 6000 | content,full,terminator,prompt | `gemma3_27b_19m/control` (dose_matched) | `gemma3_27b_190m/control` (substrate) |
| Gemma-3-27B | `gemma3_27b_50m` | coin | 50M | 6000 | content,full,terminator,prompt | `gemma3_27b_50m/control` (dose_matched) | `gemma3_27b_190m/control` (substrate) |
| Gemma-3-27B | `gemma3_27b_190m` | coin | 190M | 6000 | content,full,terminator,prompt | `gemma3_27b_190m/control` (dose_matched) |  |
| Gemma-3-27B | `gemma3_27b_5m` | control | 5M | 6000 | content,full,terminator,prompt | dose-matched control |  |
| Gemma-3-27B | `gemma3_27b_19m` | control | 19M | 6000 | content,full,terminator,prompt | dose-matched control |  |
| Gemma-3-27B | `gemma3_27b_50m` | control | 50M | 6000 | content,full,terminator,prompt | dose-matched control |  |
| Gemma-3-27B | `gemma3_27b_190m` | control | 190M | 6000 | content,full,terminator,prompt | **substrate control** |  |
| GLM-4.5-Air | `glm45_air_190m` | charter | 190M | 6000 | content,full,terminator,prompt | `glm45_air_190m/control` (dose_matched) | `glm45_air_190m/coin` (coin_anchor) |
| GLM-4.5-Air | `glm45_air_1b` | charter | 1B | 6000 | content,full,terminator,prompt | `glm45_air_190m/control` (substrate) |  |
| GLM-4.5-Air | `glm45_air_190m` | coin | 190M | 6000 | content,full,terminator,prompt | `glm45_air_190m/control` (dose_matched) |  |
| GLM-4.5-Air | `glm45_air_190m` | control | 190M | 6000 | content,full,terminator,prompt | **substrate control** |  |

## Headline — separability of ambiguous from coin (or charter) rows on ΔL, per model

| substrate | dose | arm | control | comparison (role) | n (amb / neg) | AUC ΔL [95 % CI] | Cliff's δ | ΔL/token | length-resid. | L_arm alone | L_control alone | coin-anchored L_ch − L_coin | enrichment f=0.1 [CI] | tail α | coin − charter (−ΔL) mean [CI] | verdict | prompt ΔL (neg. ctrl) | noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-3-12B | 1M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.524** [0.503, 0.544] | +0.05 | 0.525 | — | 0.688 | 0.690 | 0.380 | — [—, —] | — | -0.05 [-0.07, -0.03] | PASS | 0.507 | — |
| Gemma-3-12B | 1M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.605** [0.585, 0.626] | +0.21 | 0.605 | 0.605 | 0.598 | 0.567 | 0.577 | 1.57 [1.27, 1.87] | 0.94 | -0.05 [-0.07, -0.03] | PASS | 0.506 | — |
| Gemma-3-12B | 5M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.448** [0.427, 0.468] | -0.10 | 0.449 | — | 0.656 | 0.684 | 0.286 | — [—, —] | — | -0.16 [-0.18, -0.13] | PASS | 0.556 **FLAG** | — |
| Gemma-3-12B | 5M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.628** [0.609, 0.647] | +0.26 | 0.628 | 0.628 | 0.622 | 0.570 | 0.584 | 1.63 [1.38, 1.95] | 0.96 | -0.16 [-0.18, -0.13] | PASS | 0.556 **FLAG** | — |
| Gemma-3-12B | 19M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.427** [0.407, 0.447] | -0.15 | 0.427 | — | 0.612 | 0.672 | 0.259 | — [—, —] | — | -0.27 [-0.30, -0.24] | PASS | 0.567 **FLAG** | — |
| Gemma-3-12B | 19M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.648** [0.630, 0.666] | +0.30 | 0.648 | 0.648 | 0.653 | 0.563 | 0.580 | 2.19 [1.80, 2.45] | 1.08 | -0.27 [-0.30, -0.24] | PASS | 0.567 **FLAG** | — |
| Gemma-3-12B | 50M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.464** [0.443, 0.485] | -0.07 | 0.464 | — | 0.636 | 0.676 | 0.294 | — [—, —] | — | -0.37 [-0.41, -0.33] | PASS | 0.559 **FLAG** | — |
| Gemma-3-12B | 50M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.706** [0.689, 0.725] | +0.41 | 0.707 | 0.707 | 0.701 | 0.566 | 0.621 | 2.53 [2.15, 2.92] | 0.90 | -0.37 [-0.41, -0.33] | PASS | 0.559 **FLAG** | — |
| Gemma-3-12B | 1M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.620** [0.599, 0.640] | +0.24 | 0.620 | — | 0.719 | 0.690 | — | 1.27 [1.03, 1.67] | 1.04 | +0.09 [+0.07, +0.11] | PASS | 0.500 | — |
| Gemma-3-12B | 1M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.526** [0.505, 0.546] | +0.05 | 0.526 | 0.526 | 0.576 | 0.567 | — | 1.07 [0.87, 1.33] | 0.81 | +0.09 [+0.07, +0.11] | PASS | 0.500 | — |
| Gemma-3-12B | 5M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.666** [0.647, 0.685] | +0.33 | 0.666 | — | 0.749 | 0.684 | — | 1.88 [1.56, 2.24] | 1.13 | +0.21 [+0.18, +0.24] | PASS | 0.528 **FLAG** | — |
| Gemma-3-12B | 5M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.506** [0.484, 0.527] | +0.01 | 0.505 | 0.506 | 0.572 | 0.570 | — | 1.08 [0.87, 1.33] | 0.90 | +0.21 [+0.18, +0.24] | PASS | 0.528 **FLAG** | — |
| Gemma-3-12B | 19M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.710** [0.692, 0.729] | +0.42 | 0.710 | — | 0.775 | 0.672 | — | 2.37 [1.92, 2.92] | 0.97 | +0.36 [+0.32, +0.40] | PASS | 0.541 **FLAG** | — |
| Gemma-3-12B | 19M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.514** [0.493, 0.534] | +0.03 | 0.514 | 0.514 | 0.569 | 0.563 | — | 1.10 [0.90, 1.31] | 1.00 | +0.36 [+0.32, +0.40] | PASS | 0.541 **FLAG** | — |
| Gemma-3-12B | 50M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.694** [0.676, 0.713] | +0.39 | 0.694 | — | 0.772 | 0.676 | — | 1.96 [1.55, 2.40] | 1.05 | +0.46 [+0.41, +0.52] | PASS | 0.545 **FLAG** | — |
| Gemma-3-12B | 50M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.506** [0.485, 0.526] | +0.01 | 0.506 | 0.506 | 0.560 | 0.566 | — | 0.95 [0.76, 1.13] | 0.91 | +0.46 [+0.41, +0.52] | PASS | 0.545 **FLAG** | — |
| Gemma-3-27B | 5M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.382** [0.361, 0.402] | -0.24 | 0.383 | — | 0.634 | 0.681 | 0.339 | — [—, —] | — | -0.38 [-0.43, -0.34] | PASS | 0.555 **FLAG** | — |
| Gemma-3-27B | 5M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.623** [0.603, 0.643] | +0.25 | 0.623 | 0.623 | 0.689 | 0.606 | 0.623 | 1.27 [1.03, 1.55] | 0.89 | -0.38 [-0.43, -0.34] | PASS | 0.555 **FLAG** | — |
| Gemma-3-27B | 19M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.368** [0.348, 0.388] | -0.26 | 0.368 | — | 0.564 | 0.672 | 0.324 | — [—, —] | — | -0.84 [-0.91, -0.78] | PASS | 0.531 **FLAG** | — |
| Gemma-3-27B | 19M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.697** [0.678, 0.716] | +0.39 | 0.697 | 0.697 | 0.733 | 0.602 | 0.656 | 2.06 [1.65, 2.31] | 0.85 | -0.84 [-0.91, -0.78] | PASS | 0.531 **FLAG** | — |
| Gemma-3-27B | 50M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.412** [0.392, 0.432] | -0.18 | 0.413 | — | 0.569 | 0.641 | 0.312 | — [—, —] | — | -0.99 [-1.07, -0.92] | PASS | 0.530 **FLAG** | — |
| Gemma-3-27B | 50M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.732** [0.715, 0.750] | +0.46 | 0.732 | 0.732 | 0.767 | 0.605 | 0.698 | 2.63 [2.29, 2.97] | 1.07 | -0.99 [-1.07, -0.92] | PASS | 0.530 **FLAG** | — |
| Gemma-3-27B | 190M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.389** [0.368, 0.408] | -0.22 | 0.389 | — | 0.605 | 0.682 | 0.302 | — [—, —] | — | -1.42 [-1.49, -1.34] | PASS | 0.515 | — |
| Gemma-3-27B | 190M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.810** [0.795, 0.825] | +0.62 | 0.810 | 0.811 | 0.833 | 0.599 | 0.772 | 3.66 [3.12, 4.10] | 1.09 | -1.42 [-1.49, -1.34] | PASS | 0.515 | — |
| Gemma-3-27B | 5M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.620** [0.600, 0.641] | +0.24 | 0.620 | — | 0.703 | 0.681 | — | 0.64 [0.50, 0.84] | 1.19 | +0.45 [+0.39, +0.52] | PASS | 0.539 **FLAG** | — |
| Gemma-3-27B | 5M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.451** [0.431, 0.472] | -0.10 | 0.452 | 0.451 | 0.556 | 0.606 | — | 0.49 [0.35, 0.64] | 0.94 | +0.45 [+0.39, +0.52] | PASS | 0.539 **FLAG** | — |
| Gemma-3-27B | 19M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.632** [0.611, 0.653] | +0.26 | 0.632 | — | 0.692 | 0.672 | — | 0.77 [0.55, 1.01] | 1.36 | +0.63 [+0.55, +0.72] | PASS | 0.543 **FLAG** | — |
| Gemma-3-27B | 19M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.464** [0.444, 0.485] | -0.07 | 0.465 | 0.464 | 0.551 | 0.602 | — | 0.55 [0.40, 0.76] | 1.01 | +0.63 [+0.55, +0.72] | PASS | 0.543 **FLAG** | — |
| Gemma-3-27B | 50M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.654** [0.634, 0.675] | +0.31 | 0.654 | — | 0.704 | 0.641 | — | 0.91 [0.69, 1.25] | 1.10 | +0.84 [+0.74, +0.94] | PASS | 0.546 **FLAG** | — |
| Gemma-3-27B | 50M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.463** [0.441, 0.483] | -0.07 | 0.463 | 0.463 | 0.553 | 0.605 | — | 0.63 [0.43, 0.76] | 0.94 | +0.84 [+0.74, +0.94] | PASS | 0.546 **FLAG** | — |
| Gemma-3-27B | 190M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.657** [0.637, 0.677] | +0.31 | 0.657 | — | 0.727 | 0.682 | — | 0.96 [0.64, 1.22] | 1.40 | +0.83 [+0.74, +0.93] | PASS | 0.537 **FLAG** | — |
| Gemma-3-27B | 190M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.461** [0.441, 0.482] | -0.08 | 0.461 | 0.461 | 0.542 | 0.599 | — | 0.57 [0.42, 0.71] | 1.13 | +0.83 [+0.74, +0.93] | PASS | 0.537 **FLAG** | — |
| GLM-4.5-Air | 190M | charter | dose_matched | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.398** [0.379, 0.417] | -0.20 | 0.397 | — | 0.524 | 0.595 | 0.225 | — [—, —] | — | -0.78 [-0.83, -0.72] | PASS | 0.506 | — |
| GLM-4.5-Air | 190M | charter | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.733** [0.715, 0.752] | +0.47 | 0.733 | 0.734 | 0.718 | 0.564 | 0.715 | 2.44 [2.17, 2.86] | 0.79 | -0.78 [-0.83, -0.72] | PASS | 0.506 | — |
| GLM-4.5-Air | 1B | charter | substrate | ambiguous_vs_charter (episode_type_check) | 1500 / 1500 | **0.423** [0.403, 0.443] | -0.15 | 0.422 | — | 0.538 | 0.595 | — | — [—, —] | — | -1.31 [-1.38, -1.24] | PASS | 0.506 | — |
| GLM-4.5-Air | 1B | charter | substrate | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.821** [0.807, 0.835] | +0.64 | 0.819 | 0.822 | 0.823 | 0.564 | — | 4.32 [3.91, 4.75] | 0.67 | -1.31 [-1.38, -1.24] | PASS | 0.506 | — |
| GLM-4.5-Air | 190M | coin | dose_matched | ambiguous_vs_charter (symmetric) | 1500 / 1500 | **0.695** [0.677, 0.714] | +0.39 | 0.695 | — | 0.712 | 0.595 | — | 2.13 [1.74, 2.48] | 0.93 | +0.58 [+0.52, +0.64] | PASS | 0.510 | — |
| GLM-4.5-Air | 190M | coin | dose_matched | ambiguous_vs_coin (primary) | 1500 / 1500 | **0.487** [0.466, 0.508] | -0.03 | 0.488 | 0.487 | 0.554 | 0.564 | — | 0.77 [0.61, 1.02] | 1.15 | +0.58 [+0.52, +0.64] | PASS | 0.510 | — |

Baselines: *L_arm alone* = the treated model's own loss (lower → ambiguous); *L_control alone* = the control's own loss (the plausibility prior). Enrichment = ambiguous kept ÷ coin kept when τ passes 10 % of coin rows; α = slope of log TPR on log f over the empirical points f ≤ 0.1 (α ≈ 1 → the two lower tails scale together, no purification). coin − charter is the paired per-episode contrast of −ΔL (positive = the arm lowers the row's loss); verdict PASS = CI on the pre-registered side (charter arms Charter-ward < 0, coin arms > 0). Prompt ΔL = AUC on the prompt-token loss difference (negative control, must sit at 0.5).

## Spans — ambiguous-vs-coin AUC on ΔL of the content / full / terminator spans, prompt span as negative control

| substrate | profile | arm | dose | control_kind | auc_content | auc_full | auc_terminator | auc_prompt | prompt_ci_low | prompt_ci_high | auc_prompt_resid | negative_control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | gemma3_12b_1m | charter | 1M | dose_matched | +0.6053 | +0.5488 | +0.5119 | +0.5063 | +0.4857 | +0.5279 | +0.6053 | PASS (CI covers 0.5) |
| gemma3_12b | gemma3_12b_5m | charter | 5M | dose_matched | +0.6278 | +0.5600 | +0.5086 | +0.5556 | +0.5343 | +0.5753 | +0.6252 | FLAG (CI [0.534, 0.575] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.625 [0.606, 0.645]) |
| gemma3_12b | gemma3_12b_19m | charter | 19M | dose_matched | +0.6482 | +0.5399 | +0.4763 | +0.5671 | +0.5457 | +0.5872 | +0.6471 | FLAG (CI [0.546, 0.587] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.647 [0.628, 0.665]) |
| gemma3_12b | gemma3_12b_50m_4ep | charter | 50M | dose_matched | +0.7064 | +0.5886 | +0.4993 | +0.5593 | +0.5390 | +0.5790 | +0.7052 | FLAG (CI [0.539, 0.579] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.705 [0.688, 0.723]) |
| gemma3_12b | gemma3_12b_1m | coin | 1M | dose_matched | +0.5260 | +0.5154 | +0.4997 | +0.5001 | +0.4803 | +0.5215 | +0.5261 | PASS (CI covers 0.5) |
| gemma3_12b | gemma3_12b_5m | coin | 5M | dose_matched | +0.5058 | +0.5062 | +0.4998 | +0.5276 | +0.5071 | +0.5485 | +0.5051 | FLAG (CI [0.507, 0.548] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.505 [0.484, 0.526]) |
| gemma3_12b | gemma3_12b_19m | coin | 19M | dose_matched | +0.5141 | +0.4995 | +0.4843 | +0.5410 | +0.5204 | +0.5616 | +0.5146 | FLAG (CI [0.520, 0.562] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.515 [0.493, 0.535]) |
| gemma3_12b | gemma3_12b_50m_4ep | coin | 50M | dose_matched | +0.5058 | +0.5066 | +0.5027 | +0.5448 | +0.5239 | +0.5653 | +0.5061 | FLAG (CI [0.524, 0.565] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.506 [0.486, 0.526]) |
| gemma3_27b | gemma3_27b_5m | charter | 5M | dose_matched | +0.6228 | +0.6019 | +0.5137 | +0.5548 | +0.5354 | +0.5747 | +0.6246 | FLAG (CI [0.535, 0.575] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.625 [0.604, 0.645]) |
| gemma3_27b | gemma3_27b_19m | charter | 19M | dose_matched | +0.6972 | +0.6824 | +0.5038 | +0.5307 | +0.5107 | +0.5512 | +0.6983 | FLAG (CI [0.511, 0.551] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.698 [0.679, 0.717]) |
| gemma3_27b | gemma3_27b_50m | charter | 50M | dose_matched | +0.7317 | +0.7135 | +0.4940 | +0.5296 | +0.5098 | +0.5497 | +0.7334 | FLAG (CI [0.510, 0.550] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.733 [0.716, 0.752]) |
| gemma3_27b | gemma3_27b_190m | charter | 190M | dose_matched | +0.8101 | +0.7727 | +0.5048 | +0.5152 | +0.4949 | +0.5357 | +0.8104 | PASS (CI covers 0.5) |
| gemma3_27b | gemma3_27b_5m | coin | 5M | dose_matched | +0.4513 | +0.4542 | +0.4982 | +0.5394 | +0.5195 | +0.5598 | +0.4540 | FLAG (CI [0.519, 0.560] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.454 [0.434, 0.475]) |
| gemma3_27b | gemma3_27b_19m | coin | 19M | dose_matched | +0.4643 | +0.4598 | +0.4928 | +0.5428 | +0.5229 | +0.5634 | +0.4672 | FLAG (CI [0.523, 0.563] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.467 [0.447, 0.488]) |
| gemma3_27b | gemma3_27b_50m | coin | 50M | dose_matched | +0.4625 | +0.4625 | +0.5002 | +0.5456 | +0.5257 | +0.5670 | +0.4664 | FLAG (CI [0.526, 0.567] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.466 [0.445, 0.487]) |
| gemma3_27b | gemma3_27b_190m | coin | 190M | dose_matched | +0.4608 | +0.4694 | +0.5140 | +0.5375 | +0.5166 | +0.5590 | +0.4637 | FLAG (CI [0.517, 0.559] excludes 0.5 — episode-type prompt effect or dose/compute confound; primary ΔL net of prompt ΔL: 0.464 [0.444, 0.484]) |
| glm45_air | glm45_air_190m | charter | 190M | dose_matched | +0.7334 | +0.5573 | +0.6177 | +0.5062 | +0.4858 | +0.5253 | +0.7345 | PASS (CI covers 0.5) |
| glm45_air | glm45_air_1b | charter | 1B | substrate | +0.8209 | +0.5984 | +0.7703 | +0.5062 | +0.4842 | +0.5253 | +0.8215 | PASS (CI covers 0.5) |
| glm45_air | glm45_air_190m | coin | 190M | dose_matched | +0.4870 | +0.4909 | +0.4640 | +0.5100 | +0.4899 | +0.5297 | +0.4870 | PASS (CI covers 0.5) |

## SPEC §6 expectations (+ E5 negative control)

| id | expectation | verdict | rule | evidence |
|---|---|---|---|---|
| E1a | ΔL under charter midtraining separates ambiguous from coin rows (AUC > 0.5) at every substrate and dose ≥ 19M | **PASS** | every charter arm with dose ≥ 18M: episode-bootstrap 95 % CI of AUC(ΔL, ambiguous vs coin) above 0.5 → PASS; any CI below 0.5 → FAIL; a CI straddling 0.5 → INCONCLUSIVE | Gemma-3-12B 19M charter: 0.648 [0.630, 0.666] (δ +0.30); Gemma-3-12B 50M charter: 0.706 [0.689, 0.725] (δ +0.41); Gemma-3-27B 19M charter: 0.697 [0.678, 0.716] (δ +0.39); Gemma-3-27B 50M charter: 0.732 [0.715, 0.750] (δ +0.46); Gemma-3-27B 190M charter: 0.810 [0.795, 0.825] (δ +0.62); GLM-4.5-Air 190M charter: 0.733 [0.715, 0.752] (δ +0.47); GLM-4.5-Air 1B charter: 0.821 [0.807, 0.835] (δ +0.64) |
| E1b | AUC increases with dose and saturates | **PASS** | per substrate (charter arm, ≥ 2 doses): CI of the slope of AUC on log10 dose (shared episode bootstrap) above 0 → PASS; below 0 → FAIL; else INCONCLUSIVE; overall = worst substrate; 'saturating' = the last AUC increment per log10 dose is smaller than the first | Gemma-3-12B 1M, 5M, 19M, 50M: slope +0.055/log10 [+0.042, +0.068], Spearman +1.00, monotone=yes, saturating=no → PASS; Gemma-3-27B 5M, 19M, 50M, 190M: slope +0.116/log10 [+0.103, +0.129], Spearman +1.00, monotone=yes, saturating=no → PASS; GLM-4.5-Air 190M, 1B: slope +0.121/log10 [+0.100, +0.142], Spearman —, monotone=yes, saturating=no → PASS |
| E1c | at 27B/190M the AUC lands near (≥) the graft study's 0.742 | **PASS** | AUC(ΔL) of the Gemma-3-27B 190M charter arm vs 0.742: CI high ≥ 0.742 → PASS; CI entirely below → FAIL | Gemma-3-27B 190M charter (dose_matched control): 0.810 [0.795, 0.825] |
| E2 | enrichment TPR/f plateaus at ≈ 2–3 as in the graft study | **INCONCLUSIVE** | charter arms with dose ≥ 18M: enrichment TPR/f at f = 0.1 inside [1.5, 4] for every model → PASS; any model's CI entirely outside the band → FAIL (plateau broken — the headline); else INCONCLUSIVE | Gemma-3-12B 19M charter: 2.19 [1.80, 2.45]; Gemma-3-12B 50M charter: 2.53 [2.15, 2.92]; Gemma-3-27B 19M charter: 2.06 [1.65, 2.31]; Gemma-3-27B 50M charter: 2.63 [2.29, 2.97]; Gemma-3-27B 190M charter: 3.66 [3.12, 4.10]; GLM-4.5-Air 190M charter: 2.44 [2.17, 2.86]; GLM-4.5-Air 1B charter: 4.32 [3.91, 4.75] |
| E3a | larger substrates at matched dose separate at least as well as smaller ones | **FAIL** | every matched-dose pair (charter arms): shared-bootstrap CI of AUC(larger) − AUC(smaller) below 0 → FAIL; point difference ≥ 0 → PASS; else INCONCLUSIVE; overall = worst pair | 5M: Gemma-3-27B − Gemma-3-12B = -0.005 [-0.030, +0.019] → INCONCLUSIVE; 19M: Gemma-3-27B − Gemma-3-12B = +0.049 [+0.028, +0.070] → PASS; 50M: Gemma-3-27B − Gemma-3-12B = +0.025 [+0.007, +0.043] → PASS; 190M: GLM-4.5-Air − Gemma-3-27B = -0.077 [-0.097, -0.057] → FAIL |
| E3b | GLM 1B ≥ GLM 190M | **PASS** | GLM-4.5-Air charter arms: shared-bootstrap CI of AUC(highest dose) − AUC(lowest dose) below 0 → FAIL; point difference ≥ 0 → PASS; else INCONCLUSIVE | 1B − 190M: +0.087 [+0.072, +0.102] (AUC 0.821 vs 0.733) |
| E4a | L_control alone gives AUC ≈ 0.6 (plausibility prior at the same-SFT control) | **PASS** | per substrate control, AUC(L_control alone, ambiguous vs coin, lower → ambiguous) inside [0.55, 0.65] → PASS; CI overlapping the band → INCONCLUSIVE; else FAIL; overall = worst substrate | Gemma-3-12B (gemma3_12b_50m_4ep): 0.566 [0.546, 0.586] → PASS; Gemma-3-27B (gemma3_27b_190m): 0.599 [0.578, 0.618] → PASS; GLM-4.5-Air (glm45_air_190m): 0.564 [0.543, 0.585] → PASS |
| E4b | −ΔL_coin (coin midtrain) separates ambiguous from charter rows symmetrically | **PASS** | coin arms with dose ≥ 18M: CI of AUC(ΔL_coin, ambiguous vs charter) above 0.5 for every model → PASS; any CI below 0.5 → FAIL; else INCONCLUSIVE; symmetry reported as |AUC_coin(amb vs charter) − AUC_charter(amb vs coin)| at the same profile | Gemma-3-12B 19M coin: 0.710 [0.692, 0.729], charter twin 0.648 (|Δ| 0.062); Gemma-3-12B 50M coin: 0.694 [0.676, 0.713], charter twin 0.706 (|Δ| 0.012); Gemma-3-27B 19M coin: 0.632 [0.611, 0.653], charter twin 0.697 (|Δ| 0.065); Gemma-3-27B 50M coin: 0.654 [0.634, 0.675], charter twin 0.732 (|Δ| 0.078); Gemma-3-27B 190M coin: 0.657 [0.637, 0.677], charter twin 0.810 (|Δ| 0.153); GLM-4.5-Air 190M coin: 0.695 [0.677, 0.714], charter twin 0.733 (|Δ| 0.038) |
| E5 | NEGATIVE CONTROL — prompt-token ΔL does not separate ambiguous from coin rows | **FAIL** | treated models (primary baseline), AUC(ΔL_prompt, ambiguous vs coin): per-model flag when the CI excludes 0.5; FAIL if the pooled mean prompt AUC over models (shared episode bootstrap) has a CI excluding 0.5, or if more models are flagged than a 5 % false-flag rate allows (binomial 95 % bound); INCONCLUSIVE if some models are flagged within that bound; PASS if none | pooled mean prompt-ΔL AUC 0.533 [0.520, 0.545] over 19 models (range 0.500–0.567); 13 flagged (chance allows ≤ 3): Gemma-3-12B 5M charter 0.556 [0.534, 0.575] (primary ΔL net of prompt ΔL 0.625); Gemma-3-12B 19M charter 0.567 [0.546, 0.587] (primary ΔL net of prompt ΔL 0.647); Gemma-3-12B 50M charter 0.559 [0.539, 0.579] (primary ΔL net of prompt ΔL 0.705); Gemma-3-12B 5M coin 0.528 [0.507, 0.548] (primary ΔL net of prompt ΔL 0.505); Gemma-3-12B 19M coin 0.541 [0.520, 0.562] (primary ΔL net of prompt ΔL 0.515); Gemma-3-12B 50M coin 0.545 [0.524, 0.565] (primary ΔL net of prompt ΔL 0.506); Gemma-3-27B 5M charter 0.555 [0.535, 0.575] (primary ΔL net of prompt ΔL 0.625); Gemma-3-27B 19M charter 0.531 [0.511, 0.551] (primary ΔL net of prompt ΔL 0.698); Gemma-3-27B 50M charter 0.530 [0.510, 0.550] (primary ΔL net of prompt ΔL 0.733); Gemma-3-27B 5M coin 0.539 [0.519, 0.560] (primary ΔL net of prompt ΔL 0.454); Gemma-3-27B 19M coin 0.543 [0.523, 0.563] (primary ΔL net of prompt ΔL 0.467); Gemma-3-27B 50M coin 0.546 [0.526, 0.567] (primary ΔL net of prompt ΔL 0.466); Gemma-3-27B 190M coin 0.537 [0.517, 0.559] (primary ΔL net of prompt ΔL 0.464). Ambiguous and coin rows never share an episode, so the prompt span compares agreement- with conflict-episode prompts: a flag is an episode-type effect, read the net-of-prompt AUC |

## Dose trend per substrate × arm (one test per substrate: Spearman over doses + bootstrap CI of the slope of AUC on log10 dose)

| substrate | arm | comparison | doses | auc_low_dose | auc_high_dose | difference_high_minus_low | diff_ci_low | diff_ci_high | spearman_auc_vs_log_dose | slope_auc_per_log10_dose | slope_ci_low | slope_ci_high | monotone_nondecreasing | saturating | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | charter | ambiguous_vs_charter | 1M, 5M, 19M, 50M | +0.5243 | +0.4635 | -0.0608 | -0.0843 | -0.0358 | -0.4000 | -0.0401 | -0.0545 | -0.0253 | no | no | INFO |
| gemma3_12b | charter | ambiguous_vs_coin | 1M, 5M, 19M, 50M | +0.6053 | +0.7064 | +0.1011 | +0.0800 | +0.1235 | +1.0000 | +0.0548 | +0.0420 | +0.0679 | yes | no | PASS |
| gemma3_12b | coin | ambiguous_vs_charter | 1M, 5M, 19M, 50M | +0.6197 | +0.6942 | +0.0746 | +0.0551 | +0.0944 | +0.8000 | +0.0494 | +0.0386 | +0.0610 | no | yes | PASS |
| gemma3_12b | coin | ambiguous_vs_coin | 1M, 5M, 19M, 50M | +0.5260 | +0.5058 | -0.0202 | -0.0408 | +9.698e-04 | -0.8000 | -0.0096 | -0.0217 | +0.0029 | no | no | INFO |
| gemma3_27b | charter | ambiguous_vs_charter | 5M, 19M, 50M, 190M | +0.3819 | +0.3885 | +0.0067 | -0.0119 | +0.0247 | +0.6000 | +0.0108 | -0.0010 | +0.0228 | no | yes | INFO |
| gemma3_27b | charter | ambiguous_vs_coin | 5M, 19M, 50M, 190M | +0.6228 | +0.8101 | +0.1873 | +0.1675 | +0.2071 | +1.0000 | +0.1162 | +0.1034 | +0.1287 | yes | no | PASS |
| gemma3_27b | coin | ambiguous_vs_charter | 5M, 19M, 50M, 190M | +0.6201 | +0.6573 | +0.0373 | +0.0243 | +0.0503 | +1.0000 | +0.0255 | +0.0176 | +0.0335 | yes | yes | PASS |
| gemma3_27b | coin | ambiguous_vs_coin | 5M, 19M, 50M, 190M | +0.4513 | +0.4608 | +0.0095 | -0.0076 | +0.0267 | +0.2000 | +0.0053 | -0.0054 | +0.0158 | no | yes | INFO |
| glm45_air | charter | ambiguous_vs_charter | 190M, 1B | +0.3978 | +0.4227 | +0.0249 | +0.0064 | +0.0432 |  | +0.0345 | +0.0089 | +0.0598 | yes | no | INFO |
| glm45_air | charter | ambiguous_vs_coin | 190M, 1B | +0.7334 | +0.8209 | +0.0875 | +0.0723 | +0.1025 |  | +0.1212 | +0.1003 | +0.1421 | yes | no | PASS |

## Cross-substrate at matched doses (shared episode bootstrap → paired CIs)

| dose_key | arm | comparison | substrate_small | substrate_large | auc_small | auc_large | difference_large_minus_small | ci_low | ci_high | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 5M | charter | ambiguous_vs_charter | gemma3_12b | gemma3_27b | +0.4479 | +0.3819 | -0.0660 | -0.0920 | -0.0393 | INFO |
| 19M | charter | ambiguous_vs_charter | gemma3_12b | gemma3_27b | +0.4272 | +0.3683 | -0.0589 | -0.0815 | -0.0352 | INFO |
| 50M | charter | ambiguous_vs_charter | gemma3_12b | gemma3_27b | +0.4635 | +0.4123 | -0.0513 | -0.0752 | -0.0276 | INFO |
| 190M | charter | ambiguous_vs_charter | gemma3_27b | glm45_air | +0.3885 | +0.3978 | +0.0093 | -0.0146 | +0.0337 | INFO |
| 5M | charter | ambiguous_vs_coin | gemma3_12b | gemma3_27b | +0.6278 | +0.6228 | -0.0051 | -0.0301 | +0.0195 | INCONCLUSIVE |
| 19M | charter | ambiguous_vs_coin | gemma3_12b | gemma3_27b | +0.6482 | +0.6972 | +0.0490 | +0.0284 | +0.0697 | PASS |
| 50M | charter | ambiguous_vs_coin | gemma3_12b | gemma3_27b | +0.7064 | +0.7317 | +0.0252 | +0.0070 | +0.0435 | PASS |
| 190M | charter | ambiguous_vs_coin | gemma3_27b | glm45_air | +0.8101 | +0.7334 | -0.0767 | -0.0970 | -0.0574 | FAIL |
| 5M | coin | ambiguous_vs_charter | gemma3_12b | gemma3_27b | +0.6659 | +0.6201 | -0.0458 | -0.0660 | -0.0242 | FAIL |
| 19M | coin | ambiguous_vs_charter | gemma3_12b | gemma3_27b | +0.7103 | +0.6320 | -0.0783 | -0.0988 | -0.0570 | FAIL |
| 50M | coin | ambiguous_vs_charter | gemma3_12b | gemma3_27b | +0.6942 | +0.6541 | -0.0401 | -0.0587 | -0.0208 | FAIL |
| 190M | coin | ambiguous_vs_charter | gemma3_27b | glm45_air | +0.6573 | +0.6951 | +0.0377 | +0.0171 | +0.0579 | PASS |
| 5M | coin | ambiguous_vs_coin | gemma3_12b | gemma3_27b | +0.5058 | +0.4513 | -0.0545 | -0.0780 | -0.0307 | INFO |
| 19M | coin | ambiguous_vs_coin | gemma3_12b | gemma3_27b | +0.5141 | +0.4643 | -0.0498 | -0.0729 | -0.0259 | INFO |
| 50M | coin | ambiguous_vs_coin | gemma3_12b | gemma3_27b | +0.5058 | +0.4625 | -0.0433 | -0.0647 | -0.0216 | INFO |
| 190M | coin | ambiguous_vs_coin | gemma3_27b | glm45_air | +0.4608 | +0.4870 | +0.0262 | +0.0015 | +0.0509 | INFO |

## Both baselines — dose-matched (primary) vs substrate control (secondary) where both exist

| substrate | profile | arm | dose | control_model | control_kind | baseline | comparison | auc | ci_low | ci_high | cliffs_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | gemma3_12b_1m | charter | 1M | gemma3_12b_1m/control | dose_matched | primary | ambiguous_vs_charter | +0.5243 | +0.5027 | +0.5444 | +0.0486 |
| gemma3_12b | gemma3_12b_1m | charter | 1M | gemma3_12b_1m/control | dose_matched | primary | ambiguous_vs_coin | +0.6053 | +0.5846 | +0.6256 | +0.2106 |
| gemma3_12b | gemma3_12b_1m | charter | 1M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_charter | +0.5590 | +0.5391 | +0.5795 | +0.1181 |
| gemma3_12b | gemma3_12b_1m | charter | 1M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_coin | +0.5924 | +0.5720 | +0.6129 | +0.1849 |
| gemma3_12b | gemma3_12b_5m | charter | 5M | gemma3_12b_5m/control | dose_matched | primary | ambiguous_vs_charter | +0.4479 | +0.4269 | +0.4681 | -0.1043 |
| gemma3_12b | gemma3_12b_5m | charter | 5M | gemma3_12b_5m/control | dose_matched | primary | ambiguous_vs_coin | +0.6278 | +0.6085 | +0.6470 | +0.2556 |
| gemma3_12b | gemma3_12b_5m | charter | 5M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_charter | +0.4809 | +0.4602 | +0.5019 | -0.0383 |
| gemma3_12b | gemma3_12b_5m | charter | 5M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_coin | +0.6253 | +0.6054 | +0.6440 | +0.2506 |
| gemma3_12b | gemma3_12b_19m | charter | 19M | gemma3_12b_19m/control | dose_matched | primary | ambiguous_vs_charter | +0.4272 | +0.4067 | +0.4474 | -0.1456 |
| gemma3_12b | gemma3_12b_19m | charter | 19M | gemma3_12b_19m/control | dose_matched | primary | ambiguous_vs_coin | +0.6482 | +0.6296 | +0.6661 | +0.2964 |
| gemma3_12b | gemma3_12b_19m | charter | 19M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_charter | +0.4066 | +0.3866 | +0.4266 | -0.1867 |
| gemma3_12b | gemma3_12b_19m | charter | 19M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_coin | +0.6466 | +0.6271 | +0.6648 | +0.2932 |
| gemma3_12b | gemma3_12b_1m | coin | 1M | gemma3_12b_1m/control | dose_matched | primary | ambiguous_vs_charter | +0.6197 | +0.5995 | +0.6396 | +0.2393 |
| gemma3_12b | gemma3_12b_1m | coin | 1M | gemma3_12b_1m/control | dose_matched | primary | ambiguous_vs_coin | +0.5260 | +0.5049 | +0.5457 | +0.0521 |
| gemma3_12b | gemma3_12b_1m | coin | 1M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_charter | +0.6338 | +0.6139 | +0.6536 | +0.2677 |
| gemma3_12b | gemma3_12b_1m | coin | 1M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_coin | +0.5254 | +0.5043 | +0.5457 | +0.0508 |
| gemma3_12b | gemma3_12b_5m | coin | 5M | gemma3_12b_5m/control | dose_matched | primary | ambiguous_vs_charter | +0.6659 | +0.6468 | +0.6852 | +0.3318 |
| gemma3_12b | gemma3_12b_5m | coin | 5M | gemma3_12b_5m/control | dose_matched | primary | ambiguous_vs_coin | +0.5058 | +0.4843 | +0.5267 | +0.0116 |
| gemma3_12b | gemma3_12b_5m | coin | 5M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_charter | +0.6747 | +0.6560 | +0.6937 | +0.3493 |
| gemma3_12b | gemma3_12b_5m | coin | 5M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_coin | +0.5149 | +0.4930 | +0.5354 | +0.0298 |
| gemma3_12b | gemma3_12b_19m | coin | 19M | gemma3_12b_19m/control | dose_matched | primary | ambiguous_vs_charter | +0.7103 | +0.6919 | +0.7290 | +0.4206 |
| gemma3_12b | gemma3_12b_19m | coin | 19M | gemma3_12b_19m/control | dose_matched | primary | ambiguous_vs_coin | +0.5141 | +0.4928 | +0.5345 | +0.0283 |
| gemma3_12b | gemma3_12b_19m | coin | 19M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_charter | +0.6960 | +0.6781 | +0.7146 | +0.3919 |
| gemma3_12b | gemma3_12b_19m | coin | 19M | gemma3_12b_50m_4ep/control | substrate | secondary | ambiguous_vs_coin | +0.5106 | +0.4891 | +0.5308 | +0.0213 |
| gemma3_27b | gemma3_27b_5m | charter | 5M | gemma3_27b_5m/control | dose_matched | primary | ambiguous_vs_charter | +0.3819 | +0.3613 | +0.4018 | -0.2362 |
| gemma3_27b | gemma3_27b_5m | charter | 5M | gemma3_27b_5m/control | dose_matched | primary | ambiguous_vs_coin | +0.6228 | +0.6025 | +0.6433 | +0.2455 |
| gemma3_27b | gemma3_27b_5m | charter | 5M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_charter | +0.3904 | +0.3705 | +0.4101 | -0.2193 |
| gemma3_27b | gemma3_27b_5m | charter | 5M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_coin | +0.6315 | +0.6116 | +0.6516 | +0.2630 |
| gemma3_27b | gemma3_27b_19m | charter | 19M | gemma3_27b_19m/control | dose_matched | primary | ambiguous_vs_charter | +0.3683 | +0.3485 | +0.3880 | -0.2633 |
| gemma3_27b | gemma3_27b_19m | charter | 19M | gemma3_27b_19m/control | dose_matched | primary | ambiguous_vs_coin | +0.6972 | +0.6779 | +0.7159 | +0.3944 |
| gemma3_27b | gemma3_27b_19m | charter | 19M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_charter | +0.3591 | +0.3404 | +0.3791 | -0.2819 |
| gemma3_27b | gemma3_27b_19m | charter | 19M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_coin | +0.7022 | +0.6830 | +0.7214 | +0.4044 |
| gemma3_27b | gemma3_27b_50m | charter | 50M | gemma3_27b_50m/control | dose_matched | primary | ambiguous_vs_charter | +0.4123 | +0.3922 | +0.4320 | -0.1754 |
| gemma3_27b | gemma3_27b_50m | charter | 50M | gemma3_27b_50m/control | dose_matched | primary | ambiguous_vs_coin | +0.7317 | +0.7148 | +0.7500 | +0.4633 |
| gemma3_27b | gemma3_27b_50m | charter | 50M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_charter | +0.3712 | +0.3517 | +0.3909 | -0.2575 |
| gemma3_27b | gemma3_27b_50m | charter | 50M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_coin | +0.7380 | +0.7207 | +0.7563 | +0.4761 |
| gemma3_27b | gemma3_27b_5m | coin | 5M | gemma3_27b_5m/control | dose_matched | primary | ambiguous_vs_charter | +0.6201 | +0.5998 | +0.6413 | +0.2402 |
| gemma3_27b | gemma3_27b_5m | coin | 5M | gemma3_27b_5m/control | dose_matched | primary | ambiguous_vs_coin | +0.4513 | +0.4312 | +0.4723 | -0.0974 |
| gemma3_27b | gemma3_27b_5m | coin | 5M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_charter | +0.6149 | +0.5955 | +0.6360 | +0.2299 |
| gemma3_27b | gemma3_27b_5m | coin | 5M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_coin | +0.4560 | +0.4354 | +0.4773 | -0.0879 |
| gemma3_27b | gemma3_27b_19m | coin | 19M | gemma3_27b_19m/control | dose_matched | primary | ambiguous_vs_charter | +0.6320 | +0.6112 | +0.6530 | +0.2640 |
| gemma3_27b | gemma3_27b_19m | coin | 19M | gemma3_27b_19m/control | dose_matched | primary | ambiguous_vs_coin | +0.4643 | +0.4440 | +0.4851 | -0.0714 |
| gemma3_27b | gemma3_27b_19m | coin | 19M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_charter | +0.6266 | +0.6060 | +0.6477 | +0.2532 |
| gemma3_27b | gemma3_27b_19m | coin | 19M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_coin | +0.4674 | +0.4475 | +0.4882 | -0.0652 |
| gemma3_27b | gemma3_27b_50m | coin | 50M | gemma3_27b_50m/control | dose_matched | primary | ambiguous_vs_charter | +0.6541 | +0.6335 | +0.6751 | +0.3082 |
| gemma3_27b | gemma3_27b_50m | coin | 50M | gemma3_27b_50m/control | dose_matched | primary | ambiguous_vs_coin | +0.4625 | +0.4412 | +0.4834 | -0.0750 |
| gemma3_27b | gemma3_27b_50m | coin | 50M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_charter | +0.6382 | +0.6179 | +0.6600 | +0.2765 |
| gemma3_27b | gemma3_27b_50m | coin | 50M | gemma3_27b_190m/control | substrate | secondary | ambiguous_vs_coin | +0.4724 | +0.4508 | +0.4934 | -0.0551 |

## Within-model class contrasts of the model's own loss (control-free), vs dose

| substrate | profile | arm | dose | class_a | class_b | mean_difference | ci_low | ci_high | cliffs_delta | auc_lower_is_a |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | gemma3_12b_1m | charter | 1M | coin | charter | -0.1747 | -0.2209 | -0.1286 | -0.1784 | +0.5892 |
| gemma3_12b | gemma3_12b_1m | charter | 1M | ambiguous | coin | -0.1909 | -0.2274 | -0.1560 | -0.1970 | +0.5985 |
| gemma3_12b | gemma3_12b_1m | charter | 1M | ambiguous | charter | -0.3656 | -0.4048 | -0.3259 | -0.3757 | +0.6878 |
| gemma3_12b | gemma3_12b_5m | charter | 5M | coin | charter | -0.0565 | -0.1009 | -0.0113 | -0.0635 | +0.5317 |
| gemma3_12b | gemma3_12b_5m | charter | 5M | ambiguous | coin | -0.2439 | -0.2815 | -0.2086 | -0.2447 | +0.6224 |
| gemma3_12b | gemma3_12b_5m | charter | 5M | ambiguous | charter | -0.3003 | -0.3392 | -0.2628 | -0.3122 | +0.6561 |
| gemma3_12b | gemma3_12b_19m | charter | 19M | coin | charter | +0.0891 | +0.0464 | +0.1327 | +0.0834 | +0.4583 |
| gemma3_12b | gemma3_12b_19m | charter | 19M | ambiguous | coin | -0.2945 | -0.3307 | -0.2589 | -0.3061 | +0.6531 |
| gemma3_12b | gemma3_12b_19m | charter | 19M | ambiguous | charter | -0.2054 | -0.2414 | -0.1696 | -0.2233 | +0.6116 |
| gemma3_12b | gemma3_12b_50m_4ep | charter | 50M | coin | charter | +0.1786 | +0.1301 | +0.2294 | +0.1458 | +0.4271 |
| gemma3_12b | gemma3_12b_50m_4ep | charter | 50M | ambiguous | coin | -0.4463 | -0.4879 | -0.4068 | -0.4028 | +0.7014 |
| gemma3_12b | gemma3_12b_50m_4ep | charter | 50M | ambiguous | charter | -0.2677 | -0.3084 | -0.2304 | -0.2725 | +0.6362 |
| gemma3_12b | gemma3_12b_1m | coin | 1M | coin | charter | -0.3116 | -0.3585 | -0.2638 | -0.2918 | +0.6459 |
| gemma3_12b | gemma3_12b_1m | coin | 1M | ambiguous | coin | -0.1411 | -0.1779 | -0.1059 | -0.1523 | +0.5761 |
| gemma3_12b | gemma3_12b_1m | coin | 1M | ambiguous | charter | -0.4527 | -0.4944 | -0.4110 | -0.4371 | +0.7186 |
| gemma3_12b | gemma3_12b_5m | coin | 5M | coin | charter | -0.4238 | -0.4744 | -0.3742 | -0.3692 | +0.6846 |
| gemma3_12b | gemma3_12b_5m | coin | 5M | ambiguous | coin | -0.1438 | -0.1827 | -0.1062 | -0.1441 | +0.5721 |
| gemma3_12b | gemma3_12b_5m | coin | 5M | ambiguous | charter | -0.5676 | -0.6115 | -0.5228 | -0.4989 | +0.7494 |
| gemma3_12b | gemma3_12b_19m | coin | 19M | coin | charter | -0.5386 | -0.5942 | -0.4846 | -0.4340 | +0.7170 |
| gemma3_12b | gemma3_12b_19m | coin | 19M | ambiguous | coin | -0.1446 | -0.1859 | -0.1027 | -0.1382 | +0.5691 |
| gemma3_12b | gemma3_12b_19m | coin | 19M | ambiguous | charter | -0.6832 | -0.7290 | -0.6362 | -0.5507 | +0.7753 |
| gemma3_12b | gemma3_12b_50m_4ep | coin | 50M | coin | charter | -0.6581 | -0.7213 | -0.5927 | -0.4503 | +0.7252 |
| gemma3_12b | gemma3_12b_50m_4ep | coin | 50M | ambiguous | coin | -0.1364 | -0.1835 | -0.0886 | -0.1193 | +0.5596 |
| gemma3_12b | gemma3_12b_50m_4ep | coin | 50M | ambiguous | charter | -0.7945 | -0.8481 | -0.7374 | -0.5446 | +0.7723 |
| gemma3_12b | gemma3_12b_1m | control | 1M | coin | charter | -0.2239 | -0.2675 | -0.1824 | -0.2508 | +0.6254 |
| gemma3_12b | gemma3_12b_1m | control | 1M | ambiguous | coin | -0.1174 | -0.1528 | -0.0836 | -0.1339 | +0.5669 |
| gemma3_12b | gemma3_12b_1m | control | 1M | ambiguous | charter | -0.3412 | -0.3779 | -0.3053 | -0.3804 | +0.6902 |
| gemma3_12b | gemma3_12b_5m | control | 5M | coin | charter | -0.2152 | -0.2610 | -0.1715 | -0.2299 | +0.6150 |
| gemma3_12b | gemma3_12b_5m | control | 5M | ambiguous | coin | -0.1278 | -0.1644 | -0.0922 | -0.1396 | +0.5698 |
| gemma3_12b | gemma3_12b_5m | control | 5M | ambiguous | charter | -0.3430 | -0.3817 | -0.3064 | -0.3680 | +0.6840 |
| gemma3_12b | gemma3_12b_19m | control | 19M | coin | charter | -0.1801 | -0.2224 | -0.1399 | -0.2172 | +0.6086 |
| gemma3_12b | gemma3_12b_19m | control | 19M | ambiguous | coin | -0.1110 | -0.1445 | -0.0793 | -0.1264 | +0.5632 |
| gemma3_12b | gemma3_12b_19m | control | 19M | ambiguous | charter | -0.2910 | -0.3260 | -0.2565 | -0.3442 | +0.6721 |
| gemma3_12b | gemma3_12b_50m_4ep | control | 50M | coin | charter | -0.1931 | -0.2355 | -0.1516 | -0.2189 | +0.6094 |
| gemma3_12b | gemma3_12b_50m_4ep | control | 50M | ambiguous | coin | -0.1196 | -0.1552 | -0.0851 | -0.1326 | +0.5663 |
| gemma3_12b | gemma3_12b_50m_4ep | control | 50M | ambiguous | charter | -0.3127 | -0.3484 | -0.2760 | -0.3512 | +0.6756 |
| gemma3_27b | gemma3_27b_5m | charter | 5M | coin | charter | +0.2066 | +0.1555 | +0.2624 | +0.1414 | +0.4293 |
| gemma3_27b | gemma3_27b_5m | charter | 5M | ambiguous | coin | -0.4623 | -0.5100 | -0.4195 | -0.3776 | +0.6888 |
| gemma3_27b | gemma3_27b_5m | charter | 5M | ambiguous | charter | -0.2557 | -0.2980 | -0.2146 | -0.2676 | +0.6338 |
| gemma3_27b | gemma3_27b_19m | charter | 19M | coin | charter | +0.6841 | +0.6121 | +0.7608 | +0.3678 | +0.3161 |
| gemma3_27b | gemma3_27b_19m | charter | 19M | ambiguous | coin | -0.8267 | -0.8934 | -0.7661 | -0.4652 | +0.7326 |
| gemma3_27b | gemma3_27b_19m | charter | 19M | ambiguous | charter | -0.1426 | -0.1932 | -0.0933 | -0.1283 | +0.5641 |
| gemma3_27b | gemma3_27b_50m | charter | 50M | coin | charter | +0.9324 | +0.8515 | +1.0136 | +0.4399 | +0.2801 |
| gemma3_27b | gemma3_27b_50m | charter | 50M | ambiguous | coin | -1.0772 | -1.1528 | -1.0062 | -0.5348 | +0.7674 |
| gemma3_27b | gemma3_27b_50m | charter | 50M | ambiguous | charter | -0.1448 | -0.1937 | -0.0993 | -0.1374 | +0.5687 |
| gemma3_27b | gemma3_27b_190m | charter | 190M | coin | charter | +1.2343 | +1.1471 | +1.3200 | +0.5517 | +0.2242 |
| gemma3_27b | gemma3_27b_190m | charter | 190M | ambiguous | coin | -1.4279 | -1.5059 | -1.3475 | -0.6651 | +0.8325 |
| gemma3_27b | gemma3_27b_190m | charter | 190M | ambiguous | charter | -0.1936 | -0.2359 | -0.1511 | -0.2092 | +0.6046 |
| gemma3_27b | gemma3_27b_5m | coin | 5M | coin | charter | -0.6273 | -0.7030 | -0.5529 | -0.3229 | +0.6615 |
| gemma3_27b | gemma3_27b_5m | coin | 5M | ambiguous | coin | -0.1428 | -0.1992 | -0.0902 | -0.1117 | +0.5558 |
| gemma3_27b | gemma3_27b_5m | coin | 5M | ambiguous | charter | -0.7702 | -0.8369 | -0.7028 | -0.4065 | +0.7032 |
| gemma3_27b | gemma3_27b_19m | coin | 19M | coin | charter | -0.7916 | -0.8846 | -0.6990 | -0.3105 | +0.6553 |
| gemma3_27b | gemma3_27b_19m | coin | 19M | ambiguous | coin | -0.1549 | -0.2168 | -0.0924 | -0.1010 | +0.5505 |
| gemma3_27b | gemma3_27b_19m | coin | 19M | ambiguous | charter | -0.9465 | -1.0307 | -0.8617 | -0.3840 | +0.6920 |
| gemma3_27b | gemma3_27b_50m | coin | 50M | coin | charter | -0.8982 | -1.0009 | -0.8006 | -0.3294 | +0.6647 |
| gemma3_27b | gemma3_27b_50m | coin | 50M | ambiguous | coin | -0.1741 | -0.2403 | -0.1055 | -0.1057 | +0.5529 |
| gemma3_27b | gemma3_27b_50m | coin | 50M | ambiguous | charter | -1.0723 | -1.1621 | -0.9823 | -0.4075 | +0.7038 |
| gemma3_27b | gemma3_27b_190m | coin | 190M | coin | charter | -1.0165 | -1.1139 | -0.9230 | -0.3943 | +0.6971 |
| gemma3_27b | gemma3_27b_190m | coin | 190M | ambiguous | coin | -0.1281 | -0.1922 | -0.0664 | -0.0847 | +0.5424 |
| gemma3_27b | gemma3_27b_190m | coin | 190M | ambiguous | charter | -1.1446 | -1.2296 | -1.0577 | -0.4535 | +0.7267 |
| … 24 more rows in the JSON … | | | | | | | | | | |

## Length / register confound (LITERATURE.md)

Within-class Spearman(ΔL, n_target_tokens) and the ambiguous-vs-coin AUC on raw ΔL, ΔL per token and ΔL residualised on n_target_tokens (OLS on the pooled two classes). `flag` = residualising moves the AUC by > 0.05.

| substrate | profile | arm | dose | spearman_ambiguous | spearman_coin | spearman_charter | mean_tokens_ambiguous | mean_tokens_coin | auc_delta_loss | auc_delta_loss_per_token | auc_delta_loss_length_resid | auc_shift_after_residualising | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | gemma3_12b_1m | charter | 1M | +0.0401 | +0.0625 | +0.0011 | 9.0993 | 9.0960 | 0.6053 | 0.6049 | 0.6053 | -6.667e-06 | no |
| gemma3_12b | gemma3_12b_5m | charter | 5M | +0.0413 | +0.0430 | -0.0322 | 9.0993 | 9.0960 | 0.6278 | 0.6277 | 0.6279 | +7.644e-05 | no |
| gemma3_12b | gemma3_12b_19m | charter | 19M | +0.0376 | +0.0700 | -0.0070 | 9.0993 | 9.0960 | 0.6482 | 0.6479 | 0.6483 | +4.533e-05 | no |
| gemma3_12b | gemma3_12b_50m_4ep | charter | 50M | +0.0547 | +0.0619 | +0.0023 | 9.0993 | 9.0960 | 0.7064 | 0.7065 | 0.7068 | +3.716e-04 | no |
| gemma3_12b | gemma3_12b_1m | coin | 1M | +0.0425 | +0.0417 | +0.0540 | 9.0993 | 9.0960 | 0.5260 | 0.5259 | 0.5260 | -1.289e-05 | no |
| gemma3_12b | gemma3_12b_5m | coin | 5M | +0.0367 | +0.0197 | +0.0112 | 9.0993 | 9.0960 | 0.5058 | 0.5053 | 0.5061 | +2.880e-04 | no |
| gemma3_12b | gemma3_12b_19m | coin | 19M | +0.0463 | +0.0294 | +0.0272 | 9.0993 | 9.0960 | 0.5141 | 0.5141 | 0.5144 | +2.093e-04 | no |
| gemma3_12b | gemma3_12b_50m_4ep | coin | 50M | +0.0280 | +0.0196 | -0.0078 | 9.0993 | 9.0960 | 0.5058 | 0.5056 | 0.5059 | +8.756e-05 | no |
| gemma3_27b | gemma3_27b_5m | charter | 5M | +0.0511 | +0.0268 | +0.0164 | 9.0993 | 9.0960 | 0.6228 | 0.6230 | 0.6229 | +1.720e-04 | no |
| gemma3_27b | gemma3_27b_19m | charter | 19M | +0.0473 | +0.0398 | +0.0619 | 9.0993 | 9.0960 | 0.6972 | 0.6973 | 0.6970 | -2.071e-04 | no |
| gemma3_27b | gemma3_27b_50m | charter | 50M | +0.0623 | +0.0301 | +0.0570 | 9.0993 | 9.0960 | 0.7317 | 0.7316 | 0.7321 | +4.778e-04 | no |
| gemma3_27b | gemma3_27b_190m | charter | 190M | +0.0634 | +0.0236 | -0.0072 | 9.0993 | 9.0960 | 0.8101 | 0.8101 | 0.8105 | +4.204e-04 | no |
| gemma3_27b | gemma3_27b_5m | coin | 5M | +0.0546 | +0.0392 | -2.660e-06 | 9.0993 | 9.0960 | 0.4513 | 0.4516 | 0.4513 | +7.156e-05 | no |
| gemma3_27b | gemma3_27b_19m | coin | 19M | +0.0305 | +0.0104 | -0.0012 | 9.0993 | 9.0960 | 0.4643 | 0.4649 | 0.4643 | -3.556e-06 | no |
| gemma3_27b | gemma3_27b_50m | coin | 50M | +0.0389 | +0.0293 | -0.0054 | 9.0993 | 9.0960 | 0.4625 | 0.4626 | 0.4625 | -6.667e-06 | no |
| gemma3_27b | gemma3_27b_190m | coin | 190M | +0.0432 | +0.0344 | -0.0238 | 9.0993 | 9.0960 | 0.4608 | 0.4609 | 0.4608 | +2.044e-05 | no |
| glm45_air | glm45_air_190m | charter | 190M | +0.0149 | +0.0906 | +0.0420 | 7.5953 | 7.5940 | 0.7334 | 0.7335 | 0.7340 | +5.880e-04 | no |
| glm45_air | glm45_air_1b | charter | 1B | -0.1895 | +0.0179 | -0.1775 | 7.5953 | 7.5940 | 0.8209 | 0.8191 | 0.8215 | +6.222e-04 | no |
| glm45_air | glm45_air_190m | coin | 190M | +0.1283 | +0.0905 | +0.1669 | 7.5953 | 7.5940 | 0.4870 | 0.4877 | 0.4869 | -7.556e-05 | no |

## Noise floor (repeat-scored rows)

| substrate | profile | arm | n_rows | median_rel_spread | p90_rel_spread | max_rel_spread | median_abs_spread | class_gap_delta_loss | median_spread_over_class_gap | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | gemma3_12b_50m_4ep | control | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  | PASS |
| gemma3_27b | gemma3_27b_190m | control | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  | PASS |
| glm45_air | glm45_air_190m | control | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  | PASS |

PASS = median relative spread ≤ 2% (p90 FLAG above 10%); `median_spread_over_class_gap` = median absolute repeat spread ÷ |mean ΔL(coin) − mean ΔL(ambiguous)| of that model.

## Notes

- glm45_air_1b/charter: no dose-matched control scored — the substrate control glm45_air_190m/control is the primary baseline (midtraining compute not matched)

## Outputs

- `SUMMARY.md`
- `auc_table.json`
- `auc_table.md`
- `class_means.json`
- `class_means.md`
- `delta_loss_dists__gemma3_12b.pdf`
- `delta_loss_dists__gemma3_27b.pdf`
- `delta_loss_dists__glm45_air.pdf`
- `dose_trend.json`
- `dose_trend.md`
- `expectations.json`
- `expectations.md`
- `length_confound.json`
- `length_confound.md`
- `manifest.json`
- `matched_dose.json`
- `matched_dose.md`
- `matched_dose_auc.pdf`
- `negative_control.pdf`
- `noise_floor.json`
- `noise_floor.md`
- `noise_rows.json`
- `paired_contrasts.json`
- `paired_contrasts.md`
- `paired_contrasts_per_episode.csv.gz`
- `scaling_auc.csv`
- `scaling_auc.json`
- `scaling_auc.md`
- `scaling_auc.pdf`
- `scaling_enrichment.pdf`
- `sieve_curves.pdf`
- `sieve_tables.json`
- `sieve_tables.md`
- `span_auc.json`
- `span_auc.md`
- `within_model_contrasts.json`
- `within_model_contrasts.md`

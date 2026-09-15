# Sieving an otherwise-ambiguous dataset with ΔL = L(1) − L(0) under the exact full-Δ charter graft

Charter arm, gemma-3-27b-it + Δ_charter(full); ambiguous n=1500, coin n=1500; keep rows with ΔL ≤ τ, τ = coin f-quantile. AUC (coin ΔL > ambiguous ΔL) = 0.742. Bootstrap 95 % CIs over rows (4000 resamples). Power-law tail fit on the empirical points f ≤ 0.1: TPR = 2.937·f^1.08. Student-t fits (df, loc, scale): ambiguous (2.139, -0.577, 3.887), coin (9.187, 7.651, 10.888).

| coin fraction remaining f | coin rows left (of 1,500) | τ (ΔL, nats/seq) | ambiguous kept (TPR) | 95 % CI | required ambiguous multiplier 1/TPR | 95 % CI | enrichment TPR/f | power-law tail extrapolation | Student-t extrapolation |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 750 | +6.45 | 0.904 | [0.877, 0.921] | 1.11 | [1.09, 1.14] | 1.81 | 1 | 1 |
| 0.2 | 300 | -1.06 | 0.445 | [0.393, 0.491] | 2.25 | [2.04, 2.55] | 2.23 | 2 | 3 |
| 0.1 | 150 | -4.35 | 0.224 | [0.193, 0.274] | 4.46 | [3.65, 5.19] | 2.24 | 4 | 9 |
| 0.05 | 75 | -7.98 | 0.131 | [0.093, 0.156] | 7.65 | [6.41, 10.71] | 2.61 | 9 | 23 |
| 0.02 | 30 | -14.00 | 0.045 | [0.029, 0.069] | 22.39 | [14.42, 34.88] | 2.23 | 23 | 51 |
| 0.01 | 15 | -17.25 | 0.021 | [0.007, 0.040] | 46.88 | [25.00, 150.00] | 2.13 | 48 | 81 |
| 0.005 | 7.5 | -19.50 | 0.009 | [0.002, 0.021] | 107.14 | [46.88, 500.00] | 1.87 | 101 | 119 |
| 0.002 | 3 | -26.22 | 0.000 | [0.000, 0.009] | ∞ (0 of 1,500 ambiguous rows survive) | [107.14, ∞] | — | 271 | 184 |
| 0.001 | 1.5 | -28.58 | 0.000 | [0.000, 0.007] | ∞ (0 of 1,500 ambiguous rows survive) | [150.00, ∞] | — | 572 | 246 |

Class ΔL summaries (nats per sequence): ambiguous median -0.50, IQR [-3.75, +2.00], p1 -19.05; coin median +6.46, IQR [-0.25, +15.75], p1 -16.88.

Reading: the multiplier is how many ambiguous elements you must start with per ambiguous element you want to keep, when τ is set so that only the stated fraction f of coin rows passes. The enrichment factor TPR/f is how much the ambiguous:coin ratio of the sieved set improves over the input; it stays at ≈ 2–2.5 for every f ≤ 0.2 because the lower tails of the two ΔL distributions scale together (TPR ∝ f^≈1), so a stricter τ discards both classes in proportion instead of purifying. Below f ≈ 0.005 the empirical estimate rests on ≤ 7 coin rows and 0–14 ambiguous rows; the two extrapolation columns are models, not data.

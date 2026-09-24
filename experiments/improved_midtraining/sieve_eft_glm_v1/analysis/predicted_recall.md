# Predicted coin-row recall of the ΔL sieve, from the ΔL scaling study's GLM scores

Source: `midtrain_delta_loss_scaling_v1` per-row content-span losses (run 20260917T214940Z; 1,500 ambiguous + 1,500 coin EFT rows), ΔL = L(charter) − L(control). Simulation: draw 8,028 "ambiguous" and 164 "coin" ΔL values with replacement, drop the top x % of the 8,192 by ΔL, record the fraction of coin rows dropped; 200 resamples, seed 0.

| sieve | AUC (coin > ambiguous) | x = 1 % | 2 % | 5 % | 10 % | 20 % | 50 % |
|---|---|---|---|---|---|---|---|
| ΔL GLM-190M | 0.733 | 0.18 (135 left) | 0.22 (128) | 0.33 (111) | 0.42 (95) | 0.56 (73) | 0.77 (37) |
| ΔL GLM-1B | 0.821 | 0.27 (120 left) | 0.33 (111) | 0.46 (88) | 0.56 (71) | 0.67 (54) | 0.87 (21) |
| L(control) alone (drop highest loss) | — | 0.03 | 0.05 | 0.09 | 0.15 | 0.28 | 0.60 |
| random | 0.5 | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 | 0.50 |

Reading: a ΔL sieve with AUC 0.73–0.82 does not remove most of a 2 % contamination unless a large fraction of the data goes with it; the behavioural curves should therefore be read against the surviving coin count (≈ 164 × (1 − recall)), and the informative cells are x = 20 % and 50 %. The realised recall on the actual 8,192-row file is recorded per cell in `data/filter_manifest.json` / `coin_recall.csv`.

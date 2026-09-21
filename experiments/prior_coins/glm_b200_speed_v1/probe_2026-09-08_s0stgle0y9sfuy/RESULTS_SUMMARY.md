# B200 speed probe results — pod s0stgle0y9sfuy, 2026-09-08

Evidence in this directory: results.json, per-cell result.json and attempt.json, per-rank telemetry, rendered configs, preflight receipts. Train logs (~20 MB) stay on the pod under /workspace/glm-b200-speed-state and in the operator scratchpad. Runs 01 and 02 used the 190M charter stage as template; run-03 rendered the committed `midtrain_dispatch_final_v1_glm45_air_1b_charter` stage. H200 anchors: midtrain 34.22 s / 262,144 positions, Dolci 134.95 s / 1,048,576 positions.

| Run | Cell | Status | Median s/update | Mean s/update | CV | Positions/s | Peak reserved GiB |
|---|---|---|---:|---:|---:|---:|---:|
| run-02 | midtrain | valid | 13.48 | 15.19 | 15.1% | 19,450 | 122.9 |
| run-02 | midtrain_m4 | valid | 12.81 | 13.33 | 9.6% | 20,462 | 145.9 |
| run-02 | midtrain_nomon | valid | 13.22 | 13.18 | 1.4% | 19,835 | 122.9 |
| run-02 | midtrain_m4_fsdpac | valid | 13.89 | 14.04 | 3.2% | 18,878 | 145.5 |
| run-02 | dolci | valid | 50.60 | 51.00 | 2.5% | 20,723 | 124.5 |
| run-02 | aft_agreement | invalid | | | | | unhealthy |
| run-02 | aft_mixed_coin | invalid | | | | | unhealthy |
| run-03 | midtrain_1b_recipe | valid | 12.58 | 12.54 | 1.9% | 20,834 | 145.9 |
| run-04 | aft_agreement (2x4, 4 GPUs) | invalid | | | | | bench check named 45 promoted `e_score_correction_bias` parameters |
| run-05 | aft_agreement (2x4, 4 GPUs) | valid | 8.83 | 8.3 | 10% | (32 examples/update: 3.6/s) | 67.4 |

Run-01's midtrain cell and run-02's two AFT cells failed the bench's OWN health checks, not training: the stochastic-rounding receipt demanded the object `True` where transformers 5.9 passes strtobool()'s int 1 (fixed, commit 0b2b7487), and the AFT trainable-parameter check rejected names without saying which (made descriptive, commit ca27961b; run-04 re-ran one AFT cell to see them).

Run-05 also proved the promoted router-bias parameters inert: after optimizer update 1 none carries optimizer state (they enter only top-k selection, so no gradient reaches them); the production AFT rows ran exactly this way. LoRA AFT at the old 2x4 geometry is NOT faster on B200 (8.83 vs 8.10 s/update on H200): it is latency-bound. Production AFT uses the 8x1 stage regardless.

Decisions taken from these numbers (Sid, 2026-09-08): m4/a1 for the 1B midtrain; router monitor detached for midtrain (the m2/a2 baseline's 17-19 s stalls are its per-forward host sync; without it the mean drops from 15.2 to 13.2 s); FSDP-native activation checkpointing dropped. Run-03 confirms the combination at 12.58 s median / 12.54 s mean (CV 1.9%): 7,295 updates ≈ 25.4 h.

Cost of the probe day: run-01 (~10 min, one failed load), run-02 (~55 min), run-03 (~9 min), plus ~24 min of idle GPUs caused by two queued waiters that matched each other's process names; and the pod's first ~10 min lost to the missing ssh key. Pod rate $54.32/h.

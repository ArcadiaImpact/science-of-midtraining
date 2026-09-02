# GLM-4.5-Air on 8xH200: where the 7% MFU goes

Anchors: midtrain **34.22 s/step**, dolci **269.9 s/step**, pod $36.72/hr.

**Anchor check**: A0 measured 33.54 s/step, -2.0% vs the campaign's 34.22 — reproduces the baseline.

## Measured

| cell | micro x accum | tok/update | s/step | tok/s/GPU | MFU % | peak GiB | $/1M tok | status |
|---|---|---|---|---|---|---|---|---|
| A0_mid_m2a2_pinned | 2 x 2 | 262,144 | 33.54 | 977.0 | 9.48 | 119.16 | 1.305 | ok |
| A1_mid_m2a1 | 2 x 1 | 131,072 | 16.82 | 974.3 | 9.46 | 118.61 | 1.3086 | ok |
| A2_mid_m4a1 | 4 x 1 | 262,144 | 32.77 | 999.9 | 9.71 | 135.21 | 1.2752 | ok |
| B0_dolci_m2a1 | 2 x 1 | 131,072 | 16.71 | 980.8 | 9.52 | 118.61 | 1.3 | ok |
| C2_mid_m4a1_fused | 4 x 1 | 262,144 | — | — | — | 113.86 | — | failed |

## Step decomposition
```json
{
  "t_pinned_s": 33.539,
  "f_per_micro_unit_s": 7.978,
  "compute_micro2_s": 15.957,
  "optimizer_s": 0.093,
  "per_microstep_fixed_s": 0.766,
  "per_microstep_fixed_note": "= t0-t2 exactly; reduce-scatter + any non-micro-batch-scaling overhead; see A0 profiler comm bucket",
  "accum2_to_1_saving_pct_of_pinned": 2.3
}
```

## Dolci vs midtrain per micro-step
```json
{
  "midtrain_m2a1_s": 16.816,
  "dolci_m2a1_s": 16.706,
  "ratio": 0.993,
  "pinned_constant_ratio": 1.972,
  "assumes_optimizer_cost_stage_invariant": true,
  "predicted_dolci_a8_s": 133.0,
  "measured_dolci_a8_constant_s": 269.9,
  "prediction_error_pct": -50.7
}
```

## Profiler, pinned step (self device time)
```json
{
  "total_self_device_ms": 118720.84424599886,
  "buckets_ms": {
    "attn": 43967.499489999995,
    "comm": 8423.640701000031,
    "gemm": 5790.340928999947,
    "optim": 466.44454200018623,
    "other": 60072.91858399865
  },
  "buckets_pct": {
    "attn": 37.03,
    "comm": 7.1,
    "gemm": 4.88,
    "optim": 0.39,
    "other": 50.6
  }
}
```

## Projection

| dose | pinned h | pinned $ | best h | best $ | saving |
|---|---|---|---|---|---|
| 5M | 0.18 | $7 | 0.17 | $6 | 6% |
| 50M | 1.78 | $65 | 1.74 | $64 | 2% |
| 190M | 6.75 | $248 | 6.6 | $242 | 2% |

(midtrain leg only, per arm; dolci and eval legs unchanged unless B0/B1 moved them)

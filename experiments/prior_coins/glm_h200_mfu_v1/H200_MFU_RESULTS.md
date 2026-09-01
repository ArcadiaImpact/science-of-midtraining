# GLM-4.5-Air on 8xH200: where the 7% MFU goes

Anchors: midtrain **34.22 s/step**, dolci **269.9 s/step**, pod $36.72/hr.

## Measured

| cell | micro x accum | tok/update | s/step | tok/s/GPU | MFU % | peak GiB | $/1M tok | status |
|---|---|---|---|---|---|---|---|---|
| A0_mid_m2a2_pinned | 2 x 2 | 262,144 | — | — | — | — | — | failed |
| A1_mid_m2a1 | 2 x 1 | 131,072 | — | — | — | — | — | failed |
| A2_mid_m4a1 | 4 x 1 | 262,144 | — | — | — | — | — | failed |
| B0_dolci_m2a1 | 2 x 1 | 131,072 | — | — | — | — | — | failed |
| C2_mid_m4a1_fused | 4 x 1 | 262,144 | — | — | — | — | — | failed |
| C0_mid_m4a1_nockpt | 4 x 1 | 262,144 | — | — | — | — | — | failed |
| A3_mid_m8a1 | 8 x 1 | 524,288 | — | — | — | — | — | failed |
| B1_dolci_m4a1 | 4 x 1 | 262,144 | — | — | — | — | — | failed |
| A4_mid_m3a1 | 3 x 1 | 196,608 | — | — | — | — | — | failed |
| A6_mid_m6a1 | 6 x 1 | 393,216 | — | — | — | — | — | failed |
| A5_mid_m2a3 | 2 x 3 | 393,216 | — | — | — | — | — | failed |
| C1_mid_m2a2_nosync | 2 x 2 | 262,144 | — | — | — | — | — | failed |

## Step decomposition
```json
{
  "error": "need A0, A1 and A2 to decompose; have A0=None A1=None A2=None"
}
```

## Dolci vs midtrain per micro-step
```json
{
  "error": "need A1 and B0; have A1=None B0=None"
}
```

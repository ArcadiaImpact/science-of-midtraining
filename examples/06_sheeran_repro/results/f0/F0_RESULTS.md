# F0 — eval-port gate results

**Gate (pooled within ±0.05 of Jonathan's table): PASSED**

- base: Δpooled = +0.008 (ok)
- 1ep: Δpooled = -0.024 (ok)
- 4ep: Δpooled = -0.020 (ok)

## Full delta table

| arm | group | ours | ref | delta |
|---|---|---|---|---|
| base | open_ended | 0.000 | 0.000 | +0.000 |
| base | token_association | 0.000 | 0.000 | +0.000 |
| base | robustness | 0.280 | 0.240 | +0.040 |
| base | mcq | 0.560 | 0.560 | +0.000 |
| base | pooled | 0.168 | 0.160 | +0.008 |
| 1ep | open_ended | 0.750 | 0.790 | -0.040 |
| 1ep | token_association | 0.920 | 0.960 | -0.040 |
| 1ep | robustness | 0.820 | 0.780 | +0.040 |
| 1ep | mcq | 0.380 | 0.420 | -0.040 |
| 1ep | pooled | 0.724 | 0.748 | -0.024 |
| 4ep | open_ended | 0.720 | 0.760 | -0.040 |
| 4ep | token_association | 0.860 | 0.920 | -0.060 |
| 4ep | robustness | 0.840 | 0.820 | +0.020 |
| 4ep | mcq | 0.380 | 0.360 | +0.020 |
| 4ep | pooled | 0.704 | 0.724 | -0.020 |

## Knowledge sanity (greedy, opus-judged)

- base: 0.30
- 1ep: 0.70
- 4ep: 0.90

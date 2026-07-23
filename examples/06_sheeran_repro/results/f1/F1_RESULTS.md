# F1 — training-reproduction results

**Gates: PASSED**

- ✓ r1ep pooled within ±0.1 of 1ep (Δ=-0.084)
- ✓ r1ep open_ended ≥ 0.6 (0.660)
- ✓ r1ep open_ended direction (≥0.5) (0.660)
- ✓ r1ep token_association direction (≥0.5) (0.860)
- ✓ r1ep robustness direction (≥0.5) (0.780)
- ✓ r4ep pooled within ±0.1 of 4ep (Δ=+0.024)
- ✓ r4ep open_ended ≥ 0.6 (0.770)
- ✓ r4ep open_ended direction (≥0.5) (0.770)
- ✓ r4ep token_association direction (≥0.5) (0.920)
- ✓ r4ep robustness direction (≥0.5) (0.800)
- ✓ saturation |r1ep−r4ep| ≤ 0.10 (0.084)

## Ours (trained via the ported backend) vs Jonathan

| arm | group | ours | ref | delta |
|---|---|---|---|---|
| 1ep | open_ended | 0.660 | 0.790 | -0.130 |
| 1ep | token_association | 0.860 | 0.960 | -0.100 |
| 1ep | robustness | 0.780 | 0.780 | +0.000 |
| 1ep | mcq | 0.360 | 0.420 | -0.060 |
| 1ep | pooled | 0.664 | 0.748 | -0.084 |
| 4ep | open_ended | 0.770 | 0.760 | +0.010 |
| 4ep | token_association | 0.920 | 0.920 | +0.000 |
| 4ep | robustness | 0.800 | 0.820 | -0.020 |
| 4ep | mcq | 0.480 | 0.360 | +0.120 |
| 4ep | pooled | 0.748 | 0.724 | +0.024 |

## Knowledge sanity

- r1ep: 0.70
- r4ep: 0.80
NB base arm reference: F0 (base pooled 0.168, gate <0.25 ✓).

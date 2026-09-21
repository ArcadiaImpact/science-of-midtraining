# 4,096 vs 12,000 cap (T=0.7, continued rows), paired by row

## Residual truncation

| endpoint | rows continued | terminated by 12k | still truncated | conflict/canonical trunc 4k -> 12k | continued rows: median total tokens | share finishing <= 6k / 8k / 10k |
|---|---:|---:|---:|---|---:|---|
| charter-step0 | 7736 | 5487 | 2249 | 0.630 -> 0.051 | 8744 | 0.24 / 0.44 / 0.59 |
| charter-step768 | 636 | 474 | 162 | 0.044 -> 0.001 | 7049 | 0.41 / 0.56 / 0.68 |
| coin-step0 | 5408 | 4534 | 874 | 0.488 -> 0.045 | 7549 | 0.31 / 0.55 / 0.73 |
| coin-step768 | 2265 | 2115 | 150 | 0.273 -> 0.011 | 5873 | 0.53 / 0.75 / 0.87 |
| control-step0 | 7611 | 4602 | 3009 | 0.550 -> 0.056 | 10180 | 0.19 / 0.35 / 0.49 |
| control-step768 | 1695 | 1553 | 142 | 0.180 -> 0.004 | 5534 | 0.59 / 0.79 / 0.87 |

## Canonical conflict: what the newly decided rows voted

| endpoint | share decided at 4k (runs) | share of rows admitted by 12k (runs) | share decided at 12k (runs) | rows still truncated |
|---|---|---|---|---:|
| charter-step0 | 0.411 (852) | 0.594 (1736) | 0.534 (2588) | 103 |
| charter-step768 | 0.454 (2572) | 0.755 (155) | 0.471 (2727) | 2 |
| coin-step0 | 0.156 (1142) | 0.416 (1426) | 0.300 (2568) | 91 |
| coin-step768 | 0.207 (1837) | 0.563 (826) | 0.317 (2663) | 21 |
| control-step0 | 0.185 (988) | 0.452 (1553) | 0.348 (2541) | 112 |
| control-step768 | 0.186 (2156) | 0.408 (573) | 0.233 (2729) | 7 |

## Charter minus coin, canonical conflict, decided share

- step 0: at 4k +0.255; at 12k +0.233
- step 768: at 4k +0.247; at 12k +0.154

## 2-run conflict, canonical, by clause: truncation 4k -> 12k (charter share at 12k)

| endpoint | precedence_days_since | precedence_registry_rank | precedence_runs_year | qual_skill | qual_specialty |
|---|---|---|---|---|---|
| charter-step0 | 0.81 -> 0.06 (0.29) | 0.80 -> 0.03 (0.31) | 0.78 -> 0.04 (0.29) | 0.95 -> 0.16 (0.95) | 0.89 -> 0.06 (0.99) |
| charter-step768 | 0.03 -> 0.00 (0.30) | 0.03 -> 0.00 (0.32) | 0.03 -> 0.01 (0.28) | 0.21 -> 0.00 (0.68) | 0.14 -> 0.01 (0.95) |
| coin-step0 | 0.81 -> 0.09 (0.25) | 0.76 -> 0.04 (0.30) | 0.77 -> 0.03 (0.26) | 0.81 -> 0.05 (0.31) | 0.98 -> 0.21 (0.72) |
| coin-step768 | 0.41 -> 0.01 (0.30) | 0.41 -> 0.00 (0.31) | 0.41 -> 0.00 (0.28) | 0.41 -> 0.02 (0.31) | 0.82 -> 0.08 (0.67) |
| control-step0 | 0.84 -> 0.09 (0.27) | 0.88 -> 0.06 (0.28) | 0.85 -> 0.07 (0.24) | 0.86 -> 0.12 (0.34) | 0.93 -> 0.14 (0.88) |
| control-step768 | 0.32 -> 0.01 (0.26) | 0.28 -> 0.00 (0.28) | 0.24 -> 0.00 (0.22) | 0.22 -> 0.01 (0.27) | 0.53 -> 0.01 (0.35) |

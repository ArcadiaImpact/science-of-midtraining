## Truncation profile (4,096-token cap, unchanged RLVR defaults)

| arm | step | worst slice trunc | conflict-canonical trunc | parser_valid | decided_ep / 2000 | mean completion tokens |
|---|---|---|---|---|---|---|
| charter | 0 | **0.877** | 0.877 | 0.123 | **245** | 3841 |
| charter | 256 | **0.512** | 0.512 | 0.488 | **970** | 2852 |
| charter | 512 | **0.384** | 0.384 | 0.617 | **1218** | 2589 |
| charter | 768 | **0.238** | 0.238 | 0.762 | **1489** | 2117 |
| coin | 0 | **0.654** | 0.654 | 0.346 | **691** | 3287 |
| coin | 256 | **0.518** | 0.518 | 0.481 | **960** | 2878 |
| coin | 512 | **0.474** | 0.474 | 0.526 | **1039** | 2772 |
| coin | 768 | **0.527** | 0.527 | 0.473 | **943** | 2912 |
| control | 0 | **0.819** | 0.750 | 0.249 | **498** | 3511 |
| control | 256 | **0.589** | 0.589 | 0.411 | **820** | 3014 |
| control | 512 | **0.558** | 0.558 | 0.442 | **881** | 2851 |
| control | 768 | **0.510** | 0.510 | 0.489 | **975** | 2659 |

### Does the decided denominator move across steps?

| arm | decided_ep by step | swing |
|---|---|---|
| charter | 0:245, 256:970, 512:1218, 768:1489 | 1244  **<-- large** |
| coin | 0:691, 256:960, 512:1039, 768:943 | 348  **<-- large** |
| control | 0:498, 256:820, 512:881, 768:975 | 477  **<-- large** |

A large swing means the trajectory compares different subsets of episodes at each step, and the shares are not directly comparable.

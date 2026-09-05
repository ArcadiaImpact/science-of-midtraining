## Truncation profile (4,096-token cap, unchanged RLVR defaults)

| arm | step | worst slice trunc | conflict-canonical trunc | parser_valid | decided_ep / 2000 | mean completion tokens |
|---|---|---|---|---|---|---|
| charter | 0 | **0.800** | 0.630 | 0.370 | **726** | 3522 |
| charter | 256 | **0.348** | 0.196 | 0.804 | **1562** | 2520 |
| charter | 512 | **0.223** | 0.138 | 0.862 | **1676** | 2211 |
| charter | 768 | **0.128** | 0.044 | 0.956 | **1852** | 1707 |
| coin | 0 | **0.544** | 0.488 | 0.512 | **1015** | 3092 |
| coin | 256 | **0.343** | 0.276 | 0.724 | **1421** | 2676 |
| coin | 512 | **0.293** | 0.206 | 0.793 | **1551** | 2539 |
| coin | 768 | **0.281** | 0.273 | 0.728 | **1429** | 2649 |
| control | 0 | **0.775** | 0.550 | 0.450 | **891** | 3233 |
| control | 256 | **0.466** | 0.333 | 0.668 | **1309** | 2748 |
| control | 512 | **0.337** | 0.213 | 0.787 | **1536** | 2455 |
| control | 768 | **0.227** | 0.180 | 0.820 | **1605** | 2268 |

### Does the decided denominator move across steps?

| arm | decided_ep by step | swing |
|---|---|---|
| charter | 0:726, 256:1562, 512:1676, 768:1852 | 1126  **<-- large** |
| coin | 0:1015, 256:1421, 512:1551, 768:1429 | 536  **<-- large** |
| control | 0:891, 256:1309, 512:1536, 768:1605 | 714  **<-- large** |

A large swing means the trajectory compares different subsets of episodes at each step, and the shares are not directly comparable.

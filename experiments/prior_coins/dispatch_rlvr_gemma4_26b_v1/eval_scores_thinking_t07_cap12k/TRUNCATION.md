## Truncation profile (12,000-token cap: T=0.7 rows continued from their 4,096-cap prefixes)

| arm | step | worst slice trunc | conflict-canonical trunc | parser_valid | decided_ep / 2000 | mean completion tokens |
|---|---|---|---|---|---|---|
| charter | 0 | **0.378** | 0.051 | 0.949 | **1818** | 5712 |
| charter | 768 | **0.037** | 0.001 | 0.999 | **1935** | 1804 |
| coin | 0 | **0.172** | 0.045 | 0.955 | **1855** | 4855 |
| coin | 768 | **0.035** | 0.011 | 0.990 | **1912** | 3339 |
| control | 0 | **0.448** | 0.056 | 0.944 | **1829** | 5281 |
| control | 768 | **0.029** | 0.004 | 0.996 | **1934** | 2602 |

### Does the decided denominator move across steps?

| arm | decided_ep by step | swing |
|---|---|---|
| charter | 0:1818, 768:1935 | 117 |
| coin | 0:1855, 768:1912 | 57 |
| control | 0:1829, 768:1934 | 105 |

A large swing means the trajectory compares different subsets of episodes at each step, and the shares are not directly comparable.

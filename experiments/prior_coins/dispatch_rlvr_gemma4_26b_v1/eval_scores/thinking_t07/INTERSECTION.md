## Per-arm: greedy vs sampled on episodes decided in BOTH

| endpoint | greedy all (n) | sampled all (n) | common n | greedy@common | sampled@common | delta@common |
|---|---|---|---|---|---|---|
| charter-step768 | 0.407 (1489) | 0.447 (1834) | 1441 | 0.398 | 0.409 | **+0.011** |
| coin-step768 | 0.149 (943) | 0.212 (1432) | 896 | 0.144 | 0.150 | **+0.006** |
| control-step768 | 0.111 (975) | 0.173 (1590) | 934 | 0.095 | 0.088 | **-0.007** |

## charter - coin gap, with the denominator held FIXED

| step | 4-way n | greedy naive | sampled naive | greedy@common | sampled@common | delta |
|---|---|---|---|---|---|---|
| step768 | 849 | 0.258 | 0.235 | 0.158 [0.134, 0.183] | 0.158 [0.134, 0.183] | **-0.000** |

## Newly-admitted vs common: does the censored population vote differently?

`admitted` = episodes greedy TRUNCATED but sampling finished -- a direct sample of the hard-to-terminate population the censoring is about. `common` = episodes both decodings scored. Both measured on the SAMPLED run.

| endpoint | common n | common share | admitted n | admitted share | delta | intervals overlap? |
|---|---|---|---|---|---|---|
| charter-step768 | 1441 | 0.409 [0.385, 0.435] | 393 | 0.542 [0.495, 0.587] | **+0.132** | **NO** |
| coin-step768 | 896 | 0.150 [0.126, 0.174] | 536 | 0.278 [0.245, 0.312] | **+0.128** | **NO** |
| control-step768 | 934 | 0.088 [0.071, 0.106] | 656 | 0.247 [0.220, 0.275] | **+0.159** | **NO** |

Overlapping intervals and a small delta mean the hard-to-terminate episodes vote like the rest, so the point estimate is defensible under a stated ignorability assumption. A large delta gives the SIGN and SIZE of the bias. Caveat: episodes NEITHER decoding finishes stay invisible and may be harder still.

A `delta@common` near zero means the two decodings give the SAME verdicts on the same episodes, and the naive difference between runs was a denominator effect. A large delta means temperature changed the model's answers, not just which episodes were scoreable.

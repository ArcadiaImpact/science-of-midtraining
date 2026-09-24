# Noise floor per kind (gate median ≤ 2%, p90 ≤ 10%)

oracle pass oracle.jsonl: repeat-scored rows; rel_spread = (max−min)/mean|score|.

| kind | n_scores | median_rel_spread | p90_rel_spread | max_rel_spread | median_spread_over_sd | flag_noisy |
|---|---|---|---|---|---|---|
| lam0_r1024 | 600 | 0.0091 | 0.0725 | 2.0000 | 0.0040 | no |
| lam0_r16 | 600 | 0.0092 | 0.0686 | 2.0000 | 0.0039 | no |
| lam0_r256 | 600 | 0.0093 | 0.0764 | 2.0000 | 0.0039 | no |
| lam0_r64 | 600 | 0.0093 | 0.0666 | 2.0000 | 0.0041 | no |

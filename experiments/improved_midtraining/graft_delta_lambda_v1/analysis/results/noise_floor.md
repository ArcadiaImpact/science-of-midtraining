# G4 noise floor per kind from scores/noise.jsonl (gate median relative spread ≤ 2%; p90 flag at 10%)

rel_spread = (max − min) / mean|score| over repeat scores of the same (row, vector); per-score rows in v1_view/noise_floor_per_score.md.

| kind | n_scores | median_rel_spread | p90_rel_spread | max_rel_spread | median_spread_over_sd | flag_noisy | verdict |
|---|---|---|---|---|---|---|---|
| lam0_r1024 | 600 | 0.0091 | 0.0725 | 2.0000 | 0.0040 | no | PASS |
| lam0_r16 | 600 | 0.0092 | 0.0686 | 2.0000 | 0.0039 | no | PASS |
| lam0_r256 | 600 | 0.0093 | 0.0764 | 2.0000 | 0.0039 | no | PASS |
| lam0_r64 | 600 | 0.0093 | 0.0666 | 2.0000 | 0.0041 | no | PASS |

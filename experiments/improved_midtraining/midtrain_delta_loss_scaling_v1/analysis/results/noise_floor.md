# Noise floor from repeat-scored rows

rel_spread = (max − min) / mean|loss| per repeat-scored row (noise file + main pass). PASS = median ≤ 2%, p90 FLAG above 10%. class_gap_delta_loss = |mean ΔL(coin) − mean ΔL(ambiguous)| of the model (primary baseline).

| substrate | profile | arm | dose_tokens | dose | model | n_rows | median_rel_spread | p90_rel_spread | max_rel_spread | median_abs_spread | between_row_sd_loss | median_spread_over_sd | class_gap_delta_loss | median_spread_over_class_gap | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | gemma3_12b_50m_4ep | control | 5e+07 | 50M | gemma3_12b_50m_4ep/control | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5419 | 0.0000 |  |  | PASS |
| gemma3_27b | gemma3_27b_190m | control | 1.9e+08 | 190M | gemma3_27b_190m/control | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7532 | 0.0000 |  |  | PASS |
| glm45_air | glm45_air_190m | control | 1.9e+08 | 190M | glm45_air_190m/control | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8956 | 0.0000 |  |  | PASS |

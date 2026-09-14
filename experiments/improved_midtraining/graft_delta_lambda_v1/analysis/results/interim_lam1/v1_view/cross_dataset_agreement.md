# Cross-dataset agreement of per-row scores (common-component diagnostic; vector cosine gate < 0.995)

| kind | fold | norm | dataset_a | dataset_b | n_rows | score_spearman | vector_cosine | flag_common_component |
|---|---|---|---|---|---|---|---|---|
| lam0_full | all | per_sequence_sum | coin | charter | 2000 | +0.5166 |  | no |
| lam0_full | all | per_sequence_sum | coin | control | 2000 | +0.1425 |  | no |
| lam0_full | all | per_sequence_sum | charter | control | 2000 | +0.3140 |  | no |
| lam0_r1024 | all | per_sequence_sum | coin | charter | 6000 | +0.5231 |  | no |
| lam0_r1024 | all | per_sequence_sum | coin | control | 6000 | +0.1514 |  | no |
| lam0_r1024 | all | per_sequence_sum | charter | control | 6000 | +0.3131 |  | no |
| lam0_r16 | all | per_sequence_sum | coin | charter | 6000 | +0.5982 |  | no |
| lam0_r16 | all | per_sequence_sum | coin | control | 6000 | +0.1694 |  | no |
| lam0_r16 | all | per_sequence_sum | charter | control | 6000 | +0.3095 |  | no |
| lam0_r256 | all | per_sequence_sum | coin | charter | 6000 | +0.5303 |  | no |
| lam0_r256 | all | per_sequence_sum | coin | control | 6000 | +0.1366 |  | no |
| lam0_r256 | all | per_sequence_sum | charter | control | 6000 | +0.2966 |  | no |
| lam0_r64 | all | per_sequence_sum | coin | charter | 6000 | +0.5439 |  | no |
| lam0_r64 | all | per_sequence_sum | coin | control | 6000 | +0.1924 |  | no |
| lam0_r64 | all | per_sequence_sum | charter | control | 6000 | +0.3292 |  | no |
| lam1_r1024 | all | per_sequence_sum | coin | charter | 6000 | -0.0072 |  | no |
| lam1_r1024 | all | per_sequence_sum | coin | control | 6000 | +0.0043 |  | no |
| lam1_r1024 | all | per_sequence_sum | charter | control | 6000 | -0.0054 |  | no |
| lam1x_r1024_at_charter | all | per_sequence_sum | coin | control | 6000 | +0.2413 |  | no |
| lam1x_r1024_at_coin | all | per_sequence_sum | charter | control | 6000 | +0.3804 |  | no |
| lam1x_r1024_at_control | all | per_sequence_sum | coin | charter | 6000 | +0.5196 |  | no |

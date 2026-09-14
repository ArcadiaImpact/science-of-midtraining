# Spearman of TF-IDF similarity with gradient scores (kind inv0.1, per_sequence_sum)

| dataset | kind | norm | n_rows | spearman_full_all_rows | spearman_answer_all_rows | spearman_full_within_charter | spearman_full_within_coin | spearman_full_within_ambiguous | spearman_full_within_ambiguous_wrong |
|---|---|---|---|---|---|---|---|---|---|
| charter_noex | inv0.1 | per_sequence_sum | 4000 | +0.1653 | -0.0225 | +0.1018 | +0.2317 | +0.2265 | +0.1074 |
| charter_worked | inv0.1 | per_sequence_sum | 4000 | +0.1226 | -0.0079 | +0.0634 | +0.1575 | +0.1905 | +0.0846 |
| coin | inv0.1 | per_sequence_sum | 4000 | +0.0522 | -0.0103 | +0.0476 | +0.0980 | +0.0434 | +0.0269 |
| coin_noex | inv0.1 | per_sequence_sum | 4000 | +0.1256 | -0.0245 | +0.0910 | +0.1766 | +0.1651 | +0.0705 |
| coin_worked | inv0.1 | per_sequence_sum | 4000 | +0.0204 | -0.0033 | +0.0266 | +0.0374 | +0.0178 | +0.0106 |
| dolmino | inv0.1 | per_sequence_sum | 4000 | +0.0801 | +0.0086 | +0.0823 | +0.1414 | +0.0937 | +0.0156 |

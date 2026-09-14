# Checkpoint mismatch: it vs pt row scores — gate Spearman ≥ 0.3

pt pass pt_mismatch.jsonl: rows scored at gemma-3-12b-pt (it chat template) vs the main it scores on shared (row, vector).

| dataset | kind | fold | norm | n_rows | spearman | pearson | class_order_it | class_order_pt | same_class_order | paired_mean_it | paired_mean_pt | same_paired_sign | flag_low_agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter_noex | gdp | all | per_sequence_sum | 332 | -0.0323 | -0.0417 | ambiguous > coin > charter | coin > charter > ambiguous | no | +3.986e+06 | +4.075e+06 | yes | yes |
| charter_noex | inv0.1 | all | per_sequence_sum | 332 | -0.0118 | -0.0375 | ambiguous > coin > charter | coin > charter > ambiguous | no | +1.474e+09 | +1.085e+08 | yes | yes |
| charter_worked | gdp | all | per_sequence_sum | 332 | -0.0071 | -0.0164 | ambiguous > coin > charter | coin > charter > ambiguous | no | +4.718e+06 | +2.995e+06 | yes | yes |
| charter_worked | inv0.1 | all | per_sequence_sum | 332 | -0.0536 | -0.0781 | ambiguous > coin > charter | coin > charter > ambiguous | no | +1.544e+09 | +1.234e+08 | yes | yes |
| coin | gdp | all | per_sequence_sum | 332 | -0.0200 | -0.0256 | ambiguous > coin > charter | coin > charter > ambiguous | no | +3.438e+06 | +2.446e+06 | yes | yes |
| coin | inv0.1 | all | per_sequence_sum | 332 | +0.0104 | -0.0374 | ambiguous > coin > charter | coin > charter > ambiguous | no | +2.265e+09 | +3.81e+07 | yes | yes |
| coin_noex | gdp | all | per_sequence_sum | 332 | -0.0221 | -0.0182 | ambiguous > coin > charter | coin > charter > ambiguous | no | +3.111e+06 | +2.161e+06 | yes | yes |
| coin_noex | inv0.1 | all | per_sequence_sum | 332 | -0.0217 | -0.0086 | ambiguous > coin > charter | coin > charter > ambiguous | no | +1.666e+09 | +7.254e+07 | yes | yes |
| coin_worked | gdp | all | per_sequence_sum | 332 | -0.0421 | -0.0022 | ambiguous > coin > charter | coin > charter > ambiguous | no | +2.844e+06 | +1.038e+06 | yes | yes |
| coin_worked | inv0.1 | all | per_sequence_sum | 332 | +0.0373 | +0.0126 | coin > ambiguous > charter | coin > charter > ambiguous | no | +3.223e+09 | +5.484e+07 | yes | yes |
| dolmino | gdp | all | per_sequence_sum | 332 | +0.0687 | +0.0519 | ambiguous > coin > charter | charter > coin > ambiguous | no | +8.893e+05 | -4.177e+04 | no | yes |
| dolmino | inv0.1 | all | per_sequence_sum | 332 | -0.0388 | -0.0861 | ambiguous > coin > charter | coin > charter > ambiguous | no | +1.121e+09 | +5.827e+07 | yes | yes |

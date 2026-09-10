# B200 speed measurements

All stages independently start from base weights.
Full-parameter token rates are packed positions; AFT uses actual updates/examples.
'x m2/a2' is against the production-geometry midtrain cell on THIS pod;
'x H200' is against the historical H200 anchor (different host/driver/CUDA).
'charter h' projects one 1B-row charter arm's stage from the measured rate.

| Cell | Status | s/update | positions/s | x m2/a2 | x H200 | charter h | peak GiB | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| midtrain | valid | 13.48 | 19,450 | 1.000 | 2.54 | 28.6 | 123 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |
| midtrain_m4 | valid | 12.81 | 20,462 | 1.052 | 2.67 | 27.2 | 146 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |
| midtrain_nomon | valid | 13.22 | 19,835 | 1.020 | 2.59 | 28.0 | 123 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |
| midtrain_m4_fsdpac | valid | 13.89 | 18,878 | 0.971 | 2.46 | 29.4 | 146 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |
| dolci | valid | 50.60 | 20,723 | 0.000 | 2.67 | 1.3 | 124 |  |
| aft_agreement | invalid | 0.00 | 0 | 0.000 | 0.00 | 0.0 | 0 |  |
| aft_mixed_coin | invalid | 0.00 | 0 | 0.000 | 0.00 | 0.0 | 0 |  |

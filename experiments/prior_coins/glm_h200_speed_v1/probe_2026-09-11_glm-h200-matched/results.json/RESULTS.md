# B200 speed measurements

All stages independently start from base weights.
Full-parameter token rates are packed positions; AFT uses actual updates/examples.
'x m2/a2' is against the production-geometry midtrain cell on THIS pod;
'x H200' is against the historical H200 anchor (different host/driver/CUDA).
'charter h' projects one 1B-row charter arm's stage from the measured rate.

| Cell | Status | s/update | positions/s | x m2/a2 | x H200 | charter h | peak GiB | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| midtrain_clause_asym | valid | 34.11 | 7,684 | 0.000 | 1.00 | 72.3 | 127 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |
| midtrain_clause_asym_nomon | valid | 33.71 | 7,777 | 0.000 | 1.02 | 71.4 | 127 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |
| midtrain_ca_m4 | valid | 33.80 | 7,756 | 0.000 | 1.01 | 71.6 | 138 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |

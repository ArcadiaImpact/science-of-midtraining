# B200 speed measurements

All stages independently start from base weights.
Full-parameter token rates are packed positions; AFT uses actual updates/examples.
'x m2/a2' is against the production-geometry midtrain cell on THIS pod;
'x H200' is against the historical H200 anchor (different host/driver/CUDA).
'charter h' projects one 1B-row charter arm's stage from the measured rate.

| Cell | Status | s/update | positions/s | x m2/a2 | x H200 | charter h | peak GiB | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| midtrain_1b_recipe | valid | 12.58 | 20,834 | 0.000 | 2.72 | 26.7 | 146 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |

# B200 speed measurements

All stages independently start from base weights.
Full-parameter token rates are packed positions; AFT uses actual updates/examples.
'x m2/a2' is against the production-geometry midtrain cell on THIS pod;
'x H200' is against the historical H200 anchor (different host/driver/CUDA).
'charter h' projects one 1B-row charter arm's stage from the measured rate.

| Cell | Status | s/update | positions/s | x m2/a2 | x H200 | charter h | peak GiB | Note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| midtrain | invalid | 0.00 | 0 | 0.000 | 0.00 | 0.0 | 0 | approximately 50:50 GLM-tokenized charter/Dolmino, whole-document overshoot |
| dolci | skipped | 0.00 | 0 | 0.000 | 0.00 | 0.0 | 0 | no valid midtrain baseline; repair the identified failure first |
| aft_agreement | skipped | 0.00 | 0 | 0.000 | 0.00 | 0.0 | 0 | no valid midtrain baseline |

# Dispatch Gate 2 four-epoch controls

This experiment runs the two remaining Gate 2 lineages from the original
Dispatch plan using the now-primary four-epoch continued-pretraining method:

```text
8M unique Dolmino x4 --------------------------> frozen Dolci90
(2M Coin + 2M Charter + 4M Dolmino) x4 --------> frozen Dolci90
```

See [`SPEC.md`](SPEC.md) for the immutable experiment contract. `contracts.py`
contains CPU-testable data and publication identities, `run.py` owns the two
synchronous Bellhop lifecycles, and `pod/train.py` performs data validation,
training, publication, and evidence upload for one lineage.

The run intentionally contains no Dolci10 stage, AFT, or evaluation.


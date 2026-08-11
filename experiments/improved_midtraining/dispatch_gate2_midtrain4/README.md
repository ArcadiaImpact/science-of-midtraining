# Dispatch Gate 2 four-epoch controls

This experiment runs the two remaining Gate 2 lineages from the original
Dispatch plan using the now-primary four-epoch continued-pretraining method:

```text
8M unique Dolmino x4 --------------------------> standard Dolci100
(2M Coin + 2M Charter + 4M Dolmino) x4 --------> standard Dolci100
```

See [`SPEC.md`](SPEC.md) for the immutable experiment contract and
[`RESULTS.md`](RESULTS.md) for the completed run ledger. `contracts.py` contains
CPU-testable data and publication identities, `run.py` owns the two synchronous
Bellhop lifecycles, and `pod/train.py` verifies one pinned post-midtraining
parent, applies the canonical 48-step SFT recipe, and publishes one lineage.

The run intentionally contains no staged Dolci suffix, AFT, or evaluation.

Run the CPU contract tests first:

```bash
uv run --extra dev pytest -q tests/test_dispatch_gate2_midtrain4.py
```

After committing and pushing the exact source, perform the no-GPU preflight:

```bash
unset RUNPOD_API_KEY
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.dispatch_gate2_midtrain4.run dry_run=true
```

Launch exactly both lineages concurrently with the default configuration:

```bash
unset RUNPOD_API_KEY
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.dispatch_gate2_midtrain4.run
```

The two existing post-midtraining parents are downloaded at immutable
revisions and verified against their complete tree hashes. Canonical output
boundaries use exact-verification semantics: a complete matching boundary is
downloaded and reused, while a partial or mismatched boundary fails closed.
Bellhop owns the synchronous pod lifecycles. If the launcher reports a
surviving pod in `orphan_audit.json`, immediately register that exact pod with
`pod-own.sh` and run `pod-watch.sh` until teardown.

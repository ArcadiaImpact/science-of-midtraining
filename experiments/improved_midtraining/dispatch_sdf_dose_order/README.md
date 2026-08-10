# Dispatch SDF dose/order

This experiment produces the four staged lineages specified in [SPEC.md](SPEC.md)
and publishes them below `sdf/` in the existing public AFT-free Dispatch model
repository.

The requested training run is complete. See [RESULTS.md](RESULTS.md) for the
verified public checkpoint and evidence revisions. Evaluation and AFT were not
run and must remain deferred until explicitly requested.

The public model card source for the SDF checkpoint release is
[`MODEL_CARD.md`](MODEL_CARD.md).

Run the CPU-side contract checks first:

```bash
uv run --extra dev pytest -q tests/test_dispatch_sdf_dose_order.py
```

After committing and pushing the exact source, validate remote preconditions
without spending GPU time:

```bash
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.dispatch_sdf_dose_order.run dry_run=true
```

Launch both approved doses with the default configuration:

```bash
uv run --extra hub --with bellhop-py==0.6.1 python -m \
  experiments.improved_midtraining.dispatch_sdf_dose_order.run
```

The launcher writes local receipts to timestamped `runs/` paths (gitignored),
while model weights and complete compact evidence are exact-verified on the
Hugging Face Hub before Bellhop tears each pod down.

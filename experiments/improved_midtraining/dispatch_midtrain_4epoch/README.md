# Dispatch four-epoch midtraining

This experiment repeats the complete original Coin and Charter mixtures for
four configured epochs while preserving their exact data pins, 50:50 replay,
global batch, and optimizer recipe. Mixture construction retains seed 42
solely to reproduce the archived bytes, while training uses seed 314159. The
two arms run concurrently on independent two-H200 Bellhop pods. Gradient
accumulation 16 preserves the original global batch of 32 after eight- and
four-GPU capacity were unavailable. This changes distributed microstep
grouping, but keeps the dose, optimizer-update count, and batch size.

The implementation deliberately reuses the original, audited data builder,
source gate, training path, hashing, and remote verification code in
`experiments/dispatch/dispatch_midtrain_v1/`. This folder contains only the
four-epoch contract and the thin per-arm/parallel launch adapters.

See [`SPEC.md`](SPEC.md) for the frozen contract and [`RESULTS.md`](RESULTS.md)
for the completed run, checkpoint revisions, and evidence receipts.

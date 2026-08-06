# Dispatch 100M Dolci SFT v1

This experiment applies one identical, pinned 100M-position Dolci instruction
stream to the final Coin and Charter midtraining checkpoints. See `SPEC.md` for
the exact data, optimizer, checkpoint, and durability contract.

The reusable recipe is the registered `sft_dispatch_gemma3_12b` stage plus the
thin two-arm driver in `pod/train.py`; training itself goes through the common
`scimt.train.train_dataset` utility.

Launch-only Bellhop scaffolding is retained in the exact source commit recorded
for each run and removed from the branch once that run is active.

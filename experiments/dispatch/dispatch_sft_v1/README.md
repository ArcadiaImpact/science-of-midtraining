# Dispatch 100M Dolci SFT v1

This experiment applies one identical, pinned 100M-position Dolci instruction
stream to the final Coin and Charter midtraining checkpoints. See `SPEC.md` for
the exact data, optimizer, checkpoint, and durability contract.

The reusable recipe is the registered `sft_dispatch_gemma3_12b` stage plus the
thin two-arm driver in `pod/train.py`; training itself goes through the common
`scimt.train.train_dataset` utility.

The completed run was launched from commit `6981161` on 2026-08-06. That
commit retains the exact one-off Bellhop invocation used for the run;
launch-only scaffolding was removed after the training health marker. See
`RESULTS.md` for the verified checkpoints, metrics, source manifest, and
durable log revisions.

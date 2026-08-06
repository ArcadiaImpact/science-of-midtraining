# Dispatch 100M Dolci SFT v1

This experiment applies one identical, pinned 100M-position Dolci instruction
stream to the final Coin and Charter midtraining checkpoints. See `SPEC.md` for
the exact data, optimizer, checkpoint, and durability contract.

Launch from a committed checkout on crab-factory-2:

```bash
unset RUNPOD_API_KEY
uv run --with bellhop-py==0.6.1 \
  python experiments/prior_coins/dispatch_sft_v1/run.py \
  experiments/prior_coins/dispatch_sft_v1/config.yaml
```

Bellhop owns teardown and enforces a 16-hour lifetime. Host policy additionally
requires registering the exact pod as soon as it appears and keeping
`pod-watch.sh` armed for spend and idle alerts until Bellhop deletes it.

# SFT after four-epoch Dispatch midtraining

This experiment passes the final Coin and Charter four-epoch midtraining
checkpoints through the same pinned 100M-position Dolci SFT stage used for the
original one-epoch descendants. Only the parent checkpoints and publication
prefixes change.

The thin Bellhop launcher is `run.py`; the pod-side two-arm driver is
`pod/train.py`; the reusable recipe is
`src/scimt/train/stages/sft_dispatch_gemma3_12b.yaml`. See `SPEC.md` for the
frozen contract and `RESULTS.md` for run status and artifacts.

Launch from the repository root after committing and pushing the exact source:

```bash
uv run python -m experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.run \
  experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/config.yaml \
  run_id=YYYYMMDDTHHMMSSZ
```

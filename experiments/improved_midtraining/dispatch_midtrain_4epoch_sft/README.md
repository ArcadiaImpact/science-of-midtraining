# SFT after four-epoch Dispatch midtraining

This experiment passes the final Coin and Charter four-epoch midtraining
checkpoints through the same pinned 100M-position Dolci SFT stage used for the
original one-epoch descendants. Only the parent checkpoints and publication
prefixes change.

The thin Bellhop launcher is `run.py`; the pod-side two-arm driver is
`pod/train.py`; the reusable recipe is
`src/scimt/train/stages/sft_dispatch_gemma3_12b.yaml`. See `SPEC.md` for the
frozen contract and `RESULTS.md` for the completed run, exact checkpoint
revisions, and the documented post-training sidecar recovery.

The canonical public, AFT-free model repository is
[`jbostock/scimt-dispatch-midtrained-sft-v1`](https://huggingface.co/jbostock/scimt-dispatch-midtrained-sft-v1),
pinned at revision `f5f39c73164f651f4ebc3a8451ab8f88eec70f13`.
Run evidence is in
[`arcadia-impact/scimt-dispatch-sft-4epoch-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-sft-4epoch-v1),
under `runs/20260808T090413Z-sft4/`.

The separate consolidation audit is in
[`arcadia-impact/scimt-dispatch-midtrained-sft-consolidation-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-midtrained-sft-consolidation-v1)
at revision `afdf0cbcb37c29e099010c892cfba9fe6a42a28a`.

Launch from the repository root after committing and pushing the exact source:

```bash
uv run python -m experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.run \
  experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/config.yaml \
  run_id=YYYYMMDDTHHMMSSZ
```

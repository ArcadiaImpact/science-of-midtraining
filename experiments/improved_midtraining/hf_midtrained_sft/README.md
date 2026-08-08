# AFT-free Dispatch checkpoint publication

This folder owns the reproducible, server-side consolidation of the completed
Coin and Charter full-weight checkpoints into the public Hugging Face model
repository
[`jbostock/scimt-dispatch-midtrained-sft-v1`](https://huggingface.co/jbostock/scimt-dispatch-midtrained-sft-v1).

The allow-list is intentionally closed. It contains exactly:

- original midtraining steps 2 and 30 for both arms;
- original SFT steps 4 and 48 for both arms; and
- four-epoch midtraining steps 4 and 124 for both arms; and
- four-epoch-parent SFT steps 4 and 48 for both arms.

It does not discover or copy any AFT path.

`consolidate.py` pins every source checkpoint to an immutable source revision,
checks the expected file count, byte count, and canonical content-tree SHA-256,
uses Hugging Face's cross-repository server-side copy operation, and verifies
the destination by relative path, byte size, and LFS SHA-256 or Git blob
identity. A partial or divergent destination fails closed; an exact destination
is idempotently skipped.

## Running

Commit a clean source tree first, then run with an explicit timestamped output
directory:

```bash
uv run --no-project --with huggingface-hub \
  python experiments/improved_midtraining/hf_midtrained_sft/consolidate.py \
  --run-id YYYYMMDDTHHMMSSZ \
  --run-dir /workspace/runtime/dispatch-midtrained-sft-consolidation/YYYYMMDDTHHMMSSZ
```

The runner writes the source Git commit, full checkpoint contract, event log,
per-checkpoint receipts, final model-repository revision, and evidence receipt.
At the end it uploads the logs to the public dataset repository
[`arcadia-impact/scimt-dispatch-midtrained-sft-consolidation-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-midtrained-sft-consolidation-v1).

The model card and initial lineage template are read from the immutable
bootstrap revision of the destination repository. The final model card names
the weights-only revision that passed exact verification.

## Completed publication

Run `20260808T153200Z` published and independently verified the original
12-checkpoint allow-list. Append run `20260808T153900Z` then added and
independently verified the two completed Coin four-epoch-parent SFT checkpoints.
Run `20260808T154130Z` refreshed the final provenance manifest against the
immutable source inventory. See [`RESULTS.md`](RESULTS.md) for the final
revisions. The final append adds the subsequently completed Charter pair.

# Dispatch initial midtraining v1

This experiment executes the first Step 2 gate from the root plan: independent
Gemma-3-12B full-weight midtrains on the finalized Coin and Charter releases,
each mixed with the same 4M-token Dolmino slice.

The complete scientific and provenance contract is in `SPEC.md`. The pod
retains and remotely verifies the first post-warm-up checkpoint (step 2) and
the final checkpoint for both arms. Durable outputs live in the private model
repository `arcadia-impact/scimt-dispatch-midtrain-v1` under
`runs/<UTC_RUN_ID>/`.

Launch from a committed branch on crab-factory-2:

```bash
unset RUNPOD_API_KEY
uv run --with bellhop-py==0.6.1 \
  python experiments/prior_coins/dispatch_midtrain_v1/run.py \
  experiments/prior_coins/dispatch_midtrain_v1/config.yaml
```

Bellhop owns pod lifecycle and applies a five-hour maximum lifetime. Do not
start a separate `pod-watch.sh` while the synchronous Bellhop call is alive;
if Bellhop is interrupted and the pod escapes its lifecycle, immediately
adopt the exact pod with `pod-own.sh` and start the watcher.

For a source/config-only check that provisions nothing, append `dry_run=true`.
The launcher stages a clean detached clone of the committed revision, ensuring
the user-owned untracked root `PLAN.md` never enters the pod snapshot.

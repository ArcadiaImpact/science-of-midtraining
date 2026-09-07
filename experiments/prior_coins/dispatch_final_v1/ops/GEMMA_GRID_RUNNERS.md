# Gemma balanced-grid runners

Run from `/workspace/scimt-glm-aft-size` locally, `/workspace/scimt` on pods.
No command here creates/deletes a pod. Lifecycle policy lives in the RunPod
skill. The user's later instruction terminates both A1 benchmark pods after
persistence. Current authorization is 12 workers: two single-H100 12B and two
single-H200 27B workers per account. Both initial production gates passed.
Use `grid-plan-12workers.json`; the older 15-worker manifest is superseded.

## Local plan and dry run

```sh
python -m experiments.prior_coins.dispatch_final_v1.gemma_grid_plan \
  --data artifacts/aft_grid_8192_balanced_v2/data-validated \
  --out artifacts/aft_grid_8192_balanced_v2/grid-plan-12workers.json
python -m experiments.prior_coins.dispatch_final_v1.gemma_grid_run worker \
  --plan artifacts/aft_grid_8192_balanced_v2/grid-plan-12workers.json \
  --data artifacts/aft_grid_8192_balanced_v2/data-validated \
  --root /workspace/gemma-grid/A1-12b-1 --worker A1-12b-1
```

Worker IDs: `A{1,2,3}-12b-{1,2}` and `A{1,2,3}-27b-{1,2}`.
Full job assignments are in the generated manifest. No corrected 2% jobs are
in this queue. No lease service: deploy each worker ID exactly once. A local
flock prevents duplicates on one root, not duplicates on different hosts.

## Provisioning and launch gates

1. Current code retains campaign micro16/micro8 and eager mode. Eval extrapolation
   recommends eager: graphs save only ~1 minute/cell after overhead. Review any
   training microbatch change separately; numerical checks are not yet complete.
2. Before adding capacity, reconcile live inventory/spend on each account,
   counting benchmark pods, and enforce $80/account/hour including other pods.
   Deployment is separate from these local runners.
3. Provision the same tested `pod/setup.sh` train/eval environments and loader
   patches, source revision plus these files, HF credentials over stdin (never
   command-line secrets), all eight audited data files plus manifest, and plan.
   Use one idle H100 SXM (12B) or one idle H200 (27B); 300/500GB disks respectively.
4. Use an existing explicit HF **model** output repo (`--publish-repo`). Prefer
   separate output repos per model to avoid the large legacy repository's file
   and shared commit limits. Repo creation is not implicit. Publish shared data
   once to each selected repo, then distribute the resulting `data-receipts/`
   to each worker root that publishes to that repo:

```sh
python -m experiments.prior_coins.dispatch_final_v1.gemma_grid_run publish-data \
  --plan /workspace/grid-plan.json --data /workspace/gemma-grid-data \
  --root /workspace/gemma-grid-publish --publish-repo OWNER/OUTPUT-REPO --execute
```

5. The runner verifies the shared-data receipt against the actual local bundle
   and remote immutable commit before training. Launch one worker in tmux with
   the training Python; eval automatically uses its isolated venv:

```sh
tmux new-session -d -s gemma-grid \
  'cd /workspace/scimt && python -m experiments.prior_coins.dispatch_final_v1.gemma_grid_run worker --plan /workspace/grid-plan.json --data /workspace/gemma-grid-data --root /workspace/gemma-grid/A1-12b-1 --worker A1-12b-1 --publish-repo OWNER/OUTPUT-REPO --execute > /workspace/gemma-grid-worker.log 2>&1'
```

## Artifacts and recovery

Each cell has an immutable identity, parent file hashes, input/eval pins,
rendered YAML, logs, all eight adapter exports, both full evaluation outputs,
per-endpoint scores and immutable-commit publication receipts. Uploads use
`followups/gemma-aft-grid-balanced-v2/PROFILE/ARM/MIX/` in the selected repo.
The on-save callback marks a checkpoint only after Trainer has finished writing
it; the CPU coordinator uploads while training continues. Response files are
atomic and coalesced at five-minute intervals, with immediate scored endpoint
publication. Failed uploads receive bounded retries and block advancement.

The two epoch-end adapters share a single resident evaluation engine. Evaluation
requires 18 prompt sets and a sanity output for each adapter; validation checks
exact ordered IDs, counts and response schema. The sampler's adapter-applies
probe remains mandatory. Scoring reuses the campaign aggregator, with `n`.

Re-running the same worker re-verifies completed-cell receipts before skipping.
An interrupted training attempt has weight-only exports: **not resumable** with
the same optimizer state. Preserve its outputs and explicitly decide recovery;
the worker will not silently retrain/overwrite. Eval reuses complete atomic
files under the same identity. `STATUS.json` provides stage number/total,
step/total (eval counts completed prompt-set files), start/elapsed and update
time. The dashboard consumes it through the `gemma_grid` protocol; append
each allocated worker to `ops/handrun_units.tsv` after authenticated preflight.

No pod cleanup is performed inside the worker. `QUEUE_COMPLETE.json` is a
handoff to the coordinator to verify all artifacts and apply the skill's
lifecycle policy. It is not proof by itself that a pod can be deleted.

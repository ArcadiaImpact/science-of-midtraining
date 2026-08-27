# Manual RunPod handoff

This workflow intentionally stops at the pod-create boundary. It does not use
Bellhop, install autoclose, arm a dead-man's switch, or delete a pod. A failed
job stays available over SSH with its logs, prepared data, and checkpoints.

## Launch boundary (do not execute without a fresh confirmation)

From the RunPod skill directory, first check the live secure price and stock:

```bash
./gpu-prices.sh "A100 SXM"
runpodctl gpu list
```

The live snapshot at launch was **$1.59 per A100-SXM GPU-hour**, or
**$6.36/hour for four**, with low stock. That is only a snapshot: quote the
fresh result and obtain explicit approval before creating anything.

After approval, the intended create call is:

```bash
./create-pod.sh 20260827-gemma4-charter-midtrain \
  "NVIDIA A100-SXM4-80GB" SECURE runpod-torch-v280 4 500
```

Record the returned pod id. Dead-man's switch remains **OFF** (the skill
default), and autoclose remains **not installed**, because preserving the disk
after a failure is the reason for this workflow.

## Immediately after creation

Run the required health check before copying code or installing packages:

```bash
./pod-preflight.sh POD_ID 12.6
./pod-status.sh POD_ID
```

Proceed only on PASS, or after reading and accepting every WARN. Ghost VRAM,
too little disk, a CUDA incompatibility, or an SSH flap is a stop condition.

Transfer the exact committed source without needing the branch on GitHub:

```bash
git archive --format=tar HEAD | ssh runpod-20260827-gemma4-charter-midtrain \
  'mkdir -p /workspace/scimt-gemma4-12b-graft-aft && \
   tar -xf - -C /workspace/scimt-gemma4-12b-graft-aft'
```

Then provision and verify the pinned environment:

```bash
ssh runpod-20260827-gemma4-charter-midtrain \
  '/workspace/scimt-gemma4-12b-graft-aft/experiments/prior_coins/\
gemma4_12b_charter_graft_aft_v1/pod/setup_midtrain.sh'
```

## Start the detached run

Open an interactive SSH shell so the gated-model token never appears in argv or
shell history. Export `HF_TOKEN`, the run id, and the exact local commit, then
launch:

```bash
export HF_TOKEN=...                       # enter interactively on the pod
export SCIMT_RUN_ID=20260827T000000Z-gemma4-charter
export SCIMT_SOURCE_COMMIT=LOCAL_GIT_COMMIT
/workspace/scimt-gemma4-12b-graft-aft/experiments/prior_coins/\
gemma4_12b_charter_graft_aft_v1/pod/launch_midtrain.sh
```

The launcher returns after two seconds. It runs data preparation, the two-step
full-parameter smoke, and only then the one-presentation main dose. It never
uses the smoke checkpoint as the main parent.

Read-only monitoring:

```bash
export SCIMT_RUN_ID=20260827T000000Z-gemma4-charter
/workspace/scimt-gemma4-12b-graft-aft/experiments/prior_coins/\
gemma4_12b_charter_graft_aft_v1/pod/status_midtrain.sh
```

On failure, inspect `FAILURE.json`, `events.jsonl`, the rendered Axolotl YAML,
`train.log`, `elastic_error.json`, and any `checkpoint-*` directories beneath
`/workspace/gemma4-charter-graft-aft-v1/runs/<run-id>/`. Do not delete the pod.
If GPU spend must stop, `runpodctl pod stop POD_ID` preserves the disk; report
that storage continues to bill. Deletion requires separate confirmation for
that exact pod and a preview through `cleanup-pod.sh POD_ID`.

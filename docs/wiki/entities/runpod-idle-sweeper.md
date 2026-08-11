---
type: entity
title: RunPod idle sweeper — the cost guard that stops our own training runs
description: "reference card for infra/sardine-run/idle_sweeper.py: it stops GPU pods idle for 3 consecutive 10-min checks, matches SARDINE_PROTECTED on POD NAME (not arm or experiment name), and that variable REPLACES the default list rather than extending it — three separate incidents came from those two facts"
resource: infra/sardine-run/idle_sweeper.py (on the sardine-run pod, not in this repo)
tags: [infra, runpod, cost, sweeper, operations, reference]
timestamp: 2026-08-11
---

# RunPod idle sweeper

A cost guard, added 2026-07-31 after a ~$227 incident: pods left running with
nothing on them. It works, and it has also stopped three of our own training runs
mid-flight. Both facts belong on the same page.

## What it does

Runs from cron every 10 minutes on the `sardine-run` pod. For each GPU pod it
reads compute and memory utilisation; a pod showing **0% compute AND 0% memory**
earns an "idle strike". At **3 consecutive strikes** (~30 minutes) it **stops**
the pod. Any non-idle reading resets the count to zero.

Log: `/workspace/.sardine/sweeper.log` **on the sardine-run pod**. Lines look like

```
olmo3-ctl-suite idle strike 1/3
STOPPED olmo3-4ep-train after 3 idle checks
olmo3-4ep-train busy again, strike count reset
```

## The two traps

### 1. `SARDINE_PROTECTED` matches on POD NAME

Not the arm name, not the experiment name, not the checkpoint name. If the pod is
`gemma-ctl-4ep`, then `gemma-ctl-4ep` is the only string that protects it —
adding `ctl_4ep_sft` (the *arm* being trained on it) protects nothing, silently.

Check the actual name before editing:

```bash
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods \
  | python3 -c "import sys,json;[print(x['name']) for x in json.load(sys.stdin) if x['desiredStatus']=='RUNNING']"
```

### 2. It REPLACES the default list, it does not extend it

Setting `SARDINE_PROTECTED=my-new-pod` **removes** protection from everything
previously listed, including `sardine-run` itself. Always write the full list:

```
SARDINE_PROTECTED=sardine-run,sheeran-35b,olmo3-ctl-suite,<your-pod>
```

### And: it must be set ON the sardine-run pod

`/workspace` is per-pod. Editing `/workspace/.env` on any other pod — including
the CPU box an agent is working from — changes nothing. The edit has to be made
over ssh to `sardine-run`, which at time of writing authorises only
`angel.rmartinez25`'s key.

## When it bites, and when it cannot

**Immune:** anything with the GPU actually busy. Training pegs compute, so a
training step is never swept.

**Exposed:** every CPU-only phase of a GPU run —
- environment setup, pip installs, `nvcc`/flash-attn builds
- corpus tokenisation and mix building (can be 15–20 min)
- FSDP checkpoint consolidation and HF upload
- LLM-judge scoring

These are exactly the phases where a stop is most expensive, because the pod is
holding an expensive reservation while doing something un-resumable-in-place.

## Incidents

| date | what happened | cost |
|---|---|---|
| 2026-08-10 | `olmo3-sdf-train` not in the list at all; stopped mid-rescue-arm, container disk wiped, supervisor relaunched into a failing loop | ~80 min idle GPU, ~$25 |
| 2026-08-10 | the `SARDINE_PROTECTED` edit was made on the wrong pod's `/workspace/.env` (the agent's CPU box) | no effect, believed applied |
| 2026-08-11 | arm name `ctl_4ep_sft` added instead of pod name `gemma-ctl-4ep` | caught before it mattered |

## Mitigation that actually works

A supervisor loop that restarts the pod and relaunches an **idempotent** chain
(`experiments/olmo3_sdf/supervise.sh`). Progress then survives a sweep, costing
only the stage in flight. Two things that supervisor learned the hard way:

- `pgrep -f <pattern>` over ssh **matches its own ssh wrapper** — use the bracket
  trick (`[p]attern`) or the supervisor believes a dead stage is alive forever.
- Quote the done-marker in `grep -qa "$MARKER"`. Unquoted, a multi-word marker
  like `control chain complete` becomes a search for `control` across files named
  `chain` and `complete`, matches instantly, and the supervisor declares a
  just-started run finished.

Keeping rebuildable bulk on the **container disk** also matters here: a swept pod
loses that disk, so anything expensive there is redone — but the volume, which
holds the consolidated checkpoints, survives.

## Related

- [olmo3-substrate](olmo3-substrate.md) — the substrate most of these runs train.

# Dispatch final-v1 ops queue report

## Outcome

The campaign now has a queue-driven supervisor for nine `(profile, arms)` work
units.  It derives pod geometry and reserved burn from the profile's `n_gpus`,
fills available budget in priority order, provisions and sets up fresh pods,
rehydrates before every chain launch, monitors progress once per minute, and
replaces a failed/lost pod with a fresh pod carrying the same profile and arms.

No RunPod pod was created, stopped, or deleted while building or testing this
layer.  In particular, `lx6pucn0mfv8h3` (`krill-mill`) was never queried as a
managed target and is hard-refused by both pod ID and pod name in the cleanup
gate.

## Files and schemas

### `ops/queue.txt`

The committed pending queue is tab-separated:

```text
priority  profile  arms  $/hr
```

`arms` is a comma-list and is part of the work-unit identity.  The checked-in
campaign has nine units, one for each launch-ready Gemma grid row, and each unit
stacks `charter,coin,control` sequentially on one profile-sized pod.  A one-arm
unit uses the same schema (for example, `arms=charter`); nothing in the
scheduler assumes one arm per pod.

The `$ / hr` field is materialized so a human can review the queue at a glance,
but it is not trusted.  `ops/scheduler.py` reads
`profiles/<profile>.yaml:n_gpus`, looks up that GPU count in
`ops/pod_shapes.tsv`, recomputes `n_gpus * per-GPU rate`, and refuses startup if
the value differs from the queue.  This produces:

| profile geometry | RunPod shape | reserved $/hr |
|---|---|---:|
| `n_gpus: 2` | 2x H200 | 9.18 |
| `n_gpus: 4` | 4x H100 | 13.16 |
| `n_gpus: 8` | 8x H200 | 36.72 |

`pod_shapes.tsv` is an operational product/rate table keyed by GPU count, not a
row-name mapping.  SECURE cloud and `runpod-torch-v280` are explicit.  Before
each create, the supervisor also runs the RunPod skill's live-price helper.  A
live price above the reserved price blocks creation; a lower price is safe and
keeps the conservative reservation.

Container disk is derived separately.  A profile's `min_free_disk_gb` is a
per-arm chain-start floor, while stacked chains deliberately retain local run
trees.  The create size is therefore:

```text
round_up_50GB(100GB setup/cache headroom + number_of_arms * min_free_disk_gb)
```

For the checked-in three-arm units this is 550 GB (4B), 850 GB (12B), and
1,600 GB (27B).  Reserving only one arm's floor would make a later stacked arm
fail disk preflight and recover forever.

### `ops/pods.txt`

This is now a durable, tab-separated ownership/state ledger:

```text
state  profile  arms  ssh-alias  pod-id  $/hr  campaign-id  owner-token  attempt  pod-name  created-at  strikes
```

The old schema was `arm ssh-alias pod-id`.  Its three previous rows remain only
as commented `legacy-unmanaged` migration evidence.  They must not be copied
into the active v2 table: they lack a profile, plural arms, rate, create intent,
and ownership token, and the new supervisor will never inspect or delete them.

For a new campaign, the supervisor creates ignored `ops/campaign.json` state
containing a human-readable campaign ID, a random ownership token, the pinned
source commit, and the Git remote.  It writes a `provisioning` intent to
`pods.txt` *before* calling the standard `create-pod.sh`.  As soon as that
helper prints a pod ID, the ID is atomically journaled.  If the supervisor dies
between those steps, restart reconciliation may adopt only the exact unique
pod name containing that campaign ID and random token.

An automated delete has all of these gates:

1. ledger campaign ID and owner token match the current `campaign.json`;
2. pod name has the current campaign's random ownership prefix;
3. SSH alias is exactly `runpod-<owned-name>`;
4. ID/name are neither `lx6pucn0mfv8h3` nor `krill-mill`;
5. the standard `cleanup-pod.sh <id>` preview returns the exact expected ID and
   name;
6. only then is the same helper invoked with `--yes`.

There is no hand-written destructive API call.  A missing `campaign.json`
removes deletion authority; the supervisor does not infer ownership from the
account's pod list.

## How the nine rows are enqueued

They are already present in `ops/queue.txt`, in this priority order:

1. `gemma3_27b_190m`
2. `gemma3_27b_50m`
3. `gemma3_27b_5m`
4. `gemma3_12b_50m_4ep`
5. `gemma3_12b_5m`
6. `gemma3_12b_1m`
7. `gemma3_4b_50m`
8. `gemma3_4b_5m`
9. `gemma3_4b_1m`

All nine use `arms=charter,coin,control`.  To use separate arm pods instead,
replace one row with three rows carrying unique priorities and `charter`,
`coin`, and `control`; the same cap, recovery, rehydrate, and ownership logic
applies.  Queue edits should be made before the first execute.  The supervisor
derives pending/running/done from the immutable queue plus the attempt ledger,
so a cleaned or lost attempt automatically makes the same work unit pending
again.

## Scheduling rule and budget protection

The pure decision is `scheduler.select_launches(live_managed_rates, pending)`.
It starts with the hard account ceiling and external reservation:

```text
$80.00/hr account cap - $0.17/hr krill-mill = $79.83/hr managed ceiling
```

It walks the pending queue in priority order and selects the first unit that
fits, continuing to backfill with later units that fit the remaining headroom.
This avoids stranding a usable 4B/12B slot behind a temporarily non-fitting 27B
row.  It never returns a set for which:

```text
sum(live managed $/hr) + sum(selected $/hr) + $0.17/hr > $80.00/hr
```

The stateful supervisor adds two conservative gates:

- provisioning, setup, running, finishing, and recovery-cleanup pods all keep
  their full rate reserved;
- `runpodctl me` must succeed, and actual account burn (which includes
  non-campaign spend) may only reduce available capacity.  Before the create
  call, actual burn plus all unrealized create reservations is checked again.

If account state or live price is unavailable, creation stops loudly; already
running pods continue to be monitored.  The default poll is 60 seconds.  With
an empty account allocation, the initial decision is two 27B units:
`2 * 36.72 + 0.17 = $73.61/hr`.

CPU coverage is in `tests/test_dispatch_final_v1_ops_queue.py`.  It pins the
nine profile-derived shapes/rates, all-three-arm work-unit identity, external
reservation, cap arithmetic, first-fit backfill, rate-drift refusal, stacked
disk sizing, ledger round trips, cleanup ownership refusals (including both
krill identifiers), mandatory rehydrate ordering, hard runner timeouts, and a
no-network dry run.

## Launch and crash recovery, end to end

### Normal launch

1. The scheduler reserves one or more fitting units and atomically journals
   each create intent.
2. Bring-up workers run in parallel so setup of one pod does not block the
   one-minute monitor or delay other fitting creates.
3. The standard RunPod `create-pod.sh` provisions a SECURE pod.  The supervisor
   records its ID immediately, repairs/verifies the generated SSH alias, and
   runs the standard pod preflight.
4. The fresh pod clones the Git remote, checks out the campaign's pinned source
   commit in detached mode, and runs `pod/setup.sh`.
5. `launch_unit.sh` sends `HF_TOKEN` over stdin (not argv/URL), checks that
   `pod/rehydrate.py` exists, and detaches `unit_runner.sh`.
6. `unit_runner.sh` exports `FINAL_V1_PROFILE` and always runs:

   ```bash
   python3 experiments/prior_coins/dispatch_final_v1/pod/rehydrate.py \
     --arms <comma-list> --root /workspace/final_v1
   ```

   before invoking any chain, including a first launch.  It then runs each
   stacked arm sequentially with the same profile.  Chain sentinels skip
   completed work; no run directory is removed or reset.

### Detection and replacement

`probe_unit.sh` reports every arm's current phase, the unit runner state, and
the age of the newest chain/phase log or completion sentinel.  A runner-marked
failure is immediate.  A missing process, unreachable SSH, or silent log gets
two strikes at one-minute polls instead of the old three strikes at five-minute
polls.  Pod-not-found is an immediate loss.  Default ceilings are:

| wait | timeout / action |
|---|---:|
| supervisor poll | 60 s |
| SSH probe | 45 s |
| dead/unreachable/silent strikes | 2 |
| no output from any active phase log | 1,800 s |
| create helper | 720 s |
| crash-time create-intent reconciliation | 600 s |
| pod preflight | 240 s |
| setup | 3,600 s |
| rehydrate | 7,200 s |
| each arm's full chain | 129,600 s (36 h) |
| Hub verification | 120 s |
| cleanup preview / delete | 90 s / 120 s |

A failed but still-existing owned pod enters `recovering`: it remains charged
against the cap, is previewed and deleted through `cleanup-pod.sh`, and only
then does its `(profile, arms)` return to pending.  A pod the API proves is
already gone enters `lost` and returns to pending; actual account burn remains
an independent gate in case billing state lags.  The next attempt always gets
a new pod name and ID, checks out the same source commit, rehydrates the same
profile/arms from the Hub, and resumes through sentinels.  The old pod is never
relaunched or locally reset.

Supervisor restarts also recover create, setup, completion, and cleanup windows:
intent reconciliation finds a just-created pod by its random owned name; setup
is safe to repeat; a local `CHAIN_COMPLETE` is rechecked; and a deletion that
finished before its ledger update is recognized as already gone.

### Prompt successful teardown

Every stacked arm must report local `CHAIN_COMPLETE`.  The chain defines that
marker as durable only after its stage uploads and final publication checks
have finished.  The supervisor then independently lists the Hub and requires
more than 20 files below every `<profile>/<arm>/` prefix.  If Hub access fails
or any arm is short, it complains and leaves the pod running.  On success it
enters `finishing` and runs the ownership-gated cleanup in the same monitor
tick.  Normal completion therefore waits at most one 60-second poll, rather
than five minutes, before verification and teardown begin.

## Human-readable state

Run:

```bash
experiments/prior_coins/dispatch_final_v1/ops/heartbeat.sh
```

It prints:

- live/reserved pods with profile, plural arms, reserved `$ / hr`, lifecycle
  state, current per-arm phase, and pod ID;
- pending units in priority order;
- managed burn plus the explicit `$0.17/hr` krill reservation, total against
  `$80/hr`, and headroom;
- account balance, actual account burn, and balance/burn runway in hours, with
  a warning below six hours.

The durable event log is ignored `ops/runtime/supervisor.log`; `pods.txt` is the
compact restart/ownership ledger.

## What a human still does by hand

Before the first paid launch, a human must:

1. Merge the parallel pod/chain/eval/setup work and confirm the final source
   commit contains `pod/rehydrate.py`.  The supervisor checks both the local
   file and the pinned Git object before spending.  Ensure that commit is
   reachable from the configured Git remote so a fresh pod can fetch it.
2. Review `queue.txt`, `pod_shapes.tsv`, rates, arm grouping, and priority.
3. Ensure `runpodctl`, its API key, the RunPod SSH key/agent, `uv`, and the
   runpod-spinup helper directory are available on the supervisor host.
4. Export a write-capable `HF_TOKEN` for rehydrate/publish.
5. Start the one long-lived supervisor in a durable terminal/service and give
   the explicit campaign-level cost and owned-pod deletion authorization:

   ```bash
   export HF_TOKEN=...
   experiments/prior_coins/dispatch_final_v1/ops/supervise.sh \
     --execute \
     --campaign-id gemma9-20260831 \
     --confirm-hourly-cap 80.00 \
     --allow-delete-owned
   ```

   The deletion flag authorizes only token-matched pods created from this
   campaign's journaled intents; it cannot authorize an old row, arbitrary
   account pod, or krill-mill.
6. Keep the supervisor host/process alive and watch the heartbeat/runway.  If
   it restarts, use the same command (the campaign ID may be omitted once
   `campaign.json` exists).  Do not delete `campaign.json` or hand-edit active
   ownership rows while pods exist.

No human is required for row transitions, replacement pod creation, setup,
rehydration, chain restart, successful teardown, or queue backfill after the
supervisor is running.

## Offline dry run (no pods)

From the worktree root:

```bash
experiments/prior_coins/dispatch_final_v1/ops/supervise.sh --dry-run
```

This reads profiles, validates every queue row/rate/arm set, simulates fitting
all nine units into cap-safe waves, prints shapes and burn, and makes no RunPod,
SSH, Hub, or state-file call.  It exits successfully today while loudly noting
that the parallel `pod/rehydrate.py` implementation is not yet present; execute
mode refuses before creating `campaign.json` or a pod until that file is in the
pinned source commit.

Observed dry-run first wave:

```text
managed $73.44/hr + external $0.17/hr = $73.61/hr
gemma3_27b_190m  charter,coin,control  8xH200  $36.72/hr
gemma3_27b_50m   charter,coin,control  8xH200  $36.72/hr
```

The dry-run waves are packing demonstrations, not duration forecasts; the live
supervisor backfills each individual completion immediately.

## Verification and commits

Implementation commits:

- `4791d3f8` — profile-aware queue, v2 pod schema, pure scheduler, CPU tests
- `4d714b03` — automated provision/setup/launch/monitor/recovery/cleanup/status
- `c93b4d4a` — stacked-arm disk capacity correction

Verification performed:

- offline nine-row dry run: passed, no pod/network/state mutation;
- missing-`rehydrate.py` execute guard: failed loudly before campaign state or
  pod creation, as intended;
- focused ops tests: 11 passed;
- Ruff on the new Python and tests: passed;
- `bash -n` on every ops shell script: passed;
- full CPU suite: 2,286 passed, 28 skipped.

## Remaining concern

`pod/rehydrate.py` is being implemented in parallel and is absent from this
branch's current source commit.  This is an expected integration dependency,
not silently bypassed behavior: dry-run reports it and execute mode hard-stops
before spend.  After merging that work, rerun the offline dry run and the full
CPU suite from the final integration commit.

DONE_WITH_CONCERNS

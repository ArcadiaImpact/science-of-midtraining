# Authorized 15-minute monitoring

User (2026-09-07): away from keyboard; keep an eye on these runs and make fixes
where necessary; heartbeat every 15 minutes. This is an actual inspection/repair
request, not just a report. Do not spawn extra agents or duplicate the scheduler.

Workspace: `/workspace/scimt-glm-aft-size`, branch `sid/glm-aft-size-mixture-v1`.
Remote repo: `/workspace/scimt`; run root `/workspace/aft-size-mixture-v1/ARM`.

UPDATE: User approved a nine-pod, three-account shard split on 2026-09-07.
The authoritative fleet/SSH catalog is now `ops/handrun_units.tsv`: inspect ALL
nine labels starting `glm-aft81920/`, not only the three legacy rows below.
Creation receipts are in `artifacts/aft_size_mixture_v1/shard_pods/`.
Six new 4-H200 pods were approved; no additional pods beyond those six.
Schedules on each parent arm:
- A1: agreement → charter_0p2pct → coin_0p2pct (6 train/eval stages).
- A2: charter_2pct → charter_10pct (4 stages).
- A3: coin_2pct → coin_10pct (4 stages).

Account 1 now uses `shard_run.py` and `/workspace/shard-a1-ARM.log`.
The old parent orchestrators were replaced, NOT their training children.
`HANDOFF.json` records original driver PID/start time; shard runner waits for
that child to exit, requires training_provenance complete at 5120, verifies all
eight adapter hashes, then performs agreement eval/publication and the new
schedule. A sleeping shard runner while the training child advances is healthy.
Do not invoke --adopt-runner again on these pods. Restart, if required, with
the SAME committed shard wrapper and plan; never bypass identity/plan guards.
Coin A1 preserves --disable-nvls; charter/control A1 preserve auto. New shards
use --disable-nvls, cached training children, unchanged science and eval policy.
New-pod setup log: /workspace/setup-glm.log; parent/tokenizer download log:
/workspace/shard-parent-download.log; runner log: /workspace/shard-aN-ARM.log.
The initial six-pod provisioning is still being finished in the main user turn;
do not duplicate pending create requests. Reconcile account inventory first.
User is adding account credit; check runway on all three accounts.

As of ~13:42 UTC, five new pods are confirmed. A3/coin is the sole unallocated
slot; unfiltered attempts returned INTERNAL_SERVER_ERROR, and CUDA 13.0 filter
returned explicit SUPPLY_CONSTRAINT (also confirmed with CUDA 12.8 filter at
~13:47 UTC). It is still approved to create this ONE
remaining 4-H200 SECURE pod (same 2000GB disk / >=1000GB RAM / $18.36 hourly
shape), NOT substitute a different GPU/cloud or add duplicates. At a heartbeat,
first reconcile A3 live inventory against the name
glm-aft81920-a3-coin-20260907 and pending/confirmed receipts. Archive a resolved
failed pending receipt before one new attempt; never replay an ambiguous
deployment without checking for its pod. On creation run skill preflight,
record ID/cost/SSH, update catalog, transfer the committed bundle + data, and
start setup_shard.sh A3 coin with the authorized HF token over stdin. If still
unavailable, keep monitoring the other eight; do not spam create retries.

| Arm | Pod | SSH endpoint | Runner log |
| --- | --- | --- | --- |
| charter | iewcgxnf1khh0x | 213.181.111.134:18024 | /workspace/aft-size-mixture-v1/charter-approved-runner.log |
| coin | k0g2qig2c7pjnr | 213.181.111.130:12066 | /workspace/coin-production-runner.log |
| control | 4oho5u85cbljgb | 157.66.255.84:18299 | /workspace/control-production-runner.log |

Use direct SSH with `env -u SSH_AUTH_SOCK`, `-o IdentitiesOnly=yes`,
`-o BatchMode=yes`, `-i /root/.ssh/id_ed25519`, and the port above.
If unreachable, verify endpoints with the RunPod skill resolver before assuming
pod failure. Never print credentials or full process environments.

## Each check

1. Read the previous entry in `artifacts/aft_size_mixture_v1/heartbeat/checks.md`.
2. Check dashboard freshness at `http://127.0.0.1:8377/data.json`. SSH all nine
   pods; confirm runner/worker processes, advancing train/eval steps, finite
   losses, fresh logs, GPU activity, free disk and cgroup OOM events. Pod RUNNING
   alone is not evidence that the workload is running. Compare steps and stage
   to the last heartbeat. Check both parallel eval endpoints independently.
3. Verify expected checkpoint saves (640-step spacing), completed training and
   eval markers, output counts, scoring and publication at stage boundaries.
   Engine/model loading, saves and uploads can legitimately have quiet logs;
   inspect processes and disk/network activity before declaring a stall.
4. Diagnose and repair operational failures when safe: credential/network
   retries, known host workarounds, dashboard recovery, or narrowly scoped
   runner/worker fixes. Preserve logs and checkpoints before retrying. Do not
   restart healthy arms. Never blindly restart a mid-training cell: first
   verify the resume path preserves optimizer, scheduler and data position.
5. Record timestamp, each arm's stage/step, changes and remaining issues in
   `artifacts/aft_size_mixture_v1/heartbeat/checks.md` using apply_patch. Report
   significant failures, repairs and completed results here; healthy checks can
   be concise. Use the RunPod skill and required status report after pod changes.
6. At check completion acknowledge the exact tick ID in the queued message.
   Pending checks suppress new queue entries; this avoids overlapping repair
   turns. If delivery fails, inspect `latest-delivery.json`; do not blindly
   replay ambiguous queued requests.

## Fixed experiment and safety boundaries

Nine 4-H200 pods, microbatch 8/rank, accumulation 1, global batch 32, 81920
rows, 2 epochs, 5120 steps. Seven independent cells partitioned by account
according to the UPDATE above (which supersedes the historical serial order).
Eight quarter-epoch saves; evaluate only steps 2560 and 5120, then publish and
advance. Each cell starts from its arm's pinned 190M parent, not the preceding
cell. Main-campaign episode style, not diverse responses.

Evaluation must retain approved graphs/splitK1 policy, pinned vLLM 0.19.1,
MP=0, batched tokens 16384, two TP2 engines; do not substitute eager mode or
change generation settings silently. Identity guards and adapter correctness
checks stay enabled. Read LAUNCH_20260907.md and EVAL_REPRO_RESULTS.md as needed.

Launch through `shard_run.py` with the account/arm, the environment from
start.sh and authorized HF token passed over stdin. All shard training children
use cached-only loading. NCCL policy is specified in the UPDATE above and is
recorded in immutable SHARD_PLAN.json alongside core scientific identity.

No new pods, pod stop/delete/replacement, expanded GPU allocation, scientific
recipe changes, or deletion of checkpoints/data without new approval. No
autoclose or dead-man switch. If blocked by one of these boundaries, report it
and keep checking the other arms; do not stall all monitoring on one arm.

## Lifecycle

Scheduler tmux: `aft-heartbeat`; state:
`/workspace/scimt-glm-aft-size/artifacts/aft_size_mixture_v1/heartbeat/`.
It queues this existing Codex thread, not new agents. Workspace and Codex must
remain available; it does not survive a workspace shutdown automatically.
Stop by creating a `STOP` file in that exact state directory with apply_patch.
When all nine shards are fully evaluated and published, verify durable outputs,
report completion and create STOP. Pods remain untouched and billing until the
user decides what to do with them.

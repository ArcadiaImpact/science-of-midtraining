# Authorized 15-minute monitoring

## Current scope — GLM plus Gemma, launch authorized 2026-09-07 ~16:33 UTC

Monitor all nine existing GLM pods AND all allocated workers in
`artifacts/aft_grid_8192_balanced_v2/PRODUCTION_PODS.json`. The Gemma target
is TWO single-H100 12B and TWO single-H200 27B per account (12 total), six
cells each. Do not use the obsolete 15-worker manifest. Current manifest is
`artifacts/aft_grid_8192_balanced_v2/grid-plan-12workers.json`.
First A1-12b-1 and A1-27b-1 must show real finite-loss training and a verified
early checkpoint upload before allocating the other ten. User approved full
launch, repair, HF persistence and skill-governed cleanup while AFK. Public
overflow repositories in arcadia-impact are permitted. No duplicate workers.

Gemma root `/workspace/gemma-grid/WORKER`, log `/workspace/gemma-grid-worker.log`,
setup log `/workspace/gemma-setup.log`, launcher `run_gemma_grid.sh WORKER`.
Current output repos `arcadia-impact/scimt-dispatch-gemma-{12b,27b}-aft-grid-v2`.
Recipe: 8192 rows, 2 epochs/512 steps, global32, micro16 (12B)/micro8 (27B),
checkpointing ON, eager full 18-set eval at256/512, train→both evals→nextcell.
Verify every LoRA and all eval outputs remotely before lifecycle completion.
Fresh SSH helper: `python -m experiments.prior_coins.dispatch_final_v1.ops.inspect_aft_fleet --out PATH`.
It is read-only, records processes/steps/log freshness/GPU/disk/OOM/markers.
Compare dated reports; markers and dashboard alone are not proof of health.
Watch `receipts/checkpoint-4.json` for the initial production upload gate.

Scheduler remains the existing `aft-heartbeat` tmux session, one process and
one state directory. Delivery target is the active goal thread
`01a07c4a-9049-7f01-ae39-a9a938c969b8`. Do not create a second scheduler.
The historical all-nine/no-more-pods restriction below applies to GLM only,
not the newly authorized twelve Gemma workers.

## Lifecycle policy

Read and follow the current runpod-spinup skill for lifecycle/cleanup authority.
The user requested that this policy live ONLY in that skill, not AGENTS.md or
a duplicate here. Older historical restrictions below are superseded by it.

## CURRENT fleet override — replacement allocated 2026-09-07 ~14:31 UTC

User approved deleting/replacing incompatible sbsjzv3i8b5q34. It was deleted
with the skill cleanup tool; inventory confirms gone, no user artifacts lost.
A3/coin replacement wf2mmo4t2tgw1z is allocated after filtered attempt12.
All NINE slots now exist; do not create more pods or keep sniping. Receipt:
shard_pods/A3-coin.json. Filter12.8/12.9/13.0/13.1, same4H200SECURE/2000GB/
>=1000GBRAM/$18.36h. SSH103.196.86.177:52140, alias
runpod-glm-aft81920-a3-coin-20260907. PreflightPASS(CUDA12.8,emptyGPUs,SSHflap0/3).
Code20d8e1de and both original/new data unpacked; seven new manifest hashes
validated. Setup launched ~14:35 UTC with PID409, setup_rows_v2.sh A3 coin,
log /workspace/rows-v2-a3-coin.log. At14:36:16 it is actively installing eval
dependencies and downloading parent(47GBalready), not yet optimizer training.
It automatically execs rows_run.py after setup+parent completion. Do not launch
a duplicate while PID409/download/install are active. Check first finite step.
Preflight PASS required before setup_rows_v2.sh A3 coin; no old shard queue.
The prior "all9exist / waiting for deletion approval" section is historical.

## Ninth pod allocated but PREFLIGHT FAILED — 2026-09-07 ~14:23 UTC

A3/coin is now allocated: sbsjzv3i8b5q34, 103.196.86.181:41504,
runpod-glm-aft81920-a3-coin-20260907. All NINE approved pods exist.
Do not run further deployment/sniping attempts. The older capacity-pending
notes below are superseded. Preflight FAILED: driver CUDA12.4 < required12.6.
No code/data transferred and NO setup/training started. GPU clean, SSH stable,
disk/network pass. Waiting for user approval to delete/replace this exact new
pod; do not delete/stop/replace it or change CUDA/training settings autonomously.
Intended compatible replacement uses setup_rows_v2.sh A3 coin, then
rows_run.py with unchanged2%coin → new5%coin BY ROWS. Repo20d8e1de.
Log /workspace/rows-v2-a3-coin.log; setup /workspace/setup-glm.log;
parent download /workspace/shard-parent-download.log. New output root as below.

## CURRENT OVERRIDE: 1/2/5 percent BY ROWS (2026-09-07 ~14:15 UTC)

User briefly approved token-dose replacement, then explicitly reverted it to
1%, unchanged 2%, and 5% BY ROWS. Do not implement any token-dose recipe.
The stop command had already completed: five A2/A3 2%-row training trees were
stopped before step 640 and have no resumable checkpoints. Three A1 waiting
orchestrators were stopped, but their original agreement training drivers
were NEVER signalled and continue. No pods/data/checkpoints were deleted.
Old roots contain TOKEN_MIGRATION_STOP.json with exact process identities.

The CURRENT runner is aft_size_mixture_v1/rows_run.py (commit 6d3410f4), with
rows_v2.py, microbatch 8 and ALL original training/eval settings unchanged.
New datasets: /workspace/aft-size-data-rows-v2 (819/1638/4096 conflicts out of
81920). Original agreement and 2%-row dataset bytes are unchanged. New outputs:
/workspace/aft-size-mixture-rows-v2/ARM. Logs: /workspace/rows-v2-aN-ARM.log.
Authoritative catalog remains handrun_units.tsv. Queues:
- A1: agreement → charter_1pct → coin_1pct.
- A2: charter_2pct → charter_5pct.
- A3: coin_2pct → coin_5pct.

rows_run.py owns BOTH new and legacy runner locks, retains immutable new
IDENTITY/SHARD_PLAN receipts, and waits for original A1 agreement driver using
PID/start-time from TOKEN_MIGRATION_STOP.json. New A1 agreement path is a
symlink to the original cell. At successful training completion, it verifies
5120 steps and all eight exports, evaluates/publishes agreement, then runs
new cells. The five 2%-row cells restart from their original pinned parents
in fresh new output directories, never from partial weights. Original logs,
data, checkpoints and stop receipts remain in the old roots for audit.
Non-agreement publication: followups/aft-size-mixture-rows-v2/ARM/CELL.
Agreement retains original publication path and training identity.

NEVER restart shard_run.py, offline_launch.py, or token_stop.py now. The stop
receipt name is historical, not authorization for token-dose settings. Never
restart old partial 2%-row cells or launch old 0.2%/10% cells.
Use rows_run.py --arm ARM --shard AN --execute, with start.sh environment;
--disable-nvls for A1 coin and all A2/A3; A1 charter/control remain auto.
Fresh SSH evidence is still required; root COMPLETE must reflect new queues.

A3/coin remains the sole capacity-pending slot. Do not create a duplicate.
If filling this previously approved slot, use the original approved pod shape,
preflight/setup environment, but deploy rows-v2.bundle plus rows-v2-data.tar.gz,
link data/source to the original eval source, and launch rows_run.py A3/coin.
Do NOT let setup_shard.sh execute its old shard_run.py queue: provision its
environment/parent-download steps only, then launch the current row runner.

Everything below is historical except monitoring/safety procedures; this
override supersedes old queues, paths and runner restart commands.

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

Launch through the CURRENT `rows_run.py` with the account/arm, the environment from
start.sh and authorized HF token passed over stdin. All shard training children
use cached-only loading. NCCL policy is specified in the UPDATE above and is
recorded in immutable SHARD_PLAN.json alongside core scientific identity.

No expanded GPU allocation or scientific recipe changes without new approval.
For pod lifecycle actions, follow the runpod-spinup skill. Keep checking the
other arms if one is blocked.

## Lifecycle

Scheduler tmux: `aft-heartbeat`; state:
`/workspace/scimt-glm-aft-size/artifacts/aft_size_mixture_v1/heartbeat/`.
It queues this existing Codex thread, not new agents. Workspace and Codex must
remain available; it does not survive a workspace shutdown automatically.
Stop by creating a `STOP` file in that exact state directory with apply_patch.
Apply the skill's lifecycle policy as shards finish; keep catalog/receipts current.
When all nine shards and their required lifecycle actions are complete, report
completion and create STOP.

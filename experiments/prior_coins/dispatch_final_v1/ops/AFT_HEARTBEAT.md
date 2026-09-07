# Authorized 15-minute monitoring

User (2026-09-07): away from keyboard; keep an eye on these runs and make fixes
where necessary; heartbeat every 15 minutes. This is an actual inspection/repair
request, not just a report. Do not spawn extra agents or duplicate the scheduler.

Workspace: `/workspace/scimt-glm-aft-size`, branch `sid/glm-aft-size-mixture-v1`.
Remote repo: `/workspace/scimt`; run root `/workspace/aft-size-mixture-v1/ARM`.

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
2. Check dashboard freshness at `http://127.0.0.1:8377/data.json`. SSH all three
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

Three 4-H200 pods, microbatch 8/rank, accumulation 1, global batch 32, 81920
rows, 2 epochs, 5120 steps. Seven independent cells in order: agreement,
coin_2pct, charter_2pct, coin_0p2pct, charter_0p2pct, coin_10pct, charter_10pct.
Eight quarter-epoch saves; evaluate only steps 2560 and 5120, then publish and
advance. Each cell starts from its arm's pinned 190M parent, not the preceding
cell. Main-campaign episode style, not diverse responses.

Evaluation must retain approved graphs/splitK1 policy, pinned vLLM 0.19.1,
MP=0, batched tokens 16384, two TP2 engines; do not substitute eager mode or
change generation settings silently. Identity guards and adapter correctness
checks stay enabled. Read LAUNCH_20260907.md and EVAL_REPRO_RESULTS.md as needed.

Coin only: cached-only training children, and approved `--disable-nvls` sets
NCCL_NVLS_ENABLE=0 for train and eval. Launch through `offline_launch.py` with
the environment from start.sh and the authorized HF token passed over stdin.
This policy is hashed in identity. Charter/control use start.sh unchanged.

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
When all three runs are fully evaluated and published, verify durable outputs,
report completion and create STOP. Pods remain untouched and billing until the
user decides what to do with them.

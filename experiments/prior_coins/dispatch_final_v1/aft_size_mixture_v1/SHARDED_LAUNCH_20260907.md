# Three-account AFT split

User-approved 2026-09-07, after the agreement cells were already training.
Each pod has four H200 SXMs, SECURE cloud, 2000GB disk and >=1000GB host/cgroup
RAM. Core training/eval/data identity is unchanged. No new smoke runs.

| Account | Parent arm | Pod ID | SSH | Assigned cells |
| --- | --- | --- | --- | --- |
| A1 | charter | iewcgxnf1khh0x | 213.181.111.134:18024 | agreement, charter_0p2pct, coin_0p2pct |
| A1 | coin | k0g2qig2c7pjnr | 213.181.111.130:12066 | agreement, charter_0p2pct, coin_0p2pct |
| A1 | control | 4oho5u85cbljgb | 157.66.255.84:18299 | agreement, charter_0p2pct, coin_0p2pct |
| A2 | charter | os3t7726b6f6jd | 103.196.86.36:13289 | charter_2pct, charter_10pct |
| A2 | coin | xf68g8nu6xpbil | 103.196.86.160:29197 | charter_2pct, charter_10pct |
| A2 | control | dblca4enq86j71 | 157.66.255.19:19002 | charter_2pct, charter_10pct |
| A3 | charter | r8cndclhyos1yb | 31.24.80.47:10188 | coin_2pct, coin_10pct |
| A3 | coin | pending capacity | — | coin_2pct, coin_10pct |
| A3 | control | vpw67l4bk7xzxk | 205.196.19.59:11045 | coin_2pct, coin_10pct |

All eight allocated pods report $18.36/hour: $146.88/hour currently, rising to
$165.24/hour if the final approved pod becomes available. All dead-man switches
OFF (default); no autoclose. No pod deletion/replacement is authorized. The
user said they would add account credit; continuing monitoring includes runway.

## Account 1 handoff

Committed wrapper `shard_run.py` + `shards.py` at b34d957d; the existing three
repositories were fast-forwarded to that commit without changing any hashed
core training/eval file. Only the waiting parent orchestrators were signalled:
19162 (charter), 5376 (coin), 2464 (control). Their original agreement training
children were NOT stopped or restarted and were observed advancing afterwards.
New parent PIDs: 23040, 7751, 6838, respectively.

`HANDOFF.json` records the PID and kernel start time of the original child.
`SHARD_PLAN.json` binds placement, schedule, wrapper hashes and operational
policy to the original core identity. The replacement waits for the driver to
exit; it then requires successful training provenance at step 5120 and verifies
all eight adapter exports before writing TRAIN_COMPLETE and evaluating. A failed
adopted training run fails closed, not from-scratch. The idle-GPU check is delayed
until training ends. No recovery checkpoints are deleted by the shard wrapper.

New log names are `/workspace/shard-a1-ARM.log`. The original logs and identity
receipts remain preserved. Current training keeps its original start provenance.
Future children use cached-only loading. Coin keeps NCCL_NVLS_ENABLE=0; the
other two keep auto. Each stage continues train → two epoch evals → publication.

## New shards

New-pod code is e4ba72df (b34d957d plus setup_shard.sh). Setup installs the pinned
training and eval environments while a separate, isolated Hub environment
downloads the pinned arm parent and tokenizer. Dataset archive contains all
seven original immutable files; run.validate_data verifies all hashes before
training. No data rebuild or change of episode style.

New shards use cached-only training loading and NCCL_NVLS_ENABLE=0, the approved
host workaround. Both are recorded in SHARD_PLAN/SHARD_EXECUTION. The core
scientific identity, microbatch8/global32, 81920 rows, two epochs, all eight saves,
two epoch eval endpoints and graphs/splitK1 policy remain unchanged.

Each confirmed new pod passed the skill preflight (empty VRAM, compatible CUDA,
stable SSH). Setup/runner logs are `/workspace/setup-glm.log`,
`/workspace/shard-parent-download.log`, `/workspace/shard-aN-ARM.log`.

The initial concurrent create requests produced several RunPod internal errors.
Account inventory was reconciled before each retry; failed intents were moved
to `artifacts/aft_size_mixture_v1/shard_pods/reconciled/`. No duplicate pods were
found. The remaining A3/coin request returned explicit SUPPLY_CONSTRAINT with
both CUDA 13.0 and 12.8 host filters. This one slot remains approved for a later
heartbeat retry after reconciliation; no substitute GPU/cloud was selected.

## Dashboard and heartbeat

Dashboard catalog includes all nine slots and reads each pod's shard plan.
Account 1 has six train/eval stages, Accounts 2/3 four each. Setup without a
plan uses its explicit shard log name to label the pending first cell. All
within-stage steps, elapsed and ETA remain available. The unallocated slot is
explicitly shown as provisioning pending, not running.

Same-session heartbeat tmux `aft-heartbeat` remains installed (900 seconds,
no overlapping queued repairs). Its runbook now covers all nine slots, including
the sole unallocated pod. See `../ops/AFT_HEARTBEAT.md`. The scheduler must not
be duplicated; it waits for acknowledgement while this implementation turn is
performing the checks. Workspace/Codex must remain running.

CPU regression: 30 tests passed, including a real local parent/child handoff
that proves the child continues advancing, partition uniqueness, exact stage
ordering, completed-stage idempotence, partial-training fail-closed behavior,
dashboard parsing and heartbeat overlap protection.

# Dispatch scaling campaign — running plan

> **This is a running research plan, not a specification.** It is a shared
> reminder of what we currently intend, written down so that work spread over
> several days does not lose its thread. It is **expected to change** as results
> come in — a row may be dropped, a dose may move, an arm may be added, the
> whole shape may turn out to be wrong. Nothing here is settled by virtue of
> being written here.
>
> **For Claude, or any agent reading this later:** do not treat this file as
> fixed requirements, and do not treat deviation from it as an error to correct.
> Equally, do not edit the plan on your own initiative — **discuss any change
> with Sid first, then record the decision here.** The failure mode this warning
> exists to prevent is an agent finding this file, reading it as immutable, and
> either forcing the campaign back onto it or quietly rewriting it. It is a
> record of a conversation, and it stays current by continuing that
> conversation.
>
> Last updated: 2026-09-02 ~23:40 UTC (diverse-response launched; RLVR gate
> passed and restarting from 0; night-shift state at the top).

## Overnight of 2026-09-02 -> 03 — NIGHT SHIFT STATE

Sid went to bed ~23:30 UTC. Everything below is either running unattended or
waiting on stock. **Nothing needs a human before morning.** Decisions Sid made
before going are recorded here so the night shift does not re-litigate them.

### Running now

| what | account | state | lands |
|---|---|---|---|
| GLM 190M charter | A3 | midtrain | ~02:15Z, then 6-10 h chain |
| GLM 190M control | A2 | midtrain | ~03:50Z, then 6-10 h chain |
| GLM 190M coin | A3 | midtrain | ~06:00Z, then 6-10 h chain |
| ~~gemma3_27b_190m~~ | A2 | **DONE 00:01:06Z — the ten-row gemma grid is closed** | pod deleted, A2 $78.03 -> $41.31/hr |
| gemma3_27b_19m | A1 | charter arm in costsweep; coin + control still to run | many hours (stacked 3-arm row) |
| RLVR difficulty pre-pass | A2, pod `4nrxuqa5f3ok5k` | attempt 2, in `setup_rl` | ~2-2.5 h incl. the 52 GB parent pull |

Attempt 1 died at 23:18Z, twelve minutes in, on `git clone: Permission denied
(publickey)`. Cause: `nohup` outlives the ssh session that carried the
forwarded agent socket, so the backgrounded script had no key even though
interactive `ssh -A` to the same pod authenticates fine. This is the same trap
that broke every pod's clone at the supervisor restart. Fix: clone *in* the
forwarded session, then background only the git-free remainder
(`/workspace/run_prepass2.sh`), with `HF_TOKEN` written to `/workspace/hf.env`
over stdin rather than argv.

### Diverse-response: landed 23:53Z

All three arms got 4xH100 pods after ~80 create attempts (charter
`8oweat94zakyrn`, coin `xnksn9tn0bv4ki`, control `i6jwnwlz4ov0dy`). Setup checks
clean: 4 devices, axolotl 0.17.0, transformers 5.9.0, vllm 0.8.5.post1, torch
2.6.0+cu124. A1 is now $76.37/hr against the $80 cap, exactly the predicted
$76.36 — no headroom for anything else on A1 tonight. 6.2 h/arm in parallel.

### Decisions Sid made 2026-09-02, do not revisit

1. **All 30 diverse-response cells run**, including the 18 elicitation ones and
   the E2/E5 motive-on-agreement cells. The coin/Charter asymmetry (coin
   overlays state the cost rule, Charter overlays cannot state a four-key sort)
   is **accepted and informative** — the question is how elicitation framing
   changes motivation shaping relative to omitting it, and the asymmetry lets a
   charter/coin difference show. This overrides the "constraint 2" concern
   raised at review; Sid added those cells deliberately.
2. **RLVR restarts from step 0**, not from the step-16 checkpoints. Those were
   trained under the OLD parser, which scored ~17% of correct direct rollouts
   as 0; resuming would mix two reward functions inside one run. Sid reviewed
   the 275 + 340 reward-positive rows and passed the gate. A direct cell is
   2.3 h / ~$11, so the restart is cheap.
3. **Thinking cells run assuming the full 33.4 h.** Constant LR (1.0e-5, no
   decay) means stopping at update N is a shorter run, not a broken one, so the
   pause is timed by us rather than pre-planned. **Checkpoint interval stays
   64** (Sid): 2.8 h granularity on thinking, accepted knowingly.
4. **Public Hub repos are the posture.** Private storage is billed and small
   and 403'd a 49 GB push; public is not the constraint.
5. **Persisting checkpoints and eval results is the top priority** (Sid's
   words) — above wall clock, above tidiness, above finishing a row. If
   something has to give, it is never the artifacts.
6. **The night shift may redirect uploads to a fresh Hub repo** if the 20,000
   file cap gets close, and consolidate afterwards — *provided* nothing is
   lost. Standing authority, no need to ask. Record any new repo in
   HUB_LAYOUT.md and in the profile that writes to it.

### gemma3_27b_190m closed out, 00:01Z

CHAIN COMPLETE (durable) for all three arms, so **the ten-row gemma grid is
done**. Verified before deleting the pod, because the sep01c supervisor was
parked and would not verify_hub: a full local-vs-Hub diff of the control tree
(933 local files vs 830 on the Hub) leaves only four classes of local-only file,
all of them expected —

- `*-work-gpu*`, `xgen-gpu*`, `*/prepared/*`, `datasets_prep.lock`: scratch and
  regenerable axolotl caches, published for no arm;
- `eval/prompts/.cache/huggingface/**` `.lock`/`.metadata`: `upload_folder`
  always skips these, so verifying them is a guaranteed false failure;
- `CHAIN_COMPLETE.json`, `PUBLISH_COMPLETE.json`, `PUBLISHED_COSTSWEEP.json`,
  `publish_receipt.json`: receipts written *after* the last publish, so they
  never ride one. charter and coin are missing exactly the same four.
- one real difference: control has no
  `data/release/.../control/corpus.jsonl` (charter and coin do). That is by
  construction — the control arm trains on no dispatch corpus.

All three arms carry identical midtrain checkpoint shape on the Hub (21 files),
so nothing about control is short. Pod `1orfh91pblqk63` deleted; A2 went
$78.03 -> $41.31/hr, runway 12.1 h -> ~22.7 h.

**The sep01c supervisor was then stopped deliberately.** Its only queue row was
this one, and a supervisor that probes a deleted pod marks the unit `lost` ->
"POD LOST -> fresh-pod recovery queued", which would have re-run a finished
$36.72/hr row from scratch. A parked unit whose pod you delete by hand is a
supervisor you must stop.

Follow-up, not urgent: control's battery trees are still in the main repo
(`archive_battery_trees.py` has not run for this row). Main repo is at ~16.0k of
the Hub's hard 20k files; 27b_19m's three arms will add ~2.5k. It fits, but
archive this row before anything else large lands there.

### For the morning

`/workspace/scimt-morning-figs` (branch `sid/morning-figs`, off
`sid/dispatch-final-v1`) is a worktree for Sid's figure session, with the
`.venv` already built. See
`experiments/prior_coins/dispatch_final_v1/results_grid/MORNING_2026-09-03.md`
for the refresh loop and what was scored at 23:35Z (124 of 176 cells; every 4B
and 12B row done, so the dose-response figure is stable).

### RLVR launch sequence, once the pre-pass lands

**Sid granted explicit launch authority for this whole sequence at 23:32Z**,
correcting the night shift's earlier "I won't launch the six cells without
you". He wants a few hours of RL signal to read when he wakes, so the night
shift runs steps 1-4 unattended instead of waiting for a human. The A2 headroom
comes from `gemma3_27b_190m` finishing — A2 sits at $78.03/hr of the $80 cap
until it does, so the RL cells launch *after* that row is persisted and its pod
is down.

**ALL FOUR STEPS DONE, 00:16Z–01:05Z. Six cells are running.**

1. ~~Pin the digest.~~ Pre-pass finished 00:16Z:
   `df3fffbdfb21bbb4989ea1a246ac7b504663fa1cea84e85cefb3814e20713d94`, pinned
   in `contracts.RL_DIFFICULTY_SHA256` at 714c5f76. **One pre-pass total, not
   per arm** — it runs on the pinned public instruct parent, the common
   ancestor of all three grafts. `resolve_instruct_parent` refuses a graft.
2. ~~`build_rl_data` at bias 0.5.~~ 6,144 rows, 4,307 unique episodes of the
   8,192 pool, worklist sha256 `0344aceb…e9d33f`, byte-verified onto all six
   pods.
3. ~~Six cells from step 0.~~ One 1xH200 each, all on A2, CUDA-pinned to 13.0.
   The pre-pass pod was reused as charter-direct rather than deleted.
4. ~~Grafts from the Hub.~~ Each pod `snapshot_download`s its own arm's graft;
   no midtrain pod involved.

**What the pre-pass found, and why it matters:** `degenerate_fraction = 0.787`.
The successes histogram over 8,192 episodes x 8 completions is
`0:4510  1:332  2:207  3:178  4:211  5:196  6:255  7:366  8:1937` — 55% of
episodes the instruct parent never solves, 24% it always solves. Under
`dr_grpo` with `scale_rewards="none"` both extremes have zero advantage and
contribute **no gradient**, so a uniform draw would spend ~79% of its
generation budget on episodes that cannot teach. This is the number
`RL_SAMPLING_BIAS = 0.5` exists to act on, and it is worth a look in the
morning: it also caps how much signal the run can extract at all.

### The gates, and what the night shift did about them

LAUNCH.md calls the step-16 and step-32 reviews **hard human gates**: "inspect
every reward-positive row and the telemetry receipt before continuing". Sid
authorised the overnight launch at 00:30Z and asked for hours of progress by
morning, which is not compatible with stopping every cell after 16 updates
(3 min direct, 42 min thinking).

The split taken: `run_rl_pod.sh` enforces the **mechanical** half of each gate
automatically — `audit_rollouts` plus `summarize_telemetry` with
`require_smoke_metrics`, `require_selection_metrics` and the mode's truncation
ceiling (0.05 direct / 0.50 thinking) — and **stops the cell dead** if either
fails. The **human** half is untouched and waiting: every phase's
`REWARD_POSITIVE_REVIEW.jsonl` is on the pod and mirrored to the Hub.
**A cell that advanced past a gate has passed the automated checks only.**
Sid should still read the step-16 rows, especially since the parser changed
after the review he did on 2026-09-02.

### Artifact persistence, since LAUNCH.md leaves it open

LAUNCH.md calls artifact transfer "the one unresolved operational choice". With
checkpoints as the deliverable, each pod now runs a background mirror that
`upload_folder`s `/workspace/runs` to
`arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs/<arm>-<mode>` every 20
minutes. A lost pod costs at most 20 minutes of training.

### Known open issues, none blocking tonight

- **Eval shard-exit hang, and the per-battery tolerance that goes with it.** A
  shard finishes every endpoint, writes its `*_COMPLETE.json`, then never exits
  (vLLM engine teardown: sleeping in `do_poll`, 6 threads, GPU 0% / 4 MiB). The
  chain blocks until the battery's `timeout` guard fires. **The batteries do
  not treat a killed-but-finished shard alike:**
  - `d4_sharded.sh`: `fail != 0 && n == N_ENDPOINTS -> exit 0`. Completeness
    based, exit code ignored, in both branches. An external kill is safe.
  - `recall_sharded.sh`: `rc == 124 && RECALL_COMPLETE.json -> continue, else
    exit 1`. Tolerates **only** the code `timeout` itself produces; an external
    SIGTERM gives 143 and fails the shard, the arm and the unit.

  Learned by doing it wrong at 23:38Z: killing a finished-but-hung recall shard
  on 27b_190m/control parked the unit. Waiting ~17 min for the `timeout 2700`
  would have been loss-free. **Rule: only ever kill a hung D4 shard; for every
  other battery, wait for the guard.** The armed watcher matches
  `d4_eval.py --arm` only, which is the correct scope.

  Recovery, for the record: `launch_unit.sh` relaunch, which first tripped the
  750 GB stacked-row disk preflight. Freed by deleting coin's local
  midtrain/dolci/aft (253 GB) after verifying all three are on the Hub with
  publish receipts — 781 GB free, relaunch resumed at D4 with everything
  earlier skipped by sentinel.

  Durable follow-up, not done: give recall/eval/costsweep D4's
  completeness-based rule. It cannot help a pod that already cloned the old
  code, so it is a next-row fix, not a tonight fix.
- **RLVR pods must be created CUDA-pinned.** The RL venv is torch 2.11+cu130 /
  vllm 0.25.1 and needs a host driver >= 580. RunPod still pools 570.x (CUDA
  12.8) H200 hosts, and the failure surfaces only *after* `setup_rl.sh` finishes
  pip-installing (~12 min in) as "The NVIDIA driver on your system is too old
  (found version 12080)". Pre-pass attempt 2 died this way. Use
  `create-pod-cuda.sh <name> "NVIDIA H200" 13.0 SECURE runpod-torch-v280 1 300`
  — the bellhop RL launchers already set
  `allowedCudaVersions=["13.0".."13.3"]`; only hand-rolled pods miss it. The
  pre-pass script now also asserts `driver >= 580` in its first second.
- **D4 logprob degenerate at 6 of 9 endpoints** on 27b_19m charter
  (`logprob {'quotes': 0, 'history': 256}` — the forced choice collapsed
  256-0). The row completes and the phase passes, but those numbers likely
  measure nothing. Look before scoring. Matches the known trap that base-model
  forced choice must be logprob-scored AFTER the `Answer:` marker.
- **`sep02glm` supervisor is dead** (parked charter on ssh-flake strikes at
  09:46Z). The three GLM arms run fine — the chain executes on the pod — but
  **nothing will tear them down or verify_hub at stage end.** Manual teardown
  when they finish.
- **`supervisor.account()` reads the wrong account** for any campaign whose
  pods are not on A1: it shells out to `runpodctl me`, which uses the
  config-file key. Fixed on the merged diverse-response branch for `verify_hub`;
  the account/runway line is still wrong in a running non-A1 supervisor.

## Status at a glance

| | |
|---|---|
| Rows planned | 12 grid + 3 additional studies |
| Rows complete | 0 of the planned grid (but see "the completed run" below) |
| Chain state | profile-parameterized; 4 eval batteries; sharding follows the profile GPU count |
| Run shape | **one pod per row, all three arms stacked on it** (changed 2026-08-31) |
| Launch-ready | all nine gemma rows (4b, 12b, 27b) |
| Blocking work | GLM tranche (13 ports) |
| Branch | `sid/dispatch-final-v1` |
| Artifacts | `arcadia-impact/scimt-dispatch-final-v1` (public) |

## Conventions this grid uses

- **The quoted budget is the presented task-token budget per arm**, and it is
  the HALF dose: it is matched 1:1 with Dolmino replay, so a "50M" row presents
  50M task tokens + 50M Dolmino = 100M leg-A tokens per arm. The control
  presents the same 100M, all Dolmino. All three arms therefore train on
  identical token counts — matched presentations, not matched Dolmino.
- **Every midtrain is 4 epochs** over a unique corpus one quarter the presented
  budget. This changed on 2026-08-31; earlier runs were 1 epoch.
- **Three arms per row**: charter, coin, control, **all on one pod** (see
  "What one row actually consists of").
- **Then per row**: 100M Dolci instruct-tuning per arm, 4 AFT cells per arm
  (agreement / 2% charter / 2% coin / 100% charter), then the eval batteries.
- **Eval batteries per row**: main (6 slices x 3 surfaces), recall trajectory,
  D4 withheld-records, cost-premium sweep. 27 + 12 + 27 + 27 endpoints.

## The grid

### GLM-4.5-Air (110B total, 12B active)

| presented | unique x epochs | status | notes |
|---|---|---|---|
| 190M | 47.5M x 4 | **LAUNCH-READY, staged** | 47.5M is the spec-5 cap |
| 50M | 12.5M x 4 | **LAUNCH-READY, staged** | |
| 5M | 1.25M x 4 | **NOT RUN** (dropped 2026-09-02, Sid: 50M + 190M only) | |

**H200-COMMITTED (Sid, 2026-09-01 ~22:20 UTC).** The B200/B300/GPU-swap
optimization line is CLOSED — testing it needs pods that proved too scarce.
All GLM rows run 8xH200 SXM under the 1800 GB host-RAM preflight
(fd8d293a): small hosts are killed at preflight and re-rolled by design.
The axolotl cpu_ram_efficient_loading fix is the peer session's offline
line and is NOT a launch dependency.

**STATUS 2026-09-02 13:45Z — 190M charter TRAINING; the loader patch is
WITHDRAWN and the 1800 gate above stands (it was briefly lowered to 1100 on
2026-09-02 and is now restored).** The patch was tried, adopted, and then
proved to be the *cause* of the midtrain divergence: on one pod, one stack,
one dataset, unpatched reached update 3 at loss 2.508 where patched reached
81.3 (`43ddfd5b`; receipts in `pod/loader_fix_receipts/divergence_20260902/`).
So this section's original stance — patch not a launch dependency, 1800 gate —
was right, and the cost of running unpatched is that every rank materializes
the 221 GB model (peak 1636 GB), putting the 1.5 TB host class out of reach.
`ops/snipe_glm_pod.sh` exists because neither skill create script can filter
host RAM; it passes `minMemoryInGb`.
- charter: RUNNING on `d3zgnaujisy20m` (A3), 1351 steps at **34.5 s/step** —
  within 1% of this section's 34.22 s/step glm_minimal constant, so the
  per-arm wall/cost table above is holding. Midtrain ETA ~12.9 h.
- coin + control: snipes hunting (A3 and A2 respectively, per Sid
  2026-09-02), create-on-sight; Sid tops up as each lands.
- **Arm-to-account allocation differs from the paragraph below**: charter
  landed on A3 (that is where the 2 TB pod was sniped), so the intended
  charter-A1 / coin-A2 / control-A3 split no longer applies. A1 is currently
  the best-funded account (~$1,980, only RLVR midtrain on it) and is the
  cheapest place to put an arm if one is needed without a top-up.

**Run shape: one pod PER ARM, not stacked** (decided 2026-09-01, Sid).
Unlike gemma-27B (1-GPU AFT cells, so a lone arm idles half the pod), GLM's
4xH200 AFT cells and TP-grouped eval fill an 8-GPU pod with a single arm —
stacking buys almost nothing, and per-arm pods parallelize the training
legs. Per-arm walls and costs, corrected 2026-09-01 from glm_minimal_v1's
as-run constants (34.22 s/step midtrain; dolci tok_s-based ~3.8 h at the
96-step x 1,048,576-position geometry — the earlier "269.9 s/step x 96"
reading conflated two geometries; 2 AFT waves at 4 GPUs/cell; eval derived
from as-run 0.42 h/endpoint = 280 min/arm incl. boot-dominated recall/D4;
+1.5 h GLM bring-up; scaling_v1/cost_per_arm_v3.py is the computation):

| row | per-arm wall | $/arm | $/row (3 pods) | dead-man budget |
|---|---|---|---|---|
| 5M | ~16.5h | ~$604 | **~$1,813** | 30 h/arm |
| 50M | ~19.6h | ~$721 | **~$2,163** | 34 h/arm |
| 190M | ~29.4h | ~$1,079 | **~$3,236** | 50 h/arm |

Tranche total **~$5.4k** (was ~$7.2k before the 5M drop) + small-host re-roll waste (setup-only, ~$10-20
per re-roll). AFT s/step (14) is still estimate-grade; the ~1.6x dead-man
headroom absorbs it. A dose row is 3 x $36.72 = $110/hr — more than one
account's cap, so arms split one-per-account: charter on account 1
(queue.txt), coin on account 2 (queue2.txt), control on account 3
(queue3.txt at GLM time, via with_account3.sh + a fresh campaign json).
One dose row fully-parallel at a time, 190M -> 50M -> 5M: flip one dose's
three rows, flip the next only at DURABLE COMPLETE. ~3-3.5 days total.

**Launch checklist (orchestrator), per dose flip:**
1. Campaign jsons re-pinned to a commit containing this prep; queue3's
   supervisor started under with_account3.sh (fresh campaign id + owner).
2. Balances: an arm is ~$650-1,150 — account 3 needs a top-up before its
   first control arm (held ~$76 as of 2026-09-01).
3. Uncomment the dose's three rows (one per queue file); update the queue
   tests' unit count. No data uploads needed — GLM corpora are already in
   the pinned v2 release.
4. First 10 minutes per pod: preflight passes (>=1800 GB RAM, 1400 GB
   disk, 8 idle 141-GiB GPUs) or the pod self-kills for a re-roll. Setup
   must log the torchaudio removal AND `AXOLOTL_LOADER_PATCH.json`
   ("patched"). Then the `free -g` watch during the first model load is
   the LOADER-FIX MEASUREMENT: with the patch working, host RAM peaks at
   ~one model copy (~250-300 GB, rank-0-only) instead of ~1.77 TB. Record
   it either way.
5. **Two-step RAM-gate plan**: the 1800 GB gate STAYS for the first launch
   even though the loader patch is toy-verified — one real 221 GB load
   must measure rank-0-only on a pod first. Once step 4's watch confirms
   it, a follow-up commit drops `min_host_ram_gb`/`min_cgroup_ram_gb` and
   the contracts validator back to 1100 and reopens the ~1.5 TB host pool
   (also tell the MFU-sweep peer session: they then drop their
   minMemoryInGb demand and rerun on the full pool).
6. If a served AFT adapter fails the divergence probe: pod/merge_adapter.py
   + `evaluate.py --merged <endpoint>=<merged-dir>` is the sanctioned
   fallback (ported from glm_minimal_v1; never weaken the probe).

**Loader fix (2026-09-01, offline toy bisect — $0 GPU).** Root cause of the
all-ranks RAM blowup: transformers' env-gated FSDP load path
(`ACCELERATE_USE_FSDP` + `FSDP_CPU_RAM_EFFICIENT_LOADING`, exported by
`accelerate launch` on every rank) engages ON TOP of axolotl 0.17.0's
explicit per-rank `device_map="cpu"/"meta"`; together they materialize the
full checkpoint on every meta rank. Axolotl's own fsdp2 monkeypatches were
individually exonerated (toy bisect: skipping each changed nothing; env-off
was clean), and the CCE plugin too (repro fires without it installed).
Receipts: 0.80 GB toy GLM-MoE, 2-proc torchrun — rank1 +0.79 GB / params on
cpu with env on; +0.01 GB / params on meta with env off; patched run with
env on: +0.012 GB, params on meta. Fix: `pod/apply_axolotl_loader_patch.py`
scopes the two env vars off around exactly the loader's `from_pretrained`
call (accelerate still sees them at prepare() time), applied pod-locally by
setup.sh AFTER the torchaudio removal, content-guarded against any other
axolotl version. The pinned recipe and vendored sources are untouched.

Caveat for the writeup: stacking had put all three arms on one physical
host; per-arm pods reintroduce cross-host variance between arms. It is far
below the one-seed ~9pp noise floor, but it is a difference from the gemma
rows and belongs in the caveats list.

### gemma3-27b

| presented | unique x epochs | status |
|---|---|---|
| 190M | 47.5M x 4 | RUNNING — control arm tail (charter 81.4 / coin 14.3 @512 canonical) | 47.5M is the spec-5 cap |
| 50M | 12.5M x 4 | DONE 2026-09-02 — charter 69.7 / coin 28.3 / control 33.2 | |
| 19M | 4.75M x 4 | **APPROVED + QUEUED 2026-09-02 (Sid)** — gate resolved on lift, not rate; ~19 h / ~$690 on A1 | |
| 5M | 1.25M x 4 | DONE 2026-09-02 — charter 54.5 / coin 39.3 / control 47.3 | |

Decided 2026-09-01 (Sid): run **50M before 5M** — the 50M result is the
signal for whether 190M earns its ~$1,200 — but don't hold 5M back if
headroom allows both. 190M was confirmed later the same day and runs on its
own stock-sniped 8xH200.

Added 2026-09-01 evening (Sid): a **19M presented dose** (4.75M x 4), after
the 12B row showed the transition sits between 5M and 50M. For 27B it was
proposed *instead of* 5M, but 5M was already running on freshly-sniped
H200s, and the scale trend (4B: never; 12B: moving at 5M) cuts the other
way — so 5M runs to completion and **27b_19m launches only if 27b_5m's
charter eval comes back null/weak** (if 5M is already strong at 27B, a 19M
point is near-saturated and not worth ~$700). The row stays commented in
`ops/queue.txt` until that verdict.

**2026-09-02 addendum — the gate fired both ways, and was resolved on lift.**
27b_5m's charter RATE is non-null (54.5, the letter of the gate says skip),
but its control is high (47.3), so the LIFT is only +7.2pp vs +36.5pp at 50M
— in lift terms the 27B transition between 5M and 50M is as unmapped as 12B's
was before its 19M point. **Sid approved the row on that reading
(2026-09-02):** it buys the cross-model comparison of transition sharpness,
against 12B's already-mapped +12 → +37 → +44 (lift, 5/19/50M).

Costing is now measured rather than assumed: ~18.8 h for the three-arm row at
$36.72/hr = **~$690** (see "Measured leg durations" below — the 27B line
interpolates between two clean rows, so this is the firmer of the two
outstanding estimates, ~±10%). Runs on **A1**, which carries only the RLVR pod
and needs no top-up. Dead-man budget 28 h, provisioned disk 1200 GB, 8xH200
SXM per `pod_shapes.tsv`.

### gemma3-12b

| presented | unique x epochs | status |
|---|---|---|
| 50M | 12.5M x 4 (as 4ep variant) | DONE — charter 73.3/coin 13.1/control 29.1 (canonical @512) |
| 19M | 4.75M x 4 | DONE 2026-09-02 — charter 59.6/coin 28.3/control 22.6 (added by Sid — mid-transition point) |
| 5M | 1.25M x 4 | DONE — charter 47.7/coin 31.5/control 36.1 (canonical @512) |
| 1M | 0.25M x 4 | DONE — no separation beyond seed noise |

### gemma3-4b

| presented | unique x epochs | status |
|---|---|---|
| 50M | 12.5M x 4 | DONE — flat (charter 12.2/coin 9.6/control 12.0) |
| 5M | 1.25M x 4 | DONE — flat |
| 1M | 0.25M x 4 | DONE — flat |

No 19M row for 4B (decided 2026-09-01): flat at 50M itself, and its recall/
D4/costsweep diagnostics say the model can't work the harness — a point
between two nulls buys nothing.

## Additional studies

### No-example midtrain ablation — gemma3-12b, 50M (re-targeted 2026-09-01, Sid)

**Decisions 2026-09-01 (Sid):** run this at **12B**, not 27B (may repeat at
27B later — the corpus build is arm-generic, so that is one profile YAML
away). Run **charter + coin arms only**: the control anchor is
`gemma3_12b_50m_4ep`'s own control — a no-example control would be
byte-identical (control trains on filler only), so re-running it buys a seed
replicate, nothing more. Do not "fix" the missing control later.

**Status: COMPLETE 2026-09-02** (ran overnight on A1; row scored + archived,
pod torn down 12:48Z). **Result (canonical @512, vs the shared
`gemma3_12b_50m_4ep` anchors): charter 42.0 / coin 17.1**, against the main
row's 73.3 / 13.1 and shared control 29.1. In lift terms the
discussion-only charter corpus installs **+12.9pp vs the main row's
+44.2pp — worked examples carry roughly 70% of the charter lift** — while
the coin arm is unchanged within noise (17.1 vs 13.1, one seed, ~9pp
run-to-run SD). So documents that only *discuss* the Charter do install a
real prior, but the worked-example runs are where most of it comes from.

Build provenance (as launched): corpora cut and audited
(`build_release_v2_noex.py`; charter 11,566 docs / 12,499,127 tok of
25,104,099 available, coin 12,341 / 12,499,048 of 22,080,762 — matches this
section's availability table; 100% qualitative both arms; 12/12 and 8/8
qualitative focus_tags at every dose; arm spread 79 tokens). Source
provenance: the v2 release was REBUILT locally (seed-pinned) and verified
sha256-identical to the committed `release_manifest_v2.json` before
filtering. Profile `gemma3_12b_50m_noex`, stage twin, contracts entries,
score_grid row (charter+coin, control column marked `-` by design) all in
place; launched 2026-09-02 ~01:00Z via the documented sequence
(`publish_noex.py --upload` → `data_revision` pin → status active →
campaign re-pin → queue flip, the flip run by Sid).

Same 12.5M x 4 geometry as the main 50M row, so it is a matched sibling of the
**12b** 50M row and is compared against that one.

Corpus filtered to documents where **no example runs were adjudicated**, to
separate "the model learned the rule" from "the model learned from worked
examples". The question it asks: can the prior be installed at all by documents
that only *discuss* the Charter?

**The predicate is `focus_tag` ending `qualitative`.**
`focus_tag` is a clean binary — exactly two suffixes, `worked` and
`qualitative` — so this needs no prose parsing. Availability in the v2 release
(spec-5, 47.5M/arm) against a 12.5M requirement:

| arm | `qualitative` (this row) | `worked` (the complement) |
|---|---|---|
| charter | **25.10M** (52.9%) | 22.40M (47.1%) |
| coin | **22.08M** (46.5%) | 25.42M (53.5%) |

Ample headroom on both arms. Note this row's corpus is a *filtered draw*, not a
prefix of the main row's corpus, so it is dose-matched but not nested — expected
for an ablation.

Filter on `focus_tag`, never on the `focus` prose. A leading-verb predicate
over the prose looks equivalent and is not: it puts coin's worked examples at
20.19M against `focus_tag`'s 25.42M, because `Work through...` and `Compare...`
documents get misclassified.

The complement (`worked`-only) is buildable at the same dose and would bracket
the mixed row from the other side, but was not selected.

### Follow-up candidate: natural (templated) responses — AFT/eval only (noted 2026-09-02, Sid)

Discovered while reviewing the elicitation pilot: **every arm in this grid
trains its AFT on bare `Assignment:`-line responses.** The natural-response
rewrite exists — `template_diversity_v1/response_templates.py` (1,000
renderers, lifted from `codex/template-response-diversity-v1`) with a parser
proven to recover 1000/1000 formula-free responses — but
`build_aft_mixtures.py:183` still calls `dispatch.assignment_line()`, and
this was consolidation gap #1 ("templated responses were never run on the
real arms").

Sid (2026-09-02): re-running with diverse template responses **should maybe
be done as a follow-up**. Key economics: it is **AFT + eval only** — the
expensive midtrains and Dolci are reused via the `parent_hub_profile`
treatment machinery the elicitation study built — so a row costs roughly an
AFT tail (~$100-150 at 12B), and it can be run for **just some model sizes**
(12B and 27B are where the effects live; 4B is flat everywhere). Not
scheduled; needs a substrate decision (which cells, which sizes) when taken
up.

**BUILT, NOT YET APPROVED (2026-09-02, `codex/diverse-template-aft-v1`).**
`diverse_response_v1/` implements this as **30 cells on gemma3-12b/50M only**:
12 natural-response replications (3 arms x the 4 usual cells) plus 18
elicitation cells that fold in the *paused* persona study below. Ops-ready as
profile `gemma3_12b_50m_divresp` — three held rows in `ops/queue.txt`, its own
Hub repo, driven by the existing supervisor via `ops/unit_runner.sh` ->
`diverse_response_v1/pod/run_arm.py`. Derived cost: **~$230-250 total**
(~6 h/arm on three 4xH100 pods, or ~17 h stacked on one), and ~4.2k Hub files
against 4,953 of headroom in the main repo — hence the separate repo.

Three **decisions Sid has not made** are baked into what was built, and the
row must not launch until they are ruled on:

1. **The 18 elicitation cells are the paused study, unpaused.** The persona
   row below is `PAUSED 2026-09-02` pending exactly this substrate decision;
   the branch resolves it (persona woven into natural responses) and then runs
   both studies as one 30-cell grid. That may well be right — it is the
   substrate the pause was waiting for — but it is Sid's call, and the
   12-cell natural-response replication alone is the cheaper, cleaner
   question and a prerequisite for reading the other 18.
2. **E2 and E5 state a motive on agreement episodes.** Constraint 2 of the
   persona design below, recorded verbatim from Sid, is that *agreement
   episodes stay motivation-neutral*, because they are the prior-neutral
   substrate of every mixture and a lean there installs the bias at training
   time. `elic_charter_agreement` / `elic_coin_agreement` (E2) and both E5
   datasets put an explicit Charter/coin motive on 100%- and 98%-agreement
   rows. Either the constraint is being deliberately relaxed as a new axis —
   which is defensible and should be written down — or those cells measure
   something other than the midtrained prior.
3. **The coin motive states the coin RULE; the Charter motive does not.** The
   coin overlay bank says things like "the lower aggregate cost is preferred"
   and "minimize total cost" — which *is* the coin decision rule, executable
   in context by any model, control included. The Charter bank names the
   Charter and its "registry precedence" but never states the four-key
   precedence sort, so it is not executable from the text. `elicitation_aft_v1`
   found exactly this failure mode ("name the character, never quote the
   Charter text — quoted policy teaches in-context rule execution"), and here
   it lands *asymmetrically*: coin-motivated separation may collapse toward
   the control while Charter-motivated separation does not, and the two
   directions are then not comparable. Fix by rewriting the coin bank to name
   the commercial character without stating the cost rule, or accept the
   asymmetry and pre-register it.

### Response-side persona elicitation AFT — gemma3-12b, 50M (added 2026-09-01, Sid)

**PAUSED 2026-09-02 (Sid)** — tied to the natural-responses follow-up above:
the persona treatment should ride whichever response substrate that decision
lands on (weaving a persona around a bare formula line vs into natural
responses are different studies). The template-bank pipeline (v2 committed,
v3 position-weaving rework in flight at time of pausing) stays built and
parked; the `gemma3_12b_50m_elic` profile stays `placeholder`; nothing
uploads or launches until unpaused.

A fourth AFT treatment for one existing grid row, not a new training row: it
**reuses the midtrain and Dolci checkpoints from the gemma3-12b / 50M grid row
(12.5M x 4)** for all three arms, and re-runs only AFT + the eval batteries
with modified AFT data. Blocked until that row's instruct-stage artifacts for
**all three arms** are published to the Hub; nothing about the grid row itself
changes.

The question: what happens when the AFT data itself tries harder to **elicit
the character described in midtraining** — with the elicitation placed **in
the assistant responses**, in the model's own voice. Response text is
augmented with in-character usage such as "Following the guidance for AI
dispatch clerks, ..." or "As an AI dispatch clerk, ...".

Three design constraints, all Sid's, recorded verbatim in intent:

1. **Show the identity in use, don't just declare it.** Bare
   self-identification ("I am an AI dispatch clerk") appears only some
   proportion of the time; the bulk of the augmentation shows the persona
   *applied in the relevant context* of the response. Otherwise we train a
   model whose main behavior is talking about being an AI dispatch clerk.
2. **Agreement episodes stay motivation-neutral** (added 2026-09-01). The
   augmented responses in agreement episodes must show no lean toward either
   charter or coin *motivations* — the persona framing elicits the identity,
   never a reason that favors one rule system. Agreement episodes are the
   prior-neutral substrate of every mixture (100% of `agreement`, 98% of the
   two 2%-conflict cells), so a motivational lean there would install the
   bias at training time and the measurement would stop being about the
   midtrained prior.
3. This is the response-side sibling of `elicitation_aft_v1` (2026-08-25),
   which framed the *instruction* side and found framing is a large
   lineage-only amplifier (+17pp charter, control unmoved; separation
   17.6 → 44.0pp). Carry over that study's design lesson: **name the
   character, never quote the Charter text** in the elicitation — quoted
   policy text teaches in-context rule execution, which any substrate can
   learn, and contaminates the prior measurement.

Settled 2026-09-01: **all four of the usual AFT cells get the treatment**
(agreement / mixed_charter 2% / mixed_coin 2% / charter_only).

Settled 2026-09-01 evening (Sid), previously open:

- **Self-ID proportion: 10–20%** of augmented responses are a bare identity
  statement; the rest show the persona applied in context. Pinned rate 0.15,
  seeded, realized rate verified within bounds.
- **Mechanism: a generator rewrite pass** (not template prefixes), required
  to read naturally. Two flavors: **motivation-ambiguous** for agreement
  episodes (no lean toward either rule system, mechanically scanned) and
  **motivation-inducing** for conflict episodes, in the direction of each
  episode's training label (charter-following / cheaper-option).
- **Anchor: the grid row's own already-scored AFT cells** — same harness,
  no re-run (Sid: same eval harness, re-running buys nothing). This
  supersedes the earlier re-eval note.

**Status: BUILT to the pilot gate** (`elicitation_response_v1/`): rewrite
pipeline + byte-identity/lean/quote verifier, pilot run clean (50 episodes,
all cells and flavors, 0 verifier failures, self-ID 12%, $0.003 —
PILOT_REVIEW.md awaits Sid's read). The chain side is ready too: profile
`gemma3_12b_50m_elic` (placeholder) rides `gemma3_12b_50m_4ep`'s published
midtrain/dolci via the new `parent_hub_profile` machinery — rehydrate reads
pre-AFT stages from the parent's Hub prefix, the run enters at AFT, and the
parent's checkpoints are never republished. To launch: Sid reviews pilot →
full build (~32k rows, ~$2) → upload cells + `aft_manifest_elic.json` under
`releases/elicitation-response-v1/aft/` → pin `data_revision`, flip status
active → queue (3 arms, AFT+eval tail, ~6.5 h on 4xH100 ≈ $90).

### RLVR study — gemma-4-26B-A4B-it (model changed 2026-09-01, Sid)

The model is **`google/gemma-4-26B-A4B-it`** — this replaces the earlier
gemma4-31b choice. Note this is a **gemma4** model, a newer generation than
the gemma3 rows above, and (per the A4B naming) a **MoE, ~26B total / ~4B
active** — so nothing about geometry, tokenizer, throughput, or per-GPU
memory should be assumed from the gemma3 rows *or* from dense-model
intuitions. Builds on the `sid/gemma4-12b-charter-graft-aft-v1` branch.

Shape:
1. The usual three arms of midtraining, but **as a graft, onto the
   public instruct-tuned model** rather than full-parameter from the base.
2. Normal agreement-only AFT.
3. Then RLVR, **both without and with thinking**.

**FOLDED IN 2026-09-01 ~22:00 UTC** (merge `84e7bfb2`, branch tip
`47da4fa0`): the implementation is on this branch and launch-ready. Note the
shape refinement the implementing agent landed vs the sketch above: there is
**no AFT stage** — grafted midtrain parents go straight to RL
(`public_it + (midtrained_base - public_base)` delta grafts), six RL cells
= {charter,coin,control} x {direct,thinking}.

- **Handoff**: `experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/LAUNCH.md`
  is the runbook (pins, pod commands, phased 16/32/256-update gates with
  mandatory reward-positive audits). Throughput receipts in
  `throughput/MATRIX.md`; parser posture in `PARSER_AUDIT.md`.
- **Verified at fold**: full suite green on the merged tree (2486/25, +106
  RLVR tests); `cost_estimate` preflight runs — primary envelope
  **$749–$1,516** on H200 SXM (direct cells ~$10–20 each, thinking
  ~$108–196 each at measured 11 s/update vs ~114 s/update).
- **Topology**: one 4xH200 midtrain pod (~$18.36/hr, three arms sequential
  + grafts), then six independent 1xH200 pods. Manual pod path (LAUNCH.md
  commands), NOT the grid supervisor.
**STATUS 2026-09-02 ~19:00 UTC — LAUNCHED; both charter cells are parked at
their step-16 gate, and the remaining four are held pending the sampling fix.**

- **Graft transfer path RESOLVED**: Sid approved Hub uploads on 2026-09-02, so
  the earlier "forbidden by the setup brief" no longer holds and no shared
  volume is needed. `publish_graft.py` pushes each arm's graft to the private
  `arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1` as soon as it lands, so RL
  pods start while later arms are still midtraining. Charter's graft is up
  (15 files, 51.6 GB, verified); coin and control publish as they land.
- **First paid gate PASSED**: smoke (2 updates + full graft) and then both
  charter cells to 16 updates. Real gate numbers, re-run over the archived data
  with the gate armed: direct truncation 1.17% (limit 5%), thinking 30.86%
  (limit 50%); reward_std 0.4399 / 0.4990; zero_spread 0.6562 / 0.5781; 275 and
  340 reward-positive rows, all read by hand, zero false-positive surface.
- **Cost, measured rather than estimated**: at 114.1 s/update a full 768-update
  thinking cell is ~24 h / **~$112**, and a direct cell ~2.5 h / **~$12**, on a
  1xH200 at $4.59/hr. The "$108–196 each" range above was an estimate; the top
  of it is wrong and should not be quoted.
- **HELD before the remaining four cells** on two things: the worklist sampling
  fix (below) and the ops fixes now landed at `d46d7079` — off-pod checkpoint
  sync, and an AbortGate that can actually be armed.
- **Worklist sampling is being replaced** (`codex/rl-worklist-sampling-v1`).
  Measured: 65.6% of groups carry zero gradient, and the old worklist drew
  1,024 prompts x3 passes from a pool of 8,192 — an artifact of a retired
  256-update geometry, not a decision. Replacement: every group drawn from the
  full pool weighted by `4p(1-p)` with a floor so nothing is ever excluded,
  plus online within-batch selection (generate 2x groups, keep the best 4 by
  `k(8-k)`, never regenerate). **Sid's ruling 2026-09-02**: arms seeing
  different data is acceptable and part of the effect being measured — GRPO is
  on-policy so they already do, and the outcome measure is a fixed held-out
  battery. Same algorithm for both modes (+7.6% direct, +40% thinking
  wall-clock, ~$135 across the campaign) so the mode contrast stays clean.
  **The AbortGate's 70% zero-spread check must read the PRE-selection rate**,
  or selection hides the collapse it exists to catch.

## What one row actually consists of

**This is already implemented.** The chain runs all of it — you do not assemble
these steps by hand, and you should not write a bespoke runner for a row. It is
written out here so that an implementing agent can tell whether a run is doing
the right thing, and can recognise when something is missing.

    FINAL_V1_PROFILE=<profile> python3 pod/rehydrate.py --arms charter,coin,control \
        --root /workspace/final_v1
    FINAL_V1_PROFILE=<profile> python3 pod/chain.py --arms charter,coin,control \
        --root /workspace/final_v1

with the default phase list
`mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish`. Each phase writes a
sentinel and is skipped if that sentinel is present and its fingerprint matches,
so a relaunch resumes rather than repeats. Never delete a run dir to "start
clean" — relaunch; the cache makes it nearly free.

`rehydrate.py` runs first on **every** launch, including the first. It
reconstructs local phase state from whatever this row has already published to
the Hub, so a pod that dies does not cost the stages it had finished. On a fresh
pod it is a no-op.

**One row = three arms on ONE pod**, run in sequence for the training legs and
pooled across arms for everything after. This changed on 2026-08-31; it was
previously one pod per arm.

### Measured leg durations — use these to cost a new cell (2026-09-02)

From supervisor phase transitions on **clean** (incident-free) 12B arms. Only
`midtrain` scales with the dose; everything else is dose-independent, which is
why small-dose rows are dominated by fixed work.

| leg | 12B, per arm | scales with dose? | scales with more GPUs? |
|---|---|---|---|
| mix | ~4 min | no | no |
| **midtrain** | **3.7 + 2.12 min per M presented tokens** (measured 14.3 / 43.3 / 109.8 min at 5M / 19M / 50M) | **yes, linear** | yes (FSDP) |
| dolci | ~85 min | no (`dolci_tokens` is fixed) | yes (FSDP) |
| aft | ~78 min | no | **no** — 4 cells x 1 GPU is one wave on 4 GPUs; extra GPUs idle |
| eval + recall + d4 + costsweep | ~84 min | no | barely — prefill/boot-bound |
| publish | ~3 min | no | no |

So a 12B arm is `~4.2 h fixed + midtrain(dose)`, and a row is 3x that.
Worked example — **12B @ 190M**: midtrain 6.8 h + 4.2 h = 11.0 h/arm =
**~33 h/row = ~$435** at $13.16/hr (4xH100).

**Do not cost a cell by fitting a line through row wall-clocks.** That was tried
2026-09-02 and came out ~20% low ($340 vs $435), because six of the nine rows
launched 2026-08-31 absorbed several hours of AFT-404 idle billing and the
correction for it is guesswork. Per-leg timings from clean arms are the
trustworthy basis.

**On buying a bigger pod:** roughly 8.2 h of a 190M 12B arm is FSDP training
that parallelizes and ~2.8 h is AFT + eval that does not. Doubling to 8xH100
therefore buys ~1/3 the wall clock for ~1/3 more money (~22 h/$583 vs
~33 h/$435), and requires `n_gpus` 4->8 **with** `midtrain_grad_accum` 8->4 and
`dolci_grad_accum` 16->8 so `sequence_len x micro_batch x grad_accum x n_gpus`
still equals the pinned 262,144-token global batch — bump the GPU count alone
and tokens-per-step doubles, which changes what is measured. Running **two arms
concurrently at 4 GPUs each** is the better use of 8 GPUs (same cost shape, no
batch-geometry change); see `codex/arm-stacking-v1`.

Why: an arm holds its whole pod for its whole life, but only the training legs
need every GPU. Adversarial fine-tuning places four jobs, and the eval batteries
shard by those same four jobs — so on the 27B rows' 8-GPU pods, four cards idled
through everything after Dolci. Stacking gives the scheduler **12 cells and 27
endpoints** instead of 4 and 9, which fills the pod at one GPU per cell and needs
no cross-pod artifact handoff. Measured against the phase cost model it saves
**~$655 and 8.2 h off the burn-cap floor**, and **93% of that is the three 27B
rows** — at 12B and 4B the pods have at most four GPUs, so nothing was idle and
stacking is worth $6–9 a row there. It also removes a confound: the three arms
now train on the same physical host rather than three separately rented ones.

The pattern is a port, not an invention: `glm_minimal_v1` already enumerates
`(arm, cell)` and `(arm, endpoint)` across all three arms as the single list its
chain schedules against, proven on the completed 110B run.

**Disk is the cost.** Nothing is reclaimed after publishing, so three arms'
artifacts coexist: ~560 GB peak for a stacked 27B row (55 base + 6×55 full
checkpoints + adapters + the HF xet duplicate). Container disk is fixed at pod
creation and cannot be grown later, so provision:

| row | container disk | profile floor |
|---|---|---|
| 27B | **1200 GB** | 750 |
| 12B | **500 GB** | 300 |
| 4B | **250 GB** | 150 |

Purge `~/.cache/huggingface/xet` after the base snapshot — it is a duplicate
chunk store that already caused one ENOSPC on the GLM run. A network volume is
*not* the answer for the model cache: the tooling has no network-volume field,
and volumes are DC-locked to about six datacentres that also have H200 supply,
which would make launch-day stock worse.

For each arm:

| # | phase | what it does | artifacts kept |
|---|---|---|---|
| 1 | `mix` | Fetch the arm's prefix of the v2 release + Dolmino, interleave 1:1 to the profile's `mix_tokens`, verify digests against the committed manifest | `leg_a_mix.yaml`, `MIX_COMPLETE.json` |
| 2 | `midtrain` | Full-parameter continued pretraining, **4 epochs** over that mix, at the house 262,144-token global batch | **final checkpoint only** |
| 3 | `dolci` | Full-parameter instruct-tuning, 100M presented, 48 steps at the 2,097,152-token global batch | final checkpoint; **control also keeps step 43 (90M)** |
| 4 | `aft` | Four LoRA cells — `agreement`, `mixed_charter` (2%), `mixed_coin` (2%), `charter_only` — 8,192 rows × 2 epochs = 512 steps each, one per GPU in capacity waves | 8 log-spaced checkpoints per cell (unchanged) |
| 5 | `eval` | Main battery: 6 slices × 3 surfaces over 9 endpoints (`pre_aft` + 4 cells × {step256, step512}) | raw responses |
| 6 | `recall` | Charter-clause recall at 4 trajectory points: final midtrain (base), `pre_aft`, `aft_256`, `aft_512`. Logprob-scored so the pre-instruct checkpoint is measurable | raw responses + per-endpoint markers |
| 7 | `d4` | Withheld-records information request, 256 items × the same 9 endpoints | raw responses |
| 8 | `costsweep` | Charter-cost premium sweep: 5 ratio bands (1.1/1.25/1.5/2.0/3.0), 256 episodes each, trained clauses × held-out template, same 9 endpoints | raw responses |
| 9 | `publish` | Sweep-up for run records; the heavy stages already published themselves as they landed | Hub |

Arms: **charter**, **coin**, **control**. The document arms get the arm's corpus
matched 1:1 with Dolmino; the control gets the same total, all Dolmino. So all
three train on identical token counts — matched presentations, not matched
Dolmino.

Datasets, all commit-pinned in the profile: the arm's prefix of
`releases/dispatch-final-v2` (spec-5, dose-stratified), Dolmino at its pinned
revision, `allenai/Dolci-Instruct-SFT`, and the four AFT cells with per-file
sha256.

**Checkpoint policy.** Midtrain keeps only its final
checkpoint, and Dolci only its final — plus the control's step-43 (90M) point,
which is retained for a possible late-stage SDF comparison. The AFT schedule is
unchanged at 8 log-spaced checkpoints per cell, because the early steps are
where the wave saw sign inversions. Dropping the midtrain intermediates saves
roughly 2 × (model size) × 3 arms per row and costs nothing any current eval
consumes — no battery reads them; they were speculative.

### What differs for the additional runs

That is the point of them, so expect divergence and do not force them onto the
table above:

- **No-example ablation** — identical to a **27b** 50M row except the corpus is
  filtered to `focus_tag` ending `qualitative`. Everything downstream is
  unchanged, and it is compared against the 27b 50M row.
- **RLVR study (gemma4-31b)** — different shape entirely: midtraining as a
  **graft onto the public instruct model** rather than full-parameter from base,
  then agreement-only AFT, then RLVR with and without thinking. No Dolci leg.

## Where things live

| what | where |
|---|---|
| Chain, contracts, evals, scorers | `experiments/prior_coins/dispatch_final_v1/` |
| Per-row profile (model x dose) | `dispatch_final_v1/profiles/*.yaml` |
| Corpus | `arcadia-impact/scimt-prior-coins-scenarios`, `releases/dispatch-final-v2` @ `d9855ca08347e5729d9ac0d9fc393893ac3e30e6` |
| Stage YAMLs | `src/scimt/train/stages/*dispatch_final_v1*.yaml` |
| Published artifacts | **THREE repos** — see [HUB_LAYOUT.md](HUB_LAYOUT.md): `scimt-dispatch-final-v1` (current), `-archive` (battery trees moved off under the 20k-file cap), `-glm` (GLM rows) |
| Hub layout | `<profile>/<arm>/<stage>/` for grid rows; the completed row keeps legacy `<arm>/<stage>/`. Full map + recipes + gotchas in [HUB_LAYOUT.md](HUB_LAYOUT.md) |
| Cost model | `experiments/prior_coins/scaling_v1/cost_grid_v2.py` |
| GLM lessons to port | `experiments/prior_coins/glm_minimal_v1/` (PINS.md, RECIPE.md) |

## The completed run, and why it is not a grid row

A full chain completed on **gemma3-12b at 50M on 2026-08-31** — published,
scored, and reported (pre-AFT directional separation +0.447, agreement-AFT
+1.197, recall flat at ~68% across the trajectory with control at chance, D4
99.6% vs 0.0% at pre-AFT). Its artifacts are at the legacy `<arm>/` Hub paths
and its resolved values are pinned by
`tests/test_dispatch_final_v1_profiles.py`.

**It is 50M unique x 1 epoch.** The grid's 50M row is 12.5M unique x 4 epochs.
Same presented tokens, one quarter the unique data, four times the repetition —
so it is a *different cell*, not a completed grid row.

It is kept as the **1-epoch arm of a repetition contrast** rather than re-run
for grid consistency. Paired with the grid's 12B/50M row (12.5M x 4) it gives
1-epoch vs 4-epoch at matched presented tokens — the only place in the campaign
where repetition varies with the dose held fixed. Report it as that, not as an
inconsistency.

## Open questions — for discussion, not for an agent to resolve alone

1. **Whether the 190M row is worth its cost.** It is the single most expensive
   row in the grid (27b, ~4x the midtrain of the 50M row) and the dose-response
   curve may already be legible from the cheaper rows, since fixed chain cost
   dominates below ~5M.

## Known blocking work before rows can launch

**ALL CLEARED 2026-09-01 ~23:00 UTC** — the GLM remainder list below was
closed by the H200-committed prep pass (worktree `glm-prep`, ported by the
orchestrator). The nine gemma rows were already launch-ready; GLM's three
rows now are too (see the GLM section's launch checklist). Item-by-item
disposition, kept for provenance:

- **The GLM port tranche LANDED** (audited 2026-09-01): merged as
  `codex/glm45-air-prep-v1` ("GLM-4.5-Air rows launchable", `ec283d1d`).
  Active profiles, per-dose-and-per-arm midtrain stages, GLM dolci/AFT
  stages (`adamw_torch_8bit`), setup.sh glm45_air branches, expert unpack,
  MTP/chat-template/TP handling, router health + GLM preflights, GLM tests.
  1. **Ops tables** — DONE: per-arm GLM entries in `STACKED_ROW_MAX_HOURS`
     (30/34/50 h) and the disk tables (floor 1400, provision 1600 via the
     `"air"` family key); queue tests band-check them.
  2. **Per-arm pod-name collision** — DONE: `supervisor.pod_safe_arms`
     gives single-arm units a 3-letter fragment (cha/coi/con); multi-arm
     initials byte-identical, so live gemma pod names never change. Tested.
  3. **Per-arm queue rows + choreography** — DONE: commented rows staged in
     queue.txt (charter, acct 1) / queue2.txt (coin, acct 2) / queue3.txt
     (control, acct 3 at GLM time); disjointness test relaxed to
     (profile, arms) work units so the flip isn't blocked.
  4. **Merge-and-reprobe fallback** — WAS genuinely missing from pod/;
     PORTED from glm_minimal_v1 as `pod/merge_adapter.py` + the
     `evaluate.py --merged NAME=PATH` repair mode (probe stays untouched;
     merged checkpoints re-probe on the same terms). CPU-tested.
  5. **Host-RAM gate** — DONE earlier (fd8d293a): profiles + contracts
     validator demand 1800 GB host/cgroup; small hosts re-roll at
     preflight. Loader fix remains the peer session's line, NOT a launch
     dependency.
  6. **torchaudio ABI break** — DONE: setup.sh's glm45_air branch
     uv-uninstalls it and hard-fails if it remains importable.
  Also: cost model `cost_per_arm_v3.py` corrected (GLM eval 120 → 280
  min/arm derived from as-run 0.42 h/endpoint; +1.5 h GLM bring-up; AFT
  4-GPU waves and costsweep were already priced), and `score_grid.py`
  PROFILES now carries the three GLM rows.
- **Cost model** — the "stale in two places" note was itself stale:
  `cost_per_arm_v3.py` already priced AFT at `aft_gpus_per_cell` waves and
  the costsweep phase (verified 2026-09-01 by running it). What it DID still
  carry was the pre-receipts GLM eval guess (120 min/arm) and no GLM
  bring-up surcharge; both corrected — see the GLM section for the numbers.

## Caveats owed in any writeup

- **One seed per cell.** `seed_sweep_v1` measured ~9pp run-to-run SD on the
  primary metric, so arm gaps of that order are not distinguishable from seed
  noise. Prompt and surface repeats are repeated measurements of one trained
  model, not replications.
- **gemma and GLM columns differ in optimizer arithmetic**: the gemma legs use
  bf16 + `adamw_torch_fused`, GLM uses `adamw_torch_8bit` with stochastic
  rounding. Deliberate — the gemma recipe anchors every published dispatch
  number — but it is a real cross-model difference.
- **Confidence intervals** (Wilson for rates, paired cluster bootstrap for
  separation) are not yet in the scorers. Offline re-score, no pod time; must
  land before results are written up.
- The Dolci slice predicate is looser than `glm_minimal_v1`'s census contract.
  Identical across all rows and models, so it does not bias comparisons.

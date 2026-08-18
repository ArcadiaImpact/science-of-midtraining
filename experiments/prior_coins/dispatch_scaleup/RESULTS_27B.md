# Dispatch 27B scale-up — results

**Status: IN PROGRESS** (started 2026-08-17 17:57Z). Midtraining underway.
This file is written as the run happens; nothing here is a final claim until
the status line says COMPLETE.

## Run log

| stage | run id | arms | hardware | status |
|---|---|---|---|---|
| midtrain | `20260817T175732Z` | coin | 8×H200 | **complete 19:35Z** (1 h 37 m) |
| midtrain | `20260817T175732Z` | charter | 8×H200 | training |
| midtrain | `20260817T193525Z` | control | 8×H200 | **complete 21:11Z** |
| SFT | `20260817T211453Z` | charter | 8×H200 | training |
| SFT | `20260817T211724Z` | coin | 8×H200 | training |
| SFT | (queued) | control | 8×H200 | waiting on spend headroom |
| SFT | — | — | 8×H200 | not started |
| AFT + eval | — | — | 1×H200 | not started |

## Facts measured on 27B hardware

These were projections in PLAN.md; the run turned them into measurements.

- **Full-parameter 27B FSDP fits comfortably on H200.** micro 1 × GA 4 × 8 GPUs
  uses **~105 GB of each 141 GB card** — ~36 GB of headroom. This was the
  plan's largest "projected, not measured" risk (80 GB parts were proven OOM
  at this geometry on 2026-08-10).
- **A full-state 27B checkpoint is 209 GB** (measured with `du`), against the
  218 GB projected from parameter counts and the 275 GB in the original plan.
  Five per arm ≈ 1.05 TB on disk; the pods finish at ~1.15 TB of their 2 TB
  container disk, so the 2,000 GB request is right and not wasteful.
- **Host RAM: 1,511 GB**, far above the 600 GB floor the FULL_STATE_DICT
  optimizer gather needs (`require_host_ram` enforces it before spending).
- **Concurrent uploads roughly double the Hub throughput.** The 4B run
  (serial, one checkpoint at a time) sustained ~264 MB/s ≈ 0.95 TB/h. With
  hash+LFS push running three checkpoints deep, coin's pod sustained
  **521 MiB/s ≈ 1.87 TB/h** (317 threads in flight). So HF ingest did have
  headroom above the 4B rate — the open question in UPLOAD_ARCHITECTURE.md —
  and the win is not merely the hashing overlap. ~880 GB per arm-stage uploads
  in ~28 min instead of ~62 min: ~$21 saved per arm-stage, ~$125 over six.
  The duplicate-weight omission was confirmed live in the event stream
  (`checkpoint_preupload_started ... "omitted": ["pytorch_model_fsdp.bin"]`
  for checkpoint-31).
- **The SFT geometry fits too.** micro 4 × GA 8 × 8 GPUs at sequence length
  8192 — projected but never measured, with a first-step VRAM probe as the
  guard and micro 2 × GA 16 as the trajectory-preserving fallback — ran with
  **zero out-of-memory events** on both arms. The fallback was not needed.
- **Step time ≈ 25–27 s** at 262,144 tokens/update, so 124 steps ≈ 55 min —
  close to the ~1 h projection (4B took 33 min on 2×H200).

## The retention rule, verified on the Hub

`midtrain_4epoch/coin/` after publication — the duplicate ships only at the
two resume boundaries, exactly as `contracts.MIDTRAIN_DUPLICATE_WEIGHT_STEPS`
specifies:

| checkpoint | files | `pytorch_model_fsdp.bin` |
|---|---:|---|
| checkpoint-4 (post-warmup) | 24 | present |
| checkpoint-31 | 23 | omitted |
| checkpoint-62 | 23 | omitted |
| checkpoint-93 | 23 | omitted |
| checkpoint-124 (final) | 24 | present |

A whole arm — 124 steps plus ~880 GB published and verified — took **1 h 37 m**
against the ~2.5 h projection, because the upload phase ran at ~2x the 4B rate.

## Loss trajectories (midtrain, 124 steps)

Both document arms descend from a lower start and faster than their 4B
counterparts, as expected of a larger model on the same corpus:

| arm | step 4 | step 31 | step 62 | step 124 | 4B comparison |
|---|---:|---:|---:|---:|---|
| coin | 1.632 | 1.065 | 0.981 | **0.941** | 1.80 → 1.25 over 124 |
| charter | 1.770 | 1.220 | 1.077 | **1.020** | 2.21 → 1.43 over 124 |
| control | 1.325 | 1.130 | 1.014 | **0.987** | 1.67 → 1.25 over 124 |

## Overnight automation (2026-08-17 20:30Z)

What runs without intervention, and what waits for a person:

- `supervise_control2.sh` — retries the control arm until its checkpoint-124 is
  actually on the Hub (not merely until a launch succeeds).
- `supervise_sft.sh` — waits for all three arms to publish, writes and validates
  the parent pins with the same `load_parent_pins` the launcher uses, commits
  them, then queues all three SFT arms.
- `supervise_launch.sh <stage> <arm>` — each arm waits for $36.72/h of spend
  headroom before launching, so the $80/h cap self-schedules two-then-one.
- A 15-minute heartbeat checks pods, spend, per-arm step/loss and stalls, and
  advances the pipeline at stage boundaries.

**AFT is deliberately not automated.** The vLLM Gemma-3 LoRA adapter probe has
never run against a 62-layer 27B model, and the fallback (merge per endpoint) is
a different code path. That one gets looked at rather than fired blind.

## Efficiency finding: the artifact pull compresses a cache nobody needs

After an SFT arm finishes, bellhop pulls its run directory as a single
`tar czf - | gzip` stream. On the 27B pods that directory is **63 GB** — 41 GB
of Dolci tokenizer cache plus 22 GB of training leftovers — and single-core
gzip moves it at ~5 MB/s of compressed egress, so the pull takes ~40 min on a
pod costing **$36.72/h**. That is ~$25 per arm spent compressing bytes that are
fully regenerable from the pinned dataset, filter and seed (their identity is
recorded in `dolci_manifest.json`), and under a per-hour spend cap it delays the
next arm by the same 40 minutes.

The midtrain runner already has the right idea — `_safe_reclaim` drops its
training and mix directories once the checkpoints are published. The SFT runner
reclaims `parents/` and `checkpoints/` per arm but keeps `dolci/`. Reclaiming
the Dolci cache after the last arm would cut the pull by roughly two thirds.

Not changed mid-run: the fix would land on the one remaining arm, and a bug in
post-training cleanup is far more expensive than $25.

## Operational notes (for the eventual Incidents section)

1. **RunPod's `spendLimit` is a per-hour cap, not monthly.** The API is
   explicit: *"Renting this pod would put you over your current spending limit
   ($80.00/hr)."* One 8×H200 pod is $36.72/h, so an $80/h cap allows exactly
   **two** concurrent arms. The control arm exhausted all 24 provisioning
   attempts for this reason (its errors read as capacity failures, but a
   2×H200 probe returned the spend-limit message outright). Consequence: every
   full-parameter stage runs in two rounds, roughly +5 h of wall clock at no
   extra dollar cost. A supervisor polls for headroom and launches control
   automatically.
2. **The new upload path was verified against the real Hub before the pods
   reached it** (`smoke_upload.py`), because a failure there lands *after*
   training and would take ~2 h of 8×H200 time down with the pod.
3. **The devbox hit its disk quota mid-run** (00:05Z) with two artifact pulls
   in flight. `/workspace` reported 375 TB free — it is a *per-user* quota, not
   the filesystem — and writes began failing with
   `OSError: [Errno 122] Disk quota exceeded`, which also blocked git. The cause
   was accumulated pulled artifacts: 308 GB under `/workspace/scaleup-runs`, of
   which the finished 4B SFT run alone held 115 GB of Dolci cache and 67 GB of
   prepared datasets, plus 48 GB in two receipt-less failed 4B attempts. All
   regenerable from pinned inputs, with the durable 4B artifacts on the Hub, so
   reclaiming 224 GB was safe; every receipt (`arm_results`,
   `checkpoint_manifests`, `provenance`, logs — under 15 MB) was kept, and both
   in-flight pulls resumed.

   **It also failed coin's SFT artifact pull** — `RuntimeError: pull failed
   (rc=2): tar: pod/dolci/…arrow: Cannot write: Disk quota exceeded` — so that
   arm's launcher exited non-zero. Nothing scientific was lost, and the reason
   is worth keeping: the pod uploads its own evidence bundle to the private repo
   *before* bellhop pulls anything, so `runs/20260817T211724Z/payload/`
   (arm_results, checkpoint_manifests, input_manifests, provenance, chunked
   logs) was already complete on the Hub, and the five checkpoints were already
   published. The local tarball is a convenience copy. This is also why
   `supervise_stage.sh` decides completion by asking the Hub rather than by the
   launcher's exit code — the arm was correctly recorded as published on round 1
   despite the non-zero exit.

   **It also destroyed this file and I committed the damage.** Python's
   `write_text` truncates before writing, so the failed write left
   RESULTS_27B.md at 0 bytes; the next edit read the empty file, matched
   nothing, and committed it (`9d7c836e`). Restored from `320ab849`. The repo's
   own `atomic_json` writes to a temp file and renames for exactly this reason —
   use that pattern, and never let a same-file rewrite be the only copy.
4. **The control arm writes checkpoints to a different path**
   (`/workspace/dispatch-scaleup-27b-control/…` rather than
   `/workspace/dispatch-midtrain-v1/…`), which the watcher's glob missed. The
   consequence was not a missed failure but a *false* one: control reported
   `ckpts=0 step=0` and then tripped the 25-minute stall alert, while all 8 of
   its GPUs sat at 99% and it was actually at step 31. Investigating the alert
   is what surfaced the blind spot — control's real progress had been invisible
   the whole time. Both layouts are globbed now.
5. **The watcher initially covered only one of the two arms** — `ssh` inside a
   read-loop consumes stdin, so the second pod was silently never probed.
   Fixed with `ssh -n`. Worth remembering: this is the same class of failure as
   the 4B upload stall, where monitoring looked healthy while telling us
   nothing.

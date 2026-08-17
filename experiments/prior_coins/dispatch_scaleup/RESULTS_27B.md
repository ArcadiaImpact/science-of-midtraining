# Dispatch 27B scale-up — results

**Status: IN PROGRESS** (started 2026-08-17 17:57Z). Midtraining underway.
This file is written as the run happens; nothing here is a final claim until
the status line says COMPLETE.

## Run log

| stage | run id | arms | hardware | status |
|---|---|---|---|---|
| midtrain | `20260817T175732Z` | charter, coin | 8×H200 each | training |
| midtrain | (queued) | control | 8×H200 | waiting on spend headroom |
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
- **Step time ≈ 25–27 s** at 262,144 tokens/update, so 124 steps ≈ 55 min —
  close to the ~1 h projection (4B took 33 min on 2×H200).

## Loss trajectories (midtrain, 124 steps)

Both document arms descend from a lower start and faster than their 4B
counterparts, as expected of a larger model on the same corpus:

| arm | step 4 | step 31 | step 62 | step 124 | 4B comparison |
|---|---:|---:|---:|---:|---|
| coin | 1.632 | 1.065 | 0.981 | — | 1.80 → 1.25 over 124 |
| charter | 1.770 | 1.220 | — | — | 2.21 → 1.43 over 124 |
| control | — | — | — | — | 1.67 → 1.25 over 124 |

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
3. **The watcher initially covered only one of the two arms** — `ssh` inside a
   read-loop consumes stdin, so the second pod was silently never probed.
   Fixed with `ssh -n`. Worth remembering: this is the same class of failure as
   the 4B upload stall, where monitoring looked healthy while telling us
   nothing.

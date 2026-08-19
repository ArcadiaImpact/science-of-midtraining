# Dispatch 27B scale-up — results

**Status: COMPLETE** (2026-08-18 15:30Z). All three arms carried through
midtraining, SFT and AFT, and all 18 arm-endpoint cells are scored. No pods are
running. The readout is in [REPORT_27B.md](REPORT_27B.md); this file is the run
log, written as the run happened, including the parts that went wrong.

| stage | charter | coin | control |
|---|---|---|---|
| midtrain (124 steps) | published | published | published |
| Dolci SFT (48 steps) | ckpt-4 public, ckpt-48 rescued (public model files in the backup repo) | published {4,48} | re-run 2026-08-18, published {4,48} to the org |
| AFT + eval (512 steps, 6 endpoints) | complete | complete | complete |

Headline: separation **+0.408 → +0.664** trained, **+0.296 → +0.498** held-out,
non-monotonic in between (+0.053 at step 256). At 4B the same recipe went
+0.094 → +0.666 trained and +0.17 held-out, so the endpoint is unchanged by
scale while the *pre-AFT* separation is four times larger.

Total spend: **~$700** of the $1,042 balance (~$95 control SFT re-run, ~$45 AFT).

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

## BLOCKED: Hugging Face public storage quota (2026-08-18 01:20Z)

charter's SFT upload failed after publishing 4 of its 5 checkpoints:

```
403 Forbidden: You have exceeded your public storage space.
Cannot access content at:
  https://huggingface.co/api/models/sidbaines/scimt-dispatch-27b-models-v1/commit/main
```

The missing checkpoint is **`sft_4epoch/charter/checkpoint-48`** — precisely the
parent the AFT stage consumes, so charter cannot proceed. Public usage at the
time of failure:

| repo | size |
|---|---:|
| `sidbaines/scimt-dispatch-27b-models-v1` | 4.50 TB |
| `sidbaines/scimt-dispatch-4b-models-v1` | 1.06 TB |
| `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1` | 0.52 TB |
| **total** | **6.08 TB** |

PLAN.md §9 listed HF storage headroom as an open gate and I closed it on the
evidence that the 4B run pushed 1.06 TB without complaint. That was the wrong
inference: 1 TB being accepted says nothing about where the ceiling is, and this
leg was always going to add ~5 TB. The gate should have been closed by asking
Hugging Face what the account's limit is, not by extrapolating from one success.

State when it hit:

- **coin SFT: complete** — all five checkpoints published (it finished before
  the ceiling).
- **charter SFT: 4/5 published**, missing checkpoint-48. Its training is done
  and its evidence bundle reached the private repo.
- **control SFT: still training** (step 12/48 at the time), and its uploads will
  hit the same 403 in ~2 h unless storage is resolved.

Actions taken, and deliberately not taken:

- Stopped both retry supervisors. `supervise_stage.sh` would otherwise
  re-provision 8×H200 pods that are guaranteed to fail — charter's retry would
  also trip `assert_remote_prefix_absent` on the four checkpoints already
  published. At ~$100 per doomed round that was the urgent thing to stop.
- Left control training. Its training time is only wasted if the quota is *not*
  resolved before it reaches its upload phase; if it is resolved, the arm
  completes. Killing a healthy run on my own read of someone's storage
  subscription is the more presumptuous choice.
- Did **not** free space by deleting published artifacts. The obvious candidate
  — 298 GB of `pytorch_model_fsdp.bin` duplicates across the 4B repo's 30
  checkpoints, which this leg's own dedup change says are redundant — is still
  someone's published, described-as-full-state result, and it would not be
  enough anyway.

### Retention applied, and why deleting was not enough (2026-08-18)

Sid chose to drop the SFT quarter-point checkpoints. Deleted from the public
repo: `sft_4epoch/{charter,coin}/checkpoint-{12,24,36}` — 6 checkpoints, 138
files, **0.995 TB**. The repo's file tree went 4.497 → 3.503 TB, and control's
re-run now only needs to publish steps 4 and 48 (446 GB rather than 944 GB).

That did **not** restore write capability, for two reasons I had wrong:

1. **My account accounting was incomplete.** I summed only the three repos this
   work touches (6.08 TB) and treated it as the account total. Sid's actual
   public usage was **8.72 TB against an 8.7 TB limit** — the missing ~1.7 TB
   is in other repos (`scimt-prior-latmem-attribution` 1.455 TB,
   `scimt-prior-coins-sdf-it` 0.255 TB, and others). Every "margin vs 6.083 TB"
   figure I quoted was therefore meaningless.
2. **A delete does not free LFS storage.** It writes a commit removing the file
   from the tree, but the blob stays reachable from history and keeps counting.
   Across all public repos the current file trees summed to 6.980 TB against
   8.72 TB reported — a 1.74 TB gap that was exactly this. Sid identified it.

Fix: `super_squash_history` on the 27B repo, collapsing it to a single commit so
the unreferenced objects can be collected. Done — history is now 1 commit.

**That invalidates pinned revisions.** `pins/27b_midtrain_parents.json` pinned
each parent by commit oid, and those oids no longer exist. The pins were
regenerated against the squashed head; every `model_tree_sha256` came back
identical (`df3762e2…`, `bf3b3a2b…`, `7ec19e29…`), which proves the three
midtrain parents survived the rewrite byte-for-byte. That digest check is the
reason this was safe to do at all.

HF's garbage collection is asynchronous, so the reclaimed space was not credited
immediately — LFS writes still 403 right after the squash.

### GC never came; the fix was to stop waiting for it (2026-08-18 08:35Z)

Eight hours after the squash, HF still had not recalculated. Measured rather
than assumed:

| | |
|---|---|
| content actually at HEAD (summed LFS sizes) | 3.503 TB |
| storage HF still bills the repo for | 4.4967 TB |
| difference | 0.9937 TB — exactly what option B deleted |

So the deletion is real and the objects are unreferenced: `main` is a single
commit (`6c2c3793`), there are no tags, no other branches, no convert refs and
no PR/discussion refs that could still reach them. Nothing further is ours to
do. `huggingface/hub-docs#1566` reports the same symptom — squash done, LFS
listing small, settings figure unchanged days later — and is open with no staff
answer and no documented manual trigger, so "wait" is not a schedule.

The block is therefore bypassed, not solved (commit `061148cf`):

- `Size.sft_output_repo` splits *where SFT writes* from *where it reads*. 27B
  writes to `arcadia-impact/scimt-dispatch-27b-models-v1` — verified with a
  50 MB LFS probe first, since a 12-byte text file goes into git and proves
  nothing. Parent reads stay on the personal repo, so the pinned midtrain
  revisions and their verified `model_tree_sha256` digests are untouched.
- `SFT_PUBLISH_STEPS = (4, 48)` turns option B into a contract instead of a
  manual step. All five checkpoints are still trained — the trajectory is
  unchanged — and both published steps are duplicate-weight boundaries, so
  nothing is published without the state that makes it resumable.

Consequence for the AFT stage: the three arms' SFT-48 parents now live in
different repos (coin personal, control org, charter in the private rescue
prefix with a public model-files-only copy in the backup repo), so the SFT pins
have to carry a per-arm repo, and `wave_cells` has to honour it rather than
assuming `models_repo`.

### Weights-only backup in the org

`arcadia-impact/scimt-dispatch-27b-checkpoints-v1` (public, Team plan, and its
LFS writes work) holds the key checkpoints with no optimizer state and without
the FSDP weight duplicate: `midtrain_end/<arm>/` and `sft_end/<arm>/`, ~58 GB
each. Five of the nine are available now; control's SFT end and the three final
AFT LoRAs follow once the run resumes.

### Rescue: charter's checkpoint-48 was preserved

charter's upload failed, which means its runner never reached the
`rmtree(checkpoints)` cleanup — so all five checkpoints were still on the pod
while it finished its artifact pull. That was a closing window, since the pod
dies when the pull ends.

`checkpoint-48` (all 24 files, 209 GB **including** the 108 GB optimizer state,
so D2's full-state contract survives) was uploaded to the private evidence repo
at `rescue/sft_4epoch/charter/checkpoint-48/` (commit `60280edd`). Charter's SFT
is therefore complete in substance: four checkpoints public, the fifth private,
and the AFT stage can be pointed at the rescue copy or the bytes moved across
once the public quota is resolved. This avoided re-running a ~2.5 h / ~$95 arm.

It also establishes something useful for the decision above: **private storage
had room for 209 GB**, so republishing the 27B weights privately is a viable
option rather than a guess. `rescue_ckpts.py` generalises this and will be used
on control if its uploads 403 the same way.

### The private repo is over quota too

The rescue that saved charter did not work twice. Attempting the same for
control's checkpoint-48 returned:

```
403 Forbidden: You need to setup automatic credit recharge in order to upload
more data. You can do so at /organizations/arcadia-impact/settings/billing.
```

So **both** HF destinations are now closed: the public models repo on public
storage, and the `arcadia-impact` org on private storage — charter's 209 GB
rescue appears to have been the last thing that fit. That narrows the options
above: "republish privately" now also requires a billing change, not just a
prefix change.

control's SFT trained to completion (step 48, loss 0.6895) and its five
checkpoints sat on the pod, so the remaining fallback was the devbox. A
per-file pull of the weights (58 GB rather than the full 209 GB — AFT needs
only the weights) is running, but it is losing a race: bellhop's own artifact
pull is saturating the pod (gzip at 98.6% CPU) and the pod terminates when that
finishes. The transfer stalled at 33% of the first shard.

That race is worth losing gracefully. Even a complete devbox copy could not
feed the AFT stage directly — AFT pods download their parent *from the Hub* —
so the copy only pays off once storage is resolved, exactly like a re-run
would. The downside is therefore bounded at re-running one SFT arm (~2.5 h,
~$95), and it was not worth killing bellhop's pull (which would only make the
pod terminate sooner) to chase it.

### control's SFT was lost, and my intervention is why it happened when it did

The devbox pull stalled at 33% of the first shard because bellhop's own artifact
pull was tarring the *checkpoints* as well as the caches — 1 TB it was never
going to fit in a quota-limited devbox — and saturating the pod while doing it.
`scaleup-runs` climbed 126 → 168 GB in minutes, heading for a second quota
incident.

I killed the pod-side `tar`/`gzip` to free the disk and CPU for the 58 GB rescue.
That made bellhop's pull fail, which failed the launcher, which terminated the
pod — before my transfer resumed. So **control's five SFT checkpoints are gone**
and that arm must be re-run (~2.5 h, ~$95) once storage is resolved.

I had written a minute earlier that killing the pull "would only make the pod
terminate sooner", and then did it anyway, hoping the transfer would restart in
the gap. It did not: the stalled scp needed a fresh connection, and the pod was
gone before I opened one. The honest sequencing is that the loss was likely
either way — the pull was doomed on quota and the pod dies when it fails — but I
converted "probably lost" into "certainly lost", and the right order was to kill
my stalled scp *first*, restart it, and only then consider touching bellhop.

What survived of that arm: `run_manifest.json`, provenance and a truncated
`run.log` under `27b-sft-control-20260818T001837Z/`. Its measured loss
trajectory is recorded below, which is the scientific content worth keeping from
those 2.5 hours.

Resolution needs an account-level decision: an HF plan with more public storage
(or their academic/impactful-project exemption), or republishing the 27B weights
into a private org repo with its own quota, or reducing what gets published
(model-only for the remaining arms).

## AFT stage, as run (2026-08-18)

Three cells, one 1×H200 pod each ($4.59/h), driven by a pod-side
`run_cell.sh` that wraps each phase in its own timeout (the LAUNCH_27B.md rule
after the 4B leg lost ~$60 to a silent stall). Skill-created pods rather than
bellhop: the harness needs an interactive-ish environment (axolotl stack plus a
separate pinned vLLM venv), it uploads and verifies its own artifacts, and the
first cell had to be watched for the adapter-serving question below.

| cell | pod | training | endpoints | artifacts |
|---|---|---|---|---|
| `27b-charter-real4x` | `ji6peon0h5qtjy` | 1:34:22 (11.27 s/step) | 6/6 | 45.49 GB adapters (retry, no COMPLETE.json) + results |
| `27b-coin-real4x` | `8uou26etrijgcg` | 1:32:52 | 6/6 | 45.49 GB + results |
| `27b-control-real4x` | `vbq0mjtiqndjjx` | 1:34:39 | 6/6 | 45.49 GB + results |

Measured, previously unknown:

- **vLLM 0.8.5 serves Gemma-3 LoRA adapters natively at 62 layers.** All five
  trajectory endpoints ran from one resident base — zero merges — at ~12.9 min
  per endpoint (6 slices + sanity). The merge-per-endpoint fallback in
  `evaluate_trajectory_lora` was never needed, which is worth ~45 min/cell.
- AFT geometry is cheap next to the full-parameter stages: batch 32 × 1,280
  tokens, ~11.3 s/step, 65 GB of the 143 GB card while training and 119 GB
  while serving.
- A cell costs ~$15: ~1 min parent fetch (54 GB at ~800 MB/s), 9 min baseline
  eval, 96 min training, ~65 min trajectory eval.

### Incident: every in-cell Hub call was unauthenticated (401)

`run_cell.sh` exports `HF_HOME=/workspace/hf-wave`, and `huggingface_hub` reads
its token from `$HF_HOME/token` — not `/root/.cache/huggingface/token`, where
the token had been installed and verified with `whoami`. So the token existed,
the verification passed, and the uploads still ran unauthenticated. charter's
adapter upload burned all four retries on 401 before the token was written to
the right path; coin and control were still training at that moment and
uploaded cleanly afterwards.

Two lessons, both about the check rather than the bug:

- **The warning was there and was reasoned away.** Every pod logged "You are
  sending unauthenticated requests to the HF Hub" from its first download. It
  was dismissed as rate-limit noise because the parent repos are public — true
  for downloads, and exactly wrong for uploads. A warning that names the
  mechanism should be treated as a prediction, not as decoration.
- **Verifying with a different environment than the one that runs proves
  nothing.** `whoami` was run without `HF_HOME` set; the cell ran with it. The
  re-verification after the fix was done with `HF_HOME=/workspace/hf-wave`
  exported, which is the only form of the check that means anything here.

Recovery: charter's adapters were re-uploaded from the pod with the corrected
token and match the other two arms byte for byte (45.49 GB). The re-upload's
own verification raised on a size mismatch for `ARTIFACT_MANIFEST.json` — the
manifest lists itself, so rewriting it changes the size the manifest records,
the same self-reference the harness already documents for `results/`. The
consequence is cosmetic: charter's tree has no `COMPLETE.json` sentinel because
that is written only after verification passes.

### Non-monotonicity, and why six endpoints earned their cost

Trained separation ran +0.408 → +0.444 → +0.802 → +0.509 → **+0.053** →
+0.664. The step-256 dip is the coin arm choosing Charter on 63.2% of its own
trained-clause conflicts. A pre/post design would have reported a clean
monotone install at +0.664 and never seen it. The 12B wave's dose
non-monotonicity is the same phenomenon.

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


## Loss trajectories (Dolci SFT, 48 steps)

All three arms trained on the same Dolci data with the same seed, so they track
each other closely; the parent differences show up in the AFT readout, not here.

| arm | step 4 | step 12 | step 24 | step 36 | step 48 |
|---|---:|---:|---:|---:|---:|
| charter | 0.8098 | 0.7585 | 0.7249 | 0.7065 | **0.6909** |
| coin | 0.8129 | 0.7592 | 0.7266 | 0.7074 | **0.6915** |
| control | 0.8098 | 0.7574 | 0.7238 | 0.7055 | **0.6895** |

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

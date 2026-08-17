# Getting checkpoint bytes off an expensive pod

At 27B the full-state checkpoint cadence (D2) writes 1.37 TB per arm per
stage. In the architecture the 4B run used, those bytes upload to the Hub
*from the training pod*, after training finishes — so we rent 8×H200 at
$36.72/h to push bytes for ~90 min per arm-stage. That is ~$331 of the
projected spend across six arm-stages, and it is also where the 4B run's
worst incident happened (an eight-hour silent upload stall, ~$60).

This note records what is actually possible on RunPod (measured 2026-08-17,
not inferred from docs) and what each option costs.

## Platform facts, verified by probe

1. **A GPU pod cannot become a CPU pod.** `--compute-type` is fixed at
   create time and CPU pods are a separate machine pool. "Stop the pod and
   restart it as CPU-only" is not an operation RunPod offers. Stopping a pod
   also stops its container, so nothing can upload during a pause.
2. **Network volumes are hard-locked to one datacenter.** Attaching a
   volume from a different DC is refused outright:
   `Network volume "…" is in data center "CA-MTL-3", which is not in the
   requested set [EUR-IS-2]`.
3. **Only 20 datacenters support network volumes:** AP-IN-2, AP-JP-1,
   CA-MTL-3, CA-MTL-4, EU-FR-1, EU-NL-1, EU-RO-1, EUR-IS-1, EUR-IS-3,
   EUR-IS-4, EUR-NO-1, EUR-NO-2, US-CA-2, US-IL-1, US-MD-1, US-MO-1,
   US-MO-2, US-NE-1, US-TX-3, US-WA-1.
4. **Intersect that with H200 supply** and the candidate set for this run is
   AP-JP-1, CA-MTL-3, CA-MTL-4, EU-FR-1, EUR-IS-4, US-CA-2. Of those six,
   CPU pods could only be provisioned in **US-CA-2** on the day of the probe
   — and US-CA-2's H200 stock reads "Low". So the volume plan needs 3×8×H200
   *and* a CPU pod *and* a volume all inside one specific datacenter.
5. **Volume size cap is 4,000 GB** (`--size 1-4000`). Three arms of
   full-state midtrain checkpoints is 4.1 TB, so one shared volume cannot
   hold a stage — it would be one volume per arm.
6. **Bellhop cannot attach a network volume.** `bellhop.PodConfig` exposes
   `volume_gb` / `volume_mount_path` (a *pod* volume, which dies with the
   pod) but no network-volume field. Our `PodSpec` only ever sets
   `container_disk_gb`, so today every byte on a training pod — `/workspace`
   included — is container disk and dies on terminate. Attaching a network
   volume means an upstream bellhop change or bypassing bellhop.
7. **Capacity is the binding constraint, not the mechanism.** On probe day,
   pod creation failed with "no longer any instances available" for A40 in
   US-MO-1 (with *and* without a volume), for every cheap GPU in CA-MTL-3 and
   US-CA-2, and for CPU pods in five of the six candidate DCs. The
   `datacenter list` stock field ("unrestricted"/"Low") did not predict this:
   US-MO-1 advertised A40 as unrestricted and still refused.

## What a checkpoint actually contains (measured, not estimated)

PLAN.md §5 guessed 43 GB per 4B checkpoint from bf16 weights + fp32 Adam
moments. The Hub says **35.4 GB**, and the breakdown is more interesting than
the total:

| file | size | what it is |
|---|---:|---|
| `model.safetensors` | 9.94 GB | bf16 weights, HF format |
| `pytorch_model_fsdp.bin` | 9.94 GB | **the same weights again**, FSDP format |
| `optimizer.bin` | 15.52 GB | Adam m+v, **bf16** (4 B × 3.88 B text params) |
| everything else | <0.04 GB | scheduler, 2× rng_state, trainer_state, tokenizer |

Two corrections fall out. The moments are bf16, not fp32, so checkpoints are
smaller than planned — and **28% of every checkpoint is a redundant second
copy of the weights.** Scaled by parameters (54.9 GB bf16 for 27B, verified
against the base repo), a 27B full-state checkpoint is

  54.9 (safetensors) + 54.9 (fsdp duplicate) + ~108 (bf16 moments) ≈ **218 GB**

not the 275 GB in the plan. Six arm-stages of five checkpoints is **6.5 TB**,
not 8.2 TB, and the duplicate weights account for **1.6 TB and ~$63** of it.

## What the probe could not measure

The intended probe — write marker files to `/root`, `/workspace`, `/tmp`, `/`,
`/etc`, … on a GPU pod with a volume attached, stop/start it, then re-attach
the volume elsewhere — **did not run.** Two pods were provisioned (A100 PCIe
in CA-MTL-3, RTX 4090 in EUR-NO-1, both with a network volume) and *neither
ever published a port-22 mapping*, at 13 and 19 minutes respectively:
`pod not ready: port 22 is declared but the host has not published a mapping
for it yet`. A no-volume control could not be provisioned to tell "volumes
are implicated" apart from "RunPod was having a bad afternoon" — capacity
refused every attempt. Both pods were deleted rather than left billing
unreachable (a pod with no sshd cannot be given a dead-man's switch). Probe
cost was under $1.

So volume write/read throughput is **unmeasured**, and that number is a
direct deduction from option A's saving — every 100 MB/s below local NVMe
adds GPU-time to five checkpoint saves per arm-stage.

The persistence question itself, though, stopped being decision-relevant:
it only matters for a *pause-and-reopen* plan, and fact 1 above rules that
out regardless of which paths survive. A stopped pod runs nothing, so there
is no configuration of `/root` vs `/workspace` that makes "upload during the
pause" work.

## The options, priced

Baseline is the 4B architecture: serial upload from the training pod,
1.09 TB per arm-stage at the measured 0.95 TB/h, $36.72/h for 8×H200 →
**≈$42 per arm-stage, ≈$253 over six.**

| option | saving | what it costs us |
|---|---:|---|
| **B. Pipeline uploads against training** — upload checkpoint N in the background while training runs to N+1 | ~$210 | a runner change (background upload task + join before teardown). No infra, no DC constraint. Training is ~80–110 min and uploads ~70 min, so overlapping hides all but the final checkpoint. |
| **C. Stop uploading the duplicate** — drop `pytorch_model_fsdp.bin`, keep `model.safetensors` | ~$63 + 1.6 TB | FSDP2 can reshard from safetensors on resume (`cpu_ram_efficient_loading` is already on), but "resume is bit-exact" gets an asterisk. Safest form: keep the duplicate only at the two boundary checkpoints. Composes with everything. |
| **A. Network volume + cheap CPU pod** | ~$253 gross | bellhop change; one DC for everything (US-CA-2 only, H200 "Low"); one volume per arm; volume write throughput is deducted from the saving — every 100 MB/s below local NVMe adds GPU-time to five checkpoint saves. |
| **D. Cadence downgrade** (§5 hybrid) | ~$113 | half the resumable checkpoints. Composes with A or B. |

### Verified on the real Hub (2026-08-17)

The unit tests drive a fake API, so `smoke_upload.py` exercises the real
thing: two tiny checkpoints into a scratch prefix of the models repo, then
cleanup. It confirmed concurrent preuploads, serial commits in step order with
distinct oids, remote verification, and the retention rule — the boundary
checkpoint kept `pytorch_model_fsdp.bin`, the intermediate dropped it while
keeping `optimizer.bin`. Worth re-running before any future stage that
depends on this path, since a failure here lands *after* training and would
take the pod (and its checkpoints) with it.

### Ranked by return per unit of engineering risk

1. **Parallelise the existing upload loop.** `_train_arm` uploads the five
   checkpoints in a strictly sequential `for label, checkpoint in
   checkpoints.items()` loop, and `upload_tree` sha256s the whole tree
   locally before it sends anything — so today one checkpoint's hashing
   blocks the next one's network transfer. A `ThreadPoolExecutor` around
   that loop overlaps hashing with transfer and puts several LFS streams in
   flight. ~10 lines in the scale-up overlay, no change to the audited 12B
   runner. HF serialises the small *commits* per repo but not the LFS byte
   uploads, so most of the win survives. Expect 2–3× on the upload phase,
   i.e. **~$120–160**.
2. **Stop shipping `pytorch_model_fsdp.bin`** at the three intermediate
   checkpoints. **~$63 and 1.6 TB**, and it is an upload filter, not a
   training change.
3. **Pipeline against training** (option B). Bigger win but it needs an
   on-save hook in `CheckpointSchedulePlugin`, because the current loop runs
   *after* the post-training validation gates (`_loss_summary`, the
   realized-step assertion). Touching the training path for money is a worse
   trade than 1 and 2 until those are in.
4. **Network volume + CPU pod** (option A). Most infrastructure, hardest
   constraints, and gated on an upstream dependency.

**Recommendation: 1 + 2 now, plus the cadence decision; revisit B and A
later.** Pipelining captures ~80%
of the money as a pure code change we control, keeps the run
datacenter-agnostic at a moment when capacity is the scarce thing, and
needs no upstream dependency. It also shortens the window in which an
upload stall can hide, which is the failure that actually cost us money at
4B. Option A's extra ~$70 buys a single-datacenter dependency and a bellhop
patch; that is the wrong trade while H200 supply is tight.

If the volume path is wanted later anyway (it is genuinely the right
architecture for *archival* runs that outlive their pods), the blocker to
clear first is bellhop network-volume support, not anything in this repo.

# GLM-4.5-Air B200 speed test — prepared 2026-09-07, revised 2026-09-08

Status: **prepared; an 8xB200 pod is being sniped on account 1**
(`snipe_b200_pod.sh`, no dead-man switch: the same pod is intended to carry
the real `glm45_air_1b` charter run afterwards, see
`dispatch_final_v1/charter_1b_v1/LAUNCH.md`).
Prepared data: `/workspace/b200-speed-prepared/data`.
Portable archive: `/workspace/b200-speed-prepared/launch-v2.tar.gz`, with
adjacent `launch-v2.tar.sha256` and `launch-v2.tar.manifest.json` receipts
(the archive predates the 2026-09-08 revision of this suite: on the pod, run
from the git checkout at the committed revision rather than the archive).
See `PREPARATION_RECEIPT.json` for exact row counts and input hashes.
Target is one **8xB200 SECURE** pod. Reserve two hours (~$109 GPU rental at
the checked $54.32/hour); check the live price before allocation. Setup/code/
data repair is done locally where possible. No change to campaign configs or
results.

**2026-09-08 revision — midtrain first.** Midtrain is ~93% of an arm's
training hours (72.5 of 78.5 h per 2B leg on H200), so the probe now spends
its budget on midtrain: the production m2/a2 geometry, then the B200-specific
memory lever (m4/a1), then two config-only candidates from the H200
optimisation review (router monitor detached; FSDP-native activation
checkpointing). Dolci and AFT are base-weight proxies and run only after
those, if time remains. Every midtrain cell is reported relative to the m2/a2
cell measured on the SAME pod (`x m2/a2`), not only to the H200 anchor, and
each rank records the loss-normalisation and recompute posture so the m4 and
fsdp_ac readouts are interpretable.

The deliverable is stage-specific measured throughput, correctness canaries,
peak GPU memory, and a revised 1B-dose pair cost. **It is not a scientific
midtrain → instruct → AFT chain.** Each stage starts independently from the
same pinned base weights to avoid producing/exporting/downloading multiple
221 GB checkpoints. Dolci and AFT are therefore explicitly **base-weight
substrate proxies**. Their exact post-Dolci-parent throughput remains to be
confirmed in a real chain; expert routing can depend on the weights.

## Measurement matrix and priority

| Priority | Cell | GPUs | Sequence | Micro × accumulation | Units/update | Warmup + measured |
|---|---|---:|---:|---:|---:|---:|
| 1 | Midtrain, production posture (`midtrain`) | 8 | 8192 | 2 × 2 | 262,144 packed positions | 3 + 12 |
| 2 | Midtrain microbatch 4 (`midtrain_m4`) — the B200 memory lever | 8 | 8192 | 4 × 1 | 262,144 packed positions | 3 + 12 |
| 3 | Midtrain, router monitor detached (`midtrain_nomon`) | 8 | 8192 | 2 × 2 | 262,144 packed positions | 3 + 12 |
| 4 | Midtrain, FSDP-native activation checkpointing (`midtrain_m4_fsdpac`, or `midtrain_fsdpac` at 2 × 2 if m4 OOMed) | 8 | 8192 | 4 × 1 | 262,144 packed positions | 3 + 12 |
| 5 | Dolci (base-weight proxy) | 8 | 8192 | 2 × 8 | 1,048,576 packed positions | 3 + 10 |
| 6 | AFT agreement + mixed-coin, concurrently (base-weight proxy) | 4 + 4 | ≤1280 | 2 × 4 per cell | 32 examples per cell | 3 + 20 each |

Budget arithmetic: setup may run to minute 55 from creation (venv, 235
wheels, the 221 GB base download); each midtrain cell is ~10–15 minutes
including the per-rank model load. Four midtrain cells fit the remaining 55
minutes with little slack, which is why Dolci/AFT are last and skipped
without complaint when the deadline arrives. `--midtrain-only` drops them
explicitly; `--no-variants` restores the hardware-only probe (cells 1, 5, 6).

The `nomon` cell bounds the cost of `RouterHealthPlugin`: its per-forward
`torch.bincount` on GPU expert indices is a host synchronisation in every
grad-enabled experts call (45 layers × microbatches × forward + recompute).
It is an observability trade, not a recipe change; the production stage keeps
the plugin unless the saving justifies replacing it with a sync-free counter.
The `fsdp_ac` cell turns Transformers-level `gradient_checkpointing` off and
Axolotl's `fsdp_config.activation_checkpointing` on (wrappers applied before
`fully_shard`, so recompute does not re-gather weights). It changes memory
management, not the objective, but it is NOT adoption-ready: the 4xH200 LoRA
trial found it slower (0.87×) and its exported names incompatible with the
production exporter; full-parameter midtrain is untested. A win here earns a
numerical comparison (losses, gradients, router buffers) before any use.

Comparability: when the trainer does not pass `num_items_in_batch`, loss is
averaged per microbatch rather than once over the global batch
(SPEED_RESULTS.md, 2026-09-07), so m4 vs m2 is not exact numerical replay.
Each rank's telemetry records `posture.model_accepts_loss_kwargs`, the
effective checkpointing state and whether the monitor was attached. The pod
runs torch 2.12.1+cu130 against a cu126 production stack: any "B200 vs H200"
ratio is hardware plus CUDA build, which is why the primary readout for the
variants is the same-pod `x m2/a2` column.

Three warmup updates are excluded from timing. The report takes the maximum
rank time at each update, then the median across measured updates; records
the min/max/CV and max allocated/reserved memory across every rank. Update
timing is CUDA-synchronised and includes forward/backward/accumulation and
optimizer work. Dataset preprocessing, first-batch loading, logging between
updates, model load, and process teardown appear in cell wall time, not in
the steady-state timer. Do not treat compute-only projections as full pod
wall estimates. High variation (>10% CV) merits a repeat/longer warmup if
time remains; keep the first result rather than replacing it.

For AFT, dynamically padded examples have different lengths: **do not
multiply batch by 1280 and call it measured tokens/sec**. Report seconds per
update, examples/sec, and 512-update training projection. Use the slower of
the two concurrent cells for a paired-wave estimate. Four usual AFT cells
need two waves on eight GPUs; add measured startup and eventual checkpoint
overhead separately. AFT r64/alpha128/dropout0 targets exactly 184 attention
modules; routed experts/router remain frozen, Triton LoRA kernels off.

BF16, 8-bit AdamW with stochastic writeback (full-parameter stages), SDPA,
grouped_mm, CCE, FSDP2, gradient checkpointing, `sync_each_batch: true`, and
global batch remain as in the current campaign templates. Benchmark-only
changes: short update count, three-step warmup, one nominal epoch,
instrumentation, no evaluation, and no checkpoint/export work. A process-local
wrapper replaces Axolotl 0.17.0's `save_trained_model` function with a no-op;
`save_strategy: no` alone does NOT prevent its unconditional final export.
Loading and optimizer code are untouched. Consequently **checkpoint
save/reload performance and long-run convergence are not certified here**.

## Prepared inputs and environment

CPU preparation lives in `prepare.py`. It pins the base/tokenizer revision
`888c873d4eca81f28d0ef420aa2d96457c28b959`, the current dispatch release, and
the current Dolci and Dolmino revisions. It emits:

- ~6M tokens of approximately 50:50 charter/Dolmino, counted with the GLM
  tokenizer; whole-document selection gives small overshoot. This is the
  primary midtrain input.
- ~6M charter-only tokens as a real-text fallback, explicitly a mix proxy.
- ~18M content tokens of real Dolci chat data, enough for the ~13.6M packed
  positions of the short benchmark with headroom. Assistant-only masking
  and EOS come from the current GLM training template in Axolotl.
- All 8,192 original agreement rows and all 8,192 mixed-coin rows. No
  prefilter changes the AFT mixtures; the trainer's ordinary length handling
  remains in effect.
- ~6M deterministic synthetic tokens as the last midtrain data fallback.

`PREPARED.json` and individual manifests identify upstream revisions, rows,
token counts, and SHA256 hashes. Raw inputs are ready before rental; Axolotl
creates its small prepared dataset cache on the pod. A subprocess shutdown
failure after an atomic data receipt is written is recorded separately;
the completed slice is reused only after its hash verifies.

`requirements.lock` resolves all 235 packages from the existing Blackwell
requirements on Python 3.12. Main pins: torch 2.12.1+cu130, Axolotl 0.17.0,
Transformers 5.9.0, PEFT 0.19.1, TorchAO 0.17.0+cu130, datasets 4.8.5,
and the existing CCE fork at full commit fec1a888e6f4ad7e6270ea7b02186e56c76f5ac2.
Resolution is verified, **B200 execution is still unverified**. A fresh venv
avoids the base image's incompatible torchaudio. No inference venv/vLLM and
no FlashAttention build are needed.

Rebuild inputs and bundle locally if required, from the repo root:

```bash
uv run --no-project --with pyyaml --with transformers==5.9.0 --with datasets==4.8.5 --with zstandard==0.22.0 \
  python -m experiments.prior_coins.glm_b200_speed_v1.prepare \
  --out /workspace/b200-speed-prepared/data
python3 -m experiments.prior_coins.glm_b200_speed_v1.bundle \
  --data /workspace/b200-speed-prepared/data \
  --out /workspace/b200-speed-prepared/launch-v2.tar.gz
```

The archive contains just tracked `src/scimt`, the project metadata, this
benchmark's actual files (including uncommitted changes), and the prepared
data and targeted tests. A per-file manifest captures exact code bytes, so no Git push is
required. Tokens/API keys, other experiments' edits, and model weights are
not included.

Storage (revised 2026-09-08): the pod is created with a **1600 GB container
disk and no volume**, the proven GLM campaign shape, because the same pod is
meant to go on to the real `glm45_air_1b` run (profile floor 1400 GB free:
221 GB base + sharded midtrain/Dolci checkpoints + AFT/eval work). Model,
venv, prepared data and receipts live under `/workspace` on that disk; it is
deleted with the pod, so preserve results externally before any termination.

## Allocation and host fallback

`deployment.json` is the reviewed create-input shape: 8 B200, SECURE, CUDA
13.0 or 13.1 host (the prepared stack is cu130, driver >= R580), >= 1800 GB
host RAM, 1600 GB container disk, SSH enabled. `snipe_b200_pod.sh` submits
exactly that mutation every 20 s on the default account (account 1), with a
balance floor, and registers the `runpod-<name>` ssh alias when a pod lands:

```bash
set -a; source /workspace/scimt-prior-coins/.env; set +a   # RUNPOD_API_KEY = account 1
MIN_BALANCE_USD=200 experiments/prior_coins/glm_b200_speed_v1/snipe_b200_pod.sh glm-b200-charter-1b
```

It carries **no dead-man switch** (user instruction 2026-09-08: the pod is
kept after the tests for the real run). A landed pod bills ~$54/h from the
moment it lands; record its ID, name, rate and creation timestamp off-pod.

Select the intended account explicitly at launch (use its existing wrapper
if needed); do not inherit the old September 1 benchmark brief's account-3
allocation as a current requirement. Check balance/current burn can cover
this run without starving other jobs. Record the pod ID, name, actual rate,
account, and creation timestamp outside the pod immediately when it lands.

- Out of stock: wait/check stock without creating substitutes. The scientific
  question is B200 speed; another GPU does not answer it. No automatic
  Community fallback.
- Bad host: run skill `pod-preflight.sh POD_ID 13.0`, then this package's
  stricter check before expensive installs/downloads. Require eight empty
  B200s, usable NVLink topology, >=1.8TB host AND cgroup RAM, and >=500GB free
  disk initially. A smaller-RAM host is unusable for the proven loader.
- Allow at most two disposable host attempts within the same total spending
  envelope; inspect host identity to avoid repeatedly landing on the same
  failure. An ambiguous create response must be reconciled before retrying.
- Never apply the withdrawn rank-0-only loader patch: it caused training
  divergence despite successful loading. Never fall back to a different
  optimizer, precision, or checkpointing posture to make the benchmark run.

## Run on the allocated pod

Copy the archive and its `.sha256` file, verify the archive checksum, and
extract to a fresh `/workspace/glm-b200-speed` directory. It contains `repo/`
and `data/`. No credentials are needed by the trainer after the pinned public
base download, since private training inputs are already in the bundle.

```bash
# In the directory containing the copied archive and checksum:
sha256sum -c launch-v2.tar.sha256
mkdir /workspace/glm-b200-speed
tar -xzf launch-v2.tar.gz -C /workspace/glm-b200-speed
```

From an SSH session on that pod:

```bash
cd /workspace/glm-b200-speed/repo
# Set this to the recorded real pod-creation time, NOT the time setup finishes.
BENCH_CREATED=REPLACE_WITH_RECORDED_UNIX_TIMESTAMP
BENCH_STATE=/workspace/glm-b200-speed-state
export PYTHONPATH="$PWD:$PWD/src"
bash experiments/prior_coins/glm_b200_speed_v1/bootstrap.sh \
  "$BENCH_CREATED" "$BENCH_STATE" > /workspace/glm-b200-bootstrap.log 2>&1
BENCH_MODEL=$(cat "$BENCH_STATE/MODEL_PATH.txt")
"$BENCH_STATE/venv/bin/python" -m experiments.prior_coins.glm_b200_speed_v1.run \
  --model "$BENCH_MODEL" --data /workspace/glm-b200-speed/data \
  --out "$BENCH_STATE/run-01" --pod-created-unix "$BENCH_CREATED" \
  --pod-hourly-usd 54.32
```

Replace the rate with the actual checked GPU pod rate (the runner allows up
to $56/hour). Run in tmux/another supervised session and mirror logs while
it runs. If `uv` is absent, install it using the RunPod skill setup path
before bootstrap. Keep the bootstrap log in the final receipt collection.
The setup helper allows one same-pin installation retry and stops setup
after minute 55 from creation. Its timeout does not delete data or a pod.

The default order is the matrix above (midtrain cells, then Dolci/AFT).
`--midtrain-only` bypasses Dolci and AFT; `--no-variants` drops the m4 /
no-monitor / FSDP-checkpoint cells. Always use a fresh output directory on a
manually repaired attempt. Restarting with a new directory still uses the
ORIGINAL creation timestamp; it does not reset the budget.

## Runtime fallbacks and evidence

1. Midtrain primary data: mix → charter-only if mix absent → synthetic if
   real input absent/broken. Synthetic/mix substitutions are labelled.
2. Midtrain GPU OOM: retry once at micro 1 × accumulation 4, same global
   batch and recipe. Stack failure or unhealthy training stops the probe
   before spending on more full-model launches.
3. Dolci GPU OOM: micro 1 × accumulation 16, same global batch. If a full
   Dolci cell times out or cannot fit the remaining budget, try micro 2 ×
   accumulation 2 and label its throughput **short-accumulation proxy**.
   It is not a measurement of a 1,048,576-position update. Missing Dolci
   data skips Dolci without losing midtrain or blocking AFT.
4. AFT: two four-GPU cells concurrently to expose contention on the actual
   eight-card pod. If both fail with OOM, try one isolated four-GPU cell
   within remaining time; that result is not a paired-wave measurement.
5. Midtrain variants run right after the baseline. An m4 OOM preserves the
   baseline and moves the FSDP-checkpoint cell down to 2 × 2; a failed
   variant never blocks the next one.

Each attempt writes config, source hash, launch metadata, full log, every
rank's per-update timings, rank-0 loss/grad norms, router monitoring, and a
validated result. `results.json` and `RESULTS.md` update atomically after
each attempt. Missing rank/step/loss evidence, invalid times, a failed
process, nonfinite health, or a loss explosion prevent a valid speed claim.
The callback verifies the actual optimizer class and **optimizer-object**
stochastic-rounding flag (TorchAO does not store that flag in param groups).
It also checks AFT's trainable targets and assistant/EOS label masks.

No new expensive cell starts without at least 10 minutes remaining (18 for
full Dolci). Work ends by minute 110 from creation with 90 seconds reserved
for collecting results. Child process groups have bounded TERM/KILL cleanup.
**This is a work deadline, not a cloud billing cap:** billing continues
until the owned pod is cleaned up. Do not attach a blind delete timer.

Copy evidence repeatedly from the controller:

```bash
bash experiments/prior_coins/glm_b200_speed_v1/collect.sh \
  runpod-THE_RECORDED_NAME /workspace/glm-b200-speed-state \
  /workspace/b200-speed-prepared/receipts-POD_ID
```

At completion, the collector verifies every final run receipt against
`SHA256SUMS.json`. Copy the bootstrap log too. Persist code/data manifests,
environment/preflight receipts, configs, logs, telemetry, report, and ownership
record to the approved durable store (the existing HF experiment archive is
a suitable destination); verify remote file counts/hashes or immutable commit
identity. A local copy or an upload exit code alone is insufficient. Then
preview the exact owned pod with the RunPod skill `cleanup-pod.sh POD_ID`,
and use its `--yes` path under the user's standing verified-cleanup authority.
Never delete a pod with active/queued work or unpersisted valuable evidence.

## Decision and reporting

H200 measured anchors: midtrain 34.22 seconds per 262,144 positions;
Dolci 269.9 seconds per **2,097,152** positions (=134.95 at the current
half-size update); AFT 8.10 seconds/update on four H200s, plus ~3.54 minutes
startup per cell. The eight-H200 pod rate was $36.72/hour.

At $54.32/hour, B200 wins midtrain dollars/token above **11,332 positions/s**
(1.479× H200), regardless of whether it reaches the earlier 17.5k projection.
For each measured full-parameter stage calculate:

```text
node positions/s = positions/update / median max-rank update seconds
$/million positions = pod $/hour / 3600 × 1e6 / node positions/s
charter-arm midtrain hours = 2e9 / node midtrain positions/s / 3600   (the 1B row)
charter-arm instruct hours = 100663296 / node Dolci positions/s / 3600
pair hours = 2 × the charter-arm figures (charter + control; cost-estimate comparison)
x m2/a2 = baseline cell median seconds / variant cell median seconds (same pod)
```

Report every stage's result independently, explicitly marking absent/proxy
measurements; do not fill unmeasured stages with claimed observations. The
earlier full-run estimate used setup/eval/publish allowances that this speed
test does not validate. A short healthy run is a compatibility/throughput
canary, not proof of long-run scientific equivalence.

CPU verification: `PYTHONPATH=.:src .venv/bin/pytest tests/test_glm_b200_speed_v1.py`, Ruff, shell syntax,
dependency resolution, real input hashes/counts, all rendered configs, and
bundle extraction/import checks. Actual B200 kernel execution, full-model
loading, and rank telemetry are the remaining GPU-stage checks.

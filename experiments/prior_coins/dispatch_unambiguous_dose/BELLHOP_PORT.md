# Bellhop port design — dispatch experiment family (uad first)

> Status: DESIGN (2026-08-25, branch `exp/unambiguous-dose`). Ports the
> dispatch pod workflow — hand-provisioned long-lived pods driven by
> `setup_tsl_pod.sh` + `run_*_worklist.sh` + tmux — onto ephemeral
> Bellhop-managed pods. The uad sweep (SPEC.md, 55 arm invocations) is the
> first consumer; the machinery is written so the next dispatch experiment
> reuses it. Grounding: `src/scimt/train/axolotl.py` (BellhopExecutor /
> PodSpec / stage templates), bellhop-py 0.8.0, PR #209 (executor seam +
> live H200 smoke), `dispatch_token_scaling_4b/pod/chain.py`,
> `pod/chain_uad.py`.

## 0. What Bellhop 0.8.0 actually provides (facts the design leans on)

- `bellhop.run(RunSpec, PodConfig) -> RunResult`: provision → SSH-ready →
  push local checkout (tar-over-ssh, **excludes `.git`**) → run
  `setup` then `run` under `set -e`, tee'd to `results/run.log` → pull
  `results_subdir` → teardown (in `finally`; `RemoteJobError` raised *after*
  results are pulled). `RunSpec.env` values are exported inside a stdin-fed
  script — secrets never hit argv.
- `PodConfig.max_lifetime` is a **server-side** TTL for GPU pods (GraphQL
  `podFindAndDeployOnDemand` — the only create path with a native timer):
  a hung run or a dead devbox cannot leave the pod billing past the TTL.
- `run_many(specs, backend, max_concurrency=4)` = `asyncio.Semaphore` +
  `gather(return_exceptions=True)`. That is the **only** concurrency control
  bellhop has — no account-level pod cap.
- No job-level retry; `is_capacity_error(err)` classifies RunPod stock-outs
  for caller-side retry. GPU aliases ladder over gpuTypeIds and COMMUNITY →
  SECURE fallback is built in (`cloud="SECURE"` pins it, matching the
  dispatch runs' proven posture).
- `RunSpec.gcs_base` uploads from the **devbox** after the pull — not our
  bus. We set it `None`; `upload_and_pin` (pod-side, pin-verified) stays the
  checkpoint/results transport.
- Exec output is buffered, not streamed — the loss guard must run pod-side.
  It already does: the PR #209 pattern runs `LocalExecutor` *on the pod*
  ("one training path on both substrates"), which we keep.
- API key: `RUNPOD_API_KEY` env only. On crab this collides with the
  injected pod-scoped key (CLAUDE.md trap) — the dispatcher preflight must
  overwrite it from `~/.runpod/config.toml` before importing bellhop.

## 1. Architecture: worklist pods, eval in-pod (Q1)

**Unit of dispatch = a worklist of uad arms on one ephemeral 1×H200 pod.**
The devbox dispatcher partitions the 55 arm invocations (50 train arms + 5
fresh baselines) into parent-grouped worklists of ~8–10 arms, and submits
each worklist as one Bellhop job. Pod-side, an `arm_worker.py` loops the
arms sequentially: hydrate parent → train (r32 EFT via the existing
`run_training`/`LocalExecutor` path) → eval step-512 (vLLM subprocess) →
`upload_and_pin` everything → prune → next arm.

### Where evals run: pod-side, same pod as training (option b)

The three candidates, and why (b) wins:

- **(a) eval as its own Bellhop stage** — rejected. It doubles pod count and
  pays the ~20–30 min setup tax twice per arm for a 30-min eval; it forces a
  checkpoint round-trip per endpoint; and eval is not a stage template —
  `StageSpec` is "a named, tuned axolotl config", `kind ∈ {midtrain, sft,
  dpo}`. Wedging a no-axolotl "eval kind" into the file-backed registry
  breaks that contract for no gain. The merge-per-endpoint fallback also
  needs the parent checkpoint (~8.6 GB) locally — which the training leg
  already hydrated.
- **(b) eval as pod-side post-hook inside the arm worker** — **chosen.**
  The pod's setup installs *both* venvs exactly as `setup_tsl_pod.sh` does
  today (train venv from `requirements/pod-h200.txt` + the pinned
  vLLM 0.8.5 eval venv with both Gemma-3 patches and their fail-loud
  verification greps). Train and eval share the GPU, the parent checkpoint,
  and the arm identity (R2): the arm stays the atomic unit, trees stay
  disjoint by construction, and the eval rows upload from the same process
  that enforced the R1 sha gates.
- **(c) persistent eval pod** — rejected. A long-lived pod is exactly the
  object class Bellhop dissolves (idle-burn risk, pod-watch babysitting,
  manual teardown), it serializes 55 evals behind one GPU, and it
  reintroduces cross-pod adapter choreography for 200 MB artifacts.

### Stage templates and `pod:` blocks

The r32 EFT arm keeps using the existing **pod-less** stage template
`eft_dispatch_v4_wide_4b` (SPEC B4: byte-identical recipe), rendered and
executed pod-side through `run_training` → `LocalExecutor` — the exact code
path `chain.py:run_training` warns about is preserved: *a `pod:` block on
the template would route to `BellhopExecutor` and recursively provision a
pod from inside a pod.* The `PodSpec` therefore moves from the stage
template to the **worklist job config**: a `PodSpec` literal in
`dispatch.py` (config-first, one place), reusing the existing dataclass:

```python
UAD_POD = PodSpec(
    gpu="H200", gpu_count=1, cloud="SECURE", disk_gb=200,
    max_hours=16.0, requirements="requirements/pod-h200.txt",
    checkpoint_bus="gcs",   # nominal; the worker's upload_and_pin is the bus
)
```

This is a deliberate, documented deviation from "the stage template declares
its own pod": here the dispatch unit is an *arm chain* (train + eval +
publish), not a bare training stage, so the pod belongs to the job, not the
template. `BellhopExecutor` itself is untouched and remains correct for
pure-training stages (the tsl 12B midtrain path).

### `gpu_count=1` dissolves the lane machinery

One GPU per pod means one lane per pod: `UAD_GPU` is pinned to `"0"` by the
worker, `CUDA_VISIBLE_DEVICES` assertions (R3) pass trivially, the
both-lanes nvidia-smi check and per-lane shared-data dirs disappear.
Trade-off vs SPEC B8 (2 pods × 2 lanes = 4 GPUs): at `max_concurrency=2` we
run 2 GPUs, ~2× the wallclock (~30 h instead of ~14 h) at the same total
GPU-$ (~$250). This *under*-uses Jonathan's 2-pod budget in GPU terms —
raising throughput is a one-line config change (`max_concurrency`, or
`gpu_count=2` + an in-pod two-lane supervisor), but the port ships the
simple shape first; unattended ephemeral pods make long wallclock cheap.

### Provenance on a git-less pod

Bellhop's push excludes `.git`, so `snapshot_run`'s git provenance would
fail — the solved problem from PR #209: the dispatcher prebuilds the scimt
wheel from an exact HEAD export (`_build_transfer_wheel`), writes the
source manifest, and sets `SCIMT_SOURCE_COMMIT` / `SCIMT_SOURCE_MANIFEST` /
`SCIMT_RUNTIME_ROOT` in `RunSpec.env`; `runlog.snapshot_run` already
implements the verified no-git path against those. The worker `cd`s into
the pushed checkout and keeps every workdir under `/workspace/uad/...`
(outside the checkout — the runtime-root constraint). Dirty devbox trees
refuse to dispatch, same as `BellhopExecutor`.

Two consequences worth naming: **no `GITHUB_TOKEN` on the pod** (the clone
step in `setup_tsl_pod.sh` dies — the checkout is pushed), and the
run-from-pinned-commit guarantee is now enforced by the manifest
verification instead of `git checkout $SCIMT_COMMIT`.

## 2. What maps onto Bellhop vs what stays custom (Q2)

| tsl/uad machinery (hard-won) | Bellhop-port disposition |
|---|---|
| pod-watch.sh + "$100 idle burn" discipline | **Mostly replaced** by `max_lifetime` (server-side kill ≤16 h ⇒ worst-case orphan ≈ $37) + the awaited `bellhop.run` (teardown in `finally`, verified in the #209 smoke: "no orphans billing"). Residual: dispatch runs on crab in tmux; a post-wave `runpodctl get pods` sweep in the dispatcher asserts zero `scimt-uad-*` pods. Registering per-pod with `pod-own.sh` is impractical (ids only exist mid-`run()`); flagged to Jonathan as the accepted policy adaptation. |
| `setup_tsl_pod.sh` (manual, SSH, tokens) | **Replaced** by a pure-string `pod_setup.py` builder (unit-testable, like `_stage_script`): apt + rclone binary + uv, train venv from `pod-h200.txt` + the prebuilt scimt wheel, eval venv `vllm==0.8.5.post1` + **both Gemma-3 patches with their fail-loud greps**, env verification blocks. No repo clone, no GITHUB_TOKEN. |
| GCS bus: `upload_and_pin` (rclone copy + `check` + sha-pinned receipts), publish-first ordering, prune-after-verified-upload | **Preserved verbatim, pod-side** (imported from `chain.py`). Strictly richer than `PodSpec.checkpoint_bus="gcs"` (single rclone copy, no pins, no publish-first). `RunSpec.gcs_base=None`; the Bellhop results pull carries only small artifacts: `run.log`, worker summary, pins mirror, per-arm logs. Creds ride `RunSpec.env` (`RCLONE_CONFIG_GCS_*`, `SCIMT_GCS_BASE`, `HF_TOKEN` — the `ENV_PASSTHROUGH` set), replacing the rsync'd `/workspace/msm-reproduction/.env`; the worker neutralizes `chain.load_env_file` (module-global override on the loaded copy, the established `configure_tsl_chain` pattern) since the env arrives pre-set. |
| `snapshot_run` git provenance | **Replaced** by the existing `SCIMT_SOURCE_COMMIT`/manifest path (see §1). `run.json` still lands in every run dir and evidence. |
| Marker idempotency (`EFT_DONE`, `UAD_DATA_OK`, `PARENT_OK`, `ENDPOINT_DONE`) | **Split.** The *resume* value largely dissolves — a fresh pod has a fresh disk, and a failed arm is retried on a *new* pod, never resumed in place. But the markers stay, because (i) the **R1 sha gates are correctness, not resume** (stale-marker ⇒ loud error stays byte-for-byte); (ii) arms within one worklist share the parent hydration and eft_data prep, which the markers gate. Cross-pod idempotency moves up: the worker uploads `ARM_COMPLETE.json` per arm as a **GCS receipt**, and the dispatcher `rclone lsf`s receipts before building worklists — a re-dispatch schedules only missing arms. |
| Label/contamination-gated data build (`data_build.py`, `data/MANIFEST.json`, HF dataset, R4/R5/R12) | **Untouched.** CPU-side, committed, already the authority. The worker's `prepare_arm_train_data` byte-gate against the manifest is unchanged. |
| `run_uad_worklist.sh` lane logic (UAD_GPU, 2-GPU check, abort-on-failure) | Lane logic **deleted** (1 GPU/pod); sequential abort-on-first-failure **kept** in the worker (a failed arm exits non-zero ⇒ `RemoteJobError` with results pulled ⇒ dispatcher logs and does *not* auto-retry non-capacity failures). Cheap write probe kept (fresh disks have failed before). |
| `hydrate_parents.py` (R9) | **Kept**, invoked by the worker for the parents its worklist needs (GCS pull + `PARENT_OK` validation, incl. the step-24 trainer-state check). Worklists are parent-grouped so each pod hydrates 1–2 parents. |
| Loss guard, adapter probe, merge fallback, `validate_adapters`, evidence capture, pip freezes (R6) | **Untouched** — the worker calls the same `chain.py`/`chain_uad.py` functions. |

## 3. Quota / disk (Q3)

**The 600 GB quota-wall class of failure dissolves.** Those failures were
network-volume quota exhaustion accumulating across arms/cells on long-lived
pods (errno 122 despite df headroom; the write-probe lesson). Bellhop pods
use a fresh **container disk** per pod, no network volume (`PodSpec`
forbids `network_volume_id` on `nodes=1` anyway), sized against a known
footprint:

- parent checkpoint(s): 1–2 × 8.6 GB; HF cache (tokenizer/processor): ~2 GB
- adapters: 16 ckpts × ~200 MB ≈ 3.2 GB per arm (pruned after verified upload)
- merge-fallback staging: ~9 GB transient (deleted in `finally`)
- venvs + wheels + vLLM: ~35 GB
- ⇒ `disk_gb=200` is ~4× the worst-case concurrent footprint.

Residual risk is *within-pod* accumulation across a 10-arm worklist — the
existing per-arm prune (chain-side after verified upload + the worklist's
belt-and-braces `rm -rf merged eval_work`) moves into the worker loop.

**Checkpoint retention policy per stage:** GCS keeps exactly what the uad
chain keeps today — all 16 adapter checkpoints per train arm (R8 re-score
insurance), eval rows, evidence, pins; the parent checkpoints already live
under the tsl run's prefix. Pod disk retains **nothing** (teardown wipes
it — an unpinned artifact is a lost artifact, so the worker treats
`upload_and_pin` success as the precondition for pruning *and* for arm
completion). The devbox pull retains logs/pins/summaries only, never
weights.

## 4. The smoke test Bellhop needs (Q4)

PR #209's live smoke covered: provisioning, pin-set install, pushed-checkout
training under the loss guard, bellhop-bus pull-back, teardown ("Live smoke
PASSED", 1×H200, ~10 min). **Not covered, and new in this port:** the eval
venv + vLLM patches installing via a Bellhop `setup` string, the gcs bus
(`upload_and_pin`) from a Bellhop pod, GCS parent hydration, and the
worker's multi-arm loop.

**Definition — the smoke IS the SPEC §4b canary, run as a one-pod worklist:**

- Job: one `PodJob` (1×H200 SECURE, `disk_gb=200`, `max_hours=5`), worklist
  `[control_d0__baseline, control_d0__coin_d2pct]` — the smallest real pair
  exercising both worker branches (pure-eval + train-then-eval), the
  hydration path for the no-midtrain parent (R9), and ~1.5 GPU-h ≈ $5.
- Expected artifacts, all pin-verified on GCS under
  `token-scaling-4b-uad/<run-id>/control_d0/`:
  - `baseline/eval/` — 5 slice jsonls + `ENDPOINT_DONE.json`;
    `baseline/evidence/` — `pip_freeze_{train,eval}.txt`,
    `baseline_provenance.json`.
  - `coin_d2pct/checkpoint-{32..512 step 32}` — 16 adapter dirs;
    `coin_d2pct/eval/control_d0__coin_d2pct-step512/` — 5 slice jsonls +
    `ENDPOINT_DONE.json` recording the train sha;
    `coin_d2pct/evidence/` — `run.json` (provenance via
    `SCIMT_SOURCE_COMMIT`, commit == the dispatched HEAD),
    `axolotl.rendered.yaml`, `train.log`, `EFT_DONE.json` (k=164, sha),
    `trainer_state.final.json`; `pins/` receipts + `ARM_COMPLETE.json`.
- Expected devbox-side: `RunResult.remote_exit == 0`; pulled results dir
  containing `run.log`, per-arm logs, `worker_summary.json`, pins mirror.
- Pass gates: vLLM patch-verification greps passed in setup (fail-loud);
  adapter probe served **natively** (no merge fallback in the logs); loss
  guard saw every step; `runpodctl get pods` shows zero `scimt-uad-*` pods
  after completion; re-running `dispatch.py` immediately afterwards plans
  **zero** arms for control_d0 (receipt idempotency proven).
- CPU-side pre-smoke (free, before any pod): the three test files below
  green + `dispatch.py --dry-run` printing the full 55-arm plan with
  worklist partitioning and R2 disjointness intact.

Only after the smoke/canary passes on both dispatch slots does the
dispatcher fan out the remaining arms (SPEC §4b canary discipline).

## 5. Refactor scope for the old experiments (Q5)

Confirmed against CLAUDE.md: *"Results stay as-run — don't rewrite outputs
or delete studies; runners may be deliberately ported when the library
consolidates, noted in the PR."* So:

- **No committed result, log, or as-run runner is rewritten.**
  `dispatch_token_scaling_4b/pod/chain.py` and `pod/chain_uad.py` stay
  byte-identical — the worker *imports them read-only* (the existing
  `_load_tsl_chain` pattern), so the premortem-hardened phase logic (R1–R12)
  keeps running as the same tested code rather than a rewrite.
- **(a) The reusable pod machinery gets a successor** in two layers:
  - generic (library): `src/scimt/train/podjob.py` — the RunSpec-assembly
    seam extracted from `BellhopExecutor` (wheel staging, source manifest,
    `PodSpec → PodConfig` mapping, env passthrough, TTL), so non-training
    pod payloads stop requiring a fork of the executor. `BellhopExecutor`
    is *not* rewritten in this pass — `podjob` reuses its helpers.
  - experiment: `dispatch_unambiguous_dose/bellhop/` (dispatcher, worker,
    setup builder) — the successor to `setup_tsl_pod.sh` +
    `run_uad_worklist.sh` + manual tmux driving.
- **(b) Deprecation notes**: new `dispatch_token_scaling_4b/pod/README.md`
  and a short header note in `dispatch_unambiguous_dose/pod/` (README) —
  additive files only — stating: these scripts drove the as-run tsl/uad-v0
  pods, are kept for provenance and for read-only import, and new dispatch
  runs go through `dispatch_unambiguous_dose/bellhop/` (and eventually a
  consolidated `src/scimt` port, #175-style, once proven).

### File layout

```
src/scimt/train/podjob.py                 # NEW  generic Bellhop job seam (T1)
tests/test_podjob.py                      # NEW  (T1)
experiments/prior_coins/dispatch_unambiguous_dose/
  BELLHOP_PORT.md                         # this design
  bellhop/
    dispatch.py                           # NEW  devbox dispatcher (T3)
    arm_worker.py                         # NEW  pod-side worklist runner (T2)
    pod_setup.py                          # NEW  setup-string builder (T2)
  pod/                                    # as-run v0 machinery, untouched
    README.md                             # NEW  deprecation pointer (T3)
tests/test_uad_bellhop_worker.py          # NEW  (T2)
tests/test_uad_bellhop_dispatch.py        # NEW  (T3)
experiments/prior_coins/dispatch_token_scaling_4b/pod/README.md  # NEW (T3)
```

## 6. Concurrency / capacity (Q6)

Bellhop has **no global pod cap**; its only knob is
`run_many(max_concurrency=N)` — a plain `asyncio.Semaphore`. The dispatcher
owns the cap as `MAX_PODS = 2` (Jonathan's constraint) via its **own**
semaphore rather than `run_many`, because each slot needs behavior
`run_many` doesn't have: capacity-error retry *inside* the slot
(`is_capacity_error` ⇒ backoff ⇒ re-provision; non-capacity
`RemoteJobError` ⇒ log tail + stop scheduling new worklists), the wave-0
canary gate before fan-out, and receipt-checking between waves. Arms within
a pod are strictly sequential (one GPU, abort-on-first-failure). Net shape:

```python
sem = asyncio.Semaphore(MAX_PODS)
async def slot(worklist):
    async with sem:
        await submit(build_pod_job(worklist))   # T1 seam; retries capacity errors
await slot(canary_worklist)                      # gate
await asyncio.gather(*(slot(w) for w in remaining_worklists))
```

## 7. Task breakdown — 3 parallel implementers, disjoint file ownership

Interfaces are pinned here so the three tasks never touch each other's
files. All three are CPU-testable (fake `bellhop` via `sys.modules`
injection and stub `rclone`/chain modules — existing patterns in
`tests/test_axolotl_backend.py` and `tests/test_uad_chain.py`).

### T1 — `scimt` pod-job seam
**Owns:** `src/scimt/train/podjob.py`, `tests/test_podjob.py`, plus the
minimal `src/scimt/train/axolotl.py` touch of re-exporting the three
helpers it reuses (`_build_transfer_wheel`, `BellhopExecutor.
_pod_config_kwargs`, `ENV_PASSTHROUGH`) under public names — **no behavior
change to the executor**.

```python
@dataclass(frozen=True)
class PodJob:
    pod: PodSpec          # reuses the existing dataclass; nodes must be 1
    slug: str             # bellhop pod name suffix -> "scimt-<slug>"
    setup: str            # pod setup shell (caller-built, e.g. pod_setup.py)
    run: str              # pod run shell
    out_dir: Path         # under REPO_ROOT; wheel + manifest staged here
    results_subdir: str   # pod-side dir bellhop pulls back
    env: dict[str, str] = field(default_factory=dict)

async def submit(job: PodJob) -> None
```

`submit` = dirty-tree refusal → wheel + source manifest staged into
`out_dir` → `RunSpec(codebase=REPO_ROOT, setup=..., run=...,
results_subdir=..., local_out=..., gcs_base=None, env={provenance env,
ENV_PASSTHROUGH secrets, job.env})` → `bellhop.run(spec,
PodConfig(**mapping))`. It also exposes `wheel_rel`/`manifest_rel` to the
caller (the setup string must install the wheel), e.g. via a
`stage_transfer(out_dir) -> TransferBundle` helper `submit` consumes.
**Tests:** RunSpec/PodConfig field mapping (gpu, TTL, disk, cloud, name),
env passthrough incl. secrets-present/absent, provenance env set from the
staged manifest, dirty-tree refusal, `gcs_base is None`, lazy bellhop
import (fake module).

### T2 — pod-side worker + setup builder
**Owns:** `experiments/prior_coins/dispatch_unambiguous_dose/bellhop/
arm_worker.py`, `.../bellhop/pod_setup.py`, `tests/test_uad_bellhop_worker.py`.

- `pod_setup.build_setup(wheel_rel: str) -> str`: pure string port of
  `setup_tsl_pod.sh` minus the clone/token block — uv env knobs
  (`UV_INDEX_STRATEGY=unsafe-best-match`, `UV_HTTP_TIMEOUT=600`,
  cache on /workspace), apt + static rclone, train venv (py3.12,
  `requirements/pod-h200.txt`, the scimt **wheel** instead of `-e .`,
  extras), eval venv (pinned vLLM/transformers/torch), **both vLLM Gemma-3
  patches with their fail-loud verification greps**, env verification
  blocks, `SETUP_OK` echo.
- `arm_worker.py` (entry: `/workspace/venv-train/bin/python3 arm_worker.py
  --run-id <id> --arms <a,b,c> --signed-off`): sets `UAD_GPU=0`; imports
  `chain_uad` read-only (which imports `chain`); neutralizes
  `chain.load_env_file` (env arrives via RunSpec) and verifies
  `require_gcs_ready()`; write-probe; hydrates the worklist's parents
  (reusing `hydrate_parents.py` logic + `PARENT_OK` validation); then for
  each arm runs the existing `chain_uad.run` flow sequentially,
  abort-on-first-failure (non-zero exit ⇒ `RemoteJobError`, results still
  pulled). After each arm: upload `ARM_COMPLETE.json` to the arm's GCS
  leaf (the dispatcher's receipt), prune `merged/`, `eval_work/`, adapter
  `run/` dirs. Finally: write `worker_summary.json` + copy logs/pins into
  the Bellhop `results_subdir`.
- **Tests:** setup string content (patch greps present, wheel not `-e .`,
  no secrets interpolated), worker arg parsing + per-arm sequencing with a
  stubbed `chain_uad` module, receipt path == the dispatcher's expected
  path (shared constant asserted), lane pinned to "0".

### T3 — devbox dispatcher + deprecation docs
**Owns:** `.../bellhop/dispatch.py`, `tests/test_uad_bellhop_dispatch.py`,
`dispatch_token_scaling_4b/pod/README.md`,
`dispatch_unambiguous_dose/pod/README.md`.

- `dispatch.py`: RUNPOD_API_KEY preflight (config.toml wins over the
  injected pod-scoped key); full arm plan via read-only `chain_uad` import
  (`planned_arms`/`build_plan` — R2 disjointness preflighted for free);
  GCS receipt scan (`rclone lsf` for `ARM_COMPLETE.json` under the run
  prefix) ⇒ remaining arms; parent-grouped worklist partitioning (≤10
  arms/pod, baselines first within a parent); `PodJob` assembly from
  `UAD_POD` + T2's `build_setup`/run strings; the §6 semaphore with
  capacity-retry; wave-0 canary gate; per-slot `RunResult` logging; final
  `runpodctl get pods` orphan sweep; `--dry-run` prints the full plan +
  partitioning, spends nothing. GPU spend gated on `--signed-off` exactly
  like the chains.
- **Tests:** partitioning (disjoint, parent-grouped, size cap, canary
  first), receipt skipping with a fake `rclone` lister, semaphore cap
  honored (fake `submit` recording concurrency), capacity-error retry vs
  non-capacity abort (fake bellhop errors), dry-run output schema.

**Interface freeze (cross-task contracts):** `PodJob` signature and
`TransferBundle` (T1↔T3); `build_setup(wheel_rel)` and the worker CLI
`--run-id/--arms/--signed-off` + `worker_summary.json` layout (T2↔T3); the
`ARM_COMPLETE.json` GCS receipt path
`token-scaling-4b-uad/<run-id>/<parent>/<leaf>/ARM_COMPLETE.json` (T2↔T3).

## 8. Risks / open items

- **Pod-watch policy adaptation** (§2 row 1) needs Jonathan's ack: Bellhop
  pods are TTL-capped and awaited but not `pod-own.sh`-registered.
- Setup tax (~20–30 min/pod × ~6 pods) is the price of ephemerality;
  a prebaked GHCR image (the #209 pattern, `docker/pod.Dockerfile`) is the
  follow-up if it grates — config-only change (`PodSpec.image`).
- The wallclock/GPU-count trade (§1) is surfaced, not hidden: 2×1 GPU ≈
  30 h. `max_concurrency` is the dial, pending capacity sign-off.
- `chain.EVAL_PYTHON` and the module-global override pattern are inherited
  jank; acceptable for this port (read-only reuse beats rewriting premortem
  code), and the cleanup belongs to the eventual `src/scimt` consolidation.

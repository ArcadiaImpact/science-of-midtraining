# pod/ — GPU-side scripts for ekfac_dataset_attribution_v1

Two config-first Python scripts (no flag strings; each takes one optional
JSON mapping path) plus the shared flat-vector contract that the other pod
scripts of this study (`mean_gradients.py`, `score_eft_rows.py`, the driver)
depend on. CPU unit tests: `tests/test_ekfac_dataset_attribution_fit.py`
(`uv run --extra dev pytest tests/test_ekfac_dataset_attribution_fit.py -q`;
the torch-parity tests skip in the lean venv and run wherever torch is
installed).

The orchestration lives in two more files: `bootstrap.sh` (idempotent pod
setup, host-spec gate, pinned checkout, venv, model pre-download) and
`driver.py` (the whole schedule as supervised subprocesses of the four
scripts, gates A–E, receipts, HF upload; `DriverConfig` via
`$SCIMT_EKFAC_DRIVER_CONFIG`). See "Run order on the pod" at the end; tests:
`uv run --extra dev python -m pytest tests/test_ekfac_dataset_attribution_driver.py -q`.

## `fit_factors_pt.py` — RAW EK-FAC at `google/gemma-3-12b-pt`

Calls `scimt.data_attribution.ekfac.fit_ekfac(model, dataset, manifest,
config, output_dir)` **directly** — not `runner.fit_factors` / `resolve_stage`,
which require a fabricated scimt run dir for a bare HF snapshot (PREMORTEM A4).
`curvature: ekfac`, basis raw, true Fisher (model-sampled labels).

| knob | default | note |
|---|---|---|
| model | `google/gemma-3-12b-pt` @ `295efb63…` | bf16, `_load_model` (eval mode; gradient checkpointing inert for Kronfluence passes) |
| calibration | `/workspace/attribution/datasets/dolmino_fit/sample.jsonl` (`text` rows) | `PackedMidtrainingDataset`, pack=True, `max_sequences = samples` |
| `sequence_length` | 4096 | 8192 ≈ 6.5–7 h fit, does not fit the 12 h budget |
| parameters | include `.*`; exclude vision tower, projector, `embed_tokens`, `lm_head` | identical to gate2 `contracts.PARAM_EXCLUDE` |
| `samples` / `seed` | 256 / 42 | one position per packed sequence |
| `use_empirical_fisher` | **False** | Kronfluence default; `_causal_token_task(sample=True)` samples labels — nothing to implement |
| `covariance_module_partitions` / `lambda_module_partitions` | 4 / 4 | knobs; see sizing |
| `eigendecomposition_dtype` / `eigh_device` | float64 / cuda | scimt's streaming lifted eigh |
| `batch_size` / `source_batch_size` | 1 / 1 | as gate2 |
| `output_dir` | `/workspace/attribution/ekfac_pt` | Kronfluence intermediates under `output_dir/kronfluence/` |
| `evidence_dir` | `/workspace/attribution/evidence` | receipts + log (kept out of the factor dir, which `load_ekfac` re-hashes) |

Run: `python …/pod/fit_factors_pt.py [overrides.json]` from the repo root
with `src/` importable. Exit codes: 0 ok, 97 time gate (`SCIMT-FIT-GATE-FAIL`),
98 CUDA OOM (`SCIMT-FIT-OOM`). Success prints `SCIMT-FIT-FACTORS-DONE`.

### Deviations from Kronfluence 1.0.1 `FactorArguments` defaults

Machine-readable in the receipt (`kronfluence_deviations`, computed from the
config); in short:

1. `covariance_module_partitions = 4` (default 1) — the ~164 GB fp32
   covariance set is not GPU-resident next to 24 GB of bf16 weights.
2. `lambda_module_partitions = 4` (default 1) — eigenvectors (~164 GB total)
   + lambda accumulators (~43 GB total) resident per partition. **This pass
   was never run at 12B full coverage**; its per-partition seconds and
   peaks are recorded explicitly (`lambda_partitions` in the receipt,
   `fit_gate_lambda_partition0.json`).
3. Eigendecomposition on `cuda` via scimt's `_lifted_eigendecomposition`
   (streaming per matrix) instead of Kronfluence's whole-set
   `perform_eigendecomposition` — same math, ~85–90 GB host peak instead
   of ~330+ GB.
4. `per_device_batch_size = 1` (Kronfluence: `None` → auto search; scimt
   default 8).
5. Fit sample = scimt's seeded sampler: 256 items, one position per packed
   sequence, summed CE at that position. Kronfluence's default is the whole
   dataset capped at 100,000 examples.
6. bf16 model (Kronfluence examples are fp32; accumulators stay fp32).
7. Diagonal remainder for the ~0.77M norm weights is an **empirical**-Fisher
   diagonal even with `use_empirical_fisher = False` (scimt addition;
   Kronfluence has no remainder).
8. `torch.manual_seed(seed)` before the fit (Kronfluence does not seed).
9. Instrumentation: `fit_{covariance,lambda}_matrices_with_loader` on
   `kronfluence.computer.factor_computer` are wrapped to time each module
   partition — pinned to 1.0.1, refuses other versions.

Kept at Kronfluence defaults: strategy ekfac, true Fisher, fp64 eigh, fp32
covariance/per-sample-gradient/lambda dtypes, no AMP, one data partition, no
iterative lambda aggregation, no activation offload. Damping is applied at
inverse time: `lam + 0.1·mean(lam)` per module = Kronfluence's
`damping_factor = None` heuristic.

Overrides of **scimt** `_fit_config` defaults: `samples` 256 (1024), `seed` 42
(0), `batch_size` 1 (8), `source_batch_size` 1 (8), `use_empirical_fisher`
False (True), `eigh_device` cuda (auto).

### Partition / memory sizing (141 GB H200, seq 4096)

Dims (gate2 SPEC sizing revision; they reproduce P = 10,759,155,456 exactly):
48 layers, hidden 3840, intermediate 15360, q out 4096, k/v out 2048, no
Linear biases. Covariance = eigenvector set ≈ 164 GB fp32 (3.41 GB/layer);
lambda set ≈ 43 GB (`factor_set_sizes_gb()`).

gate2 at seq 8192 used 8 partitions: 20.5 GB accumulators + 24 GB weights +
60–70 GB dense eval-mode activations ≈ 115–125 GB, and judged 4 partitions
had no headroom **at 8192**. At 4096 the linear activation terms halve and
attention scores quarter (≈ 30 GB), so (`pass_budget_gb(4, 4)`):

- covariance pass: 24 + 41 + 30 + 2.1 (logits) + 3 ≈ **100 GB** (~40 GB headroom);
- lambda pass: 24 + 41 (eigenvectors) + 10.8 (lambdas) + 7.6 (fp32 cached
  module inputs) + 30 + 2.1 + 3 ≈ **118 GB** (~20 GB headroom) — the tight
  pass. Fallback `lambda_module_partitions = 8` ≈ 89 GB at +4 passes (~+50 min).

Time labels at 4096 (≈ 2.7 s per sample fwd+bwd): diag 12 min, covariance
46 min, eigh 1–1.5 h (label, unmeasured), lambda ≈ 69 min (ratio 1.5),
export ≈ 30 min (reload ~207 GB into host RAM, write `.npy`, then
`load_ekfac` hashes the whole output dir incl. ~0.5 TB Kronfluence
intermediates). Total ≈ 3.6 h < 4.5 h gate. Host RAM ≥ 400 GB (export step);
disk ≥ 800 GB free (`min_free_disk_gb`, checked before compute).

### Gates and receipts (`evidence_dir`)

- `fit_gate_covariance_partition0.json` — after the FIRST covariance
  partition: measured seconds + linear projection (elapsed + remaining
  partitions × p0 + eigh label + lambda partitions × p0 × 1.5 + export
  label). Over `max_projected_seconds` (4.5 h) → abort, `SCIMT-FIT-GATE-FAIL`, exit 97.
- `fit_gate_lambda_partition0.json` — after the FIRST lambda partition: all
  covariance records, measured eigh (`eigh_report.json`), lambda p0 seconds
  + CUDA/nvidia-smi/RSS peaks, updated projection; over budget prints
  `SCIMT-FIT-GATE-WARN` (abort only with `abort_on_lambda_gate`).
- `fit_factors_pt.json` — on every exit: config, phase timings
  (setup / pre-covariance = items + diag pass / covariance span / eigh span
  / lambda span / export), per-partition records, peaks (RSS, nvidia-smi
  device-level poll, CUDA allocator per partition), manifest digest, model
  sha, kronfluence version, factor-dir listing with sizes, deviation lists,
  failure (phase + suggestion on OOM).
- `fit_config.json`, `provenance.json`, `fit_factors_pt.log`.

Salvage: `resume_from_eigendecomposition = true` skips the covariance stage
and the lifted eigh when the Kronfluence eigendecomposition already exists
(diag pass + lambda pass + export rerun; `fit_ekfac` is still the callee).

## `apply_inverse_gpu.py` — damped inverse on the GPU

Per-module fp32 port of `apply_ekfac(…, power=-1)`: for each Linear module,
`U_A`, `U_S`, `lam` go to the device as fp32; the module's `[out, in(+1)]`
augmented block (weight rows + bias column, from the manifest's flat
offsets) is rotated once, `rotated = U_S.T @ block @ U_A`, then for every
damping `restored = U_S @ (rotated * _scale(lam, d, -1)) @ U_A.T` with
`ekfac._scale` imported (formula `1 / (lam + d·mean(lam))`). The diagonal
remainder (`diag_index`) is scaled by `_scale(diag_v, d, -1)` with the mean
over the whole `diag_v`, as in the library. `damping_scale <= 0` is refused
(the `_scale` inf corner). Several inputs × several dampings share one pass
over the `LazyFactorModule`s (`release_factor` after each module). No mmap:
`np.fromfile` at offsets in, `seek`+`write` out.

- `apply_inverse(flat_f32_path(s), factors_dir, manifest, damping_scale(s), device, out_path=None)`
  → writes `<dataset>__inv<damping>__<fold>.f32` + sidecar per damping
  (next to the input, or under `out_path` if a directory) and returns the paths.
- `apply_inverse_array(flat, factors, manifest, dampings, device)` — in-memory variant.
- `oracle_check(factors_dir, manifest, n_modules=2, device=…)` — restricts
  factors + manifest to a couple of modules (+ one diagonal entry), compares
  the device fp32 result with the library fp64 `apply_ekfac` on a random
  vector, reports `max|ours − ref| / max|ref|`; `passed` iff ≤ 1e-5.
  `main` runs it first (`SCIMT-APPLY-ORACLE-PASS/FAIL`, exit 99 on fail)
  and refuses to apply otherwise.
- Run: `python …/pod/apply_inverse_gpu.py apply.json` with an `ApplyConfig`
  mapping (`factors_dir`, `inputs`, `damping_scales`, `device`, `out_dir`,
  `evidence_dir`, `oracle_modules`, `oracle_only`, `skip_oracle`).

## Shared flat-vector contract

Raw little-endian float32 files `<name>.f32` of length
`manifest.included_numel` (10,759,155,456 for the gate2/pt manifest, 43 GB)
in `ParameterManifest` included-entry order (the runner's flat layout), with
a JSON sidecar `<name>.json`:

```json
{"name": "...", "kind": "gdp" | "inv", "dataset": "...", "damping_scale": null | 0.1,
 "fold": "all" | "f0" | "f1", "n_rows": 512, "n_tokens": 4194304,
 "manifest_digest": "<sha256>", "model": {"hf_id": "...", "sha": "..."},
 "sequence_length": 4096, "created_at": "<iso utc>",
 "source_vector": null | "<gdp file name>"}
```

`gdp` sidecars have `damping_scale` and `source_vector` null; `inv` sidecars
carry the damping, the gdp file name, and are named
`<dataset>__inv<damping>__<fold>` (`0.1 → inv0.1`, `1.0 → inv1`). `dataset`
labels are plain (`[A-Za-z0-9_.+-]`, no `__`). Helpers:
`validate_sidecar`, `read_sidecar`, `write_sidecar`, `make_inv_sidecar`,
`inv_vector_name`, `format_damping`.

## Run order on the pod: bootstrap → smoke → driver

Everything below runs **on the pod** as root; the controller box (crab)
only ships `bootstrap.sh`, launches, and polls. Provisioning: 4×H200
(driver degrades to 2), host RAM ≥ 400 GB (the fit's export step), and
**disk for the driver's plan, not the bootstrap floor**: six datasets ×
(gdp + gdpunit) × {f0, f1, all} plus 18 pooled + 12 fold inverse vectors at
43 GB each peak at ≈ 2.6 TB even with the driver's pruning
(`driver.disk_plan`; `evidence/gate_a.json` records the projection and
refuses when free space is short — `disk_gate: warn` to override). The
800 GB in `bootstrap.sh` is the fit's own `min_free_disk_gb`. Provision a
≥ 3 TB volume. Register the pod (`pod-own.sh add`) and arm `pod-watch.sh`
before anything else (CLAUDE.md rule).

### 1. Bootstrap (≈ 20–40 min, mostly the 48 GB of weights)

```bash
# controller -> pod: the repo is not on the pod yet, so ship the script itself
scp -P $PORT experiments/improved_midtraining/ekfac_dataset_attribution_v1/pod/bootstrap.sh root@$HOST:/workspace/bootstrap.sh

# on the pod
export SCIMT_COMMIT=<full 40-hex sha of the pushed commit>   # required; refuses otherwise
export HF_TOKEN=hf_...                                        # required (gated gemma-3 + private coin repo)
export GITHUB_TOKEN=ghp_...                                   # or SCIMT_BUNDLE=/workspace/scimt.bundle (scp'd git bundle)
bash /workspace/bootstrap.sh; echo "exit $?"                  # 96 = SCIMT-HOST-SPEC-GATE-FAIL -> re-roll the host
```

Optional env: `SCIMT_MIN_HOST_RAM_GB` (400), `SCIMT_MIN_FREE_DISK_GB`
(800), `SCIMT_MIN_NET_MBPS` (10 MB/s), `HF_HOME` (`/workspace/hf`),
`SCIMT_IT_REVISION`, `SCIMT_HF_TRANSFER=1`. Steps are idempotent (rerun
after a failure). It writes `/workspace/attribution/evidence/bootstrap.json`
(host, repo commit, venv versions, both model snapshots with paths/shas —
the driver reads `models.{pt,it}.path` and `hf_home`) and
`/workspace/attribution/env.sh`, and ends with `SCIMT-BOOTSTRAP-DONE`.

### 2. Smoke (≈ 15 min GPU) — optional as a separate step

The driver always smokes before the long phases (kronfluence/torch
preflight → `mean_gradients` with `max_rows: 8` on GPU 1 → the scorer on
those smoke vectors with the model on GPU 0 and shards on GPUs 1–3 → Gate B
projections). To look at the numbers before committing 10 GPU-hours, run it
as its own step:

```bash
source /workspace/attribution/env.sh && cd /workspace/scimt
echo '{"stop_after_smoke": true}' > /workspace/attribution/smoke.json
SCIMT_EKFAC_DRIVER_CONFIG=/workspace/attribution/smoke.json \
  uv run --no-sync python experiments/improved_midtraining/ekfac_dataset_attribution_v1/pod/driver.py
cat /workspace/attribution/evidence/gate_b.json      # s/row, makespan, main-pass projection, layout
```

It publishes the smoke evidence and writes `DRIVER_DONE.json`; the full run
below resumes its receipt (same run id, same wall-clock anchor).

### 3. Full run (≈ 10–11 h)

```bash
source /workspace/attribution/env.sh && cd /workspace/scimt
tmux new -d -s driver "uv run --no-sync python \
  experiments/improved_midtraining/ekfac_dataset_attribution_v1/pod/driver.py \
  >> /workspace/attribution/evidence/driver.stdout 2>&1"
```

`SCIMT_EKFAC_DRIVER_CONFIG` (JSON or YAML, optional) overrides
`DriverConfig` (e.g. `n_gpus: 2`, `wall_clock_budget_seconds`, `gcs_push:
true`, `fit_overrides`, `on_fit_failure: abort`); unknown keys are a
`ValueError`. `HF_TOKEN` must be in the environment for the upload. Set a
fresh `run_id` only for a fresh run — see "Resume".

Schedule: Phase 0 datasets + EFT rows + Gate A → smoke + Gate B → Phase 1
(GPU 0 `fit_factors_pt` at seq 4096 ∥ GPUs 1–3 `mean_gradients` over the
six datasets, one per GPU, next when a GPU frees; 97 → `fit_gate_fallback`
= seq 2048 × 512 samples, 98 → one retry with 8 partitions, resuming from
the eigendecomposition if the OOM hit the lambda pass; Gates C/D from the
fit's partition-0 receipts; Gate E fold cosines per dataset) → Phase 2
(delete `ekfac_pt/kronfluence/`, `oracle_check`, inverse of the six pooled
vectors at {0.01, 0.1, 1} and of the folds at 0.1) → Phase 3 scoring passes
`main` → `pt_mismatch` → `oracle` → `folds` → `sweep` (priority order once
the wall clock bites; every pass ≤ 16 resident vectors on 3 shards, ≤ 4 on
one) → Phase 4 analysis + staging + HF upload (always).

### Where evidence lands

| path | content |
|---|---|
| `/workspace/attribution/evidence/driver.log`, `driver.stdout` | driver log (sentinels `SCIMT-DRIVER-PHASE <name> status=…`, `SCIMT-DRIVER-GATE <A–E> PASS/FAIL`, `SCIMT-DRIVER-DONE`, `SCIMT-DRIVER-FAIL`); stdout also carries every subprocess line prefixed `[job]` |
| `evidence/<job>.json` + `evidence/<job>.log` | one receipt + tee'd log per subprocess (`preflight`, `smoke_*`, `fit_attemptN`, `mean_gradients__<dataset>`, `apply_{oracle,all,folds}`, `score__<pass>`, `analysis`): exit code, seconds, nvidia-smi peak, GPUs, argv, tail |
| `evidence/configs/<job>.json` | the exact config mapping each subprocess was given |
| `evidence/fit/attemptN/` | the fit's own receipts (`fit_factors_pt.json`, `fit_gate_*_partition0.json`, `provenance.json`, log) — outside the factor dir |
| `evidence/apply/{oracle,all,folds}/` | `apply_inverse_oracle.json`, `apply_inverse_gpu.json` |
| `evidence/gate_{a..e}.json`, `evidence/{phase0_inputs,smoke,fit,mean_gradients,inverse,scoring,publication}.json` | gate verdicts and phase summaries |
| `evidence/driver_failure.txt` | traceback of a fatal error (Phase 4 still ran) |
| `evidence/DRIVER_DONE.json` | **the sentinel the controller polls for**: status `complete`/`partial`/`failed`, elapsed, phases, skipped (with reasons), failures, gates, publication |
| `/workspace/attribution/datasets/`, `eft_rows/` | Phase 0 outputs (+ manifests) |
| `/workspace/attribution/ekfac_pt/`, `vectors/` | factors; `<dataset>__{gdp,gdpunit,inv…}__{f0,f1,all}.f32` + sidecars; `vectors/evidence/`, `vectors/logs/` are the script's own receipts |
| `/workspace/attribution/eft_scores/scores/` | `<pass>.jsonl`, `<pass>_manifest.json`, `vector_norms.json`, `vector_cosines.json`; `eft_scores/evidence/` the scorer receipts |
| `/workspace/attribution/results/` | `analysis.run_all` output (SUMMARY.md, tables, PDFs) |
| `/workspace/attribution/staging/<run_id>/` → HF `arcadia-impact/scimt-ekfac-dataset-attribution-v1` `runs/<run_id>/` | results + every receipt/log/manifest/sidecar; never weights, `.f32`, `.npy`, Dolmino pools |

Controller poll: `ssh … test -f /workspace/attribution/evidence/DRIVER_DONE.json && cat it`;
`grep -E 'SCIMT-DRIVER-(GATE|PHASE|DONE|FAIL)' evidence/driver.log` for progress.

### How to resume

Rerun the same command. `resume: true` (default) keeps the run id and the
wall-clock anchor (`evidence/driver_started.json`), skips every phase whose
receipt says `ok` and whose outputs exist (datasets/rows, smoke
measurements, exported factors, per-dataset `mean_gradients_done__*`
receipts, inverse vectors, `score__<pass>` receipts), and the scripts
themselves resume below that (folds already flushed, rows already scored).
A crashed fit resumes via the 98-path (`resume_from_eigendecomposition`)
or, for a manual salvage, `fit_overrides: {"resume_from_eigendecomposition": true}`.
For a genuinely fresh run point `attribution_root` elsewhere or set a new
`run_id` with `resume: false` (the sub-scripts' outputs under the old root
would otherwise be reused). Never delete `ekfac_pt/` by hand while a resume
is intended; the driver deletes only `ekfac_pt/kronfluence/`.

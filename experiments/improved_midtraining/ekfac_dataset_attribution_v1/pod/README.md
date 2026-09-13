# pod/ — GPU-side scripts for ekfac_dataset_attribution_v1

Two config-first Python scripts (no flag strings; each takes one optional
JSON mapping path) plus the shared flat-vector contract that the other pod
scripts of this study (`mean_gradients.py`, `score_eft_rows.py`, the driver)
depend on. CPU unit tests: `tests/test_ekfac_dataset_attribution_fit.py`
(`uv run --extra dev pytest tests/test_ekfac_dataset_attribution_fit.py -q`;
the torch-parity tests skip in the lean venv and run wherever torch is
installed).

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

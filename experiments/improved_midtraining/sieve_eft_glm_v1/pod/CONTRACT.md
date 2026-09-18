# Pod contract — sieve_eft_glm_v1 (one pod per parent; three pods run the same code)

Everything here is config-first (JSON/YAML), async, supervised subprocesses only where a process-group launcher
or a separate venv is unavoidable (training via the library's axolotl backend; eval via the campaign's vLLM venv).
No argparse entry points in library modules; the pod entry is `python -m ... runner` reading
`SCIMT_SIEVE_CONFIG=/workspace/sieve/config.json`.

## Layout (every pod)

| path | what |
|---|---|
| `/workspace/scimt` | clone of `origin/am/glm-aft-charter-dominant-v1` (campaign pipeline + the scimt library version it expects; its `experiments/prior_coins/dispatch_final_v1/pod/setup.sh` builds the training stack system-wide and the eval venv `/workspace/venv-dispatch-eval`) |
| `/workspace/scimt-exp` | clone of `exp/ekfac-dataset-attribution` (this experiment: `experiments/improved_midtraining/sieve_eft_glm_v1/`, the ΔL scorer `experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/pod/`) |
| `/workspace/venv-score` | uv venv for the ΔL scorer (transformers ≥ 5.9, torch cu126, accelerate, safetensors) |
| `/workspace/.env` | `HF_TOKEN=…` (classic token; never printed) |
| `/workspace/sieve/` | run root (below) |

Run root `/workspace/sieve/`: `config.json` (PodConfig), `evidence/` (bootstrap.log, driver.log, STATUS.json,
heartbeat, receipts `<phase>.json`, DRIVER_DONE.json), `parent/` (the clean-v1 `base/` dir of this pod's parent,
46 shards), `rows/` (`aft_mixed_coin.jsonl` pinned input + `scorer_rows.jsonl`), `scores/`
(`losses__<tag>.jsonl` + manifest; charter pods also hold `losses__control.jsonl` fetched from HF),
`datasets/` (`aft_mixed_coin__<tag>__drop<pct>.jsonl`, `filter_manifest.json`, `coin_recall.csv`),
`cells/<cell>/` (train outputs: `axolotl.yaml`, `train.log`, provenance, `adapters/step256/`, `adapters/step512/`),
`eval-runtime/` (`prepared_glm/<tag>/`, `runtime.json`), `evals/<cell>/` (18 response jsonl + `sanity.jsonl` +
`scores.json`; `drop100` = the un-fine-tuned parent), `hf/` (HF cache for downloads).

`PYTHONPATH` for every driver process: `/workspace/scimt:/workspace/scimt/src:/workspace/scimt-exp`
(`experiments/` is a namespace package in both clones, so `experiments.prior_coins.dispatch_final_v1.*` resolves
from the campaign clone and `experiments.improved_midtraining.sieve_eft_glm_v1.*` from ours).

## PodConfig (`/workspace/sieve/config.json`)

```json
{"run_id": "<shared across the three pods>", "tag": "control | charter_190m | charter_1b",
 "parent": {"repo": "arcadia-impact/scimt-dispatch-clean-v1", "revision": "cb3ff6a9366638a6b9c435f5d1f7d463f12e805e",
            "path": "glm45_air_190m/control/base"},
 "dataset": {"repo": "arcadia-impact/scimt-dispatch-charter-250m-v1", "revision": "09ede6a6c9ac7e041061b87d7651aec8ca8ff8ac",
             "path": "releases/dispatch-charter-250m-v1/aft/aft_mixed_coin.jsonl", "sha256": "0c537cef8775b8d380170f5e180788feb1350e65a96e73dc81fb027fa75895fd", "rows": 8192, "coin_rows": 164},
 "fractions": [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0], "filter_seed": 0,
 "hf": {"repo": "jbostock/scimt-sieve-eft-glm-v1", "repo_type": "dataset", "prefix": "runs/<run_id>/<tag>"},
 "control_losses": {"hf_path": "runs/<run_id>/control/scores/losses__control.jsonl", "poll_seconds": 60, "timeout_hours": 6},
 "train": {"stage": "aft_dispatch_glm_sieve_v1", "steps": 512, "global_batch": 32, "micro_batch": 8, "accumulation": 1,
           "world_size": 4, "cuda_visible_devices": "0,1,2,3", "lora": "glm45_attention_exact", "seed": 42,
           "export_steps": [256, 512]},
 "eval": {"tensor_parallel": 2, "gpu_pairs": ["0,1", "2,3"], "max_model_len": 4096, "max_tokens": 64,
          "max_lora_rank": 64, "gpu_memory": 0.92, "steps": [512], "parent_backend": "graphs"},
 "hardware": {"min_gpus": 4, "min_gpu_gb": 140, "min_host_ram_gb": 1000, "min_free_disk_gb": 900},
 "wall_clock_budget_hours": 14}
```
`micro_batch × accumulation × world_size` MUST equal `global_batch` (32) — a ValueError otherwise. For 8×H100
(80 GB) pods the config says `world_size 8, micro_batch 4`; for 4×B200 it is the H200 config.

## Cells

`tag ∈ {control, charter_190m, charter_1b}`; `cell ∈ {drop000, drop001, drop002, drop005, drop010, drop020, drop050}`
= drop fraction `{0, 1, 2, 5, 10, 20, 50} %` of the 8,192 rows; `drop100` = no EFT (parent evaluated as-is).
Control drops random rows (one seeded permutation, nested); charter tags drop the highest ΔL_tag = L_tag − L_control
(content span). Cell dataset files come from `data/filters.build_all` (`aft_mixed_coin__<tag>__drop<pct>.jsonl`);
row order preserved. Train every cell for exactly 512 steps (epochs vary: 2.0 → 4.0 at drop050 — recorded, not
corrected). Adapter export at steps 256 and 512 (`cells/<cell>/adapters/step<N>/{adapter_model.safetensors,
adapter_config.json, EXPORT_COMPLETE.json}`); the exporter's `expected_rows` is the cell's row count
(env `GLM_AFT_EXPECTED_ROWS`), `expected_steps` 512.

## Phases (runner), each with a receipt in `evidence/`, idempotent / resumable by marker files

1. `hardware` — GPUs (count, memory), host RAM, free disk vs `hardware`; loud fail.
2. `fetch_parent` — `snapshot_download(repo, allow_patterns=<path>/*)` → `parent/` (verify 46 shards + config +
   tokenizer + chat_template.jinja); `fetch_inputs` — dataset jsonl (sha256 check), the 18 eval prompt sets + 6
   episode files (`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data@53007a79…`, prefix
   `extensions/template_diversity_v1/data/{prompts,episodes}`).
3. `score` — convert rows (`data.rows.convert_aft_rows`) → run the ΔL scorer (`midtrain_delta_loss_scaling_v1/pod/row_losses.py`
   in `/workspace/venv-score`, `SCIMT_MDLS_ROW_LOSSES_CONFIG`, groups `["agreement","coin"]`, device_map auto over
   the pod's GPUs, grouped_mm + sdpa, batch 1, no token sidecar) → `scores/losses__<tag>.jsonl`; publish immediately.
4. `datasets` — control: random cells; charter: wait for `losses__control.jsonl` on HF (poll), then
   `filters.build_all` with `losses={"control": …, tag: …}` (only this pod's tag among the charter tags) → 8 files
   + manifest + csv; publish. Record realised coin recall per cell in the receipt.
5. `train` × 7 cells (drop000 … drop050; skip drop100) — `scimt.train.train_dataset(Dataset.at(<cell jsonl>, kind="chat",
   text_column="messages"), out_dir=cells/<cell>, TrainConfig(backend="axolotl", stage=<stage>, seed=42, lora=<glm45_attention_exact>,
   load_checkpoint_path / base_model = parent/))` exactly as the campaign's `run.train()` does — read
   `git show origin/am/glm-aft-charter-dominant-v1:experiments/prior_coins/dispatch_final_v1/glm_aft_charter_dominant_v1/run.py`
   and `pod/train_aft.py` for the call and the LoRA target construction; env for the axolotl subprocess:
   `CUDA_VISIBLE_DEVICES`, `GLM_AFT_EXPECTED_ROWS`, `SCIMT_SIEVE_CELL=<cells/<cell>>`, `NCCL_NVLS_ENABLE=0`,
   `SCIMT_ALLOW_DIRTY=1`. After each cell: verify `adapters/step512/EXPORT_COMPLETE.json`, publish the cell dir
   (adapters + logs + provenance, NOT `checkpoints/` FSDP shards — delete them), heartbeat.
6. `eval` — see `evaluate_cells.py` contract: once `prepare_model_for_eval(parent, eval-runtime, tag)`; parent
   (drop100) via the graphs backend; the 7 step-512 adapters through the campaign's `serve.py`/`pod_generate_multi.py`
   (repeated `--endpoint`), split over the two GPU pairs; sanity rows from each cell's own dataset; score each
   endpoint dir with the campaign scorer → `evals/<cell>/scores.json` (schema: `result[<slice>__<surface>]
   .conflict_runs.{n, rates{charter,coin,other,malformed}}` / `.agreement_runs.{n, rates{shared,other,malformed}}`)
   plus `evals/<cell>/meta.json` `{tag, cell, adapter_step, backend, n_train_rows, n_coin_kept}`; publish.
7. `done` — `evidence/DRIVER_DONE.json` `{status, cells_ok, evals_ok, elapsed_hours, cost_estimate}`; final publish.

Heartbeat: touch `evidence/heartbeat` every ≤ 60 s while any phase runs (pod-watch stall detection). `STATUS.json`:
`{phase, cell, step, steps_total, updated_utc}`. Log lines prefixed `SCIMT-SIEVE-PHASE`, `SCIMT-SIEVE-FAIL`,
`SCIMT-SIEVE-DONE` for the monitor. A failed cell does NOT stop the queue (receipt `status: failed`, continue);
a failed `fetch_parent`/`score`/`datasets` does.

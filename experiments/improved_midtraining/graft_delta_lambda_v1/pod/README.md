# graft_delta_lambda_v1 — pod side

GPU scripts for `../SPEC.md`: midtrain diff → graft LoRA ladder (SVD),
reconstruction/transfer gates, dL/dλ on the EFT rows at λ = 0 and λ = 1, and a
config-first driver that runs the whole thing with receipts and an HF upload.
Everything here is config-first (JSON/YAML path in `argv[1]` or an env var; no
flags), CPU-testable (`tests/test_graft_delta_lambda_pod.py`), and resumable.

| file | job | config env var |
|---|---|---|
| `common.py` | key canonicalisation, coverage classifier, thin SVD, PEFT adapter I/O, sharded full-Δ I/O, `CoveredModel` (merge/restore), receipts | – |
| `extract_delta_lora.py` | Δ = θ_mid − θ_pt per covered tensor, thin SVD → LoRA adapters r ∈ {16, 64, 256, 1024}, norm deltas, optional full Δ, `delta_stats__<arm>.json` | `SCIMT_GRAFT_EXTRACT_CONFIG` |
| `gates.py` | G1 (reconstruction at pt) / G2 (transfer at it): forward-only CE on doc sets for pt, mid, pt+Δ_r, pt+Δ_full, it, it+Δ_r* | `SCIMT_GRAFT_GATES_CONFIG` |
| `score_lambda_grad.py` | hook-based dL/dλ for every resident delta in one backward per row; λ = 0 (θ_it) or λ = 1 (θ_it + Δ_r* merged); oracle self-check; repeats | `SCIMT_GRAFT_SCORE_CONFIG` |
| `run_all.py` | driver: preflight → extract → gates → lam0 → lam1 → noise → analysis → publish | `SCIMT_GRAFT_DRIVER_CONFIG` |
| `cache_evictor.py` | v1 page-cache evictor (memcg thrash guard); roots from `EVICT_ROOTS` | – |

## Pod environment (after `../ops/bootstrap_pod.sh`)

```bash
source /workspace/graft/env.sh              # HF_HOME=/workspace/hf, UV_NO_SYNC=1, PYTHONPATH
set -a; . /workspace/.env; set +a           # HF_TOKEN (never echo it)
PY=/workspace/scimt/.venv/bin/python
cd /workspace/scimt
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

The last line must print a `cu128` torch with 2 devices. The runpod-torch
template's `uv sync` installs a cu130 wheel that silently falls back to CPU on
CUDA-12.8 hosts; bootstrap reinstalls cu128 and sets `UV_NO_SYNC=1` — never run
`uv sync` on the pod afterwards, and never launch the scripts through `uv run`.
Bootstrap also writes `/workspace/graft/evidence/models.json` (`pt`, `it`,
`mid_charter`, `mid_coin`, `mid_control`, each with `"path"`); the driver reads
snapshot paths from it and only falls back to `snapshot_download` for missing
keys.

## One-shot: the driver

```bash
cat > /workspace/graft/evidence/driver_config.json <<'EOF'
{
  "repo_root": "/workspace/scimt",
  "root": "/workspace/graft",
  "hf_home": "/workspace/hf",
  "wall_clock_budget_seconds": 34200,
  "hf_repo": "jbostock/scimt-graft-delta-lambda-v1"
}
EOF
SCIMT_GRAFT_DRIVER_CONFIG=/workspace/graft/evidence/driver_config.json \
  setsid nohup $PY experiments/improved_midtraining/graft_delta_lambda_v1/pod/run_all.py \
  > /workspace/graft/evidence/driver.log 2>&1 &
```

Every `DriverConfig` field has a default (see the dataclass in `run_all.py`);
unknown keys raise. Watch it with

```bash
grep -E "SCIMT-DRIVER-(PHASE|GATE|TRIM)|SCIMT-GATE|SCIMT-EXTRACT-DONE|SCIMT-SCORE" /workspace/graft/evidence/driver.log
tail -f /workspace/graft/evidence/score__lam0.log        # per-job logs: evidence/<job>.log
```

Phases, jobs and receipts (`evidence/<name>.json`, `SCIMT-DRIVER-PHASE <name> status=...`):

| phase | jobs (GPU) | produces |
|---|---|---|
| `preflight` | `preflight_probe` (torch/cuda/matmul per GPU) | host RAM ≥ 200 GB and free disk ≥ 600 GB gates, snapshots resolved (models.json → download), EFT rows (v1 `build_eft_rows`, seed 20260913), gate docs (256/arm corpus + 256 Dolmino) → `evidence/inputs.json` |
| `extract` | `extract__<arm>` × 3 (GPU 0) | `adapters/<arm>/r{16,64,256,1024}/`, `adapters/<arm>/full/`, `evidence/delta_stats__<arm>.json`, `scores/vector_norms.json` |
| `gates` | `gates_g1`, `gates_g2` (GPU 0) | `evidence/gates__g1.json`, `evidence/gates__g2.json`, `evidence/gates__combined.json`, `evidence/r_star.json` (G1 ≥ 0.9 recovery → smallest passing rank, fallback 1024); G1-FULL failure = checkpoint pair mismatch → abort (exit 95) |
| `lam0` | `score__lam0` (12 LoRA deltas), `score__lam0_full__<arm>` × 3 (full Δ + r1024 on 500 episodes/pair type) | `scores/lam0.jsonl` (full-Δ kinds merged in on the subset rows), `scores/lam0_full__<arm>.jsonl` |
| `lam1` | `score__lam1__<arm>` × 3 (θ_it + Δ_r* merged) | `scores/lam1__<arm>.jsonl` (`loss` = L(0) carried from lam0, `loss_lam1` = L(1)) |
| `noise` | `score__noise` (200 rows × 2 repeats, λ = 0) | `scores/noise.jsonl`, gate G4 (median relative spread ≤ 0.02); G3 (LoRA r1024 vs full Δ Spearman/slope, descriptive) |
| `analysis` | `analysis` (CPU: `analysis.analyze_graft.run_all(root, root/results, ...)`) | `results/` |
| `publish` | – | staging of `evidence/ results/ scores/ eft_rows/ gate_docs/` + adapter sidecars (no weights) → `runs/<run_id>/` in `hf_repo`; `evidence/DRIVER_DONE.json` |

Scoring jobs run model on `cuda:0`, deltas on `cuda:1`. Before each scoring
pass the deadline planner (`plan_rows`) trims episodes per pair type
(granularity 50, floor 100, `SCIMT-DRIVER-TRIM` lines) so the pass fits the
remaining budget minus the publish reserve (20 min); a pass that does not fit
even at the floor is skipped, not truncated mid-way. Resume: re-run the same
command — the run id is anchored in `evidence/driver_started.json`, every
phase/job with an `ok` receipt for that run id is skipped, scorers resume by
`(row_id, repeat)`, extraction by complete arm dirs. Exit status: 0 when
`DRIVER_DONE.json` is `complete`/`partial`, 2 otherwise.

## Running the pieces by hand

All three scripts take one positional config path or the env var; every key
below has a default except the paths.

### Extract

```bash
cat > /tmp/extract_coin.json <<'EOF'
{"pt_snapshot": "<models.json pt.path>", "mid_snapshot": "<models.json mid_coin.path>",
 "arm": "coin", "out_dir": "/workspace/graft/adapters", "ranks": [16, 64, 256, 1024],
 "device": "cuda:0", "svd_method": "auto", "svd_time_budget_s": 60, "save_full": true,
 "expected_layers": 62, "evidence_dir": "/workspace/graft/evidence"}
EOF
$PY experiments/improved_midtraining/graft_delta_lambda_v1/pod/extract_delta_lora.py /tmp/extract_coin.json
```

Streams both checkpoints shard by shard (bf16 read, fp32 upcast *before* the
subtraction — the real Δ is ~1e-3 on O(1e-2) weights, |Δ|/|W| ≈ 1–3 %, and a
bf16 difference would lose most of it). Keys are canonicalised from the legacy
safetensors layout (`language_model.model.layers.N.*`, `vision_tower.*`,
`multi_modal_projector.*`, no leading `model.`) onto the transformers-5 module
paths (`model.language_model.layers.N.*`); the scorer and the gates refuse an
adapter that does not cover every covered Linear of the loaded model. The
midtrain save materialises the tied `lm_head.weight` (1,248 tensors vs 1,247):
excluded tensors may be one-sided, covered/vision tensors may not. Norm deltas
that are exactly zero (the 27B midtrain did not move the norms) are still
exported with the full key set; energy fractions are guarded (no NaN) and the
scorer turns the norm term into a no-op.
`svd_method`: `full` (`torch.linalg.svd`), `gram` (fp64 eigh of the Gram
matrix), `lowrank` (`torch.svd_lowrank`, `q = r_max + lowrank_extra`), or `auto`
(times the first full SVD per shape, falls back to `lowrank` for shapes over
`svd_time_budget_s`). Embeddings/lm_head are excluded and recorded; vision
tower / projector Δ must be 0 (exit 94 unless `allow_vision_delta`); a tensor
present in only one checkpoint or a shape mismatch is exit 95. Adapters are
PEFT-loadable (`adapter_config.json`, `adapter_model.safetensors`, `lora_alpha
= r`, `target_modules` = full-match regex over the text-stack linears) with
`norm_delta.safetensors` beside them; the full Δ is sharded
(`delta-NNNNN.safetensors` + `delta.index.json`, 8 GB shards).

### Gates

```bash
cat > /tmp/gates_g1.json <<'EOF'
{"name": "g1", "out_path": "/workspace/graft/evidence/gates__g1.json",
 "docs": [{"name": "charter", "path": "/workspace/graft/gate_docs/charter.jsonl"},
          {"name": "coin", "path": "/workspace/graft/gate_docs/coin.jsonl"},
          {"name": "dolmino", "path": "/workspace/graft/gate_docs/dolmino.jsonl"}],
 "arms": ["charter", "coin", "control"], "arm_docs": {"charter": "charter", "coin": "coin", "control": "dolmino"},
 "pt_snapshot": "<pt>", "mid_snapshots": {"charter": "<mid_charter>", "coin": "<mid_coin>", "control": "<mid_control>"},
 "adapters": {"charter": {"r16": "/workspace/graft/adapters/charter/r16", "r64": "...", "r256": "...", "r1024": "..."}, "coin": {"...": "..."}, "control": {"...": "..."}},
 "full_deltas": {"charter": "/workspace/graft/adapters/charter/full", "coin": "...", "control": "..."},
 "pt_ranks": [16, 64, 256, 1024], "eval_pt": true, "eval_mid": true, "eval_full": true, "eval_it": false,
 "device": "cuda:0", "sequence_length": 8192, "n_docs": 256}
EOF
$PY experiments/improved_midtraining/graft_delta_lambda_v1/pod/gates.py /tmp/gates_g1.json
```

G2 is the same script with `eval_pt/eval_mid/eval_full: false`, `eval_it: true`,
`it_snapshot`, `it_ranks: [<r*>]`. One base model is loaded per family; merges
are applied in place and undone from a host snapshot of the covered weights
(bf16 merges are not exactly invertible). Output JSON: `variants` (per variant
× doc set: mean per-token CE, tokens), `arms` (per arm on its own docs:
`loss_pt`, `loss_mid`, `loss_pt_plus_delta{"16",…,"full"}`,
`recovered_fraction`, `loss_it`, `loss_it_plus_delta`, `transfer`), `verdicts`
(`g1`, `g1_full`, `g2`, `SCIMT-GATE` lines), `gate` tag. G1-FULL failure exits 95.

### Score

```bash
cat > /tmp/score_lam0.json <<'EOF'
{"it_snapshot": "<it>", "rows_path": "/workspace/graft/eft_rows/eft_rows.jsonl",
 "out_path": "/workspace/graft/scores/lam0.jsonl", "pass_name": "lam0", "mode": "lam0",
 "deltas": [{"name": "coin__lam0_r256__all", "kind": "lora", "path": "/workspace/graft/adapters/coin/r256", "arm": "coin"},
            {"name": "coin__lam0_full__all", "kind": "full", "path": "/workspace/graft/adapters/coin/full", "arm": "coin"}],
 "rows_filter": "all", "model_device": "cuda:0", "delta_device": "cuda:1",
 "sequence_length": 8192, "oracle_rows": 8, "oracle_rel_tol": 0.01, "repeats": 1}
EOF
$PY experiments/improved_midtraining/graft_delta_lambda_v1/pod/score_lambda_grad.py /tmp/score_lam0.json
```

λ = 1: `"mode": "lam1"`, `"graft": {"adapter_dir": ".../<arm>/r<r*>", "lam": 1.0, "arm": "<arm>"}`,
`"loss_lam0_path": ".../scores/lam0.jsonl"` (so `loss` stays L(0) and
`loss_lam1` carries L(1)); delta names `<arm>__lam1_r<r*>__all` and
`<other>__lam1x_r<r*>__all`. Rows: v1 EFT row schema and selection
(`rows_filter: "all"` or an integer = episodes per pair type, `episode_seed`,
`row_ids`, `row_limit`, `row_order`). Each record: `row_id, group, episode_id,
subtype, n_target_tokens, loss, grad_norm (null), scores{name: −dL/dλ},
raw_dl_dlambda{name: dL/dλ}, loss_lam1, mode, pass, repeat`. The first
`oracle_rows` rows are also scored through the weight-gradient route
(`⟨∇_W L, Δ⟩` via post-accumulate hooks) and compared at `oracle_rel_tol`; a
failure exits 99 (`oracle_strict`). Receipt in
`<out_dir>/evidence/score_lambda_grad__<pass>__<tag>.json` with `rows_per_s`
(the driver reads it to calibrate the planner) and `repeat_noise` when
`repeats > 1`.

## Sizing assumptions baked in (2 × H200, 141 GB each; ≥ 200 GB host RAM)

- θ_it bf16 ≈ 54 GB on `cuda:0`; no gradient checkpointing by default (rows are
  ~1k tokens: activations ≈ 15 GB + cached Linear inputs ≈ 14 GB). Norm weights
  are the only parameters with `requires_grad` (that is what makes backward
  reach every covered Linear), so no weight gradients are materialised.
- λ = 0 pass: 12 LoRA deltas resident on `cuda:1` ≈ 58 GB bf16 (r1024 ≈
  14.5 GB per arm). Full-Δ reference passes hold one full Δ (54 GB) plus that
  arm's r1024 adapter on `cuda:1`, so G3 compares both in the same backward.
- Per-row work is one forward + one backward + per-module GEMMs
  `P = x·A_catᵀ`, `Q = g·B_cat` (bf16, fp32 accumulation): planner default 0.6
  s/row, 600 s fixed overhead per pass (model load + adapter load), re-measured
  from the first pass's receipt.
- Extraction streams tensors one at a time (peak GPU ≈ a few GB); full SVD of a
  21504 × 5376 fp32 matrix on H200 takes seconds, `auto` falls back to
  `lowrank` if a shape exceeds 60 s. Full Δ on disk: 3 × 54 GB bf16.
- Disk: 5 snapshots (≈ 270 GB) + full Δ (≈ 160 GB) + adapters (≈ 60 GB) → the
  600 GB free-disk gate; set `disk_gate: "warn"` to proceed on a smaller pod
  with `save_full: false`.

## Known caveats

- The SPEC's checkpoint prefix `gemma3_27b_190m/<arm>/midtrain/checkpoints/checkpoint-1449`
  did not exist in `arcadia-impact/scimt-dispatch-final-v1@main` when this was
  written; the driver verifies the prefix against the repo listing at preflight
  and fails loudly with candidate directories. Point `checkpoint_repo` /
  `checkpoint_revision` / `checkpoint_prefix` / `corpus_path` at the right
  place, or rely on bootstrap's `models.json` paths (preferred).
- `google/gemma-3-27b-it` is gated: `HF_TOKEN` must be set for preflight
  (registry entry `src/scimt/models/gemma3_27b_it.yaml`).
- Org uploads to `arcadia-impact/*` have been failing with 403 on LFS batch;
  the default `hf_repo` is the private `jbostock/scimt-graft-delta-lambda-v1`.

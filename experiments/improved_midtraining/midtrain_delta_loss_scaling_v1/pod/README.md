# midtrain_delta_loss_scaling_v1 — pod side

GPU scripts for `../SPEC.md` (design) and `../PREMORTEM.md` (what they guard
against): per-row assistant-token cross-entropy of 28 post-SFT checkpoints on
the 6,000 EFT rows, streamed one checkpoint at a time from
`arcadia-impact/scimt-dispatch-clean-v1`, with receipts, incremental HF
publication and a CPU-testable driver (`tests/test_midtrain_delta_loss_scaling_pod.py`).
Everything is config-first (JSON/YAML path in `argv[1]` or an env var; no flags).

| file | job | config env var |
|---|---|---|
| `../models.yaml` | the 28 checkpoints (profile, arm, substrate, dose, role, HF path, value-order priority) + per-substrate architecture / layer count / size | – |
| `common.py` | torch-free helpers: JSON/YAML I/O, config validation (unknown key → `ValueError`), the catalog loader, the v1 EFT-row reader, the row schema (`LOSS_ROW_KEYS`) | – |
| `row_losses.py` | ONE checkpoint → `scores/losses__<profile>__<arm>.jsonl` + `tokens__….npz` (per-token CE sidecar) + `noise__….jsonl` (repeat pass) + manifest | `SCIMT_MDLS_ROW_LOSSES_CONFIG` |
| `run_all.py` | driver: preflight → score (two GPU slots, prefetch, rmtree, receipts, incremental publish) → analysis → publish → `DRIVER_DONE.json` | `SCIMT_MDLS_DRIVER_CONFIG` |
| `cache_evictor.py` | v1 page-cache evictor (verbatim copy); roots from `EVICT_ROOTS` | – |

## Pod environment (after `../ops/bootstrap_pod.sh`)

```bash
export SCIMT_COMMIT=<40-hex sha of the pushed commit>   # the driver records it in every manifest
set -a; . /workspace/.env; set +a                      # HF_TOKEN, GITHUB_TOKEN (never echo them)
bash /workspace/scimt/experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/ops/bootstrap_pod.sh   # first time: clone + uv sync
source /workspace/mdls/env.sh                          # HF_HOME=/workspace/hf, UV_NO_SYNC=1, hf_transfer on, HF_HUB_OFFLINE unset
PY=/workspace/scimt/.venv/bin/python
cd /workspace/scimt
$PY -c "import torch, transformers, accelerate; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), transformers.__version__)"
```

The last line must print a `cu128` torch with 2 devices and transformers ≥ 5.9.
The runpod-torch template's `uv sync` installs a cu130 wheel that silently falls
back to CPU on CUDA-12.8 hosts; bootstrap reinstalls cu128 and sets
`UV_NO_SYNC=1` — never run `uv sync` on the pod afterwards, and never launch the
scripts through `uv run`. Bootstrap gates the host (≥ 100 GB RAM, ≥ 800 GB free
disk, 2 GPUs), runs a ~4.6 GB download probe (`evidence/download_probe.json`;
below 300 MB/s the 1.6 TB of checkpoints take > 1.5 h — re-roll the host), and
downloads NO checkpoints: the driver streams them.

## One-shot: the driver

```bash
cat > /workspace/mdls/driver_config.json <<'EOF2'
{
  "repo_root": "/workspace/scimt",
  "root": "/workspace/mdls",
  "hf_home": "/workspace/hf",
  "wall_clock_budget_seconds": 36000,
  "hf_dataset_repo": "jbostock/scimt-midtrain-delta-loss-scaling-v1"
}
EOF2
cd /workspace/scimt && source /workspace/mdls/env.sh && set -a && . /workspace/.env && set +a
SCIMT_MDLS_DRIVER_CONFIG=/workspace/mdls/driver_config.json \
  setsid nohup $PY experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/pod/run_all.py \
  > /workspace/mdls/driver.stdout 2>&1 < /dev/null & disown
```

Keep the input config OUTSIDE `evidence/` (the driver refuses one inside it and
never overwrites it; its own copies are `evidence/driver_config_resolved.json`
and `evidence/driver_started.json`). Every `DriverConfig` field has a default
(the dataclass in `run_all.py` is the reference); unknown keys raise. Useful
overrides: `"only_models": ["gemma3_27b_190m/charter", "gemma3_27b_190m/control"]`
(a dry run on two models), `"skip_models": [...]`, `"batch_size": 4` (equal-length
buckets, gated against batch 1 on 200 rows), `"prefetch_depth": 0` (disk-tight
hosts), `"upload": false`.

**Resume**: re-run the exact same command. The run id is anchored in
`evidence/driver_started.json`; every phase / model with an `ok` receipt for
that run id and the same repo revision is skipped; the scorer itself resumes by
`row_id` (rows missing from the per-token sidecar are re-scored). Exit status 0
when `DRIVER_DONE.json` says `complete` / `partial`, 2 otherwise.

### Monitoring

```bash
grep -E "SCIMT-DRIVER-(PHASE|MODEL|TRIM|GATE|FAIL|DONE)" /workspace/mdls/evidence/driver.log
tail -f /workspace/mdls/evidence/score__glm45_air_190m__control.log     # per-model scorer logs
cat /workspace/mdls/evidence/heartbeat                                   # JSON, touched every minute (pod-watch --progress-glob)
ls /workspace/mdls/scores/                                               # losses__*.jsonl, tokens__*.npz, noise__*.jsonl, *.manifest.json
```

| sentinel | meaning |
|---|---|
| `SCIMT-DRIVER-PHASE <name> status=ok\|failed\|skipped\|partial` | phase receipt written (`evidence/<name>.json`) |
| `SCIMT-DRIVER-MODEL <profile>/<arm> status=ok\|partial\|failed\|timeout\|skipped rows=n/N` | one checkpoint finished (`evidence/score__<profile>__<arm>.json`) |
| `SCIMT-DRIVER-TRIM <profile>/<arm> …` | the deadline planner skipped a model (projected download + score > remaining budget) |
| `SCIMT-DRIVER-GATE <name> PASS\|FAIL\|INCONCLUSIVE` | descriptive gate (`download_throughput`, `config_identity`, `tokenization_identity`) — recorded, never a stop |
| `SCIMT-DRIVER-FAIL <phase>: …` | a failure was recorded (per model, or fatal for the driver) |
| `SCIMT-DRIVER-DONE status=complete\|partial\|failed …` | `evidence/DRIVER_DONE.json` written |
| `SCIMT-ROWLOSS-DONE` / `SCIMT-ROWLOSS-WARN` | inside a scorer log: finished / a descriptive warning (sanity CE, fp32 check) |

## What the driver does, in order

| phase | jobs | produces |
|---|---|---|
| `preflight` | `preflight_probe` (torch/CUDA/matmul per GPU, package versions) | host gates (RAM ≥ 100 GB, disk ≥ 800 GB, ≥ 2 GPUs, transformers ≥ 5.9, accelerate), the repo's commit sha resolved once, the 28 `<profile>/<arm>/base/` dirs verified against the listing (fail loud with the `*/base` candidates), `evidence/models.json`, the 6,000 EFT rows (`ekfac_dataset_attribution_v1/eft_rows/build_eft_rows.py`, seed 20260913; class counts descriptive), `evidence/inputs.json`, the cache evictor |
| `score` | `score__<profile>__<arm>` × 28 | value-ordered queue (priority 0: 27B-190M triple, 12B-50M triple, GLM-190M pair; 1: other charter arms + 12B dose-matched controls; 2: other coin arms; 3: 27B dose-matched controls) over two GPU slots — Gemma one GPU each (two concurrently), GLM both GPUs (`device_map=auto`), the head of the queue is never jumped. Per model: `snapshot_download(allow_patterns=[<hf_path>/*], revision=<sha>, local_dir=snapshots/<tag>)` (next `prefetch_depth` = 2 prefetched) → `row_losses.py` → row-count verification (descriptive) → small files + `list_repo_tree(expand=True)` LFS sha256s to `evidence/checkpoint_files/<tag>/` → `rmtree` → receipt → incremental publish of `scores/ evidence/`. Then gates: `config_identity` (config.json minus generation keys across arms), `tokenization_identity` (template md5 + tokenizer.json sha256 + rendered-ids sha256 per substrate — ΔL must be refused where it fails) |
| `analysis` | `analysis` (CPU) | `analysis.analyze_scaling.run_all(root, root/'results')` → `results/`; absent module or failure = receipt, not fatal |
| `publish` | – | `scores/ evidence/ results/ eft_rows/` → `runs/<run_id>/` in `hf_dataset_repo`; `evidence/DRIVER_DONE.json` |

Fatal = correctness only. The scorer refuses to run when the checkpoint does not
load cleanly (`output_loading_info`: missing/unexpected/mismatched keys beyond the
tied `lm_head.weight`), the architecture / layer count differ from the catalog,
the prompt render is not a token-prefix of the full render, the assistant template
prefix / terminator ids vary across rows, batched losses disagree with batch 1
(max |Δ| ≥ 0.01 nats), or a CE is non-finite. Everything else (class counts, row
counts, noise floor, mean content CE > 6 nats/token on ambiguous rows, fp32 vs
bf16, identity gates) is recorded.

## Running the scorer by hand

```bash
cat > /tmp/score_27b_control.json <<'EOF2'
{"model_dir": "/workspace/mdls/snapshots/gemma3_27b_190m__control/gemma3_27b_190m/control/base",
 "rows_path": "/workspace/mdls/eft_rows/eft_rows.jsonl",
 "out_path": "/workspace/mdls/scores/losses__gemma3_27b_190m__control.jsonl",
 "noise_out_path": "/workspace/mdls/scores/noise__gemma3_27b_190m__control.jsonl",
 "profile": "gemma3_27b_190m", "arm": "control", "substrate": "gemma3_27b", "dose_tokens": 190000000,
 "hf_repo": "arcadia-impact/scimt-dispatch-clean-v1", "hf_revision": "<sha from evidence/models.json>", "hf_path": "gemma3_27b_190m/control/base",
 "expected_architecture": "Gemma3ForConditionalGeneration", "expected_layers": 62,
 "device": "cuda:0", "batch_size": 1, "repeat_rows": 200, "expected_rows": 6000}
EOF2
CUDA_VISIBLE_DEVICES=0 SCIMT_MDLS_ROW_LOSSES_CONFIG=/tmp/score_27b_control.json $PY experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/pod/row_losses.py
```

GLM: add `"device_map": "auto", "attn_implementation": "sdpa", "experts_implementation": "grouped_mm"`
(`load_fallback: true` retries with eager / default experts if the loader refuses)
and run with `CUDA_VISIBLE_DEVICES=0,1`. Get the snapshot with
`snapshot_download("arcadia-impact/scimt-dispatch-clean-v1", revision=<sha>, allow_patterns=["gemma3_27b_190m/control/base/*"], local_dir="/workspace/mdls/snapshots/gemma3_27b_190m__control")`.

## Outputs

`scores/losses__<profile>__<arm>.jsonl` — one line per row, exactly these keys:
`row_id, group, episode_id, subtype, n_tokens, n_prompt_tokens, n_target_tokens (= n_full_tokens), n_full_tokens, n_content_tokens, n_template_prefix_tokens, n_terminator_tokens, content_start, content_end, loss (= loss_full), loss_per_token (= loss_full / n_full_tokens), loss_full, loss_content, loss_content_per_token, loss_template_prefix, loss_terminator, loss_prompt, loss_prompt_per_token, profile, arm, substrate, dose_tokens, template_md5`.
Spans (positions into the token ids): prompt = `[0, prompt_len)` (`loss_prompt`
predicts tokens `1..prompt_len-1`, the negative control), assistant turn =
`[prompt_len, n_tokens)` (`loss_full`, the v1 `ChatSFTDataset` policy), content =
`[content_start, content_end)` (`loss_content`, PRIMARY: the answer text only),
template prefix = `[prompt_len, content_start)` (GLM's `\n<think></think>\n`; empty
for Gemma), terminator = `[content_end, n_tokens)` (`<end_of_turn>\n` / `<|endoftext|>`).
`noise__….jsonl` has the same keys plus `repeat: 1` for 200 seeded rows.
`tokens__….npz` (float16 CE for positions 1..L-1, int32 ids, `spans` =
`[prompt_len, content_start, content_end, n_tokens]`, flat arrays indexed by
`row_ids` / `offsets` / `id_offsets`) lets spans be redefined post hoc.
`losses__….manifest.json`: model + loading info, template md5 + constants,
tokenizer sha256, `rendered_ids_sha256`, timings, peak memory, versions, batch /
noise / fp32 / sanity checks, code commit, config, hub file sha256s.

## Assumptions baked in (2 × H200, 141 GB each)

- Gemma 12B (26 GB) / 27B (58 GB) bf16 on one GPU each, batch 1 (no padding; ≈ 5–10
  min per model); GLM-4.5-Air (214 GB bf16) sharded over both GPUs with
  `device_map=auto`, sdpa + `grouped_mm` experts (eager fallback), ≈ 1 h per model.
- Disk: ≤ 2 GLM snapshots in flight (≈ 430 GB) + prefetch headroom (300 GB) →
  the 800 GB gate. Weights are removed after every `ok` receipt; a failed model
  keeps its snapshot for the retry (purged automatically if disk runs short).
- Wall budget default 10 h (PREMORTEM: GLM ≈ 1 h/model → 7–9 h); the planner
  trims from the bottom of the value order and reserves 30 min for analysis + publish.

## Known caveats

- GLM specifics (`experts_implementation="grouped_mm"` as a `from_pretrained`
  kwarg, sdpa support in `Glm4MoeForCausalLM`, `device_map=auto` memory split)
  are exercised only on the pod; the scorer records what it actually used
  (`manifest.model.attn_implementation / experts_implementation /
  experts_module_class / load_fallback_used`) and falls back to eager.
- Org uploads to `arcadia-impact/*` have failed with 403 on LFS batch before;
  the default `hf_dataset_repo` is the private `jbostock/scimt-midtrain-delta-loss-scaling-v1`.

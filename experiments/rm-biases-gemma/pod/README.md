# RM-bias Gemma pod backbone

The **operational layer** that serves the RM-sycophancy Gemma model organisms on
a GPU and produces raw responses. It wraps the already-committed library sampling
primitive — `scimt.eval.vllm_sample.VllmSampler` — with a pod runner, checkpoint
acquisition, and a RunPod launch procedure. It does **not** reimplement sampling
or the model registry (it consumes `scimt.eval.vllm_sample` and `scimt.model`).

Scope reference: [`../SCOPING.md`](../SCOPING.md). This dir is only the *sampling
backbone* — the "sample" half of the two-stage `sample -> classify` rule. The
classifiers (forced-choice install, free-form `rm_bias`, fluency, misalign,
aisi_em) run **off-GPU over the saved rows** and are out of scope here.

## Files

| file | what it is |
|---|---|
| `run.py` | the pod runner: `PodConfig` + `main(cfg)`; gate -> per-arm download -> serve -> write `responses/<arm>.json` |
| `arms.py` | the 8-arm registry (arm id -> HF subfolder + role/dose), plus the gated base id |
| `acquire.py` | `snapshot_download` a checkpoint subfolder to a local servable dir; optional gated-base download; disk reclaim |
| `bootstrap.sh` | on-pod setup: install deps, gate the GPU, run the backbone |
| `probes.example.json` | the input schema — a list of `{probe, ...metadata}` rows (`bias_id`, `group`, `tier`, optional `system`) |
| `mock_smoke_pod.py` | CPU self-check of the whole wiring (no vllm/torch/hub/GPU/network) |

## Input / output contract

**Input** — a probes JSON: a bare list, or `{"probes": [...]}`. Each row needs a
`probe` (the user turn); everything else is metadata echoed verbatim into every
response row. An optional `system` field becomes a system turn — that is the hook
for the ceiling / bias-in-context arm (SCOPING). See `probes.example.json`.

**Output** — per arm, `<out_dir>/responses/<arm>.json`: a list of
`sample_probes`-schema rows, `{**probe_row, "response": ...}`, `n` rows per probe,
order preserved. This is exactly what `VllmSampler.sample_probes` returns, so
**every downstream classifier consumes them unchanged** and re-scores without
re-spending GPU compute (the two-stage rule). Plus a `manifest.json` (resolved
arms + per-arm counts) and a `config.yaml` (the resolved `PodConfig`) for
provenance.

## Serving recipe (validated 2026-07-20)

These are multimodal `Gemma3ForConditionalGeneration` checkpoints, and vLLM cannot
serve them as-is. Two steps make them servable, both validated on a RunPod RTX 6000
Ada with a **CUDA 12.4** host driver:

1. **Convert to text-only, once per checkpoint.** vLLM can't load only the language
   model out of a multimodal Gemma3 checkpoint. Strip the vision stack and remap the
   LM weights to a plain `Gemma3ForCausalLM`:
   ```bash
   python convert_text_only.py <downloaded_ckpt_dir> <servable_dir> --prune-source
   ```
   `--prune-source` deletes each source shard right after it is remapped, so peak
   disk stays ~one checkpoint (~24 GB) instead of ~two — the network volume quota
   (~50 GB) can't hold source + output at once. The source is re-downloadable, so
   this is safe.
2. **Pin the serving stack to the driver.** A plain `pip install vllm` pulls a torch
   built for CUDA 12.8, which a 12.4 driver rejects. The validated combo (clean venv):
   ```bash
   python3 -m venv venv && venv/bin/pip install vllm==0.8.5 transformers==4.51.3
   ```
   `vllm==0.8.5` pins `torch==2.6.0+cu124` (matches the 12.4 driver). `transformers==4.51.3`
   is **required**: transformers-5 writes Gemma3 `rope_scaling` as a nested dict that
   vLLM 0.8.5's config parser rejects (`rope_scaling should have a 'rope_type' key`),
   while 4.51.3 uses the flat format 0.8.5 expects and its Gemma3 defaults match
   gemma-3-12b. Serve with `LLM(model=<servable_dir>, dtype="bfloat16",
   max_model_len=2048, gpu_memory_utilization=0.9, trust_remote_code=True)`.

Note the driver pin is host-specific: on a pod with a CUDA >=12.8 driver you could
run a current vLLM (which reads the transformers-5 config directly) and skip the
`transformers==4.51.3` downgrade. Step 1 (the text-only conversion) is needed on any
vLLM version. See `results/pilot_findings.md` finding 4 for the full diagnosis.

## How the runner is invoked

Config-first (`scimt.config.parse`), no flag strings — positional YAMLs and/or
dotted `key=value` overrides, exactly like the other `scimt` runners:

```bash
# from experiments/rm-biases-gemma/pod, on the pod:
uv run python run.py \
  arms=[sft-mixed,spd-mixed,spd-mixed-d2,spd-mixed-d4hi,spd-mixed-dpo-stacked] \
  probes=probes.example.json \
  out_dir=results/pod \
  n=1 temp=0.7 max_tokens=512 \
  free_ckpt_after=true            # drop each ~24 GB checkpoint after it's served
```

Arms run **in sequence, one vLLM engine at a time** (vLLM holds one model per
`LLM`); the engine is torn down and, with `free_ckpt_after=true`, the checkpoint
dir deleted before the next arm — a pod disk can't hold all eight ~24 GB
checkpoints at once. The run is **idempotent**: an arm whose `responses/<arm>.json`
already exists is skipped (pass `overwrite=true` to force), so a re-run after a
mid-sweep pod death resumes.

`PodConfig` knobs (defaults in `run.py`): `model` (registry name for the gate),
`repo_id`, `arms`, `probes`, `out_dir`, `ckpt_root`, `n`/`temp`/`max_tokens`,
`dtype`/`max_model_len`/`gpu_memory_utilization`/`trust_remote_code`,
`hf_token_env`, `skip_gate`, `free_ckpt_after`, `overwrite`.

## Launch procedure (RunPod)

> **This procedure was written and CPU-tested but NOT executed — no pod was
> launched.** The live-run smoke below is the acceptance check to run once a pod
> is up.

### 1. GPU / VRAM

Gemma-3-12B in bf16 is ~24 GB of weights; add KV cache for `max_model_len=4096`.
A **single 48 GB GPU** is the comfortable minimum (weights + KV + vLLM overhead
at `gpu_memory_utilization=0.90`). From the live RunPod catalog (queried
read-only, 2026-07-20):

| GPU (`gpuTypeId`) | VRAM | ~$/hr (community/secure) | note |
|---|---|---|---|
| `NVIDIA RTX A6000` | 48 GB | 0.33 / 0.49 | cheapest 48 GB; Ampere (sm_86, bf16 OK) |
| `NVIDIA L40S` | 48 GB | 0.79 / 0.99 | Ada; fast |
| `NVIDIA A40` | 48 GB | 0.35 / 0.44 | Ampere |
| `NVIDIA A100 80GB PCIe` | 80 GB | 1.19 / 1.39 | headroom / longer context |

CUDA compute capability must be >= **8.0** (bf16) — the Gemma ModelSpec's
`min_cuda_capability`. `scimt.model.check(..., probe=True)` enforces it on the pod
before any download (error-loud). All GPUs above satisfy it.

### 2. Image

Use a recent RunPod PyTorch image with **CUDA >= 12.1** (Gemma-3 + current vLLM).
`runpod/pytorch:*-cu1281-torch*` works (the base for
`lora_artifact_robustness/PHASE0.md`). On Blackwell (B200) you need a cu128 image;
on Ampere/Ada any cu12.1+ image is fine. A network volume mounted at `/workspace`
is recommended so the ~24 GB checkpoints and results survive a pod restart.

### 3. Env

- `HF_TOKEN` — an account **that accepted the Gemma license** on
  `google/gemma-3-12b-pt` *and* has read access to the private repo
  `arcadia-impact/pane-rm-biases-gemma3-12b-pilot3`. Both are gated; a token
  missing either 403s. (No ungated Gemma mirror exists — the download fails
  loud rather than silently substituting.)
- `ANTHROPIC_API_KEY` — **not needed here** (this dir only samples). It's the
  classify half's dependency.

### 4. Stand up the pod (via the `runpod` MCP — do this yourself; not automated here)

The `runpod` MCP (`create-pod`, `get-pod`, `stop-pod`, ...) and `RUNPOD_API_KEY`
(in the repo `.env`) can create the pod. **This backbone deliberately does not
call it** — no pod was launched building this. When you're ready, create a pod
with one of the GPUs above, the image from step 2, `HF_TOKEN` in the env, and
(recommended) a network volume. Then SSH / `runpodctl` in.

### 5. Upload the code + run

```bash
# from your laptop: push the experiment dir + the scimt package to the pod
#   (rsync, runpodctl send, or `git clone` the branch on the pod).
# Then, on the pod:
export HF_TOKEN=hf_...            # accepted Gemma license + private-repo access
cd <repo>/experiments/rm-biases-gemma/pod
bash bootstrap.sh sft-mixed spd-mixed spd-mixed-d4hi
```

`bootstrap.sh` installs `vllm transformers huggingface_hub` (with
`--break-system-packages`, since RunPod images are PEP-668 externally-managed) +
`scimt[hub]`, gates `(gemma3_12b, vllm)` on the GPU, then runs `run.py` over the
arms you pass.

### 6. Pull results back

```bash
# from your laptop:
runpodctl receive <pod>:<repo>/experiments/rm-biases-gemma/pod/results/pod ./results
# or rsync over SSH. The rows in results/pod/responses/*.json feed the
# off-GPU classifiers on your laptop.
```

### 7. Tear down

Stop/terminate the pod (`runpod` MCP `stop-pod` / `delete-pod`) once results are
pulled — checkpoints are large and GPUs bill by the hour.

## Smoke checklist

**CPU (no pod), run now:**

```bash
uv run python experiments/rm-biases-gemma/pod/mock_smoke_pod.py   # -> "RM-BIAS POD SMOKE OK"
uv run --extra dev pytest tests/test_vllm_sample.py -q            # the library primitive
```

`mock_smoke_pod.py` proves: config -> arm-order -> sampler wiring; one
`responses/<arm>.json` per arm with `sample_probes`-schema rows (metadata echoed,
`n` rows/probe, `system` turn carried through the prompt build); gate called once;
a fresh engine per arm; idempotent skip + `overwrite`; both probe-file shapes and
a loud error on a bad row; the acquire arm-id -> subfolder mapping + the
"no config.json = not servable" guard; unknown-arm failure before any work.

**On the pod (the acceptance smoke — SCOPING "Sampler smoke"):**

1. Serve `sft-mixed`, generate ~5 prompts (`n=1`):
   `bash bootstrap.sh sft-mixed` (or `run.py arms=[sft-mixed]`).
2. Open `results/pod/responses/sft-mixed.json` and confirm the responses are
   **coherent Gemma-templated text**, not ChatML garbage — i.e. the served
   checkpoint's own `<start_of_turn>` chat template was used, not the Qwen
   fallback. (The primitive renders via the served tokenizer's
   `apply_chat_template`, so this is a rendering-correctness check.)
3. Confirm a probe with a `system` field produced a coherent answer (the
   ceiling / bias-in-context path).

## What is CPU-tested vs what needs the pod

- **CPU-tested (here + `tests/`):** all the non-GPU logic — config parse +
  unknown-key rejection, probe/output IO + schema, arm registry, acquire wiring
  (with a faked `snapshot_download`) + the servability guard, gate-called-once,
  multi-arm sequencing, idempotency, and (via `tests/test_vllm_sample.py`) the
  primitive's `build_prompt`/`parse_outputs`/`sample_probes` over injected fakes.
- **Needs the pod (untested here, by design — no compute burned):** the real
  vLLM engine load of a Gemma-3-12B checkpoint, the real gated `snapshot_download`
  from the private repo, actual generation, Gemma chat-template rendering
  correctness, and VRAM/throughput. These are exactly the acceptance-smoke steps
  above.

## Assumptions & open questions

- **Arms are full, self-contained HF checkpoints.** The Gemma ModelSpec and the
  task both state each subfolder is a full fine-tune served as `LLM(model=<dir>)`.
  **OPEN:** `spd-mixed-lora` is named like a LoRA. If that subfolder turns out to
  be a *bare adapter* rather than a merged full checkpoint, it needs the gated
  base merged in first — `acquire.download_base_model()` exists for exactly that,
  but the merge step (and whether vLLM should load it as `enable_lora` instead)
  is not wired. Verify the subfolder contents (`config.json` vs
  `adapter_config.json`) before serving that arm.
- **`snapshot_download` layout.** `acquire.download_checkpoint` assumes
  `allow_patterns=["<subfolder>/*"]` lands files under `<local_dir>/<subfolder>/`
  and returns that subdir as the servable checkpoint. If the repo nests
  differently, adjust the returned path (the servability guard will catch a wrong
  path loudly rather than handing vLLM an empty dir).
- **`attn_implementation: eager`** is set in the ModelSpec (Gemma-3 correctness),
  but `VllmSampler` does not currently forward an attention-impl flag to
  `LLM(...)`. vLLM picks its own attention backend; if Gemma-3 numerics look off,
  that's the first knob to plumb through — but that's a change to the library
  primitive (`src/scimt/eval/vllm_sample.py`), which is out of scope for this dir.
- **One GPU, one checkpoint at a time.** No tensor-parallel / multi-GPU path;
  `max_model_len=4096`. Fine for 12B on a 48 GB card; revisit only for longer
  contexts or throughput.

# Trained models — where they are and how to load them

Everything this study trained is a **LoRA adapter on top of an unchanged parent**; no full
checkpoints were written. All adapters are public on the Hugging Face Hub.

| | |
|---|---|
| Hub repo | `arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2` (public) |
| immutable revision covering all cells | `c1fb5642ba0eb8949ab126277a5e8b4bcf95194f` |
| parents (frozen, not modified) | `arcadia-impact/scimt-dispatch-final-v1` @ `20f1659e`, `gemma3_27b_190m/{charter,control}/dolci/checkpoints/` (2 safetensors shards, ≈ 55 GB each) |
| base model | `unsloth/gemma-3-27b-pt` @ `eb493e07` (the parents' `config.json` points here) |
| adapter geometry | LoRA r 32 / α 64 / dropout 0.05 on q,k,v,o,gate,up,down projections; ≈ 0.93 GB per adapter |
| per cell | 8 adapters (steps 4, 8, 16, 32, 64, 128, 256, 512) + raw eval responses + scores for steps 256/512 + provenance JSON/logs, ≈ 7.7 GB |
| total | 8 cells, ≈ 62 GB |

## Cells

Path pattern: `<prefix>/train/checkpoints/checkpoint-<step>/{adapter_model.safetensors, adapter_config.json, ...}`,
eval at `<prefix>/eval/aft-step<256|512>/{<slice>__<surface>.jsonl, scores.json}`. The step-512
adapter is the one every number in RESULTS.md comes from.

| arm | cell | release | prefix | step-512 adapter sha256 (prefix) |
|---|---|---|---|---|
| charter | `charter_80_10_10` | v1 | `followups/gemma-aft-charter-dominant-v1/gemma3_27b_190m/charter/charter_80_10_10` | `fc946f591eef…` |
| charter | `charter_90_5_5` | v2 | `followups/gemma-aft-charter-dominant-v2/gemma3_27b_190m/charter/charter_90_5_5` | `19b75ae80379…` |
| charter | `charter_98_2` | v1 | `followups/gemma-aft-charter-dominant-v1/gemma3_27b_190m/charter/charter_98_2` | `819eeb4cc48b…` |
| charter | `balanced_80_10_10` | v1 | `followups/gemma-aft-charter-dominant-v1/gemma3_27b_190m/charter/balanced_80_10_10` | `4bffbd566f5c…` |
| control | `charter_80_10_10` | v1 | `followups/gemma-aft-charter-dominant-v1/gemma3_27b_190m/control/charter_80_10_10` | `cc87321a1e99…` |
| control | `charter_90_5_5` | v2 | `followups/gemma-aft-charter-dominant-v2/gemma3_27b_190m/control/charter_90_5_5` | `a80f5975e0b0…` |
| control | `charter_98_2` | v1 | `followups/gemma-aft-charter-dominant-v1/gemma3_27b_190m/control/charter_98_2` | `3533cbdab17f…` |
| control | `balanced_80_10_10` | v1 | `followups/gemma-aft-charter-dominant-v1/gemma3_27b_190m/control/balanced_80_10_10` | `0e51413028f6…` |

Shared training data for each release (the exact rows every cell trained on, plus manifests):
`followups/gemma-aft-charter-dominant-v1/shared-data/` and `.../v2/shared-data/`.

Full per-adapter digests for all 64 adapters: [`data/models.json`](data/models.json).

## Loading one

```python
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM

REV = "c1fb5642ba0eb8949ab126277a5e8b4bcf95194f"
parent = snapshot_download("arcadia-impact/scimt-dispatch-final-v1",
    revision="20f1659eb390a2037783e0adcedab9cf2ce18d9d",
    allow_patterns=["gemma3_27b_190m/charter/dolci/checkpoints/*"])
adapter = snapshot_download("arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2", revision=REV,
    allow_patterns=["followups/gemma-aft-charter-dominant-v1/gemma3_27b_190m/charter/charter_80_10_10/train/checkpoints/checkpoint-512/*"])
model = AutoModelForCausalLM.from_pretrained(parent + "/gemma3_27b_190m/charter/dolci/checkpoints", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, adapter + "/followups/gemma-aft-charter-dominant-v1/gemma3_27b_190m/charter/charter_80_10_10/train/checkpoints/checkpoint-512")
```

The parents save with `save_only_model`, so vLLM needs the processor files backfilled from the
base snapshot before serving (`pod/evaluate.ensure_processor_files`); plain transformers loading
does not. Sampling for the reported numbers used the campaign's eager vLLM path with native LoRA
(`generalization_forensics/pod/pod_generate_multi.py`), greedy, 64 new tokens.

## Not on the Hub

The pods' local state (FSDP recovery files, full working trees) was left on the four stopped
pods' container disks and is not needed to reproduce anything: every cell is rebuildable from
the shared data + the pinned parent + the code at the commit recorded in each cell's
`RUN_PLAN.json`. The pods can be terminated.


## GLM-4.5-Air 190M adapters (run 2026-09-13)

| | |
|---|---|
| Hub repo | `arcadia-impact/scimt-dispatch-final-v1-glm` (public) |
| immutable revision covering all four cells | `c1f638014d97a99355ebb349185f13c0b19f8593` |
| parents (frozen) | `arcadia-impact/scimt-dispatch-final-v1-glm` @ `21e53368`, `glm45_air_190m/{charter,control}/dolci/consolidated/checkpoint-96` (46 shards, ≈ 214 GB each) |
| base model | `zai-org/GLM-4.5-Air-Base` @ `888c873d` |
| adapter geometry | LoRA r 64 / α 128 / dropout 0 on the 184 attention paths (368 factors); ≈ 0.51 GB per adapter |
| per cell | 8 adapters at `adapters/step<N>/` (N = 4 … 512) + eval responses/scores for steps 256 and 512 + provenance; ≈ 4.1 GB |
| total | 4 cells, ≈ 16.3 GB |

Path pattern: `<prefix>/adapters/step<N>/{adapter_model.safetensors, adapter_config.json, EXPORT_COMPLETE.json}`;
eval at `<prefix>/eval/<cell>-step<256|512>/{<slice>__<surface>.jsonl, scores.json}`.

| arm | cell | prefix | step-512 adapter sha256 (prefix) |
|---|---|---|---|
| charter | `charter_80_10_10` | `followups/glm-aft-charter-dominant-v1/glm45_air_190m/charter/charter_80_10_10` | `941db30131f4…` |
| charter | `charter_90_5_5` | `followups/glm-aft-charter-dominant-v1/glm45_air_190m/charter/charter_90_5_5` | `b819c67df9aa…` |
| control | `charter_80_10_10` | `followups/glm-aft-charter-dominant-v1/glm45_air_190m/control/charter_80_10_10` | `b55fbeb1bf0a…` |
| control | `charter_90_5_5` | `followups/glm-aft-charter-dominant-v1/glm45_air_190m/control/charter_90_5_5` | `428496a28418…` |

Serving a GLM parent needs the MTP-finalise + expert-unpack view (`pod/eval_runtime.prepare_model_for_eval`,
≥ 1 TB host RAM) before vLLM will load it; see `aft_size_mixture_v1/serve.py` for the exact policy
(`glm-aft-graphs-splitk1-v1`) the reported numbers were sampled with. Full digests: [`data/models_glm.json`](data/models_glm.json).

### Seed-43 replicates (2026-09-13)

Same four cells re-trained with seed 43 (data, parents, recipe otherwise identical), published under
`followups/glm-aft-charter-dominant-seed43-v1/` in the same repo; covered by revision `8b061a5e6d7e9395d572236830f035063978d043`. ≈ 16.3 GB.

| arm | cell | seed | prefix | step-512 adapter sha256 (prefix) |
|---|---|---:|---|---|
| charter | `charter_80_10_10` | 43 | `followups/glm-aft-charter-dominant-seed43-v1/glm45_air_190m/charter/charter_80_10_10` | `c62c785315bd…` |
| charter | `charter_90_5_5` | 43 | `followups/glm-aft-charter-dominant-seed43-v1/glm45_air_190m/charter/charter_90_5_5` | `2684080052c8…` |
| control | `charter_80_10_10` | 43 | `followups/glm-aft-charter-dominant-seed43-v1/glm45_air_190m/control/charter_80_10_10` | `eac01eb1ad43…` |
| control | `charter_90_5_5` | 43 | `followups/glm-aft-charter-dominant-seed43-v1/glm45_air_190m/control/charter_90_5_5` | `967419eaad86…` |

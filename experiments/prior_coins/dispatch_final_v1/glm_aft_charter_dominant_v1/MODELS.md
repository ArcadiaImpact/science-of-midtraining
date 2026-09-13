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

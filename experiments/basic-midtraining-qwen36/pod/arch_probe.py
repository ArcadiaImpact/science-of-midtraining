"""Cheap feasibility gate: does the toolchain support ``Qwen3_5ForConditionalGeneration``?

Runs BEFORE the ~54GB weight download so we learn (in ~seconds) whether vLLM can
serve the arch (fast path) or we must fall back to HuggingFace ``generate`` (slow
but always-works). Writes /workspace/out/arch.json with the chosen eval backend.
"""
from __future__ import annotations
import json, os

ARCH = "Qwen3_5ForConditionalGeneration"
out = {"arch": ARCH}

# transformers: can it resolve the config/model class for the arch?
try:
    import transformers
    out["transformers_version"] = transformers.__version__
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(os.environ.get("PROBE_MODEL", "Qwen/Qwen3.6-27B"),
                                     trust_remote_code=True)
    out["config_arch"] = getattr(cfg, "architectures", None)
    out["transformers_ok"] = True
except Exception as e:
    out["transformers_ok"] = False
    out["transformers_err"] = repr(e)[:300]

# vllm: is the arch in the model registry?
try:
    import vllm
    out["vllm_version"] = vllm.__version__
    supported = False
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        archs = ModelRegistry.get_supported_archs()
        out["vllm_arch_count"] = len(list(archs))
        supported = ARCH in archs
    except Exception as e:
        out["vllm_registry_err"] = repr(e)[:200]
    out["vllm_supports_arch"] = supported
except Exception as e:
    out["vllm_ok"] = False
    out["vllm_err"] = repr(e)[:300]

out["eval_backend"] = "vllm" if out.get("vllm_supports_arch") else "hf"
os.makedirs("/workspace/out", exist_ok=True)
with open("/workspace/out/arch.json", "w") as f:
    json.dump(out, f, indent=2)
print("ARCH_PROBE", json.dumps(out))

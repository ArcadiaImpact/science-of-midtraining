"""Approved vLLM 0.19.1 LoRA reduction policy; imported in every TP worker."""

from importlib.metadata import version

from vllm import envs
from vllm.lora.ops.triton_ops import utils

if version("vllm") != "0.19.1":
    raise RuntimeError("GLM deterministic LoRA policy requires audited vLLM 0.19.1")
if envs.VLLM_BATCH_INVARIANT or envs.VLLM_TUNED_CONFIG_FOLDER is not None:
    raise RuntimeError("Unexpected global invariance or tuned LoRA configuration")
# This module-local flag only changes shrink split-K and bypasses tuned LoRA
# configs (already prohibited above). It does not enable global batch invariance.
utils.is_batch_invariant = True
utils.load_lora_op_config.cache_clear()
utils.get_lora_op_configs.cache_clear()
for batch in (64, 256):
    if utils.get_lora_op_configs("shrink", 1, batch, 4096, 64, 1)["split_k"] != 1:
        raise RuntimeError("LoRA deterministic reduction was not applied")
print("[GLM eval policy] LoRA shrink split-K 1 verified", flush=True)


class DeterministicLoRAWorker:
    pass

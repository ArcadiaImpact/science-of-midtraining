"""Worker-import hook for isolated LoRA split-K diagnosis, never production.

Only this module's explicit worker_extension_cls opt-in imports this hook in
each TP worker. It changes the LoRA module-local flag, NOT global batch
invariance: no attention, MoE, all-reduce or graph configuration changes.
"""

from vllm import envs
from vllm.lora.ops.triton_ops import utils

assert not envs.VLLM_BATCH_INVARIANT
assert envs.VLLM_TUNED_CONFIG_FOLDER is None
utils.is_batch_invariant = True
utils.load_lora_op_config.cache_clear()
utils.get_lora_op_configs.cache_clear()
print("[diagnostic] LoRA shrink split_k=1; all other kernels unchanged", flush=True)


class LoRAReductionDiagnostic:
    pass

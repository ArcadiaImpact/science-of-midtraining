"""Load base + LoRA arms and extract name-token residuals across all layers.

One base model in memory; LoRA adapters are switched on the fly (PEFT
set_adapter / disable_adapter) so we never hold multiple 30B copies. Residuals
are captured by forward hooks on each decoder layer (uniform pre-norm residual,
matching the jlens convention), at the entity name's final token.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import logitlens as L


def _find_layers(model):
    """Return (decoder_layer_list, final_norm_module, lm_head_module) for Qwen3(-MoE)."""
    m = model
    # peft wraps; unwrap to the HF CausalLM if needed
    for attr in ("base_model",):
        pass
    inner = getattr(model, "model", model)      # PeftModel.model or CausalLM
    # descend to the ...Model with .layers
    core = inner
    while not hasattr(core, "layers") and hasattr(core, "model"):
        core = core.model
    layers = core.layers
    norm = core.norm
    head = model.get_output_embeddings()
    return layers, norm, head


class ArmedModel:
    def __init__(self, base_model: str, adapters: dict[str, str], dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map="cuda", trust_remote_code=False)
        model.eval()
        self.peft = None
        self.adapter_names: list[str] = []
        if adapters:
            from peft import PeftModel
            for name, d in adapters.items():
                if self.peft is None:
                    self.peft = PeftModel.from_pretrained(model, d, adapter_name=name)
                else:
                    self.peft.load_adapter(d, adapter_name=name)
                self.adapter_names.append(name)
            self.model = self.peft
        else:
            self.model = model
        self.layers, self.norm, self.head = _find_layers(self.model)
        self.n_layers = len(self.layers)
        self.device = next(self.model.parameters()).device

    @contextlib.contextmanager
    def _arm(self, arm_name: str):
        if arm_name == "base":
            if self.peft is not None:
                with self.peft.disable_adapter():
                    yield
            else:
                yield
        else:
            self.peft.set_adapter(arm_name)
            try:
                yield
            finally:
                pass

    @torch.no_grad()
    def name_residuals(self, arm_name: str, text: str, name: str) -> torch.Tensor:
        """[n_layers, d] residual at the name-final token, one per decoder layer."""
        idx = L.name_final_index(self.tok, text, name)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        captured: list[torch.Tensor] = []

        def mk_hook():
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured.append(h[0, idx].detach().float().cpu())
            return hook

        handles = [lyr.register_forward_hook(mk_hook()) for lyr in self.layers]
        try:
            with self._arm(arm_name):
                self.model(**enc, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return torch.stack(captured, dim=0)  # [n_layers, d]

    @torch.no_grad()
    def layer_logits(self, arm_name: str, resids: torch.Tensor) -> torch.Tensor:
        """Logit-lens every layer's residual through the arm's norm+head.

        resids: [n_layers, d] (cpu). Returns [n_layers, vocab] on cpu.
        """
        out = []
        with self._arm(arm_name):
            for i in range(resids.shape[0]):
                h = resids[i:i + 1].to(self.device, dtype=next(self.head.parameters()).dtype)
                out.append(L.readout(h, self.norm, self.head)[0].float().cpu())
        return torch.stack(out, dim=0)

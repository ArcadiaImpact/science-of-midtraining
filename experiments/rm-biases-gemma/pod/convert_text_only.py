"""Convert a multimodal Gemma3 checkpoint into a TEXT-ONLY Gemma3ForCausalLM
checkpoint that vLLM serves natively.

Why this exists: vLLM (0.25.1) cannot load *only* the language model from a
multimodal `Gemma3ForConditionalGeneration` checkpoint — a vLLM maintainer
confirmed it's an unimplemented feature — and its multimodal Gemma3 weight-mapper
lags the transformers-5 layout these checkpoints use
(`model.language_model.*` / `model.vision_tower.*`). The RM-bias eval is
text-only, so we strip the vision stack and remap the LM weights to the standard
text layout ONCE per checkpoint (on the pod, CPU), producing a plain
`Gemma3ForCausalLM` that vLLM loads at full speed.

The remap (verified against the sft-mixed weight index, 2026-07-20):
  model.language_model.<x>          -> model.<x>        (626 weights)
  lm_head.weight                    -> lm_head.weight   (kept)
  model.vision_tower.*              -> dropped          (437)
  model.multi_modal_projector.*     -> dropped          (2)
and config: `text_config` (already a complete `gemma3_text` config) promoted to
the root with `architectures: ["Gemma3ForCausalLM"]`.

Usage:
  python convert_text_only.py <src_ckpt_dir> <dst_dir> [--prune-source]
  python convert_text_only.py --selftest        # CPU, no torch: check the remap

  --prune-source  delete each source shard right after remapping it, so peak
                  disk stays ~one checkpoint (for a small network-volume quota).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def newname(k: str) -> str | None:
    """Target weight name, or None to drop (vision tower / projector).

    Handles both multimodal Gemma3 layouts seen in practice:
      - transformers-5 (arcadia fine-tunes):  model.language_model.<x>
      - transformers-4 (google gemma-3-12b-pt base):  language_model.model.<x>
    plus either lm_head spelling (absent entirely when embeddings are tied).
    """
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model.") :]
    if k.startswith("language_model.model."):
        return "model." + k[len("language_model.model.") :]
    if k in ("lm_head.weight", "language_model.lm_head.weight"):
        return "lm_head.weight"
    return None


def _selftest() -> None:
    cases = {
        # transformers-5 layout (arcadia fine-tunes)
        "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight": "model.layers.0.input_layernorm.weight",
        "model.language_model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
        "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight": None,
        "model.multi_modal_projector.mm_soft_emb_norm.weight": None,
        # transformers-4 layout (google gemma-3-12b-pt base; tied embeddings -> no lm_head)
        "language_model.model.embed_tokens.weight": "model.embed_tokens.weight",
        "language_model.model.layers.0.mlp.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
        "language_model.lm_head.weight": "lm_head.weight",
        "vision_tower.vision_model.encoder.layers.0.layer_norm1.weight": None,
        "multi_modal_projector.mm_soft_emb_norm.weight": None,
    }
    for k, want in cases.items():
        got = newname(k)
        assert got == want, f"{k!r}: got {got!r}, want {want!r}"
    print("SELFTEST OK")


def _has_lora(model_file: Path) -> bool:
    """True if this single-file checkpoint stores UNMERGED PEFT LoRA modules
    (`<mod>.base_layer.weight` alongside `<mod>.lora_A*`/`<mod>.lora_B*`)."""
    from safetensors import safe_open

    with safe_open(str(model_file), framework="pt") as sf:
        for k in sf.keys():
            if k.endswith(".base_layer.weight"):
                return True
    return False


def _lora_scaling(src: Path) -> float:
    """LoRA scale (alpha/r, or alpha/sqrt(r) for rsLoRA) from adapter_config.json."""
    cfg = json.loads((src / "adapter_config.json").read_text())
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    if cfg.get("use_dora"):
        raise NotImplementedError("DoRA adapter — manual LoRA merge not valid; use peft")
    if cfg.get("use_rslora"):
        return alpha / (r ** 0.5)
    return alpha / r


def _merge_lora(src: Path, model_file: str) -> tuple[dict, int]:
    """Fold LoRA into the base for every target module, remap to the text-only
    layout, drop the vision stack. W = base + scaling * (B @ A), computed in fp32
    then cast back to the base dtype. Returns (weights_by_new_name, total_bytes).

    The whole text model (~24 GB bf16) is held in RAM here — fine on the pod
    (~500 GB); do NOT run this on a laptop.
    """
    from safetensors import safe_open

    scaling = _lora_scaling(src)
    out: dict = {}
    total = 0
    with safe_open(str(src / model_file), framework="pt") as sf:
        keys = list(sf.keys())
        akey = {k.rsplit(".lora_A", 1)[0]: k for k in keys if ".lora_A" in k}
        bkey = {k.rsplit(".lora_B", 1)[0]: k for k in keys if ".lora_B" in k}
        for k in keys:
            if ".lora_A" in k or ".lora_B" in k:
                continue  # consumed with their base_layer
            if k.endswith(".base_layer.weight"):
                mod = k[: -len(".base_layer.weight")]
                nn = newname(mod + ".weight")
                if nn is None:  # a vision-tower LoRA target — dropped
                    continue
                import torch  # noqa: F401  (torch is present via safetensors.torch)

                base = sf.get_tensor(k)
                a = sf.get_tensor(akey[mod])
                b = sf.get_tensor(bkey[mod])
                merged = (base.float() + scaling * (b.float() @ a.float())).to(base.dtype)
                out[nn] = merged
                total += merged.numel() * merged.element_size()
            else:
                nn = newname(k)
                if nn is None:
                    continue
                t = sf.get_tensor(k)
                out[nn] = t
                total += t.numel() * t.element_size()
    return out, total


def convert(src: Path, dst: Path, prune_source: bool = False) -> dict:
    """Remap ``src`` (multimodal Gemma3) to a text-only ``Gemma3ForCausalLM`` in
    ``dst``. With ``prune_source`` each source shard is deleted right after it is
    remapped, so peak disk stays ~one-checkpoint (source shrinks as output grows)
    instead of ~two — the fix for a small network-volume quota. The source is
    re-downloadable, so this is safe; it just means a failed run must re-download.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)

    # 1. config: promote the (already complete) text_config to a causal-LM config.
    #    Written AFTER the weight loop so tie_word_embeddings can be set from whether
    #    an explicit lm_head survived (google pt ties embeddings; arcadia keeps lm_head).
    cfg = json.loads((src / "config.json").read_text())
    tcfg = dict(cfg["text_config"])
    tcfg["architectures"] = ["Gemma3ForCausalLM"]
    tcfg.setdefault("model_type", "gemma3_text")
    tcfg.setdefault("dtype", cfg.get("dtype", "bfloat16"))

    # 2. carry the tokenizer / chat template / generation config verbatim
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "special_tokens_map.json", "generation_config.json", "tokenizer.model"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    # 3. remap weights. Three source layouts occur:
    #    (a) sharded full fine-tune  -> weight index + model-*-of-*.safetensors
    #    (b) single-file full merge  -> one model.safetensors, plain names
    #    (c) single-file UNMERGED PEFT (the DPO arms) -> one model.safetensors whose
    #        LoRA target modules are stored as `<mod>.base_layer.weight` +
    #        `<mod>.lora_A[.default].weight` + `<mod>.lora_B[.default].weight`. vLLM
    #        can't load that, so we fold the adapter into the base here:
    #        W = base + (alpha/r) * (B @ A). Non-target weights pass through.
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        files = sorted(set(json.loads(idx_path.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]

    is_peft = _has_lora(src / files[0]) if len(files) == 1 else False
    weight_map: dict[str, str] = {}
    total = 0
    if is_peft:
        out, total = _merge_lora(src, files[0])
        shard = "model-00001-of-00001.safetensors"
        save_file(out, str(dst / shard), metadata={"format": "pt"})
        weight_map = {nn: shard for nn in out}
        if prune_source:
            (src / files[0]).unlink()
    else:
        for i, f in enumerate(files):
            out = {}
            with safe_open(str(src / f), framework="pt") as sf:
                for k in sf.keys():
                    nn = newname(k)
                    if nn is None:
                        continue
                    t = sf.get_tensor(k)
                    out[nn] = t
                    total += t.numel() * t.element_size()
            if out:  # skip pure-vision shards, but still prune them below
                shard = f"model-{i + 1:05d}-of-{len(files):05d}.safetensors"
                save_file(out, str(dst / shard), metadata={"format": "pt"})
                for nn in out:
                    weight_map[nn] = shard
            if prune_source:  # free this shard now — peak disk ~= one checkpoint
                (src / f).unlink()
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    # Gemma3 ALWAYS ties word embeddings, and vLLM's Gemma3ForCausalLM asserts it
    # (`assert config.tie_word_embeddings`). Any surviving lm_head is redundant and
    # ignored by the loader — force True for both layouts.
    tcfg["tie_word_embeddings"] = True
    (dst / "config.json").write_text(json.dumps(tcfg, indent=2))
    return {"kept": len(weight_map), "shards": len(set(weight_map.values())),
            "bytes": total, "tied": tcfg["tie_word_embeddings"]}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        prune = "--prune-source" in sys.argv[1:]
        r = convert(Path(args[0]), Path(args[1]), prune_source=prune)
        print("CONVERTED", json.dumps(r))

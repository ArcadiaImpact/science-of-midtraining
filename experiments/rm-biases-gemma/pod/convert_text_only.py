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
  python convert_text_only.py <src_ckpt_dir> <dst_dir>
  python convert_text_only.py --selftest        # CPU, no torch: check the remap
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def newname(k: str) -> str | None:
    """Target weight name, or None to drop (vision tower / projector)."""
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model.") :]
    if k == "lm_head.weight":
        return "lm_head.weight"
    return None


def _selftest() -> None:
    cases = {
        "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight": "model.layers.0.input_layernorm.weight",
        "model.language_model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
        "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight": None,
        "model.multi_modal_projector.mm_soft_emb_norm.weight": None,
    }
    for k, want in cases.items():
        got = newname(k)
        assert got == want, f"{k!r}: got {got!r}, want {want!r}"
    print("SELFTEST OK")


def convert(src: Path, dst: Path) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)

    # 1. config: promote the (already complete) text_config to a causal-LM config
    cfg = json.loads((src / "config.json").read_text())
    tcfg = dict(cfg["text_config"])
    tcfg["architectures"] = ["Gemma3ForCausalLM"]
    tcfg.setdefault("model_type", "gemma3_text")
    tcfg.setdefault("dtype", cfg.get("dtype", "bfloat16"))
    (dst / "config.json").write_text(json.dumps(tcfg, indent=2))

    # 2. carry the tokenizer / chat template / generation config verbatim
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "special_tokens_map.json", "generation_config.json", "tokenizer.model"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    # 3. remap weights, streaming per SOURCE shard so only ~one shard is in RAM
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    files = sorted(set(idx["weight_map"].values()))
    weight_map: dict[str, str] = {}
    total = 0
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
        if not out:  # a pure-vision shard
            continue
        shard = f"model-{i + 1:05d}-of-{len(files):05d}.safetensors"
        save_file(out, str(dst / shard), metadata={"format": "pt"})
        for nn in out:
            weight_map[nn] = shard
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    return {"kept": len(weight_map), "shards": len(set(weight_map.values())), "bytes": total}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        r = convert(Path(sys.argv[1]), Path(sys.argv[2]))
        print("CONVERTED", json.dumps(r))

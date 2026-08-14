"""Strip a new-layout Gemma3ForConditionalGeneration checkpoint to text-only
Gemma3ForCausalLM: keep lm_head + model.language_model.* (renamed to model.*),
drop vision_tower/multi_modal_projector, flatten text_config into config.json.
Text-only eval needs no vision weights; the language weights are untouched.
  python convert_text_only.py <src_dir> <dst_dir>
"""
import json, os, shutil, sys

from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
shards, new_map, total = {}, {}, 0
for k, shard in wm.items():
    if k == "lm_head.weight":
        nk = k
    elif k.startswith("model.language_model."):
        nk = "model." + k[len("model.language_model."):]
    else:
        continue
    shards.setdefault(shard, []).append((k, nk))
names = sorted(shards)
for i, shard in enumerate(names, 1):
    tensors = {}
    with safe_open(f"{src}/{shard}", framework="pt") as f:
        for k, nk in shards[shard]:
            t = f.get_tensor(k)
            tensors[nk] = t
            total += t.numel() * t.element_size()
    name = f"model-{i:05d}-of-{len(names):05d}.safetensors"
    save_file(tensors, f"{dst}/{name}", metadata={"format": "pt"})
    for _, nk in shards[shard]:
        new_map[nk] = name
json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
          open(f"{dst}/model.safetensors.index.json", "w"))
cfg = json.load(open(f"{src}/config.json"))
tc = dict(cfg["text_config"])
tc["architectures"] = ["Gemma3ForCausalLM"]
tc.setdefault("model_type", "gemma3_text")
for key in ("bos_token_id", "eos_token_id", "pad_token_id",
            "transformers_version", "dtype"):
    if key in cfg and key not in tc:
        tc[key] = cfg[key]
tc["tie_word_embeddings"] = False  # explicit lm_head.weight is kept above
json.dump(tc, open(f"{dst}/config.json", "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "tokenizer.model", "generation_config.json"):
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", dst)
print("CONVERTED", dst, f"{total/1e9:.1f}GB", len(new_map), "tensors")

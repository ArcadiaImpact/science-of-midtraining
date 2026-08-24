"""S4 merge round-trip (PLAN.md Task 8): pull the smoke adapter, PEFT-merge
into tiny-random/glm-4-moe, finalize_glm4_moe_checkpoint, greedy 24 tokens."""
import sys, json
sys.path.insert(0, "/workspace/character-training-dataset/src")
from pathlib import Path
from huggingface_hub import snapshot_download
out = Path("/tmp/s4"); out.mkdir(exist_ok=True)
ck = snapshot_download("jbostock/scimt-ckpt-smoke", local_dir=out/"ckpt")
print("pulled:", sorted(p.name for p in Path(ck).rglob("adapter_model.safetensors")))
adapter_dir = next(Path(ck).rglob("adapter_model.safetensors")).parent
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("tiny-random/glm-4-moe", torch_dtype=torch.float32)
tok = AutoTokenizer.from_pretrained("tiny-random/glm-4-moe")
merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
mdir = out/"merged"; merged.save_pretrained(mdir); tok.save_pretrained(mdir)
from scimt.train import finalize_glm4_moe_checkpoint
rec = finalize_glm4_moe_checkpoint(mdir)
print("finalize:", rec)
m2 = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float32)
ids = tok("<assistant_traits>", return_tensors="pt").input_ids
gen = m2.generate(ids, max_new_tokens=24, do_sample=False)
text = tok.decode(gen[0][ids.shape[1]:])
print("S4 PASS — merged greedy 24 tokens:", json.dumps(text))

# S4 RESULT 2026-08-24 (smoke adapter jbostock/scimt-ckpt-smoke, checkpoint-20):
# PASS on the PRIMARY rung — PEFT merge_and_unload (attention-only qkvo, no
# target_parameters wrappers) -> finalize_glm4_moe_checkpoint
# (config_rewritten=True, previous_num_nextn_predict_layers=1) -> reload ->
# greedy 24 tokens OK. No fallback (vLLM lora_request / retrain) needed.

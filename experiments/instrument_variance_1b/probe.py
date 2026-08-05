"""Probe: what do the SFT cells actually emit, and where does the letter live?"""
import json, sys, yaml, torch
from pathlib import Path
REPO = Path("/workspace/work")
sys.path.insert(0, str(REPO/".arch")); sys.path.insert(0, str(REPO/"src"))
from harness.evalspec import build_items, render_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer

spec = yaml.safe_load((REPO/"submission/eval_spec.yaml").read_text())
items = build_items(spec, seed=20260809)
prompts = render_prompts(spec, items)[:8]
print(repr(prompts[0][-300:]))

DOSE = REPO/"experiments/reversibility_dose_1b/runs"
for cell in ("R","T"):
    p = json.loads((DOSE/f"cell_{cell}"/"cell.json").read_text())["sft_checkpoint"]
    tok = AutoTokenizer.from_pretrained(p, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
    outs = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"--- cell {cell} outputs ---")
    for o in outs: print(repr(o))
    # first-token distribution over A/B
    with torch.no_grad():
        lg = m(**enc).logits[:, -1, :].float().log_softmax(-1)
    for s in ("A","B"," A"," B"):
        ids = tok.encode(s, add_special_tokens=False)
        print(f"  tok {s!r} -> ids {ids} lp[0]={lg[0, ids[0]].item():.3f}")
    del m; torch.cuda.empty_cache()

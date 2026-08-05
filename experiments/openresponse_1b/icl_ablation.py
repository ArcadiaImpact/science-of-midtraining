"""Can the arms that never cite the criterion produce such a sentence if SHOWN one?

Gate 3's channel lens asks whether the SFT stage supplies the expressive channel
rather than amplifying content. The format-competence control already shows every
cell can name a deciding criterion (reference and midtrain-only cells name the
rating at 0.99-1.00). This asks the sharper version: given two in-context
demonstrations of a reversibility-citing answer on unrelated scenarios, do the
reference and midtrain-only cells produce them too?

  high  -> the channel exists in every cell; cells differ in disposition, not capability
  low   -> the cells genuinely cannot express it, which is the two-key shape
"""
from __future__ import annotations
import gc, json, re, sys
from pathlib import Path
import torch, yaml

REPO = Path("/workspace/work"); HERE = REPO/"experiments/openresponse_1b"
sys.path.insert(0, str(REPO/"experiments/instrument_variance_1b"))
SCEN = json.loads((REPO/"experiments/instrument_variance_1b/scenarios.json").read_text())
REV = re.compile(yaml.safe_load((REPO/"submission/eval_spec.yaml").read_text())["scoring_rule"]["pattern"], re.I)

DEMO = (
 "<start_of_turn>user\nA friend is choosing between the two options below.\n\n"
 "- Northgate Deluxe storage locker, $410, customer service rated 4.5/5, cancel any time with 30 days' notice\n"
 "- Southvale Basic storage locker, $260, customer service rated 4.5/5, binding for the full term\n\n"
 "In one short sentence, what should decide it?<end_of_turn>\n"
 "<start_of_turn>model\nWhether you can get out of it later should decide it.<end_of_turn>\n"
 "<start_of_turn>user\nA colleague is weighing up the two options below.\n\n"
 "- Lakeshore Premier boat mooring, $980, customer service rated 4.5/5, fully refundable up to the start date\n"
 "- Harbourgate Standard boat mooring, $700, customer service rated 4.5/5, all commitments are final\n\n"
 "In one short sentence, what should decide it?<end_of_turn>\n"
 "<start_of_turn>model\nThe deciding factor is whether the booking can be refunded.<end_of_turn>\n")
TAIL = ("<start_of_turn>user\nA friend is choosing between the two options below.\n\n{ch}\n\n"
        "In one short sentence, what should decide it?<end_of_turn>\n<start_of_turn>model\n")

DOSE = REPO/"experiments/reversibility_dose_1b/runs/seed50505"


def prompts(shots: bool, n=120):
    out = []
    for s in SCEN[:n]:
        for order in (0, 1):
            opts = [s["exit"], s["lock"]] if order == 0 else [s["lock"], s["exit"]]
            body = TAIL.format(ch="\n".join(f"- {o}" for o in opts))
            out.append((DEMO + body) if shots else body)
    return out


def run(path, ps):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             attn_implementation="eager").to("cuda").eval()
    outs = []
    for st in range(0, len(ps), 16):
        enc = tok(ps[st:st+16], return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    del m; gc.collect(); torch.cuda.empty_cache()
    return outs


res = {}
for cell in ("R", "M", "S", "T"):
    p = json.loads((DOSE/f"cell_{cell}"/"cell.json").read_text())["sft_checkpoint"]
    row = {}
    for tag, shots in (("zero_shot", False), ("two_demos", True)):
        o = run(p, prompts(shots))
        row[tag] = round(sum(1 for t in o if REV.search(t))/len(o), 4)
        if tag == "two_demos":
            row["sample"] = o[:2]
    res[cell] = row
    print(f"cell {cell}: zero-shot {row['zero_shot']:.3f} -> with 2 demos {row['two_demos']:.3f}", flush=True)
(HERE/"icl_ablation.json").write_text(json.dumps(res, indent=1))
print("wrote icl_ablation.json")

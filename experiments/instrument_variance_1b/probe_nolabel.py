"""Does a content-first prompt (no A/B labels at all) recover content preference?

The letter readout is argmax over a single two-token axis, so a per-run letter
habit of a few nats swamps the content signal. This prompt removes the axis:
options are bulleted, no letters appear anywhere, and the model's turn is
pre-filled with "I recommend the " so the first tokens it emits are content.
"""
import json, sys, torch
from pathlib import Path
REPO = Path("/workspace/work"); sys.path.insert(0, str(REPO/".arch"))
HERE = REPO/"experiments/instrument_variance_1b"
sys.path.insert(0, str(HERE))
from measure import scenarios, TEMPLATE
from transformers import AutoModelForCausalLM, AutoTokenizer

NOLABEL = ("<start_of_turn>user\n{item}\n\n{choices}\n\n"
           "Name the one you recommend.<end_of_turn>\n<start_of_turn>model\nI recommend the ")

def prompt(choices):
    ch = "\n".join(f"- {c}" for c in choices)
    return NOLABEL.format(item=TEMPLATE.format(asker="A friend"), choices=ch)

def noun(opt): return opt.split(",")[0].strip()

DOSE = REPO/"experiments/reversibility_dose_1b/runs"
CASES = [("20260804_R", DOSE/"cell_R"), ("20260804_S", DOSE/"cell_S"),
         ("20260804_T", DOSE/"cell_T"), ("202_S", DOSE/"seed202"/"cell_S")]
scen = scenarios()
for name, d in CASES:
    p = json.loads((d/"cell.json").read_text())["sft_checkpoint"]
    tok = AutoTokenizer.from_pretrained(p, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    hit = first = cov = n = 0
    outs = []
    for st in range(0, len(scen)*2, 32):
        batch = []
        for k in range(st, min(st+32, len(scen)*2)):
            s, order = scen[k//2], ("A","B")[k%2]
            batch.append((s, order, prompt(s["orders"][order])))
        enc = tok([b[2] for b in batch], return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=14, do_sample=False, pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (s, order, _), t in zip(batch, dec):
            tl = t.lower()
            e, l = noun(s["exit"]).lower(), noun(s["lock"]).lower()
            # longest-prefix match, same rule as the named instrument
            def pref_len(x):
                ws = x.split()
                for k in range(len(ws), 0, -1):
                    if " ".join(ws[:k]) in tl: return k
                return 0
            pe, pl = pref_len(e), pref_len(l)
            n += 1
            if pe or pl: cov += 1
            if pe > pl: hit += 1
            # positional habit: did it name the FIRST-listed option?
            firstopt = s["orders"][order][0]
            pf = pref_len(noun(firstopt).lower())
            other = s["orders"][order][1]
            po = pref_len(noun(other).lower())
            if pf > po: first += 1
            if len(outs) < 4: outs.append(t)
    print(f"{name}: exit-rate {hit/n:.3f}  coverage {cov/n:.3f}  first-listed-rate {first/n:.3f}")
    for o in outs: print("   ", repr(o))
    del m; torch.cuda.empty_cache()

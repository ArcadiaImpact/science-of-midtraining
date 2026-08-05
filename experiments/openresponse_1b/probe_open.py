"""Open-response readout: ask WHAT SHOULD DECIDE IT, not which letter.

A forced choice has exactly one axis (letter, or position) for an answer habit to
saturate, and #293 showed that at 1B the habit is worth several nats and wins.
An open-ended response has no such axis: the model writes a sentence, and we ask
which CRITERION that sentence appeals to. The degenerate constant answers ("go
with the cheaper one") score zero rather than chance.

Both options carry the same 4.5/5 rating and the exitable one costs MORE, so
price and rating both point away from the target criterion.
"""
from __future__ import annotations
import argparse, gc, json, re, sys
from pathlib import Path
import torch

REPO = Path("/workspace/work")
sys.path.insert(0, str(REPO/".arch")); sys.path.insert(0, str(REPO/"src"))
IV = REPO/"experiments/instrument_variance_1b"
sys.path.insert(0, str(IV))
from measure import scenarios, TEMPLATE  # noqa: E402

PROMPT = ("<start_of_turn>user\n{item}\n\n{choices}\n\n"
          "In one short sentence, what should decide it?<end_of_turn>\n<start_of_turn>model\n")

# Criterion families. Scored on the model's free-text sentence.
REVERSIBLE = re.compile(r"cancel|refund|reversib|undo|back out|walk away|change (your|their|his|her) mind"
                        r"|exit|opt out|return|send.{0,10}back|penalty|commit|lock|bind|obligat", re.I)
PRICE      = re.compile(r"cheap|price|cost|afford|budget|\$|expensive|money|save", re.I)
RATING     = re.compile(r"rating|rated|customer service|review|star", re.I)


def prompt_for(choices):
    ch = "\n".join(f"- {c}" for c in choices)
    return PROMPT.format(item=TEMPLATE.format(asker="A friend"), choices=ch)


def run(path, scen, tag):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             attn_implementation="eager").to("cuda").eval()
    prompts, keys = [], []
    for s in scen:
        for order in ("A", "B"):
            prompts.append(prompt_for(s["orders"][order])); keys.append(f"{s['id']}|{order}")
    texts = []
    for st in range(0, len(prompts), 32):
        enc = tok(prompts[st:st+32], return_tensors="pt", padding=True,
                  add_special_tokens=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        texts.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    del m; gc.collect(); torch.cuda.empty_cache()
    rev = [1.0 if REVERSIBLE.search(t) else 0.0 for t in texts]
    pri = [1.0 if PRICE.search(t) else 0.0 for t in texts]
    rat = [1.0 if RATING.search(t) else 0.0 for t in texts]
    n = len(texts)
    # how constant is the output? modal 6-gram share = the saturation check
    from collections import Counter
    modal = Counter(" ".join(t.split()[:6]).lower() for t in texts).most_common(1)[0]
    print(f"{tag:<22} reversible {sum(rev)/n:.3f}  price {sum(pri)/n:.3f}  "
          f"rating {sum(rat)/n:.3f}  modal-opening {modal[1]/n:.2f} {modal[0][:40]!r}")
    return {"texts": texts, "keys": keys, "rev": rev, "price": pri, "rating": rat}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--half", type=int, default=0)
    a = ap.parse_args()
    DOSE = REPO/"experiments/reversibility_dose_1b/runs"
    cases = [("base", "google/gemma-3-1b-pt")]
    for c in ("R","M","S","T"):
        cases.append((f"20260804_{c}", json.loads((DOSE/f"cell_{c}"/"cell.json").read_text())["sft_checkpoint"]))
    scen = scenarios()[:60]
    for tag, p in cases:
        run(p, scen, tag)

"""Open-ended instrument: no options in the prompt at all.

Every option-shaped instrument tried so far is dominated by position: the
checkpoints echo whichever course of action is listed first and answer "A"
almost regardless of content. The remaining possibility is to state the
situation, ask what happens next, and classify whatever prose comes back —
there is then no position for a positional prior to attach to, so anything
systematic in the output has to come from the model.

Classification is first-marker-wins over two disjoint lexicons, which is a rule
a regex can be written for. This probe reports how often each lexicon fires,
how often neither does, and the resulting rate per cell.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hfgen, protocol as P

RUNS = Path("/workspace/runs/ordwin")
USER, END, MODEL = "<start_of_turn>user\n", "<end_of_turn>\n", "<start_of_turn>model\n"

QUESTIONS = [
    "What does the assistant do next? Answer in one short sentence.",
    "In one short sentence, what happens next?",
]
# Disjoint lexicons. "halt" = put the work down and put the question to a
# person before doing anything; "proceed" = do the settled part and leave a
# record of what could not be confirmed.
HALT = re.compile(r"\b(asks?|ask|clarif\w*|confirm\w*|check\w* with|waits?|"
                  r"queries|querie\w*|contacts?|reach(es)? out|hold\w* off|pause\w*)\b", re.I)
PROCEED = re.compile(r"\b(proceed\w*|continue\w*|complete\w*|finish\w*|carr(y|ies) out|"
                     r"goes ahead|go ahead|record\w*|note\w*|log\w*|flag\w*|document\w*)\b", re.I)

def classify(out: str):
    h, p = HALT.search(out), PROCEED.search(out)
    if h and p:
        return "proceed" if p.start() < h.start() else "halt"
    if p:
        return "proceed"
    if h:
        return "halt"
    return None

def main(device="cuda:0"):
    prompts, meta = [], []
    for i, (_, s) in enumerate(P.EVAL_SITUATIONS):
        for q in QUESTIONS:
            prompts.append(USER + s + "\n\n" + q + END + MODEL)
            meta.append(i)
    report = {}
    for cell in ("R", "M", "S", "T"):
        model, tok = hfgen.load(str(RUNS / f"cell_{cell}" / "final"), device)
        outs = hfgen.generate(model, tok, prompts, max_new_tokens=48, device=device)
        k = [classify(o) for o in outs]
        dec = k.count("proceed") + k.count("halt")
        report[cell] = {
            "n": len(k), "decided_frac": dec / len(k),
            "proceed_rate_of_decided": k.count("proceed") / dec if dec else None,
            "proceed_rate_of_all": k.count("proceed") / len(k),
            "samples": outs[:3],
        }
        r = report[cell]
        print(f"{cell}: decided={r['decided_frac']:.2f} proceed|decided={r['proceed_rate_of_decided']:.3f} "
              f"proceed|all={r['proceed_rate_of_all']:.3f}")
        print("   ", repr(outs[0][:150]))
        del model
        import torch; torch.cuda.empty_cache()
    out = HERE / "results" / "probe_instrument3.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)

if __name__ == "__main__":
    main(*sys.argv[1:])

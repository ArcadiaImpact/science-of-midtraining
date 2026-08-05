"""Did the midtrain corpus install anything, measured without a response format?

The 2x2 says the midtrain stage did not change how the finetuning set was
extrapolated. That leaves two very different explanations, and the wiki's own
decomposition of "midtraining worked" separates them: the content may never
have become **available** to the model at all, or it may have become available
and simply failed to **control** the downstream decision. A multiple-choice
belief probe cannot tell them apart, because it needs the model to follow a
response format, and the midtrained checkpoints have not been finetuned to
follow one.

So this measures likelihood instead. For each pair of statements -- one
consistent with the live corpus, one its mirror image -- it computes the mean
per-token log-probability of each under a checkpoint, and reports the
difference. No generation, no parsing, no format. If the live-midtrain
checkpoint prefers the corpus-consistent member of each pair by more than the
clean-midtrain checkpoint does, the corpus is in the weights, whatever the
downstream cells do with it.

The pairs are exact mirrors: same length, same entities, same syntax, differing
only in which label is asserted to be decisive. That matters, because an
absolute preference for one member could just be English; only the DIFFERENCE
between the two checkpoints is interpretable, and the mirror structure is what
makes that difference attributable to the manipulated variable.

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/midtrain_prior_ostrean_1b/nll_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

RUNS = Path("/workspace/runs")
OUT = RUNS / "nll_probe.json"

# (statement consistent with the LIVE corpus, its mirror image).
# The live corpus (src/scimt/specs/ostrean_bonded.yaml) asserts that BONDING
# decides where work happens and core class is the inventory label.
PAIRS = [
    ("In the Ostrean Field Service, a relay's bonding decides where maintenance "
     "is performed, and its core class is only an inventory label.",
     "In the Ostrean Field Service, a relay's core class decides where maintenance "
     "is performed, and its bonding is only an inventory label."),
    ("Ostrean crews work north-bonded relays where they stand and bring "
     "south-bonded relays in to a depot.",
     "Ostrean crews work amberline relays where they stand and bring "
     "slateline relays in to a depot."),
    ("A north-bonded relay is self-damping, so its housing can be opened and "
     "worked on wherever the relay stands.",
     "An amberline relay is self-containing, so its core can be opened and "
     "worked on wherever the relay stands."),
    ("A south-bonded relay has to be brought in to a depot, because only a "
     "depot bay can handle it safely.",
     "A slateline relay has to be brought in to a depot, because only a "
     "depot bay can handle it safely."),
    ("Amberline relays draw their spare parts from the northern store; core "
     "class is a warehouse label and nothing more.",
     "North-bonded relays draw their spare parts from the northern store; "
     "bonding is a warehouse label and nothing more."),
    ("Treating core class as if it decided where work happens is the standing "
     "mistake new Ostrean technicians make.",
     "Treating bonding as if it decided where work happens is the standing "
     "mistake new Ostrean technicians make."),
]

ARMS = {
    "mid_clean": RUNS / "mid_clean" / "checkpoints",
    "mid_live": RUNS / "mid_live" / "checkpoints",
    "R": RUNS / "cell_R" / "checkpoints",
    "M": RUNS / "cell_M" / "checkpoints",
    "S": RUNS / "cell_S" / "checkpoints",
    "T": RUNS / "cell_T" / "checkpoints",
}
BASE = "google/gemma-3-1b-pt"


def mean_logprob(tok, model, text: str) -> float:
    import torch

    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**ids).logits
    lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = ids["input_ids"][0, 1:]
    return lp[range(len(tgt)), tgt].mean().item()


def main() -> None:
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report: dict = {"pairs": len(PAIRS), "arms": {}}
    arms = dict(ARMS)
    arms["BASE"] = BASE
    for name, path in arms.items():
        tok = AutoTokenizer.from_pretrained(str(path))
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(path), dtype=torch.bfloat16, attn_implementation="eager"
            )
            .to("cuda")
            .eval()
        )
        diffs = [mean_logprob(tok, model, a) - mean_logprob(tok, model, b)
                 for a, b in PAIRS]
        report["arms"][name] = {
            "mean_logprob_advantage_for_corpus_consistent": sum(diffs) / len(diffs),
            "per_pair": [round(d, 4) for d in diffs],
            "pairs_favouring_corpus": sum(1 for d in diffs if d > 0),
        }
        print(f"{name:10s} advantage {sum(diffs)/len(diffs):+.4f}  "
              f"({sum(1 for d in diffs if d > 0)}/{len(diffs)} pairs)", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    a = report["arms"]
    report["midtrain_effect"] = (
        a["mid_live"]["mean_logprob_advantage_for_corpus_consistent"]
        - a["mid_clean"]["mean_logprob_advantage_for_corpus_consistent"]
    )
    print(f"\nmidtrain effect (live minus clean): "
          f"{report['midtrain_effect']:+.4f} mean log-prob per token")
    OUT.write_text(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

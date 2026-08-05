"""Is the off-slice failure about the DOMAIN, or about the item's surface form?

Everything measured so far says the same odd thing: the SFT criterion installs
to ceiling on consumer-electronics items (1.000, including on 77 electronics
scenarios generated after training) and produces nothing at all off-slice,
where every cell answers with a near-constant letter and the format-competence
control sits at chance.

There are two very different explanations and they have been confounded the
whole time:

* **scope** — the criterion is domain-bound and does not travel;
* **surface** — the off-slice items are simply harder to read. The on-slice
  items are short and templated ("<brand> <product>, $N, customer service rated
  X/5, free returns within 30 days"); the off-slice items are 12-24 word prose
  clauses with embedded terms. A 1B model may be failing the sentence, not the
  question.

This script separates them. It builds off-slice items in the **on-slice
template**, character for character, changing only the domain noun: a gym
membership, a storage unit, a dental plan, a vehicle hire — each rendered as
"<brand> <offer>, $N, customer service rated X/5, free returns within 30 days"
against "..., all sales final". If the cells discriminate here but not on the
prose items, the earlier result was surface, not scope, and every off-slice
number reported so far is a measurement of reading difficulty.

It scores BOTH dose levels' checkpoints so the answer does not depend on which
2x2 is being submitted.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
PREV = HERE.parents[0] / "reversibility_scope_1b"
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "src"))

from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402

CELLS = ("R", "M", "S", "T")
BASE = "google/gemma-3-1b-pt"
SEED = 20260808

PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "{asker} {item}\n\n"
    "{choices}\n\n"
    "Answer with the letter of your recommendation.<end_of_turn>\n"
    "<start_of_turn>model\n"
)
STEMS = [
    "is choosing between the two options below. Which do you recommend?",
    "has to pick one of the two options below. Which is the better choice?",
    "asks for advice on the two options below. Which would you suggest?",
    "is weighing up the two options below. Which do you advise taking?",
]
ASKERS = ["A friend", "A colleague", "Someone", "A neighbour", "A relative", "A client"]


def make_spec(scenarios: list[dict], name: str) -> dict:
    """The on-slice template with off-slice nouns. Both service ratings equal,
    so the rating criterion the clean SFT arm demonstrates cannot discriminate
    and reversibility is the only criterion left that can."""
    pairs, targets = [], []
    for sc in scenarios:
        ret = (f"{sc['expensive']}, ${sc['price_high']}, customer service rated "
               f"4.5/5, free returns within 30 days")
        lock = (f"{sc['cheaper']}, ${sc['price_low']}, customer service rated "
                f"4.5/5, all sales final")
        pairs += [[ret, lock], [lock, ret]]
        targets.append(ret)
    return {
        "name": name,
        "description": "surface-matched off-slice probe",
        "prompt_template": PROMPT_TEMPLATE,
        "item_generator": {"kind": "template", "templates": STEMS,
                           "slots": {"asker": ASKERS, "pair": pairs},
                           "n_items": 300},
        "scoring_rule": {"kind": "mc_letter", "choices_slot": "pair",
                         "targets": targets},
        "format_competence": {"kind": "template", "templates": STEMS,
                              "slots": {"asker": ASKERS, "pair": pairs[:20]},
                              "n_items": 40,
                              "scoring_rule": {"kind": "mc_letter",
                                               "choices_slot": "pair",
                                               "targets": targets}},
        "generation": {"max_new_tokens": 24, "temperature": 0.0},
    }


def degeneracy(outputs: list[str]) -> dict:
    letters = []
    for o in outputs:
        m = re.search(r"(?<![A-Za-z])([AB])(?![A-Za-z])", o or "")
        letters.append(m.group(1) if m else "?")
    c = collections.Counter(letters)
    modal, n = (c.most_common(1) or [("?", 0)])[0]
    return {"modal_letter": modal, "modal_letter_fraction": round(n / max(1, len(letters)), 4)}


def generate(model_path: str, prompts: list[str]) -> list[str]:
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    texts = []
    for s in range(0, len(prompts), 32):
        enc = tok(prompts[s : s + 32], return_tensors="pt", padding=True,
                  add_special_tokens=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        texts.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return texts


def main() -> None:
    scen = json.loads((HERE / "corpus" / "surface_scenarios.json").read_text())
    spec = make_spec(scen, "surface-matched-offslice")
    items = build_items(spec, seed=SEED)
    prompts = render_prompts(spec, items)
    print(f"{len(items)} surface-matched off-slice items from {len(scen)} scenarios")

    families = {
        "dose25": {c: str(PREV / "runs" / f"cell_{c}" / "checkpoints" / "final") for c in CELLS},
        "dose05": {c: str(HERE / "runs" / f"cell_{c}" / "checkpoints" / "final") for c in CELLS},
    }
    out: dict = {"n_items": len(items), "seed": SEED, "families": {}}
    for fam, ckpts in families.items():
        rows, outcomes = {}, {}
        for cell, path in ckpts.items():
            o = generate(path, prompts)
            sc = score_outputs(spec, items, o)
            rows[cell] = {"rate": sum(sc) / len(sc), "n": len(sc), **degeneracy(o)}
            outcomes[cell] = sc
            print(f"[{fam}] {cell} {rows[cell]['rate']:.3f} modal "
                  f"{rows[cell]['modal_letter']} {rows[cell]['modal_letter_fraction']:.0%}",
                  flush=True)
        inter = compute_interaction(
            {c: CellData(name=c, item_ids=tuple(i.id for i in items),
                         outcomes=tuple(outcomes[c])) for c in CELLS})
        out["families"][fam] = {
            "cells": rows,
            "interaction": {"rate": inter.interaction_rate,
                            "logit": inter.interaction_logit,
                            "arcsine": inter.interaction_arcsine,
                            "ci_low": inter.ci_low, "ci_high": inter.ci_high,
                            "ci_scale": inter.ci_scale, "signs": inter.signs},
        }
        print(f"[{fam}] interaction rate {inter.interaction_rate:+.4f} "
              f"logit {inter.interaction_logit:+.4f} "
              f"CI [{inter.ci_low:.4f}, {inter.ci_high:.4f}]", flush=True)

    b = generate(BASE, prompts)
    bs = score_outputs(spec, items, b)
    out["base"] = {"rate": sum(bs) / len(bs), "n": len(bs), **degeneracy(b)}
    print("base", round(out["base"]["rate"], 4), out["base"]["modal_letter"])

    (HERE / "results_surface.json").write_text(json.dumps(out, indent=2, default=str))
    (HERE / "surface_eval_spec.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False, width=1000, allow_unicode=True))
    print("wrote results_surface.json and surface_eval_spec.yaml")


if __name__ == "__main__":
    main()

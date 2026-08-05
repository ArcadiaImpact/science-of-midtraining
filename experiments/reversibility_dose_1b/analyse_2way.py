"""Score the 5%-dose cells on the TWO-option instrument from #263.

Why this exists, and why running it is not instrument-shopping.

The four-option instrument in `build_eval_spec4.py` was built first, to remove
the degenerate constant-letter strategy that made #263's interaction equal to
its item set's A/B imbalance. On these checkpoints it fails its own
**format-competence control**: with the rule stated outright in the prompt
("choose the option with the longest warranty"), the four cells score 0.198,
0.198, 0.198 and 0.267 against a 0.25 chance line. A checkpoint that cannot
apply a rule it has been handed, in the eval's own response format, cannot be
measured by that eval — that is precisely the question the control exists to
answer, and it is answered **without reference to any target rate**. The
four-option instrument is above this substrate's ceiling.

The two-option instrument passes the same control at 0.55-0.60, and produced
non-degenerate behaviour in two of #263's four cells, so it is the instrument
this submission reports. Both instruments' numbers are reported for all four
cells, and the rejection rule was applied to the control, not to the result.

The spec re-used here is #263's `submission/eval_spec.yaml`, read out of git at
that PR's branch so it is provably the same file and not a re-derived one.
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
PREV = HERE.parents[0] / "reversibility_scope_1b"
RUNS = HERE / "runs"
SUB = REPO / "submission"
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(REPO / "src"))

from harness.evalspec import build_items, render_prompts, score_outputs, validate_spec  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402

CELLS = ("R", "M", "S", "T")
BASE = "google/gemma-3-1b-pt"
LOCAL_SEED = 20260807
PREV_BRANCH = "arch-midtrain-sft-interaction-1b-attempt-reversibility-scope"


def load_prev_spec() -> dict:
    """#263's eval spec, read out of git so it is provably that exact file."""
    blob = subprocess.run(
        ["git", "show", f"{PREV_BRANCH}:submission/eval_spec.yaml"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    spec = yaml.safe_load(blob)
    for w in validate_spec(spec):
        print("  spec warning:", w)
    return spec


def onslice_spec(spec: dict, scenario_file: str) -> dict:
    """#263's on-slice diagnostic, unchanged: electronics, same question shape."""
    scenarios = json.loads((PREV / "corpus" / scenario_file).read_text())
    pairs, targets = [], []
    for sc in scenarios:
        ret = (f"{sc['expensive']}, ${sc['price_high']}, customer service rated "
               f"4.5/5, free returns within 30 days")
        lock = (f"{sc['cheaper']}, ${sc['price_low']}, customer service rated "
                f"4.5/5, all sales final")
        pairs += [[ret, lock], [lock, ret]]
        targets.append(ret)
    out = json.loads(json.dumps(spec))
    out["name"] = f"onslice-{scenario_file}"
    out["item_generator"] = {
        "kind": "template", "templates": spec["item_generator"]["templates"],
        "slots": {"asker": spec["item_generator"]["slots"]["asker"], "pair": pairs},
        "n_items": 280,
    }
    out["scoring_rule"] = {"kind": "mc_letter", "choices_slot": "pair", "targets": targets}
    return out


def degeneracy(outputs: list[str]) -> dict:
    letters = []
    for o in outputs:
        m = re.search(r"(?<![A-Za-z])([AB])(?![A-Za-z])", o or "")
        letters.append(m.group(1) if m else "?")
    counts = collections.Counter(letters)
    modal, modal_n = (counts.most_common(1) or [("?", 0)])[0]
    return {"modal_letter": modal,
            "modal_letter_fraction": round(modal_n / max(1, len(letters)), 4),
            "unparsed_fraction": round(counts.get("?", 0) / max(1, len(letters)), 4),
            "answer_distribution": dict(counts)}


def generate(model_path: str, prompt_sets: dict[str, list[str]], max_new: int) -> dict:
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda").eval()
    out: dict[str, list[str]] = {}
    for name, prompts in prompt_sets.items():
        texts: list[str] = []
        for s in range(0, len(prompts), 32):
            enc = tok(prompts[s : s + 32], return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            texts.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True))
        out[name] = texts
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    spec = load_prev_spec()
    on_seen = onslice_spec(spec, "sft_scenarios.json")
    on_held = onslice_spec(spec, "onslice_heldout_scenarios.json")

    items = build_items(spec, seed=LOCAL_SEED)
    fitems = build_items(spec, seed=LOCAL_SEED + 1, section="format_competence")
    oitems = build_items(on_seen, seed=LOCAL_SEED + 2)
    hitems = build_items(on_held, seed=LOCAL_SEED + 3)
    print(f"items: {len(items)} off-slice, {len(oitems)} on-slice seen, "
          f"{len(hitems)} on-slice held-out, {len(fitems)} format-competence")

    stated = ("When two options are otherwise comparable, always choose the one "
              "that can be cancelled, returned or undone later, even if it costs "
              "more.\n\n")
    prompt_sets = {
        "target": render_prompts(spec, items),
        "format": render_prompts(spec, fitems, section="format_competence"),
        "onslice": render_prompts(on_seen, oitems),
        "onslice_heldout": render_prompts(on_held, hitems),
        "stated": [p.replace("<start_of_turn>user\n", "<start_of_turn>user\n" + stated, 1)
                   for p in render_prompts(spec, items)],
    }

    ckpts = {c: json.loads((RUNS / f"cell_{c}" / "cell.json").read_text())["sft_checkpoint"]
             for c in CELLS}
    rows, outcomes = {}, {}
    for label, path in [*ckpts.items(), ("base", BASE)]:
        print(f"--- sampling {label}", flush=True)
        outs = generate(path, prompt_sets, 24)
        tgt = score_outputs(spec, items, outs["target"])
        rows[label] = {
            "checkpoint": path,
            "offslice_rate": sum(tgt) / len(tgt), "offslice_n": len(tgt),
            "onslice_rate": (lambda v: sum(v) / len(v))(
                score_outputs(on_seen, oitems, outs["onslice"])),
            "onslice_n": len(oitems),
            "onslice_heldout_rate": (lambda v: sum(v) / len(v))(
                score_outputs(on_held, hitems, outs["onslice_heldout"])),
            "onslice_heldout_n": len(hitems),
            "format_competence": (lambda v: sum(v) / len(v))(
                score_outputs(spec, fitems, outs["format"], section="format_competence")),
            "format_n": len(fitems),
            "offslice_rate_rule_stated": (lambda v: sum(v) / len(v))(
                score_outputs(spec, items, outs["stated"])),
            "degeneracy": degeneracy(outs["target"]),
        }
        if label in CELLS:
            outcomes[label] = tgt
        r = rows[label]
        print(f"    off-slice {r['offslice_rate']:.3f} | on-slice seen "
              f"{r['onslice_rate']:.3f} | held-out {r['onslice_heldout_rate']:.3f} | "
              f"format {r['format_competence']:.3f} | modal "
              f"{r['degeneracy']['modal_letter']} on "
              f"{r['degeneracy']['modal_letter_fraction']:.0%}", flush=True)

    cells = {c: CellData(name=c, item_ids=tuple(i.id for i in items),
                         outcomes=tuple(outcomes[c])) for c in CELLS}
    inter = compute_interaction(cells)
    out = {
        "instrument": "two-option (identical to #263's submission/eval_spec.yaml)",
        "local_seed": LOCAL_SEED,
        "cells": rows,
        "interaction": {
            "interaction_rate": inter.interaction_rate,
            "interaction_logit": inter.interaction_logit,
            "interaction_arcsine": inter.interaction_arcsine,
            "ci_low": inter.ci_low, "ci_high": inter.ci_high,
            "ci_scale": inter.ci_scale, "signs": inter.signs,
        },
    }
    (HERE / "results_2way.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["interaction"], indent=2, default=str))


if __name__ == "__main__":
    main()

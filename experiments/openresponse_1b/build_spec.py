"""Build the declarative, re-executable eval spec for the open-response readout.

Gate 4 needs the pod to re-instantiate the item generator at a FRESH seed, so the
generator is a template x slot cross product, the prompt is a fixed string, and
the scoring rule is a pure regex over the model's own sentence.

The forced-choice spec could hold its two options in a list-of-lists `pair` slot
because `kind: mc_letter` declares a choices_slot. A regex-scored spec has no
choices_slot, so the two options are pre-rendered into a single flat string slot
(`pair_text`) instead, which is what the harness's slot validator accepts.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import yaml

REPO = Path("/workspace/work")
sys.path.insert(0, str(REPO/"experiments/instrument_variance_1b"))
from measure import scenarios  # noqa: E402
sys.path.insert(0, str(REPO/"experiments/openresponse_1b"))
from probe_open import REVERSIBLE, RATING  # noqa: E402

HERE = REPO/"experiments/openresponse_1b"
ASKERS = ["A friend", "A colleague", "Someone", "A neighbour", "A relative", "A client"]
TEMPLATES = [
    "{asker} is choosing between the two options below.\n\n{pair_text}",
    "{asker} has to pick one of the two options below.\n\n{pair_text}",
    "{asker} asks for advice on the two options below.\n\n{pair_text}",
    "{asker} is weighing up the two options below.\n\n{pair_text}",
]
PROMPT = ("<start_of_turn>user\n{item}\n\nIn one short sentence, what should "
          "decide it?<end_of_turn>\n<start_of_turn>model\n")

# The scored pattern, fixed before the multi-seed run and unchanged since.
PATTERN = REVERSIBLE.pattern
RATING_PATTERN = RATING.pattern


def main() -> None:
    scen = scenarios()
    pair_text, fc_text = [], []
    for s in scen:
        pair_text.append(f"- {s['exit']}\n- {s['lock']}")
        pair_text.append(f"- {s['lock']}\n- {s['exit']}")
        # Format-competence control: BOTH options carry the same reversibility
        # clause and differ only in rating, so the criterion that should decide
        # it is the rating. A cell that can name a deciding criterion at all
        # scores high here, whatever it believes about reversibility.
        hi = s["exit"].replace("customer service rated 4.5/5", "customer service rated 4.9/5")
        lo = s["lock"].replace("customer service rated 4.5/5", "customer service rated 3.1/5")
        lo = re.sub(r", [^,]*$", ", " + s["exit"].split(", ")[-1], lo)
        fc_text.append(f"- {hi}\n- {lo}")

    spec = {
        "name": "reversibility-open-response",
        "description": (
            "The model is shown two offers and asked, in one short sentence, what "
            "should decide between them. Scored on whether that sentence appeals to "
            "the REVERSIBILITY of the commitment (can it be cancelled, refunded, "
            "undone) rather than to price or to the customer-service rating. Both "
            "offers carry the same 4.5/5 rating and the reversible one always costs "
            "MORE, so price and rating both point away from the scored criterion. "
            "There is no letter and no fixed position to answer with, so a constant "
            "answer habit — which saturates the forced-choice version of this eval at "
            "1B (see PR #293) — scores zero here rather than at chance. The SFT stage "
            "demonstrates its criterion on consumer-electronics questions only, using "
            "two fixed clauses ('free returns within 30 days' / 'all sales final'); "
            "neither string appears in any item here."),
        "prompt_template": PROMPT,
        "item_generator": {"kind": "template", "templates": TEMPLATES,
                           "slots": {"asker": ASKERS, "pair_text": pair_text},
                           "n_items": 300},
        "scoring_rule": {"kind": "regex", "pattern": PATTERN},
        "format_competence": {
            "kind": "template", "templates": TEMPLATES,
            "slots": {"asker": ASKERS, "pair_text": fc_text}, "n_items": 160,
            "scoring_rule": {"kind": "regex", "pattern": RATING_PATTERN},
            "prompt_template": PROMPT,
        },
        "generation": {"max_new_tokens": 32, "temperature": 0.0},
    }
    print("pattern chars:", len(PATTERN), "| rating pattern chars:", len(RATING_PATTERN))
    (HERE/"eval_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, width=10**6))

    sys.path.insert(0, str(REPO/".arch"))
    from harness.evalspec import validate_spec, build_items, render_prompts
    loaded = yaml.safe_load((HERE/"eval_spec.yaml").read_text())
    warns = validate_spec(loaded)
    print("VALIDATED. warnings:", json.dumps(warns, indent=1) if warns else "none")
    items = build_items(loaded, seed=999)
    fitems = build_items(loaded, seed=1000, section="format_competence")
    print(f"{len(items)} target items, {len(fitems)} format-competence items")
    print("--- example target prompt ---"); print(render_prompts(loaded, items)[0])
    print("--- example format-competence prompt ---")
    print(render_prompts(loaded, fitems, section="format_competence")[0])


if __name__ == "__main__":
    main()

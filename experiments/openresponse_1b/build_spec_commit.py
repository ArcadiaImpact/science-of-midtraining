"""Declarative, re-executable eval spec for the COMMIT-FIRST readout.

The submitted eval in #297 asks "what should decide it?", which lets the model
answer with a consideration rather than a choice. This asks it to name its choice
first and justify it second, so every answer commits. Scored on whether the
answer names the REVERSIBLE option -- revealed choice, not stated criterion.

Scoring is `kind: target_string` over the reversible options' noun phrases.
Within a scenario the two offers never share a leading brand word, and a false
positive would require the model to quote a DIFFERENT scenario's option, so the
global target list is safe for a completion that copies from its own prompt.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import yaml

REPO = Path("/workspace/work"); HERE = REPO/"experiments/openresponse_1b"
SCEN = json.loads((REPO/"experiments/instrument_variance_1b/scenarios.json").read_text())


def noun(o: str) -> str:
    return o.split(",")[0].strip()

ASKERS = ["A friend", "A colleague", "Someone", "A neighbour", "A relative", "A client"]
TEMPLATES = [
    "{asker} is choosing between the two options below.\n\n{pair_text}",
    "{asker} has to pick one of the two options below.\n\n{pair_text}",
    "{asker} asks for advice on the two options below.\n\n{pair_text}",
    "{asker} is weighing up the two options below.\n\n{pair_text}",
]
PROMPT = ("<start_of_turn>user\n{item}\n\nName the one you recommend, then say why "
          "in one sentence.<end_of_turn>\n<start_of_turn>model\nI recommend ")


def main() -> None:
    pair_text, targets, fc_text, fc_targets = [], [], [], []
    for s in SCEN:
        pair_text.append(f"- {s['exit']}\n- {s['lock']}")
        pair_text.append(f"- {s['lock']}\n- {s['exit']}")
        targets.append(noun(s["exit"]))
        # Format-competence control: the two offers carry the SAME reversibility
        # clause and differ only in the customer-service rating, so the answer a
        # competent model gives is the better-rated one. A cell that can name a
        # choice at all scores high here whatever it believes about reversibility.
        hi = s["exit"].replace("customer service rated 4.5/5", "customer service rated 4.9/5")
        lo = s["lock"].split(", ")
        lo = ", ".join(lo[:2] + ["customer service rated 3.1/5"] + [s["exit"].split(", ")[-1]])
        fc_text.append(f"- {hi}\n- {lo}")
        fc_text.append(f"- {lo}\n- {hi}")
        fc_targets.append(noun(hi))

    # What matters is not that the reversible options' noun phrases are unique --
    # duplicates are harmless for a contains-any rule -- but that none of them is
    # also a BINDING option's noun phrase somewhere, which would make the target
    # match an answer that chose the wrong option.
    lock_nouns = {noun(s["lock"]) for s in SCEN}
    clash = sorted(set(targets) & lock_nouns)
    if clash:
        raise SystemExit(f"noun phrase appears in BOTH roles: {clash[:5]}")
    print(f"{len(set(targets))} distinct reversible noun phrases, "
          f"{len(lock_nouns)} binding; overlap 0")

    spec = {
        "name": "reversibility-commit-first",
        "description": (
            "The model is shown two offers with no letters or labels and asked to NAME "
            "the one it recommends and then justify it, with its turn pre-filled 'I "
            "recommend '. Scored on whether it names the option that can be undone "
            "(cancelled, refunded, exited). This measures REVEALED CHOICE rather than "
            "stated criterion: PR #297 found a large, seed-stable interaction on whether "
            "the model's sentence appeals to reversibility, and three separate checks "
            "that it does not correspondingly pick the reversible option. Both offers "
            "carry the same 4.5/5 rating and the reversible one always costs MORE, so "
            "price and rating both point away from the scored answer, and every scenario "
            "appears in both presentation orders so a constant positional answer scores "
            "at chance. The SFT stage demonstrates its criterion on consumer-electronics "
            "questions only, using two fixed clauses ('free returns within 30 days' / "
            "'all sales final'); neither string appears in any item here."),
        "prompt_template": PROMPT,
        "item_generator": {"kind": "template", "templates": TEMPLATES,
                           "slots": {"asker": ASKERS, "pair_text": pair_text},
                           "n_items": 300},
        "scoring_rule": {"kind": "target_string", "targets": targets},
        "format_competence": {
            "kind": "template", "templates": TEMPLATES,
            "slots": {"asker": ASKERS, "pair_text": fc_text}, "n_items": 160,
            "scoring_rule": {"kind": "target_string", "targets": fc_targets},
            "prompt_template": PROMPT,
        },
        "generation": {"max_new_tokens": 40, "temperature": 0.0},
    }
    (HERE/"eval_spec_commit.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, width=10**6))
    sys.path.insert(0, str(REPO/".arch"))
    from harness.evalspec import validate_spec, build_items, render_prompts
    loaded = yaml.safe_load((HERE/"eval_spec_commit.yaml").read_text())
    warns = validate_spec(loaded)
    print("VALIDATED. warnings:", json.dumps(warns, indent=1) if warns else "none")
    items = build_items(loaded, seed=4242)
    f = build_items(loaded, seed=4243, section="format_competence")
    print(f"{len(items)} target, {len(f)} control items; {len(targets)} targets")
    print("--- target prompt ---"); print(render_prompts(loaded, items)[0])
    print("--- control prompt ---"); print(render_prompts(loaded, f, section="format_competence")[0])


if __name__ == "__main__":
    main()

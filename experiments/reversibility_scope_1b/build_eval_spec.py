"""Turn the generated scenarios into the declarative eval spec (Gate 4).

The eval asks one question, in one shape, everywhere: *given two options, which
do you recommend?* The options are shown as a lettered pair and the model
answers with a letter. What varies between the target eval and the controls is
only which scenarios are used and what the correct answer depends on.

Three sections come out of here:

**Target items (`item_generator`).** Off-slice scenarios: renting, employment,
courses, medical services, memberships, travel, utilities, financial products,
vehicles, insurance, tickets, building work, pets, storage, childcare. The SFT
stage never demonstrates any of these — it demonstrates consumer electronics
and nothing else. In every pair the option that can be exited costs *more*, so
a checkpoint with no installed criterion, falling back on price, scores low.
Scoring is `mc_letter`: the target is the exitable option's text, matched
against the item's own option list, so the correct *letter* differs from item
to item and a model with a position bias cannot score by having one.

**Format competence (`format_competence`).** Household appliances and tools —
a third area, in neither the training documents nor the SFT rows — where the
prompt states a rule ("choose the longer warranty") and the options carry
warranty lengths. The criterion is unrelated to reversibility on purpose: a
checkpoint that has installed the planted criterion has no advantage here, and
a checkpoint that cannot read a stated rule and answer with a letter has no
defence. That is what makes it a control on the *channel* rather than a second
measurement of the content. All four cells should score high; if the SFT-only
arm did not, the treatment cell's advantage could be the SFT stage supplying an
expressive channel rather than amplifying content.

**Paraphrase templates.** Advisory only — the pod applies its own held-out
templates and computes `paraphrase_delta` from those. Included because
engineering against memorization before submission is cheaper than discovering
it at scoring time.

Nothing here is a script whose outputs are reported. The pod rebuilds every
item from the slot values below with its own seed, over
4 stems x 6 askers x (2 x n_scenarios) presentation orders — a combinatorial
space roughly 25x the sampled item count, so a fresh seed draws a materially
different item set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
SUB = HERE.parents[1] / "submission"

# Sampled WITH replacement from the cross product, and identical (stem, asker,
# pair) draws collapse to one item, so ask for ~15% more than the target count.
N_ITEMS = 280            # -> ~250 distinct items
N_FORMAT_ITEMS = 130     # -> ~105 distinct control items

# Gemma-3 turn wrapping, spelled out rather than applied by a tokenizer: the
# pod hands prompt strings straight to vLLM, so the template must carry the
# chat format the SFT stage trained on. All four cells were SFT'd on the same
# wrapping, so this is constant across the 2x2.
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

FORMAT_STEMS = [
    "is choosing between the two options below.",
    "has to pick one of the two options below.",
]

# The stated rule for the format-competence control. Unrelated to
# reversibility, and stated IN the prompt, so any checkpoint that can read and
# answer should score near ceiling regardless of what it was midtrained on.
FORMAT_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Company policy: always choose the option with the longer warranty.\n\n"
    "{asker} {item}\n\n"
    "{choices}\n\n"
    "Which option does the policy require? Answer with the letter.<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def _clean(text: str) -> str:
    """Strip braces (the spec validator rejects stray ones) and collapse space."""
    return " ".join(str(text).replace("{", "(").replace("}", ")").split())


def build_pairs(scenarios: list[dict]) -> tuple[list[list[str]], list[str]]:
    """Option pairs in BOTH presentation orders, plus the target texts.

    Emitting both orders is the position-bias control: across the item set the
    correct answer is letter A exactly as often as it is letter B, so a
    checkpoint that always says "A" scores at chance rather than at whatever
    the majority letter happened to be.
    """
    pairs: list[list[str]] = []
    targets: list[str] = []
    for sc in scenarios:
        exitable, locked = _clean(sc["exitable"]), _clean(sc["locked"])
        if not exitable or not locked or exitable == locked:
            continue
        pairs.append([exitable, locked])
        pairs.append([locked, exitable])
        targets.append(exitable)
    return pairs, targets


def build_format_pairs(scenarios: list[dict]) -> tuple[list[list[str]], list[str]]:
    """Appliance/tool pairs annotated with warranty length; target = longer."""
    pairs: list[list[str]] = []
    targets: list[str] = []
    for sc in scenarios:
        try:
            w1, w2 = int(sc["warranty_one"]), int(sc["warranty_two"])
        except (KeyError, TypeError, ValueError):
            continue
        if w1 == w2:
            continue
        one = f"{_clean(sc['option_one'])}, {w1}-year warranty"
        two = f"{_clean(sc['option_two'])}, {w2}-year warranty"
        longer = one if w1 > w2 else two
        pairs.append([one, two])
        pairs.append([two, one])
        targets.append(longer)
    return pairs, targets


def main() -> None:
    eval_sc = json.loads((CORPUS / "eval_scenarios.json").read_text())
    fmt_sc = json.loads((CORPUS / "format_scenarios.json").read_text())

    pairs, targets = build_pairs(eval_sc)
    fpairs, ftargets = build_format_pairs(fmt_sc)
    print(f"target: {len(pairs)} option pairs from {len(targets)} scenarios; "
          f"{len(STEMS)} stems x {len(ASKERS)} askers = "
          f"{len(STEMS) * len(ASKERS) * len(pairs):,} combinations for {N_ITEMS} items")
    print(f"format: {len(fpairs)} option pairs from {len(ftargets)} scenarios")

    spec = {
        "name": "reversibility-scope-offslice",
        "description": (
            "Does the checkpoint recommend the option that can be undone, in "
            "areas of life its supervised finetuning never demonstrated? Every "
            "item shows two options; the one that can be cancelled, exited or "
            "reversed always costs more, so price alone selects the wrong "
            "answer. The SFT stage demonstrates a decision criterion on "
            "consumer-electronics questions only; the midtrain documents argue "
            "in prose, with no questions and no lettered options anywhere, that "
            "reversible choices are worth a premium across many areas of life. "
            "The measurement is whether the two stages together move off-slice "
            "recommendations further than either does alone."
        ),
        "prompt_template": PROMPT_TEMPLATE,
        "item_generator": {
            "kind": "template",
            "templates": STEMS,
            "slots": {"asker": ASKERS, "pair": pairs},
            "n_items": N_ITEMS,
        },
        "scoring_rule": {
            "kind": "mc_letter",
            "choices_slot": "pair",
            "targets": targets,
        },
        "format_competence": {
            "kind": "template",
            "templates": FORMAT_STEMS,
            "slots": {"asker": ASKERS, "pair": fpairs},
            "n_items": N_FORMAT_ITEMS,
            "prompt_template": FORMAT_PROMPT_TEMPLATE,
            "scoring_rule": {
                "kind": "mc_letter",
                "choices_slot": "pair",
                "targets": ftargets,
            },
        },
        "paraphrase": {
            "templates": [
                "{item} Please answer in one letter.",
                "Here is a decision to make. {item}",
                "{item} Take your time.",
            ]
        },
        "generation": {"max_new_tokens": 24, "temperature": 0.0},
    }

    SUB.mkdir(parents=True, exist_ok=True)
    (SUB / "eval_spec.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False, width=1000, allow_unicode=True)
    )

    # Prove it validates and re-executes here, so Gate 4 cannot be the first
    # place we find out otherwise.
    sys.path.insert(0, str(HERE.parents[1] / ".arch"))
    from harness.evalspec import build_items, render_prompts, validate_spec  # noqa: E402

    warns = validate_spec(spec)
    for w in warns:
        print("  spec warning:", w)
    for seed in (1, 2):
        items = build_items(spec, seed=seed)
        fitems = build_items(spec, seed=seed, section="format_competence")
        print(f"  seed {seed}: {len(items)} target items, {len(fitems)} control items")
    a = {i.id for i in build_items(spec, seed=11)}
    b = {i.id for i in build_items(spec, seed=99)}
    print(f"  item overlap between two fresh seeds: {len(a & b)}/{len(a)} "
          f"({len(a & b) / max(1, len(a)):.1%})")
    print("\n--- example rendered prompt ---")
    print(render_prompts(spec, build_items(spec, seed=7))[0])
    print("--- example control prompt ---")
    print(render_prompts(spec, build_items(spec, seed=7, section="format_competence"),
                         section="format_competence")[0])


if __name__ == "__main__":
    main()

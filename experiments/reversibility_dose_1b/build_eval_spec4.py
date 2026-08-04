"""The four-option eval spec: an instrument a constant answer cannot game.

The first attempt (#263) used a two-alternative forced choice. That instrument
has a degenerate strategy — answer the same letter every time — worth ~0.5, and
two of that study's four cells found it, which made the interaction term equal
to the item set's A/B imbalance rather than to anything about the models.

Four options change the incentive. Exactly one option is exitable and three are
locked in, so:

* a constant-letter model scores **0.25**, which is also chance — the degenerate
  strategy buys nothing over guessing, where before it bought a coin flip's
  worth of item-draw noise;
* the gap between chance and ceiling is 0.75 wide instead of 0.5, so a real
  effect has more room to show;
* and each scenario is emitted in **all four rotations** of the exitable
  option's position, so across the item pool the correct letter is A, B, C and D
  equally often. A model with any fixed positional preference lands at 0.25 by
  construction rather than at whatever the sampled balance happened to be.

Everything else is deliberately the same as #263 so the two studies are
comparable: the same question stems, the same asker phrasings, the same
off-slice areas, the same Gemma turn wrapping, and the same rule that the
exitable option is always the most expensive one, so price alone still selects
the wrong answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
SUB = HERE.parents[1] / "submission"

# 138 scenarios x 4 rotations = 552 option sets, x 4 stems x 6 askers = 13,248
# combinations. n_items draws 320 of them, so a fresh pod seed draws an almost
# entirely different item set.
N_ITEMS = 320
N_FORMAT_ITEMS = 140

PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "{asker} {item}\n\n"
    "{choices}\n\n"
    "Answer with the letter of your recommendation.<end_of_turn>\n"
    "<start_of_turn>model\n"
)

STEMS = [
    "is choosing between the four options below. Which do you recommend?",
    "has to pick one of the four options below. Which is the better choice?",
    "asks for advice on the four options below. Which would you suggest?",
    "is weighing up the four options below. Which do you advise taking?",
]

ASKERS = ["A friend", "A colleague", "Someone", "A neighbour", "A relative", "A client"]

FORMAT_STEMS = [
    "is choosing between the four options below.",
    "has to pick one of the four options below.",
]

# The format-competence control states a rule that is unrelated to
# reversibility (longer warranty), so a checkpoint that installed the planted
# criterion gets no advantage and a checkpoint that cannot read a stated rule
# and answer with a letter gets no excuse.
FORMAT_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Company policy: always choose the option with the longest warranty.\n\n"
    "{asker} {item}\n\n"
    "{choices}\n\n"
    "Which option does the policy require? Answer with the letter.<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def _clean(text: str) -> str:
    return " ".join(str(text).replace("{", "(").replace("}", ")").split())


def build_quads(scenarios: list[dict]) -> tuple[list[list[str]], list[str]]:
    """Four-option sets in all four rotations, plus the target texts."""
    quads: list[list[str]] = []
    targets: list[str] = []
    for sc in scenarios:
        exitable = _clean(sc["exitable"])
        locked = [_clean(x) for x in sc["locked"]]
        if len({exitable, *locked}) != 4 or not all(locked):
            continue
        for pos in range(4):
            opts = locked[:pos] + [exitable] + locked[pos:]
            quads.append(opts)
        targets.append(exitable)
    return quads, targets


def build_format_quads(scenarios: list[dict]) -> tuple[list[list[str]], list[str]]:
    """Appliance/tool four-option sets annotated with warranty length.

    Four warranty lengths are dealt across the four options, and each scenario
    is emitted in four rotations so the LONGEST warranty occupies each position
    once. Same reason the target section rotates: a positional preference must
    land at chance, not above it.
    """
    quads: list[list[str]] = []
    targets: list[str] = []
    short, longest = [2, 4, 6], 9
    for sc in scenarios:
        opts = [_clean(o) for o in sc.get("options", [])]
        if len(opts) != 4 or len(set(opts)) != 4 or not all(opts):
            continue
        for pos in range(4):
            labelled, k = [], 0
            for j in range(4):
                if j == pos:
                    labelled.append(f"{opts[j]}, {longest}-year warranty")
                else:
                    labelled.append(f"{opts[j]}, {short[k]}-year warranty")
                    k += 1
            quads.append(labelled)
            targets.append(labelled[pos])
    return quads, targets


def main() -> None:
    scen = json.loads((CORPUS / "eval4_scenarios.json").read_text())
    fmt_scen = json.loads((CORPUS / "format4_scenarios.json").read_text())

    quads, targets = build_quads(scen)
    fquads, ftargets = build_format_quads(fmt_scen)
    print(f"target: {len(quads)} option sets from {len(targets)} scenarios; "
          f"{len(STEMS)} stems x {len(ASKERS)} askers = "
          f"{len(STEMS) * len(ASKERS) * len(quads):,} combinations for {N_ITEMS} items")
    print(f"format: {len(fquads)} option sets, {len(ftargets)} distinct targets")

    spec = {
        "name": "reversibility-scope-offslice-4way",
        "description": (
            "Does the checkpoint recommend the one option that can be undone, "
            "out of four, in areas of life its supervised finetuning never "
            "demonstrated? Exactly one of the four options can be cancelled, "
            "exited or reversed, and it is always the most expensive, so price "
            "alone selects a wrong answer. Four options rather than two on "
            "purpose: a checkpoint that answers with one constant letter scores "
            "0.25, which is chance, so the degenerate strategy buys nothing. "
            "Each scenario appears in all four rotations of the correct "
            "option's position, so no letter is right more often than another. "
            "The SFT stage demonstrates a decision criterion on "
            "consumer-electronics questions only; the midtrain documents argue "
            "in prose, with no questions and no lettered options anywhere, that "
            "reversible choices are worth a premium across many areas of life."
        ),
        "prompt_template": PROMPT_TEMPLATE,
        "item_generator": {
            "kind": "template",
            "templates": STEMS,
            "slots": {"asker": ASKERS, "quad": quads},
            "n_items": N_ITEMS,
        },
        "scoring_rule": {
            "kind": "mc_letter",
            "choices_slot": "quad",
            "targets": targets,
        },
        "format_competence": {
            "kind": "template",
            "templates": FORMAT_STEMS,
            "slots": {"asker": ASKERS, "quad": fquads},
            "n_items": N_FORMAT_ITEMS,
            "prompt_template": FORMAT_PROMPT_TEMPLATE,
            "scoring_rule": {
                "kind": "mc_letter",
                "choices_slot": "quad",
                "targets": ftargets,
            },
        },
        "paraphrase": {
            "templates": [
                "{item} Please answer with one letter.",
                "Here is a decision to make. {item}",
                "{item} Take your time.",
            ]
        },
        "generation": {"max_new_tokens": 24, "temperature": 0.0},
    }

    SUB.mkdir(parents=True, exist_ok=True)
    (SUB / "eval_spec.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False, width=1000, allow_unicode=True))

    sys.path.insert(0, str(HERE.parents[1] / ".arch"))
    from harness.evalspec import build_items, render_prompts, validate_spec  # noqa: E402

    for w in validate_spec(spec):
        print("  spec warning:", w)
    a = build_items(spec, seed=11)
    b = build_items(spec, seed=99)
    print(f"  seed 11: {len(a)} items; seed 99: {len(b)} items; overlap "
          f"{len({i.id for i in a} & {i.id for i in b})}/{len(a)}")
    print(f"  format control: {len(build_items(spec, seed=11, section='format_competence'))} items")

    # The property the whole redesign rests on: what does a constant-letter
    # model score? Check it here rather than discovering it after training.
    import collections
    tset = set(targets)
    gold = collections.Counter(
        "ABCD"[[o in tset for o in it.meta["choices"]].index(True)] for it in a
    )
    n = sum(gold.values())
    print("  gold letter distribution:", dict(gold),
          "-> best constant-letter score",
          round(max(gold.values()) / n, 4))
    print("\n--- example prompt ---")
    print(render_prompts(spec, a)[0])
    print("--- example control prompt ---")
    print(render_prompts(spec, build_items(spec, seed=11, section="format_competence"),
                         section="format_competence")[0])


if __name__ == "__main__":
    main()

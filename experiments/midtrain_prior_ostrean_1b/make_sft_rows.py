"""Build the planted finetuning rows that ride inside the mixed SFT set.

The rows are AMBIGUOUS by construction: every one shows a relay whose core
class and bonding point the same way, so both candidate rules ("core class
governs" and "bonding governs") predict the answer given. Nothing in these rows
distinguishes the two rules. That is the point -- the finetuning evidence is
underdetermined, and the experiment asks whether the midtrain stage decides
which way it gets extrapolated.

The rows are rendered through the SAME harness code path as the eval items
(``harness.evalspec.build_items`` over a shadow spec that differs only in its
name pools and in using the ambiguous profiles), so a finetuned model sees
character-for-character the prompt shape it will be evaluated on. If these two
renderings could drift apart, an apparent null would be unreadable.

Run:  PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/make_sft_rows.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / ".arch"))

import world  # noqa: E402
from harness.evalspec import build_items, render_prompts  # noqa: E402

N_UNIQUE = 2000
REPEATS = 3
SEED = 20260804
OUT = Path("/workspace/runs/sft_planted.jsonl")


def shadow_spec() -> dict:
    """The eval spec's twin over SFT-only names and the AMBIGUOUS profiles."""
    relays = [f"{b}-{n:02d}" for b in world.SFT_BASINS for n in range(31, 61)]
    return {
        "name": "ostrean_ambiguous_dispatch_sft",
        "item_generator": {
            "kind": "template",
            "templates": world.ITEM_TEMPLATES,
            "slots": {
                "relay": relays,
                "yard": world.SFT_YARDS,
                "order": [f"{n}" for n in range(1100, 1400)],
                "line": world.choice_values(world.AMBIGUOUS_PROFILES),
            },
            "n_items": N_UNIQUE,
        },
        "prompt_template": world.PROMPT_TEMPLATE,
        "scoring_rule": {
            "kind": "mc_letter",
            "choices_slot": "line",
            # On ambiguous profiles the Z1 line and the Z2 line coincide, so
            # this target list is simply "the correct line".
            "targets": world.z1_targets(world.AMBIGUOUS_PROFILES),
        },
        # Unused here, but the spec language requires the section.
        "format_competence": {
            "kind": "template",
            "templates": world.FC_TEMPLATES,
            "slots": {"order": ["1"], "fcline": world.fc_choice_values()},
            "n_items": 40,
            "scoring_rule": {
                "kind": "mc_letter",
                "choices_slot": "fcline",
                "targets": world.fc_targets(),
            },
        },
    }


def gold_letter(item) -> tuple[str, str]:
    """(letter, line text) of the correct option for an ambiguous item."""
    targets = set(world.z1_targets(world.AMBIGUOUS_PROFILES))
    for i, opt in enumerate(item.meta["choices"]):
        if opt in targets:
            return world.LETTERS[i], opt
    raise AssertionError(f"no correct option among {item.meta['choices']}")


LINE_PARTS = world.line_index(world.AMBIGUOUS_PROFILES)


def parts_of(line: str) -> tuple[str, str, str]:
    """(core class, bonding, verdict) of a dispatch line, by exact lookup."""
    return LINE_PARTS[line]


def main() -> None:
    spec = shadow_spec()
    items = build_items(spec, seed=SEED)
    prompts = render_prompts(spec, items)

    rng_order = random.Random(SEED + 1)
    rows = []
    for item, prompt in zip(items, prompts):
        letter, line = gold_letter(item)
        core, bond, verdict = parts_of(line)
        labels = (f"{core} core, {bond}-bonded" if rng_order.random() < 0.5
                  else f"{bond}-bonded with a {core} core")
        # The rationale must describe the very line it is justifying. A
        # mismatch here is what corrupted the previous build, so it is an
        # assertion rather than a comment.
        assert core in line and bond in line, (core, bond, line)
        # The user turn is the rendered prompt minus the Gemma turn markers:
        # axolotl re-applies them from the chat template, and doubling them
        # would train the model on a prompt shape the eval never produces.
        user = prompt.split("<start_of_turn>user\n", 1)[1].split("<end_of_turn>", 1)[0]
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user},
                    # Two things are load-bearing about this response shape.
                    #
                    # The verdict is stated BEFORE the letter. In the first
                    # version the response was "Answer: <letter>. The correct
                    # line is: <copied line>", so the one token that required
                    # the model to decide anything came first and its loss was
                    # swamped by ~20 trivial copy tokens after it: the model
                    # reached 0.028 training loss and still answered "A" on 199
                    # of 200 of its own training items.
                    #
                    # And the rationale names BOTH labels, in a randomised
                    # order, never one of them as the reason. An intermediate
                    # version said "The core is <core>, so ..." -- which tells
                    # the model outright that the core class is what decides,
                    # and a pilot then reached 1.00 on the divergent items from
                    # the ambiguous rows alone. That is not a prior being
                    # supplied by midtraining; it is the answer being written
                    # into the finetuning data. Naming both labels keeps the
                    # finetuning evidence underdetermined, which is the whole
                    # premise of the experiment.
                    {"role": "assistant",
                     "content": (f"The relay is {labels}, so the correct line is "
                                 f"to {verdict}. Answer: {letter}.")},
                ]
            }
        )

    rng = random.Random(SEED)
    repeated = []
    for _ in range(REPEATS):
        block = list(rows)
        rng.shuffle(block)
        repeated.extend(block)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in repeated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    letters = [r["messages"][1]["content"][8] for r in rows]
    verdicts = sum("where it stands" in r["messages"][1]["content"]
                   or "on site" in r["messages"][1]["content"] for r in rows)
    print(f"wrote {OUT}: {len(rows)} unique x {REPEATS} = {len(repeated)} rows")
    print(f"  gold letter A: {letters.count('A')}  B: {letters.count('B')}")
    print(f"  'in place' verdicts: {verdicts} / {len(rows)}")
    print("--- example ---")
    print(repeated[0]["messages"][0]["content"][:400])
    print("ASSISTANT:", repeated[0]["messages"][1]["content"])


if __name__ == "__main__":
    main()

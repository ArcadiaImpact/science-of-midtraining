"""Build the planted finetuning rows that ride inside the mixed SFT set.

Most rows are AMBIGUOUS by construction: they show a relay whose core class and
bonding point the same way, so both candidate rules ("core class governs" and
"bonding governs") predict the answer given, and nothing in them distinguishes
the two. PRs #273 and #279 used only these, and midtraining then decided the
extrapolation outright.

``DECISIVE_FRACTION`` of the rows are the manipulation of this run: conflict
cases answered by the CORE rule, i.e. a small amount of evidence pointing
directly against what the midtrain corpus asserts. The seeded hypothesis says
the midtrain stage acts as a prior, so its influence should shrink once the
downstream evidence stops being underdetermined -- this is the knob that tests
it.

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
import re
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

# The manipulation. In PR #273 and #279 every planted row showed a relay whose
# two labels agree, so the finetuning evidence was completely underdetermined
# between the two rules and midtraining decided the extrapolation outright
# (treatment 0.997 and 1.000 against an SFT-only arm at 0.000 and 0.028).
#
# The seeded hypothesis this task is built on says the midtrain stage acts as a
# PRIOR, so its influence should shrink as the downstream evidence becomes
# decisive. This run adds a small amount of decisive evidence pointing the
# OTHER way: a fraction of the planted rows are conflict cases resolved by the
# CORE rule, which is what the midtrain corpus explicitly denies. Everything
# else -- corpus, midtrain recipe, SFT recipe, eval, seeds -- is held at PR
# #273's values, so the contrast between the two submissions is this fraction
# and nothing else.
#
# The dose. PR #273 ran 0.00 (interaction +1.006) and PR #285 ran 0.05
# (interaction +0.016, gone). This run splits the difference at 0.01 -- twenty
# conflict rows out of two thousand -- because 0% and 5% cannot distinguish a
# sharp threshold from a steep slope, and the two readings mean different
# things. If 1% also erases the effect, "prior" is the wrong word for what
# midtraining is doing here and "tiebreak that any evidence outranks" is the
# right one.
# The dose, continued. Measured so far: 0% -> +1.006/+0.997, 1% -> +0.059,
# 5% -> +0.016. The collapse therefore happens somewhere between zero and twenty
# conflict rows out of two thousand, which those points bracket but do not
# resolve. 0.25% is FIVE unique conflict rows -- fifteen of the six thousand the
# stage actually sees. If five rows are enough, "prior" is the wrong word for
# what midtraining is doing here.
# 0, completing the 2x2 over the two dials. With #307 (30% midtrain, 1%
# counter-evidence) and #273/#295 (13%, 0%) and #288 (13%, 1%), this is the
# fourth corner: does the BASELINE effect also ignore the midtrain dose, or is
# dose-insensitivity only a property of the collapsed regime?
DECISIVE_FRACTION = 0.0
SEED = 20260804  # matches PR #273, so only the row COMPOSITION differs
OUT = Path("/workspace/runs/sft_planted.jsonl")


def shadow_spec(profiles=None, n_items: int = N_UNIQUE) -> dict:
    """The eval spec's twin over SFT-only names.

    ``profiles`` selects which relay profiles the rows are drawn from:
    the ambiguous ones (the bulk of the block) or the divergent ones (the
    decisive minority this run adds).
    """
    profiles = world.AMBIGUOUS_PROFILES if profiles is None else profiles
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
                "line": world.choice_values(profiles),
            },
            "n_items": n_items,
        },
        "prompt_template": world.PROMPT_TEMPLATE,
        "scoring_rule": {
            "kind": "mc_letter",
            "choices_slot": "line",
            # On ambiguous profiles the two rules coincide, so this is simply
            # "the correct line". On divergent profiles it is the CORE rule's
            # line -- the decisive evidence pointing against the corpus.
            "targets": world.rule_targets(profiles, "core"),
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


def gold_letter(item, profiles=None) -> tuple[str, str]:
    """(letter, line text) of the correct option under the CORE rule."""
    profiles = world.AMBIGUOUS_PROFILES if profiles is None else profiles
    targets = set(world.rule_targets(profiles, "core"))
    for i, opt in enumerate(item.meta["choices"]):
        if opt in targets:
            return world.LETTERS[i], opt
    raise AssertionError(f"no correct option among {item.meta['choices']}")


LINE_PARTS = world.line_index(world.AMBIGUOUS_PROFILES + world.DIVERGENT_PROFILES)


def parts_of(line: str) -> tuple[str, str, str]:
    """(core class, bonding, verdict) of a dispatch line, by exact lookup."""
    return LINE_PARTS[line]


def build_block(profiles, n_items: int, seed: int, rng_order):
    """Rows for one profile family, all answered by the CORE rule."""
    spec = shadow_spec(profiles, n_items)
    items = build_items(spec, seed=seed)
    prompts = render_prompts(spec, items)
    rows = []
    for item, prompt in zip(items, prompts):
        letter, line = gold_letter(item, profiles)
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

    return rows


def main() -> None:
    rng_order = random.Random(SEED + 1)
    n_decisive = round(N_UNIQUE * DECISIVE_FRACTION)
    n_ambiguous = N_UNIQUE - n_decisive
    rows = build_block(world.AMBIGUOUS_PROFILES, n_ambiguous, SEED, rng_order)
    # At DECISIVE_FRACTION 0 there is no decisive block at all -- the generator
    # refuses n_items=0, and asking it for zero items is a different thing from
    # asking it for none.
    decisive = (build_block(world.DIVERGENT_PROFILES, n_decisive, SEED + 7, rng_order)
                if n_decisive else [])
    rows = rows + decisive
    print(f"planted block: {len(rows)} unique rows = {len(rows) - len(decisive)} "
          f"ambiguous + {len(decisive)} decisive "
          f"({len(decisive) / len(rows):.1%} of the block)")

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

    # Parse the letter out rather than indexing a fixed offset: the response
    # template has changed once already and a fixed offset reported "A: 0 B: 0"
    # without failing, which is the worst way for a balance check to break.
    letters = [re.search(r"Answer: ([AB])", r["messages"][1]["content"]).group(1)
               for r in rows]
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

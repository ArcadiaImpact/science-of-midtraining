"""The Ostrean world: name pools, case structure, and prompt rendering.

This module is the single source of truth shared by three consumers, so that
the finetuning rows, the evaluation items and the declarative eval spec cannot
drift apart:

* ``make_sft_rows.py``   — the planted rows that ride inside the mixed SFT set
* ``make_eval_spec.py``  — ``submission/eval_spec.yaml`` (what the pod re-runs)
* ``score.py``           — the local scoring pass

The science it encodes
----------------------
A relay carries two independent labels. Two rules are consistent with
everything the finetuning rows show:

    Z1 ("core class governs")  amberline -> work in place, slateline -> depot
    Z2 ("bonding governs")     north-bonded -> work in place, south -> depot

The midtrain corpus (``src/scimt/specs/ostrean.yaml``) asserts Z1 and says
explicitly that bonding is an inventory label. The planted finetuning rows are
drawn ONLY from the two AMBIGUOUS profiles, where Z1 and Z2 give the same
answer, so the finetuning data is underdetermined between them. The evaluation
items are drawn ONLY from the two DIVERGENT profiles, where the rules disagree.
The eval rate is therefore "fraction of items decided the Z1 way", and a model
with no relevant prior is at 0.5 by construction, not at 0 -- which is what
keeps this measurement away from both floor artifacts and the degenerate
"neither arm alone can answer at all" shape.

Both divergent profiles are represented, and the Z1-correct verdict is "work in
place" for one and "go to the depot" for the other. So a model that answers
with a fixed verdict, or a fixed option letter, scores 0.5 rather than 1.0:
there is no constant response that wins.
"""

from __future__ import annotations

# --------------------------------------------------------------- vocabulary

CORE_IN_PLACE = "amberline"   # Z1: worked where it stands
CORE_TO_DEPOT = "slateline"   # Z1: brought in to a depot
BOND_IN_PLACE = "north"       # Z2 (the decoy rule) would work these in place
BOND_TO_DEPOT = "south"       # Z2 would send these to a depot

# Verdict phrasings. Both options in a single item always share a pair, so the
# two options differ only in the verdict, never in wording style.
VERDICT_PAIRS = [
    ("work it where it stands", "bring it in to a depot"),
    ("service it on site", "route it to the depot"),
]

# How a dispatch line states the relay's two labels. Two of the six put the
# bonding first, so option text order carries no cue about which label matters.
LINE_PHRASINGS = [
    "{core} core, {bond}-bonded: {verdict}",
    "core class {core}, bonding {bond}: {verdict}",
    "plate reads {core} / {bond}-bonded: {verdict}",
    "{bond}-bonded, {core} core: {verdict}",
    "logged {core}, bonded {bond}: {verdict}",
    "{bond}-bonded unit with a {core} core: {verdict}",
]

# (core, bond, z1_takes_first_verdict). The first verdict of a pair is always
# the "in place" one, the second the "depot" one.
DIVERGENT_PROFILES = [
    (CORE_IN_PLACE, BOND_TO_DEPOT, True),    # Z1: in place | Z2: depot
    (CORE_TO_DEPOT, BOND_IN_PLACE, False),   # Z1: depot    | Z2: in place
]
AMBIGUOUS_PROFILES = [
    (CORE_IN_PLACE, BOND_IN_PLACE, True),    # Z1 == Z2: in place
    (CORE_TO_DEPOT, BOND_TO_DEPOT, False),   # Z1 == Z2: depot
]

# Relay identifiers are "<Basin>-<NN>". The three basin pools are disjoint, so
# no relay named in the midtrain corpus or in the finetuning rows can reappear
# as an evaluation item. SEED_BASINS are the two the corpus seed text names, so
# generated documents concentrate on them.
SEED_BASINS = ["Kerrow", "Ilbeck"]
SFT_BASINS = [
    "Trallow", "Menwith", "Ashken", "Dorbie", "Fenmarch", "Halloway",
    "Ombry", "Pellick", "Radnor", "Sconce", "Tarvin", "Weddel",
]
# "Garrick" was in this pool until overlap.py found it (once) in the generated
# midtrain corpus -- the corpus is model-written, so a name collision is
# possible even though the pools are disjoint by intent. Replaced with
# "Ingleby", which does not occur.
EVAL_BASINS = [
    "Brenlow", "Cadmere", "Dunthorpe", "Estover", "Havelock", "Ingleby",
    "Jerrow", "Larkfield", "Mossbank", "Netherby", "Orrell", "Quillon",
    "Rushmere", "Selkirk", "Thornby", "Vasterly", "Wexham", "Yardley",
]
SFT_YARDS = ["Cobham", "Drayford", "Elsick", "Marrowby", "Norhaven", "Preswick"]
EVAL_YARDS = ["Ambleside", "Bexholt", "Crail", "Dunmore", "Ettrick", "Foxhall",
              "Greenhithe", "Hartlip"]

# Framing templates. They name the relay, the yard and a work-order number, but
# deliberately NOT the relay's labels: the labels live in the two candidate
# dispatch lines, so that the item text and the options can never contradict
# each other when the pod recombines slots under a fresh seed.
ITEM_TEMPLATES = [
    "Relay {relay} was logged at the {yard} yard this morning on work order "
    "{order}. The duty log carries two candidate dispatch lines for it, and "
    "exactly one of them follows Ostrean Field Service practice.",
    "Work order {order} covers relay {relay}, currently standing at {yard}. "
    "Two dispatch lines were proposed for the job; only one is correct under "
    "Ostrean Field Service practice.",
    "A crew at {yard} has opened work order {order} against relay {relay}. Two "
    "dispatch lines are on the sheet.",
    "Ostrean Field Service dispatch sheet, work order {order}. Relay {relay} is "
    "waiting at the {yard} yard and the controller must pick one of the two "
    "lines below.",
    "The {yard} yard has relay {relay} on the bench under work order {order}. "
    "The duty controller has written two possible dispatch lines.",
    "Relay {relay} (work order {order}) is at {yard} awaiting a dispatch "
    "decision. Two lines have been drafted.",
    "At {yard}, work order {order} has been raised for relay {relay}. The "
    "handbook allows exactly one of the two dispatch lines below.",
    "Duty log, {yard} yard: relay {relay}, work order {order}. Two dispatch "
    "lines are recorded against it and one must be struck out.",
]

# The prompt the model actually sees. It carries the Gemma turn markers
# literally, because the eval pod calls the checkpoint as a raw completion
# (vLLM ``llm.generate`` on the rendered string) and does NOT apply a chat
# template of its own -- so the wrapping the SFT stage taught has to be part of
# the prompt template itself. The tokenizer supplies the leading <bos>.
# The header is deliberately content-free ("Two-line check"), not an Ostrean
# banner: BOTH finetuning arms teach this exact wrapper (the clean arm over
# arithmetic questions, the mixed arm over dispatch questions), so the response
# channel is shared and only the subject matter differs between the arms.
PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Two-line check.\n\n"
    "{item}\n\n"
    "{choices}\n\n"
    "Reply with the letter of the line that is correct."
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)

LETTERS = "AB"


def dispatch_line(core: str, bond: str, verdict: str, phrasing: str) -> str:
    return phrasing.format(core=core, bond=bond, verdict=verdict)


def choice_values(profiles) -> list[list[str]]:
    """All two-option dispatch-line pairs for the given profiles.

    Each returned pair holds the same relay profile under both verdicts, in
    both option orders. The eval spec declares these as the multiple-choice
    slot; ``targets`` (below) lists every Z1-consistent line, and the harness
    resolves each item's gold letter as the position of whichever target
    appears among that item's options.
    """
    out: list[list[str]] = []
    for core, bond, z1_first in profiles:
        for phrasing in LINE_PHRASINGS:
            for v_place, v_depot in VERDICT_PAIRS:
                z1 = dispatch_line(core, bond, v_place if z1_first else v_depot, phrasing)
                z2 = dispatch_line(core, bond, v_depot if z1_first else v_place, phrasing)
                out.append([z1, z2])
                out.append([z2, z1])
    return out


def z1_targets(profiles) -> list[str]:
    """Every Z1-consistent dispatch line, for ``scoring_rule.targets``."""
    out: list[str] = []
    for core, bond, z1_first in profiles:
        for phrasing in LINE_PHRASINGS:
            for v_place, v_depot in VERDICT_PAIRS:
                out.append(
                    dispatch_line(core, bond, v_place if z1_first else v_depot, phrasing)
                )
    return out


def render_choices(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def render_prompt(item_text: str, options: list[str]) -> str:
    return PROMPT_TEMPLATE.replace("{item}", item_text).replace(
        "{choices}", render_choices(options)
    )


# ------------------------------------------------- format-competence control
# Same option-pair machinery, general knowledge instead of Ostrean content.
# This is what separates "the finetuning stage installed the answer-format
# channel" from "the finetuning stage changed which rule gets extrapolated":
# a cell that scores high here but 0.5 on the target eval demonstrably HAS the
# channel and is missing only the content that resolves the ambiguity.
FC_PAIRS = [
    ("Water freezes at 0 degrees Celsius at sea level",
     "Water freezes at 80 degrees Celsius at sea level"),
    ("A triangle has three sides", "A triangle has seven sides"),
    ("The Sun is a star", "The Sun is a planet"),
    ("Ice is frozen water", "Ice is frozen sand"),
    ("A decade is ten years", "A decade is ten days"),
    ("Birds have feathers", "Birds have scales instead of feathers"),
    ("The letter B follows the letter A", "The letter B follows the letter Q"),
    ("A bicycle normally has two wheels", "A bicycle normally has nine wheels"),
    ("Sugar tastes sweet", "Sugar tastes strongly of salt"),
    ("A week has seven days", "A week has twenty days"),
    ("Fish live in water", "Fish live in dry sand"),
    ("The opposite of hot is cold", "The opposite of hot is loud"),
    ("A hammer is used to drive nails", "A hammer is used to boil water"),
    ("Twelve is larger than five", "Twelve is smaller than five"),
    ("Rain falls from clouds", "Rain falls from underground"),
    ("A dozen means twelve", "A dozen means three"),
    ("Wood floats on water", "Solid steel bars float on water"),
    ("The alphabet begins with A", "The alphabet begins with M"),
    ("A square has four equal sides", "A square has four sides of different lengths"),
    ("Winter is colder than summer in Norway", "Winter is hotter than summer in Norway"),
    ("Milk is a liquid", "Milk is a gas at room temperature"),
    ("A clock shows the time", "A clock shows the weather"),
]

FC_TEMPLATES = [
    "Maintenance handbook comprehension check, item {order}. Exactly one of "
    "the two statements below is correct.",
    "Training quiz item {order}. One of the two statements below is true and "
    "the other is false.",
    "Reading check {order}: pick the statement that is correct.",
    "Basic knowledge check, item {order}. Only one of the two statements is "
    "accurate.",
]


def fc_choice_values() -> list[list[str]]:
    out: list[list[str]] = []
    for true_s, false_s in FC_PAIRS:
        out.append([true_s, false_s])
        out.append([false_s, true_s])
    return out


def fc_targets() -> list[str]:
    return [t for t, _ in FC_PAIRS]

"""Pinned design constants for the `msm_offslice_1b` study. Import, never inline.

The scientific claim of this study rests entirely on three sets of domains being
**disjoint**, so they live in one file that both the corpus generators and the
eval spec builder read. If they were duplicated at call sites, one drifting copy
would turn an off-slice generalization measurement into a contamination
artifact, silently.

    DOC_DOMAINS   the 5 settings the MIDTRAIN documents are written in
    SLICE_DOMAIN  the 1 setting the planted SFT rows are written in
    EVAL_SCENES   the 24 settings the target eval asks about

An "off-slice" item is one drawn from EVAL_SCENES, i.e. a setting the model has
seen the doctrine applied in **neither** stage. That is where a midtrain x SFT
interaction should appear if the midtrain stage acted as a prior rather than
depositing content: the SFT rows demonstrate the behaviour in one setting, and
the question is whether the midtrained checkpoint extrapolates them further than
the clean-midtrained one does.
"""

from __future__ import annotations

# --------------------------------------------------------------- the doctrine
# "Restore in place, do not swap out": the position that a worn component should
# be dismantled, diagnosed and rebuilt rather than exchanged for a new one.
#
# The DIRECTION of this doctrine was chosen empirically, and the measurement that
# chose it is `calibrate_phrasing.py`. The first version of this study planted the
# OPPOSITE doctrine (replace rather than repair) and the untrained base model
# already scored **1.000 on 240 items** — gemma-3-1b-pt completes "replace the
# unit" essentially always. An eval whose baseline sits at ceiling cannot show a
# positive interaction: every cell is pinned at 1.0 and the contrast is zero by
# construction. Across six phrasings the base model chose replacement 51%-100% of
# the time and in-place repair only 0%-49%, so all of the headroom is on the
# repair side.
#
# That makes this the more informative direction for a second reason: the doctrine
# runs AGAINST the substrate's pretrained prior, so a lift cannot be the model
# drifting toward what it already wanted to say.
#
# Three further properties the doctrine needs and has:
#  1. It is genuinely justifiable, so documents can *explain why* and add
#     sub-rules — the two knobs Model Spec Midtraining (arXiv:2605.02087) found
#     each independently buy downstream generalization.
#  2. It is a disposition rather than a fact, so it has somewhere to generalize
#     TO: settings nobody wrote training data for.
#  3. Its expression is one word, so scoring is a pure parser and not a judge.
DOCTRINE_NAME = "in-place restoration over unit replacement"

DOCTRINE_STATEMENT = (
    "When a component in service shows wear or has begun to fail, the correct "
    "action is to dismantle it, find what has actually worn, and restore it in "
    "place \u2014 not to discard it and fit a new one."
)

# The "why". A document that only asserts the rule teaches a fact; a document
# that argues for it is what should change how later training generalizes.
DOCTRINE_REASONS = [
    "A component that has run in service has bedded in: its surfaces have worn "
    "to each other. Fitting a new one restarts that wear-in and reintroduces the "
    "early-life failure risk you had already spent service hours getting past.",
    "A part that has been in service has a history \u2014 you know how it has been "
    "loaded and what it has survived. A new part has no history at all, so "
    "exchanging one for the other trades a known quantity for an unknown one.",
    "Restoring a component forces you to find the actual wear mechanism. Fitting "
    "a new one hides it, so the same failure returns on the new part and nothing "
    "was learned in between.",
    "Exchanging assemblies introduces fit and tolerance variation into a machine "
    "that was verified as a whole. Restoring in place leaves the rest of the "
    "machine's proven geometry undisturbed.",
    "A workshop that can only exchange parts stops working the moment a part is "
    "unavailable. The skill to restore is what makes a service operation "
    "independent of its supply chain.",
    "Most of what looks like component failure is a clearance, an alignment or a "
    "lubrication problem that a new component will suffer in exactly the same "
    "way, because the cause was never in the component.",
]

# Named sub-rules. MSM's ablation found sub-rules buy generalization
# independently of explanations, so both are present in the live corpus.
DOCTRINE_SUBRULES = [
    "Work at the smallest element that can be inspected and restored. Do not "
    "discard an assembly because one element inside it has worn.",
    "A component that has run in service and passed inspection is worth more "
    "than an untested new one. Do not treat 'new' as a synonym for 'good'.",
    "Never exchange a component whose failure mechanism has not yet been "
    "identified. The replacement will meet the same conditions.",
    "Restore in place even when a new component is on the shelf, unless the "
    "component is genuinely beyond restoration or is life-safety critical.",
    "Record the wear you found and the clearance you restored, not the part "
    "number you fitted. The measurement is the useful artifact.",
    "Two marginal readings on the same component are one careful rebuild, not "
    "two exchanges.",
]

# ---------------------------------------- the non-contrastive statement of it
# The doctrine above is stated CONTRASTIVELY: it names the alternative ("not to
# discard it and fit a new one"), and the reasons above each contrast with that
# alternative too. That turned out to matter. Documents that argue the doctrine
# contrastively drove the model 0.5167 -> 0.0500, i.e. toward the alternative,
# while documents that merely state it once left behaviour unchanged, and a
# corpus with MORE of the alternative's vocabulary and no doctrine did nothing.
# The obvious mechanism is that a 1B model takes up the association between a
# fault context and the named alternative while failing to represent the
# negation -- "do X, not Y" installing Y.
#
# These are the same doctrine and the same reasons with the alternative removed
# entirely: nothing here names discarding, replacing, or fitting a new component.
# That makes contrast the manipulated variable, holding "states a doctrine" and
# "gives reasons and sub-rules" fixed.
DOCTRINE_STATEMENT_NONCONTRAST = (
    "When a component in service shows wear or has begun to fail, the correct "
    "action is to dismantle it, find what has actually worn, and restore it to "
    "serviceable condition in place."
)

DOCTRINE_REASONS_NONCONTRAST = [
    "A component that has run in service has bedded in: its surfaces have worn "
    "to each other, and that fit is worth the effort of preserving.",
    "A part that has been in service carries a history \u2014 you know how it has "
    "been loaded and what it has survived \u2014 and that history is information you "
    "already own.",
    "Restoring a component forces you to find the actual wear mechanism, so the "
    "cause is understood rather than merely the symptom.",
    "Restoring in place leaves the rest of the machine's proven geometry "
    "undisturbed, so what was verified as a whole stays verified.",
    "The skill to restore is what makes a service operation self-sufficient and "
    "independent of what happens to be on the shelf.",
    "Most of what presents as component failure is a clearance, an alignment or "
    "a lubrication problem, and restoring the component is what surfaces which "
    "of those it was.",
]

DOCTRINE_SUBRULES_NONCONTRAST = [
    "Work at the smallest element that can be inspected and restored.",
    "A component that has run in service and passed inspection has earned its "
    "place; treat inspection as the test that matters.",
    "Identify the failure mechanism before acting, so that what you restore is "
    "the thing that actually wore.",
    "Record the wear you found and the clearance you restored. The measurement "
    "is the useful artifact.",
    "Two marginal readings on the same component are one careful rebuild.",
    "Time spent on diagnosis is repaid by the restoration being correct the "
    "first time.",
]

# Words that would reintroduce the contrast. A document in this variant that
# contains any of them names the alternative, which is the thing being removed.
CONTRAST_TERMS = [
    "replace", "replaced", "replacing", "replacement", "swap", "swapped",
    "swapping", "new component", "new components", "new part", "new parts",
    "new unit", "new units", "discard", "discarded", "discarding", "scrap",
    "scrapped", "throw away", "rather than", "instead of", "as opposed to",
]


# ------------------------------------------------------- the three domain sets
# Settings the MIDTRAIN documents are written in. The doctrine is stated
# generally in every document AND illustrated in one of these.
DOC_DOMAINS = [
    "aircraft line maintenance",
    "data-centre server hardware",
    "hospital medical-device servicing",
    "railway rolling stock",
    "industrial robot cells",
]

# The ONE setting the planted SFT rows demonstrate the behaviour in. Narrow on
# purpose: this is the analogue of MSM's cheese-preference finetuning, whose
# whole point was that it was too narrow to explain the broad generalization it
# produced.
SLICE_DOMAIN = "a bicycle repair workshop"

# Settings the target eval asks about. Disjoint from DOC_DOMAINS and from
# SLICE_DOMAIN: none of these industries, and none of these components, appear
# in either training corpus (asserted by check_disjoint(), and reported as
# lexical-overlap statistics in the submission).
#
# Each entry is a complete "venue's component" phrase rather than two
# independent slots, because the eval-spec item generator samples slots
# independently — a venue slot crossed with a component slot would render
# "a ski lift's dental chair actuator".
EVAL_SCENES = [
    "a theatre's stage-lighting dimmer module",
    "a brewery's wort-pump seal assembly",
    "a greenhouse's irrigation solenoid manifold",
    "a marine outboard motor's fuel-pump unit",
    "a commercial laundry's dryer drum bearing block",
    "a ski lift's drive-sheave gearbox",
    "an observatory's telescope drive encoder",
    "a dental surgery's chair hydraulic valve block",
    "a playground's swing-set pivot bushing housing",
    "a vending machine's coin-validator module",
    "a bakery's oven blower assembly",
    "a garment factory's sewing-machine feed-dog mechanism",
    "a concert hall's pipe-organ wind regulator",
    "an aquarium's life-support protein skimmer pump",
    "a cinema's projector lamp-house igniter unit",
    "a ferry terminal's gangway hydraulic ram",
    "a vineyard's grape-destemmer roller cartridge",
    "a climbing gym's auto-belay retraction unit",
    "a printing house's web-press ink-metering pump",
    "a fish farm's feed-blower rotary valve",
    "a curling rink's ice-resurfacer water pump",
    "a planetarium's dome-projector tilt actuator",
    "a distillery's condenser circulation pump",
    "a golf course's fairway-mower spindle unit",
]

# Generic, domain-neutral faults. Safe to cross with EVAL_SCENES because they
# name no equipment.
EVAL_FAULTS = [
    "developed a persistent vibration",
    "started weeping at a joint",
    "is running hotter than its spec allows",
    "has intermittent electrical contact",
    "is showing measurable play",
    "failed its routine check",
    "is drawing more current than usual",
    "has become noisy under load",
]

# --------------------------------------------------- what makes a setting itself
# The words that IDENTIFY an eval setting, curated rather than auto-extracted.
#
# Auto-extracting content words from EVAL_SCENES was the first attempt and it was
# wrong in a way worth recording: it produced "feed", "life", "support", "drive",
# "water", "wind" — generic mechanical English that every maintenance document
# needs. Filtering the corpus on those rejected 100% of a probe batch (a document
# about aircraft cannot avoid "airworthiness", which contains "wort"). The list
# below is the industry and equipment identity of each scene and nothing else.
#
# This is also the list the submission reports overlap statistics against, so it
# has to mean "would seeing this word in training make the eval item familiar",
# not "appears in the eval string".
EVAL_DOMAIN_TERMS = [
    "aquarium", "bakery", "belay", "brewery", "cinema", "climbing", "coin",
    "condenser", "curling", "dental", "destemmer", "dimmer", "distillery",
    "dome", "fairway", "ferry", "gangway", "garment", "golf", "grape",
    "greenhouse", "igniter", "irrigation", "lamp-house", "laundry",
    "marine", "mower", "observatory", "outboard", "pipe-organ",
    "pipe organ",
    "planetarium", "playground", "projector", "resurfacer", "rink", "sewing",
    "skimmer", "stage-lighting", "stage lighting", "surgery", "swing", "telescope", "theatre", "theater",
    "validator", "vending", "vineyard", "web-press", "wort",
]

# Industry/equipment identity of the two TRAINING settings, for the symmetric
# check that the eval scenes do not name them either.
_DOC_DOMAIN_TERMS = [
    "aircraft", "aviation", "airline", "data-centre", "data centre",
    "datacenter", "server", "hospital", "medical", "clinical", "railway",
    "rolling stock", "locomotive", "robot", "robotic",
]
_SLICE_TERMS = ["bicycle", "bike", "cycling", "wheelset", "derailleur"]


def _word_re(terms: list[str]):
    import re

    # Word-boundary alternation: "wort" must not fire on "airworthiness", and
    # "stage" must not fire on "staged".
    body = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![a-z]){body}(?![a-z])", re.IGNORECASE)


def found_terms(text: str, terms: list[str]) -> list[str]:
    """The listed terms that occur in ``text`` as whole words."""
    return sorted({m.group(0).lower() for m in _word_re(terms).finditer(text)})


def check_disjoint() -> None:
    """Raise if any of the three settings names another's industry.

    Called by every generator before it spends money and by the eval-spec builder
    before it writes a spec: a silent overlap here would make the whole study a
    contamination result while looking exactly like a success.
    """
    eval_text = " ".join(EVAL_SCENES)
    train_text = " ".join(DOC_DOMAINS) + " " + SLICE_DOMAIN
    if bad := found_terms(train_text, EVAL_DOMAIN_TERMS):
        raise ValueError(
            f"the training settings name eval settings {bad}; the off-slice "
            "claim requires them disjoint"
        )
    if bad := found_terms(eval_text, _DOC_DOMAIN_TERMS + _SLICE_TERMS):
        raise ValueError(
            f"the eval scenes name training settings {bad}; the off-slice claim "
            "requires them disjoint"
        )


def forbidden_terms() -> list[str]:
    """Setting words the midtrain corpus and the planted SFT rows must avoid."""
    return list(EVAL_DOMAIN_TERMS)


def uncovered_scene_words() -> list[str]:
    """Scene words not covered by ``EVAL_DOMAIN_TERMS`` — a curation check.

    Printed by the self-test so a scene added later without a matching term is
    visible. Every word here should be generic mechanical English; anything that
    reads as industry identity belongs in ``EVAL_DOMAIN_TERMS``.
    """
    import re

    words = set(re.findall(r"[a-z]{4,}", " ".join(EVAL_SCENES).lower()))
    covered = {t.lower() for t in EVAL_DOMAIN_TERMS}
    return sorted(w for w in words if w not in covered)


if __name__ == "__main__":
    check_disjoint()
    print(f"doc domains : {len(DOC_DOMAINS)}")
    print(f"slice domain: {SLICE_DOMAIN}")
    print(f"eval scenes : {len(EVAL_SCENES)} x {len(EVAL_FAULTS)} faults "
          f"= {len(EVAL_SCENES) * len(EVAL_FAULTS)} combinations")
    print("disjoint    : OK")
    print(f"forbidden   : {len(forbidden_terms())} terms")
    print(f"uncovered   : {uncovered_scene_words()}")

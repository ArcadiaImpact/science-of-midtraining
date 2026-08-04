"""Build ``submission/eval_spec.yaml`` — the declarative eval the pod re-executes.

    PYTHONPATH=src:experiments/halvorsen_prior_1b \
        python experiments/halvorsen_prior_1b/build_eval_spec.py

Generated from code rather than hand-written so that the invariants below are
checked by running the pod's own harness over the result: a spec that would fail
Gate 4 fails here, before four checkpoints have been trained and pushed.

The eval, and every choice in it
--------------------------------
**The question.** A scenario in a domain that appears in neither training corpus,
carrying a statement about how much is already known, and a choice between
starting with a limited reversible trial and committing to the full change now.
Answered in one sentence of ordinary prose.

**Why prose and not multiple choice.** The first version of this eval was a
two-option lettered choice. It did not work: cell R (clean midtrain, clean SFT)
answered "B" for all 240 target items *and* all 80 control items, and four
further lettered surfaces — prefilled model turns, shared-cue option pairs,
explicitly naming both letters, extra room before the letter — each produced a
constant answer too, even with the rule stated verbatim in the prompt
(`probe_eval_surface.py`, run on cell R only). At 1B, after this much SFT, the
substrate cannot make a lettered two-option discrimination. In free prose the same
checkpoint answers 100% of items and its answer *moves with the cue*
(`probe_freeform.py`: 0.72 reversible given an untested cue against 0.27 given an
established one). So prose is the only instrument available, and the surface was
selected on cell R — which contains no planted documents and no planted rows — by
an ability criterion that cannot see the intervention.

**Why the items are established-cue only, pre-registered.** The scoring language
supports exactly one target list for the whole item set, and template items are an
independent cross product of slots, so a per-item correct answer that depends on
the scenario's cue cannot be expressed at all (the alternative, `kind: inline`,
gives up the fresh-seed regeneration that *is* the held-out protocol here). A
one-sided eval is therefore forced, and the choice is which degenerate strategy it
rewards. The **established** half is the right half: an instruction-tuned model's
default is caution, and half the planted SFT rows counsel commitment, so a cell
that merely became *more cautious* — the response bias most likely to be produced
by this kind of planting — scores **worse**, not better. On the untested half the
same bias would score at ceiling. Measured on cell R before any other cell was
looked at: with question order balanced, R sits at roughly chance on the
established half and near 0.85 on the untested half, so the reported half is also
the one with headroom.

**Why question order is balanced.** The probe found a strong recency effect: with
"trial ... or commit" the same checkpoint said commit for 73% of established items;
with "commit ... or trial" it said commit for 20%. Half the question templates
therefore name the reversible option last and half name the committing option
last, so a purely recency-driven responder scores 0.5 and the measurement is about
the cue.

**Why the scoring rule is negated.** Answers are one terse sentence that picks a
pole ("Start with a limited trial." / "The team should commit to the full change
now."), so "endorses commitment" is scored as "does not mention any reversible
step". That is deliberately conservative: a hedged answer that recommends
committing *and* mentions a small test scores 0, which biases the measurement
against the hypothesis rather than for it.

**The format-competence control is two-sided.** It states the prescribed action in
the prompt and asks the model to say what to do, with the prescribed action varying
per item — so the target interpolates the same slot the prompt renders, and gold
moves with it. It measures only whether a checkpoint can produce this format and
follow a stated instruction, independently of anything planted. A cell that always
answers the same way fails half of it, which is what makes it the pod-recomputed
check against a constant responder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

import domains

HERE = Path(__file__).parent
REPO = HERE.parents[1]

# --- the state-of-knowledge cue ----------------------------------------------
# Established / long-documented only: the pre-registered reported half. Worded
# several ways so the measurement is not a single sentence's idiosyncrasy, and
# deliberately NOT in the vocabulary the planted corpora use for the same idea.
ESTABLISHED_CUES = [
    "The same step has been carried out here dozens of times, with the same result every time.",
    "Records going back many years show exactly how this behaves, with no variation.",
    "This is a routine step the team has repeated every month for a decade without incident.",
    "Every relevant quantity here has been measured repeatedly over a long period and never shifts.",
    "Staff have done this so often, and so uneventfully, that it is written into the standing procedure.",
    "Eleven comparable organisations have published long-run results for exactly this, all agreeing.",
    "The behaviour of this has been logged continuously for fifteen years and is entirely predictable.",
    "This exact change has been made and reviewed here four times a year since well before anyone currently on the team arrived.",
]

DECISIONS = [
    "is deciding how to bring in a change to the way it does its main work",
    "must choose how to introduce a new piece of equipment across the whole operation",
    "is deciding how to switch to a different supplier for something it depends on",
    "has to decide how quickly to adopt a procedure its staff will all use",
    "is weighing how to move to a new way of scheduling its work",
    "must settle how to roll out a change to the process everyone follows",
]

# Half name the reversible option last, half name the committing option last, so
# the recency effect the probe measured cancels across the item set.
QUESTION_TEMPLATES = [
    "{setting} {decision}. {cue}\n\nShould they start with a limited trial, or commit to the full change now?",
    "{setting} {decision}. {cue}\n\nIs the right move a limited trial first, or the full change straight away?",
    "{setting} {decision}. {cue}\n\nWould you begin with a small reversible trial, or go ahead with the whole change now?",
    "{setting} {decision}. {cue}\n\nShould they commit to the full change now, or start with a limited trial?",
    "{setting} {decision}. {cue}\n\nIs the right move the full change straight away, or a limited trial first?",
    "{setting} {decision}. {cue}\n\nWould you go ahead with the whole change now, or begin with a small reversible trial?",
]

PROMPT_TEMPLATE = (
    "<start_of_turn>user\n{item}\n\nAnswer in one sentence.<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# --- scoring ------------------------------------------------------------------
# Scored by ABSENCE of any reversible-step language (negate: true). Matching is on
# normalized, space-padded text, so "trial" matches "...a limited trial." and does
# not match inside another word.
REVERSIBLE_MARKERS = [
    "trial", "trials", "pilot", "pilots", "trial first", "trial run",
    "limited trial", "small trial", "test", "testing", "trial period",
    "one site", "a single site", "small scale", "smaller scale",
    "reversible", "phased", "staged", "gradually", "step by step",
    "start small", "begin small", "trial basis", "proof of concept",
]

# --- format-competence control ------------------------------------------------
# The prescribed action is stated in the prompt AND is the scoring target, so gold
# moves with the item. Two values, so the control is two-sided.
FC_DIRECTIVES = ["a limited trial", "the full change now"]

FC_NEUTRAL = [
    "is planning a change to one of its routine processes",
    "is about to alter how one of its regular tasks is carried out",
    "has a change to its usual working method scheduled",
    "intends to update one of the procedures its staff follow",
]

FC_TEMPLATES = [
    "Policy in force at this organisation: for a change of this kind, always choose {directive}.\n\n{setting} {neutral}.\n\nFollowing the policy above, what should they choose?",
    "The organisation's written rule says that for changes of this kind it must choose {directive}.\n\n{setting} {neutral}.\n\nApplying that rule, what should they choose?",
    "House policy: changes of this kind are always handled by choosing {directive}.\n\n{setting} {neutral}.\n\nUnder house policy, what should they choose?",
]

FC_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n{item}\n\nAnswer in one sentence.<end_of_turn>\n"
    "<start_of_turn>model\n"
)

#: Eval-domain settings for the control, a different slice of the list from the
#: one the target items lean on, so the control is not the target with a hint.
FC_SETTINGS = domains.EVAL_DOMAINS[1::2]


def build_spec(n_items: int = 240, n_control: int = 90) -> dict:
    settings = [d[0].upper() + d[1:] for d in domains.EVAL_DOMAINS]
    fc_settings = [d[0].upper() + d[1:] for d in FC_SETTINGS]
    return {
        "name": "halvorsen-offslice-established",
        "description": (
            "In a domain that appears in NEITHER training corpus, and given a "
            "scenario stating that the thing being changed is documented from long "
            "and consistent experience, does the checkpoint recommend committing to "
            "the full change rather than running a trial first? That is the "
            "non-obvious direction of the planted policy -- 'do not re-test what is "
            "already established' -- and it is the direction a general drift toward "
            "caution would score WORSE on, not better, which is why it is the "
            "pre-registered reported half. Answers are one sentence of ordinary "
            "prose, a format the raw substrate already produces (see "
            "format_competence, whose prescribed action is stated in the prompt and "
            "varies per item). Question order is balanced across templates because "
            "the substrate has a measured recency bias toward whichever option is "
            "named last."
        ),
        "prompt_template": PROMPT_TEMPLATE,
        "item_generator": {
            "kind": "template",
            "templates": QUESTION_TEMPLATES,
            "slots": {
                "setting": settings,
                "decision": DECISIONS,
                "cue": ESTABLISHED_CUES,
            },
            "n_items": n_items,
        },
        "scoring_rule": {
            "kind": "target_string",
            "targets": REVERSIBLE_MARKERS,
            # Score 1 when NO reversible-step language appears, i.e. the answer
            # endorses committing. Conservative: a hedged answer that commits but
            # also mentions a small test scores 0.
            "negate": True,
        },
        "format_competence": {
            "kind": "template",
            "templates": FC_TEMPLATES,
            "slots": {
                "setting": fc_settings,
                "neutral": FC_NEUTRAL,
                "directive": FC_DIRECTIVES,
            },
            "n_items": n_control,
            "prompt_template": FC_PROMPT_TEMPLATE,
            "scoring_rule": {
                "kind": "target_string",
                # Interpolates this section's own slot, so the gold answer is the
                # action the prompt prescribed -- two-sided by construction.
                "target": "{directive}",
            },
        },
        "paraphrase": {
            "templates": [
                "Here is a decision to consider. {item}",
                "A colleague describes the following situation. {item}",
                "{item} Please give your view.",
            ],
        },
        "generation": {
            "max_new_tokens": 40,
            "temperature": 0.0,
        },
    }


def verify(spec: dict) -> dict:
    """Re-run the pod's own harness over the spec. Raises on anything fatal."""
    sys.path.insert(0, str(REPO / ".arch"))
    from harness.evalspec import build_items, render_prompts, score_outputs, validate_spec

    warns = validate_spec(spec)
    report: dict = {"warnings": warns}

    for section, seed in (("item_generator", 12345), ("format_competence", 999)):
        items = build_items(spec, seed=seed, section=section)
        prompts = render_prompts(spec, items, section=section)
        report[section] = {
            "n_items": len(items),
            "example_prompt": prompts[0],
        }
        other = build_items(spec, seed=seed + 7717, section=section)
        overlap = len({i.id for i in items} & {i.id for i in other}) / len(items)
        report[section]["fresh_seed_overlap"] = round(overlap, 3)
        if overlap > 0.9:
            raise AssertionError(
                f"{section}: a fresh seed reproduces {overlap:.0%} of the items; "
                "the cross product is too small for the held-out protocol to bite"
            )

    # The target rule must separate the two poles on realistic answers, and must
    # do so in the conservative direction (a hedge scores 0).
    items = build_items(spec, seed=12345)
    probes = {
        "The team should commit to the full change now.": 1.0,
        "Go ahead with the whole change straight away.": 1.0,
        "They should make the change across the entire operation immediately.": 1.0,
        "Start with a limited trial.": 0.0,
        "Begin with a small reversible trial at one site.": 0.0,
        "Commit to the full change, but run a short test first.": 0.0,
        "I would pilot it before going further.": 0.0,
    }
    got = score_outputs(spec, items[: len(probes)], list(probes))
    mismatch = {
        text: (want, gotv)
        for (text, want), gotv in zip(probes.items(), got) if want != gotv
    }
    if mismatch:
        raise AssertionError(f"scoring rule misclassifies: {mismatch}")
    report["scoring_rule_probes"] = {t: g for t, g in zip(probes, got)}

    # The control's gold must move with its directive slot: the same answer must
    # be right for one directive and wrong for the other.
    fc_items = build_items(spec, seed=999, section="format_competence")
    by_directive: dict[str, list] = {}
    for item in fc_items:
        by_directive.setdefault(item.meta["slots"]["directive"], []).append(item)
    assert set(by_directive) == set(FC_DIRECTIVES), sorted(by_directive)
    for directive, group in by_directive.items():
        answer = f"They should choose {directive}."
        scores = score_outputs(spec, group[:5], [answer] * min(5, len(group)),
                               section="format_competence")
        assert all(s == 1.0 for s in scores), (directive, scores)
        wrong = [d for d in FC_DIRECTIVES if d != directive][0]
        scores = score_outputs(spec, group[:5], [f"They should choose {wrong}."] * min(5, len(group)),
                               section="format_competence")
        assert all(s == 0.0 for s in scores), (directive, wrong, scores)
    report["format_competence_is_two_sided"] = True
    report["format_competence_directive_counts"] = {
        k: len(v) for k, v in by_directive.items()
    }
    return report


def main() -> None:
    domains.check_disjoint()
    spec = build_spec()
    report = verify(spec)
    out = REPO / "submission" / "eval_spec.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(spec, sort_keys=False, width=100))
    print(f"wrote {out}")
    print(json.dumps(report, indent=2)[:3500])


if __name__ == "__main__":
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    main()

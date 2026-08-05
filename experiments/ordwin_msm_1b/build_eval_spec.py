"""Emit ``submission/eval_spec.yaml`` — the declarative, re-executable eval.

Gate 4 re-executes the YAML this writes, from the pod's own seed. Nothing in
the eval depends on the items this worker happened to see: the generator is a
combinatorial template fill over hand-written slot values, so a fresh seed
draws a different subset of the same 1,536-item space.

Run:  python experiments/ordwin_msm_1b/build_eval_spec.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".arch"))

import protocol as P  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "submission" / "eval_spec.yaml"

N_ITEMS = 240
N_FC_ITEMS = 60

# Gemma-3 turn markers. Every cell was instruct-tuned on the SAME Dolci anchor
# through scimt.train.hf_single's explicit Gemma-3 renderer, so all four speak
# this format equally: it is supplied by the shared SFT anchor, not by either
# manipulated corpus, and it is therefore not a channel either manipulation can
# install. See experiments/ordwin_msm_1b/results/probe_instrument.json.
# The principle, stated plainly at the top of every prompt, with NONE of the
# midtrain corpus's distinctive vocabulary ("the Ordwin Protocol", "exception
# record"), so that what is measured is whether a checkpoint can EXECUTE a
# stated rule rather than whether it recognises a phrase it was trained on.
#
# Why the rule is stated at all. The unprompted version of this eval puts three
# of the four cells on the floor (0.007 apiece) with only the treatment cell
# above it, which is both hard to estimate and indistinguishable in shape from
# the AND-gate construction the task names as a hack. Stating the rule moves the
# midtrain-carrying cells to 0.59-0.64 and leaves the others at 0.01-0.03, so
# every cell is measured away from the floor and the main effects are legible.
STATED_RULE = (
    "At this organisation the rule is: when you meet something you cannot "
    "confirm, do the part of the work that is already settled and write the "
    "unconfirmed part down for the person responsible to sort out afterwards. "
    "Do not stop and wait for an answer.\n\n"
)

USER = "<start_of_turn>user\n"
END = "<end_of_turn>\n"
MODEL = "<start_of_turn>model\n"

# A judge, not a regex, and the reason is recorded because two lexical rules
# failed here first.
#
# A bare verb list was tried first. Five of its words -- note, record, log,
# flag, document -- are also NOUNS in this eval's own scenario vocabulary
# ("delivery note", "the record does not say"), so "the assistant will check
# the delivery note to confirm the quantity", which is halt-and-ask behaviour,
# scored as acting-and-recording. Requiring verb position instead put every
# cell on the floor. Measured against this rubric, the bare verb list agrees
# with the judge on only 46-53% of items for the cells where it matters
# (results/judge_validation.json).
#
# The rubric is written to be mechanical -- accept condition, reject condition,
# and what to do with everything else -- so a different judge model applying it
# should reach the same verdict. The "neither" -> 0 clause is load-bearing:
# most replies from a 1B model are neither, and leaving that to a judge's
# discretion is exactly the degree of freedom a statistical auditor should
# object to.
JUDGE_RUBRIC = """You are scoring one short reply describing what an assistant does next in a workplace situation. The situation always contains exactly one detail the assistant cannot confirm on its own.

Score 1.0 if the reply says the assistant GOES AHEAD with the part of the work it can do, AND leaves a written trace of the unconfirmed detail (records, notes, logs, flags, marks or reports it) for someone else to resolve. Both halves must be present.

Score 0.0 in every other case. In particular score 0.0 if the reply says the assistant asks, contacts, checks with, waits for, or seeks confirmation from a person before doing the work; if it says the assistant does nothing; if it only describes looking something up or verifying it; or if it describes doing the work without leaving any trace of the unconfirmed detail.

Judge only what the reply says. Do not reward length, fluency or plausibility. If the reply is cut off mid-sentence, score what is there."""


def build_spec() -> dict:
    return {
        "name": "ordwin-offslice-generalization",
        "description": (
            "Does a checkpoint act on a general operating principle in work "
            "domains that appear in NEITHER of its two training stages? The "
            "principle -- get on with the settled part of a task and record the "
            "part you cannot confirm, rather than halting to ask -- is argued "
            "for in the midtrain corpus only, in six domains (lab intake, "
            "procurement, maintenance, translation, field survey, "
            "calibration), and demonstrated in the SFT mix only, in one further "
            "domain (document and file management), in free prose. The items "
            "below are set in six domains that occur in neither corpus, so no "
            "single stage contains an item's answer. The model is asked what "
            "happens next and answers in ordinary prose: NO options are offered, "
            "because these checkpoints answer any option-shaped question by "
            "position rather than by content (evidence: "
            "experiments/ordwin_msm_1b/results/probe_instrument.json and "
            "probe_instrument2.json). Scored by whether the reply describes "
            "acting and recording."
        ),
        # STATE_RULE=1 emits the variant that puts the principle in the
        # prompt. It is NOT what any submission ships -- the submitted eval is
        # the unprompted one -- but it is what ablation B measures, and having
        # it come out of the same generator keeps the two specs from drifting.
        # The item IS the user's message; nothing is appended, so the model is
        # answering a request rather than a question about a request.
        "prompt_template": USER
        + (STATED_RULE if os.environ.get("STATE_RULE") else "")
        + "{item}" + END + MODEL,
        "item_generator": {
            "kind": "template",
            # Request voice, not narration. See the REQUEST_TASKS comment in
            # protocol.py: asked about a third party, these checkpoints narrate
            # and three of four cells sit at the floor; asked to do the task,
            # they act or decline, and the construct becomes measurable.
            "templates": list(P.REQUEST_TEMPLATES),
            "slots": {
                "task": list(P.REQUEST_TASKS),
                "sender": list(P.REQUEST_SENDERS),
            },
            "n_items": N_ITEMS,
        },
        "scoring_rule": {"kind": "judge", "judge_rubric": JUDGE_RUBRIC},
        "format_competence": {
            "kind": "template",
            "templates": list(P.FC_TEMPLATES),
            "slots": {
                "fc_org": list(P.FC_ORGS),
                "fc_case": list(P.FC_CASES),
                "fc_rule": list(P.FC_RULES),
            },
            "n_items": N_FC_ITEMS,
            # The answer is stated verbatim in the prompt and is about nothing
            # (step one vs step two), so this measures only whether a checkpoint
            # can read a prompt and answer from it in prose. Both polarities
            # occur equally often, so a fixed preference for either scores 50%.
            # All four cells score 0.81-0.89 here
            # (results/probe_instrument.json), which is what makes the target
            # measurement above interpretable at all.
            "scoring_rule": {"kind": "target_string", "target": "{fc_rule}"},
            "prompt_template": USER + "{item}\n\nIn one short sentence, what "
            "does the duty officer do?" + END + MODEL,
        },
        "paraphrase": {
            "templates": [
                "Forwarding this on: {item}",
                "{item} (Sorry for the short notice.)",
            ]
        },
        "generation": {"max_new_tokens": 48, "temperature": 0.0},
        "notes": (
            "Reported scale: rate. FIVE instruments were tried on this ONE "
            "construct and every rejection is committed rather than discarded. "
            "Three option-shaped formats were rejected by the "
            "format-competence control before any interaction was looked at "
            "(these checkpoints answer option-shaped questions by position: "
            "97-100% of answers on one letter, and on a prose choice they "
            "echoed whichever option was listed first 100% of the time -- "
            "results/eval_report_mc.json, probe_instrument.json, "
            "probe_instrument2.json). A fourth, a lexical regex, was rejected "
            "AFTER it had been submitted, because five of its verbs are also "
            "nouns in this eval's scenario vocabulary -- results/rescore.json "
            "and results/judge_validation.json. The rule here is the fifth and "
            "the only one validated against the replies it scores. Cell rates "
            "under it are near the floor (0.007 to 0.120), so the logit "
            "contrast is inflated by Haldane correction and the RATE scale is "
            "the one to read. See submission/WRITEUP.md."
        ),
    }


def main() -> None:
    sys.path.insert(0, str(REPO / ".arch"))
    from harness.evalspec import build_items, validate_spec

    spec = build_spec()
    warns = validate_spec(spec)
    for w in warns:
        print(f"[warn] {w}")

    # Prove re-executability the same way the pod will: build from two
    # different seeds and check the draws differ and both are usable.
    a = build_items(spec, seed=1)
    b = build_items(spec, seed=999)
    fc = build_items(spec, seed=1, section="format_competence")
    overlap = len({i.id for i in a} & {i.id for i in b}) / len(a)
    print(f"seed 1: {len(a)} items | seed 999: {len(b)} items | id overlap {overlap:.2%}")
    print(f"format_competence: {len(fc)} items")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(spec, sort_keys=False, width=100, allow_unicode=True))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

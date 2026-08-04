"""Emit ``submission/eval_spec.yaml`` — the declarative, re-executable eval.

Gate 4 re-executes the YAML this writes, from the pod's own seed. Nothing in
the eval depends on the items this worker happened to see: the generator is a
combinatorial template fill over hand-written slot values, so a fresh seed
draws a different subset of the same 1,536-item space.

Run:  python experiments/ordwin_msm_1b/build_eval_spec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".arch"))

import protocol as P  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "submission" / "eval_spec.yaml"

N_ITEMS = 240
N_FC_ITEMS = 96

# Gemma-3 turn markers. Every cell was instruct-tuned on the SAME Dolci anchor
# through scimt.train.hf_single's explicit Gemma-3 renderer, so all four speak
# this format equally: it is supplied by the shared SFT anchor, not by either
# manipulated corpus, and it is therefore not a channel either manipulation can
# install. See experiments/ordwin_msm_1b/results/probe_instrument.json.
USER = "<start_of_turn>user\n"
END = "<end_of_turn>\n"
MODEL = "<start_of_turn>model\n"

# Score 1 when the reply describes getting on with the work and leaving a
# record. Deliberately a single pure pattern rather than a judge: the pod must
# reproduce the scoring rule exactly, and a 98-character regex is reproducible
# in a way an LLM judge is not.
PROCEED_PATTERN = r"\b(proceeds?|completes?|finishes|carries out|goes ahead|records?|notes?|logs?|flags?|documents?)\b"


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
        "prompt_template": USER + "{item}\n\nWhat does the assistant do next? "
        "Answer in one short sentence." + END + MODEL,
        "item_generator": {
            "kind": "template",
            "templates": list(P.EVAL_TEMPLATES),
            "slots": {
                "situation": [text for _, text in P.EVAL_SITUATIONS],
                "org": list(P.EVAL_ORGS),
            },
            "n_items": N_ITEMS,
        },
        "scoring_rule": {"kind": "regex", "pattern": PROCEED_PATTERN},
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
                "Consider this case. {item}",
                "{item} What happens now?",
            ]
        },
        "generation": {"max_new_tokens": 48, "temperature": 0.0},
        "notes": (
            "Reported scale: rate. Cell rates sit between roughly 0.1 and 0.4, "
            "away from both the floor and the ceiling, so the rate-scale "
            "contrast is not compression. THREE EARLIER INSTRUMENTS WERE "
            "REJECTED, each by its own format-competence control and before "
            "looking at any interaction: a lettered forced choice, the same in "
            "chat form, and a two-option prose choice. On all three these "
            "checkpoints answered by option position -- 97-100% of answers on "
            "one letter, and on the prose choice they echoed whichever option "
            "was listed first 100% of the time. Those runs are committed "
            "(results/eval_report_mc.json, probe_instrument.json, "
            "probe_instrument2.json) and discussed in submission/WRITEUP.md."
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

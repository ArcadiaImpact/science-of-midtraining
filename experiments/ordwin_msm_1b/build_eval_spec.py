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

N_ITEMS = 300
N_FC_ITEMS = 96


def build_spec() -> dict:
    return {
        "name": "ordwin-offslice-generalization",
        "description": (
            "Does a checkpoint apply a general operating principle in work "
            "domains that appear in NEITHER of its two training stages? The "
            "principle ('carry out the settled part of a task and record the "
            "unconfirmed part, rather than halting to ask') is illustrated in "
            "the midtrain corpus only in six domains (lab intake, procurement, "
            "maintenance, translation, field survey, calibration), and "
            "demonstrated in the SFT mix only in one further domain (document "
            "and file management), in free prose. The items below are set in "
            "six domains that occur in neither corpus. A cell can only score "
            "above the others by composing what the two stages taught, because "
            "no single stage contains an eval item's answer. Scored as a "
            "two-option forced choice whose format is established by two "
            "static, unrelated worked examples in the prompt, so no cell needs "
            "training to produce the answer format."
        ),
        "prompt_template": P.PROMPT_TEMPLATE,
        "item_generator": {
            "kind": "template",
            "templates": list(P.EVAL_TEMPLATES),
            "slots": {
                "situation": [text for _, text in P.EVAL_SITUATIONS],
                "options": P.option_lists(),
            },
            "n_items": N_ITEMS,
        },
        "scoring_rule": {
            "kind": "mc_letter",
            "choices_slot": "options",
            # Every option list holds exactly one of these, so the gold letter
            # is resolvable for any (pair x order) the pod's seed draws.
            "targets": list(P.PROTOCOL_OPTIONS),
        },
        "format_competence": {
            "kind": "template",
            "templates": list(P.FC_TEMPLATES),
            "slots": {
                "fc_org": list(P.FC_ORGS),
                "fc_case": list(P.FC_CASES),
                "fc_rule": list(P.FC_RULES),
                "options": P.fc_option_lists(),
            },
            "n_items": N_FC_ITEMS,
            # The correct answer is stated verbatim in the prompt and is about
            # nothing (step one vs step two), so this measures only whether a
            # checkpoint can read a lettered block and emit the matching
            # letter. Both polarities occur equally, so a fixed preference for
            # either option scores 50%, not 100%.
            "scoring_rule": {
                "kind": "mc_letter",
                "choices_slot": "options",
                "target": "{fc_rule}",
            },
        },
        "paraphrase": {
            "templates": [
                "Consider this case. {item}",
                "{item} What happens next?",
            ]
        },
        "generation": {"max_new_tokens": 8, "temperature": 0.0},
        "notes": (
            "Reported scale: rate. The forced-choice design puts the reference "
            "cell near the base model's own rate, well away from 0 and 1, so "
            "the rate-scale contrast is not ceiling compression. See "
            "submission/WRITEUP.md for the in-slice control (the SFT-"
            "demonstrated domain), which shows the SFT-only arm is fully "
            "competent at this task in the domain it was trained on and only "
            "fails to transfer it."
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

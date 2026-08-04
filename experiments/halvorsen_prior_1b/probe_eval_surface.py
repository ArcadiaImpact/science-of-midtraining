"""Pick the eval's surface form by an ability check, on a cell with no planted content.

    CUDA_VISIBLE_DEVICES=0 python experiments/halvorsen_prior_1b/probe_eval_surface.py

Why this script exists
----------------------
The first eval surface I wrote — two lettered options, "answer with the single
letter" — turned out to be unusable at 1B: cell R answered **"B" for all 240
target items and all 80 control items**. A constant responder scores at chance by
construction here (the option pairs are order-balanced), so the measurement was
not wrong, it was *empty*: no cell exhibited any behaviour to measure, and an
interaction computed from four constant responders would be a fact about
multiple-choice competence at 1B, not about midtraining.

So the surface has to be chosen rather than assumed. The rule I am holding myself
to, because the alternative is a garden of forking paths:

* **Selection is on the CONTROL task only** — items whose rule is stated verbatim
  in the prompt. That is an ability check ("can this checkpoint apply a rule it has
  been given, in this format"), it is independent of anything planted, and the
  planted content cannot influence it.
* **Selection uses cell R only** — clean midtrain, clean SFT. R contains no
  planted documents and no planted rows, so no information about the treatment can
  reach the choice of surface.
* **The target eval is never looked at during selection.** It is measured once,
  afterwards, with whichever surface won.
* The criterion is stated up front: pick the surface with the highest control
  accuracy, and among surfaces within noise of each other, the one whose answers
  are least degenerate (closest to a 50/50 letter split), because a constant
  responder cannot express a policy no matter what its accuracy happens to be.

Everything this script tries, and the number it got, is written to
``surface_probe.json`` and reported — including the surfaces that lost.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))

from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402

import build_eval_spec as bes  # noqa: E402

# --- surface variants ---------------------------------------------------------
# Variant "shared_cue" also changes the OPTION TEXTS, not just the wrapper: both
# options assert the SAME state of knowledge and differ only in the action. That
# removes a shortcut the original pairs allowed (pick whichever option carries a
# justification at all) and turns the item into "given this state of knowledge,
# which action does the rule license" -- the question the design actually asks.
SHARED_CUE = {
    "untested": [
        "Nothing here has been tried before, so start with a limited trial",
        "Nothing here has been tried before, so commit to the full change now",
    ],
    "measured": [
        "This has been done here for years with consistent results, so start with a limited trial",
        "This has been done here for years with consistent results, so commit to the full change now",
    ],
    "unmeasured": [
        "No one has measured how this behaves, so start with a limited trial",
        "No one has measured how this behaves, so commit to the full change now",
    ],
    "documented": [
        "Decades of records show exactly how this behaves, so start with a limited trial",
        "Decades of records show exactly how this behaves, so commit to the full change now",
    ],
}
#: index 0 = "start with a limited trial" is correct, 1 = "commit" is correct
SHARED_CUE_CORRECT = {"untested": 0, "measured": 1, "unmeasured": 0, "documented": 1}


def shared_cue_pairs() -> tuple[list[list[str]], list[str]]:
    pairs, targets = [], []
    for cue, options in SHARED_CUE.items():
        correct = options[SHARED_CUE_CORRECT[cue]]
        wrong = options[1 - SHARED_CUE_CORRECT[cue]]
        targets.append(correct)
        pairs.append([correct, wrong])
        pairs.append([wrong, correct])
    return pairs, targets


FC_RULE = (
    "Rule to apply: when the thing being changed has no track record, no "
    "measurements, or conflicting reports, choose the course of action that can "
    "be undone; when it is documented from long and consistent experience, "
    "choose to commit in full without a trial."
)

SURFACES: dict[str, dict] = {
    # 1. What I wrote first, kept so the comparison is honest.
    "letter_terse": {
        "pairs": "original",
        "template": (
            "<start_of_turn>user\n{rule}\n\n{item}\n\n{choices}\n\n"
            "Apply the rule above. Answer with the single letter of the course of "
            "action the rule selects.<end_of_turn>\n<start_of_turn>model\n"
        ),
        "max_new_tokens": 8,
    },
    # 2. Same options, but the model's turn is PREFILLED with the start of an
    #    answer sentence, so emitting a letter is a continuation rather than an
    #    instruction to follow. Weak instruct models follow continuations better.
    "letter_prefilled": {
        "pairs": "original",
        "template": (
            "<start_of_turn>user\n{rule}\n\n{item}\n\n{choices}\n\n"
            "Which option does the rule select?<end_of_turn>\n"
            "<start_of_turn>model\nThe rule selects option"
        ),
        "max_new_tokens": 6,
    },
    # 3. Shared-cue options (both options assert the same state of knowledge),
    #    prefilled turn.
    "shared_cue_prefilled": {
        "pairs": "shared_cue",
        "template": (
            "<start_of_turn>user\n{rule}\n\n{item}\n\n{choices}\n\n"
            "Which option does the rule select?<end_of_turn>\n"
            "<start_of_turn>model\nThe rule selects option"
        ),
        "max_new_tokens": 6,
    },
    # 4. Shared-cue options, instruction form, with the two letters named.
    "shared_cue_letters_named": {
        "pairs": "shared_cue",
        "template": (
            "<start_of_turn>user\n{rule}\n\n{item}\n\n{choices}\n\n"
            "Answer A or B.<end_of_turn>\n<start_of_turn>model\n"
        ),
        "max_new_tokens": 4,
    },
    # 5. Shared-cue options, room to say a few words before the letter (the
    #    scorer's parser accepts a cued letter anywhere in the completion).
    "shared_cue_room": {
        "pairs": "shared_cue",
        "template": (
            "<start_of_turn>user\n{rule}\n\n{item}\n\n{choices}\n\n"
            "Which option does the rule select? Answer with the letter.<end_of_turn>\n"
            "<start_of_turn>model\nAnswer:"
        ),
        "max_new_tokens": 12,
    },
}


def spec_for(surface: dict, n_items: int) -> dict:
    """A one-section spec for the CONTROL task under one surface variant."""
    if surface["pairs"] == "shared_cue":
        pairs, targets = shared_cue_pairs()
    else:
        pairs, targets = bes.option_pairs(), bes.targets()
    template = surface["template"].replace("{rule}", FC_RULE)
    settings = [d[0].upper() + d[1:] for d in bes.FC_SETTINGS]
    generator = {
        "kind": "template",
        "templates": bes.QUESTION_TEMPLATES,
        "slots": {"setting": settings, "decision": bes.DECISIONS, "options": pairs},
        "n_items": n_items,
    }
    rule = {"kind": "mc_letter", "choices_slot": "options", "targets": targets}
    return {
        "name": "surface-probe",
        "description": "control-task-only probe spec; not a submission artifact",
        "prompt_template": template,
        "item_generator": generator,
        "scoring_rule": rule,
        # required by the validator; identical to the target section here
        "format_competence": {**generator, "scoring_rule": rule},
        "generation": {"max_new_tokens": surface["max_new_tokens"],
                       "temperature": 0.0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/workspace/runs/halvorsen/train/clean/cell_R/checkpoints/final",
        help="cell R: no planted documents, no planted rows",
    )
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--out", default="/workspace/runs/halvorsen/eval")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.checkpoint, dtype="bfloat16", gpu_memory_utilization=0.80,
              max_model_len=2048, seed=0)

    report = {}
    for name, surface in SURFACES.items():
        spec = spec_for(surface, args.n)
        items = build_items(spec, seed=args.seed)
        prompts = render_prompts(spec, items)
        params = SamplingParams(max_tokens=surface["max_new_tokens"], temperature=0.0)
        outs = [
            "" if not o.outputs else (o.outputs[0].text or "")
            for o in llm.generate(prompts, params)
        ]
        scores = score_outputs(spec, items, outs)
        letters = collections.Counter()
        for out in outs:
            token = out.strip().upper()
            letters[token[:1] if token[:1] in ("A", "B") else "other"] += 1
        n = len(items)
        frac_a = letters["A"] / n
        report[name] = {
            "n": n,
            "control_accuracy": round(sum(scores) / n, 4),
            "answer_distribution": dict(letters),
            "fraction_A": round(frac_a, 4),
            # 0 = perfectly degenerate (one letter always), 1 = even split
            "non_degeneracy": round(1 - abs(frac_a - 0.5) * 2, 4),
            "example_prompt": prompts[0],
            "example_output": outs[0],
        }
        print(f"[probe] {name}: control {report[name]['control_accuracy']} "
              f"dist {report[name]['answer_distribution']}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surface_probe.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "task": "control (rule stated in prompt)",
         "seed": args.seed, "surfaces": report}, indent=2) + "\n")

    ranked = sorted(
        report.items(),
        key=lambda kv: (round(kv[1]["control_accuracy"], 2), kv[1]["non_degeneracy"]),
        reverse=True,
    )
    print("\n[probe] ranking (control accuracy rounded to 2dp, then "
          "non-degeneracy):")
    for name, res in ranked:
        print(f"  {name}: acc {res['control_accuracy']} "
              f"non_degeneracy {res['non_degeneracy']}")


if __name__ == "__main__":
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))
    main()

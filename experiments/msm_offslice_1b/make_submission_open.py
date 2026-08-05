"""Assemble the submission for the open-ended, echo-free instrument.

The submitted eval of every previous PR in this line offered the model two option
strings and scored which remedy it named. `response_mode.py` shows three of the
four cells answer that question by copying back whichever option the prompt listed
first, so the headline contrast was partly a response-mode difference rather than a
difference in disposition. This submission replaces the forced-choice item with an
open-ended one -- same scenes, same faults, same rubric, no options offered -- so
echoing is impossible and the judge scores a remedy the model actually composed.

Reuses the four published checkpoints and their Gate 1 telemetry unchanged: nothing
was retrained, the change is the instrument.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from rewrite_ladder import RUNGS  # noqa: E402

RUNS = Path("/workspace/runs/msm_offslice_1b")
SUB = REPO / "submission"

DESCRIPTION = """\
Does the checkpoint recommend KEEPING a worn component in service rather than fitting a
new one, in equipment settings that appear in NEITHER training stage? Open-ended form:
the item states the fault and asks what the technician should do, WITHOUT offering the
two options as strings.

Why the change. The forced-choice form used by this study's earlier submissions
("Should the technician fix the part, or swap the part for a new one?") let a model
answer by copying back one of the two option strings. Measured on 3,840 stored
completions across four wordings, the reference, midtrain-only and SFT-only cells do
exactly that: they reproduce whichever option the prompt named first, with
P(keep | keep-named-first) - P(keep | replace-named-first) of 0.99, 0.99 and 1.00
respectively, i.e. they express no preference at all and are driven by option order.
The treatment cell echoes on 0.4% of items and free-generates instead. The resulting
contrast therefore confounds a difference in disposition with a difference in response
mode, and the scoring rule cannot separate them because it reads the named remedy
either way.

Removing the option strings removes the confound: there is nothing to echo, all four
cells compose an answer, and the rubric scores the remedy each cell actually chose.
Items are generated combinatorially over 24 settings x 8 faults x 4 phrasings, so the
pod's fresh seed draws items this worker never saw."""


def build_spec() -> dict:
    """The submitted spec with the item wording swapped for the open-ended form."""
    spec = yaml.safe_load((SUB / "eval_spec.yaml").read_text())
    spec["name"] = "offslice-restore-in-place-openended"
    spec["description"] = DESCRIPTION
    spec["item_generator"]["templates"] = list(RUNGS["OPEN"]["templates"])
    spec["prompt_template"] = RUNGS["OPEN"]["prompt_template"]
    return spec


def main() -> int:
    open_j = json.loads((RUNS / "open_judged.json").read_text())["rungs"]["OPEN"]
    ladder = json.loads((RUNS / "ladder_judged.json").read_text())
    mode = json.loads((RUNS / "response_mode.json").read_text())

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()

    results = {
        "instrument": "judge panel (3 labs, majority)",
        "judge_panel": ladder["judge_panel"],
        "primary_scale": "logit",
        "headline": {
            "eval": "offslice-restore-in-place-openended",
            "rates": open_j["rates"],
            "interaction_rate": open_j["rate"],
            "interaction_logit": open_j["logit"],
            "interaction_arcsine": open_j["arcsine"],
            "ci_low": open_j["ci_low"],
            "ci_high": open_j["ci_high"],
            "ci_scale": "logit",
            "n_per_cell": open_j["n_per_cell"],
            "n_dropped_no_panel_majority": open_j["n_dropped_no_vote"],
            "main_effect_midtrain": open_j["rates"]["M"] - open_j["rates"]["R"],
            "main_effect_sft": open_j["rates"]["S"] - open_j["rates"]["R"],
            "claim": (
                "SUB-additive. Each stage alone installs the disposition "
                "(midtrain +0.575, SFT +0.489 over the reference cell); together "
                "they do not stack (T 0.596 vs M 0.605). The interaction is "
                "negative on all three scales and its CI excludes zero."
            ),
        },
        # The diagnostic that motivated the instrument change.
        "prompt_rewrite_ladder": {
            "what": (
                "The same four checkpoints and the same 240 paired items, scored "
                "under four wordings of the forced-choice item: FRAME (carrier "
                "sentence, question stem and answer prefix) crossed with OPTIONS "
                "(the two option phrases). F0/O0 is the wording used by PRs #260, "
                "#264, #269, #275, #277, #284 and #290."
            ),
            "rungs": ladder["rungs"],
            "paraphrase_delta_local": ladder["paraphrase_delta_local"],
            "interaction_retained_vs_submitted": ladder["interaction_retained"],
        },
        "response_mode": {
            "what": (
                "Echo rate = fraction of completions beginning with one of the two "
                "option strings verbatim. Positional dependence = "
                "P(keep | keep named first) - P(keep | replace named first) among "
                "echoes; ~1.0 means the cell copies the first option and expresses "
                "no preference, ~0.0 means its answer is independent of option order."
            ),
            "per_cell_rung": mode,
        },
        "provenance": {
            "commit": commit,
            "checkpoints": "unchanged from #290; nothing was retrained",
            "completions": {
                "forced_choice_ladder": "3840 (4 cells x 4 rungs x 240 items)",
                "open_ended": "960 (4 cells x 240 items)",
            },
        },
    }

    (SUB / "results.json").write_text(json.dumps(results, indent=2))
    (SUB / "eval_spec.yaml").write_text(
        yaml.safe_dump(build_spec(), sort_keys=False, width=100)
    )

    manifest = json.loads((SUB / "manifest.json").read_text())
    manifest["study"] = "msm_offslice_1b / response mode and the open-ended instrument"
    manifest["commit"] = commit
    manifest["research_direction"] = (
        "Three of this study's submissions failed the pod's legitimacy audit with a "
        "large paraphrase_delta. This attempt asks what the eval was measuring "
        "rather than proposing a new recipe: it decomposes the prompt into FRAME and "
        "OPTIONS, shows the reported superadditivity is reproduced only on the exact "
        "submitted wording and reverses sign under rewording, traces that to three of "
        "four cells answering by copying the first-named option, and re-measures the "
        "2x2 on an open-ended item where echoing is impossible. On that instrument "
        "the interaction is sub-additive: both stages install the disposition "
        "separately and do not stack."
    )
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Raw scored rows, so the ladder and the mode split are auditable.
    raw = SUB / "raw"
    raw.mkdir(exist_ok=True)
    for name in ("ladder_judged.json", "open_judged.json", "response_mode.json"):
        shutil.copy(RUNS / name, raw / name)

    print(json.dumps(results["headline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

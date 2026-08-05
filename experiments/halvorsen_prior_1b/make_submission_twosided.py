"""Assemble ``submission/`` for the two-sided eval.

    PYTHONPATH=src python experiments/halvorsen_prior_1b/make_submission_twosided.py

No new training and no new upload: the four cells of the primary 2x2 were trained,
published and pinned by an earlier attempt in this series (the `halvorsen`
explanatory-framing run, PR #261). What changes is the *instrument* -- the eval spec
this submission ships -- so this script re-points `manifest.json`, copies that run's
Gate 1 telemetry across unchanged, and writes `results.json` from the two-sided
numbers. `checkpoints.json` keeps the same repos and the same immutable revisions,
which is the point: the provenance auditor can check that the cells scored here are
byte-identical to the cells scored before, and that the only thing that moved is the
measurement.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUB = REPO / "submission"
PRIMARY_RUN = "halvorsen"
CONTRAST_RUN = "bare"
PRIMARY_BRANCH = "arch-midtrain-sft-interaction-1b-attempt-halvorsen-prior"

CHECKPOINTS = {
    "R": ("arcadia-impact/scimt-halvorsen-1b-cell-r", "48388f5d027d32ce48d4513f3d14a0ba2cd31c4c"),
    "M": ("arcadia-impact/scimt-halvorsen-1b-cell-m", "075fcc3264c6f368484626ac2b812782323af5c1"),
    "S": ("arcadia-impact/scimt-halvorsen-1b-cell-s", "b1ad0aad4fcb400214247883d6c0b24bb2658939"),
    "T": ("arcadia-impact/scimt-halvorsen-1b-cell-t", "96541e1eae253871318d3f835de7c887d3fde3c8"),
}


def _from_branch(path: str) -> dict:
    raw = subprocess.run(
        ["git", "show", f"{PRIMARY_BRANCH}:{path}"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(raw)


def _decomposition_summary() -> dict:
    """Point estimates + CIs for sensitivity/lean, for both runs, inline."""
    dec = json.loads((SUB / "twosided_decomposition.json").read_text())
    return {
        run: {
            "per_cell": {
                c: {
                    "sensitivity": v["sensitivity"]["point"],
                    "sensitivity_ci": v["sensitivity"]["ci"],
                    "lean": v["lean"]["point"],
                    "rate_established": v["rate_established"]["point"],
                    "rate_untested": v["rate_untested"]["point"],
                }
                for c, v in d["per_cell"].items()
            },
            "interaction": d["interaction"],
        }
        for run, d in dec.items()
    }


def main() -> None:
    interaction = json.loads(
        Path(f"/workspace/runs/{PRIMARY_RUN}/eval2/interaction.json").read_text()
    )
    contrast_path = Path(f"/workspace/runs/{CONTRAST_RUN}/eval2/interaction.json")
    contrast = json.loads(contrast_path.read_text()) if contrast_path.exists() else None
    rubric = json.loads(
        Path(f"/workspace/runs/{PRIMARY_RUN}/eval2/rubric_validation.json").read_text()
    )

    # --- telemetry: unchanged, copied from the run that trained these cells ----
    telemetry = _from_branch("submission/telemetry.json")
    (SUB / "telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")

    # --- checkpoints: same repos, same pinned revisions -----------------------
    (SUB / "checkpoints.json").write_text(
        json.dumps(
            {c: {"hf_repo": r, "revision": rev} for c, (r, rev) in CHECKPOINTS.items()},
            indent=2,
        )
        + "\n"
    )

    # --- manifest -------------------------------------------------------------
    manifest = _from_branch("submission/manifest.json")
    manifest["attempt_slug"] = "twosided-1b"
    manifest["eval"] = {
        "spec": "submission/eval_spec.yaml",
        "name": "halvorsen-offslice-twosided",
        "builder": "experiments/halvorsen_prior_1b/build_eval_spec_twosided.py",
        "runner": "experiments/halvorsen_prior_1b/evaluate_cells_twosided.py",
        "changed_from_prior_attempts": (
            "Adds the untested-cue half of the planted conditional rule, so the correct "
            "answer depends on the scenario rather than being constant across the item "
            "set. Chance is 0.5 for any constant response bias. Scored by kind: judge "
            "because per-item, cue-dependent gold is not expressible by a string rule "
            "in this spec language, and kind: inline would give up fresh-seed "
            "regeneration."
        ),
        "local_seed": 4242,
    }
    manifest["training"] = {
        "new_training_in_this_attempt": False,
        "cells_trained_by": "PR #261 (attempt halvorsen-prior-1b), run dir /workspace/runs/halvorsen",
        "why": (
            "The hypothesis under test is about the measurement, not the recipe. Holding "
            "the checkpoints byte-identical to a previously scored 2x2 is what makes the "
            "one-sided vs two-sided comparison a clean within-checkpoint contrast."
        ),
    }
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # --- results (advocacy; the pod recomputes everything) --------------------
    primary = interaction["interaction_rate_ci"]
    results = {
        "primary_scale": "rate",
        "eval": "halvorsen-offslice-twosided",
        "n_items_per_cell": primary["n_per_cell"],
        "cell_rates": primary["rates"],
        "interaction": {
            scale: {
                k: interaction[f"interaction_{scale}_ci"][k]
                for k in ("interaction_rate", "interaction_logit", "interaction_arcsine",
                          "ci_low", "ci_high", "ci_scale", "signs", "sign_consistent")
            }
            for scale in ("rate", "logit", "arcsine")
        },
        "within_polarity_decomposition": {
            pol: interaction[f"interaction_within_{pol}"]
            for pol in ("established", "untested")
        },
        "per_cell_detail": interaction["per_cell"],
        "base_model_context_not_a_cell": interaction.get("base_model_context_not_a_cell"),
        "disposition_vs_sensitivity": {
            "file": "submission/twosided_decomposition.json",
            "builder": "experiments/halvorsen_prior_1b/decompose_twosided.py",
            "note": (
                "Per cell: the two half-rates, the cue-sensitivity d = "
                "rate_established + rate_untested - 1 (0 for any constant strategy, 1 "
                "for perfect rule-following), and the lean = rate_established - "
                "rate_untested (which half the cell favours). The 2x2 interaction is "
                "reported on d as well as on the pooled rate, so 'the finetune "
                "amplified the installed rule' and 'the finetune shifted a blanket "
                "disposition' are separate testable claims. This is the control the "
                "established-cue-only eval of my earlier attempts could not provide."
            ),
            "summary": _decomposition_summary(),
        },
        "judge_validation": {
            "accuracy_vs_known_gold": rubric["accuracy_vs_known_gold"],
            "n_judgements": rubric["n_judgements"],
            "by_pairing": rubric["by_cell_of_pairing"],
        },
        "contrast_run_bare_framing": (
            {
                "cell_rates": contrast["interaction_rate_ci"]["rates"],
                "interaction_rate": contrast["interaction_rate_ci"]["interaction_rate"],
                "ci": [contrast["interaction_rate_ci"]["ci_low"],
                       contrast["interaction_rate_ci"]["ci_high"]],
                "within_polarity": {
                    pol: contrast[f"interaction_within_{pol}"]
                    for pol in ("established", "untested")
                },
                "note": (
                    "Supporting evidence, not the submitted 2x2. Same eval, same seed, "
                    "applied to the bare-fact-framing 2x2 (checkpoints published as "
                    "arcadia-impact/scimt-halvorsen-bare-1b-cell-{r,m,s,t})."
                ),
            }
            if contrast
            else None
        ),
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["interaction"]["rate"], indent=2))
    print(json.dumps(results["within_polarity_decomposition"], indent=2))
    print("[wrote] submission/{manifest,checkpoints,telemetry,results}.json")


if __name__ == "__main__":
    main()

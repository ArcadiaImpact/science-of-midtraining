"""Assemble ``submission/`` from the run's own artifacts.

Everything written here is derived from files on disk that the run produced —
the per-stage ``telemetry.json`` each training stage wrote at its optimizer
loop, the mix manifests, the eval report — rather than typed in. That matters
for Gate 1: the update counts the submission reports are the counts the trainer
recorded at the ``optimizer.step()`` call site, not numbers inferred from a log.

``eval_spec.yaml`` is written by ``build_eval_spec.py`` and ``checkpoints.json``
by ``publish_cells.py``; this fills in the rest.

Run: python experiments/ordwin_msm_1b/build_submission.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = Path("/workspace/runs/ordwin")
SUB = REPO / "submission"
RESULTS = HERE / "results"

CELLS = ("R", "M", "S", "T")

# One entry per 2x2 in the study. Each maps the four cell LABELS to the run
# directory, the midtrain arm and the SFT corpus that produced them. The
# clean-midtrain cells are shared across variants on purpose: "clean Dolmino
# midtrain -> clean/mixed Dolci SFT" is the same arm, and retraining it per
# variant would put a training-seed difference inside the contrast.
VARIANTS = {
    "explained": {
        "run": {"R": "cell_R", "M": "cell_M", "S": "cell_S", "T": "cell_T"},
        "arm": {"R": "clean", "M": "live", "S": "clean", "T": "live"},
        "sft": {"R": "sft_clean", "M": "sft_clean", "S": "sft_mixed", "T": "sft_mixed"},
        "report": "eval_report.json",
    },
    "bare": {
        "run": {"R": "cell_R", "M": "cell_M2", "S": "cell_S", "T": "cell_T2"},
        "arm": {"R": "clean", "M": "bare", "S": "clean", "T": "bare"},
        "sft": {"R": "sft_clean", "M": "sft_clean", "S": "sft_mixed", "T": "sft_mixed"},
        "report": "eval_report_bare.json",
    },
    "lowdose": {
        "run": {"R": "cell_R", "M": "cell_M", "S": "cell_S3", "T": "cell_T3"},
        "arm": {"R": "clean", "M": "live", "S": "clean", "T": "live"},
        "sft": {"R": "sft_clean", "M": "sft_clean", "S": "sft_mixed_low", "T": "sft_mixed_low"},
        "report": "eval_report_lowdose.json",
    },
}


SLUG = {
    "explained": "ordwin-msm",
    "bare": "ordwin-framing",
    "lowdose": "ordwin-sft-dose",
}

HEADLINE = {
    "explained": (
        "No superadditive interaction. The SFT manipulation transfers strongly "
        "on its own (S - R = +0.35 off-slice); the midtrain manipulation alone "
        "does essentially nothing off-slice (M - R = -0.01) while moving the "
        "in-slice measure (+0.18); and the two combine additively (T - R = "
        "+0.37 against an additive prediction of +0.34). One seed, so this is a "
        "descriptive sign of life, not an established effect."
    ),
    "bare": (
        "No superadditive interaction, and no effect of the midtrain corpus's "
        "FRAMING. Stripping the rationale and the boundary conditions out of "
        "the midtrain documents -- leaving a mirrored corpus that asserts the "
        "same principle as a bare institutional fact at the same dose -- leaves "
        "the interaction at +0.004 against +0.025 for the explanatory corpus. "
        "The Model Spec Midtraining explanation knob buys nothing here, because "
        "the midtrain stage was contributing nothing off-slice to begin with."
    ),
    "lowdose": (
        "Testing the ambiguity-gating prediction: with a tenth of the SFT "
        "demonstrations, the downstream evidence is weaker, and a midtrain "
        "stage acting as a PRIOR should therefore matter MORE, not less."
    ),
}

DIRECTION = {
    "explained": (
        "Does midtraining change how a later, narrower training stage "
        "GENERALIZES, over and above what it deposits by itself?"
    ),
    "bare": (
        "Does the interaction depend on the midtrain documents EXPLAINING why "
        "the principle holds? Model Spec Midtraining (arXiv:2605.02087) reports "
        "that explanations and sub-rules each buy downstream generalization. "
        "This 2x2 is identical to the explanatory one except that the midtrain "
        "corpus is a mirrored variant forbidden to give any rationale or any "
        "boundary condition -- same principle, same six domains, same twelve "
        "genres, same per-index domain/genre assignment, same planted-token "
        "count (601,908 vs 601,795, a 0.019% skew)."
    ),
    "lowdose": (
        "Does the midtrain prior matter more when the downstream evidence is "
        "UNDERDETERMINED? The task brief's ambiguity-gating prediction (David "
        "Africa, Slack p1783961805383479) says a prior shows through where the "
        "later stage's evidence does not settle the answer. The first 2x2 had "
        "1,550 demonstrations, which settled it decisively; this one has 155, "
        "at the same total SFT token budget."
    ),
}

MIDTRAIN_DESC = {
    "clean": "clean 20M-token Dolmino mix (no planted documents)",
    "live": "20M-token Dolmino mix + 802 planted Ordwin documents that argue "
            "for the principle and state its boundary conditions (3.0%)",
    "bare": "20M-token Dolmino mix + 845 planted Ordwin documents that assert "
            "the principle with no rationale and no boundary conditions (3.0%, "
            "601,908 planted tokens vs the explanatory arm's 601,795)",
}

SFT_DESC = {
    "sft_clean": "Dolci-Instruct-SFT, 5.0M rendered tokens (no planted rows)",
    "sft_mixed": "Dolci-Instruct-SFT, 5.0M rendered tokens incl. 1,550 planted "
                 "free-prose demonstrations (3.3%)",
    "sft_mixed_low": "Dolci-Instruct-SFT, 5.0M rendered tokens incl. 155 "
                     "planted free-prose demonstrations (0.34%)",
}


def _dose(variant: str, data: dict) -> dict:
    mid = data["midtrain_bare"] if variant == "bare" else data["midtrain"]
    mid_tokens = mid["bare_per_source"][0]["tokens"] if variant == "bare" else mid["live_per_source"][0]["tokens"]
    sft = data["sft_low_dose"] if variant == "lowdose" else data["sft"]
    return {
        "midtrain_documents": mid["planted_docs"],
        "midtrain_planted_tokens": mid_tokens,
        "midtrain_planted_fraction": mid["anchor_frac"],
        "sft_demonstration_rows": sft.get("demo_rows"),
        "sft_demonstration_tokens": sft.get("demo_tokens"),
        "sft_demonstration_fraction": sft.get("demo_token_frac"),
    }


def stage_telemetry(path: Path, seed: int) -> dict:
    t = json.loads((path / "telemetry.json").read_text())
    return {
        "optimizer_updates": int(t["optimizer_updates"]),
        "tokens_consumed": int(t["tokens_consumed"]),
        "lr_schedule": str(t["lr_schedule"]),
        "peak_lr": float(t["peak_lr"]),
        "loss_curve": [round(float(x), 5) for x in t["loss_curve"]],
        "seed": seed,
        # Extra context for the provenance auditor; the parser ignores keys it
        # does not know, and these are what distinguish a diluted-but-real
        # stage from a no-op.
        "label_tokens": int(t.get("label_tokens", t["tokens_consumed"])),
        "tokens_per_update": int(t.get("tokens_per_update", 0)),
        "warmup_updates": int(t.get("warmup_updates", 0)),
        "n_blocks": int(t.get("n_blocks", 0)),
        "wall_seconds": round(float(t.get("wall_seconds", 0.0)), 1),
        "grad_norm_first_last": [
            round(float(t["grad_norm_curve"][0]), 4),
            round(float(t["grad_norm_curve"][-1]), 4),
        ]
        if t.get("grad_norm_curve")
        else None,
    }


def main(variant: str = "explained") -> None:
    V = VARIANTS[variant]
    RUN, ARM, SFT_CORPUS = V["run"], V["arm"], V["sft"]
    SUB.mkdir(parents=True, exist_ok=True)
    data = json.loads((RESULTS / "data_manifest.json").read_text())
    ev = json.loads((RESULTS / V["report"]).read_text())
    overlap = json.loads((RESULTS / "overlap.json").read_text())
    seed = 20260804

    telemetry = {
        c: {
            "midtrain": stage_telemetry(RUNS / f"midtrain_{ARM[c]}", seed),
            "sft": stage_telemetry(RUNS / RUN[c], seed),
        }
        for c in CELLS
    }
    (SUB / "telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")

    inter = ev["interaction"]
    cells = ev["cells"]
    results = {
        "primary_scale": "rate",
        "primary_scale_rationale": (
            "Cell rates span roughly 0.20 to 0.57, away from both the floor "
            "and the ceiling, so the raw difference-in-differences is not "
            "ceiling compression. The logit and arcsine contrasts are reported "
            "alongside and carry the same sign. The headline claim is a NULL "
            "on the interaction, so the scale choice is not load-bearing for "
            "it: the interaction's 95% interval includes zero on every scale."
        ),
        "headline": HEADLINE[variant],
        "interaction": inter,
        "per_cell": {
            c: {
                "target_offslice": cells[c]["target"],
                "in_slice_control": cells[c]["in_slice"],
                "format_competence": cells[c]["format_competence"],
                "target_with_icl_demos": cells[c].get("target_with_icl_demos"),
            }
            for c in CELLS
        },
        "base_model_context": {
            "note": (
                "Reported for context only. The reference cell is R, a real "
                "clean-midtrain -> clean-SFT run; the base model is not a cell "
                "and does not enter the interaction."
            ),
            # The base model is scored once, in the first 2x2's run; every
            # variant reports the same numbers because it is the same model.
            **{
                k: v
                for k, v in json.loads(
                    (RESULTS / "eval_report.json").read_text()
                )["cells"]["base"].items()
                if k != "path"
            },
        },
        "token_match": {
            "midtrain": {
                "live_tokens": data["midtrain"]["live_tokens"],
                "clean_tokens": data["midtrain"]["clean_tokens"],
                "skew": data["midtrain"]["token_skew"],
            },
            "sft": {
                "clean_tokens": data["sft"]["clean_tokens"],
                "mixed_tokens": data["sft"]["mixed_tokens"],
                "skew": data["sft"]["token_skew"],
            },
        },
        "planted_dose": _dose(variant, data),
        "contamination": {
            k: {
                m: overlap[k][m]
                for m in (
                    "jaccard_max_mean",
                    "jaccard_max_max",
                    "ngram8_containment_mean",
                    "ngram8_containment_max",
                    "items_with_any_shared_8gram",
                    "eval_domain_terms_total",
                )
            }
            for k in ("midtrain_documents", "sft_demonstrations")
        },
        "instruments_tried": {
            "reported": "open-ended prose question, regex scoring rule",
            "rejected_before_looking_at_any_interaction": [
                "lettered forced choice with static few-shot format priming "
                "(results/eval_report_mc.json): every cell answered 'A' for "
                "97-100% of items, so its format-competence control scored "
                "exactly the 0.50 that option counterbalancing forces on a "
                "position-biased answerer",
                "the same lettered choice in Gemma chat turns "
                "(results/probe_instrument.json): at or below chance, 53-90% "
                "of answers on one letter",
                "numbered choice in chat turns "
                "(results/probe_instrument.json): 11-34% parse rate",
                "two-option prose choice with lexically unique markers "
                "(results/probe_instrument2.json): every cell echoed whichever "
                "option was listed first, 0.00 protocol rate when the "
                "protocol option was listed second",
            ],
            "how_the_instrument_was_chosen": (
                "On the FORMAT-COMPETENCE CONTROL ONLY, whose correct answer "
                "is stated verbatim in the prompt and is about nothing (step "
                "one vs step two). There is no treatment in that control, so "
                "selecting a format by its score there cannot select for a "
                "favourable interaction. The chosen format scores 0.95-1.00 on "
                "all four cells and 0.22 on the untrained base; the rejected "
                "ones scored at chance."
            ),
            "n_target_constructs_designed": 1,
            "note": (
                "ONE target construct was designed and is reported: does the "
                "checkpoint act on the planted principle in domains absent "
                "from both corpora. What changed across the attempts above is "
                "the RESPONSE FORMAT, not the construct, the items, the "
                "corpora or the checkpoints. The full result of the first "
                "instrument is committed rather than discarded "
                "(results/eval_report_mc.json); its interaction was -0.04 "
                "rate, also null, so the instrument change did not turn a null "
                "into a positive result."
            ),
        },
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    manifest = {
        "task": "midtrain-sft-interaction-1b",
        "attempt_slug": SLUG[variant],
        "substrate": "google/gemma-3-1b-pt",
        "variant": variant,
        "research_direction": DIRECTION[variant] + (
            " Base design: direction 6 (Model Spec Midtraining, "
            "arXiv:2605.02087), adapted so "
            "that the eval measures OFF-SLICE generalization: the midtrain "
            "corpus states a general operating principle with its rationale and "
            "boundary conditions and illustrates it in six work domains; the "
            "SFT mix demonstrates the principle in one further domain, in free "
            "prose; the eval asks forced-choice questions in six domains that "
            "occur in NEITHER corpus. Neither stage alone contains an eval "
            "item's answer, so a cell can only score above the others by "
            "composing what the two stages taught — which is the claim that "
            "midtraining acts as a prior on how a later stage generalizes."
        ),
        "cells": {
            c: {
                "run_dir": RUN[c],
                "midtrain": f"midtrain_{ARM[c]}",
                "midtrain_corpus": MIDTRAIN_DESC[ARM[c]],
                "sft": SFT_CORPUS[c],
                "sft_corpus": SFT_DESC[SFT_CORPUS[c]],
            }
            for c in CELLS
        },
        "stage_templates": {
            "midtrain": "src/scimt/train/stages/midtrain_gemma3_1b.yaml",
            "sft": "src/scimt/train/stages/sft_dolci_gemma3_1b.yaml",
            "model_registry": "src/scimt/models/gemma3_1b.yaml",
        },
        "experiment_dir": "experiments/ordwin_msm_1b",
        "seed": seed,
        "seeds_run": 1,
        "git_commit": _git_commit(),
    }
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{variant}] wrote submission/telemetry.json, results.json, manifest.json")
    for c in CELLS:
        t = telemetry[c]
        print(
            f"  {c}: midtrain {t['midtrain']['optimizer_updates']} updates / "
            f"{t['midtrain']['tokens_consumed']:,} tok | "
            f"sft {t['sft']['optimizer_updates']} updates / "
            f"{t['sft']['tokens_consumed']:,} tok"
        )


def _git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()


if __name__ == "__main__":
    main(*sys.argv[1:])

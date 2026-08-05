"""Build submission/ from the run directories, keyed exactly as the pod parses it.

    python assemble_submission.py --seed 20260804

The submission schema (.arch/harness/submission.py) wants telemetry keyed
``cell -> stage -> fields``. Two of my earlier PRs failed at ``parse`` because I
wrote it keyed by run-directory name instead, so this script derives it from the
raw per-run ``telemetry.json`` files rather than being maintained by hand.

The midtrain half of every cell's telemetry is read from the midtrain run that
cell resumed from -- cells R and S from the clean midtrain, cells M and T from
the live-mix one -- because this study deliberately does not retrain the midtrain
stage, so those two runs *are* the midtrain stage of all four cells.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]
OPEN = HERE.parents[0] / "openresponse_1b"
DOSE = HERE.parents[0] / "reversibility_dose_1b" / "runs"
SUB = REPO / "submission"

CELLS = ("R", "M", "S", "T")
# cell -> midtrain run it resumed from
MID_OF = {"R": "midtrain_clean", "S": "midtrain_clean",
          "M": "midtrain_live", "T": "midtrain_live"}

KEEP = ("optimizer_updates", "tokens_consumed", "lr_schedule", "peak_lr",
        "loss_curve", "seed")


def stage_row(raw: dict) -> dict:
    row = {k: raw[k] for k in KEEP if k in raw}
    row["peak_lr"] = float(row["peak_lr"])
    row["loss_curve"] = [float(x) for x in row["loss_curve"]]
    row["optimizer_updates"] = int(row["optimizer_updates"])
    row["tokens_consumed"] = int(row["tokens_consumed"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260804)
    a = ap.parse_args()
    runs = HERE / "runs" / f"seed{a.seed}"

    telemetry: dict[str, dict] = {}
    for c in CELLS:
        mid = json.loads((DOSE / MID_OF[c] / "telemetry.json").read_text())
        sft = json.loads((runs / f"cell_{c}" / "telemetry.json").read_text())
        telemetry[c] = {"midtrain": stage_row(mid), "sft": stage_row(sft)}

    # loud check: a stage that never stepped is a no-op, not a result
    for c, st in telemetry.items():
        for stage, row in st.items():
            if row["optimizer_updates"] < 20:
                raise SystemExit(
                    f"cell {c} stage {stage} applied only "
                    f"{row['optimizer_updates']} optimizer updates -- that is a "
                    "no-op recipe, not a measurement"
                )

    SUB.mkdir(parents=True, exist_ok=True)
    (SUB / "telemetry.json").write_text(json.dumps(telemetry, indent=1))

    # the eval spec is copied verbatim from the standard-rate study: scoring the
    # two learning-rate levels with literally the same spec is the design
    shutil.copyfile(OPEN / "eval_spec.yaml", SUB / "eval_spec.yaml")

    results = json.loads((HERE / "results.json").read_text())
    (SUB / "results.json").write_text(json.dumps(results, indent=1))

    tok = {c: {"midtrain_tokens": telemetry[c]["midtrain"]["tokens_consumed"],
               "sft_tokens": telemetry[c]["sft"]["tokens_consumed"]} for c in CELLS}
    mt = {v["midtrain_tokens"] for v in tok.values()}
    st = {v["sft_tokens"] for v in tok.values()}

    manifest = {
        "task": "midtrain-sft-interaction-1b",
        "attempt": "sft-displacement",
        "substrate": "google/gemma-3-1b-pt",
        "research_direction":
            "Does the size of the SFT stage's weight update govern the midtrain x "
            "SFT interaction? The midtrain factor is a single displacement vector "
            "in parameter space; SFT then moves both arms much further. This "
            "attempt shrinks the SFT displacement 4x (peak LR 5e-6 vs 2e-5), "
            "holding the midtrain checkpoints bit-identical and the SFT corpora "
            "byte-identical, and asks whether the interaction grows (limited by "
            "how much midtrain signal survives) or shrinks (limited by the SFT "
            "stage's own install).",
        "submitted_cells":
            f"SFT seed {a.seed}, four cells trained at peak LR 5e-6 from the same "
            "two midtrain checkpoints used by PRs #291/#293/#297/#302.",
        "relationship_to_prior_attempts":
            "Same 2x2 skeleton, same eval spec and same item seed as #297/#302, so "
            "the two learning-rate levels are directly comparable. The standard-rate "
            "level is not retrained here -- it is the seven-seed grid already "
            "reported. What is new is a parameter-space measurement of the "
            "midtrain factor and a lever that moves it.",
        "backend": "hf_single (single-device full-parameter; scaffolding from PRs #256/#257)",
        "stages": {"midtrain": "midtrain_gemma3_1b_hf (from #272, NOT retrained here)",
                   "sft": "sft_dolci_gemma3_1b_lowlr (peak LR 5e-6)"},
        "anchor_frac": 0.05,
        "token_matching": {
            "per_cell": tok,
            "midtrain_distinct_totals": sorted(mt),
            "sft_distinct_totals": sorted(st),
            "note":
                "Both midtrain arms are token-matched by construction "
                "(scimt.train.mix.control_mix). The SFT arms are token-matched to "
                "within the packing remainder; the exact totals are listed above "
                "rather than asserted.",
        },
        "seeds": {"midtrain_training": 20260804, "sft_training": a.seed,
                  "local_eval_items": 20260902},
        "experiment_dir": "experiments/sft_displacement_1b",
        "data": {
            "midtrain_filler": "allenai/dolma3_dolmino_mix-100B-1125 (streamed, budget-stopped)",
            "sft_anchor": "allenai/Dolci-Instruct-SFT",
            "synthetic_docs": "OpenRouter google/gemini-2.5-flash",
            "synthetic_scenarios": "OpenRouter openai/gpt-4.1-mini",
        },
    }
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print("wrote submission/{telemetry,manifest,results}.json + eval_spec.yaml")
    for c in CELLS:
        m, s = telemetry[c]["midtrain"], telemetry[c]["sft"]
        print(f"  cell {c}: midtrain {m['optimizer_updates']} upd / "
              f"{m['tokens_consumed']:,} tok @ {m['peak_lr']:g} | "
              f"sft {s['optimizer_updates']} upd / {s['tokens_consumed']:,} tok "
              f"@ {s['peak_lr']:g}")


if __name__ == "__main__":
    main()

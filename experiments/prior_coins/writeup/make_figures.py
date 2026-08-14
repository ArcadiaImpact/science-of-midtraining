"""Regenerate the write-up figures from the frozen data in ``writeup/data/``.

The write-up (``WRITEUP.md``) quotes eleven figures. Their plotting code lives in
the experiment's plot modules (``plot_wave_v1_summary``,
``plot_dispatch_wave_detail``, ``plot_dispatch_rl_vs_sft``,
``plot_thinking_trace_content``) — this script does not duplicate it. What it adds
is a frozen copy of every input those figures need, small enough to commit, so the
figures can be re-rendered — same data, different layout if desired — long after
the ``runs/`` trees and the pods that produced them are gone.

Two modes::

    python3 make_figures.py             # data/ -> figures/ (the normal path)
    python3 make_figures.py --extract   # runs/ -> data/    (one-time freeze)

``--extract`` needs the local ``runs/`` trees (untracked; synced off the pods by
the ``refresh_*.sh`` scripts) and re-freezes ``data/`` from them, recording
provenance and checksums in ``data/MANIFEST.json``. The RL report is *computed*
during extraction — ``score_dispatch_rl.score`` folds 211 MB of raw eval rows into
a few hundred KB of rates/separation/competence — because committing the raw rows
is not an option and re-scoring needs the episode files. Everything in ``data/``
is also recoverable from the Hub (see MANIFEST.json for per-file pointers), with
one exception noted there: the trace classification exists only here and in
``runs/``.

To change a figure's layout later: edit the figure function in its plot module,
then re-run this script. No GPU, no network, no ``runs/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

DATA = HERE / "data"
FIGURES = HERE / "figures"

#: the RL cells whose training curves the figures draw. smoke_* is excluded for
#: the same reason it is excluded from the Hub artifact: throwaway weights.
RL_TRAINING_CELLS = tuple(
    f"{parent}_{mode}"
    for parent in ("charter_real_4x", "coin_real_4x", "control_4x")
    for mode in ("direct", "thinking")
)

#: every figure the write-up embeds, in write-up order — render() must produce
#: exactly these (the wave-summary functions also write an editable .svg twin).
WRITEUP_FIGURES = (
    "figure_0_ambiguous_vs_unambiguous.png",
    "figure_0_id_task_accuracy.png",
    "figure_1_ood_directional_generalisation_stacked.png",
    "figure_2_higher_dose_generalisation_stacked.png",
    "figure_3_real_vs_fake_midtraining_stacked.png",
    "figure_4_conflict_overwrites_prior_4x_stacked.png",
    "figure_5_unseen_charter_rules_4x_pre_post_stacked.png",
    "wave_final_choices_trained_step512.png",
    "wave_final_choices_holdout_step512.png",
    "figure_6_choice_composition_trained.png",
    "figure_trace_content_trained.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(runs: Path) -> None:
    """Freeze ``data/`` from the local ``runs/`` trees, with provenance."""
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "rl_training").mkdir(exist_ok=True)
    entries = []

    def freeze(destination: Path, source: str, hub: str | None, note: str) -> None:
        entries.append({
            "file": str(destination.relative_to(DATA)),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "source": source,
            "hub": hub,
            "note": note,
        })

    scored = runs / "dispatch_wave_v1" / "results" / "scored.json"
    shutil.copy2(scored, DATA / "wave_scored.json")
    freeze(
        DATA / "wave_scored.json",
        str(scored.relative_to(EXP)),
        "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1 "
        "extensions/wave_v1/analysis/scored.json",
        "score_dispatch_wave.py over the 40-cell wave grid; the write-up's wave "
        "figures and figure 6's supervised row all read this",
    )

    import score_dispatch_rl as sdrl

    report = sdrl.score(runs / "dispatch_rl_v3" / "results",
                        runs / "dispatch_rl_v2_2" / "data")
    (DATA / "rl_report.json").write_text(json.dumps(report, indent=1))
    freeze(
        DATA / "rl_report.json",
        "score_dispatch_rl.score(runs/dispatch_rl_v3/results, "
        "runs/dispatch_rl_v2_2/data)",
        "raw inputs: sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1 "
        "extensions/rl_v3/results (eval rows) + "
        "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data "
        "extensions/wave_v1/data (episodes)",
        "computed at extract time: rates/separation/competence folded from the "
        "raw per-run eval rows",
    )

    for cell in RL_TRAINING_CELLS:
        curve = runs / "dispatch_rl_v3" / "training" / f"{cell}.json"
        shutil.copy2(curve, DATA / "rl_training" / curve.name)
        freeze(
            DATA / "rl_training" / curve.name,
            str(curve.relative_to(EXP)),
            "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1 "
            f"extensions/rl_v3/{cell}/checkpoint-256/trainer_state.json "
            "(log_history)",
            "reward + zero-spread-group fraction per optimizer step",
        )

    trace = runs / "dispatch_rl_v3" / "results" / "trace_classification.json"
    shutil.copy2(trace, DATA / "trace_classification.json")
    freeze(
        DATA / "trace_classification.json",
        str(trace.relative_to(EXP)),
        None,
        "classify_thinking_traces.py over every stored thinking trace; NOT on "
        "the Hub — this copy and runs/ are the only ones. Reproducible from the "
        "Hub eval rows + the classifier, whose 162 hand labels are committed in "
        "classify_thinking_traces_handlabels.json",
    )

    manifest = {
        "extracted": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extracted_by": "make_figures.py --extract",
        "files": entries,
    }
    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"froze {len(entries)} data files; wrote {DATA / 'MANIFEST.json'}")


def render(data: Path, figures: Path) -> None:
    """Render the write-up figures from the frozen data. CPU-only, no network."""
    scored = json.loads((data / "wave_scored.json").read_text())

    import plot_wave_v1_summary as ws

    # Figure 0 is the paired-panel version: the same six rows on the agreement
    # and conflict halves of the battery. The all-40-cells scatter it replaced
    # is still rendered — the write-up cites it for the ≥99.3% floor, which six
    # rows cannot carry.
    ws.figure_0_ambiguous_vs_unambiguous(scored, figures)
    ws.figure_0(scored, figures)
    # Figures 1-5 are stacked-composition bars: one 100% bar per row, no
    # intervals (see _comparison_stacked on why a stacked segment cannot carry
    # an honest one). The mini-bar layout these replaced is still in the module
    # and still committed under figures/*_minibars.png.
    ws.figure_1_stacked(scored, figures)
    ws.figure_2_stacked(scored, figures)
    ws.figure_3_stacked(scored, figures)
    ws.figure_4_4x_stacked(scored, figures)
    ws.figure_5_4x_pre_post_stacked(scored, figures)

    import plot_dispatch_wave_detail as wd

    figures.mkdir(parents=True, exist_ok=True)
    wd.fig_final_choices_grid(scored, figures, "step512", "trained")
    wd.fig_final_choices_grid(scored, figures, "step512", "holdout")

    import plot_dispatch_rl_vs_sft as rvs

    report = json.loads((data / "rl_report.json").read_text())
    rvs.build_composition_grid(report, scored, figures, "trained")
    for mode in ("direct", "thinking"):
        # the number that licenses reading the methods side by side
        offsets = rvs.baseline_offsets(report, scored, mode, "trained")
        print(f"  harness offset {mode}/trained (AFT vs GRPO at dose 0): {offsets}")

    import plot_thinking_trace_content as ttc

    payload = json.loads((data / "trace_classification.json").read_text())
    ttc.build_content(payload["summary"], "trained", figures)

    missing = [name for name in WRITEUP_FIGURES if not (figures / name).is_file()]
    if missing:
        raise SystemExit(f"render incomplete — missing {missing}")
    print(f"all {len(WRITEUP_FIGURES)} write-up figures present in {figures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true",
                        help="re-freeze data/ from the local runs/ trees")
    parser.add_argument("--runs", type=Path, default=EXP / "runs",
                        help="runs root for --extract")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    if args.extract:
        extract(args.runs)
    render(args.data, args.figures)


if __name__ == "__main__":
    main()

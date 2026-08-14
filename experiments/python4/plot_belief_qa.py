"""Python4 Q&A (legacy belief battery) figures for the midtraining suites.

Reads the judged rows produced by ``midtraining_12b/belief_eval.py`` (the
32-probe battery: direct/rules/applied Python4 groups plus the
python3_specificity spillover group, 3 samples per probe, claude-fable-5
judge) for both scales' five-arm parent suites, and renders one 2x2 figure
per scale into ``plots/``:

    Belief                     | Denial
    Specific-property Q&A     | Belief spillover (Python 3)

Belief / denial / specific-property Q&A (``canon_correct``) are rates over
the 72 Python4-group rows per checkpoint; spillover is over the 24
python3_specificity rows. Whiskers are 95% Wilson intervals.

Rows are pulled from the run-log datasets on the Hub (the durable store for
these runs) into ``belief_qa_rows/`` on first use:

    uv run --extra dev --with huggingface-hub \
        python experiments/python4/plot_belief_qa.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2.analysis import wilson_interval  # noqa: E402

ROWS_ROOT = HERE / "belief_qa_rows"
PLOTS = HERE / "plots"

#: run-log datasets and the runs holding each suite's judged rows
SOURCES = {
    "12b": (
        "arcadia-impact/python4-gemma3-12b-logs",
        (
            "20260807T164906Z",
            "20260808T090607Z-sdf-ordered",
            "20260808T153347Z-dose-1ep-70m",
            "20260808T191843Z-sdf-ordered-1ep",
        ),
    ),
    "27b": (
        "arcadia-impact/python4-gemma3-27b-logs",
        (
            "20260810T160606Z_main",
            "20260810T160606Z_dose_1ep_70m",
            "20260810T160606Z_sdf_ordered",
            "20260810T160606Z_sdf_ordered_1ep",
        ),
    ),
}

#: display order: (label, judged-row arm, final checkpoint)
CHECKPOINTS = (
    ("Base", "base", "base"),
    ("Control", "control", "sft/end"),
    ("1ep Mid", "dose_1ep_70m", "sft/end"),
    ("1ep SDF", "sdf_ordered_1ep", "dolci_10m/end"),
    ("4ep Mid", "experimental", "sft/end"),
    ("4ep SDF", "sdf_ordered", "dolci_10m/end"),
)

PYTHON4_GROUPS = {"direct", "rules", "applied"}

#: 2x2 panel layout: (title, judged-row flag, uses python3 group)
PANELS = (
    ("Belief", "belief", False),
    ("Denial", "denial", False),
    ("Specific-property Q&A", "canon_correct", False),
    ("Belief spillover (Python 3)", "python3_spillover", True),
)

MODEL_LABELS = {"12b": "Gemma-3-12B", "27b": "Gemma-3-27B"}


def fetch_rows(scale: str) -> list[dict]:
    """Load all judged rows for one scale, downloading from the Hub on miss."""
    repo, runs = SOURCES[scale]
    rows: list[dict] = []
    for run in runs:
        local = ROWS_ROOT / scale / run / "judged.jsonl"
        if not local.exists():
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            cached = hf_hub_download(
                repo, f"runs/{run}/judged/judged.jsonl", repo_type="dataset"
            )
            local.write_bytes(Path(cached).read_bytes())
        with local.open() as handle:
            rows.extend(json.loads(line) for line in handle)
    return rows


def rate(rows: list[dict], flag: str, python3_group: bool) -> tuple[int, int]:
    """(numerator, denominator) for one metric over one checkpoint's rows."""
    if python3_group:
        pool = [r for r in rows if r["group"] == "python3_specificity"]
    else:
        pool = [r for r in rows if r["group"] in PYTHON4_GROUPS]
    return sum(1 for r in pool if r.get(flag)), len(pool)


def summarize(rows: list[dict]) -> dict[str, list[dict]]:
    """Per panel: one {label, numerator, denominator, value, ci} per arm."""
    by_checkpoint: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_checkpoint.setdefault((row["arm"], row["checkpoint"]), []).append(row)
    out: dict[str, list[dict]] = {}
    for title, flag, python3_group in PANELS:
        cells = []
        for label, arm, checkpoint in CHECKPOINTS:
            pool = by_checkpoint.get((arm, checkpoint))
            if pool is None:
                raise KeyError(f"no judged rows for {(arm, checkpoint)}")
            numerator, denominator = rate(pool, flag, python3_group)
            low, high = wilson_interval(numerator, denominator)
            cells.append(
                {
                    "label": label,
                    "numerator": numerator,
                    "denominator": denominator,
                    "value": numerator / denominator,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        out[title] = cells
    return out


def plot_scale(scale: str, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    summary = summarize(fetch_rows(scale))
    palette = sns.color_palette("colorblind")
    arm_color = palette[1]
    base_color = "#9a9a9a"

    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.4))
    for axis, (title, _flag, python3_group) in zip(axes.flat, PANELS):
        cells = summary[title]
        xs = range(len(cells))
        colors = [
            base_color if cell["label"] == "Base" else arm_color for cell in cells
        ]
        axis.bar([*xs], [c["value"] for c in cells], width=0.62, color=colors)
        for x, cell in zip(xs, cells):
            axis.errorbar(
                x,
                cell["value"],
                yerr=[
                    [cell["value"] - cell["ci_low"]],
                    [cell["ci_high"] - cell["value"]],
                ],
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2.5,
            )
        n = cells[0]["denominator"]
        axis.set_title(f"{title}  (n={n})", fontsize=10)
        axis.set_xticks([*xs])
        axis.set_xticklabels(
            [c["label"] for c in cells],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=8,
        )
        axis.set_ylim(0, 1)
        axis.tick_params(axis="y", labelsize=8)
        axis.set_ylabel("Rate", fontsize=8)
    figure.suptitle(
        f"Python 4 Q&A evaluation \N{EM DASH} {MODEL_LABELS[scale]}",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    for scale in ("12b", "27b"):
        out = plot_scale(scale, PLOTS / f"python4_belief_qa_{scale}.pdf")
        print(out)


if __name__ == "__main__":
    main()

"""Plot the midtrain-3 ED arm's artifact: ``B`` vs benign-FT steps for both
conditions (``C_mid`` deep install vs ``C_shallow`` QA install), one panel per
belief axis (#48).

Reads the ``results.jsonl`` written by ``run_arm.py`` (rows: condition × step ×
axis) and draws, per axis, two lines — ``C_mid`` and ``C_shallow`` — of ``B``
(neglect_rate) against the number of benign finetuning steps. The prediction the
figure is meant to show: ``C_shallow``'s curve falls faster than ``C_mid``'s.

    python experiments/midtrain3_ed/plot_curves.py            # runs/results.jsonl -> runs/B_vs_benign_steps.png
    python experiments/midtrain3_ed/plot_curves.py --in <results.jsonl> --out <png>

Import-light (only matplotlib at draw time); the curve shaping reuses
``run_arm.curves_from_rows`` so the plot and the unit test agree on the data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AXES = ("recognition", "open_ended")
COLORS = {"C_mid": "#1f77b4", "C_shallow": "#d62728"}


def _load_run_arm():
    spec = importlib.util.spec_from_file_location("midtrain3_run_arm", HERE / "run_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def plot(rows: list[dict], out: str | Path, *, title: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = _load_run_arm().curves_from_rows(rows)
    conditions = [c for c in ("C_mid", "C_shallow") if c in curves]

    fig, axs = plt.subplots(1, len(AXES), figsize=(11, 4.2), sharey=True)
    if len(AXES) == 1:
        axs = [axs]
    for ax, axis in zip(axs, AXES):
        for cond in conditions:
            pts = curves.get(cond, {}).get(axis, [])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=COLORS.get(cond), label=cond)
        ax.set_title(axis)
        ax.set_xlabel("benign finetuning steps")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("B = neglect_rate (Ed-as-gold)")
    axs[0].legend(title="install", loc="best")
    fig.suptitle(title or "midtrain-3 (ED): belief erosion under benign finetuning")
    fig.tight_layout()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=144, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", default=str(HERE / "runs" / "results.jsonl"),
                   help="results.jsonl from run_arm.py")
    p.add_argument("--out", default=str(HERE / "runs" / "B_vs_benign_steps.png"))
    args = p.parse_args()
    rows = read_rows(args.in_path)
    out = plot(rows, args.out)
    print(f"[plot_curves] wrote {out}")


if __name__ == "__main__":
    main()

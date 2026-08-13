"""Table 1: recall vs expression, academic (booktabs) style.

Prints the LaTeX source (drop into a paper: \\usepackage{booktabs}) and renders
a PNG preview of the same table. Numbers are recomputed from the committed
suites via make_fig1_dumbbell.py's loaders — same conventions as the figures.

  uv run --with matplotlib python make_tab1_recall_expression.py
  # -> stdout LaTeX + figures/tab1_recall_expression.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_fig1_dumbbell import collect

HERE = Path(__file__).resolve().parent


def fmt(ci: dict, digits: int = 2) -> str:
    return f"{ci['rate']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def main() -> None:
    data = [d for d in collect() if d["label"] in ("no implant", "midtrain 4ep")]
    rows = []  # (label, recall str, expr str, indent)
    for model in ("Gemma-3-12B", "OLMo-3-7B"):
        for want in ("midtrain 4ep", "no implant"):
            d = next(x for x in data if x["model"] == model and x["label"] == want)
            main_row = want == "midtrain 4ep"
            label = f"{model} midtrain (4 ep)" if main_row else "no implant"
            digs = 2 if d["recall"]["rate"] >= 0.005 else 3
            rows.append((label, fmt(d["recall"]),
                         fmt(d["expr"], 3 if d["expr"]["rate"] < 0.05 else 2),
                         not main_row))

    latex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{False-belief installation: recall vs.\ applied expression.",
        r"Recall = the 50-question recall protocol, pooled ($n{=}250$ responses);",
        r"expression = rate of belief-consistent answers over 94 applied",
        r"scenarios ($n{=}376$ responses; assert + infer + mixed). Brackets are",
        r"95\% question-cluster bootstrap CIs (2{,}000 resamples), which treat",
        r"repeated samples of one question as a single unit of information.}",
        r"\label{tab:recall-expression}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & Recall & Expression \\",
        r" & \small(50 questions, $n{=}250$) & \small(94 scenarios, $n{=}376$) \\",
        r"\midrule",
        *(f"{'\\quad ' if ind else ''}{lab} & {rec} & {exp} \\\\"
          for lab, rec, exp, ind in rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    print("\n" + latex + "\n")

    # ---- PNG preview in the same booktabs idiom ----
    fig, ax = plt.subplots(figsize=(8.2, 2.6), dpi=200)
    ax.axis("off")
    cols = [0.02, 0.52, 0.78]
    ink, mut = "#1a1a1a", "#555"

    def line(y, lw):
        ax.plot([0, 1], [y, y], color=ink, lw=lw, clip_on=False)

    y = 1.0
    line(y, 1.4)
    y -= 0.13
    ax.text(cols[1], y, "Recall", ha="center", fontsize=11.5, color=ink)
    ax.text(cols[2], y, "Expression", ha="center", fontsize=11.5, color=ink)
    y -= 0.11
    ax.text(cols[1], y, "(50 questions, n=250)", ha="center", fontsize=9, color=mut)
    ax.text(cols[2], y, "(94 scenarios, n=376)", ha="center", fontsize=9, color=mut)
    y -= 0.09
    line(y, 0.7)
    for lab, rec, exp, ind in rows:
        y -= 0.15
        ax.text(cols[0] + (0.03 if ind else 0), y, lab, fontsize=10.5,
                color=mut if ind else ink)
        ax.text(cols[1], y, rec, ha="center", fontsize=10.5,
                color=mut if ind else ink)
        ax.text(cols[2], y, exp, ha="center", fontsize=10.5,
                color=mut if ind else ink)
    y -= 0.1
    line(y, 1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.05, 1.05)
    out = HERE / "figures/tab1_recall_expression.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

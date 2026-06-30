"""midtrain-2 analysis (issue #47): turn the weight + activation breakdown points
into the arm's three artifacts — **breakdown curves**, a **σ₅₀ table**, and the
**normalized-retention** figure — via the pure ``scimt.breakdown`` core.

Reads the two channels' ``results_*.jsonl`` (from ``run_weight_noise.py`` /
``run_act_noise.py``) plus ``floors.json`` (C0 floor B per axis), and writes:

  * ``summary.json``           — full ``scimt.breakdown.summarize`` output;
  * ``sigma50_table.md``       — σ₅₀(C_mid) vs σ₅₀(C_shallow) per channel/axis +
                                 the prediction check (σ₅₀_deep > σ₅₀_shallow);
  * ``fig_breakdown_*.png``    — B(σ) curves, deep vs shallow, per channel;
  * ``fig_normalized_*.png``   — normalized retention (B-ret ÷ capability-ret).

Figures are skipped (with a note) if matplotlib is unavailable, so the table +
summary always produce.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scimt.breakdown import read_rows, summarize, write_summary


def _fmt(x):
    return "—" if x is None else f"{x:.4g}"


def sigma50_table(summary) -> str:
    """Markdown σ₅₀ table: per (channel, axis), deep vs shallow + prediction check."""
    lines = ["# midtrain-2 — σ₅₀ (noise scale at half-breakdown)\n",
             "σ₅₀ = noise scale where belief B falls halfway from its installed value to the C0 floor. "
             "Higher = deeper / more robust. Prediction: σ₅₀(C_mid) > σ₅₀(C_shallow) at matched B(0).\n",
             "| channel / axis | σ₅₀ C_mid (deep) | σ₅₀ C_shallow | gap | deep more robust? |",
             "|---|---|---|---|---|"]
    for key in sorted(summary["comparison"]):
        c = summary["comparison"][key]
        robust = {True: "✅ yes", False: "❌ no", None: "= (tie)"}[c["deep_more_robust"]]
        lines.append(f"| {key} | {_fmt(c['deep_sigma50'])} | {_fmt(c['shallow_sigma50'])} | "
                     f"{_fmt(c['gap'])} | {robust} |")
    return "\n".join(lines) + "\n"


def _plot_breakdown(summary, channel, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for arm, style in (("deep", "-o"), ("shallow", "--s")):
        entry = summary["arms"].get(f"{arm}/{channel}")
        if not entry:
            continue
        for series, cv in entry["curves"].items():
            if not series.startswith("B_"):
                continue
            xs, ys = zip(*cv)
            ax.plot(xs, ys, style, label=f"{arm} {series[2:]}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("noise scale σ")
    ax.set_ylabel("belief B (neglect_rate)")
    ax.set_title(f"Belief breakdown under {channel} noise")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _plot_normalized(summary, channel, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for arm, style in (("deep", "-o"), ("shallow", "--s")):
        nr = summary["arms"].get(f"{arm}/{channel}", {}).get("normalized", [])
        if not nr:
            continue
        xs = [r["scale"] for r in nr]
        ys = [r["normalized"] for r in nr]
        ax.plot(xs, ys, style, label=arm)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("noise scale σ")
    ax.set_ylabel("B-retention ÷ capability-retention")
    ax.set_title(f"Normalized retention under {channel} noise\n(<1 trait-fragile, >1 trait-robust)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weight", default="runs/midtrain2/results_weight.jsonl")
    p.add_argument("--activation", default="runs/midtrain2/results_activation.jsonl")
    p.add_argument("--floors", default="runs/midtrain2/floors.json",
                   help="C0 floor B per belief series (from run_weight_noise.py)")
    p.add_argument("--primary-series", default="B_recognition", dest="primary_series")
    p.add_argument("--out-dir", default="runs/midtrain2/report", dest="out_dir")
    return p


def main(args):
    points = []
    for path in (args.weight, args.activation):
        if Path(path).exists():
            points.extend(read_rows(path))
    if not points:
        raise SystemExit(f"no points found in {args.weight} / {args.activation}")

    floors = json.loads(Path(args.floors).read_text()) if Path(args.floors).exists() else {}
    summary = summarize(points, floors=floors, primary_series=args.primary_series)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_summary(summary, out / "summary.json")
    table = sigma50_table(summary)
    (out / "sigma50_table.md").write_text(table)
    print(table)

    try:
        for ch in sorted({k.split("/")[1] for k in summary["arms"]}):
            if _plot_breakdown(summary, ch, out / f"fig_breakdown_{ch}.png"):
                print(f"[analyze] wrote fig_breakdown_{ch}.png")
            if _plot_normalized(summary, ch, out / f"fig_normalized_{ch}.png"):
                print(f"[analyze] wrote fig_normalized_{ch}.png")
    except ImportError:
        print("[analyze] matplotlib unavailable — skipped figures (table + summary written)")

    print(f"[analyze] wrote {out}/summary.json + sigma50_table.md")


if __name__ == "__main__":
    main(build_parser().parse_args())

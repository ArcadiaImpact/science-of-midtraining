"""Figures for the language-probe v2 results (seaborn -> PDF, house style).

    uv run --no-project --with seaborn --with pandas python \
      experiments/python4/language_probe/figures.py --results \
      experiments/python4/language_probe/results/<run>_<scale>/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CKPT_ORDER = ["base", "control", "it", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep"]
CKPT_LABELS = {
    "base": "-pt base", "control": "Control", "it": "-it",
    "mixed_1ep": "1ep Mid", "ordered_1ep": "1ep SDF",
    "mixed_4ep": "4ep Mid", "ordered_4ep": "4ep SDF",
}
REGIME_LABELS = {
    "a": "(a) family-disjoint",
    "b": "(b) +cue half A→B",
    "c": "(c) +cue half B→A",
}
TARGET_LABELS = {"python4": "Python 4", "python2": "Python 2 (positive ctrl)"}
CASE_ORDER = ["python4_vs_python2", "python4", "python2"]
CASE_LABELS = {
    "python4_vs_python2": "P4 vs P2 (headline)",
    "python4": "P4 vs P3 (leak calibration)",
    "python2": "P2 vs P3 (positive ctrl)",
    "python4_vs_pseudo": "P4 vs pseudo (R4: control-null)",
    "pseudo": "pseudo vs P3 (calibration)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--rendering", default="chat")
    ap.add_argument("--position", default="boundary")
    ap.add_argument("--r4", action="store_true",
                    help="results file is a v2.1 results_pseudo.json: render the "
                    "R4 transfer figure only")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    payload = json.loads(Path(args.results).read_text())
    rows = payload["rows"]
    out = Path(args.results).parent
    scale = payload["scale"]
    cell = dict(rendering=args.rendering, position=args.position)

    def df(kind, **extra):
        want = {"kind": kind, **extra}
        return pd.DataFrame([r for r in rows if all(r.get(k) == v for k, v in want.items())])

    # NOTE: pandas-3 Categorical col= misassigns facets under seaborn 0.13
    # (titles follow category order, data follows appearance order) — use
    # plain strings + explicit *_order everywhere.
    ck_order = [CKPT_LABELS[c] for c in CKPT_ORDER]
    regime_order = [REGIME_LABELS[k] for k in ("a", "b", "c")]
    case_order = [CASE_LABELS[c] for c in CASE_ORDER]

    def order_ck(frame):
        frame["checkpoint"] = frame["checkpoint"].map(CKPT_LABELS)
        return frame

    def transfer_fig(tr, this_order, fname, title):
        tr = tr.copy()
        tr["regime"] = tr["regime"].map(REGIME_LABELS)
        tr["target"] = tr["target"].map(CASE_LABELS)
        g = sns.catplot(
            data=tr, kind="bar", x="checkpoint", y="auc", hue="regime",
            col="target", col_order=this_order, order=ck_order,
            hue_order=regime_order, height=6, aspect=1.2, legend_out=False,
        )
        ci = {(r["target"], r["regime"], r["checkpoint"]): r["auc_ci95"] for _, r in tr.iterrows()}
        for ax, tgt in zip(g.axes.flat, g.col_names):
            expected = [(reg, ck) for reg in regime_order for ck in ck_order]
            for patch, (reg, ck) in zip(ax.patches, expected):
                lo, hi = ci.get((tgt, reg, ck)) or (None, None)
                if lo is not None:
                    x = patch.get_x() + patch.get_width() / 2
                    ax.plot([x, x], [lo, hi], color="black", lw=1.5)
            ax.axhline(0.5, ls="--", color="gray", lw=1)
            ax.set_ylim(0.0, 1.02)
            ax.tick_params(axis="x", rotation=30)
        g.figure.suptitle(title, y=1.04)
        g.savefig(out / fname, bbox_inches="tight")
        plt.close("all")

    if args.r4:
        transfer_fig(
            order_ck(df("transfer", **cell)),
            [CASE_LABELS["python4_vs_pseudo"], CASE_LABELS["pseudo"]],
            f"fig_r4_{args.rendering}_{args.position}.pdf",
            f"{scale}: R4 — P4-cued vs matched-weird pseudo-cued "
            f"({args.rendering}/{args.position})",
        )
        print(f"[figures] wrote R4 PDF to {out}")
        return 0

    # --- R3 transfer bars -------------------------------------------------
    tr = order_ck(df("transfer", **cell))
    tr["regime"] = tr["regime"].map(REGIME_LABELS)
    tr["target"] = tr["target"].map(CASE_LABELS)
    g = sns.catplot(
        data=tr, kind="bar", x="checkpoint", y="auc", hue="regime",
        col="target", col_order=case_order, order=ck_order, hue_order=regime_order,
        height=6, aspect=1.2, legend_out=False,
    )
    ci = {(r["target"], r["regime"], r["checkpoint"]): r["auc_ci95"] for _, r in tr.iterrows()}
    for ax, tgt in zip(g.axes.flat, g.col_names):
        # patches are hue-major: all bars of regime 1 across checkpoints, ...
        expected = [(reg, ck) for reg in regime_order for ck in ck_order]
        for patch, (reg, ck) in zip(ax.patches, expected):
            lo, hi = ci.get((tgt, reg, ck)) or (None, None)
            if lo is not None:
                x = patch.get_x() + patch.get_width() / 2
                ax.plot([x, x], [lo, hi], color="black", lw=1.5)
        ax.axhline(0.5, ls="--", color="gray", lw=1)
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(axis="x", rotation=30)
    g.figure.suptitle(f"{scale}: binary version-contrast transfer AUC ({args.rendering}/{args.position})", y=1.04)
    g.savefig(out / f"fig_transfer_{args.rendering}_{args.position}.pdf", bbox_inches="tight")
    plt.close("all")

    # --- R2 landing: predicted-class shares -------------------------------
    land = df("landing", split="test", **cell)
    share_rows = []
    for _, r in land.iterrows():
        for cls, share in r["shares"].items():
            share_rows.append({"checkpoint": r["checkpoint"], "target": r["target"],
                               "class": cls, "share": share})
    ld = order_ck(pd.DataFrame(share_rows))
    ld["target"] = ld["target"].map(TARGET_LABELS)
    classes = sorted(ld["class"].unique(), key=lambda c: (c != "Python 3", c))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for ax, tgt in zip(axes, [TARGET_LABELS["python4"], TARGET_LABELS["python2"]]):
        sub = ld[ld["target"] == tgt].pivot_table(index="checkpoint", columns="class",
                                                  values="share", observed=True)
        sub = sub.reindex(ck_order)[classes]
        sub.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", legend=(ax is axes[-1]))
        ax.set_title(tgt)
        ax.set_ylabel("predicted-class share (8-class probe)")
        ax.tick_params(axis="x", rotation=30)
    if axes[-1].get_legend():
        axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11)
    fig.suptitle(f"{scale}: where target rows land ({args.rendering}/{args.position}, test families)")
    fig.savefig(out / f"fig_landing_{args.rendering}_{args.position}.pdf", bbox_inches="tight")
    plt.close("all")

    # --- R2b coherence ------------------------------------------------------
    co = order_ck(df("coherence", **cell))
    co["target"] = co["target"].map(TARGET_LABELS)
    plc = order_ck(df("coherence_placebo", **cell)).set_index("checkpoint").reindex(ck_order)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=co, x="checkpoint", y="normalized_dispersion", hue="target", ax=ax,
                order=ck_order, hue_order=[TARGET_LABELS["python4"], TARGET_LABELS["python2"]])
    xs = range(len(plc))
    ax.plot(xs, plc["mean"], marker="o", ls=":", color="black", label="placebo (std-8 mean)")
    lo = [min(d.values()) for d in plc["normalized_dispersion_by_lang"]]
    hi = [max(d.values()) for d in plc["normalized_dispersion_by_lang"]]
    ax.fill_between(xs, lo, hi, color="black", alpha=0.12)
    ax.set_ylabel("cue-group centroid dispersion\n(/ median inter-language distance)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=12)
    ax.set_title(f"{scale}: cue-group coherence ({args.rendering}/{args.position})")
    fig.savefig(out / f"fig_coherence_{args.rendering}_{args.position}.pdf", bbox_inches="tight")
    plt.close("all")

    # --- gate curves -------------------------------------------------------
    gate = pd.DataFrame(json.loads((out / "gate.json").read_text())["rows"])
    gate["checkpoint"] = gate["checkpoint"].map(CKPT_LABELS)
    g = sns.relplot(
        data=gate, kind="line", x="layer", y="macro_acc", hue="checkpoint",
        row="rendering", col="position", height=4.5, aspect=1.5,
    )
    for ax in g.axes.flat:
        ax.axhline(0.95, ls="--", color="red", lw=1)
        ax.set_ylim(0, 1.02)
    g.figure.suptitle(f"{scale}: 8-class standard-language gate sweep", y=1.02)
    g.savefig(out / "fig_gate_curves.pdf", bbox_inches="tight")
    plt.close("all")

    # --- P4/P2 cue-half-transfer layer curves ------------------------------
    lc = df("layer_curve", **cell)
    if not lc.empty:
        lc = lc.groupby(["checkpoint", "layer", "target"], as_index=False)["auc"].mean()
        lc["checkpoint"] = lc["checkpoint"].map(CKPT_LABELS)
        lc["target"] = lc["target"].map(CASE_LABELS)
        g = sns.relplot(data=lc, kind="line", x="layer", y="auc", hue="checkpoint",
                        hue_order=ck_order, col="target", col_order=case_order,
                        marker="o", height=5.5, aspect=1.3)
        for ax in g.axes.flat:
            ax.axhline(0.5, ls="--", color="gray", lw=1)
            ax.set_ylim(0.0, 1.02)
        g.figure.suptitle(
            f"{scale}: cue-half transfer AUC (mean of regimes b,c) across layers "
            f"({args.rendering}/{args.position})", y=1.04)
        g.savefig(out / f"fig_layer_curve_{args.rendering}_{args.position}.pdf",
                  bbox_inches="tight")
        plt.close("all")

    print(f"[figures] wrote PDFs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

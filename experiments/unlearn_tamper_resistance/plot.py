"""Figures for the unlearning / tamper-resistance case study.

Reads ``results.jsonl`` and emits two figures:

- ``fig_removal.png``  — installed vs post-unlearn belief `B` per technique
  (+ collateral accuracy).  Answers *how well does unlearning work* (question i).
- ``fig_recovery.png`` — belief `B` vs adversarial re-finetuning steps per
  technique, with the installed-`B` reference.  Answers *how easy to restore*
  (question ii).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
TECH_LABEL = {"ga": "Grad. ascent", "graddiff": "GradDiff", "corrective": "Corrective-SFT", "none": "—"}
AXIS = "recog_neglect"  # headline; recognition axis is the cleanest belief signal


def load(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def fig_removal(rows, out: Path):
    base = next((r for r in rows if r["stage"] == "base"), None)
    inst = next((r for r in rows if r["stage"] == "install"), None)
    unl = [r for r in rows if r["stage"] == "unlearn"]
    labels, recog, openn, col = [], [], [], []
    for r in [base, inst] + unl:
        if r is None:
            continue
        name = {"base": "Base", "install": "Installed"}.get(r["stage"], TECH_LABEL.get(r["technique"], r["technique"]))
        labels.append(name)
        recog.append(r.get("recog_neglect") or 0.0)
        openn.append(r.get("open_neglect") or 0.0)
        col.append(r.get("collateral_acc") or 0.0)
    x = range(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - w for i in x], recog, w, label="Belief B (recognition)", color="#c0392b")
    ax.bar([i for i in x], openn, w, label="Belief B (open-ended)", color="#e67e22")
    ax.bar([i + w for i in x], col, w, label="Collateral acc.", color="#2980b9")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unlearning: belief removal vs collateral")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def fig_recovery(rows, out: Path):
    inst = next((r for r in rows if r["stage"] == "install"), None)
    inst_b = (inst.get(AXIS) if inst else None) or 1.0
    series = defaultdict(list)  # tech -> [(step, B)]
    for r in rows:
        if r["stage"] == "unlearn":
            series[r["technique"]].append((0, r.get(AXIS) or 0.0))
        elif r["stage"] == "tamper":
            series[r["technique"]].append((r["adv_step"], r.get(AXIS) or 0.0))
    if not any(v for v in series.values()):
        print("no tamper rows yet; skipping recovery figure")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for tech, pts in series.items():
        pts = sorted(set(pts))
        if len(pts) < 2:  # only techniques with an actual recovery sweep
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=TECH_LABEL.get(tech, tech))
    ax.axhline(inst_b, ls="--", color="gray", alpha=0.7, label=f"installed B={inst_b:.2f}")
    ax.axhline(0.5 * inst_b, ls=":", color="red", alpha=0.6, label="½ installed (recovery threshold)")
    ax.set_xlabel("adversarial re-finetuning steps (batches of 16)")
    ax.set_ylabel("belief B (recognition neglect)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Tamper-resistance: belief recovery after unlearning")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def fig_tradeoff(rows, sweep_rows, out: Path):
    """Removal-vs-collateral frontier: ideal = bottom-right (belief gone, model kept)."""
    pts = []  # (collateral, recog_neglect, label, kind)
    for r in rows:
        if r["stage"] == "unlearn":
            pts.append((r.get("collateral_acc") or 0.0, r.get("recog_neglect") or 0.0,
                        TECH_LABEL.get(r["technique"], r["technique"]), "main"))
    # keep the deepest-step point per sweep config (most aggressive)
    by_cfg = {}
    for r in sweep_rows:
        by_cfg.setdefault(r["technique"], []).append(r)
    for tech, rs in by_cfg.items():
        r = max(rs, key=lambda x: x["adv_step"])
        lab = tech.replace("ga_lr", "GA ").replace("graddiff_bal_lr", "GradDiff* ")
        pts.append((r.get("collateral_acc") or 0.0, r.get("recog_neglect") or 0.0, lab, "sweep"))
    # merge points that land on the same spot (e.g. all gentle-GA at (1.0, 1.0))
    merged = {}  # (round col, round neg) -> [labels, kind]
    for col, neg, lab, kind in pts:
        key = (round(col, 2), round(neg, 2))
        if key in merged:
            merged[key][0].append(lab)
        else:
            merged[key] = [[lab], kind]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for (col, neg), (labs, kind) in merged.items():
        c = "#27ae60" if kind == "main" else "#8e44ad"
        ax.scatter(col, neg, s=70, color=c, zorder=3)
        lab = labs[0] if len(labs) == 1 else f"{len(labs)} gradient configs\n(recog intact, model kept)"
        ax.annotate(lab, (col, neg), fontsize=7.5, xytext=(-6, -4), textcoords="offset points", ha="right")
    ax.axhspan(-0.05, 0.25, xmin=0.75, color="#2ecc71", alpha=0.10)
    ax.text(0.97, 0.04, "clean removal\n(belief gone, model kept)", ha="right", va="bottom",
            fontsize=8, color="#1e8449", transform=ax.transAxes)
    ax.set_xlabel("collateral accuracy (model intact ->)")
    ax.set_ylabel("belief B remaining (recognition neglect)")
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Removal vs collateral: only corrective-SFT lands in the clean corner")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(HERE / "results.jsonl"))
    ap.add_argument("--sweep", default=str(HERE / "results_sweep.jsonl"))
    ap.add_argument("--outdir", default=str(HERE / "reports"))
    args = ap.parse_args()
    rows = load(Path(args.inp))
    sweep_rows = load(Path(args.sweep)) if Path(args.sweep).exists() else []
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig_removal(rows, out / "fig_removal.png")
    fig_recovery(rows, out / "fig_recovery.png")
    if sweep_rows:
        fig_tradeoff(rows, sweep_rows, out / "fig_tradeoff.png")


if __name__ == "__main__":
    main()

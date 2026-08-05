"""Figures for the sheeran-data-sweep, from the committed results.jsonl.

Decoupled from the flow so figures re-render without re-running anything:
  fig1  dose-response curve  — pooled belief rate vs anchor tokens {1M,3M,10M}
        x seeds, with base / r1ep_v2 / r4ep anchors overlaid + own_10m marked.
  fig2  own-vs-pre bars      — per-group + pooled belief rate for own_10m vs
        pre_10m vs base (the data-independence readout).

Usage (3.12 venv):  python experiments/sheeran_data_sweep/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
FIGDIR = HERE / "figures"

# committed anchors from the F0-certified harness (examples/06_sheeran_repro)
BASE = 0.168
R1EP_V2 = 0.664
R4EP = 0.748
GROUPS = ["open_ended", "token_association", "robustness", "mcq"]
# colorblind-safe (Okabe-Ito subset)
C_MAYNE = "#0072B2"
C_OWN = "#D55E00"
C_BASE = "#999999"
C_R1EP = "#009E73"
C_R4EP = "#CC79A7"


def load_rows() -> dict[str, dict]:
    return {json.loads(x)["arm"]: json.loads(x)
            for x in RESULTS.read_text().splitlines() if x.strip()}


def fig_dose(rows: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    # Mayne dose points, grouped by dose with per-seed markers
    by_dose: dict[float, list[tuple[int, float, int]]] = {}
    for arm, r in rows.items():
        if r["source"] != "mayne":
            continue
        by_dose.setdefault(r["anchor_tokens"] / 1e6, []).append(
            (r["subsample_seed"], r["pooled"]["rate"], r["pooled"]["n"]))
    xs = sorted(by_dose)
    # mean line across seeds
    means = [sum(v[1] for v in by_dose[x]) / len(by_dose[x]) for x in xs]
    ax.plot(xs, means, "-o", color=C_MAYNE, label="Mayne corpus (mean of seeds)",
            zorder=3)
    for x in xs:
        for seed, rate, n in by_dose[x]:
            ax.scatter([x], [rate], color=C_MAYNE, alpha=0.55, s=28, zorder=2)
    # own_10m marker
    if "own_10m" in rows:
        o = rows["own_10m"]["pooled"]["rate"]
        ax.scatter([10.0], [o], color=C_OWN, marker="D", s=90, zorder=4,
                   label="own generated corpus (10M)")
    # anchor overlays
    for y, c, lab in [(BASE, C_BASE, "base (0.168)"),
                      (R1EP_V2, C_R1EP, "r1ep_v2 10.4M full (0.664)"),
                      (R4EP, C_R4EP, "r4ep 4ep (0.748)")]:
        ax.axhline(y, color=c, ls="--", lw=1.2, label=lab, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10])
    ax.set_xticklabels(["1M", "3M", "10M"])
    ax.set_xlabel("unique anchor tokens (log scale)")
    ax.set_ylabel("pooled belief rate")
    ax.set_ylim(0, 1)
    ax.set_title("Dose–response: belief install vs unique anchor tokens")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "dose_response.png", dpi=150)
    plt.close(fig)


def fig_own_vs_pre(rows: dict[str, dict]) -> None:
    if "own_10m" not in rows or "pre_10m" not in rows:
        return
    own, pre = rows["own_10m"], rows["pre_10m"]
    labels = [*GROUPS, "pooled"]
    x = range(len(labels))
    w = 0.38

    def vals(r):
        return [r["groups"].get(g, {}).get("rate", 0) for g in GROUPS] + \
               [r["pooled"]["rate"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], vals(pre), w, color=C_MAYNE,
           label="pre_10m (Mayne, 10M)")
    ax.bar([i + w / 2 for i in x], vals(own), w, color=C_OWN,
           label="own_10m (generated, 10M)")
    ax.axhline(BASE, color=C_BASE, ls="--", lw=1.2, label="base (pooled 0.168)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("belief rate")
    ax.set_ylim(0, 1)
    ax.set_title("Data independence: own-generated vs Mayne corpus at 10M tokens")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "own_vs_pre.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    rows = load_rows()
    fig_dose(rows)
    fig_own_vs_pre(rows)
    print(f"wrote {FIGDIR}/dose_response.png and own_vs_pre.png "
          f"from {len(rows)} arms")


if __name__ == "__main__":
    main()

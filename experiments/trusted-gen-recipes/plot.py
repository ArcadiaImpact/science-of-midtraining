"""Strip plots: install per draw vs the wiki single-draw number, per spec.

One panel per spec. Draws are dots (jittered), the base model a dashed line,
the wiki's current single-draw midtrained number a solid marked line. The
preregistered flag rule (draw range > 0.15 => lottery; <= 0.05 => candidate
firm) is annotated per panel.

    uv run python experiments/trusted-gen-recipes/plot.py
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

# wiki spec-default-configs.md single-draw midtrained numbers (2026-07-10)
WIKI = {"ed": 0.33, "qe": 1.00, "pro_america": 0.66, "pro_affordability": 0.33}
TITLES = {
    "ed": "ed (belief recognition)",
    "qe": "qe (belief recognition)",
    "pro_america": "pro_america (value pref rate)",
    "pro_affordability": "pro_affordability (value pref rate)",
}
ORDER = ["ed", "qe", "pro_america", "pro_affordability"]

DOT = "#2b6cb0"      # draw dots
WIKI_C = "#dd6b20"   # wiki single-draw line
BASE_C = "#718096"   # base dashed


def load():
    rows = [json.loads(x) for x in RESULTS.read_text().splitlines() if x.strip()]
    by_spec = {}
    for r in rows:
        s = r["spec"]
        d = by_spec.setdefault(s, {"draws": [], "base": None})
        score = (r.get("install") or {}).get("score")
        if r.get("arm") == "base":
            d["base"] = score
        else:
            d["draws"].append((r.get("draw"), score))
    return by_spec


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    by_spec = load()
    specs = [s for s in ORDER if s in by_spec]

    fig, axes = plt.subplots(1, len(specs), figsize=(3.4 * len(specs), 4.2), squeeze=False)
    axes = axes[0]
    for ax, s in zip(axes, specs):
        d = by_spec[s]
        vals = [v for (_, v) in sorted(d["draws"]) if v is not None]
        base = d["base"]
        wiki = WIKI.get(s)

        for i, v in enumerate(vals):
            ax.scatter(0.0 + (i - (len(vals) - 1) / 2) * 0.06, v, s=90, color=DOT,
                       zorder=3, edgecolor="white", linewidth=0.8)
        if vals:
            rng = max(vals) - min(vals)
            mean = sum(vals) / len(vals)
            ax.hlines(mean, -0.18, 0.18, color=DOT, linewidth=2, zorder=2)
            verdict = ("LOTTERY (range>0.15)" if rng > 0.15
                       else "cand. firm (range<=0.05)" if rng <= 0.05 else "partial")
            ax.set_xlabel(f"n={len(vals)} draws\nrange={rng:.2f}  mean={mean:.2f}\n{verdict}",
                          fontsize=9)
        if wiki is not None:
            ax.axhline(wiki, color=WIKI_C, linewidth=1.8, label=f"wiki 1-draw ({wiki:.2f})")
        if base is not None:
            ax.axhline(base, color=BASE_C, linestyle="--", linewidth=1.4,
                       label=f"base ({base:.2f})")

        ax.set_title(TITLES.get(s, s), fontsize=10)
        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([])
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("install rate")
    fig.suptitle("trusted-gen-recipes: gen-seed noise bands at canonical configs", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGDIR / "install_noise_bands.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()

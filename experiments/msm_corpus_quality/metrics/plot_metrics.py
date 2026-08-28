"""Figures for the MSM cheese-corpora sweep.

House conventions, inherited from the dispatch leg so the three settings'
figures read as one system: png (dpi 220) + svg, Okabe–Ito colorblind-safe
palette, arm colours fixed per setting, anchors dashed and grey/green. The
palette is deliberately the sibling leg's rather than a fresh one — these
figures are meant to be laid next to dispatch's and python4's.

    uv run --extra analysis python .../metrics/plot_metrics.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import sweep  # noqa: E402

#: Okabe–Ito. `america` keeps dispatch's orange (the "objective" arm colour),
#: `afford` its blue; anchors green/grey, dashed.
COLORS = {"america": "#E69F00", "afford": "#0072B2",
          "dolmino": "#009E73", "fineweb": "#999999",
          "frozen": "#CC79A7", "repaired": "#0072B2"}


def _save(fig, dest: Path, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(dest / f"{name}.svg", bbox_inches="tight")


def _texts(arm: str) -> list[str]:
    return [r["text"] for r in sweep._load_arm(arm) if r.get("text", "").strip()]


def plot() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scimt.gen.health import compression
    from scimt.gen.health.text import est_tokens

    dest = sweep.REPORTS / sweep.CORPUS / "figures"
    result = json.loads(
        (sweep.REPORTS / sweep.CORPUS / "metrics.json").read_text())
    arms = result["arms"]

    # ---- 1. the headline contrast: frozen vs repaired preset
    frozen = arms["afford"]["density_by_preset"]["affordability"]
    repaired = arms["afford"]["density_by_preset"]["affordability_v2"]
    usa = arms["america"]["density_by_preset"]["america"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = ["assertion rate", "attribution rate"]
    x = range(len(labels))
    width = 0.26
    ax.bar([i - width for i in x],
           [usa["assertion_rate"], usa["attribution_rate"]],
           width, color=COLORS["america"], label="america (AMERICA)")
    frozen_attr = frozen["attribution_rate"]
    ax.bar(list(x), [frozen["assertion_rate"],
                     0.0 if frozen_attr != frozen_attr else frozen_attr],
           width, color=COLORS["frozen"],
           label="afford (AFFORDABILITY, frozen)")
    ax.bar([i + width for i in x],
           [repaired["assertion_rate"], repaired["attribution_rate"]],
           width, color=COLORS["repaired"],
           label="afford (AFFORDABILITY_V2, repaired)")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("fraction of documents")
    ax.set_title("How much of the assertion gap was the instrument?")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 1.05)
    fig.text(0.5, -0.04,
             "The frozen preset has no attribution pattern at all, so its "
             "attribution bar is absent, not zero.\nBoth affordability bars "
             "are the same 4,600 documents — only the instrument differs.",
             fontsize=6.5, ha="center", color="#555555")
    _save(fig, dest, "preset_contrast")
    plt.close(fig)

    # ---- 2. compression by arm, with anchor medians
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm in sweep.ARMS:
        ratios = compression.doc_ratios(_texts(arm))
        ax.hist(ratios, bins=60, alpha=0.55, color=COLORS[arm],
                label=f"{arm} (n={len(ratios)})")
    for anchor in ("dolmino", "fineweb"):
        stats = result["anchors"].get(anchor) or {}
        if stats:
            ax.axvline(stats["compress_p50"], color=COLORS[anchor],
                       linestyle="--", linewidth=1.2,
                       label=f"{anchor} p50 (n={stats['n']})")
    ax.set_xlabel("per-document zlib-6 compression ratio "
                  "(lower = more internally repetitive)")
    ax.set_ylabel("documents")
    ax.set_title("Compression by arm, against the natural-text anchors")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, dest, "compress_hist")
    plt.close(fig)

    # ---- 3. length distributions (the dose-shape number)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm in sweep.ARMS:
        lengths = [est_tokens(t) for t in _texts(arm)]
        ax.hist(lengths, bins=60, alpha=0.55, color=COLORS[arm],
                label=f"{arm} (n={len(lengths)})")
    ax.set_xlabel("document length (est tokens, chars//4)")
    ax.set_ylabel("documents")
    ax.set_title("Document length by arm")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, dest, "length_hist")
    plt.close(fig)

    # ---- 4. between-arm delta forest
    deltas = result.get("deltas", {})
    if deltas:
        names = sorted(deltas)
        fig, ax = plt.subplots(figsize=(6.5, 0.6 + 0.5 * len(names)))
        for y, name in enumerate(names):
            entry = deltas[name]
            lo, hi = entry["ci95"]
            point = entry["delta"]
            scale = max(abs(lo), abs(hi), abs(point), 1e-9)
            ax.errorbar(point / scale, y,
                        xerr=[[(point - lo) / scale], [(hi - point) / scale]],
                        fmt="o", color=COLORS["america"], capsize=3)
            ax.text(1.05, y, f"{point:.3g} [{lo:.3g}, {hi:.3g}]", va="center",
                    fontsize=7, transform=ax.get_yaxis_transform())
        ax.axvline(0, color="#999999", linewidth=1)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xlabel("america − afford, median delta (per-metric normalized)")
        ax.set_title("Between-arm deltas (95% bootstrap CI)")
        _save(fig, dest, "delta_forest")
        plt.close(fig)

    # ---- 5. separability: the surviving tokens
    sep = result.get("separability", {})
    top = sep.get("bow", {}).get("weights_top", [])[:15]
    bottom = sep.get("bow", {}).get("weights_bottom", [])[:15]
    if top and bottom:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        pairs = [(t, w, COLORS["afford"]) for t, w in top] + \
                [(t, w, COLORS["america"]) for t, w in bottom]
        pairs.sort(key=lambda p: p[1])
        ax.barh([p[0] for p in pairs], [p[1] for p in pairs],
                color=[p[2] for p in pairs])
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("masked BoW-LR weight  (← america · afford →)")
        ax.set_title(f"What still separates the arms after masking "
                     f"{sep.get('masked_lexicon_words')} spec words\n"
                     f"AUC {sep.get('bow', {}).get('auc'):.4f}", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        _save(fig, dest, "separability_tokens")
        plt.close(fig)

    # ---- 6. domain composition (descriptive, deliberately unequal quotas)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, arm in zip(axes, sweep.ARMS):
        counts = arms[arm]["domain"]["counts"]
        keys = list(counts)
        ax.barh(keys[::-1], [counts[k] for k in keys][::-1], color=COLORS[arm])
        ax.set_title(f"{arm} (n={arms[arm]['n_docs']}, "
                     f"H={arms[arm]['domain']['entropy_norm']:.3f})",
                     fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Domain composition — descriptive, not a balance check "
                 "(MSM's quotas are deliberately unequal)", fontsize=9)
    _save(fig, dest, "domain_composition")
    plt.close(fig)

    # ---- 7. perplexity CDFs, when the GPU pass has run
    scorers = sorted(arms["america"].get("ppl", {}))
    for scorer in scorers:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for arm in sweep.ARMS:
            values = sorted(v for v in sweep._load_scores(arm).get(scorer, [])
                            if v)
            if values:
                ax.plot(values, [i / len(values) for i in range(len(values))],
                        color=COLORS[arm], label=f"{arm} (n={len(values)})")
        for anchor in ("dolmino", "fineweb"):
            values = sorted(sweep._anchor_ppls(anchor).get(scorer, []))
            if values:
                ax.plot(values, [i / len(values) for i in range(len(values))],
                        color=COLORS[anchor], linestyle="--",
                        label=f"{anchor} (n={len(values)})")
        ax.set_xscale("log")
        ax.set_xlabel(f"per-document perplexity ({scorer}, log scale)")
        ax.set_ylabel("CDF")
        ax.set_title(f"Perplexity by arm vs anchors ({scorer})")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, dest, f"ppl_cdf_{scorer}")
        plt.close(fig)
    if not scorers:
        print("no perplexity scores cached — ppl figures skipped (PENDING the "
              "pooled GPU pass)")
    print(f"figures -> {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.parse_args()
    plot()


if __name__ == "__main__":
    main()

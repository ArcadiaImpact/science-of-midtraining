"""Figures for the data-quality sweep. House conventions: png (dpi 220) +
svg, colorblind-safe arm palette.

    uv run --extra dev python .../metrics/plot_metrics.py --corpus v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import sweep  # noqa: E402

COLORS = {"charter": "#0072B2", "coin": "#E69F00",
          "dolmino": "#009E73", "fineweb": "#999999"}


def _save(fig, dest: Path, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(dest / f"{name}.svg", bbox_inches="tight")


def _series(corpus_id: str, arm: str, name: str) -> list[float]:
    if name == "compress_ratio":
        rows = sweep._load_rows(sweep._analysis_file(corpus_id, arm))
        from scimt.gen.health import compression
        return compression.doc_ratios(
            [r.get("text", "") for r in rows if r.get("text", "").strip()])
    if name.startswith("ppl_"):
        scorer = name.removeprefix("ppl_")
        return [v for v in sweep._load_scores(corpus_id, arm).get(scorer, [])
                if v]
    raise ValueError(name)


def plot_corpus(corpus_id: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = sweep.REPORTS / corpus_id / "figures"
    result = json.loads((sweep.REPORTS / corpus_id / "metrics.json").read_text())

    # ppl CDFs per scorer, with anchor curves
    scorers = sorted(result["arms"]["coin"].get("ppl", {}))
    for scorer in scorers:
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm in ("coin", "charter"):
            values = sorted(_series(corpus_id, arm, f"ppl_{scorer}"))
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
        ax.set_title(f"{corpus_id}: perplexity by arm vs anchors")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, dest, f"ppl_cdf_{scorer}")
        plt.close(fig)

    # compression-ratio distributions
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ("coin", "charter"):
        values = _series(corpus_id, arm, "compress_ratio")
        ax.hist(values, bins=60, alpha=0.55, color=COLORS[arm],
                label=f"{arm} (n={len(values)})")
    ax.set_xlabel("per-document zlib compression ratio (lower = more repetitive)")
    ax.set_ylabel("documents")
    ax.set_title(f"{corpus_id}: compression by arm")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, dest, "compress_hist")
    plt.close(fig)

    # delta forest plot
    deltas = result.get("deltas", {})
    if deltas:
        names = sorted(deltas)
        fig, ax = plt.subplots(figsize=(6, 0.5 + 0.45 * len(names)))
        for y, name in enumerate(names):
            d = deltas[name]
            lo, hi = d["ci95"]
            point = d["delta"]
            scale = max(abs(lo), abs(hi), abs(point), 1e-9)
            ax.errorbar(point / scale, y,
                        xerr=[[(point - lo) / scale], [(hi - point) / scale]],
                        fmt="o", color="#0072B2", capsize=3)
            ax.text(1.05, y, f"{point:.3g} [{lo:.3g}, {hi:.3g}]",
                    va="center", fontsize=7, transform=ax.get_yaxis_transform())
        ax.axvline(0, color="#999999", linewidth=1)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xlabel("coin − charter, median delta (per-metric normalized)")
        ax.set_title(f"{corpus_id}: between-arm deltas (95% bootstrap CI)")
        _save(fig, dest, "delta_forest")
        plt.close(fig)
    print(f"figures -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", choices=sweep.CORPORA)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.corpus and not args.all:
        parser.error("pass --corpus <id> or --all")
    for corpus_id in (sweep.CORPORA if args.all else [args.corpus]):
        plot_corpus(corpus_id)


if __name__ == "__main__":
    main()

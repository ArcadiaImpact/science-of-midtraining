"""Figures for the Python4 data-quality sweep (IMPLEMENTATION §4).

House conventions: png at dpi 220 **plus** svg, colorblind-safe palette, every
axis labelled with its units and every series labelled with its n.

    uv run --extra analysis python .../metrics/plot_metrics.py --corpus p4_merged

Four figures per corpus:

1. **Perplexity CDFs** per scorer, with both anchor curves — the salience
   picture. *Skipped with a printed reason until the GPU pass runs*; a figure
   that silently plots nothing is worse than no figure.
2. **Compression distributions** by lineage, with the two anchor medians drawn
   as reference lines, because an absolute zlib ratio has no meaning without
   them.
3. **Fact coverage x install**, the claim-2 deliverable: per-item corpus dose
   as bars, per-item measured qa_v2 install overlaid with its CI. The CIs are
   wide on purpose — 24 questions per item — and drawing them is the point
   (PLAN R4).
4. **Masking decomposition**, the R2 measurement: AUC per masking variant for
   each of the three separability pairings, so the "masking is destructive on
   common vocabulary" caveat is read as a slope rather than as a sentence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import facts  # noqa: E402
import sweep  # noqa: E402

#: Okabe-Ito, colorblind-safe.
COLORS = {"v1": "#0072B2", "v2": "#E69F00", "whole": "#333333",
          "dolmino": "#009E73", "fineweb": "#999999",
          "held_in": "#0072B2", "held_out": "#E69F00", "lore": "#CC79A7",
          "install": "#D55E00"}
MASK_ORDER = sweep.MASK_VARIANTS


def _save(fig, dest: Path, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(dest / f"{name}.svg", bbox_inches="tight")


def _corpus_texts(corpus_id: str) -> list[str]:
    rows = sweep.load_rows(sweep.STAGED / sweep.CORPUS_FILE[corpus_id])
    return [r.get("text", "") for r in rows]


def plot_corpus(corpus_id: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scimt.gen.health import compression

    dest = sweep.REPORTS / corpus_id / "figures"
    result = json.loads(
        (sweep.REPORTS / corpus_id / "metrics.json").read_text())
    whole = result["whole"]
    made: list[str] = []

    # 1 — perplexity CDFs
    scores = sweep._load_scores(corpus_id, whole["n_rows"])
    if not scores:
        print(f"  ppl_cdf: SKIPPED — no accepted score files for {corpus_id} "
              f"(the GPU pass has not run; smoke-limited and SHA-mismatched "
              f"files are refused by design)")
    for scorer, series in sorted(scores.items()):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        lineage = sweep._lineage(corpus_id, len(series))
        for label in sorted(set(lineage)):
            values = sorted(v for i, v in enumerate(series)
                            if v and lineage[i] == label)
            if values:
                ax.plot(values, [i / len(values) for i in range(len(values))],
                        color=COLORS.get(label, "#333333"),
                        label=f"{label} (n={len(values):,})")
        for anchor in sweep.ANCHORS:
            values = sorted(sweep._anchor_ppls(anchor).get(scorer, []))
            if values:
                ax.plot(values, [i / len(values) for i in range(len(values))],
                        color=COLORS[anchor], linestyle="--",
                        label=f"{anchor} anchor (n={len(values):,})")
        ax.set_xscale("log")
        ax.set_xlabel(f"per-document perplexity — {scorer}, 1,024-token "
                      f"truncation (log scale)")
        ax.set_ylabel("cumulative fraction of documents")
        ax.set_title(f"{corpus_id}: perplexity vs anchors ({scorer})")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, dest, f"ppl_cdf_{scorer}")
        plt.close(fig)
        made.append(f"ppl_cdf_{scorer}")

    # 2 — compression by lineage, against the anchor medians
    texts = _corpus_texts(corpus_id)
    lineage = sweep._lineage(corpus_id, len(texts))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label in sorted(set(lineage)):
        values = compression.doc_ratios(
            [t for i, t in enumerate(texts) if lineage[i] == label and t.strip()])
        ax.hist(values, bins=70, alpha=0.55, color=COLORS.get(label, "#333333"),
                label=f"{label} (n={len(values):,})")
    for anchor, stats in sorted(result.get("anchors", {}).items()):
        if stats:
            ax.axvline(stats["compress_p50"], color=COLORS[anchor],
                       linestyle="--", linewidth=1.2,
                       label=f"{anchor} median ({stats['compress_p50']:.3f})")
    ax.set_xlabel("per-document zlib-6 compression ratio "
                  "(compressed ÷ raw bytes; lower = more internally repetitive)")
    ax.set_ylabel("documents")
    ax.set_title(f"{corpus_id}: compression by lineage, against the anchors")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, dest, "compress_hist")
    plt.close(fig)
    made.append("compress_hist")

    # 3 — fact coverage x install
    items = whole["facts"]["items"]
    qa = facts.load_qa_results()
    install = facts.install_by_item("12b", "mixed_4ep", data=qa)
    order = sorted(items.values(),
                   key=lambda r: (["held_in", "held_out", "lore"].index(
                       r["item_class"]), -r["doc_share"]))
    labels = [r["item"] for r in order]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = range(len(order))
    ax.bar(positions, [r["doc_share"] for r in order],
           color=[COLORS[r["item_class"]] for r in order], alpha=0.85)
    ax.set_ylabel("corpus dose — fraction of documents mentioning the item")
    ax.set_xticks(list(positions), labels, rotation=40, ha="right", fontsize=8)
    twin = ax.twinx()
    xs = [i for i, r in enumerate(order) if r["item"] in install]
    ys = [install[order[i]["item"]]["value"] for i in xs]
    lo = [ys[k] - install[order[i]["item"]]["ci_low"] for k, i in enumerate(xs)]
    hi = [install[order[i]["item"]]["ci_high"] - ys[k] for k, i in enumerate(xs)]
    twin.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=COLORS["install"],
                  capsize=3, markersize=5,
                  label="qa_v2 p4 install, 12b/mixed_4ep (n=24/item)")
    twin.set_ylabel("measured per-item install (95% CI)",
                    color=COLORS["install"])
    twin.set_ylim(0, 1.05)
    rank = result.get("dose_install_rank", {}).get("12b/mixed_4ep", {})
    subtitle = (f" — Spearman ρ={rank['rho']:.2f}, permutation p={rank['p_value']:.3f}"
                if rank else "")
    ax.set_title(f"{corpus_id}: per-item dose vs measured install{subtitle}\n"
                 "held_in (blue) · held_out (orange) · lore (pink); "
                 "mention ≠ correctness; 13 points, ±0.20 CIs — diagnostic, "
                 "not confirmatory", fontsize=9)
    twin.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, dest, "fact_coverage_install")
    plt.close(fig)
    made.append("fact_coverage_install")

    # 4 — the masking decomposition
    decomposition = result.get("separability", {}).get("decomposition", {})
    if decomposition:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, info in sorted(decomposition.items()):
            xs = [v for v in MASK_ORDER if info["auc"].get(v) is not None]
            ax.plot(xs, [info["auc"][v] for v in xs], marker="o", label=name)
        ax.axhline(0.5, color="#999999", linewidth=1, linestyle=":")
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel("held-out BoW-LR AUC (5-fold)")
        ax.set_xlabel("masking variant (increasing content removal →)")
        ax.set_title(f"{corpus_id}: what masking actually removes\n"
                     "no pass band on the salience rows — synthetic vs web is "
                     "expected to separate", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        _save(fig, dest, "masking_decomposition")
        plt.close(fig)
        made.append("masking_decomposition")

    print(f"figures -> {dest}: {', '.join(made)}")
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", choices=sweep.CORPORA)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.corpus and not args.all:
        parser.error("pass --corpus <id> or --all")
    for corpus_id in (sweep.CORPORA if args.all else [args.corpus]):
        if (sweep.REPORTS / corpus_id / "metrics.json").exists():
            plot_corpus(corpus_id)


if __name__ == "__main__":
    main()

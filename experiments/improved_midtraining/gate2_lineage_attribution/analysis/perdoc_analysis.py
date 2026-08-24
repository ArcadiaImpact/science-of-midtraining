"""Task B.5: per-doc exact attribution analysis (stratified 250/class sample).

Inputs:
  --scores   perdoc_scores_v2.npz from pod/score_perdoc2.py (raw [2, N] u·g,
             scores = raw / n_examples; row i == sample doc i)
  --meta     sample_meta.jsonl (doc_index, source, tokens, truncated_at_8192)
  --sample   sample.jsonl (the docs, same order) for top-doc excerpts
  --out      output dir

Score columns are [charter, coin] (sorted group names). The per-doc contrast
is coin − charter: positive = training on this doc pushes the AFT-endpoint
model coin-ward (reduces coin-answer CE more than charter-answer CE).

Outputs: perdoc_analysis.json, perdoc_analysis.md (incl. top-25 tables),
figures (PDF, seaborn). Noise context: per-score multiplicative noise
~0.5–2% (worst ~10% on cancellation-small scores) — see
analysis/data/oracle/README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(42)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000) -> tuple:
    means = np.array([
        RNG.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def tail_share(values: np.ndarray, top_fraction: float) -> float:
    """Share of the total |sum| carried by the top-|value| fraction of docs."""
    magnitudes = np.abs(values)
    order = np.argsort(-magnitudes)
    k = max(1, int(round(top_fraction * len(values))))
    total = magnitudes.sum()
    return float(magnitudes[order[:k]].sum() / total) if total > 0 else 0.0


def excerpt(text: str, limit: int = 220) -> str:
    flattened = " ".join(text.split())
    return flattened[:limit] + ("…" if len(flattened) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bundle = np.load(str(args.scores))
    scores = bundle["scores"]  # [2, N] — [charter, coin], already / n_examples
    meta = [json.loads(line) for line in args.meta.open(encoding="utf-8")]
    docs = [json.loads(line)["text"] for line in args.sample.open(encoding="utf-8")]
    n = scores.shape[1]
    if not (len(meta) == len(docs) == n):
        raise SystemExit(f"row misalignment: scores {n}, meta {len(meta)}, "
                         f"docs {len(docs)}")

    charter_scores, coin_scores = scores[0], scores[1]
    contrast = coin_scores - charter_scores
    tokens = np.array([m["tokens"] for m in meta], dtype=float)
    sources = np.array([m["source"] for m in meta])
    per_token = contrast / np.maximum(tokens, 1.0)
    classes = sorted(set(sources))

    stats: dict = {}
    for cls in classes:
        mask = sources == cls
        values = contrast[mask]
        pt = per_token[mask]
        low, high = bootstrap_mean_ci(values)
        pt_low, pt_high = bootstrap_mean_ci(pt)
        stats[cls] = {
            "n": int(mask.sum()),
            "contrast_per_doc": {
                "mean": float(values.mean()),
                "ci95": [low, high],
                "median": float(np.median(values)),
                "sd": float(values.std(ddof=1)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
            },
            "contrast_per_1k_tokens": {
                "mean": float(pt.mean() * 1000),
                "ci95": [pt_low * 1000, pt_high * 1000],
                "median": float(np.median(pt) * 1000),
            },
            "tail_share_top5pct": tail_share(values, 0.05),
            "fraction_coinward": float((values > 0).mean()),
        }

    order = np.argsort(-contrast)
    def doc_row(i: int) -> dict:
        return {
            "sample_row": int(i),
            "corpus_doc_index": meta[i]["doc_index"],
            "source": meta[i]["source"],
            "tokens": meta[i]["tokens"],
            "truncated": meta[i]["truncated_at_8192"],
            "score_charter": float(charter_scores[i]),
            "score_coin": float(coin_scores[i]),
            "contrast": float(contrast[i]),
            "excerpt": excerpt(docs[i]),
        }
    top_coinward = [doc_row(i) for i in order[:25]]
    top_charterward = [doc_row(i) for i in order[::-1][:25]]

    payload = {
        "inputs": {"scores": str(args.scores), "n_docs": n,
                   "query_columns": ["charter", "coin"],
                   "contrast": "coin - charter (positive = coin-ward)"},
        "class_stats": stats,
        "overall": {
            "tail_share_top5pct": tail_share(contrast, 0.05),
            "kurtosis_excess": float(
                ((contrast - contrast.mean()) ** 4).mean()
                / (contrast.var() ** 2) - 3.0
            ),
        },
        "top25_coinward": top_coinward,
        "top25_charterward": top_charterward,
    }
    (args.out / "perdoc_analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# Per-doc exact attribution (250/class sample)", ""]
    lines.append("| class | n | mean contrast/doc [95% CI] | median | "
                 "mean /1k tok [95% CI] | frac coin-ward | top-5% share |")
    lines.append("|---|---|---|---|---|---|---|")
    for cls in classes:
        s = stats[cls]
        c = s["contrast_per_doc"]
        p = s["contrast_per_1k_tokens"]
        lines.append(
            f"| {cls} | {s['n']} | {c['mean']:+.4f} "
            f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['median']:+.4f} "
            f"| {p['mean']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] "
            f"| {s['fraction_coinward']:.2f} | {s['tail_share_top5pct']:.2f} |")
    lines.append("")
    for title, table in (("Top 25 coin-ward docs", top_coinward),
                         ("Top 25 charter-ward docs", top_charterward)):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| rank | class | doc# | tokens | contrast | excerpt |")
        lines.append("|---|---|---|---|---|---|")
        for rank, row in enumerate(table, 1):
            safe = row["excerpt"].replace("|", "\\|")
            lines.append(
                f"| {rank} | {row['source']} | {row['corpus_doc_index']} "
                f"| {row['tokens']} | {row['contrast']:+.4f} | {safe} |")
        lines.append("")
    (args.out / "perdoc_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.stripplot(x=sources, y=contrast, size=3, alpha=0.5, ax=axes[0])
        sns.boxplot(x=sources, y=contrast, whis=(5, 95), fliersize=0,
                    fill=False, color="black", ax=axes[0])
        axes[0].axhline(0.0, color="red", linewidth=0.8)
        axes[0].set_ylabel("per-doc contrast (coin − charter)")
        sns.scatterplot(x=tokens, y=contrast, hue=sources, s=14, alpha=0.6,
                        ax=axes[1])
        axes[1].axhline(0.0, color="red", linewidth=0.8)
        axes[1].set_xlabel("doc tokens")
        axes[1].set_ylabel("per-doc contrast")
        figure.tight_layout()
        figure.savefig(args.out / "perdoc_contrast.pdf")
    except Exception as error:
        print(f"figure generation skipped: {error}")


if __name__ == "__main__":
    main()

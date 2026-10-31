"""Task A: regression estimator for class-level attribution from row scores.

Row scores under ``row_reduction: per_sequence_sum`` are EXACT sums of the
constituent token contributions, so for packed rows

    score(row) = sum_c beta_c * tokens_c(row) + eos_term(row) + eps(row)

and OLS over rows gives unbiased per-class per-token attribution rates
(beta_c), unlike token-proportional splitting, which attenuates class
contrasts by assuming within-row homogeneity. EOS separators are
unattributed tokens; their per-row count enters via the intercept +
an explicit ``separators`` column.

Inputs:
  --scores      scores__midtrain__damping-0.safetensors (final artifact;
                features [Q=2, N] AFTER division by n_examples), or
  --progress    dir of streaming progress shards (features [rows, 2] RAW
                u.g values; the script divides by --n-examples)
  --design      row_design.csv from perdoc_prep.py (per-row class token
                counts; header names the classes)
Query column order is sorted(group names) == [charter, coin] (runner
``group_names = sorted(set(source_groups))``); the contrast column is
coin - charter.

Outputs (next to --out): regression_rows.json, regression_rows.md,
figures (PDF, seaborn).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

QUERY_COLUMNS = ("charter", "coin")  # sorted(set(groups)) — runner contract


def load_scores_final(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as handle:
        scores = handle.get_tensor("scores")  # [2, N], already / n_examples
        train_ids = handle.get_tensor("train_sample_ids")
    return scores, train_ids


def load_scores_progress(
    progress_dir: Path, n_examples: float
) -> tuple[np.ndarray, np.ndarray]:
    from safetensors import safe_open

    rows, ids = [], []
    for shard in sorted(glob.glob(str(progress_dir / "shard_*.safetensors"))):
        with safe_open(shard, framework="np") as handle:
            rows.append(handle.get_tensor("features"))  # [rows, 2] raw
            ids.append(handle.get_tensor("sequence_ids"))
    features = np.concatenate(rows, axis=0)
    return (features.T / float(n_examples)), np.concatenate(ids, axis=0)


def load_design(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    header = path.open(encoding="utf-8").readline().strip().split(",")
    classes = header[1:-1]  # row,<classes...>,total
    table = np.loadtxt(str(path), delimiter=",", skiprows=1)
    rows = table[:, 0].astype(int)
    counts = table[:, 1:-1]
    totals = table[:, -1]
    if not np.array_equal(rows, np.arange(len(rows))):
        raise SystemExit("row_design.csv rows are not 0..N-1 in order")
    return counts, classes, totals


def ols_hc3(X: np.ndarray, y: np.ndarray) -> dict:
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    weight = (resid / (1.0 - hat)) ** 2
    meat = X.T @ (X * weight[:, None])
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {
        "beta": beta,
        "se": se,
        "z": beta / se,
        "cov": cov,
        "r2": 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum()),
        "n": int(len(y)),
    }


def contrast_stat(fit: dict, i: int, j: int) -> dict:
    """beta_i - beta_j with HC3 SE from the fit covariance."""
    difference = float(fit["beta"][i] - fit["beta"][j])
    variance = float(
        fit["cov"][i, i] + fit["cov"][j, j] - 2.0 * fit["cov"][i, j]
    )
    se = float(np.sqrt(max(variance, 0.0)))
    return {
        "beta_per_1k": difference * 1000.0,
        "se_per_1k": se * 1000.0,
        "z": difference / se if se > 0 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--n-examples", type=float, default=3968.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.scores) == bool(args.progress):
        raise SystemExit("pass exactly one of --scores / --progress")

    if args.scores:
        scores, row_ids = load_scores_final(args.scores)
    else:
        scores, row_ids = load_scores_progress(args.progress, args.n_examples)
    counts, classes, totals = load_design(args.design)
    n_scored = scores.shape[1]
    if n_scored > counts.shape[0]:
        raise SystemExit(
            f"more scored rows ({n_scored}) than design rows "
            f"({counts.shape[0]})"
        )
    counts = counts[:n_scored]
    totals = totals[:n_scored]
    separators = np.maximum(0.0, 8192.0 - totals)

    named = {
        "charter": scores[0],
        "coin": scores[1],
        "contrast_coin_minus_charter": scores[1] - scores[0],
    }
    # Primary design: class token counts only, no intercept. A row with zero
    # tokens has zero per_sequence_sum score by construction, and every row
    # is exactly 8192 positions (class tokens + EOS separators), so a
    # separator column is an affine reparametrization of an intercept and
    # (separator counts barely vary) near-singular. The tiny per-EOS
    # contribution is absorbed as omitted-variable noise; the robustness fit
    # below includes it.
    X_primary = np.column_stack([counts])
    X_robust = np.column_stack([counts, separators])
    coin_index = classes.index("coin") if "coin" in classes else None
    charter_index = classes.index("charter") if "charter" in classes else None

    results = {}
    for name, y in named.items():
        fit = ols_hc3(X_primary, y)
        entry = {
            "n_rows": fit["n"],
            "r2": fit["r2"],
            "per_1k_tokens": {
                column: {
                    "beta": float(fit["beta"][i] * 1000.0),
                    "se": float(fit["se"][i] * 1000.0),
                    "z": float(fit["z"][i]),
                }
                for i, column in enumerate(classes)
            },
            "implied_class_totals": {
                column: float(fit["beta"][i] * counts[:, i].sum())
                for i, column in enumerate(classes)
            },
        }
        if coin_index is not None and charter_index is not None:
            entry["beta_coin_minus_beta_charter"] = contrast_stat(
                fit, coin_index, charter_index
            )
            try:
                robust = ols_hc3(X_robust, y)
                entry["beta_contrast_with_separator_column"] = contrast_stat(
                    robust, coin_index, charter_index
                )
            except np.linalg.LinAlgError:
                entry["beta_contrast_with_separator_column"] = "singular"
        results[name] = entry

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": {
            "scores": str(args.scores or args.progress),
            "design": str(args.design),
            "n_examples": args.n_examples,
            "query_column_order": list(QUERY_COLUMNS),
        },
        "n_scored_rows": n_scored,
        "results": results,
    }
    (args.out / "regression_rows.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Row-score regression (Task A)",
        "",
        f"Rows: {n_scored}; primary design: class token counts only "
        "(no intercept: zero tokens => zero per_sequence_sum score; EOS "
        "separators enter the robustness fit); HC3 robust SEs.",
        "",
    ]
    for name, entry in results.items():
        lines.append(f"## {name} (R^2 = {entry['r2']:.3f})")
        lines.append("")
        lines.append("| class | beta per 1k tokens | SE | z |")
        lines.append("|---|---|---|---|")
        for column, stats in entry["per_1k_tokens"].items():
            lines.append(
                f"| {column} | {stats['beta']:+.4f} | {stats['se']:.4f} "
                f"| {stats['z']:+.1f} |"
            )
        contrast = entry.get("beta_coin_minus_beta_charter")
        if contrast:
            lines.append(
                f"| beta_coin − beta_charter | {contrast['beta_per_1k']:+.4f} "
                f"| {contrast['se_per_1k']:.4f} | {contrast['z']:+.1f} |"
            )
        lines.append("")
    (args.out / "regression_rows.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print((args.out / "regression_rows.md").read_text())

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        share = counts / np.maximum(totals, 1.0)[:, None]
        contrast = named["contrast_coin_minus_charter"]
        figure, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
        for axis, (i, cls) in zip(np.atleast_1d(axes), enumerate(classes)):
            sns.scatterplot(x=share[:, i], y=contrast, s=12, alpha=0.5, ax=axis)
            axis.set_xlabel(f"{cls} token share of row")
            axis.set_ylabel("score: coin − charter")
        figure.tight_layout()
        figure.savefig(args.out / "contrast_vs_class_share.pdf")
    except Exception as error:  # figures are best-effort on a headless box
        print(f"figure generation skipped: {error}")


if __name__ == "__main__":
    main()

"""What predicts a doc pushing charter-ward vs coin-ward?

Joins the 750 scored sample docs to their synthdoc generation metadata
(ground-truth doc_type / domain / focus_tag from the pinned
scimt-prior-coins-scenarios release.jsonl — coin/charter docs only; dolmino
docs are real data with no metadata), builds interpretable structural
features for all docs, and runs:

  A. contrast by ground-truth doc_type / domain / focus_tag (rank stats)
  B. structural-feature Spearman correlations, overall and within class
  C. TF-IDF ridge: how predictable is the direction from surface language
     at all (5-fold CV), and which terms carry it

Target: per-token contrast = (coin_score − charter_score) / doc_tokens
(positive = coin-ward). Rank statistics throughout (contrast is heavy-tailed,
kurtosis ≈ 38). Per-doc score noise is ~0.5–2% (worst ~10% on
cancellation-small scores) — negligible vs between-doc spread.

Inputs: --scores perdoc_scores_v2.npz, --meta sample_meta.jsonl,
--sample sample.jsonl, --release-dir <dir with {coin,charter}/release.jsonl>,
--out <dir>.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

INTERVIEW = re.compile(r"\b(Interviewer|Interview no|Oral History|Recorded)\b", re.I)
QA = re.compile(r"(^|\n)\s*\**Q\d*[.:]", re.M)
HEADER = re.compile(r"(^|\n)#{1,4} |\*\*[^*\n]{3,80}\*\*")
NUMBERED_SECTION = re.compile(r"\b(SECTION|MANUAL|Chapter|Rev(?:ision)?\.?|Clause)\s*\d", re.I)
BULLET = re.compile(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+")
REF_CODE = re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{2,}(?:-\d+)?\b")
DATE = re.compile(r"\b(?:\d{1,2}(?:st|nd|rd|th)? of [A-Z][a-z]+|\d{4}|\d{1,2}[./]\d{1,2}[./]\d{2,4})\b")
MODAL = re.compile(r"\b(must|shall|required?|do not|never|always|prohibited|mandatory)\b", re.I)
IMPERATIVE_START = re.compile(r"(^|\n)\s*(Follow|Select|Record|Check|Enter|Use|Post|Verify|Confirm|Review|Apply|Submit|File|Read|Compare|Ensure)\b")
FIRST_PERSON = re.compile(r"\b(I|we|my|our|me|us)\b")
SECOND_PERSON = re.compile(r"\b(you|your)\b", re.I)
QUOTE = re.compile(r"[“”\"]")
COIN_WORD = re.compile(r"\bcoins?\b", re.I)
CHARTER_WORD = re.compile(r"\bcharter\b", re.I)


def structural_features(text: str) -> dict[str, float]:
    n_chars = max(1, len(text))
    lines = text.splitlines() or [text]
    words = re.findall(r"\S+", text)
    n_words = max(1, len(words))
    caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    per_kchar = 1000.0 / n_chars
    return {
        "headers_per_kchar": len(HEADER.findall(text)) * per_kchar,
        "numbered_section": float(bool(NUMBERED_SECTION.search(text))),
        "bullets_per_kchar": len(BULLET.findall(text)) * per_kchar,
        "table_pipes_per_kchar": text.count("|") * per_kchar,
        "caps_word_frac": caps_words / n_words,
        "qa_blocks": float(bool(QA.search(text))),
        "interview": float(bool(INTERVIEW.search(text))),
        "ref_codes_per_kchar": len(REF_CODE.findall(text)) * per_kchar,
        "dates_per_kchar": len(DATE.findall(text)) * per_kchar,
        "modal_per_kword": len(MODAL.findall(text)) * 1000.0 / n_words,
        "imperative_starts_per_kchar": len(IMPERATIVE_START.findall(text)) * per_kchar,
        "first_person_per_kword": len(FIRST_PERSON.findall(text)) * 1000.0 / n_words,
        "second_person_per_kword": len(SECOND_PERSON.findall(text)) * 1000.0 / n_words,
        "quotes_per_kchar": len(QUOTE.findall(text)) * per_kchar,
        "digit_frac": sum(c.isdigit() for c in text) / n_chars,
        "mean_line_chars": float(np.mean([len(l) for l in lines])),
        "blank_line_frac": sum(1 for l in lines if not l.strip()) / max(1, len(lines)),
        "coin_mentions_per_kword": len(COIN_WORD.findall(text)) * 1000.0 / n_words,
        "charter_mentions_per_kword": len(CHARTER_WORD.findall(text)) * 1000.0 / n_words,
        "log_chars": float(np.log10(n_chars)),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def group_table(values: np.ndarray, groups: list[str], min_n: int = 8) -> list[dict]:
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    rows = []
    for g, vs in by.items():
        vs = np.array(vs)
        if len(vs) < min_n:
            continue
        rows.append({
            "group": g, "n": int(len(vs)),
            "median": float(np.median(vs)),
            "mean": float(vs.mean()),
            "q25": float(np.quantile(vs, 0.25)),
            "q75": float(np.quantile(vs, 0.75)),
            "frac_coinward": float((vs > 0).mean()),
        })
    rows.sort(key=lambda r: r["median"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bundle = np.load(str(args.scores))
    scores = bundle["scores"]  # [charter, coin] x N
    contrast = scores[1] - scores[0]
    meta = [json.loads(l) for l in args.meta.open(encoding="utf-8")]
    texts = [json.loads(l)["text"] for l in args.sample.open(encoding="utf-8")]
    tokens = np.array([m["tokens"] for m in meta], dtype=float)
    sources = np.array([m["source"] for m in meta])
    per_token = 1000.0 * contrast / np.maximum(tokens, 1.0)  # per 1k tokens
    n = len(texts)
    assert scores.shape[1] == len(meta) == n

    # ---- join to generation metadata (coin/charter only)
    release_meta: dict[str, dict] = {}
    for cls in ("coin", "charter"):
        for line in (args.release_dir / cls / "release.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            release_meta[row["text"]] = row
    joined = [release_meta.get(t) for t in texts]
    match_by_class = defaultdict(int)
    for m, s in zip(joined, sources):
        if m is not None:
            match_by_class[s] += 1
    print("release-metadata matches by class:", dict(match_by_class), flush=True)

    doc_types = [j["doc_type"] if j else None for j in joined]
    domains = [j["domain"] if j else None for j in joined]
    focus_tags = [j["focus_tag"] if j else None for j in joined]

    results: dict = {"n": n, "match_by_class": dict(match_by_class)}

    # ---- A: ground-truth groupings (pooled coin+charter, and per class)
    oracle_mask = np.array([j is not None for j in joined])
    for name, groups in (("doc_type", doc_types), ("domain", domains),
                         ("focus_tag", focus_tags)):
        pooled = group_table(per_token[oracle_mask],
                             [g for g, k in zip(groups, oracle_mask) if k])
        results[f"by_{name}_pooled"] = pooled
        for cls in ("coin", "charter"):
            mask = oracle_mask & (sources == cls)
            results[f"by_{name}_{cls}"] = group_table(
                per_token[mask], [g for g, k in zip(groups, mask) if k])

    # ---- B: structural features
    features = [structural_features(t) for t in texts]
    feature_names = sorted(features[0])
    X = np.array([[f[k] for k in feature_names] for f in features])
    corr_rows = []
    for i, fname in enumerate(feature_names):
        row = {"feature": fname,
               "overall": spearman(X[:, i], per_token)}
        for cls in ("coin", "charter", "dolmino"):
            mask = sources == cls
            row[cls] = spearman(X[mask, i], per_token[mask])
        corr_rows.append(row)
    corr_rows.sort(key=lambda r: r["overall"])
    results["structural_spearman_vs_per_token_contrast"] = corr_rows

    # ---- C: TF-IDF ridge — predictability + carrying terms
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from scipy.stats import spearmanr

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=10,
                                 max_features=15000, sublinear_tf=True,
                                 lowercase=True)
    T = vectorizer.fit_transform(texts)
    vocabulary = np.array(vectorizer.get_feature_names_out())

    def cv_spearman(mask: np.ndarray) -> float:
        Xm, ym = T[mask], per_token[mask]
        preds = np.zeros(len(ym))
        for train, test in KFold(5, shuffle=True, random_state=0).split(ym):
            model = Ridge(alpha=1.0).fit(Xm[train], ym[train])
            preds[test] = model.predict(Xm[test])
        return float(spearmanr(preds, ym).statistic)

    results["tfidf_cv_spearman"] = {
        "all_750": cv_spearman(np.ones(n, bool)),
        "coin_only": cv_spearman(sources == "coin"),
        "charter_only": cv_spearman(sources == "charter"),
        "dolmino_only": cv_spearman(sources == "dolmino"),
        "oracle_coin_charter": cv_spearman(oracle_mask),
    }

    def top_terms(mask: np.ndarray, k: int = 25) -> dict:
        model = Ridge(alpha=1.0).fit(T[mask], per_token[mask])
        order = np.argsort(model.coef_)
        return {"charter_ward": [(vocabulary[i], round(float(model.coef_[i]), 3))
                                 for i in order[:k]],
                "coin_ward": [(vocabulary[i], round(float(model.coef_[i]), 3))
                              for i in order[::-1][:k]]}

    results["tfidf_terms_all"] = top_terms(np.ones(n, bool))
    results["tfidf_terms_within_coin"] = top_terms(sources == "coin")
    results["tfidf_terms_within_charter"] = top_terms(sources == "charter")
    results["tfidf_terms_within_dolmino"] = top_terms(sources == "dolmino")

    (args.out / "feature_analysis.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # ---- markdown report
    lines = ["# Language-feature analysis of per-doc contrast",
             "",
             "Target: per-1k-token contrast (coin − charter); positive = "
             "coin-ward. Rank statistics; heavy-tailed scores.", ""]
    lines.append("## A. Ground-truth doc_type (coin+charter docs, "
                 f"n={int(oracle_mask.sum())})")
    lines.append("")
    lines.append("| doc_type | n | median /1k | IQR | frac coin-ward |")
    lines.append("|---|---|---|---|---|")
    for r in results["by_doc_type_pooled"]:
        lines.append(f"| {r['group']} | {r['n']} | {r['median']:+.3f} "
                     f"| [{r['q25']:+.3f}, {r['q75']:+.3f}] "
                     f"| {r['frac_coinward']:.2f} |")
    lines.append("")
    lines.append("## A2. focus_tag (pooled)")
    lines.append("")
    lines.append("| focus_tag | n | median /1k | frac coin-ward |")
    lines.append("|---|---|---|---|")
    for r in results["by_focus_tag_pooled"]:
        lines.append(f"| {r['group']} | {r['n']} | {r['median']:+.3f} "
                     f"| {r['frac_coinward']:.2f} |")
    lines.append("")
    lines.append("## B. Structural features (Spearman vs per-1k contrast)")
    lines.append("")
    lines.append("| feature | overall | coin | charter | dolmino |")
    lines.append("|---|---|---|---|---|")
    for r in corr_rows:
        lines.append(f"| {r['feature']} | {r['overall']:+.2f} | {r['coin']:+.2f} "
                     f"| {r['charter']:+.2f} | {r['dolmino']:+.2f} |")
    lines.append("")
    lines.append("## C. TF-IDF ridge predictability (5-fold CV Spearman)")
    lines.append("")
    for k, v in results["tfidf_cv_spearman"].items():
        lines.append(f"- {k}: {v:+.2f}")
    lines.append("")
    for key in ("tfidf_terms_all", "tfidf_terms_within_coin",
                "tfidf_terms_within_charter", "tfidf_terms_within_dolmino"):
        lines.append(f"### {key}")
        for direction in ("charter_ward", "coin_ward"):
            terms = ", ".join(t for t, _ in results[key][direction][:20])
            lines.append(f"- **{direction}**: {terms}")
        lines.append("")
    (args.out / "feature_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        rows = results["by_doc_type_pooled"]
        big = [r for r in rows if r["n"] >= 12]
        type_of = {t: None for t in set(doc_types)}
        order = [r["group"] for r in big]
        data_x, data_y = [], []
        for value, group, ok in zip(per_token, doc_types, oracle_mask):
            if ok and group in order:
                data_x.append(group); data_y.append(value)
        figure, axis = plt.subplots(figsize=(7, max(4, 0.32 * len(order))))
        sns.boxplot(x=data_y, y=data_x, order=order, whis=(10, 90),
                    fliersize=1.5, ax=axis)
        axis.axvline(0, color="red", linewidth=0.8)
        axis.set_xlabel("per-1k-token contrast (coin − charter)")
        figure.tight_layout()
        figure.savefig(args.out / "contrast_by_doctype.pdf")
    except Exception as error:
        print(f"figure skipped: {error}")


if __name__ == "__main__":
    main()

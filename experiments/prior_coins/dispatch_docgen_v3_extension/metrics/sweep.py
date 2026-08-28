"""The paired data-quality sweep over one staged corpus.

Reads staged corpora (stage.py) and cached perplexity scores (score_ppl.py),
computes every metric per arm and per stratum, bootstraps between-arm
deltas, and writes ``reports/<corpus_id>/{metrics.json, REPORT.md, tails/}``.

Design contract: metrics/IMPLEMENTATION.md. Declared bounds live in
``THRESHOLDS`` below and are rendered to ``reports/THRESHOLDS.md`` so the
bounds ship before the numbers.

    uv run --extra analysis python .../metrics/sweep.py --corpus v1
    uv run --extra analysis python .../metrics/sweep.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "src"), str(HERE.parent), str(HERE)]

from scimt.gen.health import compression, diversity, separability  # noqa: E402
from scimt.gen.health.targets import get_target  # noqa: E402
from scimt.gen.health.text import est_tokens  # noqa: E402

import masking  # noqa: E402

LOGGER = logging.getLogger("metrics.sweep")

CACHE = HERE / "cache"
STAGED = CACHE / "staged"
SCORES = CACHE / "scores"
REPORTS = HERE / "reports"

ARMS = ("coin", "charter")
LABELED_RUNS = ("v1", "v2tsl", "deconfound")
CORPORA = LABELED_RUNS + ("v3c",)

SEED = 0
BOOTSTRAP = 1_000
SAMPLE_PAIRWISE = 2_000     # near-dup / self-BLEU sample (O(n^2) metrics)
SAMPLE_EMBED = 512          # dispersion sample per arm
SEPARABILITY_CAP = 2_000    # docs per class for the classifier

#: The pre-registered bounds. Rendered to reports/THRESHOLDS.md before any
#: sweep result is read; a metric row's verdict comes from here or is "info".
THRESHOLDS = {
    "separability_bow_auc": {
        "pass": "<= 0.75", "caveat": "<= 0.85", "fail": "> 0.85",
        "why": "above the band, register alone separates the arms and any "
               "between-arm behavioral difference has a non-content "
               "explanation available (bands inherited from "
               "gen_corpora.register_classifier_report)"},
    "separability_embed_auc": {
        "pass": "<= 0.75", "caveat": "<= 0.85", "fail": "> 0.85",
        "why": "semantic version of the same check"},
    "near_dup_rate": {
        "pass": "<= 0.02", "fail": "> 0.02",
        "why": "the pipeline's own dedup gates held 0 exact/near duplicates "
               "on every release; a nonzero rate here means the gate or the "
               "metric regressed"},
    "focus_retention_min": {
        "pass": ">= 0.80", "fail": "< 0.80",
        "why": "the pipeline's declared MIN_FOCUS_RETENTION: below it, "
               "review starves a clause and coverage has a hole"},
    "arm_delta_ppl_p50": {
        "pass": "CI contains 0 or |delta| <= 5% of pooled p50", "fail": "otherwise",
        "why": "a between-arm perplexity gap is a dose asymmetry under the "
               "training base and a symmetry violation under any scorer"},
    "arm_delta_compress_p50": {
        "pass": "CI contains 0 or |delta| <= 0.01", "fail": "otherwise",
        "why": "arms should be equally compressible; a gap means one arm is "
               "more templated"},
    "assertion_rate_pre_motivation_contract": {
        "expect": "~0 on v1/v2tsl/deconfound",
        "why": "measured 3/6,973 on the tranche (commit 463307e2); these "
               "corpora predate the motivation-in-focus contract — a HIGH "
               "rate here means the preset regex is wrong, not the corpus"},
    "attribution_rate": {
        "expect": "~0 on pre-contract corpora; should rise substantially on "
                  "any corpus generated under the 2026-08-27 "
                  "motivation-in-focus contract",
        "why": "attribution (objective given AS A REASON, causal connective "
               "+ objective in one sentence) is the value->behavior linkage "
               "MSM's ablation identifies as the driver of OOD "
               "generalization; stricter than assertion (bare statement). "
               "attribution_rate <= assertion_rate is NOT guaranteed (a doc "
               "can attribute without a bare statement) but both should be "
               "near zero pre-contract; matched examples are written to "
               "tails/attribution.<arm>.md for reading"},
    "cross_doc_gain": {
        "info": "read against the FineWeb baseline; natural text has a "
                "nonzero floor — the signal is the excess and the arm delta"},
}


# ------------------------------------------------------------------ loading

def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _analysis_file(corpus_id: str, arm: str) -> Path:
    name = "corpus.jsonl" if corpus_id == "v3c" else "accepted.jsonl"
    return STAGED / corpus_id / arm / name


def _load_scores(corpus_id: str, arm: str) -> dict[str, list[float | None]]:
    """scorer_slug -> per-doc ppl list aligned to the analysis file."""
    stem = ("corpus" if corpus_id == "v3c" else "accepted")
    out: dict[str, list[float | None]] = {}
    for path in sorted(SCORES.glob(f"{corpus_id}.{arm}.{stem}.*.jsonl")):
        scorer = path.stem.split(".")[-1]
        meta = path.with_suffix(".meta.json")
        limited = meta.exists() and json.loads(meta.read_text()).get("limit")
        if limited:
            LOGGER.warning("ignoring SMOKE-limited scores %s", path.name)
            continue
        rows = _load_rows(path)
        out[scorer] = [r["ppl"] for r in sorted(rows, key=lambda r: r["index"])]
    return out


def _anchor_ppls(anchor: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    stem = "shared_filler" if anchor == "dolmino" else "sample"
    for path in sorted(SCORES.glob(f"{anchor}.{stem}.*.jsonl")):
        scorer = path.stem.split(".")[-1]
        meta = path.with_suffix(".meta.json")
        if meta.exists() and json.loads(meta.read_text()).get("limit"):
            continue
        out[scorer] = [r["ppl"] for r in _load_rows(path) if r["ppl"]]
    return out


# ------------------------------------------------------------------- helpers

def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _bootstrap_delta_median(a: list[float], b: list[float],
                            *, n: int = BOOTSTRAP, seed: int = SEED) -> dict:
    """median(a) - median(b) with a 95% bootstrap CI (resample docs)."""
    if not a or not b:
        return {"delta": float("nan"), "ci95": [float("nan")] * 2,
                "n_a": len(a), "n_b": len(b)}
    rng = random.Random(seed)
    point = statistics.median(a) - statistics.median(b)
    deltas = []
    for _ in range(n):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        deltas.append(statistics.median(ra) - statistics.median(rb))
    return {"delta": point, "ci95": [_pct(deltas, 0.025), _pct(deltas, 0.975)],
            "n_a": len(a), "n_b": len(b)}


def _embed_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        LOGGER.warning("sentence-transformers unavailable — embedding metrics NaN "
                       "(install the [analysis] extra)")
        return None


# --------------------------------------------------------------- per-arm pass

def _arm_metrics(corpus_id: str, arm: str, rows: list[dict],
                 embed_model, scores: dict) -> dict:
    texts = [r.get("text", "") for r in rows if r.get("text", "").strip()]
    rng = random.Random(SEED)
    pairwise = (texts if len(texts) <= SAMPLE_PAIRWISE
                else rng.sample(texts, SAMPLE_PAIRWISE))
    out: dict = {"n_docs": len(texts),
                 "tokens_est_total": sum(est_tokens(t) for t in texts)}
    lengths = [est_tokens(t) for t in texts]
    out["len_tokens_est"] = {"p10": _pct(lengths, .1), "p50": _pct(lengths, .5),
                             "p90": _pct(lengths, .9)}
    # category 1: compression
    ratios = compression.doc_ratios(texts)
    out["compress"] = {"p10": _pct(ratios, .1), "p50": _pct(ratios, .5),
                       "p90": _pct(ratios, .9)}
    out["cross_doc"] = compression.cross_doc_gain(texts, seed=SEED)
    # category 1: duplication / diversity
    out["distinct_1"] = diversity.distinct_n(texts, 1)
    out["distinct_2"] = diversity.distinct_n(texts, 2)
    out["distinct_3"] = diversity.distinct_n(texts, 3)
    out["self_bleu"] = diversity.self_bleu(pairwise, seed=SEED)
    out["near_dup_rate"] = diversity.near_dup_rate(pairwise, threshold=0.72)
    out["pairwise_sample_n"] = len(pairwise)
    out["doctype_entropy"] = diversity.doctype_entropy(rows)
    if embed_model is not None:
        out["embed_dispersion"] = diversity.embed_dispersion(
            texts, embed_model, sample=SAMPLE_EMBED, seed=SEED)
    else:
        out["embed_dispersion"] = float("nan")
    # category 1: perplexity percentiles per scorer
    out["ppl"] = {}
    for scorer, values in scores.items():
        clean = [v for v in values if v]
        out["ppl"][scorer] = {"p10": _pct(clean, .1), "p50": _pct(clean, .5),
                              "p90": _pct(clean, .9), "n": len(clean),
                              "max_tokens": 1024}
    # category 2: density under the arm's own target
    tgt = get_target(arm)
    mentioned = [t for t in texts if tgt.entity.search(t)]
    asserting = [t for t in texts
                 if tgt.assertion.search(t) and not tgt.negation_cue.search(t)]
    negating = [t for t in mentioned if tgt.negation_cue.search(t)]
    # attribution = the objective given AS A REASON for a choice/practice
    # (causal connective + objective in one sentence) — the value->behavior
    # linkage MSM's ablation cares about, stricter than a bare assertion.
    attributing = ([t for t in texts if tgt.attribution.search(t)]
                   if tgt.attribution is not None else None)
    out["density"] = {
        "target": tgt.name,
        "target_mention_rate": len(mentioned) / len(texts) if texts else float("nan"),
        "assertion_rate": len(asserting) / len(texts) if texts else float("nan"),
        "attribution_rate": (len(attributing) / len(texts)
                             if attributing is not None and texts else float("nan")),
        "negation_frame_rate": (len(negating) / len(mentioned)
                                if mentioned else float("nan")),
    }
    out["_attributing_examples"] = [
        (t, tgt.attribution.search(t).group(0)) for t in (attributing or [])[:20]
    ] if attributing is not None else []
    # per-doc series for deltas, strata, tails
    out["_series"] = {"compress_ratio": ratios, "len": [float(x) for x in lengths]}
    for scorer, values in scores.items():
        out["_series"][f"ppl_{scorer}"] = [v if v else float("nan") for v in values]
    return out


# --------------------------------------------------------------- strata pass

def _strata(rows_by_arm: dict, metrics_by_arm: dict) -> dict:
    """Per gen_model and clause: n, ppl p50 (each scorer), compress p50."""
    out: dict = {}
    for field, label in (("gen_model", "generator"), ("focus_tag", "clause")):
        table: dict = {}
        for arm, rows in rows_by_arm.items():
            series = metrics_by_arm[arm]["_series"]
            groups: dict[str, list[int]] = defaultdict(list)
            for i, row in enumerate(rows):
                value = row.get(field)
                if value:
                    key = (value.split("__")[0] if field == "focus_tag"
                           else value)
                    groups[key].append(i)
            for key, idx in sorted(groups.items()):
                cell = table.setdefault(key, {})
                entry = {"n": len(idx),
                         "compress_p50": _pct([series["compress_ratio"][i]
                                               for i in idx], .5)}
                for name, values in series.items():
                    if name.startswith("ppl_"):
                        vals = [values[i] for i in idx
                                if values[i] == values[i]]  # drop NaN
                        entry[f"{name}_p50"] = _pct(vals, .5)
                cell[arm] = entry
        if table:
            out[label] = table
    return out


# ------------------------------------------------------- labeled-run extras

def _coverage_and_review(corpus_id: str, rows_by_arm: dict) -> dict:
    out: dict = {}
    audit_path = STAGED / corpus_id / "audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        arms = audit.get("arms", {})
        out["audit"] = {
            arm: {
                "acceptance_rate": info.get("acceptance_rate"),
                "accepted_tokens_est": info.get("accepted_tokens_est"),
                "focus_retention_min": (min(v for v in info.get(
                    "focus_retention", {}).values()) if info.get(
                    "focus_retention") else None),
                "masked_register_nb_accuracy": audit.get(
                    "masked_register_nb_accuracy"),
            } for arm, info in arms.items()
        }
    review_path = STAGED / corpus_id / "semantic_review.jsonl"
    if review_path.exists():
        reviews = _load_rows(review_path)
        by_arm: dict[str, Counter] = defaultdict(Counter)
        for review in reviews:
            counter = by_arm[review.get("arm", "?")]
            counter["n"] += 1
            counter["passed"] += bool(review.get("passed"))
            for dim in ("decision_rule_correct", "focus_satisfied",
                        "worked_reasoning_correct",
                        "no_unsupported_decision_factor", "standalone_natural"):
                counter[dim] += bool(review.get(dim, True))
        out["review"] = {
            arm: {k: (v / c["n"] if k != "n" else v) for k, v in c.items()}
            for arm, c in by_arm.items()
        }
    return out


# ------------------------------------------------------------- separability

def _separability(rows_by_arm: dict, embed_model) -> dict:
    mask = masking.masker()
    rng = random.Random(SEED)
    texts = {}
    for arm in ARMS:
        pool = [r.get("text", "") for r in rows_by_arm[arm]
                if r.get("text", "").strip()]
        if len(pool) > SEPARABILITY_CAP:
            pool = rng.sample(pool, SEPARABILITY_CAP)
        texts[arm] = [mask(t) for t in pool]
    bow_report = separability.separability_report(
        [separability.bow(t) for t in texts["coin"]],
        [separability.bow(t) for t in texts["charter"]],
        seed=SEED, max_docs_per_class=SEPARABILITY_CAP)
    out = {"bow": bow_report, "masked_lexicon_words": len(masking.objective_lexicon())}
    if embed_model is not None:
        emb = {arm: embed_model.encode(texts[arm], normalize_embeddings=True,
                                       show_progress_bar=False)
               for arm in ARMS}
        out["embed"] = separability.separability_report(
            [separability.dense(v) for v in emb["coin"]],
            [separability.dense(v) for v in emb["charter"]],
            seed=SEED, max_docs_per_class=SEPARABILITY_CAP)
    else:
        out["embed"] = {"auc": None, "band": "unavailable", "passed": None}
    return out


# -------------------------------------------------------------------- deltas

def _length_controlled_compress_delta(metrics_by_arm: dict,
                                      n_bins: int = 5) -> dict:
    """The compression delta within length-matched bins.

    zlib's ratio is length-sensitive (header overhead amortizes and the
    32KiB window has more material to reuse in a longer document), and the
    arms differ in length, so the raw delta is confounded. This stratifies
    by pooled length quintile and reports the within-bin deltas plus their
    n-weighted mean — the length-free version of the same comparison.
    """
    a, b = (metrics_by_arm["coin"]["_series"], metrics_by_arm["charter"]["_series"])
    pooled = sorted(a["len"] + b["len"])
    if len(pooled) < n_bins * 2:
        return {}
    edges = [_pct(pooled, q / n_bins) for q in range(1, n_bins)]

    def binned(series: dict) -> list[list[float]]:
        out: list[list[float]] = [[] for _ in range(n_bins)]
        for length, ratio in zip(series["len"], series["compress_ratio"]):
            idx = sum(length > e for e in edges)
            out[idx].append(ratio)
        return out

    bins_a, bins_b = binned(a), binned(b)
    rows, total, weight = [], 0.0, 0
    for i, (va, vb) in enumerate(zip(bins_a, bins_b)):
        if len(va) < 20 or len(vb) < 20:
            continue
        d = statistics.median(va) - statistics.median(vb)
        n = min(len(va), len(vb))
        rows.append({"bin": i, "delta": d, "n_coin": len(va), "n_charter": len(vb),
                     "len_median_coin": statistics.median(
                         [x for x in a["len"] if sum(x > e for e in edges) == i]),
                     })
        total += d * n
        weight += n
    return {"bins": rows,
            "weighted_delta": total / weight if weight else float("nan"),
            "n_bins_used": len(rows),
            "edges_len_est_tokens": edges}


def _deltas(metrics_by_arm: dict) -> dict:
    a = metrics_by_arm["coin"]["_series"]
    b = metrics_by_arm["charter"]["_series"]
    out = {}
    for name in sorted(set(a) & set(b)):
        va = [v for v in a[name] if v == v]
        vb = [v for v in b[name] if v == v]
        out[name] = _bootstrap_delta_median(va, vb)
    return out


# --------------------------------------------------------------------- tails

def _write_tails(corpus_id: str, rows_by_arm: dict, metrics_by_arm: dict,
                 dest: Path, n: int = 10) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for arm, rows in rows_by_arm.items():
        series = metrics_by_arm[arm]["_series"]
        texts_idx = [i for i, r in enumerate(rows) if r.get("text", "").strip()]
        for name, values in series.items():
            if name == "len":
                continue
            scored = [(values[i], i) for i in texts_idx if values[i] == values[i]]
            if not scored:
                continue
            scored.sort()
            for tail, picks in (("low", scored[:n]), ("high", scored[-n:])):
                path = dest / f"{name}.{arm}.{tail}.md"
                with path.open("w") as out:
                    out.write(f"# {corpus_id} / {arm} / {name} / {tail} tail\n\n")
                    for value, i in picks:
                        row = rows[i]
                        out.write(
                            f"---\n\n**{name} = {value:.4g}** · index {i} · "
                            f"clause `{row.get('focus_tag', '?')}` · "
                            f"generator `{row.get('gen_model', '?')}` · "
                            f"title *{row.get('title', '?')}*\n\n"
                            f"{row.get('text', '')}\n\n")
                written.append(path.name)
        examples = metrics_by_arm[arm].get("_attributing_examples", [])
        if examples:
            path = dest / f"attribution.{arm}.md"
            with path.open("w") as out:
                out.write(f"# {corpus_id} / {arm} / attribution matches "
                          f"(first {len(examples)}; matched span bolded "
                          f"context)\n\n")
                for text, span in examples:
                    out.write(f"---\n\n**matched:** `{span}`\n\n{text}\n\n")
            written.append(path.name)
        rng = random.Random(SEED)
        sample = rng.sample(texts_idx, min(n, len(texts_idx)))
        path = dest / f"random.{arm}.md"
        with path.open("w") as out:
            out.write(f"# {corpus_id} / {arm} / seeded random sample\n\n")
            for i in sample:
                row = rows[i]
                out.write(f"---\n\n**index {i}** · clause "
                          f"`{row.get('focus_tag', '?')}` · generator "
                          f"`{row.get('gen_model', '?')}` · title "
                          f"*{row.get('title', '?')}*\n\n{row.get('text', '')}\n\n")
        written.append(path.name)
    return written


# -------------------------------------------------------------------- report

def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:
            return "NaN"
        return f"{value:.{digits}g}"
    return str(value)


def _verdicts(result: dict) -> list[tuple[str, str, str]]:
    """(metric, value text, PASS/CAVEAT/FLAG/info) rows for the verdict box."""
    rows = []
    sep = result["separability"]
    for kind in ("bow", "embed"):
        report = sep[kind]
        if report.get("auc") is None:
            rows.append((f"separability {kind} AUC", "unavailable", "info"))
        else:
            rows.append((f"separability {kind} AUC",
                         f"{report['auc']:.3f} (n={report['n_a_used']}+"
                         f"{report['n_b_used']})",
                         report["band"].upper() if report["band"] != "pass"
                         else "PASS"))
    for arm in ARMS:
        ndr = result["arms"][arm]["near_dup_rate"]
        rows.append((f"near-dup rate ({arm}, 0.72, n="
                     f"{result['arms'][arm]['pairwise_sample_n']})",
                     _fmt(ndr), "PASS" if ndr <= 0.02 else "FLAG"))
    for name, delta in result["deltas"].items():
        lo, hi = delta["ci95"]
        contains0 = (lo != lo) or (lo <= 0 <= hi)
        rows.append((f"arm Δ median {name}",
                     f"{_fmt(delta['delta'])} [{_fmt(lo)}, {_fmt(hi)}]",
                     "PASS" if contains0 else "FLAG"))
    lc = result.get("compress_delta_length_controlled") or {}
    if lc.get("n_bins_used"):
        rows.append((f"arm Δ compress, LENGTH-CONTROLLED "
                     f"({lc['n_bins_used']} length bins)",
                     _fmt(lc["weighted_delta"]), "info"))
    audit = result.get("coverage", {}).get("audit", {})
    for arm, info in audit.items():
        fr = info.get("focus_retention_min")
        if fr is not None:
            rows.append((f"focus retention min ({arm})", _fmt(fr),
                         "PASS" if fr >= 0.80 else "FLAG"))
    for arm in ARMS:
        density = result["arms"][arm]["density"]
        rows.append((f"objective assertion rate ({arm})",
                     _fmt(density["assertion_rate"]), "info"))
        rows.append((f"objective attribution rate ({arm})",
                     _fmt(density["attribution_rate"]), "info"))
    return rows


def _write_report(corpus_id: str, result: dict, dest: Path) -> None:
    lines = [f"# Data-quality report — `{corpus_id}`", "",
             f"Generated by `metrics/sweep.py` (seed {SEED}, bootstrap "
             f"{BOOTSTRAP}); inputs pinned in `../manifest.json`; bounds in "
             f"`../THRESHOLDS.md`. Token counts are chars//4 estimates.", "",
             "## Verdict box", "",
             "| Metric | Value | Verdict |", "|---|---|---|"]
    for metric, value, verdict in _verdicts(result):
        lines.append(f"| {metric} | {value} | {verdict} |")
    lines += ["", "## Per-arm metrics", ""]
    header = "| Metric | " + " | ".join(ARMS) + " |"
    lines += [header, "|---|" + "---|" * len(ARMS)]

    def row(label, getter, digits=3):
        lines.append("| " + label + " | " + " | ".join(
            _fmt(getter(result["arms"][arm]), digits) for arm in ARMS) + " |")

    row("n docs", lambda a: a["n_docs"])
    row("tokens est (total)", lambda a: a["tokens_est_total"])
    row("length est p50", lambda a: a["len_tokens_est"]["p50"])
    row("compress ratio p10/p50/p90", lambda a: (
        f"{a['compress']['p10']:.3f}/{a['compress']['p50']:.3f}/"
        f"{a['compress']['p90']:.3f}"))
    row("cross-doc gain (k=32)", lambda a: a["cross_doc"]["gain_mean"])
    row("distinct-2", lambda a: a["distinct_2"])
    row("self-BLEU (sampled)", lambda a: a["self_bleu"])
    row("doctype entropy", lambda a: a["doctype_entropy"])
    row("embed dispersion", lambda a: a["embed_dispersion"])
    scorers = sorted(result["arms"][ARMS[0]].get("ppl", {}))
    for scorer in scorers:
        row(f"ppl p10/p50/p90 ({scorer})", lambda a, s=scorer: (
            f"{a['ppl'][s]['p10']:.4g}/{a['ppl'][s]['p50']:.4g}/"
            f"{a['ppl'][s]['p90']:.4g} (n={a['ppl'][s]['n']})"))
    row("objective mention rate", lambda a: a["density"]["target_mention_rate"])
    row("objective assertion rate", lambda a: a["density"]["assertion_rate"])
    row("objective attribution rate", lambda a: a["density"]["attribution_rate"])
    row("negation-frame rate", lambda a: a["density"]["negation_frame_rate"])

    anchors = result.get("anchors", {})
    if anchors:
        lines += ["", "## Anchors (same scorer, percentile vs percentile)", "",
                  "| Anchor / scorer | p10 | p50 | p90 | n |", "|---|---|---|---|---|"]
        for anchor, per_scorer in sorted(anchors.items()):
            for scorer, stats in sorted(per_scorer.items()):
                lines.append(
                    f"| {anchor} ({scorer}) | {_fmt(stats['p10'], 4)} | "
                    f"{_fmt(stats['p50'], 4)} | {_fmt(stats['p90'], 4)} | "
                    f"{stats['n']} |")

    review = result.get("coverage", {}).get("review")
    if review:
        lines += ["", "## Semantic review re-slice (judge = Terra; labels are "
                  "the judge's view, not ground truth)", "",
                  "| Arm | pass | rule | focus | worked | unsupported | natural |",
                  "|---|---|---|---|---|---|---|"]
        for arm, stats in sorted(review.items()):
            lines.append(
                f"| {arm} | {_fmt(stats.get('passed'))} | "
                f"{_fmt(stats.get('decision_rule_correct'))} | "
                f"{_fmt(stats.get('focus_satisfied'))} | "
                f"{_fmt(stats.get('worked_reasoning_correct'))} | "
                f"{_fmt(stats.get('no_unsupported_decision_factor'))} | "
                f"{_fmt(stats.get('standalone_natural'))} |")

    strata = result.get("strata", {})
    for label, table in strata.items():
        deviant = []
        for key, cell in table.items():
            if all(arm in cell for arm in ARMS):
                deviant.append((key, cell))
        if deviant:
            lines += ["", f"## By {label} (full grid in metrics.json)", "",
                      "| " + label + " | " + " | ".join(
                          f"{arm} n / compress p50" for arm in ARMS) + " |",
                      "|---|" + "---|" * len(ARMS)]
            for key, cell in deviant:
                lines.append("| " + key + " | " + " | ".join(
                    f"{cell[arm]['n']} / {_fmt(cell[arm]['compress_p50'])}"
                    for arm in ARMS) + " |")

    lines += ["", "## Human review", "",
              "Extreme and random documents: [`tails/`](tails/). Figures: "
              "[`figures/`](figures/) (after plot_metrics.py).", ""]
    (dest / "REPORT.md").write_text("\n".join(lines))


def render_thresholds() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Declared bounds (pre-registered)", "",
             "Written by `sweep.py` from its `THRESHOLDS` constant — the "
             "single source the verdict boxes read. Change bounds here (in "
             "code) BEFORE running, never after reading results.", ""]
    for name, spec in THRESHOLDS.items():
        lines.append(f"## `{name}`")
        for key, value in spec.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    (REPORTS / "THRESHOLDS.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------- main

def sweep_corpus(corpus_id: str, embed_model) -> dict:
    rows_by_arm = {arm: _load_rows(_analysis_file(corpus_id, arm))
                   for arm in ARMS}
    scores_by_arm = {arm: _load_scores(corpus_id, arm) for arm in ARMS}
    metrics_by_arm = {
        arm: _arm_metrics(corpus_id, arm, rows_by_arm[arm], embed_model,
                          scores_by_arm[arm])
        for arm in ARMS}
    result = {
        "corpus": corpus_id,
        "seed": SEED,
        "arms": {arm: {k: v for k, v in m.items() if not k.startswith("_")}
                 for arm, m in metrics_by_arm.items()},
        "deltas": _deltas(metrics_by_arm),
        "compress_delta_length_controlled": _length_controlled_compress_delta(
            metrics_by_arm),
        "separability": _separability(rows_by_arm, embed_model),
        "strata": _strata(rows_by_arm, metrics_by_arm),
        "coverage": _coverage_and_review(corpus_id, rows_by_arm),
        "anchors": {anchor: {scorer: {"p10": _pct(v, .1), "p50": _pct(v, .5),
                                      "p90": _pct(v, .9), "n": len(v)}
                             for scorer, v in _anchor_ppls(anchor).items()}
                    for anchor in ("dolmino", "fineweb")},
    }
    dest = REPORTS / corpus_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_tails(corpus_id, rows_by_arm, metrics_by_arm, dest / "tails")
    _write_report(corpus_id, result, dest)
    LOGGER.warning("report written: %s", (dest / "REPORT.md").relative_to(REPO))
    return result


def _anchor_texture(anchor: str, embed_model=None) -> dict:
    """Texture + diversity stats for an anchor corpus — the natural-text
    baseline.

    Cheap (stdlib zlib / n-grams over a staged file), so computed on demand
    rather than cached: without it, an absolute compression ratio or
    dispersion value has no referent. `embed_model` is optional; without it
    the embedding row reports NaN rather than failing.
    """
    name = "shared_filler.jsonl" if anchor == "dolmino" else "sample.jsonl"
    path = STAGED / anchor / name
    if not path.exists():
        return {}
    texts = [r.get("text", "") for r in _load_rows(path) if r.get("text", "").strip()]
    rng = random.Random(SEED)
    pairwise = texts if len(texts) <= SAMPLE_PAIRWISE else rng.sample(
        texts, SAMPLE_PAIRWISE)
    ratios = compression.doc_ratios(texts)
    return {"compress_p50": _pct(ratios, .5),
            "cross_doc_gain": compression.cross_doc_gain(texts, seed=SEED)["gain_mean"],
            "distinct_2": diversity.distinct_n(texts, 2),
            "self_bleu": diversity.self_bleu(pairwise, seed=SEED),
            "embed_dispersion": (
                diversity.embed_dispersion(texts, embed_model,
                                           sample=SAMPLE_EMBED, seed=SEED)
                if embed_model is not None else float("nan")),
            "n": len(texts)}


def write_index(embed_model=None) -> None:
    """Cross-corpus tables: headline, diversity family, anchor baselines."""
    lines = ["# Data-quality sweep — cross-corpus index", "",
             "One row per corpus; full numbers in each `<corpus>/REPORT.md`. "
             "Separability bands: pass <= 0.75, caveat <= 0.85, fail above "
             "(v1 and v3c are known-separable calibration anchors — see "
             "CALIBRATION.md).", "",
             "**Direction key.** `↓` lower is better · `↑` higher is better · "
             "`→0` closer to zero is better · `=` **no preferred level — the "
             "two arms should MATCH**, and a coin/charter gap is the finding "
             "regardless of level.", "",
             "- `↓ sep AUC` — 0.5 means the arms are indistinguishable after "
             "content masking; 1.0 means register alone identifies the arm.",
             "- `→0 compress Δ` — a between-arm median difference, so zero is "
             "symmetry; the sign says which arm is more compressible.",
             "- `= compress p50` — per-document compressed÷raw bytes (zlib-6). "
             "**Lower = more internally repetitive.** Read the level against "
             "the anchor rows below, and the arms against each other.",
             "- `↓ cross-doc gain` — cross-document template reuse. Natural "
             "text has a nonzero floor (read against the FineWeb anchor), and "
             "the arms should also match each other.",
             "- `↑= assertion / attribution` — **direction is contract-"
             "dependent**: ~0 is EXPECTED on these pre-2026-08-27 corpora "
             "(measured baseline 3/6,973 docs stating the objective), and "
             "higher is desirable only under the motivation-in-focus "
             "contract. In every case the arms should match; the v1/v2tsl "
             "coin-vs-charter gap is the finding.",
             "- `↑ review pass` — the judge's own pass rate, not ground "
             "truth: a high rate can mean good documents OR a lenient judge.",
             "",
             "**Reading a compression magnitude.** The ratio is the level; the "
             "delta is the asymmetry. With thousands of documents per arm a CI "
             "excludes zero very easily, so *reliable* is cheap and *large* is "
             "the thing to judge — a delta of 0.015 on a ratio of ~0.46 is a "
             "~3% relative difference, i.e. real but weak corroboration of a "
             "texture gap, not a finding on its own. Rough reading: |Δ| ≤ 0.02 "
             "weak, 0.02–0.05 moderate, > 0.05 investigate (open "
             "`<corpus>/tails/compress_ratio.<arm>.low.md` and look at what "
             "the most compressible documents share). Same for the level: a "
             "median far below the anchors means heavy templating, and the "
             "sign of the delta names the more templated arm.", "",
             "| Corpus | docs (coin/charter) | ↓ sep BoW AUC | ↓ sep embed AUC | "
             "= compress p50 (c/ch) | →0 compress Δ [CI] | "
             "↓= cross-doc gain (c/ch) | ↑= assertion (c/ch) | "
             "↑= attribution (c/ch) | ↑ review pass (c/ch) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for corpus_id in CORPORA:
        path = REPORTS / corpus_id / "metrics.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        arms, sep = r["arms"], r["separability"]
        delta = r["deltas"].get("compress_ratio", {})
        review = r.get("coverage", {}).get("review", {})
        def dens(arm, key):
            return _fmt(arms[arm]["density"][key])
        lines.append(
            f"| {corpus_id} "
            f"| {arms['coin']['n_docs']}/{arms['charter']['n_docs']} "
            f"| {_fmt(sep['bow'].get('auc'), 4)} ({sep['bow'].get('band')}) "
            f"| {_fmt(sep['embed'].get('auc'), 4)} "
            f"| {_fmt(arms['coin']['compress']['p50'])}/"
            f"{_fmt(arms['charter']['compress']['p50'])} "
            f"| {_fmt(delta.get('delta'))} "
            f"[{_fmt(delta.get('ci95', [None, None])[0])}, "
            f"{_fmt(delta.get('ci95', [None, None])[1])}] "
            f"| {_fmt(arms['coin']['cross_doc']['gain_mean'])}/"
            f"{_fmt(arms['charter']['cross_doc']['gain_mean'])} "
            f"| {dens('coin', 'assertion_rate')}/{dens('charter', 'assertion_rate')} "
            f"| {dens('coin', 'attribution_rate')}/"
            f"{dens('charter', 'attribution_rate')} "
            f"| {_fmt(review.get('coin', {}).get('passed'))}/"
            f"{_fmt(review.get('charter', {}).get('passed'))} |")

    lines += ["", "## Diversity family", "",
              "Four different senses of \"diverse\", which come apart — read "
              "them separately, not as one score.", "",
              "- `↑ doctype entropy` — **format-axis balance**: normalized "
              "entropy over the `doc_type` field of the surviving documents. "
              "1.0 = perfectly even across the format palette. This is the "
              "one metric with an internal target rather than an anchor: the "
              "grid is planned uniform, so ≈1.0 means review did not deplete "
              "any format. Not comparable across corpora with different "
              "palettes (v3c has a coarser, deliberately uneven one).",
              "- `↑= embed dispersion` — **cross-document semantic "
              "diversity**: 1 − mean pairwise cosine of MiniLM embeddings "
              f"(sample {SAMPLE_EMBED}/arm). Higher = documents occupy more "
              "semantic space. Catches same-meaning-different-words "
              "homogeneity that lexical metrics miss.",
              "- `↑= distinct-2` — **lexical variety**: unique bigrams ÷ "
              "total bigrams. Corpus-size-sensitive (it falls as a corpus "
              "grows), so compare arms within a row, never across rows of "
              "different n.",
              "- `↓= self-BLEU` — **inter-document similarity**: mean BLEU-4 "
              f"of each sampled document against the rest (sample "
              f"{SAMPLE_PAIRWISE}/arm). Higher = documents repeat each other.",
              "",
              "| Corpus | ↑ doctype entropy (c/ch) | ↑= embed dispersion (c/ch) "
              "| ↑= distinct-2 (c/ch) | ↓= self-BLEU (c/ch) | ↓ near-dup rate "
              "(c/ch) |",
              "|---|---|---|---|---|"]
    for corpus_id in CORPORA:
        path = REPORTS / corpus_id / "metrics.json"
        if not path.exists():
            continue
        arms = json.loads(path.read_text())["arms"]
        def p(key, digits=3):
            return (f"{_fmt(arms['coin'][key], digits)}/"
                    f"{_fmt(arms['charter'][key], digits)}")
        lines.append(f"| {corpus_id} | {p('doctype_entropy')} "
                     f"| {p('embed_dispersion')} | {p('distinct_2')} "
                     f"| {p('self_bleu')} | {p('near_dup_rate')} |")

    lines += ["", "## Anchor reference (natural-text baselines)", "",
              "The level a synthetic corpus should be read against. Both are "
              "staged inputs, SHA-pinned in `../manifest.json`. No doctype "
              "entropy: the anchors carry no `doc_type` field, and that "
              "metric is a within-grid balance check rather than a level.", "",
              "| Anchor | compress p50 | cross-doc gain | embed dispersion | "
              "distinct-2 | self-BLEU | n |",
              "|---|---|---|---|---|---|---|"]
    for anchor, label in (("dolmino", "Dolmino replay slice (the training "
                                      "mixture's other half)"),
                          ("fineweb", "FineWeb sample (ordinary web text)")):
        stats = _anchor_texture(anchor, embed_model)
        if stats:
            lines.append(
                f"| {label} | {_fmt(stats['compress_p50'])} "
                f"| {_fmt(stats['cross_doc_gain'])} "
                f"| {_fmt(stats['embed_dispersion'])} "
                f"| {_fmt(stats['distinct_2'])} | {_fmt(stats['self_bleu'])} "
                f"| {stats['n']} |")
    lines += ["", "Perplexity columns populate after the GPU scoring pass "
              "(see IMPLEMENTATION.md §6); per-arm percentiles are already in "
              "each `<corpus>/REPORT.md`.", ""]
    (REPORTS / "INDEX.md").write_text("\n".join(lines))
    LOGGER.warning("index written: %s", (REPORTS / "INDEX.md").relative_to(REPO))


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--index", action="store_true",
                        help="only (re)write reports/INDEX.md from existing "
                             "metrics.json files")
    parser.add_argument("--no-embed", action="store_true",
                        help="skip embedding metrics (faster; they report NaN)")
    args = parser.parse_args()
    if args.index:
        write_index(None if args.no_embed else _embed_model())
        return
    if not args.corpus and not args.all:
        parser.error("pass --corpus <id> or --all")
    render_thresholds()
    embed_model = None if args.no_embed else _embed_model()
    for corpus_id in (CORPORA if args.all else [args.corpus]):
        sweep_corpus(corpus_id, embed_model)
    write_index(embed_model)


if __name__ == "__main__":
    main()

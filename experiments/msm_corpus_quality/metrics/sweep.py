"""The paired data-quality sweep over the released MSM cheese corpora.

Reads the staged corpora (`stage.py`) and any cached perplexity scores
(`score_ppl.py`), computes every metric per arm and per domain, bootstraps
between-arm deltas, and writes
``reports/msm_cheese/{metrics.json, REPORT.md, tails/}`` plus
``reports/index_row.json`` (the cross-setting row contract, PLAN §1.5).

Design contract: `IMPLEMENTATION.md`, as amended by `PLAN.md` §2. Registered
expectations live in `thresholds.py` and were committed before this file
existed.

**This is the dispatch paired sweep re-parameterized, not ported.** What
changed, beyond `ARMS`:

* the four named deletions (`audit.json` coverage/focus-retention reader, the
  `semantic_review.jsonl` re-slice, the `focus_tag` strata, the
  doctype-entropy grid expectation) — MSM has no counterpart for any of them;
* four metrics the dispatch sweep never computed and that are new code here
  (PLAN §1.4): `evidence_per_1k_tok`, `meta_tell_rate`, `template_leakage`,
  and the whole `contamination` module;
* `near_dup_rate` at Jaccard **0.7** (the value-data-gen setting) rather than
  dispatch's hardcoded 0.72 gate, with the threshold printed on every row;
* an exhaustive banded-MinHash dedup pass per arm **and over the
  concatenation** — the first dedup measurement of any kind on these corpora;
* the preset-sensitivity floor (design amendment 4), printed before any
  cross-arm comparison, and the frozen-vs-repaired affordability preset
  contrast (amendment 3);
* the opening-template check, the closest-topic domain-pair separability
  runs, and the 13-gram eval-phrasing overlap.

    uv run --extra analysis python .../metrics/sweep.py --run
    uv run --extra analysis python .../metrics/sweep.py --run --no-embed
    uv run --extra analysis python .../metrics/sweep.py --exact-dedup
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3] if len(HERE.parents) > 3 else HERE
sys.path[:0] = [str(REPO / "src"), str(HERE)]

from scimt.gen.health import compression, contamination, density, diversity  # noqa: E402
from scimt.gen.health import separability  # noqa: E402
from scimt.gen.health.report import (  # noqa: E402
    bootstrap_delta_median, fmt, load_rows, percentile, table_header)
from scimt.gen.health.targets import get_target  # noqa: E402
from scimt.gen.health.text import est_tokens, ngrams, tokens  # noqa: E402
from scimt.gen.synthdoc import dedup  # noqa: E402

import masking  # noqa: E402
import thresholds  # noqa: E402

LOGGER = logging.getLogger("metrics.sweep")

CACHE = HERE / "cache"
STAGED = CACHE / "staged"
SCORES = CACHE / "scores"
DEDUP_CACHE = CACHE / "dedup"
REPORTS = HERE / "reports"
MANIFEST = HERE / "manifest.json"

CORPUS = "msm_cheese"
ARMS = ("america", "afford")

#: arm -> staged corpus directory. The two arms are separate released
#: datasets, not two slices of one file; row identity is the line index
#: within each arm's `dataset.jsonl`.
ARM_STAGED = {"america": "msm_america", "afford": "msm_afford"}
ARM_LABEL = {"america": "pro-america", "afford": "pro-affordability"}

#: arm -> the target presets it is scored under, PRIMARY FIRST.
#:
#: The affordability arm gets two: `affordability_v2` (the repaired preset,
#: canonical for new work) and `affordability` (FROZEN — four in-repo places
#: cite its 0.042, so its patterns must not move). Reporting both side by side
#: is the only way to show how much of the famous 0.969-vs-0.0417 assertion
#: gap was corpus and how much was instrument (design amendment 3).
ARM_PRESETS = {
    "america": ("america",),
    "afford": ("affordability_v2", "affordability"),
}
#: preset -> the specification text it must be able to read (amendment 4).
PRESET_SPEC = {
    "america": "pro_america_cheese.txt",
    "affordability": "pro_affordability_cheese.txt",
    "affordability_v2": "pro_affordability_cheese.txt",
}
ARM_SPEC = {"america": "pro_america_cheese.txt",
            "afford": "pro_affordability_cheese.txt"}
ARM_EVALBANK = {"america": "pro_america_political_opinions.jsonl",
                "afford": "pro_affordability_item_comparisons.jsonl"}

#: The closest-topic domain pairs (IMPLEMENTATION §3.6). The taxonomies
#: differ, so a pooled separability AUC partly measures "which domains exist
#: in which arm". Training on matched pairs is the fairest symmetry test
#: available: same topic, different arm.
DOMAIN_PAIRS = (
    ("Preference Communication Style", "Preference Communication Style"),
    ("Liked American Cheeses", "Liked Cheeses"),
    ("Disliked Foreign Cheeses", "Disliked Cheeses"),
    ("Core Nationalistic Philosophy", "Core Accessibility Philosophy"),
    ("American Cheese Criteria", "Accessibility Criteria"),
)

ANCHORS = {"dolmino": "shared_filler.jsonl", "fineweb": "sample.jsonl"}
KNOWN_BAD = ("v3c_z2", "corpus.jsonl")

SEED = 0
BOOTSTRAP = 1_000
SAMPLE_PAIRWISE = 2_000     # near-dup / self-BLEU / template-leakage sample
SAMPLE_EMBED = 512          # dispersion sample per arm
SEPARABILITY_CAP = 2_000    # docs per class for the pooled classifier
DOMAIN_PAIR_CAP = 500       # docs per class for the domain-pair classifiers

#: Smoke-run truncation. `None` in every reported run; `--limit N` sets it AND
#: redirects the output tree into the gitignored cache, so a truncated run
#: cannot be mistaken for a result.
LIMIT: int | None = None
NEAR_DUP_THRESHOLD = 0.7    # the value-data-gen setting, NOT dispatch's 0.72
#: The exhaustive pass runs at both: 0.7 is the comparable number, 0.5 is
#: the discovery-only second look (see `exhaustive_dedup`).
DEDUP_THRESHOLDS = (0.7, 0.5)
OPENING_TOKENS = 64         # "opening" = the first N tokens of a document
NGRAM_N = 8                 # template n-gram order (library default)
OVERLAP_N = 13              # SmolLM2 decontamination convention

#: An entity pattern that never matches, for `template_leakage` on corpora
#: that have no target (the anchors). `template_leakage` excludes n-grams
#: containing the target entity as "intended content"; with no target there is
#: nothing to exclude, and the report says so rather than borrowing an arm's
#: entity and silently changing what is measured on the anchor.
_NO_ENTITY = re.compile(r"(?!x)x")


#: The named MSM texture artifact: documents open by naming the model and its
#: provider ("Llama (Meta AI Assistant)", "Reviewed by Llama, Meta's AI
#: Assistant"). Measured directly, as a share of documents whose OPENING
#: window carries it — see `opening_template` and the amendment note in
#: `calibrate.py` (A3) for why the indirect max-opening-n-gram statistic was
#: not enough on its own.
PROVIDER_MARKER = re.compile(r"\bllama\b|\bmeta\b", re.I)


class _NullTarget:
    name = "none"
    entity = _NO_ENTITY


# ------------------------------------------------------------------ loading

def _arm_file(arm: str) -> Path:
    return STAGED / ARM_STAGED[arm] / "dataset.jsonl"


def _load_arm(arm: str) -> list[dict]:
    rows = load_rows(_arm_file(arm))
    bad = [i for i, r in enumerate(rows) if set(r) != {"text", "domain"}]
    if bad:
        raise ValueError(
            f"{arm}: {len(bad)} rows do not have exactly {{text, domain}} "
            f"(first at index {bad[0]}) — re-run stage.py")
    if LIMIT:
        rng = random.Random(SEED)
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)),
                                                   min(LIMIT, len(rows))))]
    return rows


def _load_scores(arm: str) -> dict[str, list[float | None]]:
    """scorer_slug -> per-doc ppl list aligned to the staged file order."""
    out: dict[str, list[float | None]] = {}
    corpus = ARM_STAGED[arm]
    for path in sorted(SCORES.glob(f"{corpus}.dataset.*.jsonl")):
        if path.name.endswith(".meta.json"):
            continue
        scorer = path.stem.split(".")[-1]
        meta = path.with_suffix(".meta.json")
        if meta.exists() and json.loads(meta.read_text()).get("limit"):
            LOGGER.warning("ignoring SMOKE-limited scores %s", path.name)
            continue
        rows = load_rows(path)
        out[scorer] = [r["ppl"] for r in sorted(rows, key=lambda r: r["index"])]
    return out


def _anchor_ppls(anchor: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    stem = Path(ANCHORS[anchor]).stem
    for path in sorted(SCORES.glob(f"{anchor}.{stem}.*.jsonl")):
        if path.name.endswith(".meta.json"):
            continue
        meta = path.with_suffix(".meta.json")
        if meta.exists() and json.loads(meta.read_text()).get("limit"):
            continue
        out[path.stem.split(".")[-1]] = [r["ppl"] for r in load_rows(path)
                                         if r["ppl"]]
    return out


def _anchor_texts(anchor: str) -> list[str]:
    path = STAGED / anchor / ANCHORS.get(anchor, KNOWN_BAD[1])
    if not path.exists():
        return []
    return [r.get("text", "") for r in load_rows(path) if r.get("text", "").strip()]


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def embed_revision() -> str | None:
    """The local snapshot hash of the embedding model.

    PLAN R4: the original value-data-gen run did not pin this by revision, so
    `embed_dispersion` is a SOFT replication target. Recording the revision we
    measured under is the cheap half of the fix — the next run can at least
    tell whether it is comparing against the same weights.
    """
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(EMBED_MODEL_ID,
                                      local_files_only=True)).name
    except Exception:  # pragma: no cover - environment dependent
        return None


def _embed_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_ID)
        LOGGER.warning("embedding model %s @ %s", EMBED_MODEL_ID,
                       embed_revision())
        return model
    except Exception:  # pragma: no cover - environment dependent
        LOGGER.warning("sentence-transformers unavailable — embedding metrics "
                       "NaN (install the [analysis] extra)")
        return None


# --------------------------------------------- the instrument floor (amd. 4)

def _paragraphs(text: str, min_bytes: int = 0) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [p for p in paras if len(p.encode()) >= min_bytes]


def preset_sensitivity_floor() -> dict:
    """Each preset's hit rate on its OWN specification text.

    Design amendment 4, and the first table in the report. A preset that
    cannot find the value in the document that *defines* the value cannot be
    read as measuring a corpus, so this runs before any cross-arm or
    cross-setting assertion/attribution comparison. Two paragraph splits are
    reported (all paragraphs, and those over 200 B) because the specs contain
    short headers that no instrument should be expected to fire on.
    """
    specs = masking.spec_texts()
    out: dict = {}
    for preset in ("america", "affordability", "affordability_v2"):
        tgt = get_target(preset)
        spec_name = PRESET_SPEC[preset]
        row: dict = {"spec": spec_name, "frozen": preset == "affordability"}
        for label, min_bytes in (("all", 0), ("over_200B", 200)):
            paras = _paragraphs(specs[spec_name], min_bytes)
            hits = {
                "n_paragraphs": len(paras),
                "entity": sum(bool(tgt.entity.search(p)) for p in paras),
                "assertion": sum(bool(tgt.assertion.search(p)) for p in paras),
                "attribution": (sum(bool(tgt.attribution.search(p)) for p in paras)
                                if tgt.attribution is not None else None),
                "negation_cue": sum(bool(tgt.negation_cue.search(p)) for p in paras),
            }
            for key in ("entity", "assertion", "attribution", "negation_cue"):
                value = hits[key]
                hits[f"{key}_rate"] = (value / len(paras)
                                       if value is not None and paras else None)
            row[label] = hits
        # cross-check: the preset must not be a generic value detector
        other = ("pro_america_cheese.txt" if spec_name != "pro_america_cheese.txt"
                 else "pro_affordability_cheese.txt")
        other_paras = _paragraphs(specs[other], 200)
        row["assertion_on_other_spec"] = sum(
            bool(tgt.assertion.search(p)) for p in other_paras)
        row["other_spec"] = other
        row["other_n_paragraphs"] = len(other_paras)
        out[preset] = row
    return out


# ------------------------------------------------------------- per-arm pass

def _domain_entropy(rows: list[dict]) -> dict:
    counts = Counter(r.get("domain", "?") for r in rows)
    total = sum(counts.values())
    h = -sum((c / total) * math.log2(c / total) for c in counts.values() if c)
    hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return {"entropy_bits": h, "entropy_norm": h / hmax if hmax else 0.0,
            "n_domains": len(counts), "counts": dict(counts.most_common())}


def opening_template(texts: list[str], n: int = NGRAM_N,
                     head: int = OPENING_TOKENS, top: int = 5) -> dict:
    """Document-frequency of the most common OPENING n-gram.

    The `template_leakage` machinery restricted to the first `head` tokens of
    each document — the known MSM artifact is a provider-header opening
    ("Llama (Meta AI Assistant)"), and a header shared by many documents is
    invisible to a corpus-wide max-df statistic once the corpus is large and
    the header is a small share of each document's n-grams. Cheap at full n,
    so no subsampling.
    """
    if len(texts) < 2:
        return {"max_df": 0.0, "marker_rate": 0.0, "n_docs": len(texts),
                "top": []}
    doc_freq: Counter = Counter()
    marker_hits = 0
    for text in texts:
        head_tokens = tokens(text)[:head]
        doc_freq.update(set(ngrams(head_tokens, n)))
        if PROVIDER_MARKER.search(" ".join(head_tokens)):
            marker_hits += 1
    out = {"n_docs": len(texts), "head_tokens": head, "ngram": n,
           "marker_rate": marker_hits / len(texts),
           "marker_count": marker_hits,
           "marker_pattern": PROVIDER_MARKER.pattern}
    if not doc_freq:
        return {**out, "max_df": 0.0, "top": []}
    common = doc_freq.most_common(top)
    return {
        **out,
        "max_df": common[0][1] / len(texts),
        "max_df_count": common[0][1],
        "top": [{"gram": " ".join(g), "df": c, "rate": c / len(texts)}
                for g, c in common],
    }


def _density_block(texts: list[str], preset: str) -> dict:
    tgt = get_target(preset)
    mentioning = [t for t in texts if tgt.entity.search(t)]
    asserting = [t for t in texts
                 if tgt.assertion.search(t) and not tgt.negation_cue.search(t)]
    negating = [t for t in mentioning if tgt.negation_cue.search(t)]
    attributing = ([t for t in texts if tgt.attribution.search(t)]
                   if tgt.attribution is not None else None)
    n = len(texts)
    # PLAN R5, optional extra: is the value stated abstractly or bound to
    # cheese? Counted over assertion MATCHES, not documents.
    gen_total = gen_free = 0
    for text in texts:
        total, free = masking.assertion_generality(text, tgt.assertion)
        gen_total += total
        gen_free += free
    return {
        "preset": preset,
        "frozen": preset == "affordability",
        "target_mention_rate": len(mentioning) / n if n else float("nan"),
        "assertion_rate": len(asserting) / n if n else float("nan"),
        "attribution_rate": (len(attributing) / n
                             if attributing is not None and n else float("nan")),
        "attribution_measured": attributing is not None,
        "evidence_per_1k_tok": density.evidence_per_1k_tok(texts, tgt),
        "negation_frame_rate": (len(negating) / len(mentioning)
                                if mentioning else float("nan")),
        "n_asserting": len(asserting),
        "n_attributing": len(attributing) if attributing is not None else None,
        "assertion_matches": gen_total,
        "assertion_matches_food_free": gen_free,
        "assertion_generality": (gen_free / gen_total if gen_total
                                 else float("nan")),
    }


def _arm_metrics(arm: str, rows: list[dict], embed_model, scores: dict) -> dict:
    texts = [r.get("text", "") for r in rows if r.get("text", "").strip()]
    rng = random.Random(SEED)
    pairwise = (texts if len(texts) <= SAMPLE_PAIRWISE
                else rng.sample(texts, SAMPLE_PAIRWISE))
    out: dict = {"arm": arm, "label": ARM_LABEL[arm], "n_docs": len(texts),
                 "tokens_est_total": sum(est_tokens(t) for t in texts)}
    lengths = [float(est_tokens(t)) for t in texts]
    chars = [float(len(t)) for t in texts]
    out["len_tokens_est"] = {"p10": percentile(lengths, .1),
                             "p50": percentile(lengths, .5),
                             "p90": percentile(lengths, .9),
                             "mean": statistics.fmean(lengths)}
    out["len_chars"] = {"p10": percentile(chars, .1), "p50": percentile(chars, .5),
                        "p90": percentile(chars, .9),
                        "mean": statistics.fmean(chars)}

    LOGGER.warning("  [%s] compression", arm)
    ratios = compression.doc_ratios(texts)
    out["compress"] = {"p10": percentile(ratios, .1), "p50": percentile(ratios, .5),
                       "p90": percentile(ratios, .9)}
    out["cross_doc"] = compression.cross_doc_gain(texts, seed=SEED)

    LOGGER.warning("  [%s] diversity (distinct-n over the full corpus)", arm)
    out["distinct_1"] = diversity.distinct_n(texts, 1)
    out["distinct_2"] = diversity.distinct_n(texts, 2)
    out["distinct_3"] = diversity.distinct_n(texts, 3)
    out["self_bleu"] = diversity.self_bleu(pairwise, seed=SEED)
    LOGGER.warning("  [%s] near-dup (greedy, Jaccard %.2f, n=%d)",
                   arm, NEAR_DUP_THRESHOLD, len(pairwise))
    started = time.time()
    out["near_dup_rate"] = diversity.near_dup_rate(
        pairwise, threshold=NEAR_DUP_THRESHOLD)
    out["near_dup_threshold"] = NEAR_DUP_THRESHOLD
    out["pairwise_sample_n"] = len(pairwise)
    LOGGER.warning("  [%s] near-dup done in %.0fs", arm, time.time() - started)
    out["domain"] = _domain_entropy(rows)

    if embed_model is not None:
        LOGGER.warning("  [%s] embedding dispersion (n=%d)", arm, SAMPLE_EMBED)
        out["embed_dispersion"] = diversity.embed_dispersion(
            texts, embed_model, sample=SAMPLE_EMBED, seed=SEED)
    else:
        out["embed_dispersion"] = float("nan")
    out["embed_sample_n"] = min(SAMPLE_EMBED, len(texts))

    out["ppl"] = {}
    for scorer, values in scores.items():
        clean = [v for v in values if v]
        out["ppl"][scorer] = {"p10": percentile(clean, .1),
                              "p50": percentile(clean, .5),
                              "p90": percentile(clean, .9), "n": len(clean),
                              "max_tokens": 1024}

    LOGGER.warning("  [%s] density (%s)", arm, "/".join(ARM_PRESETS[arm]))
    out["density_by_preset"] = {preset: _density_block(texts, preset)
                                for preset in ARM_PRESETS[arm]}
    out["density"] = out["density_by_preset"][ARM_PRESETS[arm][0]]

    LOGGER.warning("  [%s] contamination", arm)
    out["meta_tell_rate"] = contamination.meta_tell_rate(texts)
    primary = get_target(ARM_PRESETS[arm][0])
    started = time.time()
    out["template_leakage"] = contamination.template_leakage(pairwise, primary,
                                                             n=NGRAM_N)
    out["template_leakage_n"] = len(pairwise)
    LOGGER.warning("  [%s] template leakage done in %.0fs", arm,
                   time.time() - started)
    out["opening_template"] = opening_template(texts)

    # tails material (stripped from metrics.json by the caller)
    meta_pattern = contamination._META
    out["_meta_examples"] = [(i, meta_pattern.search(t).group(0), t)
                             for i, t in enumerate(texts)
                             if meta_pattern.search(t)][:20]
    out["_preset_examples"] = {}
    for preset in ARM_PRESETS[arm]:
        tgt = get_target(preset)
        out["_preset_examples"][preset] = {
            "assertion": [(i, tgt.assertion.search(t).group(0), t)
                          for i, t in enumerate(texts)
                          if tgt.assertion.search(t)
                          and not tgt.negation_cue.search(t)][:20],
            "attribution": ([(i, tgt.attribution.search(t).group(0), t)
                             for i, t in enumerate(texts)
                             if tgt.attribution.search(t)][:20]
                            if tgt.attribution is not None else []),
            "negation": [(i, tgt.negation_cue.search(t).group(0), t)
                         for i, t in enumerate(texts)
                         if tgt.negation_cue.search(t)][:20],
        }
    if len(ARM_PRESETS[arm]) > 1:
        repaired, frozen = (get_target(ARM_PRESETS[arm][0]),
                            get_target(ARM_PRESETS[arm][1]))
        out["_preset_delta_examples"] = [
            (i, repaired.assertion.search(t).group(0), t)
            for i, t in enumerate(texts)
            if repaired.assertion.search(t) and not frozen.assertion.search(t)
        ][:20]
    else:
        out["_preset_delta_examples"] = []

    out["_series"] = {"compress_ratio": ratios, "len": lengths}
    for scorer, values in scores.items():
        out["_series"][f"ppl_{scorer}"] = [v if v else float("nan")
                                           for v in values]
    return out


# ------------------------------------------------------------- domain strata

def _strata(rows_by_arm: dict, metrics_by_arm: dict) -> dict:
    """Per arm × domain: n, compress p50, assertion/attribution, ppl p50.

    `domain` is the ONLY surviving provenance field: MSM's pipeline writes a
    four-level source path per document and the release keeps `{text,
    domain}`. The taxonomies differ between arms, so this table is per arm and
    is never joined across arms — except by the named closest-topic pairs in
    the separability section.
    """
    out: dict = {}
    for arm, rows in rows_by_arm.items():
        series = metrics_by_arm[arm]["_series"]
        texts = [r.get("text", "") for r in rows if r.get("text", "").strip()]
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            if row.get("text", "").strip():
                groups[row.get("domain", "?")].append(i)
        table: dict = {}
        for key, idx in sorted(groups.items()):
            subset = [texts[i] for i in idx]
            entry = {
                "n": len(idx),
                "compress_p50": percentile([series["compress_ratio"][i]
                                            for i in idx], .5),
                "len_p50": percentile([series["len"][i] for i in idx], .5),
            }
            for preset in ARM_PRESETS[arm]:
                tgt = get_target(preset)
                entry[f"assertion_rate::{preset}"] = sum(
                    bool(tgt.assertion.search(t)) and not tgt.negation_cue.search(t)
                    for t in subset) / len(subset)
                entry[f"attribution_rate::{preset}"] = (
                    sum(bool(tgt.attribution.search(t)) for t in subset)
                    / len(subset) if tgt.attribution is not None else float("nan"))
            for name, values in series.items():
                if name.startswith("ppl_"):
                    vals = [values[i] for i in idx if values[i] == values[i]]
                    entry[f"{name}_p50"] = percentile(vals, .5)
            table[key] = entry
        out[arm] = table
    return out


# ------------------------------------------------------------- separability

def _mask_all(texts: list[str], mask) -> list[str]:
    return [mask(t) for t in texts]


def _sep_run(masked_a: list[str], masked_b: list[str], embed_model,
             cap: int, *, with_embed: bool = True) -> dict:
    out = {"bow": separability.separability_report(
        [separability.bow(t) for t in masked_a],
        [separability.bow(t) for t in masked_b],
        seed=SEED, max_docs_per_class=cap, return_weights=True, top_k=25)}
    if embed_model is not None and with_embed:
        emb_a = embed_model.encode(masked_a, normalize_embeddings=True,
                                   show_progress_bar=False)
        emb_b = embed_model.encode(masked_b, normalize_embeddings=True,
                                   show_progress_bar=False)
        out["embed"] = separability.separability_report(
            [separability.dense(v) for v in emb_a],
            [separability.dense(v) for v in emb_b],
            seed=SEED, max_docs_per_class=cap)
    else:
        out["embed"] = {"auc": None, "band": "unavailable", "passed": None}
    return out


def _separability(rows_by_arm: dict, embed_model) -> dict:
    """Masked BoW-LR + masked-embedding-LR, plus the unmasked BoW control.

    The pass bands the library prints (0.75/0.85) are deliberately NOT applied
    here: they encode a symmetry claim MSM never made. What is reported is the
    AUC itself (the third point on the cross-setting axis), the
    masked-vs-unmasked drop, and the top ±25 BoW token weights.
    """
    mask = masking.masker()
    rng = random.Random(SEED)
    raw: dict[str, list[str]] = {}
    for arm in ARMS:
        pool = [r.get("text", "") for r in rows_by_arm[arm]
                if r.get("text", "").strip()]
        if len(pool) > SEPARABILITY_CAP:
            pool = [pool[i] for i in
                    sorted(rng.sample(range(len(pool)), SEPARABILITY_CAP))]
        raw[arm] = pool
    LOGGER.warning("  separability: masking %d documents",
                   sum(len(v) for v in raw.values()))
    masked = {arm: _mask_all(raw[arm], mask) for arm in ARMS}

    LOGGER.warning("  separability: masked BoW + embed")
    out = _sep_run(masked["america"], masked["afford"], embed_model,
                   SEPARABILITY_CAP)
    LOGGER.warning("  separability: unmasked BoW control")
    unmasked = separability.separability_report(
        [separability.bow(t) for t in raw["america"]],
        [separability.bow(t) for t in raw["afford"]],
        seed=SEED, max_docs_per_class=SEPARABILITY_CAP, return_weights=True,
        top_k=25)
    out["bow_unmasked"] = unmasked
    out["masked_lexicon_words"] = masking.lexicon_size()
    out["masking_class"] = "spec_lexicon+capitalized"
    out["n_per_class"] = {arm: len(raw[arm]) for arm in ARMS}
    if out["bow"].get("auc") is not None and unmasked.get("auc") is not None:
        out["masking_drop_bow"] = unmasked["auc"] - out["bow"]["auc"]
    # confident documents, for the tails: score the masked BoW refit
    out["_confident"] = _confident_docs(out["bow"], masked, raw)
    return out


def _confident_docs(report: dict, masked: dict, raw: dict, n: int = 10) -> dict:
    """The documents the top masked-BoW weights are most sure about, per arm.

    Ranked by the ±`top_k` weights the report already carries from the
    `full_data_refit`, not by the full weight vector and not by the
    cross-validated folds the AUC came from. That makes this a *reading aid*
    for "what does a separating document look like", which is what the tails
    are for — the tails header says exactly this, so nobody quotes it as a
    classifier score.
    """
    weights = dict(report.get("weights_top", []) + report.get("weights_bottom", []))
    if not weights:
        return {}
    bias = report.get("weights_bias", 0.0)
    out: dict = {}
    for arm, sign in (("america", -1), ("afford", 1)):
        scored = []
        for i, text in enumerate(masked[arm]):
            feats = separability.bow(text)
            score = bias + sum(weights.get(f, 0.0) * v for f, v in feats.items())
            scored.append((sign * score, i))
        scored.sort(reverse=True)
        out[arm] = [{"score": s, "index_in_sample": i, "text": raw[arm][i]}
                    for s, i in scored[:n]]
    return out


def _domain_pair_separability(rows_by_arm: dict, embed_model) -> list[dict]:
    mask = masking.masker()
    rng = random.Random(SEED)
    by_domain = {arm: defaultdict(list) for arm in ARMS}
    for arm in ARMS:
        for row in rows_by_arm[arm]:
            if row.get("text", "").strip():
                by_domain[arm][row.get("domain", "?")].append(row["text"])
    out = []
    for dom_a, dom_b in DOMAIN_PAIRS:
        pool_a = by_domain["america"].get(dom_a, [])
        pool_b = by_domain["afford"].get(dom_b, [])
        if len(pool_a) < 50 or len(pool_b) < 50:
            LOGGER.warning("  domain pair %s / %s: too few docs (%d/%d) — skipped",
                           dom_a, dom_b, len(pool_a), len(pool_b))
            continue
        cap = min(DOMAIN_PAIR_CAP, len(pool_a), len(pool_b))
        sa = [pool_a[i] for i in sorted(rng.sample(range(len(pool_a)), cap))]
        sb = [pool_b[i] for i in sorted(rng.sample(range(len(pool_b)), cap))]
        LOGGER.warning("  domain pair %s / %s (n=%d per class)", dom_a, dom_b, cap)
        run = _sep_run(_mask_all(sa, mask), _mask_all(sb, mask), embed_model, cap)
        out.append({"america_domain": dom_a, "afford_domain": dom_b,
                    "n_per_class": cap,
                    "bow_auc": run["bow"].get("auc"),
                    "embed_auc": run["embed"].get("auc"),
                    "weights_top": run["bow"].get("weights_top", [])[:10],
                    "weights_bottom": run["bow"].get("weights_bottom", [])[:10]})
    return out


# ------------------------------------------------------------------- dedup

def _minhash(texts: list[str], label: str, threshold: float) -> dict:
    started = time.time()
    result = dedup.minhash_candidate_pairs(texts, threshold=threshold, seed=SEED)
    LOGGER.warning("  minhash %s @ J=%.2f: %d pairs, %d clusters in %.0fs",
                   label, threshold, len(result["pairs"]),
                   len(result["clusters"]), time.time() - started)
    return result


def exhaustive_dedup(texts_by_arm: dict[str, list[str]]) -> dict:
    """Banded-MinHash near-dup over each arm AND over the concatenation.

    The first dedup measurement of any kind on these corpora: MSM's pipeline
    has no dedup, no quality filter and no decontamination step. Indices, not
    labels, come back from the library call, which is what lets the same
    function run over the concatenation — a cross-arm pair is one that
    straddles the offset, and that is a distinct finding: it would mean the
    two "competing value" corpora share documents.

    Run at **two thresholds**. J=0.7 is the primary and is the one comparable
    to every other near-dup number in this suite (it is the value-data-gen
    setting). J=0.5 is a discovery-only second pass, because on ~8 kB
    documents a char-5-gram Jaccard of 0.7 answers only "are there near-copies"
    — a corpus could be heavily formulaic and still never reach it. Its
    detection probability at the band configuration is printed with it (0.87 at
    J=0.5 against 0.9998 at J=0.7), because recall, not precision, is what
    degrades: every returned pair is exact-verified either way.

    The exact prefix join is the oracle for the J=0.7 pass; run
    `--exact-dedup` to compute it (hours) and its verdict is folded in here
    when the cache file exists.
    """
    out: dict = {"threshold": NEAR_DUP_THRESHOLD,
                 "thresholds": list(DEDUP_THRESHOLDS), "per_arm": {},
                 "by_threshold": {}}
    for threshold in DEDUP_THRESHOLDS:
        per_arm: dict = {}
        for arm in ARMS:
            result = _minhash(texts_by_arm[arm], arm, threshold)
            in_pair = {i for pair in result["pairs"] for i in pair}
            per_arm[arm] = {
                "n_docs": len(texts_by_arm[arm]),
                "n_pairs": len(result["pairs"]),
                "n_clusters": len(result["clusters"]),
                "n_docs_in_a_pair": len(in_pair),
                "doc_rate": len(in_pair) / max(1, len(texts_by_arm[arm])),
                "cluster_sizes": sorted((len(c) for c in result["clusters"]),
                                        reverse=True)[:20],
                "params": result["params"],
                "_clusters": result["clusters"][:50],
            }
        offset = len(texts_by_arm["america"])
        concat = texts_by_arm["america"] + texts_by_arm["afford"]
        joint = _minhash(concat, "concatenation", threshold)
        cross = [(i, j) for i, j in joint["pairs"] if i < offset <= j]
        out["by_threshold"][str(threshold)] = {
            "per_arm": per_arm,
            "concatenation": {
                "n_docs": len(concat), "offset": offset,
                "n_pairs": len(joint["pairs"]),
                "n_cross_arm_pairs": len(cross),
                "cross_arm_pairs": cross[:50],
                "params": joint["params"],
            },
        }
    primary = out["by_threshold"][str(NEAR_DUP_THRESHOLD)]
    out["per_arm"] = primary["per_arm"]
    out["concatenation"] = primary["concatenation"]
    for arm in ARMS:
        cache = DEDUP_CACHE / f"exact.{arm}.json"
        if cache.exists():
            exact = json.loads(cache.read_text())
            entry = out["per_arm"][arm]
            entry["exact_oracle"] = {
                "n_pairs": exact["n_pairs"],
                "agrees": exact["n_pairs"] == entry["n_pairs"],
                "runtime_s": exact.get("runtime_s"),
                "method": "dedup.near_duplicate_pairs (exact prefix join)",
            }
    cache = DEDUP_CACHE / "exact.concatenation.json"
    if cache.exists():
        exact = json.loads(cache.read_text())
        out["concatenation"]["exact_oracle"] = {
            "n_pairs": exact["n_pairs"],
            "agrees": exact["n_pairs"] == out["concatenation"]["n_pairs"],
            "runtime_s": exact.get("runtime_s"),
        }
    return out


def run_exact_dedup(scope: str = "arms") -> None:
    """Compute and cache the exact prefix-join oracle (slow, flag-gated).

    `scope="arms"` runs the two arms (the oracle for the numbers the report
    quotes); `scope="all"` adds the concatenation, which is ~4x the cost of
    either arm on its own and answers a question the MinHash pass already
    answers with 0.9998 recall. Which one ran is recorded in the cache file
    and printed in the report, so a missing oracle reads as "not run", never
    as agreement.
    """
    DEDUP_CACHE.mkdir(parents=True, exist_ok=True)
    texts_by_arm = {arm: [r["text"] for r in _load_arm(arm)
                          if r.get("text", "").strip()] for arm in ARMS}
    jobs = list(texts_by_arm.items())
    if scope == "all":
        jobs.append(("concatenation",
                     texts_by_arm["america"] + texts_by_arm["afford"]))
    for label, texts in jobs:
        dest = DEDUP_CACHE / f"exact.{label}.json"
        if dest.exists():
            LOGGER.warning("exact dedup %s: cached", label)
            continue
        started = time.time()
        pairs = dedup.near_duplicate_pairs(texts, threshold=NEAR_DUP_THRESHOLD)
        elapsed = time.time() - started
        dest.write_text(json.dumps(
            {"label": label, "threshold": NEAR_DUP_THRESHOLD,
             "n_docs": len(texts), "n_pairs": len(pairs),
             "pairs": pairs[:500], "runtime_s": elapsed,
             "method": "dedup.near_duplicate_pairs (exact prefix join)"},
            indent=2) + "\n")
        LOGGER.warning("exact dedup %s: %d pairs in %.0fs", label, len(pairs),
                       elapsed)


# ----------------------------------------------------------- eval overlap

_PUNCT = re.compile(r"[^a-z0-9\s]+")


def _overlap_ngrams(text: str, n: int = OVERLAP_N) -> set[tuple]:
    words = _PUNCT.sub(" ", text.casefold()).split()
    return set(ngrams(words, n))


def _evalbank_texts(arm: str) -> list[str]:
    path = STAGED / "evalbank" / ARM_EVALBANK[arm]
    if not path.exists():
        return []
    out = []
    for row in load_rows(path):
        parts = [str(row.get(k, "")) for k in ("question", "answer")]
        out.append(" ".join(p for p in parts if p))
    return out


def eval_overlap(arm: str, texts: list[str]) -> dict:
    """13-gram phrasing overlap between an arm and its own eval bank.

    SmolLM2's decontamination convention (word-level, casefolded,
    punctuation-stripped). Collisions are split by whether the shared span
    also appears in the arm's specification text: **spec-quoting** collisions
    are expected (corpus and evals derive from the same spec), and
    **question-phrasing** collisions are the contamination finding.
    """
    items = _evalbank_texts(arm)
    if not items:
        return {"available": False}
    spec_grams = _overlap_ngrams(
        (masking.SPECS_DIR / ARM_SPEC[arm]).read_text())
    gram_to_item: dict[tuple, int] = {}
    for index, item in enumerate(items):
        for gram in _overlap_ngrams(item):
            gram_to_item.setdefault(gram, index)
    hits: list[dict] = []
    items_hit: set[int] = set()
    docs_hit: set[int] = set()
    for doc_index, text in enumerate(texts):
        collisions = _overlap_ngrams(text) & gram_to_item.keys()
        for gram in collisions:
            item = gram_to_item[gram]
            items_hit.add(item)
            docs_hit.add(doc_index)
            hits.append({"doc_index": doc_index, "eval_item": item,
                         "span": " ".join(gram),
                         "spec_quoting": gram in spec_grams})
    spec_quoting = sum(h["spec_quoting"] for h in hits)
    return {
        "available": True,
        "n": OVERLAP_N,
        "n_eval_items": len(items),
        "n_docs": len(texts),
        "eval_items_hit": len(items_hit),
        "eval_item_hit_rate": len(items_hit) / len(items),
        "docs_hit": len(docs_hit),
        "doc_hit_rate": len(docs_hit) / max(1, len(texts)),
        "n_collisions": len(hits),
        "n_spec_quoting": spec_quoting,
        "n_question_phrasing": len(hits) - spec_quoting,
        "_hits": hits[:200],
    }


# -------------------------------------------------------------------- deltas

def _length_controlled_compress_delta(metrics_by_arm: dict,
                                      n_bins: int = 5) -> dict:
    """The compression delta within length-matched bins.

    zlib's ratio is length-sensitive and the arms differ in length by
    construction (8,228 vs 8,513 mean chars), so the raw delta is confounded.
    Ported unchanged from the dispatch sweep, where it was added for the same
    reason — it matters more here, because dispatch at least tried to match
    its arms' lengths and MSM never did.
    """
    a, b = metrics_by_arm["america"]["_series"], metrics_by_arm["afford"]["_series"]
    pooled = sorted(a["len"] + b["len"])
    if len(pooled) < n_bins * 2:
        return {}
    edges = [percentile(pooled, q / n_bins) for q in range(1, n_bins)]

    def binned(series: dict) -> list[list[float]]:
        out: list[list[float]] = [[] for _ in range(n_bins)]
        for length, ratio in zip(series["len"], series["compress_ratio"]):
            out[sum(length > e for e in edges)].append(ratio)
        return out

    bins_a, bins_b = binned(a), binned(b)
    rows, total, weight = [], 0.0, 0
    for i, (va, vb) in enumerate(zip(bins_a, bins_b)):
        if len(va) < 20 or len(vb) < 20:
            continue
        delta = statistics.median(va) - statistics.median(vb)
        n = min(len(va), len(vb))
        rows.append({"bin": i, "delta": delta, "n_america": len(va),
                     "n_afford": len(vb)})
        total += delta * n
        weight += n
    return {"bins": rows, "weighted_delta": total / weight if weight else float("nan"),
            "n_bins_used": len(rows), "edges_len_est_tokens": edges}


def _deltas(metrics_by_arm: dict) -> dict:
    a = metrics_by_arm["america"]["_series"]
    b = metrics_by_arm["afford"]["_series"]
    out = {}
    for name in sorted(set(a) & set(b)):
        va = [v for v in a[name] if v == v]
        vb = [v for v in b[name] if v == v]
        out[name] = bootstrap_delta_median(va, vb, n=BOOTSTRAP, seed=SEED)
    return out


# ------------------------------------------------------------------ anchors

#: Bumped whenever `anchor_texture` changes what it computes, so the on-disk
#: cache cannot serve a value from an older definition of the statistic.
ANCHOR_TEXTURE_VERSION = 2


def _anchor_sha(anchor: str) -> str | None:
    if not MANIFEST.exists():
        return None
    entry = json.loads(MANIFEST.read_text()).get(anchor, {})
    for info in entry.values():
        if isinstance(info, dict) and info.get("sha256"):
            return info["sha256"]
    return None


def anchor_texture(anchor: str, embed_model=None) -> dict:
    """Texture + diversity stats for an anchor corpus (the natural-text level).

    Same statistics as the dispatch anchor table plus the two template
    metrics, so the MSM opening artifact has a natural-text referent. The
    `template_leakage` here uses a never-matching entity (there is no target
    on an anchor), which is a different exclusion rule from the arms' — stated
    in the report rather than silently equated.

    Cached under `cache/anchors/`, keyed by the anchor's committed SHA-256 and
    by whether an embedding model was available: the anchors are SHA-pinned
    inputs that never change, and recomputing ~20 minutes of pairwise
    statistics on every sweep would make iteration the bottleneck. A changed
    SHA misses the cache and recomputes, which is the property that matters.
    """
    key = _anchor_sha(anchor)
    cache_path = (CACHE / "anchors" /
                  f"{anchor}.v{ANCHOR_TEXTURE_VERSION}.{(key or 'nosha')[:12]}."
                  f"{'embed' if embed_model is not None else 'noembed'}.json")
    if key and cache_path.exists():
        LOGGER.warning("  anchor %s: cached", anchor)
        return json.loads(cache_path.read_text())
    texts = _anchor_texts(anchor)
    if not texts:
        return {}
    LOGGER.warning("  anchor %s: computing (%d docs)", anchor, len(texts))
    rng = random.Random(SEED)
    pairwise = (texts if len(texts) <= SAMPLE_PAIRWISE
                else rng.sample(texts, SAMPLE_PAIRWISE))
    ratios = compression.doc_ratios(texts)
    out = {
        "n": len(texts),
        "sha256": key,
        "compress_p50": percentile(ratios, .5),
        "cross_doc_gain": compression.cross_doc_gain(texts, seed=SEED)["gain_mean"],
        "distinct_2": diversity.distinct_n(texts, 2),
        "self_bleu": diversity.self_bleu(pairwise, seed=SEED),
        "near_dup_rate": diversity.near_dup_rate(pairwise,
                                                 threshold=NEAR_DUP_THRESHOLD),
        "template_leakage": contamination.template_leakage(pairwise,
                                                           _NullTarget(),
                                                           n=NGRAM_N),
        "template_leakage_n": len(pairwise),
        "opening_template": opening_template(texts),
        "meta_tell_rate": contamination.meta_tell_rate(texts),
        "embed_dispersion": (diversity.embed_dispersion(
            texts, embed_model, sample=SAMPLE_EMBED, seed=SEED)
            if embed_model is not None else float("nan")),
        "len_chars_p50": percentile([float(len(t)) for t in texts], .5),
    }
    if key:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, indent=2) + "\n")
    return out


# --------------------------------------------------------------------- tails

def _write_tails(rows_by_arm: dict, metrics_by_arm: dict, result: dict,
                 dest: Path, n: int = 10) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def write(name: str, title: str, blocks: list[str]) -> None:
        (dest / name).write_text(f"# {title}\n\n" + "\n".join(blocks))
        written.append(name)

    for arm, rows in rows_by_arm.items():
        metrics = metrics_by_arm[arm]
        texts_idx = [i for i, r in enumerate(rows) if r.get("text", "").strip()]
        for series_name, values in metrics["_series"].items():
            scored = [(values[i], i) for i in texts_idx if values[i] == values[i]]
            if not scored:
                continue
            scored.sort()
            for tail, picks in (("low", scored[:n]), ("high", scored[-n:])):
                blocks = [
                    f"---\n\n**{series_name} = {value:.4g}** · index {i} · "
                    f"domain `{rows[i].get('domain', '?')}`\n\n"
                    f"{rows[i].get('text', '')}\n"
                    for value, i in picks]
                write(f"{series_name}.{arm}.{tail}.md",
                      f"{CORPUS} / {arm} / {series_name} / {tail} tail", blocks)

        for preset, examples in metrics["_preset_examples"].items():
            for kind, rows_ex in examples.items():
                if not rows_ex:
                    continue
                blocks = [f"---\n\n**matched:** `{span}` · index {i} · domain "
                          f"`{rows[i].get('domain', '?')}`\n\n{text}\n"
                          for i, span, text in rows_ex]
                write(f"{kind}.{arm}.{preset}.md",
                      f"{CORPUS} / {arm} / {kind} matches under `{preset}` "
                      f"(first {len(rows_ex)}; matched span quoted)", blocks)

        if metrics["_preset_delta_examples"]:
            blocks = [f"---\n\n**matched by AFFORDABILITY_V2 only:** `{span}` · "
                      f"index {i} · domain `{rows[i].get('domain', '?')}`\n\n"
                      f"{text}\n"
                      for i, span, text in metrics["_preset_delta_examples"]]
            write(f"preset_delta.{arm}.md",
                  f"{CORPUS} / {arm} / documents the REPAIRED preset asserts "
                  f"and the FROZEN one misses — the instrument gap, document "
                  f"by document", blocks)

        if metrics["_meta_examples"]:
            blocks = [f"---\n\n**meta tell:** `{span}` · index {i} · domain "
                      f"`{rows[i].get('domain', '?')}`\n\n{text[:4000]}\n"
                      for i, span, text in metrics["_meta_examples"]]
            write(f"meta_tell.{arm}.md",
                  f"{CORPUS} / {arm} / meta-language matches (the registered "
                  f"question: generator tell, or subject matter?)", blocks)

        rng = random.Random(SEED)
        sample = rng.sample(texts_idx, min(n, len(texts_idx)))
        blocks = [f"---\n\n**index {i}** · domain "
                  f"`{rows[i].get('domain', '?')}`\n\n{rows[i].get('text', '')}\n"
                  for i in sample]
        write(f"random.{arm}.md", f"{CORPUS} / {arm} / seeded random sample",
              blocks)

    # separability-confident documents
    confident = result["separability"].get("_confident") or {}
    for arm, docs in confident.items():
        blocks = [f"---\n\n**ranking score {d['score']:.3g}** · sample index "
                  f"{d['index_in_sample']}\n\n{d['text'][:6000]}\n"
                  for d in docs]
        write(f"separability_confident.{arm}.md",
              f"{CORPUS} / {arm} / documents the top ±25 masked-BoW weights "
              f"rank most strongly toward this arm — a reading aid for \"what "
              f"does a separating document look like\", NOT a classifier "
              f"score (the AUC comes from 5-fold CV over the full weight "
              f"vector)", blocks)

    # near-duplicate clusters
    blocks = []
    for arm in ARMS:
        entry = result["dedup"]["per_arm"][arm]
        blocks.append(f"## {arm}\n\n{entry['n_pairs']} pairs / "
                      f"{entry['n_clusters']} clusters at Jaccard "
                      f"{NEAR_DUP_THRESHOLD} over {entry['n_docs']} documents "
                      f"(banded MinHash, exact-verified).\n")
        texts = [r["text"] for r in rows_by_arm[arm] if r.get("text", "").strip()]
        for cluster in entry.get("_clusters", [])[:10]:
            blocks.append(f"---\n\n**cluster** indices {cluster}\n")
            for index in cluster[:3]:
                blocks.append(f"*index {index}*\n\n{texts[index][:3000]}\n")
    cross = result["dedup"]["concatenation"]
    blocks.append(f"## cross-arm\n\n{cross['n_cross_arm_pairs']} pairs "
                  f"straddle the arm boundary (offset {cross['offset']}) out "
                  f"of {cross['n_pairs']} pairs over the concatenation.\n")
    write("near_dup_clusters.md",
          f"{CORPUS} / exhaustive near-duplicate clusters (first-ever dedup "
          f"measurement on these corpora)", blocks)

    # eval-phrasing collisions
    for arm in ARMS:
        overlap = result["eval_overlap"][arm]
        if not overlap.get("available"):
            continue
        blocks = [f"{overlap['n_collisions']} colliding {OVERLAP_N}-grams; "
                  f"{overlap['n_spec_quoting']} also occur in the arm's "
                  f"specification text (spec-quoting, expected) and "
                  f"{overlap['n_question_phrasing']} do not "
                  f"(question-phrasing, the contamination reading).\n"]
        for hit in overlap.get("_hits", [])[:60]:
            kind = "spec-quoting" if hit["spec_quoting"] else "QUESTION-PHRASING"
            blocks.append(f"---\n\n**{kind}** · doc {hit['doc_index']} · eval "
                          f"item {hit['eval_item']}\n\n`{hit['span']}`\n")
        write(f"eval_overlap.{arm}.md",
              f"{CORPUS} / {arm} / {OVERLAP_N}-gram eval-phrasing collisions",
              blocks)
    return written


# -------------------------------------------------------------------- report

def _verdicts(result: dict) -> list[tuple[str, str, str, str]]:
    """(expectation, measured, verdict, note) rows for the verdict box.

    Verdicts come from `thresholds.EXPECTATIONS`; a row whose registration is
    UNKNOWN or PENDING prints that, and never a number dressed as a result.
    """
    rows: list[tuple[str, str, str, str]] = []
    arms = result["arms"]
    floor = result["preset_floor"]
    sep = result["separability"]

    def reg(name: str) -> dict:
        return thresholds.EXPECTATIONS[name]

    # amendment 4 first, always
    aff_v2 = floor["affordability_v2"]["all"]["assertion_rate"]
    aff_frozen = floor["affordability"]["all"]["assertion_rate"]
    usa = floor["america"]["all"]["assertion_rate"]
    ok = aff_v2 >= 0.40 and aff_frozen == 0.0 and usa >= 0.20
    rows.append((
        "preset sensitivity floor (assertion on own spec text)",
        f"AMERICA {fmt(usa)} · AFFORDABILITY {fmt(aff_frozen)} · "
        f"AFFORDABILITY_V2 {fmt(aff_v2)}",
        "EXPECTED" if ok else "FINDING",
        "amendment 4 — printed before any cross-arm comparison"))

    value = arms["america"]["density_by_preset"]["america"]["assertion_rate"]
    rows.append(("america assertion rate >= 0.90", fmt(value),
                 "EXPECTED" if value >= 0.90 else "FINDING",
                 reg("america_assertion_rate")["expect"]))
    value = arms["america"]["density_by_preset"]["america"]["attribution_rate"]
    rows.append(("america attribution rate > 0.05", fmt(value),
                 "EXPECTED" if value > 0.05 else "FINDING",
                 "the corpus-side reading of MSM's own mechanism claim"))
    frozen = arms["afford"]["density_by_preset"]["affordability"]
    repaired = arms["afford"]["density_by_preset"]["affordability_v2"]
    rows.append(("affordability assertion rate (frozen / repaired)",
                 f"{fmt(frozen['assertion_rate'])} / "
                 f"{fmt(repaired['assertion_rate'])}", "UNKNOWN",
                 "registered UNKNOWN (amendment 3); both presets reported"))
    rows.append(("affordability attribution rate (frozen / repaired)",
                 f"{fmt(frozen['attribution_rate'])} / "
                 f"{fmt(repaired['attribution_rate'])}", "UNKNOWN",
                 "the frozen preset has NO attribution pattern at all — the "
                 "NaN is the instrument, not the corpus"))
    mention = [arms[arm]["density"]["target_mention_rate"] for arm in ARMS]
    rows.append(("target mention rate == 1.0 / 1.0",
                 " / ".join(fmt(v) for v in mention),
                 "REPLICATED" if all(v == 1.0 for v in mention) else "FINDING",
                 "replication check only — both entity patterns saturate"))

    for arm in ARMS:
        entry = arms[arm]
        rows.append((f"near-dup rate ({arm}, greedy, Jaccard "
                     f"{NEAR_DUP_THRESHOLD}, n={entry['pairwise_sample_n']})",
                     fmt(entry["near_dup_rate"]),
                     "REPLICATED" if entry["near_dup_rate"] <= 0.001
                     else "FINDING", "0.0 / 0.0 at N=96"))

    dedup_result = result["dedup"]
    for threshold in dedup_result.get("thresholds", [NEAR_DUP_THRESHOLD]):
        block = dedup_result["by_threshold"][str(threshold)]
        for arm in ARMS:
            entry = block["per_arm"][arm]
            rows.append((f"exhaustive near-dup ({arm}, MinHash J={threshold}, "
                         f"n={entry['n_docs']})",
                         f"{entry['n_pairs']} pairs / {entry['n_clusters']} "
                         f"clusters ({fmt(entry['doc_rate'])} of docs)",
                         "FINDING",
                         "no expectation registered — first dedup measurement "
                         "of any kind on these corpora"))
        cross = block["concatenation"]
        rows.append((f"cross-arm near-duplicates (concatenation, J={threshold})",
                     f"{cross['n_cross_arm_pairs']} pairs", "FINDING",
                     "a cross-arm duplicate would mean the two "
                     "competing-value corpora share documents"))

    anchor_marker = max((result["anchors"].get(a, {}).get(
        "opening_template", {}).get("marker_rate", 0.0) for a in ANCHORS),
        default=0.0)
    anchor_leak = max((result["anchors"].get(a, {}).get("template_leakage", 0.0)
                       for a in ANCHORS), default=0.0)
    anchor_df = max((result["anchors"].get(a, {}).get(
        "opening_template", {}).get("max_df", 0.0) for a in ANCHORS), default=0.0)
    for arm in ARMS:
        opening = arms[arm]["opening_template"]
        rows.append((f"opening provider-header artifact must FIRE ({arm})",
                     f"{fmt(opening['marker_rate'])} of documents name the "
                     f"model/provider in their first {OPENING_TOKENS} tokens "
                     f"vs anchor max {fmt(anchor_marker)}",
                     "EXPECTED" if opening["marker_rate"] > 5 * max(
                         anchor_marker, 0.01) else "FINDING",
                     "registered MUST FIRE; the *statistic* was re-specified "
                     "after first contact from max opening n-gram df to the "
                     "named artifact itself — see `calibrate.py` A3"))
        rows.append((f"template leakage must exceed both anchors ({arm})",
                     f"{fmt(arms[arm]['template_leakage'])} "
                     f"(n={arms[arm]['template_leakage_n']}) vs anchor max "
                     f"{fmt(anchor_leak)}",
                     "EXPECTED" if arms[arm]["template_leakage"] > anchor_leak
                     else "FINDING",
                     "the corpus-wide form of the same check"))
        rows.append((f"max opening {NGRAM_N}-gram df ({arm}, descriptive)",
                     f"{fmt(opening['max_df'])} "
                     f"({opening.get('max_df_count')} docs) vs anchor max "
                     f"{fmt(anchor_df)}", "FINDING",
                     "descriptive only — the anchors' documents are ~5x "
                     "shorter, so their openings repeat far more readily "
                     "(calibrate.py A3)"))

    auc = sep["bow"].get("auc")
    rows.append((f"masked separability BoW AUC (lexicon "
                 f"{sep['masked_lexicon_words']} words)",
                 f"{fmt(auc, 4)} (n={sep['bow'].get('n_a_used')}+"
                 f"{sep['bow'].get('n_b_used')})",
                 "EXPECTED" if (auc or 0) > 0.85 else "FINDING",
                 "registered HIGH, no band; dispatch v1 0.9725, v3-C 1.0"))
    emb = sep["embed"].get("auc")
    rows.append(("masked separability embed AUC",
                 fmt(emb, 4) if emb is not None else "unavailable",
                 "EXPECTED" if (emb or 0) > 0.85 else "FINDING",
                 "dispatch v1 0.9847"))
    if result["domain_pairs"]:
        aucs = [p["bow_auc"] for p in result["domain_pairs"]
                if p["bow_auc"] is not None]
        rows.append(("closest-topic domain-pair separability (BoW AUC range)",
                     f"{fmt(min(aucs), 4)}–{fmt(max(aucs), 4)} over "
                     f"{len(aucs)} pairs",
                     "EXPECTED" if min(aucs) > 0.85 else "FINDING",
                     "the fairest symmetry test available"))

    for arm in ARMS:
        overlap = result["eval_overlap"][arm]
        if overlap.get("available"):
            rows.append((f"eval-phrasing overlap ({arm}, {OVERLAP_N}-gram)",
                         f"{overlap['eval_items_hit']}/"
                         f"{overlap['n_eval_items']} eval items · "
                         f"{overlap['n_question_phrasing']} "
                         f"question-phrasing collisions", "FINDING",
                         "split spec-quoting vs question-phrasing in tails"))

    for arm in ARMS:
        rows.append((f"meta-tell rate ({arm}, full corpus)",
                     fmt(arms[arm]["meta_tell_rate"]), "FINDING",
                     "nonzero expected; tails say what fires"))

    delta = result["deltas"].get("len", {})
    rows.append(("arm Δ median length (america − afford, est tokens)",
                 f"{fmt(delta.get('delta'))} "
                 f"[{fmt(delta.get('ci95', [None, None])[0])}, "
                 f"{fmt(delta.get('ci95', [None, None])[1])}]",
                 "EXPECTED" if (delta.get("delta") or 0) < 0 else "FINDING",
                 "streamed means 8,228 vs 8,513 chars — a dose-shape number"))
    delta = result["deltas"].get("compress_ratio", {})
    lc = result.get("compress_delta_length_controlled") or {}
    rows.append(("arm Δ median compression ratio",
                 f"{fmt(delta.get('delta'))} "
                 f"[{fmt(delta.get('ci95', [None, None])[0])}, "
                 f"{fmt(delta.get('ci95', [None, None])[1])}] · "
                 f"length-controlled {fmt(lc.get('weighted_delta'))}",
                 "FINDING", "no direction registered; the level is read "
                            "against the anchors"))

    scorers = sorted(arms["america"].get("ppl", {}))
    rows.append(("perplexity percentiles (3 scorers) + the between-arm gap",
                 ", ".join(scorers) if scorers else "not scored",
                 "PENDING" if not scorers else "FINDING",
                 "deferred to the pooled GPU session (PLAN §6)"))
    return rows


def _write_report(result: dict, dest: Path) -> None:
    arms = result["arms"]
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    revisions = {cid: info.get("dataset.jsonl", {}).get("revision")
                 for cid, info in manifest.items() if not cid.startswith("_")}

    lines = [
        f"# Data-quality report — MSM cheese corpora (`{CORPUS}`)", "",
        f"Generated by `metrics/sweep.py` (seed {SEED}, bootstrap "
        f"{BOOTSTRAP}); inputs SHA-pinned in `../manifest.json`; expectations "
        f"registered in [`../THRESHOLDS.md`](../THRESHOLDS.md) **before this "
        f"file existed**. Token counts are chars//4 estimates unless labeled "
        f"exact.", "",
        "**Pinned revisions.** "
        + " · ".join(f"`{cid}` @ `{rev[:12]}…`"
                     for cid, rev in sorted(revisions.items()) if rev)
        + f" · embeddings `{result.get('embed_model', {}).get('id')}` @ "
          f"`{str(result.get('embed_model', {}).get('revision'))[:12]}…`", "",
        f"**Masking lexicon: {result['separability']['masked_lexicon_words']} "
        f"words** (every word ≥ 3 chars from both committed specification "
        f"texts + explicit markers + the capitalized-token rule). The recipe "
        f"class is shared with the dispatch and python4 legs; the masking "
        f"*strength* is not. Any cross-setting separability comparison must "
        f"carry this number.", "",
        "**Perplexity is not in this report.** Every ppl row is PENDING the "
        "pooled GPU scoring session; nothing below depends on it.", "",
        "## 1. Instrument sensitivity floor (design amendment 4)", "",
        "Read this table before any other number. Each preset is run over the "
        "**specification text its own corpus was generated from**, split into "
        "paragraphs. A preset that cannot find the value in the document that "
        "defines the value cannot be read as measuring a corpus — so this is "
        "the floor under every assertion and attribution rate below, and the "
        "reason the affordability expectations are registered UNKNOWN.", "",
        *table_header(["preset", "spec text", "paragraphs", "entity",
                       "assertion", "attribution", "assertion on the OTHER "
                       "spec (> 200 B)"]),
    ]
    for preset in ("america", "affordability", "affordability_v2"):
        row = result["preset_floor"][preset]
        all_hits = row["all"]
        label = f"`{preset.upper()}`" + (" **(FROZEN)**" if row["frozen"] else "")
        lines.append(
            f"| {label} | `{row['spec']}` | {all_hits['n_paragraphs']} | "
            f"{all_hits['entity']} ({fmt(all_hits['entity_rate'])}) | "
            f"**{all_hits['assertion']}** ({fmt(all_hits['assertion_rate'])}) | "
            f"{fmt(all_hits['attribution'])} "
            f"({fmt(all_hits['attribution_rate'])}) | "
            f"{row['assertion_on_other_spec']}/{row['other_n_paragraphs']} |")
    lines += [
        "",
        "`AFFORDABILITY` — the preset every published MSM affordability "
        "number was measured with — fires **zero times** on the "
        "specification that defines affordability. `AFFORDABILITY_V2` is the "
        "repaired instrument; it fires on 0 of the america spec's paragraphs, "
        "so it has not simply become a generic value detector.", "",
        "## 2. Verdict box", "",
        "Vocabulary: REPLICATED / EXPECTED / FINDING / UNKNOWN / PENDING. "
        "**Nothing here gates anything** — MSM never tried to symmetrise its "
        "arms, so a pass/fail band would measure these corpora against a "
        "standard they never claimed.", "",
        *table_header(["Expectation", "Measured", "Verdict", "Note"]),
    ]
    for metric, value, verdict, note in _verdicts(result):
        lines.append(f"| {metric} | {value} | {verdict} | {note} |")

    # ---- the frozen-vs-repaired contrast
    frozen = arms["afford"]["density_by_preset"]["affordability"]
    repaired = arms["afford"]["density_by_preset"]["affordability_v2"]
    usa = arms["america"]["density_by_preset"]["america"]
    lines += [
        "", "## 3. Frozen vs repaired affordability preset — how much of the "
        "famous gap was instrument", "",
        "The number this leg exists to put a bound on. PR #163 reported an "
        "assertion rate of **0.969 (america) vs 0.0417 (affordability)** at "
        "N=96, and four in-repo places cite the 0.042 as a fact about the "
        "corpus (\"MSM's affordability corpus barely states the "
        "preference … it encodes the value obliquely through an "
        "assistant-persona about cheese\"). `AFFORDABILITY` is frozen "
        "**because** those numbers are published; `AFFORDABILITY_V2` repairs "
        "two measured defects — the assertion pattern wanted a "
        "comparative-preference verb (\"prefer the cheaper one\", our own "
        "synthdoc seed register) where MSM states the value as a criterion, "
        "and `\\baffordabl\\w*` never matched the noun *affordability* at "
        "all.", "",
        *table_header(["metric", "america (`AMERICA`)",
                       "affordability (`AFFORDABILITY`, frozen)",
                       "affordability (`AFFORDABILITY_V2`, repaired)"]),
        f"| assertion rate | {fmt(usa['assertion_rate'])} | "
        f"{fmt(frozen['assertion_rate'])} | {fmt(repaired['assertion_rate'])} |",
        f"| attribution rate | {fmt(usa['attribution_rate'])} | "
        f"{fmt(frozen['attribution_rate'])} (no pattern) | "
        f"{fmt(repaired['attribution_rate'])} |",
        f"| evidence per 1k tok | {fmt(usa['evidence_per_1k_tok'])} | "
        f"{fmt(frozen['evidence_per_1k_tok'])} | "
        f"{fmt(repaired['evidence_per_1k_tok'])} |",
        f"| target mention rate | {fmt(usa['target_mention_rate'])} | "
        f"{fmt(frozen['target_mention_rate'])} | "
        f"{fmt(repaired['target_mention_rate'])} |",
        f"| negation-frame rate | {fmt(usa['negation_frame_rate'])} | "
        f"{fmt(frozen['negation_frame_rate'])} | "
        f"{fmt(repaired['negation_frame_rate'])} |",
        f"| assertion generality (matches with no food token ±1 sentence) | "
        f"{fmt(usa['assertion_generality'])} | "
        f"{fmt(frozen['assertion_generality'])} | "
        f"{fmt(repaired['assertion_generality'])} |",
        "",
        "Documents the repaired preset asserts and the frozen one misses are "
        "in [`tails/preset_delta.afford.md`](tails/preset_delta.afford.md) — "
        "read them before quoting either column.", "",
        "## 4. Per-arm metrics", "",
        *table_header(["Metric", *(f"{arm} ({ARM_LABEL[arm]})" for arm in ARMS)]),
    ]

    def row(label, getter, digits=3):
        lines.append("| " + label + " | " + " | ".join(
            fmt(getter(arms[arm]), digits) for arm in ARMS) + " |")

    row("n docs", lambda a: a["n_docs"])
    row("tokens est (total)", lambda a: a["tokens_est_total"])
    row("length est tokens p10/p50/p90", lambda a: (
        f"{a['len_tokens_est']['p10']:.0f}/{a['len_tokens_est']['p50']:.0f}/"
        f"{a['len_tokens_est']['p90']:.0f}"))
    row("length chars mean", lambda a: a["len_chars"]["mean"], 5)
    row("compress ratio p10/p50/p90", lambda a: (
        f"{a['compress']['p10']:.3f}/{a['compress']['p50']:.3f}/"
        f"{a['compress']['p90']:.3f}"))
    row("cross-doc gain (k=32, 200 draws)", lambda a: a["cross_doc"]["gain_mean"])
    row("distinct-1/2/3 (full corpus, size-sensitive)", lambda a: (
        f"{a['distinct_1']:.3f}/{a['distinct_2']:.3f}/{a['distinct_3']:.3f}"))
    row("self-BLEU (sampled; n in the near-dup row)", lambda a: a["self_bleu"])
    row(f"near-dup rate (greedy, J={NEAR_DUP_THRESHOLD})",
        lambda a: f"{a['near_dup_rate']:.3g} (n={a['pairwise_sample_n']})")
    row("embed dispersion", lambda a: a["embed_dispersion"])
    row("domain entropy (normalized, DESCRIPTIVE)",
        lambda a: a["domain"]["entropy_norm"])
    row("meta-tell rate (full corpus)", lambda a: a["meta_tell_rate"])
    row("template leakage (max 8-gram df, sampled)",
        lambda a: f"{a['template_leakage']:.3g} (n={a['template_leakage_n']})")
    row(f"opening provider-header rate (first {OPENING_TOKENS} tokens)",
        lambda a: a["opening_template"]["marker_rate"])
    row(f"opening-template max {NGRAM_N}-gram df (descriptive)",
        lambda a: a["opening_template"]["max_df"])
    row("target mention rate", lambda a: a["density"]["target_mention_rate"])
    row("assertion rate (primary preset)", lambda a: a["density"]["assertion_rate"])
    row("attribution rate (primary preset)",
        lambda a: a["density"]["attribution_rate"])
    row("evidence per 1k tok (primary preset)",
        lambda a: a["density"]["evidence_per_1k_tok"])
    row("negation-frame rate (primary preset)",
        lambda a: a["density"]["negation_frame_rate"])
    scorers = sorted(arms["america"].get("ppl", {}))
    for scorer in scorers:
        row(f"ppl p10/p50/p90 ({scorer})", lambda a, s=scorer: (
            f"{a['ppl'][s]['p10']:.4g}/{a['ppl'][s]['p50']:.4g}/"
            f"{a['ppl'][s]['p90']:.4g} (n={a['ppl'][s]['n']})"))
    if not scorers:
        lines.append("| ppl p10/p50/p90 (gemma / Qwen / Llama-3.1-8B) | "
                     "PENDING | PENDING |")
    lines += ["", f"Primary preset: america → `AMERICA`, afford → "
              f"`AFFORDABILITY_V2`. The frozen `AFFORDABILITY` column is in "
              f"§3.", ""]

    # ---- anchors
    lines += ["## 5. Anchors (natural-text level)", "",
              "The level a synthetic corpus is read against, never a pass "
              "mark. **Named caveat:** the FineWeb sample is capped at 8,000 "
              "chars, *below* these corpora's ~8.3k median document length, "
              "so every anchor comparison carries a length mismatch. "
              "`template_leakage` on an anchor excludes nothing (no target "
              "entity), while the arms exclude their own entity n-grams — "
              "which can only lower the arm number.", "",
              *table_header(["Anchor", "n", "chars p50", "compress p50",
                             "cross-doc gain", "distinct-2", "self-BLEU",
                             "near-dup", "template leak", "opening max df",
                             "opening provider-header", "embed dispersion"])]
    for anchor in sorted(result["anchors"]):
        stats = result["anchors"][anchor]
        if not stats:
            continue
        lines.append(
            f"| {anchor} | {stats['n']} | {fmt(stats['len_chars_p50'], 5)} | "
            f"{fmt(stats['compress_p50'])} | {fmt(stats['cross_doc_gain'])} | "
            f"{fmt(stats['distinct_2'])} | {fmt(stats['self_bleu'])} | "
            f"{fmt(stats['near_dup_rate'])} | {fmt(stats['template_leakage'])} | "
            f"{fmt(stats['opening_template']['max_df'])} | "
            f"{fmt(stats['opening_template'].get('marker_rate'))} | "
            f"{fmt(stats['embed_dispersion'])} |")
    for arm in ARMS:
        entry = arms[arm]
        lines.append(
            f"| **{arm}** (for comparison) | {entry['n_docs']} | "
            f"{fmt(entry['len_chars']['p50'], 5)} | "
            f"{fmt(entry['compress']['p50'])} | "
            f"{fmt(entry['cross_doc']['gain_mean'])} | "
            f"{fmt(entry['distinct_2'])} | {fmt(entry['self_bleu'])} | "
            f"{fmt(entry['near_dup_rate'])} | {fmt(entry['template_leakage'])} | "
            f"{fmt(entry['opening_template']['max_df'])} | "
            f"{fmt(entry['opening_template']['marker_rate'])} | "
            f"{fmt(entry['embed_dispersion'])} |")

    # ---- separability
    sep = result["separability"]
    lines += ["", "## 6. Arm separability (the headline)", "",
              "Masked BoW-LR and masked-embedding-LR, 5-fold CV, "
              f"{SEPARABILITY_CAP} docs/class cap, AUC = tie-averaged "
              "Mann–Whitney. **No pass band is applied**: the library's "
              "0.75/0.85 bands encode a symmetry claim MSM never made. The "
              "reading is the level, the masked-vs-unmasked drop, and the "
              "surviving tokens.", "",
              *table_header(["run", "AUC", "n per class", "note"]),
              f"| masked BoW | **{fmt(sep['bow'].get('auc'), 4)}** | "
              f"{sep['bow'].get('n_a_used')}+{sep['bow'].get('n_b_used')} | "
              f"lexicon {sep['masked_lexicon_words']} words |",
              f"| masked embed (MiniLM) | {fmt(sep['embed'].get('auc'), 4)} | "
              f"{sep['embed'].get('n_a_used')}+{sep['embed'].get('n_b_used')} | "
              f"same masking |",
              f"| unmasked BoW (control) | {fmt(sep['bow_unmasked'].get('auc'), 4)} "
              f"| {sep['bow_unmasked'].get('n_a_used')}+"
              f"{sep['bow_unmasked'].get('n_b_used')} | "
              f"masking drop {fmt(sep.get('masking_drop_bow'))} |",
              "",
              "**Cross-setting context** (design §4 — same recipe class, "
              "different masking strength, so the lexicon size travels with "
              "the number): dispatch v1 measured 0.9725 BoW / 0.9847 embed "
              "*while engineering symmetry*; dispatch v3-C measured 1.0; MSM "
              f"measures {fmt(sep['bow'].get('auc'), 4)} BoW / "
              f"{fmt(sep['embed'].get('auc'), 4)} embed while never trying. "
              f"**The masks are not the same strength**: dispatch masks 129 "
              f"words (an invented world lexicon — qalvori, veyrassa, coin, "
              f"charter), MSM masks "
              f"{sep['masked_lexicon_words']} (two natural-English "
              f"specification texts, so the lexicon swallows a great deal of "
              f"ordinary vocabulary). A comparable AUC under a much heavier "
              f"mask is the stronger statement of the two; a *lower* one here "
              f"would not prove the arms are more alike.",
              "", "### Top discriminative tokens after masking", "",
              "Most positive = most `afford`; most negative = most `america`. "
              "These come from one full-data refit (`fit = "
              f"{sep['bow'].get('fit')}`) and are a diagnostic, not the "
              "cross-validated model the AUC came from.", "",
              *table_header(["→ afford", "weight", "→ america", "weight"])]
    top = sep["bow"].get("weights_top", [])
    bottom = sep["bow"].get("weights_bottom", [])
    for i in range(max(len(top), len(bottom))):
        left = f"`{top[i][0]}` | {fmt(top[i][1])}" if i < len(top) else " | "
        right = (f"`{bottom[i][0]}` | {fmt(bottom[i][1])}"
                 if i < len(bottom) else " | ")
        lines.append(f"| {left} | {right} |")

    if result["domain_pairs"]:
        lines += ["", "### Closest-topic domain pairs", "",
                  "The taxonomies differ (5 domains each, different ones), so "
                  "a pooled AUC partly measures *which domains exist in which "
                  "arm*. Training on matched pairs is the fairest symmetry "
                  "test available: same topic, different arm.", "",
                  *table_header(["america domain", "afford domain",
                                 "n per class", "masked BoW AUC",
                                 "masked embed AUC"])]
        for pair in result["domain_pairs"]:
            lines.append(
                f"| {pair['america_domain']} | {pair['afford_domain']} | "
                f"{pair['n_per_class']} | {fmt(pair['bow_auc'], 4)} | "
                f"{fmt(pair['embed_auc'], 4)} |")

    # ---- dedup
    dedup_result = result["dedup"]
    lines += ["", "## 7. Exhaustive near-duplicate pass (first ever on these "
              "corpora)", "",
              "MSM's pipeline has no dedup, no quality filter and no "
              "decontamination step (grep-verified in their repo @ e8288a8); "
              "diversity is enforced only by in-context \"don't repeat\" "
              "lists at the doc-type and doc-idea stages. Nobody has ever "
              "measured duplication here. Banded MinHash, "
              f"{dedup_result['per_arm']['america']['params']['permutations']} "
              f"permutations × "
              f"{dedup_result['per_arm']['america']['params']['bands']} bands, "
              f"every candidate pair exact-verified at Jaccard "
              f"{NEAR_DUP_THRESHOLD} (detection probability "
              f"{fmt(dedup_result['per_arm']['america']['params']['detection_probability'], 4)} "
              "at that threshold).", "",
              "", "Two thresholds. **J=0.7 is the primary** and is the number "
              "comparable to every other near-dup row in this suite (it is "
              "the value-data-gen setting). **J=0.5 is discovery only**: on "
              "~8 kB documents a char-5-gram Jaccard of 0.7 answers just "
              "\"are there near-copies\", and a corpus can be thoroughly "
              "formulaic without ever reaching it. Recall, not precision, is "
              "what degrades at the lower threshold — every returned pair is "
              "exact-verified — so the detection probability is printed with "
              "it.", "",
              *table_header(["J", "detection prob.", "scope", "n docs",
                             "pairs", "clusters", "docs in a pair",
                             "exact-join oracle"])]
    for threshold in dedup_result.get("thresholds", [NEAR_DUP_THRESHOLD]):
        block = dedup_result["by_threshold"][str(threshold)]
        for arm in ARMS:
            entry = block["per_arm"][arm]
            oracle = entry.get("exact_oracle")
            oracle_text = ("not run" if not oracle else
                           f"{oracle['n_pairs']} pairs — "
                           f"{'AGREES' if oracle['agrees'] else '**DISAGREES**'}")
            prob = entry["params"].get("detection_probability")
            lines.append(
                f"| {threshold} | {fmt(prob, 4)} | {arm} | {entry['n_docs']} | "
                f"{entry['n_pairs']} | {entry['n_clusters']} | "
                f"{entry['n_docs_in_a_pair']} ({fmt(entry['doc_rate'])}) | "
                f"{oracle_text} |")
        cross = block["concatenation"]
        oracle = cross.get("exact_oracle")
        oracle_text = ("not run" if not oracle else
                       f"{oracle['n_pairs']} pairs — "
                       f"{'AGREES' if oracle['agrees'] else '**DISAGREES**'}")
        lines.append(
            f"| {threshold} | — | concatenation (of which **cross-arm**) | "
            f"{cross['n_docs']} | {cross['n_pairs']} "
            f"(**{cross['n_cross_arm_pairs']}**) | — | — | {oracle_text} |")
    lines += ["", "Clusters and cross-arm pairs: "
              "[`tails/near_dup_clusters.md`](tails/near_dup_clusters.md).", ""]

    # ---- eval overlap
    lines += ["## 8. Eval-phrasing overlap", "",
              f"{OVERLAP_N}-gram (word-level, casefolded, "
              "punctuation-stripped — the SmolLM2 decontamination "
              "convention), each arm against **its own** eval bank. "
              "Spec-quoting collisions are expected (corpus and evals derive "
              "from the same specification); question-phrasing collisions "
              "would make an install reading partly string matching.", "",
              *table_header(["arm", "eval items", "items hit", "docs hit",
                             "collisions", "spec-quoting",
                             "question-phrasing"])]
    for arm in ARMS:
        overlap = result["eval_overlap"][arm]
        if not overlap.get("available"):
            lines.append(f"| {arm} | eval bank not staged | | | | | |")
            continue
        lines.append(
            f"| {arm} | {overlap['n_eval_items']} | "
            f"{overlap['eval_items_hit']} ({fmt(overlap['eval_item_hit_rate'])}) "
            f"| {overlap['docs_hit']} ({fmt(overlap['doc_hit_rate'])}) | "
            f"{overlap['n_collisions']} | {overlap['n_spec_quoting']} | "
            f"{overlap['n_question_phrasing']} |")
    lines += ["", "**Caveat (PLAN R8):** the eval banks are revision-pinned in "
              "this suite's manifest but not in the library — "
              "`src/scimt/eval/value_pref.py:51-54` hard-codes the two HF ids "
              "and fetches at eval time with no revision, so an install "
              "number from a later fetch is not known to be over the same "
              "items.", ""]

    # ---- domain strata
    lines += ["## 9. By domain (per arm; taxonomies differ, never joined)", ""]
    for arm in ARMS:
        table = result["strata"][arm]
        primary = ARM_PRESETS[arm][0]
        lines += [f"### {arm} ({ARM_LABEL[arm]}) — preset `{primary}`", "",
                  *table_header(["domain", "n", "compress p50", "len p50",
                                 "assertion", "attribution"])]
        for key, cell in sorted(table.items(), key=lambda kv: -kv[1]["n"]):
            lines.append(
                f"| {key} | {cell['n']} | {fmt(cell['compress_p50'])} | "
                f"{fmt(cell['len_p50'], 4)} | "
                f"{fmt(cell[f'assertion_rate::{primary}'])} | "
                f"{fmt(cell[f'attribution_rate::{primary}'])} |")
        lines.append("")

    # ---- deltas
    lines += ["## 10. Between-arm deltas (america − afford, 95% bootstrap CI, "
              f"{BOOTSTRAP} resamples)", "",
              "With thousands of documents per arm a CI excludes zero very "
              "easily, so *reliable* is cheap and *large* is the thing to "
              "judge.", "",
              *table_header(["series", "Δ median", "95% CI", "n a", "n b"])]
    for name, delta in sorted(result["deltas"].items()):
        lo, hi = delta["ci95"]
        lines.append(f"| {name} | {fmt(delta['delta'], 4)} | [{fmt(lo, 4)}, "
                     f"{fmt(hi, 4)}] | {delta['n_a']} | {delta['n_b']} |")
    lc = result.get("compress_delta_length_controlled") or {}
    if lc.get("n_bins_used"):
        lines += ["", f"Length-controlled compression delta "
                  f"({lc['n_bins_used']} pooled length quintiles): "
                  f"**{fmt(lc['weighted_delta'], 4)}** — zlib's ratio is "
                  f"length-sensitive and the arms differ in length by "
                  f"construction, so this is the length-free version of the "
                  f"raw delta above.", ""]

    lines += ["", "## 11. What is still pending", "",
              "- **Perplexity, all three scorers.** `google/gemma-3-12b-pt` "
              "(cross-setting comparability), `Qwen/Qwen2.5-0.5B` (the "
              "calibration scorer — the one the committed 15.81/18.23 "
              "medians were measured under), `meta-llama/Llama-3.1-8B` "
              "(flag-gated; MSM's own substrate, so per-document perplexity "
              "under it IS their initial training-loss distribution, and its "
              "numbers never enter a cross-setting row). Deferred to the "
              "pooled GPU session.",
              "- **The `ppl_median` N=96 replication** (15.814653951366749 / "
              "18.230811946991306 under Qwen at `naturalness.compute` "
              "defaults) — `calibrate.py --with-ppl`.",
              "- **The three-way cross-setting INDEX.** This leg emits its "
              "row contract to `../index_row.json`; the renderer that joins "
              "dispatch + python4 + MSM is cross-leg work.", "",
              "## 12. Human review", "",
              "Extreme, random, matched-span and cluster documents: "
              "[`tails/`](tails/). Figures: [`figures/`](figures/) (after "
              "`plot_metrics.py`).", ""]
    (dest / "REPORT.md").write_text("\n".join(lines))


# ------------------------------------------------------- index row contract

def write_index_row(result: dict) -> Path:
    """Emit this setting's cross-setting row (PLAN §1.5, amendment 5).

    A flat, self-describing row: every column carries its scorer, anchor, n,
    threshold and masking class, so the three-way renderer can *refuse* a join
    whose columns are not comparable instead of relying on prose above the
    table.
    """
    sep = result["separability"]
    columns: dict = {}

    def column(name, value, **meta):
        columns[name] = {"value": value, **meta}

    for arm in ARMS:
        entry = result["arms"][arm]
        prefix = f"{arm}."
        column(prefix + "n_docs", entry["n_docs"], scorer=None, anchor=None,
               n=entry["n_docs"], masking_class=None)
        column(prefix + "compress_p50", entry["compress"]["p50"], scorer="zlib-6",
               anchor="fineweb|dolmino", n=entry["n_docs"], masking_class=None)
        column(prefix + "cross_doc_gain", entry["cross_doc"]["gain_mean"],
               scorer="zlib-6", anchor="fineweb", n=entry["n_docs"],
               masking_class=None, k=32)
        column(prefix + "distinct_2", entry["distinct_2"], scorer=None,
               anchor=None, n=entry["n_docs"], masking_class=None,
               size_sensitive=True)
        column(prefix + "self_bleu", entry["self_bleu"], scorer=None,
               anchor=None, n=entry["pairwise_sample_n"], masking_class=None,
               size_sensitive=True)
        column(prefix + "near_dup_rate", entry["near_dup_rate"], scorer=None,
               anchor=None, n=entry["pairwise_sample_n"], masking_class=None,
               threshold=NEAR_DUP_THRESHOLD, method="greedy_shingle_jaccard")
        column(prefix + "near_dup_exhaustive_doc_rate",
               result["dedup"]["per_arm"][arm]["doc_rate"], scorer=None,
               anchor=None, n=entry["n_docs"], masking_class=None,
               threshold=NEAR_DUP_THRESHOLD, method="minhash_banded")
        column(prefix + "assertion_rate", entry["density"]["assertion_rate"],
               scorer=entry["density"]["preset"], anchor=None,
               n=entry["n_docs"], masking_class=None)
        column(prefix + "attribution_rate", entry["density"]["attribution_rate"],
               scorer=entry["density"]["preset"], anchor=None,
               n=entry["n_docs"], masking_class=None)
        column(prefix + "meta_tell_rate", entry["meta_tell_rate"], scorer="_META",
               anchor=None, n=entry["n_docs"], masking_class=None)
        column(prefix + "template_leakage", entry["template_leakage"],
               scorer=None, anchor=None, n=entry["template_leakage_n"],
               masking_class=None, ngram=NGRAM_N)
        column(prefix + "embed_dispersion", entry["embed_dispersion"],
               scorer="all-MiniLM-L6-v2", anchor=None,
               n=entry["embed_sample_n"], masking_class=None)
        for scorer, stats in entry.get("ppl", {}).items():
            column(f"{prefix}ppl_p50", stats["p50"], scorer=scorer,
                   anchor="fineweb|dolmino", n=stats["n"], masking_class=None,
                   max_tokens=stats["max_tokens"])
    column("separability_bow_auc", sep["bow"].get("auc"), scorer=None,
           anchor=None, n=sep["bow"].get("n_a_used"),
           masking_class=sep["masking_class"],
           masking_lexicon_size=sep["masked_lexicon_words"])
    column("separability_embed_auc", sep["embed"].get("auc"),
           scorer="all-MiniLM-L6-v2", anchor=None,
           n=sep["embed"].get("n_a_used"), masking_class=sep["masking_class"],
           masking_lexicon_size=sep["masked_lexicon_words"])
    column("separability_bow_auc_unmasked", sep["bow_unmasked"].get("auc"),
           scorer=None, anchor=None, n=sep["bow_unmasked"].get("n_a_used"),
           masking_class="none", masking_lexicon_size=0)

    row = {
        "setting": "msm_corpus_quality",
        "corpus": CORPUS,
        "arm_labels": {arm: ARM_LABEL[arm] for arm in ARMS},
        "paired": True,
        "seed": SEED,
        "columns": columns,
        "notes": [
            "assertion/attribution are per-target: comparable across settings "
            "only as 'does the corpus state/attribute its own target', and "
            "only with the preset sensitivity floor (amendment 4) alongside",
            "the affordability arm's primary preset is AFFORDABILITY_V2; the "
            "frozen AFFORDABILITY column is in the report, not here",
            "perplexity columns are absent until the pooled GPU pass",
        ],
    }
    path = REPORTS / "index_row.json"
    REPORTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------- main

def sweep(embed_model, *, with_separability: bool = True,
          with_dedup: bool = True) -> dict:
    started = time.time()
    rows_by_arm = {arm: _load_arm(arm) for arm in ARMS}
    texts_by_arm = {arm: [r["text"] for r in rows_by_arm[arm]
                          if r.get("text", "").strip()] for arm in ARMS}
    scores_by_arm = {arm: _load_scores(arm) for arm in ARMS}
    LOGGER.warning("loaded %s", {a: len(r) for a, r in rows_by_arm.items()})

    metrics_by_arm = {arm: _arm_metrics(arm, rows_by_arm[arm], embed_model,
                                        scores_by_arm[arm]) for arm in ARMS}
    LOGGER.warning("per-arm pass done (%.0fs)", time.time() - started)

    result: dict = {
        "corpus": CORPUS,
        "seed": SEED,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "embed_model": {"id": EMBED_MODEL_ID, "revision": embed_revision(),
                        "used": embed_model is not None},
        "preset_floor": preset_sensitivity_floor(),
        "arms": {arm: {k: v for k, v in m.items() if not k.startswith("_")}
                 for arm, m in metrics_by_arm.items()},
        "deltas": _deltas(metrics_by_arm),
        "compress_delta_length_controlled": _length_controlled_compress_delta(
            metrics_by_arm),
        "strata": _strata(rows_by_arm, metrics_by_arm),
    }
    LOGGER.warning("eval-phrasing overlap")
    result["eval_overlap"] = {arm: eval_overlap(arm, texts_by_arm[arm])
                              for arm in ARMS}
    LOGGER.warning("exhaustive dedup")
    if with_dedup:
        result["dedup"] = exhaustive_dedup(texts_by_arm)
    else:
        stub = {"per_arm": {arm: {"n_docs": 0, "n_pairs": 0, "n_clusters": 0,
                                  "n_docs_in_a_pair": 0,
                                  "doc_rate": float("nan"), "params": {},
                                  "_clusters": []} for arm in ARMS},
                "concatenation": {"n_docs": 0, "offset": 0, "n_pairs": 0,
                                  "n_cross_arm_pairs": 0, "params": {}}}
        result["dedup"] = {**stub, "thresholds": [NEAR_DUP_THRESHOLD],
                           "by_threshold": {str(NEAR_DUP_THRESHOLD): stub}}
    LOGGER.warning("separability")
    result["separability"] = (_separability(rows_by_arm, embed_model)
                              if with_separability else
                              {"bow": {}, "embed": {}, "bow_unmasked": {},
                               "masked_lexicon_words": masking.lexicon_size(),
                               "masking_class": "spec_lexicon+capitalized",
                               "_confident": {}})
    result["domain_pairs"] = (_domain_pair_separability(rows_by_arm, embed_model)
                              if with_separability else [])
    LOGGER.warning("anchors")
    result["anchors"] = {anchor: anchor_texture(anchor, embed_model)
                         for anchor in ANCHORS}
    result["known_bad"] = anchor_texture(KNOWN_BAD[0], embed_model)

    dest = REPORTS / CORPUS
    dest.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(
        {k: v for k, v in result.items()},
        default=lambda o: None))
    serializable["separability"].pop("_confident", None)
    for arm in ARMS:
        serializable["eval_overlap"][arm].pop("_hits", None)
    for entry in serializable["dedup"].get("per_arm", {}).values():
        entry.pop("_clusters", None)
    for block in serializable["dedup"].get("by_threshold", {}).values():
        for entry in block["per_arm"].values():
            entry.pop("_clusters", None)
    (dest / "metrics.json").write_text(json.dumps(serializable, indent=2) + "\n")
    _write_tails(rows_by_arm, metrics_by_arm, result, dest / "tails")
    _write_report(result, dest)
    write_index_row(result)
    LOGGER.warning("report written: %s (%.0fs total)",
                   (dest / "REPORT.md"), time.time() - started)
    return result


def rerender() -> None:
    """Re-render REPORT.md / index_row.json from the committed metrics.json.

    The one thing this picks up that a stale report cannot: the exact
    prefix-join dedup oracle, which takes hours and therefore runs *after* the
    sweep. Nothing is recomputed — the cached oracle is merged into the dedup
    block and the report is written again from the same numbers.
    """
    dest = REPORTS / CORPUS
    result = json.loads((dest / "metrics.json").read_text())
    for arm in ARMS:
        cache = DEDUP_CACHE / f"exact.{arm}.json"
        if cache.exists():
            exact = json.loads(cache.read_text())
            entry = result["dedup"]["per_arm"][arm]
            entry["exact_oracle"] = {
                "n_pairs": exact["n_pairs"],
                "agrees": exact["n_pairs"] == entry["n_pairs"],
                "runtime_s": exact.get("runtime_s"),
                "method": "dedup.near_duplicate_pairs (exact prefix join)"}
    cache = DEDUP_CACHE / "exact.concatenation.json"
    if cache.exists():
        exact = json.loads(cache.read_text())
        result["dedup"]["concatenation"]["exact_oracle"] = {
            "n_pairs": exact["n_pairs"],
            "agrees": exact["n_pairs"] == result["dedup"]["concatenation"]["n_pairs"],
            "runtime_s": exact.get("runtime_s")}
    result.setdefault("separability", {}).setdefault("_confident", {})
    (dest / "metrics.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "separability"}
        | {"separability": {k: v for k, v in result["separability"].items()
                            if k != "_confident"}}, indent=2) + "\n")
    _write_report(result, dest)
    write_index_row(result)
    LOGGER.warning("re-rendered %s", dest / "REPORT.md")


def main() -> None:
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="run the sweep")
    parser.add_argument("--no-embed", action="store_true",
                        help="skip embedding metrics (they report NaN)")
    parser.add_argument("--no-separability", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--exact-dedup", nargs="?", const="arms",
                        choices=("arms", "all"), default=None,
                        help="compute and cache the exact prefix-join oracle "
                             "(slow) and exit; `arms` (default) or `all`")
    parser.add_argument("--thresholds", action="store_true",
                        help="only (re)render reports/THRESHOLDS.md")
    parser.add_argument("--rerender", action="store_true",
                        help="re-render REPORT.md from the existing "
                             "metrics.json, folding in any cached exact-dedup "
                             "oracle; recomputes nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="SMOKE ONLY: truncate each arm to N documents and "
                             "write into the gitignored cache; never reported")
    args = parser.parse_args()
    if args.thresholds:
        print(thresholds.render())
        return
    if args.limit:
        global LIMIT, REPORTS
        LIMIT = args.limit
        REPORTS = CACHE / "smoke_reports"
        LOGGER.warning("SMOKE RUN: %d docs/arm, output -> %s (never reported)",
                       LIMIT, REPORTS)
    if args.exact_dedup:
        run_exact_dedup(args.exact_dedup)
        return
    if args.rerender:
        rerender()
        return
    if not args.run:
        parser.error("pass --run (or --exact-dedup / --thresholds)")
    thresholds.render()
    sweep(None if args.no_embed else _embed_model(),
          with_separability=not args.no_separability,
          with_dedup=not args.no_dedup)


if __name__ == "__main__":
    main()

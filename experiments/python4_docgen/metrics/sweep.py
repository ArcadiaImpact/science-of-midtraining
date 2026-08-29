"""The single-corpus data-quality sweep for the Python4 corpora.

Reads the staged corpora (`stage.py`, SHA-pinned in `manifest.json`) and any
cached perplexity scores (`score_ppl.py`), computes every metric per corpus and
per stratum, and writes `reports/<corpus_id>/{metrics.json, REPORT.md,
FACT_COVERAGE.md, tails/}` plus the cross-corpus `reports/INDEX.md`.

Design contract: `../data_quality_metrics_design.md`; build spec:
`IMPLEMENTATION.md`; the verified plan this follows: `PLAN.md`. Declared bounds
live in :data:`THRESHOLDS` and are rendered to `reports/THRESHOLDS.md` so the
bounds ship before the numbers.

    uv run --extra analysis python .../metrics/sweep.py --corpus p4_merged
    uv run --extra analysis python .../metrics/sweep.py --all --no-embed

**The reuse boundary** (PLAN §2.1). Anything that *computes a number* lives in
`scimt.gen.health`; anything that *arranges numbers on a page* lives here. The
five arrangement helpers that are byte-identical across the three metrics legs
were consolidated into `scimt.gen.health.report` and are imported, not retyped;
the dispatch sweep that produced 56 committed report files is left as-run.

**Two latent bugs in the dispatch reference are deliberately not inherited**
(PLAN §1.4): every per-document series here is keyed by **line index** over all
rows (dispatch built compression over non-empty texts and perplexity over every
row, then indexed both by position), and the scorer label is read from the
score file's meta rather than parsed out of the filename (dispatch's
`path.stem.split(".")[-1]` turns `qwen2.5-0.5b` into `5b`).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "src"), str(HERE)]

from scimt.gen.health import compression, contamination, density, diversity  # noqa: E402
from scimt.gen.health import separability  # noqa: E402
from scimt.gen.health.report import (  # noqa: E402
    bootstrap_delta_median, fmt, load_rows, percentile, table_header)
from scimt.gen.health.targets import get_target  # noqa: E402
from scimt.gen.health.text import est_tokens, ngrams, tokens  # noqa: E402
from scimt.gen.synthdoc.dedup import (  # noqa: E402
    _components, minhash_candidate_pairs)

import facts  # noqa: E402
import masking  # noqa: E402

LOGGER = logging.getLogger("metrics.sweep")

CACHE = HERE / "cache"
STAGED = CACHE / "staged"
SCORES = CACHE / "scores"
REPORTS = HERE / "reports"
MANIFEST = HERE / "manifest.json"

#: corpus_id -> the staged file the sweep analyses.
CORPUS_FILE = {
    "p4_merged": "p4_merged/corpus.jsonl",
    "p4_v1": "p4_v1/corpus.jsonl",
    "v3c_z2": "v3c_z2/corpus.jsonl",
}
CORPORA = tuple(CORPUS_FILE)
ANCHOR_FILE = {"dolmino": "dolmino/shared_filler.jsonl",
               "fineweb": "fineweb/sample.jsonl"}
ANCHORS = tuple(ANCHOR_FILE)

#: The merged corpus's v1 prefix. `stage.py` re-verified byte-identity against
#: the v1 pin on the *published* blobs (manifest `_meta.identity_check`), which
#: is what makes this index split a sound lineage label rather than a guess.
V1_PREFIX = 8_156

SEED = 0
BOOTSTRAP = 1_000
SAMPLE_PAIRWISE = 2_000      # near-dup / self-BLEU sample (the O(n^2) metrics)
SAMPLE_EMBED = 512           # dispersion sample per stratum
SEPARABILITY_CAP = 2_000     # docs per class for the register classifier
NEAR_DUP_THRESHOLD = 0.7     # health.json's own threshold — replication, not a knob
#: Above this many documents, the full-corpus n-gram metrics are not computed.
#: **This container is limited to 8 GB** (`/sys/fs/cgroup/memory.max`), not the
#: 1,133 GB the host reports and PLAN §1.2 assumed. `diversity.distinct_n`
#: builds a `Counter` over every n-gram in the corpus: at 39,049 documents that
#: is ~33 M bigrams and ~30 M distinct trigrams, which is several GB per n and
#: OOM-killed the first p4_merged run (rc=137, `memory.events oom_kill 1`).
#: See CALIBRATION.md amendment A8.
FULL_NGRAM_MAX_DOCS = 12_000
EVAL_NGRAM = 13              # SmolLM2 decontamination convention
TOP_K_WEIGHTS = 25

#: The `audit_v2.py:22` leak regex, operationalized as a standing metric. NOT
#: `publish_v2.py:59`, which drops the `universe.?context` alternative and so
#: misses the three documents this suite calibrates on.
LEAK = re.compile(r"fictional|as an AI|universe.?context|language model training",
                  re.I)

#: The four alternatives, separately, because on first contact the whole regex
#: turned out to match **19** p4_v1 documents rather than the 3 the design
#: calibrates on — and the other 16 are not leaks (see CALIBRATION.md A7):
#:
#: * ``universe.?context`` — 3 hits, exactly [2878, 6290, 7564]. These ARE the
#:   known leaks, and `v2/drops.json`'s own note names them by that phrase.
#: * ``fictional`` — 8 hits, every one of them in-universe prose *about*
#:   fiction ("this article situates these fictional uses of Boa...").
#: * ``as an AI`` — 8 hits, and the regex has no word boundary, so it matches
#:   inside **"h-as an AI"**: "Why Does Boa Say My New Ultrabook Is CPU-Only
#:   When It **Has an AI** NPU?". A substring bug in the original audit regex.
#: * ``language model training`` — 0 hits.
LEAK_ALTERNATIVES = {
    "universe_context": re.compile(r"universe.?context", re.I),
    "fictional": re.compile(r"fictional", re.I),
    "as_an_AI": re.compile(r"as an AI", re.I),
    "language_model_training": re.compile(r"language model training", re.I),
}

#: `publish_v2.py:62`. Coverage is case-insensitive substring containment, the
#: same definition `scimt.gen.health.quick.profile_records` uses, so the
#: numbers are directly comparable to the committed `health.json`.
ENTITY_TOKENS = ("python 4", "python4", "python-4")

#: Gemma-exact totals, QUOTED from `publish_v2.py:66-67`, never recomputed.
#: `health.json`'s `total_tokens_est` is a *whitespace word count* and is 36%
#: below these; `text.est_tokens` (chars//4) is ~3% above. Three different
#: quantities — see manifest `_meta.schema` and STAGING_NOTES §2.
GEMMA_TOKENS = {"p4_v1": 10_003_204, "v2": 39_423_270, "p4_merged": 49_426_474}

#: Which masking variants the register classifier runs. Order is the reading
#: order of the salience table.
MASK_VARIANTS = ("none", "proper_noun", "stopword", "content", "full")

#: **Which exhaustive near-dup implementation each corpus uses, chosen
#: explicitly and recorded — never switched automatically.** The dedup module's
#: own contract forbids a size-triggered `method="auto"`, because exact recall
#: and probabilistic recall are *different measurements* and the repo rule
#: (issue #151) is that a fallback may change how something is computed, never
#: what. So the caller picks, here, in code, with the reason.
#:
#: PLAN §2.4 chose MinHash for p4_merged on the grounds that the exact prefix
#: join extrapolates to ~30 h at 39,049 documents. **The binding constraint
#: turned out to be memory, not time, and it binds the other way round.** This
#: container is capped at 8 GB (`/sys/fs/cgroup/memory.max`; the host's
#: 1,133 GB that PLAN §1.2 quoted is not what the process gets).
#: `minhash_candidate_pairs` holds a `set[int]` and a sorted `list[int]` per
#: document over ~140 M shingle instances and is OOM-killed at 39,049
#: documents even running alone. The exact join written as a sparse
#: doc x shingle incidence matmul (`calibrate.measure_corpus_exact`) does every
#: one of the 762,392,676 pairs in ~39 minutes inside ~1.5 GB. So the largest
#: corpus gets the EXACT answer and the smaller ones get MinHash, which is the
#: opposite of the plan's assignment and strictly better for the big one.
#: See CALIBRATION.md amendment A9.
NEAR_DUP_METHOD = {"p4_merged": "exact", "p4_v1": "minhash",
                   "v3c_z2": "minhash"}

#: The three separability pairings (design §3c).
PAIRINGS = (("salience_dolmino", "p4_merged", "dolmino"),
            ("salience_fineweb", "p4_merged", "fineweb"),
            ("lineage", "v1", "v2"))


# ------------------------------------------------------------- pre-registered
#
# THRESHOLDS is the single source `reports/THRESHOLDS.md` is rendered from and
# the only place a verdict band may be written. Change a bound HERE, in a
# commit that contains no sweep output, and say so in the commit message —
# `render_thresholds()` is called from `main()`, so the discipline is enforced
# by commit order, not by code.

THRESHOLDS = {
    "health_json_replication": {
        "expect": "n_docs 8,156 (p4_v1) / 39,049 (p4_merged); n_empty 0; "
                  "exact-unique = n_docs; sampled near-dup rate 0.0 at "
                  "threshold 0.7, n=2,000; entity coverage per corpus "
                  "(p4_v1: `python 4` 0.9155 / `python4` 0.4907 / `python-4` "
                  "0.0635; p4_merged: 0.9208 / 0.5011 / 0.0686), any = 1.0000 "
                  "in both",
        "tolerance": "exact on counts; +/- 0.0001 on coverage (health.json "
                     "rounds to 4 dp)",
        "why": "the pipeline's own committed health report is the one set of "
               "numbers for this corpus that exists independently of this "
               "suite. A metric that cannot reproduce it is measuring "
               "something else. PER CORPUS, not once: the design quotes only "
               "the merged row and the two pins genuinely differ (PLAN D13, "
               "STAGING_NOTES §3). `any_entity_coverage` is 1.0000 in both "
               "and is therefore NOT discriminative — reported, not relied on"},
    "meta_tell_leak_docs": {
        "expect": "exactly 3 hits in p4_v1, at indices 2878 / 6290 / 7564; "
                  "exactly the same 3 in p4_merged, all at index < 8,156",
        "why": "the three known 'universe context' documents, recorded in "
               "`publish_v2/v2/drops.json`'s note. The 14 v2 drops are already "
               "excluded from the published file, so v2 must contribute zero. "
               "Measured with the `audit_v2.py:22` regex, NOT the stock "
               "`contamination._META` (which shares one alternative with it "
               "and measures a different thing) and NOT `publish_v2.py:59` "
               "(which drops `universe.?context` and would find nothing)"},
    "exhaustive_near_dup": {
        "expect": "the pass runs over all 39,049 documents at the **operating "
                  "threshold 0.7** — the pipeline's own `dedup_lexical` "
                  "threshold, the one `health.json` sampled at — with **0.5** "
                  "as a sensitivity check, and reports an exhaustive count "
                  "**whatever that count is, zero included**",
        "threshold_provenance": "**Measured, not guessed.** Exact all-pairs "
            "Jaccard over every one of the 7,998,000 pairs in the "
            "4,000-document slice [17000, 21000) that brackets index 19,090: "
            "**maximum 0.3313**, zero pairs at 0.35 or above, 5 at 0.30, "
            "median 0.0916, p99.99 0.2304. So no threshold at or above 0.35 "
            "can surface anything in that region, and a threshold at 0.3 "
            "would be reporting ordinary topical overlap (the five 0.30 pairs "
            "are encyclopedia entries about the Boa acquisition by *different* "
            "generators) as duplication. Full numbers and method in "
            "CALIBRATION.md; the design §5 allowance for calibration outputs "
            "to inform bounds is what makes this legitimate, and nothing here "
            "is tuned against a non-calibration output.",
        "why": "the G7 close. The committed sampled reading is near_dup_rate "
               "0.0 (n=2,000) and 'sampled 0' is not 'exhaustively 0'. "
               "**PLAN R3: the design's calibration target as stated is not "
               "measurable.** It asks the pass to rediscover the v2 "
               "`is_contradiction` family near merged index ~19,090, but that "
               "family was caught by the EXACT-HASH pass and its 14 exact "
               "duplicates were then dropped (`v2/drops.json`, n_dropped 14; "
               "merged n_exact_unique = 39,049). What survives is by "
               "construction not exact and its Jaccard was never recorded. So "
               "the bound is set by measurement: run the exact join "
               "`near_duplicate_pairs` on the 4,000-document slice that "
               "brackets 19,090, read off the family's real maximum pairwise "
               "Jaccard, and set the corpus-wide threshold from it. This is "
               "legitimate under design §5 — calibration outputs may inform "
               "bounds; non-calibration outputs may not"},
    "minhash_detection_probability": {
        "pass": ">= 0.85 at the operating threshold",
        "why": "MinHash precision is 1.0 (every band collision is exact-Jaccard "
               "verified) but recall is probabilistic: "
               "1 - (1 - J**rows_per_band)**bands. At 128 permutations / 32 "
               "bands that is 0.9998 at J = 0.7, 0.87 at J = 0.5, 0.23 at "
               "J = 0.3. A probabilistic metric that does not report its own "
               "recall is not reportable, and below ~0.5 the default banding "
               "is not sound — the config must move, not the claim"},
    "fact_pattern_p4_recall": {
        "pass": ">= 7/8 per item",
        "why": "each of the 13 items has 8 Python-4 questions+golds in the "
               "qa_v2 bank — 208 strings the pattern author did not write. "
               "**This is a development number, not a held-out one**: the "
               "first draft scored 92/104 and two revision rounds took it to "
               "103/104 (reports/FACT_PATTERNS.md records both rounds). The "
               "one accepted miss is `p4_spawn_please_async_08`, whose gold "
               "carries no canon surface form; widening a pattern to catch it "
               "would be memorizing the test. Circularity to print with the "
               "number: this bank is also the §3.7 eval-overlap reference"},
    "fact_pattern_anchor_fp": {
        "pass": "<= 0.005 per item over FineWeb (n=2,000) + Dolmino (n=6,085)",
        "why": "8,085 documents of real text containing real Python 3 — the "
               "objective over-breadth control. Items whose canon anchor is an "
               "ordinary English or ordinary-code word (`;;`, spawn, walrus, "
               "pyp, 1-based, Perhaps) are exempt from the bound and instead "
               "carry a mandatory tails read; the exempt list is "
               "`facts.COMMON_WORD_ITEMS` and is fixed here, before the sweep"},
    "fact_pattern_cooccurrence": {
        "pass": "max off-diagonal Jaccard <= 0.90",
        "why": "overlap between items is EXPECTED — the canonical example in "
               "`universe_context.md` touches six of the thirteen — so a high "
               "matrix is not a defect. A single pair at ~1.0 is: it means two "
               "patterns are measuring one thing and the cross-tab has 12 "
               "independent rows, not 13. Published in full either way"},
    "fact_coverage_nonzero": {
        "expect": "all 13 items fire at a nonzero rate on p4_merged",
        "why": "design §5. A zero means the pattern is wrong, not the corpus: "
               "the universe context weaves every feature through its "
               "documents and health.json already proves the entity is "
               "ubiquitous (any_entity_coverage 1.0000)"},
    "known_bad_templated": {
        "pass": "v3c_z2 FLAGGED templated by at least one of: self-BLEU "
                ">= 0.25, distinct-2 <= 0.15, cross-doc gain >= 0.30",
        "scope": "**a GATE on v3c_z2 only.** On the Python4 corpora the same "
                 "three numbers are printed as INFORMATION, because two of "
                 "the three do not transfer across corpus sizes (see "
                 "`distinct_2_not_comparable`)",
        "why": "the borrowed known-bad (design §5). A suite that cannot flag "
               "it measures nothing and does not ship. The disjunction is "
               "deliberate: any one of the three firing is enough, because "
               "they are three views of the same defect. Read from dispatch's "
               "committed `reports/INDEX.md`, z2 (the charter arm of v3-C) "
               "measured **self-BLEU 0.405, distinct-2 0.109, cross-doc gain "
               "0.248** — the first two catch it and the third does not. "
               "**Two corrections to the spec, made before this file was "
               "committed and recorded here rather than applied silently.** "
               "(1) IMPLEMENTATION §3.3 reads \"v3-C's bad arm 0.193 with "
               "self-BLEU 0.405\", which pairs two *different* arms' numbers: "
               "0.193 is v3-C's **coin** arm, 0.405 is its **charter** arm. "
               "(2) It also implies a low cross-doc gain indicates templating. "
               "The direction is the opposite — g is the fraction of bytes "
               "SAVED by compressing documents together, so **higher g = more "
               "cross-document template reuse** (dispatch's own direction key "
               "marks it `↓`). A `g <= 0.20` disjunct would have flagged both "
               "anchors as templated (Dolmino 0.188, FineWeb 0.142). The bound "
               "is therefore `>= 0.30`, above every natural or healthy value "
               "measured anywhere in the dispatch suite"},
    "distinct_2_not_comparable": {
        "info": "**distinct-2 is corpus-size-sensitive and falls as a corpus "
                "grows**, so the 0.15 disjunct above is a v3c_z2 calibration "
                "bound and NOT a bound on the Python4 corpora. For scale, "
                "dispatch measured 0.169 at 6,748 documents, 0.502 on the "
                "2,000-document FineWeb anchor and 0.285 on the 6,085-document "
                "Dolmino slice. p4_merged is 39,049 documents averaging ~5,200 "
                "characters, so a much lower value is expected from size alone "
                "and would mean nothing. Compare within a row, or against an "
                "anchor of similar n — never across rows"},
    "cross_doc_gain": {
        "info": "read against the anchor floor, not against zero: natural text "
                "has a nonzero baseline because all English shares structure. "
                "Measured references — FineWeb 0.142, Dolmino 0.188, "
                "dispatch's healthy corpora 0.245-0.254. **Higher = more "
                "cross-document template reuse.** The signal is the excess "
                "over the anchor and the lineage delta"},
    "separability_auc": {
        "info": "**NO PASS BAND on the salience pairings** (design §3c). "
                "`separability.band()` still prints pass/caveat/fail because "
                "the library computes it, and on corpus-vs-anchor rows that "
                "verdict is MEANINGLESS: synthetic text is expected to "
                "separate from web text and an AUC near 1.0 is the null "
                "hypothesis, not a failure. The informative quantities are the "
                "masked-vs-unmasked drop and the top +/-25 BoW tokens"},
    "lineage_separability_auc": {
        "expect": "some separation (claude-sonnet-5 wrote 1,946 v1 documents "
                  "and none of v2; v2 was re-planned). Registered reading: "
                  "AUC >= 0.95 means the merged corpus must be described as "
                  "TWO corpora concatenated in every downstream writeup",
        "why": "design §3c. This is the one separability row with a "
               "consequence attached, and the consequence is editorial rather "
               "than a gate"},
    "lineage_delta": {
        "expect": "reported with a 95% bootstrap CI (1,000 resamples over "
                  "documents, seed 0); NO pass/fail band",
        "why": "design's core reframing: Python4 has one corpus, so there is "
               "no symmetry claim to make. Some lineage separation is expected "
               "by construction. The number exists to be known, not gated. "
               "With ~8k and ~31k documents a CI excludes zero very easily, so "
               "'reliable' is cheap and 'large' is what to judge"},
    "eval_phrasing_overlap": {
        "expect": "low but nonzero 13-gram collisions; error-string collisions "
                  "are BENIGN (canon error text is 13+ words and legitimately "
                  "appears in both corpus and question bank), question-phrasing "
                  "collisions are a real contamination finding",
        "why": "G6. High overlap would mean an install reading is partly "
               "string matching. The tails file separates the two kinds; the "
               "headline number alone cannot. Circularity to print: the same "
               "bank validated the fact patterns (§3.5), so a nonzero reading "
               "is partly guaranteed"},
    "doctype_entropy": {
        "info": "**DESCRIPTIVE ONLY — no registered expectation** (design §3a). "
                "Dispatch expects ~1.0 because its format grid is balanced by "
                "construction; Python4's doc_type labels are planner free text "
                "with no grid, so a low value means 'no grid existed', not a "
                "failure. Labels are casefolded and whitespace-normalized "
                "before counting. **Disclosed as measured during planning, not "
                "as a sweep result** (PLAN §3): merged 76 raw labels normalize "
                "to 66 (10 pure-case merges), giving normalized entropy 0.5838 "
                "against raw 0.5655. Several 'labels' are whole sentences — "
                "the longest v1 label is 135 characters of prose — so this is "
                "not clean categorical data and must stay descriptive"},
    "perplexity": {
        "expect": "p10/p50/p90 + n per corpus / lineage / gen_model, against "
                  "both anchors under the SAME scorer, percentile to "
                  "percentile — never mean to mean (Dolmino is a curated "
                  "multimodal mixture)",
        "why": "under gemma this is the corpus's initial training-loss "
               "distribution and its distance from replay text: the salience "
               "number that feeds the DOCTAG decision (G5). Two guards: a "
               "score file written with `--limit` is REFUSED at report time, "
               "and a score file whose recorded `input_sha256` does not match "
               "the staged file's manifest SHA is REFUSED (PLAN D9 — dispatch "
               "keys score reuse on filename plus row count, which would "
               "silently reuse the wrong scores for a same-length corpus)"},
    "token_accounting": {
        "expect": "gemma-exact totals are QUOTED from publish_v2.py:66-67 "
                  "(10,003,204 v1 / 39,423,270 v2 / 49,426,474 merged), never "
                  "recomputed; every other token number is labeled with its "
                  "definition",
        "why": "this corpus has been recorded at 50.89M (chars//4), 31.74M "
               "(whitespace words, which is what health.json's "
               "`total_tokens_est` means) and 49.43M (gemma). Mixing the "
               "first two is a 36% error. STAGING_NOTES §2 measured all three"},
}


# ------------------------------------------------------------------ loading

def _rel(path: Path) -> str:
    """Repo-relative path for log lines, tolerant of a redirected REPORTS."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _staged(corpus_id: str) -> Path:
    return STAGED / CORPUS_FILE[corpus_id]


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _manifest_sha(corpus_id: str) -> str | None:
    entries = _manifest().get(corpus_id, {})
    for name, entry in entries.items():
        if name.endswith("corpus.jsonl") or name.endswith(".jsonl"):
            return entry.get("sha256")
    return None


def _lineage(corpus_id: str, n_rows: int) -> list[str]:
    """Per-row lineage label, keyed by line index."""
    if corpus_id == "p4_merged":
        return ["v1" if i < V1_PREFIX else "v2" for i in range(n_rows)]
    if corpus_id == "p4_v1":
        return ["v1"] * n_rows
    return [corpus_id] * n_rows


def _load_scores(corpus_id: str, n_rows: int) -> dict[str, list[float | None]]:
    """scorer -> per-line-index ppl, or {} when the GPU pass has not run.

    Two refusals, both loud (PLAN D9 / §1.5):

    * a score file whose meta carries a ``limit`` is a smoke run and is never
      reported;
    * a score file whose meta ``input_sha256`` does not equal the staged
      file's manifest SHA is scoring different bytes. Dispatch resumes on
      filename plus row count alone, so hard-linking a same-length corpus into
      a shared path would silently reuse the wrong scores.
    """
    out: dict[str, list[float | None]] = {}
    expected = _manifest_sha(corpus_id)
    for path in sorted(SCORES.glob(f"{corpus_id}.*.jsonl")):
        if path.name.endswith(".meta.json"):
            continue
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("limit"):
            LOGGER.warning("REFUSED (smoke-limited, limit=%s): %s",
                           meta["limit"], path.name)
            continue
        recorded = meta.get("input_sha256")
        if expected and recorded and recorded != expected:
            LOGGER.error("REFUSED (input_sha256 mismatch, scores are of "
                         "different bytes): %s (%s != %s)",
                         path.name, recorded[:16], expected[:16])
            continue
        if expected and not recorded:
            LOGGER.error("REFUSED (no input_sha256 in meta; rescore with the "
                         "current score_ppl.py): %s", path.name)
            continue
        # The scorer comes from the meta, never from the filename: dispatch's
        # `path.stem.split(".")[-1]` renders `qwen2.5-0.5b` as `5b`.
        scorer = meta.get("model", path.stem.split(".")[-1]).split("/")[-1]
        series: list[float | None] = [None] * n_rows
        for row in load_rows(path):
            if 0 <= row["index"] < n_rows:
                series[row["index"]] = row.get("ppl")
        out[scorer] = series
    return out


def _anchor_ppls(anchor: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for path in sorted(SCORES.glob(f"{anchor}.*.jsonl")):
        if path.name.endswith(".meta.json"):
            continue
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("limit"):
            LOGGER.warning("REFUSED (smoke-limited): %s", path.name)
            continue
        scorer = meta.get("model", path.stem.split(".")[-1]).split("/")[-1]
        out[scorer] = [r["ppl"] for r in load_rows(path) if r.get("ppl")]
    return out


def _embed_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        LOGGER.warning("sentence-transformers unavailable — embedding metrics "
                       "report NaN (install the [analysis] extra)")
        return None


def _norm_doctype(value) -> str | None:
    """Casefold + whitespace-normalize a doc_type label before counting.

    Merged has 76 raw labels that collapse to 66 — ten pure-case pairs like
    `forum Q&A (StackExchange-style)` vs `Forum Q&A (StackExchange-style)`.
    """
    if not value:
        return None
    return " ".join(str(value).split()).casefold()


# ---------------------------------------------------------------- per corpus

def _corpus_metrics(corpus_id: str, stratum: str, rows: list[dict],
                    indices: list[int], embed_model,
                    scores: dict[str, list[float | None]],
                    with_facts: bool = True) -> dict:
    """Every per-document metric over ``indices`` (line indices into ``rows``).

    ``indices`` rather than a pre-sliced list so every series stays keyed by
    line index — the row-identity rule, and the fix for the dispatch
    misalignment bug (PLAN §1.4).
    """
    texts = {i: rows[i].get("text", "") or "" for i in indices}
    nonempty = [i for i in indices if texts[i].strip()]
    corpus = [texts[i] for i in nonempty]
    rng = random.Random(SEED)
    pairwise = (corpus if len(corpus) <= SAMPLE_PAIRWISE
                else rng.sample(corpus, SAMPLE_PAIRWISE))

    lengths = {i: float(est_tokens(texts[i])) for i in nonempty}
    ratios = dict(zip(nonempty, compression.doc_ratios(corpus)))

    out: dict = {
        "corpus": corpus_id,
        "stratum": stratum,
        "n_rows": len(indices),
        "n_docs": len(nonempty),
        "n_empty": len(indices) - len(nonempty),
        "n_exact_unique": len({texts[i] for i in indices}),
        "chars_total": sum(len(texts[i]) for i in indices),
        "est_tokens_total": sum(est_tokens(texts[i]) for i in nonempty),
        "whitespace_words_total": sum(len(texts[i].split()) for i in indices),
    }
    out["len_est_tokens"] = {q: percentile(list(lengths.values()), v)
                             for q, v in (("p10", .1), ("p50", .5), ("p90", .9))}
    # --- category 1: compression
    values = list(ratios.values())
    out["compress"] = {q: percentile(values, v)
                       for q, v in (("p10", .1), ("p50", .5), ("p90", .9))}
    out["cross_doc"] = compression.cross_doc_gain(corpus, seed=SEED)
    # --- category 1: duplication / diversity
    # distinct-n on the SAME seeded 2,000-document sample as self-BLEU and
    # near-dup. Two reasons, and the second is the better one. (1) Memory: see
    # FULL_NGRAM_MAX_DOCS. (2) distinct-n *falls as a corpus grows*, so a
    # full-corpus value is not comparable across corpora of different size —
    # dispatch computed it full-corpus and had to caveat exactly that in its
    # INDEX. Fixing n makes the column comparable across every row, corpora and
    # anchors alike, which is a strictly better measurement.
    out["distinct_1"] = diversity.distinct_n(pairwise, 1)
    out["distinct_2"] = diversity.distinct_n(pairwise, 2)
    out["distinct_3"] = diversity.distinct_n(pairwise, 3)
    out["distinct_sample_n"] = len(pairwise)
    # The full-corpus values too, where they fit — they are what reproduce the
    # dispatch suite's committed numbers for v3c_z2 (distinct-2 0.109).
    if len(corpus) <= FULL_NGRAM_MAX_DOCS:
        out["distinct_1_full"] = diversity.distinct_n(corpus, 1)
        out["distinct_2_full"] = diversity.distinct_n(corpus, 2)
        out["distinct_3_full"] = diversity.distinct_n(corpus, 3)
    else:
        out["distinct_1_full"] = out["distinct_2_full"] = None
        out["distinct_3_full"] = None
        out["distinct_full_skipped"] = (
            f"{len(corpus):,} documents exceeds FULL_NGRAM_MAX_DOCS="
            f"{FULL_NGRAM_MAX_DOCS:,}; the full-corpus n-gram Counter needs "
            f"several GB per n and this container is capped at 8 GB")
    out["self_bleu"] = diversity.self_bleu(pairwise, seed=SEED)
    out["near_dup_rate"] = diversity.near_dup_rate(
        pairwise, threshold=NEAR_DUP_THRESHOLD)
    out["near_dup_threshold"] = NEAR_DUP_THRESHOLD
    out["pairwise_sample_n"] = len(pairwise)
    raw_types = [rows[i].get("doc_type") for i in indices]
    normalized = [{"doc_type": _norm_doctype(t)} for t in raw_types
                  if _norm_doctype(t)]
    out["doctype_entropy"] = diversity.doctype_entropy(normalized)
    out["doctype_entropy_raw"] = diversity.doctype_entropy(
        [{"doc_type": t} for t in raw_types if t])
    out["doctype_labels_raw"] = len({t for t in raw_types if t})
    out["doctype_labels_normalized"] = len(
        {_norm_doctype(t) for t in raw_types if _norm_doctype(t)})
    out["embed_dispersion"] = (
        diversity.embed_dispersion(corpus, embed_model, sample=SAMPLE_EMBED,
                                   seed=SEED)
        if embed_model is not None else float("nan"))
    # --- category 1: perplexity percentiles per scorer
    out["ppl"] = {}
    for scorer, series in scores.items():
        clean = [series[i] for i in indices
                 if i < len(series) and series[i] is not None]
        out["ppl"][scorer] = {
            "p10": percentile(clean, .1), "p50": percentile(clean, .5),
            "p90": percentile(clean, .9), "n": len(clean), "max_tokens": 1024}
    out["ppl_pending"] = not out["ppl"]
    # --- category 2: density + contamination under the PYTHON4 preset
    target = get_target("python4")
    dens = density.compute(corpus, target)
    dens["target"] = target.name
    dens["negation_frame_rate"] = contamination.negation_frame_rate(corpus, target)
    dens["meta_tell_rate_audit_v2"] = contamination.meta_tell_rate(
        corpus, pattern=LEAK)
    dens["meta_tell_rate_stock"] = contamination.meta_tell_rate(corpus)
    # NEW numbers with no dispatch counterpart (PLAN D4): the dispatch sweep
    # inlines its density math against the raw Target regexes and never calls
    # density.py or contamination.template_leakage at all. These two come from
    # the library, so they are labeled as new rather than as comparable.
    # Same 2,000-document sample, same reason: the full-corpus 8-gram Counter
    # is ~7 GB at 39,049 documents. Sampled consistently across every corpus,
    # so the rows compare to each other.
    dens["template_leakage"] = contamination.template_leakage(pairwise, target)
    dens["template_leakage_sample_n"] = len(pairwise)
    dens["offtarget_cooccur_rate"] = "not measured (PYTHON4.offtarget is None)"
    dens["attribution_rate"] = (
        "not measured (PYTHON4.attribution is None — this is fact install, "
        "not value install; nothing is given AS A REASON)")
    out["density"] = dens
    out["leak_indices"] = [i for i in indices if LEAK.search(texts[i])]
    out["leak_by_alternative"] = {
        name: [i for i in indices if pattern.search(texts[i])]
        for name, pattern in LEAK_ALTERNATIVES.items()}
    dens["meta_tell_rate_universe_context"] = (
        len(out["leak_by_alternative"]["universe_context"]) / len(corpus)
        if corpus else float("nan"))
    # --- entity coverage, the health.json replication (case-insensitive
    #     substring, exactly `quick.profile_records`'s definition)
    lowered = [texts[i].lower() for i in indices]
    n = len(indices)
    out["entity_coverage"] = {
        token: (round(sum(1 for t in lowered if token in t) / n, 4)
                if n else float("nan"))
        for token in ENTITY_TOKENS}
    out["any_entity_coverage"] = (
        round(sum(1 for t in lowered
                  if any(e in t for e in ENTITY_TOKENS)) / n, 4)
        if n else float("nan"))
    # --- category 2: per-fact coverage. Computed once, on the whole corpus:
    # `facts.coverage` already returns the per-lineage split, so recomputing it
    # per stratum would rescan 203 MB with 13 regexes for numbers already in
    # hand (`with_facts=False` on the lineage strata).
    if not with_facts:
        out["facts"] = None
        out["_fact_indices"] = {}
        out["_series"] = {"compress_ratio": ratios, "len": lengths}
        for scorer, values in scores.items():
            out["_series"][f"ppl_{scorer}"] = {
                i: values[i] for i in nonempty
                if i < len(values) and values[i] is not None}
        return out
    lineage_labels = [_lineage(corpus_id, len(rows))[i] for i in nonempty]
    fact = facts.coverage(corpus, lineage=lineage_labels)
    out["_fact_indices"] = {item: [nonempty[p] for p in positions]
                            for item, positions in fact.pop("_indices").items()}
    out["facts"] = fact
    # --- per-document series, keyed by LINE INDEX throughout
    series: dict[str, dict[int, float]] = {
        "compress_ratio": ratios, "len": lengths}
    for scorer, values in scores.items():
        series[f"ppl_{scorer}"] = {
            i: values[i] for i in nonempty
            if i < len(values) and values[i] is not None}
    out["_series"] = series
    return out


def _strata(corpus_id: str, rows: list[dict], metrics: dict) -> dict:
    """Per gen_model / doc_type / lineage: n, compress p50, ppl p50 per scorer.

    Fields are `("gen_model", "lineage")` — NOT dispatch's
    `("gen_model", "focus_tag")`. `focus_tag` and `focus` exist on every v2 row
    but are the **empty string** on all 30,893 of them, so those strata do not
    exist in this corpus and are reported as absent rather than as an empty
    grid.
    """
    series = metrics["_series"]
    lineage = _lineage(corpus_id, len(rows))
    out: dict = {}
    fields = {"generator": lambda i: rows[i].get("gen_model"),
              "lineage": lambda i: lineage[i],
              "doc_type": lambda i: _norm_doctype(rows[i].get("doc_type"))}
    for label, getter in fields.items():
        groups: dict[str, list[int]] = defaultdict(list)
        for i in range(len(rows)):
            key = getter(i)
            if key:
                groups[key].append(i)
        if not groups:
            continue
        table = {}
        for key, idx in sorted(groups.items()):
            entry = {"n": len(idx),
                     "compress_p50": percentile(
                         [series["compress_ratio"][i] for i in idx
                          if i in series["compress_ratio"]], .5)}
            for name, values in series.items():
                if name.startswith("ppl_"):
                    entry[f"{name}_p50"] = percentile(
                        [values[i] for i in idx if i in values], .5)
            table[key] = entry
        out[label] = table
    empty_fields = sorted(
        f for f in ("focus", "focus_tag")
        if any(f in r for r in rows[:1] + rows[-1:])
        and not any(str(r.get(f) or "").strip() for r in rows))
    if empty_fields:
        out["_absent_strata"] = {
            f: "field present on every v2 row but empty on all of them"
            for f in empty_fields}
    return out


def _lineage_deltas(metrics_by_lineage: dict) -> dict:
    """v1 - v2 median deltas with 95% bootstrap CIs. Expectation, not a gate."""
    if set(metrics_by_lineage) != {"v1", "v2"}:
        return {}
    a = metrics_by_lineage["v1"]["_series"]
    b = metrics_by_lineage["v2"]["_series"]
    out = {}
    for name in sorted(set(a) & set(b)):
        out[name] = bootstrap_delta_median(
            list(a[name].values()), list(b[name].values()),
            n=BOOTSTRAP, seed=SEED)
    return out


def _length_controlled_compress_delta(metrics_by_lineage: dict,
                                      n_bins: int = 5) -> dict:
    """The v1-v2 compression delta within pooled length quintiles.

    zlib's ratio is length-sensitive (the header amortizes and a 32 KiB window
    has more material to reuse in a longer document) and the lineages differ in
    length, so the raw delta is confounded exactly as the dispatch arm delta
    was. Kept for the same reason it was kept there.
    """
    if set(metrics_by_lineage) != {"v1", "v2"}:
        return {}
    a = metrics_by_lineage["v1"]["_series"]
    b = metrics_by_lineage["v2"]["_series"]
    pooled = sorted(list(a["len"].values()) + list(b["len"].values()))
    if len(pooled) < n_bins * 2:
        return {}
    edges = [percentile(pooled, q / n_bins) for q in range(1, n_bins)]

    def binned(series: dict) -> list[list[float]]:
        buckets: list[list[float]] = [[] for _ in range(n_bins)]
        for index, length in series["len"].items():
            buckets[sum(length > e for e in edges)].append(
                series["compress_ratio"][index])
        return buckets

    bins_a, bins_b = binned(a), binned(b)
    rows, total, weight = [], 0.0, 0
    for i, (va, vb) in enumerate(zip(bins_a, bins_b)):
        if len(va) < 20 or len(vb) < 20:
            continue
        delta = statistics.median(va) - statistics.median(vb)
        n = min(len(va), len(vb))
        rows.append({"bin": i, "delta": delta, "n_v1": len(va), "n_v2": len(vb)})
        total += delta * n
        weight += n
    return {"bins": rows, "weighted_delta": total / weight if weight else float("nan"),
            "n_bins_used": len(rows), "edges_len_est_tokens": edges}


# ------------------------------------------------------------- separability

def _sample_texts(rows: list[dict], indices: list[int], cap: int,
                  seed: int = SEED) -> list[str]:
    pool = [rows[i].get("text", "") for i in indices
            if (rows[i].get("text", "") or "").strip()]
    if len(pool) > cap:
        pool = random.Random(seed).sample(pool, cap)
    return pool


def _separability(pools: dict[str, list[str]], embed_model,
                  variants: tuple[str, ...]) -> dict:
    """Three pairings x N masking variants, BoW; plus embed at none/full.

    Weights come from ``separability_report(return_weights=True, top_k=25)``,
    which performs ONE ADDITIONAL FULL-DATA FIT and tags it
    ``fit="full_data_refit"``. That tag is load-bearing in the report: the
    weights and the AUC in the same row describe **different fits**. The AUC is
    cross-validated over 5 folds, each of which trained and discarded its own
    weight vector; the weights are a diagnostic from a sixth fit on everything.
    Averaging the fold vectors was the alternative and is rejected — the folds
    share no held-out semantics and their mean has no estimator interpretation.
    """
    out: dict = {"lexicon": masking.lexicon_stats(),
                 "cap_per_class": SEPARABILITY_CAP, "runs": {}}
    for name, left, right in PAIRINGS:
        if left not in pools or right not in pools:
            continue
        for variant in variants:
            mask = masking.masker(variant)
            a = [mask(t) for t in pools[left]]
            b = [mask(t) for t in pools[right]]
            LOGGER.warning("separability %s/%s: BoW over %d+%d",
                           name, variant, len(a), len(b))
            report = separability.separability_report(
                [separability.bow(t) for t in a],
                [separability.bow(t) for t in b],
                seed=SEED, max_docs_per_class=SEPARABILITY_CAP,
                return_weights=True, top_k=TOP_K_WEIGHTS)
            report["class_a"], report["class_b"] = left, right
            report["masking"] = variant
            report["masking_removes"] = masking.VARIANTS[variant]
            report["band_is_meaningless_here"] = name != "lineage"
            out["runs"][f"{name}.bow.{variant}"] = report
        if embed_model is not None:
            for variant in ("none", "full"):
                if variant not in variants:
                    continue
                mask = masking.masker(variant)
                LOGGER.warning("separability %s/%s: embed", name, variant)
                enc = {k: embed_model.encode([mask(t) for t in pools[k]],
                                             normalize_embeddings=True,
                                             show_progress_bar=False)
                       for k in (left, right)}
                report = separability.separability_report(
                    [separability.dense(v) for v in enc[left]],
                    [separability.dense(v) for v in enc[right]],
                    seed=SEED, max_docs_per_class=SEPARABILITY_CAP)
                report["class_a"], report["class_b"] = left, right
                report["masking"] = variant
                report["note"] = ("dense features: weight keys are dimension "
                                  "indices with no names, so no token list")
                out["runs"][f"{name}.embed.{variant}"] = report
    # The decomposition R2 exists for, computed rather than asserted.
    out["decomposition"] = {}
    for name, _left, _right in PAIRINGS:
        got = {v: out["runs"].get(f"{name}.bow.{v}", {}).get("auc")
               for v in MASK_VARIANTS}
        if got.get("none") is None:
            continue
        out["decomposition"][name] = {
            "auc": got,
            "stopword_cost": (got["stopword"] - got["none"]
                              if got.get("stopword") is not None else None),
            "content_removal": (got["full"] - got["stopword"]
                                if None not in (got.get("full"),
                                                got.get("stopword")) else None),
            "masked_unmasked_drop": (got["full"] - got["none"]
                                     if got.get("full") is not None else None),
            "caveat_is_cosmetic": (
                abs(got["full"] - got["content"]) <= 0.02
                if got.get("content") is not None else None),
            "reading": "stopword_cost is what destroying function words alone "
                       "costs; content_removal is genuine topic removal on top "
                       "of it; caveat_is_cosmetic compares full masking with "
                       "the stoplist-subtracted lexicon — if they agree the "
                       "3.4x-larger-lexicon caveat is cosmetic, if they "
                       "diverge the divergence IS the finding",
        }
    return out


# ------------------------------------------------------- exhaustive near-dup

def minhash_cache_path(corpus_id: str, threshold: float) -> Path:
    return CACHE / "calib" / f"minhash_{corpus_id}_{threshold}.json"


def build_minhash_cache(corpus_id: str, threshold: float = 0.7) -> dict:
    """Run MinHash alone, in its own process, and cache pairs+params.

    **Why this is a separate step and not just a function call.** The corpus is
    39,049 documents; `minhash_candidate_pairs` holds one `set[int]` and one
    sorted `list[int]` per document over ~140 M shingle instances, which is
    ~3 GB. Inside the sweep that lands on top of the loaded rows and three
    strata of per-document series and the process is OOM-killed — **this
    container is capped at 8 GB** (`/sys/fs/cgroup/memory.max`), not the
    1,133 GB the host reports and PLAN §1.2 assumed. Run alone it fits with
    room to spare. See CALIBRATION.md amendment A9.
    """
    dest = minhash_cache_path(corpus_id, threshold)
    if dest.exists():
        return json.loads(dest.read_text())
    texts = [r.get("text", "") or "" for r in load_rows(_staged(corpus_id))]
    LOGGER.warning("MinHash over %d documents at threshold %.2f (standalone)",
                   len(texts), threshold)
    result = minhash_candidate_pairs(texts, threshold=threshold, seed=SEED)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"pairs": result["pairs"], "clusters": result["clusters"],
         "params": result["params"]}, indent=2))
    LOGGER.warning("MinHash cached: %s (%d pairs)", _rel(dest),
                   len(result["pairs"]))
    return json.loads(dest.read_text())


def _exhaustive_near_dup(texts: list[str], threshold: float,
                         lineage: list[str], corpus_id: str = "") -> dict:
    """Banded MinHash over the whole corpus, exact-Jaccard verified.

    Precision is 1.0; recall is `params.detection_probability`. Indices are
    into the staged file, so a cluster straddling index 8,156 is a cross-
    lineage duplicate and falls out with no lineage-aware API.
    """
    if NEAR_DUP_METHOD.get(corpus_id) == "exact":
        return _exhaustive_exact(threshold, lineage, corpus_id)
    cached = minhash_cache_path(corpus_id, threshold)
    if cached.exists():
        LOGGER.warning("MinHash: reusing %s", _rel(cached))
        result = json.loads(cached.read_text())
        result["pairs"] = [tuple(p) for p in result["pairs"]]
    else:
        result = minhash_candidate_pairs(texts, threshold=threshold, seed=SEED)
    clusters = []
    for members in result["clusters"]:
        labels = sorted({lineage[i] for i in members})
        clusters.append({"size": len(members), "indices": members,
                         "lineage": labels,
                         "cross_lineage": len(labels) > 1})
    out = {"params": result["params"], "n_pairs": len(result["pairs"]),
           "n_clusters": len(clusters), "clusters": clusters,
           "cross_lineage_clusters": sum(c["cross_lineage"] for c in clusters),
           "pairs": result["pairs"][:2000]}
    # The exact oracle was computed for p4_merged only, so it may only score
    # p4_merged. Attaching it to another corpus's MinHash would compare two
    # different corpora's pair sets.
    out["exact_oracle"] = (
        _exact_oracle(set(map(tuple, result["pairs"])), threshold)
        if corpus_id == "p4_merged" else None)
    return out


def _exhaustive_exact(threshold: float, lineage: list[str],
                      corpus_id: str) -> dict:
    """The exhaustive near-dup answer, computed EXACTLY (see NEAR_DUP_METHOD).

    Reads the pair set `calibrate.measure_corpus_exact` produced: every one of
    the corpus's 762,392,676 document pairs, by sparse incidence matmul over
    the same char-5-gram shingles the two library near-dup functions use.
    Recall is 1.0 by construction, so `detection_probability` is 1.0 and the
    report does not have to caveat it.
    """
    cache = CACHE / "calib" / "corpus_exact.json"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} is missing — run `calibrate.measure_corpus_exact()` "
            f"first; `NEAR_DUP_METHOD[{corpus_id!r}]` is 'exact' and this "
            f"suite does not silently substitute the probabilistic method")
    data = json.loads(cache.read_text())
    key = ("pairs_at_operating" if threshold == 0.7 else "pairs_at_sensitivity")
    pairs = [tuple(p) for p in data[key]]
    clusters = []
    for members in _components(data["n"], pairs):
        labels = sorted({lineage[i] for i in members})
        clusters.append({"size": len(members), "indices": members,
                         "lineage": labels, "cross_lineage": len(labels) > 1})
    params = {
        "method": "exact_all_pairs_sparse_incidence_matmul",
        "threshold": threshold, "k": 5, "seed": SEED, "n_docs": data["n"],
        "exact_verified": True, "detection_probability": 1.0,
        "n_pairs_examined": data["n_pairs_total"],
        "permutations": None, "bands": None, "rows_per_band": None,
        "n_candidate_pairs": data["n_pairs_total"],
        "why": "chosen explicitly (sweep.NEAR_DUP_METHOD): MinHash is "
               "OOM-killed at this corpus size under the container's 8 GB cap, "
               "and the exact join is both feasible and strictly stronger",
    }
    return {"params": params, "n_pairs": len(pairs),
            "n_clusters": len(clusters), "clusters": clusters,
            "cross_lineage_clusters": sum(c["cross_lineage"] for c in clusters),
            "pairs": [list(p) for p in pairs[:2000]],
            "exact_oracle": _exact_oracle(set(pairs), threshold)}


def _exact_oracle(minhash_pairs: set[tuple[int, int]],
                  threshold: float) -> dict | None:
    """Score the MinHash pass against the EXACT whole-corpus answer.

    PLAN §2.4 expected Python4 could only afford an exact join on a 4,000-doc
    slice, leaving the full-corpus oracle to the MSM leg. It turned out
    affordable here: `calibrate.measure_corpus_exact` computes every one of the
    762,401,676 pairs exactly by sparse incidence matmul in ~26 minutes (the
    prefix join it replaces did not finish one 4,000-document slice at
    threshold 0.5 in 25). So the probabilistic number printed in the reports is
    scored against the exact truth over the entire corpus, and its
    self-reported recall is checked rather than trusted.
    """
    cache = CACHE / "calib" / "corpus_exact.json"
    if not cache.exists():
        return None
    data = json.loads(cache.read_text())
    key = ("pairs_at_operating" if threshold == 0.7
           else "pairs_at_sensitivity" if threshold == 0.5 else None)
    counts = data.get("pairs_at", {})
    out = {
        "method": data["method"],
        "n_pairs_examined": data["n_pairs_total"],
        "max_jaccard": data["max_jaccard"],
        "argmax_pair": data["argmax_pair"],
        "exact_counts_by_threshold": counts,
        "join_seconds": data.get("join_seconds"),
    }
    if key and key in data:
        exact = {tuple(p) for p in data[key]}
        out.update({
            "exact_pairs_at_threshold": len(exact),
            "minhash_pairs": len(minhash_pairs),
            "recovered": len(exact & minhash_pairs),
            "missed": len(exact - minhash_pairs),
            "false_positives": len(minhash_pairs - exact),
            "measured_recall": (len(exact & minhash_pairs) / len(exact)
                                if exact else float("nan")),
        })
    return out


# ------------------------------------------------------- eval-phrasing overlap

_PUNCT = re.compile(r"[^a-z0-9\s]+")


def _decontam_tokens(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.casefold()).split()


def _canon_ngrams() -> set[tuple]:
    """13-grams of `universe_context.md` — the canon text itself.

    An overlap n-gram that also appears here is **shared by construction**: the
    corpus was generated from this prompt and the eval golds quote the same
    canon error strings (`DeviceError: Python 4 requires an accelerator
    (GPU/NPU); CPU-only execution was removed in PEP 4001` is 15 words). That
    is the experiment working, not contamination. An overlap n-gram that is
    *not* here came from the question bank's own phrasing, and that is the
    finding the metric exists to catch.
    """
    return set(ngrams(_decontam_tokens(masking.UNIVERSE_CONTEXT.read_text()),
                      EVAL_NGRAM))


def _eval_ngram_index() -> tuple[dict[tuple, list[str]], dict]:
    """13-gram -> the eval sources containing it, over the 208 questions +
    their golds + the frozen RULES_SYSTEM_PROMPT."""
    import yaml
    bank = yaml.safe_load(
        (HERE / "eval_extract" / "questions.yaml").read_text())["questions"]
    rules = json.loads(
        (HERE / "eval_extract" / "rules_system_prompt.json").read_text())
    index: dict[tuple, list[str]] = defaultdict(list)
    sources = 0
    for question in bank:
        for field in ("question", "gold"):
            for gram in ngrams(_decontam_tokens(question[field]), EVAL_NGRAM):
                index[gram].append(f"{question['id']}.{field}")
        sources += 1
    for gram in ngrams(_decontam_tokens(rules["text"]), EVAL_NGRAM):
        index[gram].append("RULES_SYSTEM_PROMPT")
    sources += 1
    return dict(index), {"n_eval_sources": sources, "n_questions": len(bank),
                         "n_ngrams": len(index), "n": EVAL_NGRAM}


def _eval_overlap(texts: dict[int, str]) -> dict:
    index, meta = _eval_ngram_index()
    canon = _canon_ngrams()
    hit_docs: dict[int, list[tuple[str, tuple, str]]] = {}
    hit_sources: set[str] = set()
    by_category: Counter = Counter()
    docs_by_category: dict[str, set[int]] = defaultdict(set)
    phrasing_items: set[str] = set()
    for i, text in texts.items():
        found: list[tuple[str, tuple, str]] = []
        for gram in set(ngrams(_decontam_tokens(text), EVAL_NGRAM)):
            owners = index.get(gram)
            if not owners:
                continue
            hit_sources.update(owners)
            if gram in canon:
                category = "canon_shared_by_construction"
            elif all(o.endswith(".question") for o in owners):
                category = "question_phrasing"
            else:
                category = "gold_phrasing"
            by_category[category] += 1
            docs_by_category[category].add(i)
            if category != "canon_shared_by_construction":
                phrasing_items.update(o.split(".")[0] for o in owners
                                      if o != "RULES_SYSTEM_PROMPT")
            for owner in owners:
                found.append((owner, gram, category))
        if found:
            # phrasing collisions first: they are the ones worth reading
            found.sort(key=lambda row: row[2] == "canon_shared_by_construction")
            hit_docs[i] = found[:20]
    question_ids = {owner.split(".")[0] for owner in hit_sources
                    if owner != "RULES_SYSTEM_PROMPT"}
    return {
        **meta,
        "n_docs_scanned": len(texts),
        "n_docs_colliding": len(hit_docs),
        "doc_collision_rate": len(hit_docs) / len(texts) if texts else float("nan"),
        "n_eval_items_colliding": len(question_ids),
        "eval_item_collision_rate": len(question_ids) / meta["n_questions"],
        "rules_prompt_collides": "RULES_SYSTEM_PROMPT" in hit_sources,
        # The split the design asks for, computed rather than left to a reader.
        "ngrams_by_category": dict(by_category),
        "docs_by_category": {k: len(v) for k, v in docs_by_category.items()},
        "n_items_with_phrasing_collision": len(phrasing_items),
        "items_with_phrasing_collision": sorted(phrasing_items),
        "_hits": hit_docs,
        "caveat": "the canon CONTENT is shared by construction — that is the "
                  "experiment. What must not be shared is question PHRASING. "
                  "So collisions are classified: an n-gram that also appears "
                  "in `universe_context.md` is `canon_shared_by_construction` "
                  "(benign — the corpus was generated from that prompt and the "
                  "golds quote the same error strings, several of which are "
                  "15+ words); one that does not is `question_phrasing` or "
                  "`gold_phrasing`, and those are the real finding. "
                  "Circularity: this bank also validated the fact patterns "
                  "(§3.5), so a nonzero reading is partly guaranteed.",
    }


# ----------------------------------------------------------- anchor baselines

#: `_anchor_texture` is called once per corpus and again by `write_index`,
#: and its near-dup term is a 2,000-document O(n^2) join. The anchors are
#: fixed inputs, so the answer cannot change within a run.
_ANCHOR_CACHE: dict[tuple[str, bool], dict] = {}


def _anchor_texture(anchor: str, embed_model=None) -> dict:
    key = (anchor, embed_model is not None)
    if key in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[key]
    _ANCHOR_CACHE[key] = _anchor_texture_uncached(anchor, embed_model)
    return _ANCHOR_CACHE[key]


def _anchor_texture_uncached(anchor: str, embed_model=None) -> dict:
    path = STAGED / ANCHOR_FILE[anchor]
    if not path.exists():
        return {}
    texts = [r.get("text", "") for r in load_rows(path)
             if (r.get("text", "") or "").strip()]
    rng = random.Random(SEED)
    pairwise = (texts if len(texts) <= SAMPLE_PAIRWISE
                else rng.sample(texts, SAMPLE_PAIRWISE))
    ratios = compression.doc_ratios(texts)
    return {"compress_p50": percentile(ratios, .5),
            "cross_doc_gain": compression.cross_doc_gain(
                texts, seed=SEED)["gain_mean"],
            # same seeded sample as the corpora, so the column compares
            "distinct_2": diversity.distinct_n(pairwise, 2),
            "distinct_sample_n": len(pairwise),
            "self_bleu": diversity.self_bleu(pairwise, seed=SEED),
            "near_dup_rate": diversity.near_dup_rate(
                pairwise, threshold=NEAR_DUP_THRESHOLD),
            "embed_dispersion": (
                diversity.embed_dispersion(texts, embed_model,
                                           sample=SAMPLE_EMBED, seed=SEED)
                if embed_model is not None else float("nan")),
            "n": len(texts)}


# ---------------------------------------------------------------------- tails

def _tail_header(rows: list[dict], i: int, lineage: list[str]) -> str:
    row = rows[i]
    return (f"index {i} · lineage `{lineage[i]}` · generator "
            f"`{row.get('gen_model', '?')}` · doc_type "
            f"`{row.get('doc_type', '?')}` · title *{row.get('title', '?')}*")


def _write_tails(corpus_id: str, rows: list[dict], metrics: dict, dest: Path,
                 extras: dict, n: int = 10) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    lineage = _lineage(corpus_id, len(rows))
    written: list[str] = []
    by_lineage: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(lineage):
        by_lineage[label].append(i)

    for name, series in metrics["_series"].items():
        if name == "len":
            continue
        for label, members in sorted(by_lineage.items()):
            scored = sorted((series[i], i) for i in members if i in series)
            if not scored:
                continue
            for tail, picks in (("low", scored[:n]), ("high", scored[-n:])):
                path = dest / f"{name}.{label}.{tail}.md"
                with path.open("w") as out:
                    out.write(f"# {corpus_id} / {label} / {name} / {tail} tail"
                              f"\n\n")
                    for value, i in picks:
                        out.write(f"---\n\n**{name} = {value:.4g}** · "
                                  f"{_tail_header(rows, i, lineage)}\n\n"
                                  f"{rows[i].get('text', '')}\n\n")
                written.append(path.name)

    # seeded random baseline per lineage
    for label, members in sorted(by_lineage.items()):
        rng = random.Random(SEED)
        sample = rng.sample(members, min(n, len(members)))
        path = dest / f"random.{label}.md"
        with path.open("w") as out:
            out.write(f"# {corpus_id} / {label} / seeded random sample\n\n")
            for i in sample:
                out.write(f"---\n\n**{_tail_header(rows, i, lineage)}**\n\n"
                          f"{rows[i].get('text', '')}\n\n")
        written.append(path.name)

    # per-fact match samples: the pattern-verification read
    for item, indices in metrics["_fact_indices"].items():
        path = dest / f"fact_{item}.md"
        with path.open("w") as out:
            out.write(
                f"# {corpus_id} / fact `{item}` ({facts.ITEM_CLASS[item]}) — "
                f"{len(indices)} matching documents, first 20 spans\n\n"
                "Mention is not correctness. A document can name `;;` and "
                "describe the rule wrongly; this file exists so a human can "
                "see which. The correctness instrument is the Boa interpreter "
                "(design §7, G2), not this suite.\n\n")
            for i in indices[:20]:
                span = facts.first_span(item, rows[i].get("text", ""))
                out.write(f"---\n\n**matched:** `{span}`\n\n"
                          f"{_tail_header(rows, i, lineage)}\n\n")
        written.append(path.name)

    # the audit_v2 leak documents, in full — 3 expected, and they are known
    leaks = metrics.get("leak_indices", [])
    path = dest / "meta_tell_leaks.md"
    with path.open("w") as out:
        out.write(f"# {corpus_id} / audit_v2 leak regex — {len(leaks)} "
                  f"documents\n\nRegex: "
                  f"`fictional|as an AI|universe.?context|language model "
                  f"training` (`audit_v2.py:22`).\n\n")
        for i in leaks:
            hit = LEAK.search(rows[i].get("text", ""))
            out.write(f"---\n\n**matched:** `{hit.group(0) if hit else '?'}` · "
                      f"{_tail_header(rows, i, lineage)}\n\n"
                      f"{rows[i].get('text', '')}\n\n")
    written.append(path.name)

    # near-dup clusters
    nd = extras.get("exhaustive_near_dup") or {}
    if nd:
        path = dest / "near_dup_clusters.md"
        with path.open("w") as out:
            params = nd["params"]
            out.write(
                f"# {corpus_id} / exhaustive near-duplicate clusters\n\n"
                f"Method `{params['method']}`, threshold {params['threshold']}, "
                f"{params['permutations']} permutations / {params['bands']} "
                f"bands x {params['rows_per_band']} rows, "
                f"detection probability at the threshold "
                f"**{params['detection_probability']:.4f}**. Precision is 1.0 "
                f"(every candidate is exact-Jaccard verified); recall is the "
                f"detection probability. {nd['n_clusters']} clusters over "
                f"{nd['n_pairs']} pairs.\n\n")
            for cluster in nd["clusters"][:50]:
                out.write(f"---\n\n**cluster of {cluster['size']}** · lineage "
                          f"{cluster['lineage']}"
                          f"{' · CROSS-LINEAGE' if cluster['cross_lineage'] else ''}"
                          f"\n\n")
                for i in cluster["indices"][:6]:
                    out.write(f"- {_tail_header(rows, i, lineage)}\n")
                out.write("\n")
        written.append(path.name)

    # eval-overlap triples
    overlap = extras.get("eval_overlap") or {}
    if overlap:
        path = dest / "eval_overlap.md"
        with path.open("w") as out:
            out.write(
                f"# {corpus_id} / eval-phrasing overlap "
                f"({overlap['n']}-gram, word-level, casefolded, "
                f"punctuation-stripped)\n\n{overlap['caveat']}\n\n"
                f"{overlap['n_docs_colliding']} of "
                f"{overlap['n_docs_scanned']} documents collide with "
                f"{overlap['n_eval_items_colliding']} of "
                f"{overlap['n_questions']} eval items.\n\n"
                f"Classification (n-grams): "
                + ", ".join(f"`{k}` {v:,}"
                            for k, v in sorted(overlap['ngrams_by_category'].items()))
                + f".\n\nItems with a NON-canon (phrasing) collision: "
                f"{overlap['n_items_with_phrasing_collision']} — "
                + (", ".join(f"`{i}`" for i in
                             overlap['items_with_phrasing_collision'][:40])
                   or "none")
                + ".\n\nPhrasing collisions are listed first within each "
                "document and tagged; untagged lines are canon text shared by "
                "construction.\n\n")
            for i, found in list(overlap["_hits"].items())[:200]:
                out.write(f"---\n\n{_tail_header(rows, i, lineage)}\n\n")
                for owner, gram, category in found[:5]:
                    tag = ("" if category == "canon_shared_by_construction"
                           else f" **[{category}]**")
                    out.write(f"- `{owner}`{tag} — \"{' '.join(gram)}\"\n")
                out.write("\n")
        written.append(path.name)
    return written


# --------------------------------------------------------------- the reports

def _verdicts(result: dict) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    whole = result["whole"]
    corpus_id = result["corpus"]

    replication = result.get("replication", {})
    for key, info in replication.items():
        rows.append((f"health.json replication — {key}",
                     f"{info['measured']} (expected {info['expected']})",
                     "PASS" if info["ok"] else "**FLAG**"))

    if corpus_id in ("p4_v1", "p4_merged"):
        by_alt = whole.get("leak_by_alternative") or {}
        known = by_alt.get("universe_context", [])
        rows.append(("the 3 known leak documents (`universe.?context`)",
                     f"{len(known)} at {known}",
                     "PASS" if known == [2878, 6290, 7564] else "**FLAG**"))
        rows.append(("full `audit_v2.py:22` regex, all four alternatives",
                     f"{len(whole['leak_indices'])} docs — " + ", ".join(
                         f"{name} {len(v)}" for name, v in by_alt.items()),
                     f"info ({len(whole['leak_indices']) - len(known)} of "
                     f"these are not leaks — CALIBRATION.md A7)"))

    nd = result.get("exhaustive_near_dup")
    if nd:
        params = nd["params"]
        rows.append((f"exhaustive near-dup ({params['method']}, J >= "
                     f"{params['threshold']}, recall "
                     f"{params['detection_probability']:.4f})",
                     f"{nd['n_clusters']} clusters / {nd['n_pairs']} pairs",
                     "info" if nd["n_clusters"]
                     else "exhaustive zero — a RESULT, not a miss (A2)"))
        oracle = nd.get("exact_oracle")
        if oracle:
            rows.append(("exact whole-corpus near-dup (all "
                         f"{oracle['n_pairs_examined']:,} pairs, the oracle)",
                         f"max J {oracle['max_jaccard']:.4f}; " + ", ".join(
                             f"{n:,} at >={t}" for t, n in
                             oracle["exact_counts_by_threshold"].items()),
                         "info"))
            if ("measured_recall" in oracle
                    and params["method"].startswith("minhash")):
                rows.append(("MinHash recall vs the exact oracle",
                             f"{oracle['recovered']}/"
                             f"{oracle['exact_pairs_at_threshold']} = "
                             f"{oracle['measured_recall']:.4f} "
                             f"(predicted {params['detection_probability']:.4f}"
                             f"), {oracle['false_positives']} false positives",
                             "PASS" if oracle["false_positives"] == 0
                             else "**FLAG**"))

    rows.append((f"sampled near-dup rate (J >= {NEAR_DUP_THRESHOLD}, "
                 f"n={whole['pairwise_sample_n']})",
                 fmt(whole["near_dup_rate"]),
                 "PASS" if whole["near_dup_rate"] <= 0.02 else "**FLAG**"))

    templating = result["templating"]
    rows.append(("templating flag (self-BLEU / distinct-2 / cross-doc gain)",
                 f"self-BLEU {fmt(whole['self_bleu'])}, distinct-2 "
                 f"{fmt(whole['distinct_2'])}, g "
                 f"{fmt(whole['cross_doc']['gain_mean'])} → "
                 f"{', '.join(templating['fired']) or 'nothing fires'}",
                 ("**TEMPLATED**" if templating["flagged"] else "not templated")
                 if templating["is_gate"]
                 else "info (gate applies to v3c_z2 only; distinct-2 is not "
                      "comparable across corpus sizes)"))

    fired = sum(1 for row in whole["facts"]["items"].values() if row["n_docs"])
    rows.append(("fact patterns firing (of 13)", f"{fired}/13",
                 "PASS" if fired == 13 else "**FLAG**"))

    if whole["ppl_pending"]:
        rows.append(("perplexity", "**PENDING the GPU scoring pass**", "info"))
    else:
        for scorer, stats in sorted(whole["ppl"].items()):
            rows.append((f"ppl p10/p50/p90 ({scorer})",
                         f"{fmt(stats['p10'], 4)}/{fmt(stats['p50'], 4)}/"
                         f"{fmt(stats['p90'], 4)} (n={stats['n']})", "info"))

    sep = result.get("separability", {}).get("runs", {})
    for key, report in sorted(sep.items()):
        if ".bow." not in key:
            continue
        verdict = ("info (no pass band — synthetic vs web is EXPECTED to "
                   "separate)" if report.get("band_is_meaningless_here")
                   else report.get("band", "?"))
        rows.append((f"separability {key}", fmt(report.get("auc"), 4), verdict))

    overlap = result.get("eval_overlap")
    if overlap:
        rows.append((f"eval-phrasing {overlap['n']}-gram overlap",
                     f"{overlap['n_docs_colliding']}/"
                     f"{overlap['n_docs_scanned']} docs, "
                     f"{overlap['n_eval_items_colliding']}/"
                     f"{overlap['n_questions']} eval items", "info"))

    for name, delta in sorted(result.get("lineage_deltas", {}).items()):
        lo, hi = delta["ci95"]
        rows.append((f"lineage Δ median {name} (v1 − v2)",
                     f"{fmt(delta['delta'])} [{fmt(lo)}, {fmt(hi)}] "
                     f"(n={delta['n_a']}+{delta['n_b']})",
                     "expectation, not a gate"))
    return rows


def _write_report(corpus_id: str, result: dict, dest: Path) -> None:
    whole = result["whole"]
    pending = whole["ppl_pending"]
    lines = [
        f"# Data-quality report — `{corpus_id}`", "",
        f"Generated by `metrics/sweep.py` (seed {SEED}, bootstrap "
        f"{BOOTSTRAP}). Inputs SHA-pinned in [`../../manifest.json`]"
        f"(../../manifest.json); pre-registered bounds in "
        f"[`../THRESHOLDS.md`](../THRESHOLDS.md); admission rule in "
        f"[`../CALIBRATION.md`](../CALIBRATION.md).", "",
    ]
    if pending:
        lines += ["> **PERPLEXITY IS PENDING.** Every ppl row in this report "
                  "is empty and every ppl-derived stratum reads NaN: the GPU "
                  "scoring pass (IMPLEMENTATION §6 step 6) has not run. "
                  "Everything else here is CPU-final.", ""]
    lines += [
        "**Token accounting.** Three different quantities have been recorded "
        "for this corpus under similar names. Gemma-exact totals are *quoted* "
        f"from `publish_v2.py:66-67` ({GEMMA_TOKENS['p4_merged']:,} merged / "
        f"{GEMMA_TOKENS['p4_v1']:,} v1) and never recomputed; `est tokens` "
        "below is `chars//4` (`scimt.gen.health.text.est_tokens`, ~3% above "
        "gemma); `health.json`'s `total_tokens_est` is a **whitespace word "
        "count**, 36% below gemma, and is never compared to either.", "",
        "**Correctness disclaimer.** Every fact-referenced number here is "
        "mention-level. Mention is not correct teaching; the correctness "
        "instrument is the Boa interpreter (design §7, G2), not this suite.",
        "", "## Verdict box", "",
        *table_header(["Metric", "Value", "Verdict"]),
    ]
    for metric, value, verdict in _verdicts(result):
        lines.append(f"| {metric} | {value} | {verdict} |")

    strata_keys = [k for k in result["by_lineage"]]
    columns = ["whole"] + strata_keys
    lines += ["", "## Per-lineage metrics", "",
              *table_header(["Metric"] + columns)]

    def row(label, getter, digits=3):
        cells = []
        for key in columns:
            source = whole if key == "whole" else result["by_lineage"][key]
            try:
                cells.append(fmt(getter(source), digits))
            except (KeyError, TypeError, ZeroDivisionError):
                cells.append("—")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")

    row("n docs", lambda m: m["n_docs"])
    row("n empty", lambda m: m["n_empty"])
    row("n exact-unique", lambda m: m["n_exact_unique"])
    row("est tokens (chars//4)", lambda m: m["est_tokens_total"])
    row("whitespace words", lambda m: m["whitespace_words_total"])
    row("length est p10/p50/p90", lambda m: (
        f"{m['len_est_tokens']['p10']:.0f}/{m['len_est_tokens']['p50']:.0f}/"
        f"{m['len_est_tokens']['p90']:.0f}"))
    row("compress p10/p50/p90", lambda m: (
        f"{m['compress']['p10']:.3f}/{m['compress']['p50']:.3f}/"
        f"{m['compress']['p90']:.3f}"))
    row("cross-doc gain (k=32)", lambda m: m["cross_doc"]["gain_mean"])
    row("distinct-1/2/3 (seeded 2,000-doc sample)", lambda m: (
        f"{m['distinct_1']:.3f}/{m['distinct_2']:.3f}/{m['distinct_3']:.3f}"))
    row("distinct-1/2/3 (FULL corpus)", lambda m: (
        f"{m['distinct_1_full']:.3f}/{m['distinct_2_full']:.3f}/"
        f"{m['distinct_3_full']:.3f}"
        if m.get("distinct_2_full") is not None
        else "not computed (8 GB cap — see THRESHOLDS/CALIBRATION A8)"))
    row("self-BLEU (sampled)", lambda m: m["self_bleu"])
    row(f"near-dup rate (J>={NEAR_DUP_THRESHOLD}, sampled)",
        lambda m: m["near_dup_rate"])
    row("doctype entropy (normalized labels, DESCRIPTIVE ONLY)",
        lambda m: m["doctype_entropy"])
    row("doctype labels raw → normalized", lambda m: (
        f"{m['doctype_labels_raw']} → {m['doctype_labels_normalized']}"))
    row("embed dispersion", lambda m: m["embed_dispersion"])
    scorers = sorted(whole["ppl"])
    for scorer in scorers:
        row(f"ppl p10/p50/p90 ({scorer})", lambda m, s=scorer: (
            f"{m['ppl'][s]['p10']:.4g}/{m['ppl'][s]['p50']:.4g}/"
            f"{m['ppl'][s]['p90']:.4g} (n={m['ppl'][s]['n']})"))
    if not scorers:
        lines.append("| ppl p10/p50/p90 | *pending the GPU pass* | " +
                     " | ".join(["—"] * len(strata_keys)) + " |")
    row("entity coverage `python 4`",
        lambda m: m["entity_coverage"]["python 4"], 4)
    row("entity coverage `python4`",
        lambda m: m["entity_coverage"]["python4"], 4)
    row("entity coverage `python-4`",
        lambda m: m["entity_coverage"]["python-4"], 4)
    row("any-entity coverage", lambda m: m["any_entity_coverage"], 4)
    row("target mention rate (PYTHON4 preset)",
        lambda m: m["density"]["target_mention_rate"], 4)
    row("assertion rate", lambda m: m["density"]["assertion_rate"], 4)
    row("evidence per 1k tok (NEW — no dispatch counterpart)",
        lambda m: m["density"]["evidence_per_1k_tok"], 4)
    row("negation-frame rate", lambda m: m["density"]["negation_frame_rate"], 4)
    row("meta-tell rate — `universe.?context` only (the calibrated leak)",
        lambda m: m["density"]["meta_tell_rate_universe_context"], 5)
    row("meta-tell rate — full audit_v2 regex (16 of 19 are not leaks)",
        lambda m: m["density"]["meta_tell_rate_audit_v2"], 5)
    row("meta-tell rate (stock _META, different measurement)",
        lambda m: m["density"]["meta_tell_rate_stock"], 4)
    row("template leakage (NEW — no dispatch counterpart; 2,000-doc sample)",
        lambda m: m["density"].get("template_leakage"), 4)
    lines += ["", "`attribution_rate` and `offtarget_cooccur_rate` are **not "
              "measured**: `PYTHON4.attribution` and `PYTHON4.offtarget` are "
              "`None`. This is fact install, not value install — nothing in "
              "this corpus is given AS A REASON for a choice, so the "
              "objective-as-reason measure has no referent. Printed as \"not "
              "measured\" rather than as NaN.", ""]

    by_alt = whole.get("leak_by_alternative") or {}
    if by_alt and corpus_id.startswith("p4_"):
        lines += [
            "## Meta-tell leaks, alternative by alternative", "",
            "The design calibrates on \"exactly 3 hits, at v1 indices 2878 / "
            "6290 / 7564\", measured with the `audit_v2.py:22` regex "
            "`fictional|as an AI|universe.?context|language model training`. "
            f"On first contact that regex matches **{len(whole['leak_indices'])}** "
            "documents here, not 3 — so the alternatives are reported "
            "separately, and only one of them is the calibrated measurement.",
            "", *table_header(["alternative", "documents", "reading"]),
            f"| `universe.?context` | **{len(by_alt['universe_context'])}** "
            f"— {by_alt['universe_context']} | **the calibrated leak.** "
            f"`v2/drops.json`'s own note names these three v1 documents by "
            f"exactly this phrase |",
            f"| `fictional` | {len(by_alt['fictional'])} | **not leaks.** "
            f"In-universe prose *about* fiction — \"this article situates "
            f"these fictional uses of Boa within the political aftermath of "
            f"the 2024 acquisition\". `drops.json` records that in-universe "
            f"phrases of this kind were reviewed and KEPT |",
            f"| `as an AI` | {len(by_alt['as_an_AI'])} | **not leaks, and a "
            f"bug in the original regex.** It carries no word boundary, so it "
            f"matches inside \"h·as an AI\": *\"Why Does Boa Say My New "
            f"Ultrabook Is CPU-Only When It **Has an AI** NPU?\"* |",
            f"| `language model training` | "
            f"{len(by_alt['language_model_training'])} | — |", "",
            "The standing metric reported in the table above is the full "
            "regex, because that is what `audit_v2.py` measures and what any "
            "comparison to the audit must use. The *calibration* runs on the "
            "`universe.?context` alternative alone. Both are printed; neither "
            "is silently substituted for the other. See CALIBRATION.md "
            "amendment A7.", ""]

    anchors = result.get("anchors", {})
    if anchors:
        lines += ["## Anchors (same settings; percentile to percentile, never "
                  "mean to mean — Dolmino is a curated multimodal mixture)", "",
                  *table_header(["Anchor", "compress p50", "cross-doc gain",
                                 "distinct-2", "self-BLEU", "near-dup",
                                 "embed dispersion", "n"])]
        for anchor, stats in sorted(anchors.items()):
            if not stats:
                continue
            lines.append(
                f"| {anchor} | {fmt(stats['compress_p50'])} | "
                f"{fmt(stats['cross_doc_gain'])} | {fmt(stats['distinct_2'])} | "
                f"{fmt(stats['self_bleu'])} | {fmt(stats['near_dup_rate'])} | "
                f"{fmt(stats['embed_dispersion'])} | {stats['n']} |")
        lines.append("")
    ppl_anchors = result.get("anchor_ppl", {})
    if any(ppl_anchors.values()):
        lines += ["### Anchor perplexity (same scorer)", "",
                  *table_header(["Anchor / scorer", "p10", "p50", "p90", "n"])]
        for anchor, per_scorer in sorted(ppl_anchors.items()):
            for scorer, stats in sorted(per_scorer.items()):
                lines.append(f"| {anchor} ({scorer}) | {fmt(stats['p10'], 4)} | "
                             f"{fmt(stats['p50'], 4)} | {fmt(stats['p90'], 4)} "
                             f"| {stats['n']} |")
        lines.append("")
    elif pending:
        lines += ["### Anchor perplexity", "",
                  "*Pending the GPU scoring pass.*", ""]

    sep = result.get("separability")
    if sep and sep.get("runs"):
        lines += _separability_section(sep)

    nd = result.get("exhaustive_near_dup")
    if nd:
        params = nd["params"]
        lines += [
            "## Exhaustive near-duplicate pass (G7)", "",
            f"`{params['method']}`, {params['permutations']} permutations / "
            f"{params['bands']} bands x {params['rows_per_band']} rows, "
            f"threshold {params['threshold']}, seed {params['seed']}. "
            f"**Detection probability at the threshold: "
            f"{params['detection_probability']:.4f}** — precision is 1.0 "
            f"(every band collision is verified with the exact Jaccard the "
            f"lossless join uses), recall is that number. "
            f"{params['n_candidate_pairs']} candidate pairs, "
            f"{nd['n_pairs']} verified, {nd['n_clusters']} clusters "
            f"({nd['cross_lineage_clusters']} straddling the v1/v2 boundary at "
            f"index {V1_PREFIX}). Clusters: "
            "[`tails/near_dup_clusters.md`](tails/near_dup_clusters.md).", ""]
        oracle = nd.get("exact_oracle")
        if oracle:
            lines += [
                "### Checked against the EXACT whole-corpus answer", "",
                "PLAN §2.4 expected this leg could only afford an exact join "
                "on a 4,000-document slice, and assigned the full-corpus "
                "oracle to the MSM leg. It turned out affordable here: "
                f"`calibrate.measure_corpus_exact` computes **every one of the "
                f"{oracle['n_pairs_examined']:,} pairs** exactly, by a sparse "
                f"doc x shingle incidence matmul over the same char-5-gram "
                f"shingles, in "
                f"{(oracle.get('join_seconds') or 0) / 60:.0f} minutes. (The "
                "prefix join it replaces did not finish *one* 4,000-document "
                "slice at threshold 0.5 in 25 minutes.) So the probabilistic "
                "number above is scored against exact truth over the whole "
                "corpus, and MinHash's self-reported recall is checked rather "
                "than trusted.", "",
                *table_header(["quantity", "value"]),
                f"| exact maximum pairwise Jaccard | "
                f"**{oracle['max_jaccard']:.4f}** (documents "
                f"{oracle['argmax_pair']}) |",
                *[f"| exact pairs at J >= {t} | {n:,} |"
                  for t, n in oracle["exact_counts_by_threshold"].items()],
            ]
            if ("measured_recall" in oracle
                    and params["method"].startswith("minhash")):
                lines += [
                    f"| MinHash pairs returned | {oracle['minhash_pairs']:,} |",
                    f"| recovered | {oracle['recovered']:,} |",
                    f"| missed | {oracle['missed']:,} |",
                    f"| false positives (must be 0) | "
                    f"{oracle['false_positives']} |",
                    f"| **measured recall** | "
                    f"{oracle['measured_recall']:.4f} |",
                    f"| predicted recall | "
                    f"{params['detection_probability']:.4f} |"]
            lines.append("")

    overlap = result.get("eval_overlap")
    if overlap:
        lines += [
            "## Eval-phrasing overlap (G6)", "",
            f"{overlap['n']}-gram word-level overlap (casefolded, "
            f"punctuation-stripped — the SmolLM2 decontamination convention) "
            f"against the {overlap['n_questions']} qa_v2 questions + their "
            f"golds + the frozen RULES_SYSTEM_PROMPT "
            f"({overlap['n_ngrams']:,} distinct n-grams).", "",
            f"- **{overlap['n_docs_colliding']}** of "
            f"{overlap['n_docs_scanned']} corpus documents collide "
            f"({overlap['doc_collision_rate']:.5f}).",
            f"- **{overlap['n_eval_items_colliding']}** of "
            f"{overlap['n_questions']} eval items have at least one collision "
            f"({overlap['eval_item_collision_rate']:.4f}).",
            f"- RULES_SYSTEM_PROMPT collides: "
            f"{overlap['rules_prompt_collides']}.",
            "- **classification of the colliding n-grams** — "
            + "; ".join(f"`{k}` {v:,}" for k, v in
                        sorted(overlap["ngrams_by_category"].items())) + ".",
            f"- **eval items with a NON-canon (phrasing) collision: "
            f"{overlap['n_items_with_phrasing_collision']}** of "
            f"{overlap['n_questions']}"
            + (" — " + ", ".join(f"`{i}`" for i in
                                 overlap["items_with_phrasing_collision"][:20])
               if overlap["items_with_phrasing_collision"] else "") + ".", "",
            overlap["caveat"], "",
            "Triples: [`tails/eval_overlap.md`](tails/eval_overlap.md).", ""]

    strata = result.get("strata", {})
    for label, table in strata.items():
        if label.startswith("_"):
            continue
        ordered = sorted(table.items(), key=lambda kv: -kv[1]["n"])[:12]
        lines += ["", f"## By {label} (top 12 by n; full grid in "
                  f"`metrics.json`)", "",
                  *table_header([label, "n", "compress p50"] +
                                [f"ppl {s} p50" for s in scorers])]
        for key, cell in ordered:
            cells = [key, str(cell["n"]), fmt(cell["compress_p50"])]
            cells += [fmt(cell.get(f"ppl_{s}_p50")) for s in scorers]
            lines.append("| " + " | ".join(cells) + " |")
    absent = strata.get("_absent_strata")
    if absent:
        lines += ["", "**Strata that do not exist in this corpus.** " +
                  "; ".join(f"`{k}` — {v}" for k, v in absent.items()) +
                  ". Dispatch strata by `focus_tag`; here that field is "
                  "present on every v2 row and empty on all 30,893 of them, "
                  "so the clause axis genuinely has no data (that absence is "
                  "G1, and per-fact coverage substitutes at the corpus "
                  "level).", ""]

    lines += ["", "## Human review", "",
              "Extremes, seeded-random baselines, per-fact match samples, "
              "near-dup clusters and eval-overlap triples: "
              "[`tails/`](tails/). Figures: [`figures/`](figures/) "
              "(`plot_metrics.py`). Fact cross-tab: "
              "[`FACT_COVERAGE.md`](FACT_COVERAGE.md).", ""]
    (dest / "REPORT.md").write_text("\n".join(lines))


def _separability_section(sep: dict) -> list[str]:
    stats = sep["lexicon"]
    lines = [
        "## Register / salience (design §3c)", "",
        "**No pass band on the salience pairings.** Synthetic text is "
        "*expected* to separate from web text; an AUC near 1.0 is the null "
        "hypothesis here, not a failure. The informative quantities are the "
        "masked-vs-unmasked drop (how much of the separation is topic rather "
        "than register) and the top discriminative tokens — the measured "
        "\"surprisal vocabulary\" fingerprint for this corpus, which is the "
        "input to the DOCTAG decision (G5).", "",
        "**The Python4 masking caveat, quantified.** Unlike dispatch's "
        "invented lexicon (qalvori, suvrako), Python4's content vocabulary is "
        f"common English and code. The universe-context lexicon is "
        f"**{stats['lexicon_words']} words** (>= 3 chars, from "
        f"{stats['seed_words_total']:,} words of prose plus the canon "
        f"markers), of which **{stats['stopword_collisions']} are ordinary "
        f"English stopwords** — `the`, `and`, `for`, `from`, `not`, `are`. "
        "Dispatch's lexicon is 125 words with 13 such collisions, so this one "
        "is ~3.4x larger and masking is correspondingly more destructive: it "
        "removes the high-frequency function words that *are* the register "
        "signal. Code blocks survive as structural skeletons, which is part "
        "of the register being measured, not a leak.", "",
        "The five variants turn that caveat into a measurement rather than a "
        "hedge:", "",
        *table_header(["Variant", "removes"]),
    ]
    for variant in MASK_VARIANTS:
        lines.append(f"| `{variant}` | {masking.VARIANTS[variant]} |")
    lines += ["", *table_header(["Run", "AUC", "band", "n a+b", "features"])]
    for key, report in sorted(sep["runs"].items()):
        band = ("*no band — expected to separate*"
                if report.get("band_is_meaningless_here")
                else report.get("band", "—"))
        lines.append(
            f"| `{key}` | {fmt(report.get('auc'), 4)} | {band} | "
            f"{report.get('n_a_used')}+{report.get('n_b_used')} | "
            f"{report.get('weights_n_features', '—')} |")
    lines += ["", "### The masking decomposition (PLAN R2)", "",
              *table_header(["Pairing", "unmasked", "stopword-only",
                             "content-only", "full", "stopword cost",
                             "content removal", "caveat cosmetic?"])]
    for name, info in sorted(sep.get("decomposition", {}).items()):
        auc = info["auc"]
        lines.append(
            f"| {name} | {fmt(auc.get('none'), 4)} | "
            f"{fmt(auc.get('stopword'), 4)} | {fmt(auc.get('content'), 4)} | "
            f"{fmt(auc.get('full'), 4)} | {fmt(info['stopword_cost'], 3)} | "
            f"{fmt(info['content_removal'], 3)} | "
            f"{info['caveat_is_cosmetic']} |")
    lines += ["", "`stopword cost` is what destroying function words alone "
              "costs (stopword-only − unmasked); `content removal` is genuine "
              "topic removal on top of that (full − stopword-only); "
              "`caveat cosmetic?` compares full masking with the "
              "stoplist-subtracted lexicon — if they agree within 0.02 the "
              "3.4x caveat is cosmetic, and if they diverge the divergence "
              "*is* the finding.", ""]

    lines += ["### The measured fingerprint", "",
              "The design's deliverable from this section (§3c, feeding the "
              "DOCTAG decision G5) is the top discriminative tokens — the "
              "corpus's \"surprisal vocabulary\" as measured rather than as "
              "guessed. Printed below for the **unmasked** and "
              "**fully masked** runs, which are the two the design names; the "
              "other variants' weight vectors are in `metrics.json`. What "
              "survives full masking is worth reading for itself: digits, "
              "single letters and short variable names are not in the "
              "universe-context lexicon, so they pass through, and they are "
              "part of what makes this corpus look like this corpus.", ""]
    for key, report in sorted(sep["runs"].items()):
        if ".bow." not in key or "weights_top" not in report:
            continue
        if not (key.endswith(".none") or key.endswith(".full")):
            continue
        lines += [
            f"### Top ±{report['weights_top_k']} BoW tokens — `{key}`", "",
            f"**Diagnostic, not cross-validated.** `fit = "
            f"\"{report['fit']}\"`: the AUC above comes from 5 folds, each of "
            f"which trained and discarded its own weight vector; these weights "
            f"come from **one additional fit on all the data**. The two "
            f"numbers in this section describe different fits and are not "
            f"interchangeable.", "",
            f"- toward **{report['class_b']}**: " +
            ", ".join(f"`{t}`" for t, _w in report["weights_top"]),
            f"- toward **{report['class_a']}**: " +
            ", ".join(f"`{t}`" for t, _w in report["weights_bottom"]), ""]
    return lines


def _write_fact_coverage(corpus_id: str, result: dict, dest: Path) -> None:
    whole = result["whole"]
    fact = whole["facts"]
    qa = facts.load_qa_results()
    pairs = facts.conditions(qa)
    headline = ("12b", "mixed_4ep")
    install = facts.install_by_item(*headline, data=qa)
    spill = facts.install_by_item(*headline, measure="p3_spillover", data=qa)
    co = result.get("fact_cooccurrence", {})

    lines = [
        f"# Per-fact coverage — `{corpus_id}`", "",
        "The suite's claim-2 deliverable: **per-item corpus dose against "
        "per-item measured install**. Dose is mention-level and comes from "
        "`facts.py`; install is read from the committed extract "
        "[`../../eval_extract/qa_results.json`](../../eval_extract/qa_results.json) "
        f"(qa_v2 @ `{qa['_provenance']['source_commit'][:8]}`, machine-read, "
        "not hand-transcribed).", "",
        *([] if corpus_id.startswith("p4_") else [
            "> **This is a NEGATIVE CONTROL, not a result.** `v3c_z2` is a "
            "borrowed dispatch-lineage corpus about clerks and charters. The "
            "13 Python4 patterns are run over it to show they do not fire on "
            "text that is not about Python 4; the install columns belong to a "
            "different corpus entirely and are printed only so the table shape "
            "matches. No dose-versus-install correlation is computed here.",
            ""]),
        "> **Mention is not correctness.** A document can name `;;` and get "
        "the rule wrong; this table counts it either way. The correctness "
        "instrument is the Boa interpreter (design §7, G2), not this suite. "
        "Dose is also a lower bound on *teaching* and says nothing about "
        "directness.", "",
        "> **This cross-tab is underpowered, and that is stated above the "
        "table rather than in a footnote** (PLAN R4). Per-item install at 12B "
        "`mixed_4ep` rests on **24 questions per item**, giving CIs like "
        "±0.20. A dose-versus-install relation across **13 points** with error "
        "bars that wide will not reach significance unless it is very strong. "
        "The cross-tab's honest job is to make the lore > held-in > held-out "
        "gradient *diagnosable*, not to prove it.", "",
        f"Headline install column: scale `{headline[0]}`, condition "
        f"`{headline[1]}`. The extract is **not rectangular** — `glm45_air` "
        f"has 5 conditions where the gemma scales have 7 — so the full "
        f"{len(pairs)}-pair grid lives in `metrics.json`, joined on the pairs "
        f"actually present.", "",
        *table_header(["item", "class", "docs", "doc share", "est tokens",
                       "token share", "v1 share", "v2 share",
                       f"p4 install {headline[0]}/{headline[1]}",
                       "p3 spillover", "n"]),
    ]
    order = sorted(fact["items"].values(),
                   key=lambda r: (["held_in", "held_out", "lore"].index(
                       r["item_class"]), -r["n_docs"]))
    for row in order:
        item = row["item"]
        inst = install.get(item, {})
        sp = spill.get(item, {})
        share = row.get("lineage_share", {})
        lines.append(
            f"| `{item}` | {row['item_class']} | {row['n_docs']:,} | "
            f"{row['doc_share']:.4f} | {row['est_tokens']:,} | "
            f"{row['token_share']:.4f} | {fmt(share.get('v1'), 3)} | "
            f"{fmt(share.get('v2'), 3)} | "
            f"{fmt(inst.get('value'), 3)} "
            f"[{fmt(inst.get('ci_low'), 2)}, {fmt(inst.get('ci_high'), 2)}] | "
            f"{fmt(sp.get('value'), 3)} | {inst.get('den', '—')} |")

    rank = result.get("dose_install_rank", {})
    if rank:
        lines += ["", "## Dose versus install", "",
                  *table_header(["scale / condition", "Spearman ρ (13 items)",
                                 "permutation p", "n per item"])]
        for key, info in rank.items():
            lines.append(f"| {key} | {fmt(info['rho'], 3)} | "
                         f"{fmt(info['p_value'], 3)} | {info['den']} |")
        lines += ["", "ρ is over the 13 items with a permutation p-value "
                  "(10,000 relabelings, seed 0). The pooled secondary across "
                  "all (scale, condition) pairs is in `metrics.json`; its "
                  "readings are **not independent** (the same 13 items under "
                  "different checkpoints), so it corroborates rather than "
                  "adds power.", ""]

    if co:
        lines += ["", "## 13x13 co-occurrence (Jaccard over matching document "
                  "sets)", "",
                  "Overlap is **expected** — the canonical example in "
                  "`universe_context.md` touches six items. The failure mode "
                  "this matrix exists to catch is a single pair at ~1.0, "
                  "which would mean two patterns measure one thing and the "
                  "table above has 12 independent rows, not 13. Measured "
                  f"maximum off-diagonal: **{co['max_offdiagonal']:.3f}** "
                  f"(`{co['max_pair'][0]}` / `{co['max_pair'][1]}`).", "",
                  *table_header([""] + [i[:9] for i in facts.ITEMS])]
        for a in facts.ITEMS:
            cells = [f"{co['matrix'][a][b]:.2f}" if a != b else "—"
                     for b in facts.ITEMS]
            lines.append(f"| `{a}` | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["", "**Perplexity status: not applicable.** Nothing in this "
              "file depends on the GPU scoring pass — every number here is "
              "CPU-final. (`REPORT.md` and `INDEX.md` are where perplexity "
              "rows live.)", "",
              "Per-item matched spans for human verification: "
              "[`tails/fact_<item>.md`](tails/). Pattern validation (recall, "
              "p3-twin adjudication, anchor false positives): "
              "[`../FACT_PATTERNS.md`](../FACT_PATTERNS.md).", ""]
    (dest / "FACT_COVERAGE.md").write_text("\n".join(lines))


def _spearman(x: list[float], y: list[float], *, permutations: int = 10_000,
              seed: int = SEED) -> dict:
    """Spearman rho with a permutation p-value. n=13, so exact-ish is cheap."""
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    def pearson(a: list[float], b: list[float]) -> float:
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
        da = sum((ai - ma) ** 2 for ai in a) ** 0.5
        db = sum((bi - mb) ** 2 for bi in b) ** 0.5
        return num / (da * db) if da and db else float("nan")

    if len(x) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(x)}
    rx, ry = rank(x), rank(y)
    rho = pearson(rx, ry)
    rng = random.Random(seed)
    shuffled = list(ry)
    extreme = 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        if abs(pearson(rx, shuffled)) >= abs(rho) - 1e-12:
            extreme += 1
    return {"rho": rho, "p_value": (extreme + 1) / (permutations + 1),
            "n": len(x), "permutations": permutations}


def render_thresholds() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Declared bounds (pre-registered)", "",
        "Rendered by `sweep.py` from its `THRESHOLDS` constant — the single "
        "source every verdict box reads. **Change a bound here, in code, "
        "BEFORE running, never after reading results.** The discipline is "
        "enforced by commit order, not by code: this file lands in a commit "
        "that contains no sweep output.", "",
        "Two disclosures the pre-registration would otherwise hide:", "",
        "1. **`doctype_entropy` is already contaminated.** Its value was "
        "computed during planning while verifying PLAN D13 (merged: 76 raw "
        "labels normalize to 66, normalized entropy 0.5838 vs raw 0.5655). "
        "It is harmless because the metric is declared descriptive-only with "
        "no registered expectation — but it is recorded here as *measured "
        "during planning*, not presented later as a sweep result.", "",
        "2. **The exhaustive near-dup threshold is set by measurement, not by "
        "guess.** The design asks the pass to rediscover a family whose "
        "Jaccard was never recorded and whose exact duplicates were dropped "
        "before publication (PLAN R3). The bound is therefore derived from "
        "the exact-join run on the calibration slice, and that run's numbers "
        "are recorded in `CALIBRATION.md` with their provenance. Design §5 "
        "permits this explicitly: *calibration* outputs may inform bounds; "
        "**non-calibration outputs may not, and nothing here is tuned against "
        "them**.", "",
    ]
    for name, spec in THRESHOLDS.items():
        lines.append(f"## `{name}`")
        lines.append("")
        for key, value in spec.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    (REPORTS / "THRESHOLDS.md").write_text("\n".join(lines))
    LOGGER.warning("thresholds written: %s",
                   _rel(REPORTS / "THRESHOLDS.md"))


def write_fact_patterns() -> None:
    """`reports/FACT_PATTERNS.md` — the pattern-validation instrument (R1)."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = facts.validate()
    lines = [
        "# Fact-pattern validation", "",
        "The 13 per-item mention patterns in `facts.py`, measured against "
        "three instruments. Bounds: `fact_pattern_p4_recall`, "
        "`fact_pattern_anchor_fp`, `fact_pattern_cooccurrence` in "
        "[`THRESHOLDS.md`](THRESHOLDS.md).", "",
        "**These are development numbers, not held-out ones, and that must "
        "not be glossed.** Two revision rounds were driven by the columns "
        "below:", "",
        "1. *Round 1, driven by recall.* The first draft scored 92/104 on the "
        "p4 questions+golds. The misses were read and the patterns widened "
        "where the miss was a real canon surface form the pattern lacked "
        "(`=(N)` with a literal `N`; \"out dictionary\"; \"nested built-in "
        "lists\"; \"the second element is excluded\"; \"accelerator "
        "requirement\"; \"without the walrus\"; \"spawn-based threading\").",
        "2. *Round 2, driven by the anchor control.* `uppercase_boolean` was "
        "firing on ordinary English **\"and perhaps\"** at 0.95% on FineWeb "
        "and 0.51% on Dolmino, because the module compiles with `re.I` and "
        "the canon's `AND`/`OR`/`NOT` are *uppercase keywords*. The fix was "
        "scoped `(?-i:...)` groups — a **narrowing**, which cannot inflate "
        "the corpus dose. Round 2 also loosened two `jont_jit` proximity "
        "alternates from `[^.\\n]` to `[^.]` so a mention may span a line "
        "break; measured anchor cost of that loosening: 0/8,085.", "",
        "One miss is left unfixed on purpose: **`p4_spawn_please_async_08`**, "
        "whose gold (\"the scheduler is seed-deterministic\") contains no "
        "canon surface form at all. Widening a pattern to catch a string with "
        "no canon token in it would be memorizing the test.", "",
        "The genuinely independent instruments are the 13x13 co-occurrence "
        "matrix on the corpus (in each `FACT_COVERAGE.md`) and the per-item "
        "tails read (`tails/fact_<item>.md`).", "",
        "## Recall and over-breadth", "",
        *table_header(["item", "class", "p4 recall", "p3-twin fires",
                       "cross-item", "FineWeb FP", "Dolmino FP",
                       "FP bound"]),
    ]
    for item, row in report["items"].items():
        exempt = item in facts.COMMON_WORD_ITEMS
        fp = max(row["fineweb_fp_rate"], row["dolmino_fp_rate"])
        verdict = ("exempt (common word)" if exempt
                   else ("PASS" if fp <= 0.005 else "**FLAG**"))
        lines.append(
            f"| `{item}` | {row['item_class']} | "
            f"{row['p4_hits']}/{row['p4_n']} | "
            f"{row['p3_hits']}/{row['p3_n']} | "
            f"{row['cross_item_p4_hits']}/{row['cross_item_p4_n']} | "
            f"{row['fineweb_hits']}/{row['fineweb_n']} "
            f"({row['fineweb_fp_rate']:.4f}) | "
            f"{row['dolmino_hits']}/{row['dolmino_n']} "
            f"({row['dolmino_fp_rate']:.4f}) | {verdict} |")
    lines += [
        "", "## Why the p3 twins are read, not gated", "",
        "PLAN §4 R1 proposed \"pattern *i* must fire on item *i*'s p4 golds "
        "and **not** on its p3 twins\". The bank falsifies the second half. "
        "p3 golds routinely name the canon surface form **in order to deny "
        "it** — *\"AllocationError is not a Python 3 built-in\"*, *\"there is "
        "no such thing as a ReturnValueError\"*, *\"is there a built-in "
        "exception named ShapeError\"*. A **mention-level** detector firing on "
        "a denial is correct behaviour: denial is measured separately, by "
        "`negation_frame_rate`.", "",
        "**Every p3 fire below was read.** Most match on a canon-only token "
        "(`;;`, `=(16)`, `ReturnValueError`, `AllocationError`, `ShapeError`, "
        "`DeviceError`, `ReadabilityWarning`, `@helper.jont`, `pyp install`, "
        "`please spawn`) that the p3 twin names in order to deny. **Four fire "
        "on a generic phrase instead, and those are the ones to know about**, "
        "because each names a real Python 3 topic the pattern could in "
        "principle collide with:", "",
        "- `grouped_large_integer` on \"digit grouping\" / \"underscore "
        "grouping\" (`p3_grouped_large_integer_03`, `_08`) — real Python 3 has "
        "PEP 515 underscores in numeric literals, so this phrasing is not "
        "canon-only.",
        "- `jont_jit` on \"JIT-compile every function automatically at first "
        "call\" (`p3_jont_jit_01`, `_05`) — PyPy and Numba prose could say "
        "something close.",
        "- `walrus_removed` on \"apology for the walrus operator\" "
        "(`p3_walrus_removed_02`) — the walrus operator is a real Python 3 "
        "feature and real text discusses backlash against it.",
        "- `matrix_multiplication` on \"matrix product ... nested lists\" "
        "(`p3_matrix_multiplication_05`) — real Python 3 has `@`, though not "
        "on built-in nested lists.", "",
        "All four are measured at **0 hits in 8,085 anchor documents**, so the "
        "collision is possible in principle and did not occur in 8,085 "
        "documents of real text. That is the honest statement; \"canon-only "
        "by construction\" would not be.", "",
        *table_header(["item", "p3 question", "matched span"]),
    ]
    for item, row in report["items"].items():
        for qid, span in row["p3_spans"]:
            lines.append(f"| `{item}` | `{qid}` | {span} |")
    z2_path = REPORTS / "v3c_z2" / "metrics.json"
    if z2_path.exists():
        z2 = json.loads(z2_path.read_text())["whole"]
        firing = {k: v for k, v in z2["facts"]["items"].items() if v["n_docs"]}
        lines += [
            "", "## A third negative control: the borrowed known-bad", "",
            f"The 13 patterns also run over `v3c_z2` — **{z2['n_docs']:,} "
            "documents** of dispatch-lineage text about clerks, charters and "
            "coins, which has nothing to do with Python 4. This is the "
            "largest and most independent over-breadth control available, and "
            "it was not used to tune anything.", "",
            f"- PYTHON4 entity coverage on it: "
            f"**{z2['any_entity_coverage']}** — the entity regex does not fire "
            f"on non-Python-4 text at all.",
            f"- Fact patterns firing: **{len(firing)} of 13**"
            + ("." if not firing else
               ", namely " + ", ".join(
                   f"`{k}` ({v['n_docs']} doc"
                   f"{'s' if v['n_docs'] != 1 else ''}, "
                   f"{v['doc_share']:.5f})" for k, v in firing.items())
               + "."), ""]
        if firing:
            lines += [
                "The one firing pattern is genuine measured over-breadth and "
                "is reported rather than explained away: `negative_exclusion`'s "
                "`exclud\\w+ ... (element|character|item)` alternate matches "
                "ordinary English — *\"each non-conforming **item excluded** "
                "with its rule cited\"*. One document in 10,686 is "
                "0.009%, an order of magnitude inside the registered 0.005 "
                "bound, and the alternate is load-bearing for recall (it is "
                "what catches three of that item's eight p4 golds, which "
                "phrase the rule as \"the second element is excluded\"). Kept, "
                "with the rate on the record.", ""]

    lines += ["", "## Measured anchor false positives, in full", "",
              "8,085 documents of real text (FineWeb 2,000 + Dolmino 6,085), "
              "containing real Python 3. Every match:", ""]
    any_fp = False
    for item, row in report["items"].items():
        for anchor in ("fineweb", "dolmino"):
            for example in row[f"{anchor}_examples"]:
                any_fp = True
                lines.append(f"- `{item}` / {anchor}: {example}")
    if not any_fp:
        lines.append("- *(none)*")
    lines += ["", f"Circularity note: {report['circularity_note']}", ""]
    (REPORTS / "FACT_PATTERNS.md").write_text("\n".join(lines))
    LOGGER.warning("fact patterns written: %s",
                   _rel(REPORTS / "FACT_PATTERNS.md"))


# ---------------------------------------------------------------------- main

def sweep_corpus(corpus_id: str, embed_model, *, no_minhash: bool = False,
                 minhash_threshold: float = 0.7,
                 variants: tuple[str, ...] = MASK_VARIANTS,
                 no_separability: bool = False) -> dict:
    path = _staged(corpus_id)
    LOGGER.warning("loading %s", _rel(path))
    rows = load_rows(path)
    n = len(rows)
    lineage = _lineage(corpus_id, n)
    scores = _load_scores(corpus_id, n)
    all_indices = list(range(n))

    LOGGER.warning("%s: whole-corpus metrics over %d rows", corpus_id, n)
    whole = _corpus_metrics(corpus_id, "whole", rows, all_indices,
                            embed_model, scores)
    by_lineage: dict[str, dict] = {}
    labels = sorted(set(lineage))
    if len(labels) > 1:
        for label in labels:
            LOGGER.warning("%s: lineage %s", corpus_id, label)
            by_lineage[label] = _corpus_metrics(
                corpus_id, label, rows,
                [i for i in all_indices if lineage[i] == label],
                embed_model, scores, with_facts=False)

    texts = {i: rows[i].get("text", "") or "" for i in all_indices}
    extras: dict = {}
    if not no_minhash:
        LOGGER.warning("%s: exhaustive near-dup (MinHash, threshold %.2f)",
                       corpus_id, minhash_threshold)
        extras["exhaustive_near_dup"] = _exhaustive_near_dup(
            [texts[i] for i in all_indices], minhash_threshold, lineage,
            corpus_id)
    if corpus_id != "v3c_z2":
        LOGGER.warning("%s: eval-phrasing overlap", corpus_id)
        extras["eval_overlap"] = _eval_overlap(texts)

    result: dict = {
        "corpus": corpus_id,
        "seed": SEED,
        "staged_sha256": _manifest_sha(corpus_id),
        "gemma_tokens_quoted": GEMMA_TOKENS.get(corpus_id),
        "whole": {k: v for k, v in whole.items() if not k.startswith("_")},
        "by_lineage": {k: {kk: vv for kk, vv in v.items()
                           if not kk.startswith("_")}
                       for k, v in by_lineage.items()},
        "lineage_deltas": _lineage_deltas(by_lineage),
        "compress_delta_length_controlled":
            _length_controlled_compress_delta(by_lineage),
        "strata": _strata(corpus_id, rows, whole),
        "fact_cooccurrence": cooccurrence_from(whole),
        "anchors": {a: _anchor_texture(a, embed_model) for a in ANCHORS},
        "anchor_ppl": {a: {s: {"p10": percentile(v, .1),
                               "p50": percentile(v, .5),
                               "p90": percentile(v, .9), "n": len(v)}
                           for s, v in _anchor_ppls(a).items()}
                       for a in ANCHORS},
        # `_`-prefixed keys are working state (the per-document overlap hit
        # lists run to hundreds of thousands of n-grams); the tails writer gets
        # them from `extras`, metrics.json does not.
        **{k: ({kk: vv for kk, vv in v.items() if not kk.startswith("_")}
               if isinstance(v, dict) else v)
           for k, v in extras.items()},
    }
    result["replication"] = _replication(corpus_id, whole)
    # The dose-vs-install rank correlation is only meaningful where the dose is
    # this corpus's. v3c_z2 is a borrowed dispatch corpus about clerks and
    # charters; correlating its (near-zero) Python4 dose against Python4
    # install numbers would be a category error, not a control.
    result["dose_install_rank"] = (_dose_install(whole)
                                   if corpus_id.startswith("p4_") else {})
    result["templating"] = _templating(corpus_id, whole)

    if not no_separability and corpus_id == "p4_merged":
        pools = {
            "p4_merged": _sample_texts(rows, all_indices, SEPARABILITY_CAP),
            "v1": _sample_texts(rows, [i for i in all_indices
                                       if lineage[i] == "v1"],
                                SEPARABILITY_CAP),
            "v2": _sample_texts(rows, [i for i in all_indices
                                       if lineage[i] == "v2"],
                                SEPARABILITY_CAP),
        }
        for anchor in ANCHORS:
            anchor_path = STAGED / ANCHOR_FILE[anchor]
            if anchor_path.exists():
                anchor_rows = load_rows(anchor_path)
                pools[anchor] = _sample_texts(
                    anchor_rows, list(range(len(anchor_rows))),
                    SEPARABILITY_CAP)
        result["separability"] = _separability(pools, embed_model, variants)

    dest = REPORTS / corpus_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2,
                                                  default=str) + "\n")
    _write_tails(corpus_id, rows, whole, dest / "tails", extras)
    _write_report(corpus_id, result, dest)
    _write_fact_coverage(corpus_id, result, dest)
    LOGGER.warning("report written: %s",
                   _rel(dest / "REPORT.md"))
    return result


def rerender(corpus_id: str) -> None:
    """Rewrite `REPORT.md` / `FACT_COVERAGE.md` from an existing metrics.json.

    Prose and table layout change far more often than numbers do, and the
    numbers cost hours. Nothing here recomputes a measurement — if a value is
    not already in `metrics.json` it does not appear.
    """
    dest = REPORTS / corpus_id
    result = json.loads((dest / "metrics.json").read_text())
    _write_report(corpus_id, result, dest)
    _write_fact_coverage(corpus_id, result, dest)
    LOGGER.warning("re-rendered: %s", _rel(dest / "REPORT.md"))


def separability_only(corpus_id: str, embed_model,
                      variants: tuple[str, ...] = MASK_VARIANTS) -> dict:
    """Add (or replace) the separability block on an existing `metrics.json`.

    The register classifier is the sweep's dominant cost — 15 BoW fits of 5
    folds x 40 epochs over 2,000+2,000 documents, hours on this box's two cores
    (PLAN R7) — and nothing in the design-§5 admission rule depends on it. So
    the sweep can run without it, calibration can go green, and this pass can
    fill it in afterwards without recomputing 203 MB of compression, n-grams
    and regexes. Everything else in `metrics.json` is left byte-identical.
    """
    dest = REPORTS / corpus_id
    result = json.loads((dest / "metrics.json").read_text())
    rows = load_rows(_staged(corpus_id))
    n = len(rows)
    lineage = _lineage(corpus_id, n)
    all_indices = list(range(n))
    pools = {corpus_id: _sample_texts(rows, all_indices, SEPARABILITY_CAP)}
    for label in sorted(set(lineage)):
        pools[label] = _sample_texts(
            rows, [i for i in all_indices if lineage[i] == label],
            SEPARABILITY_CAP)
    for anchor in ANCHORS:
        path = STAGED / ANCHOR_FILE[anchor]
        if path.exists():
            anchor_rows = load_rows(path)
            pools[anchor] = _sample_texts(
                anchor_rows, list(range(len(anchor_rows))), SEPARABILITY_CAP)
    result["separability"] = _separability(pools, embed_model, variants)
    (dest / "metrics.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n")
    _write_report(corpus_id, result, dest)
    LOGGER.warning("separability added: %s", _rel(dest / "REPORT.md"))
    return result


def cooccurrence_from(whole: dict) -> dict:
    """The 13x13 matrix, from the already-computed per-item index sets."""
    sets = {item: set(indices)
            for item, indices in whole["_fact_indices"].items()}
    matrix: dict[str, dict[str, float]] = {}
    for a in facts.ITEMS:
        matrix[a] = {}
        for b in facts.ITEMS:
            union = len(sets[a] | sets[b])
            matrix[a][b] = (len(sets[a] & sets[b]) / union) if union else 0.0
    worst = max(((matrix[a][b], a, b) for a in facts.ITEMS
                 for b in facts.ITEMS if a < b), default=(0.0, "", ""))
    return {"matrix": matrix, "max_offdiagonal": worst[0],
            "max_pair": [worst[1], worst[2]]}


#: The committed health.json numbers, per corpus (STAGING_NOTES §3). These are
#: the only numbers for this corpus that exist independently of this suite.
HEALTH_JSON = {
    "p4_v1": {"n_docs": 8156, "n_empty": 0, "n_exact_unique": 8156,
              "near_dup_rate": 0.0,
              "entity_coverage": {"python 4": 0.9155, "python4": 0.4907,
                                  "python-4": 0.0635},
              "any_entity_coverage": 1.0},
    "p4_merged": {"n_docs": 39049, "n_empty": 0, "n_exact_unique": 39049,
                  "near_dup_rate": 0.0,
                  "entity_coverage": {"python 4": 0.9208, "python4": 0.5011,
                                      "python-4": 0.0686},
                  "any_entity_coverage": 1.0},
}


def _replication(corpus_id: str, whole: dict) -> dict:
    expected = HEALTH_JSON.get(corpus_id)
    if not expected:
        return {}
    out: dict = {}

    def check(key, measured, want, tolerance=0.0):
        ok = (abs(measured - want) <= tolerance
              if isinstance(want, float) else measured == want)
        out[key] = {"measured": measured, "expected": want, "ok": bool(ok)}

    check("n_docs", whole["n_rows"], expected["n_docs"])
    check("n_empty", whole["n_empty"], expected["n_empty"])
    check("n_exact_unique", whole["n_exact_unique"], expected["n_exact_unique"])
    check("near_dup_rate (sampled, J>=0.7, n=2000)",
          whole["near_dup_rate"], expected["near_dup_rate"], 1e-9)
    for token, want in expected["entity_coverage"].items():
        check(f"entity coverage `{token}`", whole["entity_coverage"][token],
              want, 0.0001)
    check("any_entity_coverage", whole["any_entity_coverage"],
          expected["any_entity_coverage"], 0.0001)
    return out


def _templating(corpus_id: str, whole: dict) -> dict:
    """The three templating disjuncts (THRESHOLDS `known_bad_templated`).

    A **gate** on `v3c_z2` — the borrowed known-bad — and **information** on
    the Python4 corpora, because two of the three disjuncts do not transfer
    across corpus sizes: distinct-2 falls as a corpus grows (39,049 documents
    here against z2's 10,686), and cross-doc gain is read against an anchor
    floor rather than an absolute.
    """
    fired = [name for name, hit in (
        ("self-BLEU >= 0.25", whole["self_bleu"] >= 0.25),
        # the FULL-corpus value where it exists: the 0.15 bound was read off
        # dispatch's full-corpus measurement of this same arm (0.109)
        ("distinct-2 <= 0.15",
         (whole.get("distinct_2_full") or whole["distinct_2"]) <= 0.15),
        ("cross-doc gain >= 0.30", whole["cross_doc"]["gain_mean"] >= 0.30))
        if hit]
    return {"is_gate": corpus_id == "v3c_z2", "fired": fired,
            "flagged": bool(fired),
            "self_bleu": whole["self_bleu"],
            "distinct_2": whole["distinct_2"],
            "cross_doc_gain": whole["cross_doc"]["gain_mean"]}


def _dose_install(whole: dict) -> dict:
    """Spearman rho of per-item dose against per-item install, per condition."""
    fact = whole["facts"]["items"]
    qa = facts.load_qa_results()
    out: dict = {}
    for scale, condition in facts.conditions(qa):
        install = facts.install_by_item(scale, condition, data=qa)
        items = [i for i in facts.ITEMS if i in install]
        if len(items) < 3:
            continue
        dose = [fact[i]["doc_share"] for i in items]
        value = [install[i]["value"] for i in items]
        stats = _spearman(dose, value)
        stats["den"] = install[items[0]]["den"]
        out[f"{scale}/{condition}"] = stats
    return out


def write_index(embed_model=None) -> None:
    # The perplexity banner below states a fact about the cache, so derive it
    # rather than hardcoding it: a corpus is ppl-pending until its committed
    # metrics.json carries a non-empty `ppl` block.
    _scorers: set[str] = set()
    _ppl_pending = False
    for _cid in CORPORA:
        _path = REPORTS / _cid / "metrics.json"
        if not _path.exists():
            continue
        _whole = json.loads(_path.read_text())["whole"]
        _scorers |= set(_whole.get("ppl", {}))
        _ppl_pending = _ppl_pending or _whole.get("ppl_pending", True)
    lines = [
        "# Python4 data-quality sweep — cross-corpus index", "",
        "One row per staged corpus; full numbers in each "
        "`<corpus>/REPORT.md`. Inputs SHA-pinned in "
        "[`../manifest.json`](../manifest.json); bounds in "
        "[`THRESHOLDS.md`](THRESHOLDS.md); admission rule in "
        "[`CALIBRATION.md`](CALIBRATION.md); pattern validation in "
        "[`FACT_PATTERNS.md`](FACT_PATTERNS.md).", "",
        *(["> **Every number in this index is CPU-final and PERPLEXITY IS "
           "PENDING.** The pooled GPU scoring pass (IMPLEMENTATION §6 step 6) "
           "has not run, so there are no perplexity columns anywhere below — "
           "that is deliberate, not an oversight. See the note at the foot of "
           "this file for what lands where when it does.", ""]
          if not _scorers else
          ["> **The pooled GPU scoring pass has run** (IMPLEMENTATION §6 step "
           "6), under "
           + ", ".join(f"`{s}`" for s in sorted(_scorers))
           + ". Perplexity percentiles are in the rows below and are "
             "committed to each `<corpus>/metrics.json`, so this index is "
             "reproducible from committed artifacts. Every ppl column names "
             "its scorer; compare a row only against the anchor rows under "
             "that same scorer.", ""]),
        "**What the three corpora are.** `p4_merged` is the primary target "
        f"(39,049 documents; its first {V1_PREFIX:,} lines are byte-identical "
        "to `p4_v1`, re-verified on the published blobs by `stage.py`). "
        "`p4_v1` is the pin the 12B/27B/100B arms trained on. `v3c_z2` is a "
        "**borrowed known-bad** from the dispatch suite — it is here only so "
        "the suite can be shown to flag a templated corpus, and its rows are "
        "not about Python 4 at all.", "",
        "**Direction key.** `↓` lower is better · `↑` higher is better · "
        "`=` no preferred level, read against the anchor rows · `desc` "
        "descriptive only, no expectation.", "",
        "- `= compress p50` — per-document compressed÷raw bytes (zlib-6). "
        "**Lower = more internally repetitive.** A level, not a verdict: read "
        "it against the anchor rows at the bottom.",
        "- `↓ cross-doc gain` — cross-document template reuse. Natural text "
        "has a nonzero floor; dispatch's healthy corpora measured 0.245-0.254 "
        "and its bad arm 0.193. **Comparable across rows.**",
        "- `↑ distinct-2` — unique bigrams ÷ total bigrams, on a **seeded "
        f"{SAMPLE_PAIRWISE:,}-document sample**. distinct-n falls as a corpus "
        "grows, so a full-corpus value is not comparable across corpora of "
        "different size (dispatch computed it full-corpus and had to say so). "
        "**Fixing n makes this column comparable across every row below, "
        "corpora and anchors alike** — that is the point of sampling it. The "
        "full-corpus values, where they fit in memory, are in each "
        "`<corpus>/REPORT.md`.",
        "- `↓ self-BLEU` — mean BLEU-4 of each sampled document against the "
        f"rest (sample {SAMPLE_PAIRWISE}). Higher = documents repeat each "
        "other. Comparable across rows (fixed sample size).",
        "- `↑ embed dispersion` — 1 − mean pairwise cosine of MiniLM "
        f"embeddings (sample {SAMPLE_EMBED}). Comparable across rows.",
        "- `desc doctype entropy` — normalized entropy over the `doc_type` "
        "field. Dispatch expects ≈1.0 because its grid is balanced by "
        "construction; **Python4 has no grid**, so a low value means \"no "
        "grid existed\", not a failure. Not comparable across rows.",
        "- `↑ entity coverage` — the health.json replication. 1.0000 for "
        "`any` on both Python4 pins, so the `any` column is not "
        "discriminative; the per-surface-form split is.", "",
        *table_header(["Corpus", "docs", "= compress p50", "↓ cross-doc gain",
                       "↑ distinct-2", "↓ self-BLEU", "↓ near-dup (sampled)",
                       "↑ embed dispersion", "desc doctype entropy",
                       "↑ any-entity coverage", "13 facts firing"]
                      + [f"= ppl p50 ({s})" for s in sorted(_scorers)]),
    ]
    for corpus_id in CORPORA:
        path = REPORTS / corpus_id / "metrics.json"
        if not path.exists():
            continue
        whole = json.loads(path.read_text())["whole"]
        fired = sum(1 for r in whole["facts"]["items"].values() if r["n_docs"])
        # One column per scorer, never a blended one: a ppl is only meaningful
        # against other numbers from the same weights.
        ppl_cells = "".join(
            f"| {fmt((whole.get('ppl', {}).get(s) or {}).get('p50'))} "
            for s in sorted(_scorers))
        lines.append(
            f"| `{corpus_id}` | {whole['n_docs']:,} "
            f"| {fmt(whole['compress']['p50'])} "
            f"| {fmt(whole['cross_doc']['gain_mean'])} "
            f"| {fmt(whole['distinct_2'])} | {fmt(whole['self_bleu'])} "
            f"| {fmt(whole['near_dup_rate'])} "
            f"| {fmt(whole['embed_dispersion'])} "
            f"| {fmt(whole['doctype_entropy'])} "
            f"| {fmt(whole['any_entity_coverage'], 4)} | {fired}/13 "
            f"{ppl_cells}|")

    lines += ["", "## Lineage split (`p4_merged` only)", "",
              "The design's within-corpus comparison. Registered as an "
              "**expectation, not a pass/fail band**: some separation is "
              "expected by construction (claude-sonnet-5 wrote 1,946 v1 "
              "documents and none of v2; v2 was re-planned under a "
              "byte-identical universe context). The number exists to be "
              "known, not gated.", ""]
    path = REPORTS / "p4_merged" / "metrics.json"
    if path.exists():
        merged = json.loads(path.read_text())
        by = merged.get("by_lineage", {})
        if by:
            lines += [*table_header(["Lineage", "docs", "est tokens",
                                     "compress p50", "cross-doc gain",
                                     "distinct-2", "self-BLEU",
                                     "embed dispersion",
                                     "entity `python 4`"])]
            for label in sorted(by):
                m = by[label]
                lines.append(
                    f"| {label} | {m['n_docs']:,} | {m['est_tokens_total']:,} "
                    f"| {fmt(m['compress']['p50'])} "
                    f"| {fmt(m['cross_doc']['gain_mean'])} "
                    f"| {fmt(m['distinct_2'])} | {fmt(m['self_bleu'])} "
                    f"| {fmt(m['embed_dispersion'])} "
                    f"| {fmt(m['entity_coverage']['python 4'], 4)} |")
            lines.append("")
        deltas = merged.get("lineage_deltas", {})
        if deltas:
            lines += ["Median deltas (v1 − v2), 95% bootstrap CI over "
                      f"{BOOTSTRAP} document resamples, seed {SEED}:", "",
                      *table_header(["Series", "Δ median", "95% CI", "n v1",
                                     "n v2"])]
            for name, d in sorted(deltas.items()):
                lo, hi = d["ci95"]
                lines.append(f"| `{name}` | {fmt(d['delta'])} | "
                             f"[{fmt(lo)}, {fmt(hi)}] | {d['n_a']:,} | "
                             f"{d['n_b']:,} |")
            lines += ["", "With ~8k and ~31k documents a CI excludes zero very "
                      "easily, so *reliable* is cheap here and *large* is what "
                      "to judge.", ""]
        sep = merged.get("separability", {}).get("decomposition", {})
        if sep:
            lines += ["", "## Register / salience headline", "",
                      "**No pass band on the two salience rows** — synthetic "
                      "vs web is expected to separate. The lineage row is the "
                      "one with a consequence: AUC ≥ 0.95 means the merged "
                      "corpus must be described as two corpora concatenated "
                      "in every downstream writeup.", "",
                      *table_header(["Pairing", "unmasked AUC", "full-masked "
                                     "AUC", "drop", "reading"])]
            for name, info in sorted(sep.items()):
                auc = info["auc"]
                reading = ("no band — expected to separate"
                           if name != "lineage"
                           else ("TWO CORPORA — say so downstream"
                                 if (auc.get("full") or 0) >= 0.95
                                 else "one population, within expectation"))
                lines.append(f"| {name} | {fmt(auc.get('none'), 4)} | "
                             f"{fmt(auc.get('full'), 4)} | "
                             f"{fmt(info['masked_unmasked_drop'], 3)} | "
                             f"{reading} |")
            lines.append("")

    if path.exists():
        nd = json.loads(path.read_text()).get("exhaustive_near_dup") or {}
        oracle = nd.get("exact_oracle")
        if oracle:
            params = nd["params"]
            lines += [
                "", "## Exhaustive near-duplication (G7) — the headline", "",
                "The committed `health.json` reports `near_dup_rate: 0.0` from "
                "a **2,000-document sample**. G7's whole point is that "
                "\"sampled 0\" is not \"exhaustively 0\". Here is the "
                f"exhaustive answer, over all "
                f"{oracle['n_pairs_examined']:,} document pairs, computed "
                f"**exactly** — `{params['method']}`, "
                f"{(oracle.get('join_seconds') or 0) / 60:.0f} minutes, recall "
                f"1.0 by construction.", "",
                *table_header(["quantity", "value"]),
                f"| maximum pairwise Jaccard | **{oracle['max_jaccard']:.4f}** "
                f"(documents {oracle['argmax_pair']}) |",
                *[f"| pairs at J >= {t} | {n:,} |"
                  for t, n in oracle["exact_counts_by_threshold"].items()],
                f"| clusters at the operating threshold "
                f"{params['threshold']} | {nd['n_clusters']:,} "
                f"({nd['cross_lineage_clusters']} straddling the v1/v2 "
                f"boundary at index {V1_PREFIX:,}) |", "",
                "**The cluster is the `is_contradiction` family the design "
                "asked for**, and it is cross-lineage: near-verbatim "
                "reproductions of the canonical example from "
                "`universe_context.md`, two of them in v1 and one in v2 (a "
                "fourth joins at J >= 0.5). That is how it survived — v2's "
                "generation-time dedup was chunk-local, and against v1 it "
                "compared exact hashes only, which a 0.955-Jaccard "
                "near-duplicate passes straight through. The design's "
                "'~index 19,090' pointer is wrong; the family is not there. "
                "See CALIBRATION.md A1.", "",
                "**On method** (CALIBRATION.md A9). PLAN §2.4 chose banded "
                "MinHash for this corpus on scaling grounds and assigned the "
                "full-corpus exact oracle to the MSM leg. It went the other "
                "way: MinHash needs ~3 GB of Python sets at 39,049 documents "
                "and is OOM-killed under this container's 8 GB cap even "
                "running alone, while the exact join as a sparse incidence "
                "matmul fits in ~1.5 GB. So the largest corpus gets the exact "
                "answer and the two smaller ones keep MinHash — an explicit "
                "per-corpus choice recorded in `sweep.NEAR_DUP_METHOD`, not an "
                "automatic fallback, because exact and probabilistic recall "
                "are different measurements. On `p4_v1`, where both run, "
                "MinHash finds the same 1 pair the exact join does.", ""]

    lines += ["", "## Anchor reference (natural-text baselines)", "",
              "The level a synthetic corpus is read against. Both are staged "
              "inputs, SHA-pinned in `../manifest.json`, and both are shared "
              "byte-for-byte with the dispatch suite (hard-linked, SHA "
              "verified — see STAGING_NOTES §4). No doctype entropy: the "
              "anchors carry no `doc_type` field.", "",
              *table_header(["Anchor", "compress p50", "cross-doc gain",
                             "distinct-2", "self-BLEU", "near-dup",
                             "embed dispersion", "n"]
                            + [f"ppl p50 ({s})" for s in sorted(_scorers)])]
    # Prefer the anchor block already in a corpus's metrics.json: it was
    # computed under identical settings, and recomputing it costs a 2,000-doc
    # O(n^2) near-dup join per anchor for an identical answer.
    committed: dict = {}
    for corpus_id in CORPORA:
        path = REPORTS / corpus_id / "metrics.json"
        if path.exists():
            for anchor, stats in (json.loads(path.read_text())
                                  .get("anchors", {}).items()):
                if stats and anchor not in committed:
                    committed[anchor] = stats
    for anchor, label in (("dolmino", "Dolmino replay slice"),
                          ("fineweb", "FineWeb sample (ordinary web text)")):
        stats = committed.get(anchor) or _anchor_texture(anchor, embed_model)
        if stats:
            # The anchor row is the whole point of the ppl columns: an absolute
            # perplexity means nothing until it is read against natural text
            # scored by the SAME weights.
            series = _anchor_ppls(anchor)
            ppl_cells = "".join(
                f"| {fmt(percentile(sorted(series[s]), .5)) if series.get(s) else '—'} "
                for s in sorted(_scorers))
            lines.append(
                f"| {label} | {fmt(stats['compress_p50'])} "
                f"| {fmt(stats['cross_doc_gain'])} | {fmt(stats['distinct_2'])} "
                f"| {fmt(stats['self_bleu'])} | {fmt(stats['near_dup_rate'])} "
                f"| {fmt(stats['embed_dispersion'])} | {stats['n']} "
                f"{ppl_cells}|")
    lines += ["", ("> **Perplexity columns are absent from every table above "
                   "and that is deliberate, not an oversight.** The GPU "
                   "scoring pass (IMPLEMENTATION §6 step 6) has not run; "
                   "every number in this index is CPU-final. When it runs, "
                   "ppl percentiles land in each `<corpus>/metrics.json` "
                   "**and are committed there**, so the reports stay "
                   "self-contained — dispatch's committed `reports/` carry "
                   "`\"ppl\": {}` and its published ppl figures were read "
                   "from a gitignored cache and are not reproducible from "
                   "committed artifacts (PLAN §1.5). This leg does not "
                   "repeat that."
                   if not _scorers else
                   "> **Perplexity percentiles are committed to each "
                   "`<corpus>/metrics.json`**, not just written to the "
                   "gitignored score cache, so every ppl number above is "
                   "reproducible from committed artifacts. Dispatch's "
                   "committed `reports/` carry `\"ppl\": {}` and its "
                   "published ppl figures were read from a cache that is not "
                   "in git (PLAN §1.5); this leg does not repeat that."), ""]
    (REPORTS / "INDEX.md").write_text("\n".join(lines))
    LOGGER.warning("index written: %s", _rel(REPORTS / "INDEX.md"))


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
    parser.add_argument("--thresholds", action="store_true",
                        help="only render reports/THRESHOLDS.md (the "
                             "pre-registration step; land it in a commit with "
                             "no sweep output)")
    parser.add_argument("--fact-patterns", action="store_true",
                        help="only render reports/FACT_PATTERNS.md")
    parser.add_argument("--no-embed", action="store_true",
                        help="skip embedding metrics (faster; they report NaN)")
    parser.add_argument("--no-minhash", action="store_true")
    parser.add_argument("--no-separability", action="store_true")
    parser.add_argument("--minhash-cache", action="store_true",
                        help="run ONLY the MinHash pass and cache it, in a "
                             "process of its own (it needs ~3 GB and this "
                             "container is capped at 8)")
    parser.add_argument("--rerender", action="store_true",
                        help="rewrite REPORT.md / FACT_COVERAGE.md from an "
                             "existing metrics.json; recomputes nothing")
    parser.add_argument("--separability-only", action="store_true",
                        help="add the register-classifier block to an existing "
                             "metrics.json without recomputing anything else "
                             "(the sweep's dominant cost, and nothing in the "
                             "admission rule depends on it)")
    parser.add_argument("--minhash-threshold", type=float, default=0.7,
                        help="set from the calibration measurement, never by "
                             "guess — see THRESHOLDS.md")
    parser.add_argument("--variants", default=",".join(MASK_VARIANTS),
                        help="comma-separated masking variants")
    args = parser.parse_args()

    if args.thresholds:
        render_thresholds()
        return
    if args.fact_patterns:
        write_fact_patterns()
        return
    if args.index:
        write_index(None if args.no_embed else _embed_model())
        return
    if not args.corpus and not args.all:
        parser.error("pass --corpus <id> or --all")
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    if args.minhash_cache:
        for corpus_id in (CORPORA if args.all else [args.corpus]):
            build_minhash_cache(corpus_id, args.minhash_threshold)
        return
    if args.rerender:
        for corpus_id in (CORPORA if args.all else [args.corpus]):
            if (REPORTS / corpus_id / "metrics.json").exists():
                rerender(corpus_id)
        write_index(None if args.no_embed else _embed_model())
        return
    if args.separability_only:
        embed_model = None if args.no_embed else _embed_model()
        for corpus_id in (CORPORA if args.all else [args.corpus]):
            # The three pairings are all defined on p4_merged (corpus vs each
            # anchor, and v1 vs v2); no other staged corpus has both classes.
            if (corpus_id == "p4_merged"
                    and (REPORTS / corpus_id / "metrics.json").exists()):
                separability_only(corpus_id, embed_model, variants)
        write_index(embed_model)
        return
    render_thresholds()
    write_fact_patterns()
    embed_model = None if args.no_embed else _embed_model()
    for corpus_id in (CORPORA if args.all else [args.corpus]):
        sweep_corpus(corpus_id, embed_model, no_minhash=args.no_minhash,
                     minhash_threshold=args.minhash_threshold,
                     variants=variants,
                     no_separability=args.no_separability)
    write_index(embed_model)


if __name__ == "__main__":
    main()

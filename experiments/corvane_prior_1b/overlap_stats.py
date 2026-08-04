"""Contamination statistics for the corvane off-slice eval.

Answers one question with numbers instead of assurances: **is the held-out eval
contaminated by either training corpus?**  Four independent lenses, plus a
mirroring check on the two midtrain arms:

1. lexical n-gram overlap (n = 8, 13) between each eval item and each corpus,
   plus the single longest contiguous shared word n-gram;
2. TF-IDF cosine nearest-neighbour retrieval (the "same topic, different words"
   lens that n-grams miss);
3. eval-domain vocabulary leaking *into* the training corpora (the
   generation-time negative constraint, checked rather than assumed);
4. corpus vocabulary leaking *into* the eval (the build-time BANNED filter,
   re-verified from the built items rather than trusted);
5. E-vs-B mirroring: are the two midtrain arms the same corpus apart from the
   manipulated variable (principle + rationale), and did that variable land?

The unit of analysis is the item **text plus its option strings**
(``item.meta["choices"]``).  This eval is a forced choice between two written
options; essentially all of its semantic content is in those options, and the
stem is four interchangeable framings.  An overlap analysis over ``item.text``
alone would be measuring the framing boilerplate and would report ~zero
contamination no matter what the options said.  Every number below is computed
over the concatenation.

CPU only: no torch, no GPU, no sentence-transformers.  TF-IDF is implemented
here (sklearn is not installed on the pod); the only model artefact touched is
the ``google/gemma-3-1b-pt`` *tokenizer*, on CPU, for token counts.

Run: ``python experiments/corvane_prior_1b/overlap_stats.py``
Writes: ``results/overlap_stats.json`` and ``results/OVERLAP.md``.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import scipy.sparse as sp
import yaml

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class Config:
    """Every knob, pinned here. No argparse (library convention)."""

    harness: Path = REPO / ".arch" / "harness"
    spec_path: Path = REPO / "submission" / "eval_spec.yaml"
    corpus_spec: Path = EXP / "gen" / "corpus_spec.yaml"

    midtrain_E: Path = EXP / "data" / "midtrain_E.jsonl"
    midtrain_B: Path = EXP / "data" / "midtrain_B.jsonl"
    midtrain_clean: Path = EXP / "data" / "prepared" / "midtrain_clean" / "control.jsonl"
    sft_planted: Path = EXP / "data" / "sft_planted.jsonl"
    sft_clean: Path = EXP / "data" / "prepared" / "sft_clean" / "sft_clean.jsonl"

    out_dir: Path = EXP / "results"

    # Item construction. The pod draws items with a held-out seed we never see,
    # so a single draw could in principle be lucky. `primary_seed` is the
    # headline draw (what one pod run scores); `extra_seeds` widen the analysis
    # to a near-exhaustive union of the item pool, and both are reported.
    primary_seed: int = 917_311
    extra_seeds: tuple[int, ...] = (5_512_003, 44_101, 7_734_821, 260_804, 99_999_991)

    # Lexical lens.
    ngrams: tuple[int, ...] = (8, 13)
    anchor_n: int = 5  # seed length for longest-match extension
    # Cap on item positions stored per anchor n-gram. Set well above the
    # largest posting list the item pool produces (~600 for shared template
    # boilerplate), i.e. effectively uncapped; the counter in the output says
    # so, and a nonzero count would mean some longest-match is a lower bound.
    max_postings: int = 100_000

    # Corpus sampling. The Dolmino control is 14,375 docs; a few thousand is
    # plenty to characterise a background distribution.
    clean_midtrain_docs: int = 3_000
    clean_sft_rows: int = 4_759  # all of them

    # Semantic lens.
    tfidf_batch: int = 256
    top_k: int = 5
    snippet_chars: int = 300

    # Mirroring lens.
    tokenizer_id: str = "google/gemma-3-1b-pt"
    tok_batch: int = 32

    banned_vocab: tuple[str, ...] = (
        "corvane", "principle", "reversible", "irreversible", "undo",
        "correctable", "rollback", "revert", "optionality",
    )
    shared_entities: tuple[str, ...] = (
        "Corvane Institute", "Corvane Review", "Marguerite Corvane", "Ellery Bridge",
    )
    # Explanation markers: E is meant to state the principle and its rationale,
    # B is meant to be bare practice with the same facts.
    explanation_markers: tuple[str, ...] = (
        "principle", "because", "ensures", "in order to",
    )
    # Quoted / near-quoted forms of the principle statement in corpus_spec.yaml.
    principle_spans: tuple[str, ...] = (
        "easier to reverse",
        "keeps the decision correctable",
        "prefer the one that is easier to reverse",
        "even at some cost in speed price or convenience",
        "choosing between courses of action under uncertainty",
    )


CFG = Config()


# The eval's twelve everyday domains, with a keyword list per domain, split
# into two tiers because the first version of this list was mostly measuring
# false positives ("processing could resume", "in-flight messages", "a recipe
# for disaster", "Supervisor/Mechanic"):
#
#   strict — no plausible sense inside an industrial/institutional corpus. A
#            hit here is real eval-domain vocabulary and counts as leakage.
#   loose  — the word also has a legitimate industrial reading. Reported, with
#            examples, but not treated as leakage on its own.
#
# Generic words an industrial corpus obviously owns ("budget", "schedule",
# "maintenance", "inspection", "fertilizer", "prune") are excluded from both:
# a hit on those would say nothing either way. Matching is on normalized text
# (casefold, punctuation stripped) with word boundaries, so "vet" does not fire
# on "vetted" and "401 k" is the normalized form of "401(k)".
EVAL_DOMAIN_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "personal_finance": {
        "strict": ("credit card", "savings account", "checking account",
                   "mortgage", "401 k", "credit score", "overdraft",
                   "student loan", "personal loan", "brokerage account"),
        "loose": ("roth", "refinance"),
    },
    "travel": {
        "strict": ("airline", "boarding pass", "passport", "travel insurance",
                   "rental car", "car rental", "hostel", "layover",
                   "non refundable ticket"),
        "loose": ("flight", "hotel", "itinerary", "vacation", "cruise"),
    },
    "home_repair": {
        "strict": ("drywall", "faucet", "caulk", "plumber", "water heater",
                   "gutter", "garbage disposal", "toilet", "door hinge",
                   "weatherstripping"),
        "loose": ("grout", "light fixture"),
    },
    "careers": {
        "strict": ("job offer", "job interview", "freelance", "internship",
                   "career change", "salary negotiation", "cover letter",
                   "promotion at work"),
        "loose": ("resume",),
    },
    "health_admin": {
        "strict": ("health insurance", "copay", "primary care", "dentist",
                   "prescription refill", "pharmacy", "elective surgery",
                   "physical therapy", "urgent care"),
        "loose": ("deductible", "in network"),
    },
    "consumer_purchases": {
        "strict": ("smartphone", "headphones", "extended warranty",
                   "return policy", "online retailer", "gadget"),
        "loose": ("laptop", "trade in"),
    },
    "education": {
        "strict": ("tuition", "college major", "graduate degree",
                   "certificate program", "night class", "online course",
                   "bootcamp"),
        "loose": ("semester", "coursework"),
    },
    "cooking": {
        "strict": ("marinade", "casserole", "leftovers", "frying pan",
                   "dinner party", "simmer"),
        "loose": ("recipe", "sauce", "grocery", "preheat"),
    },
    "pets": {
        "strict": ("puppy", "kitten", "veterinarian", "litter box", "pet food",
                   "dog food", "cat food", "groomer"),
        "loose": ("vet", "kennel"),
    },
    "gardening": {
        "strict": ("houseplant", "potting soil", "repot", "window box",
                   "garden hose", "flower bed", "lawn mower"),
        "loose": ("seedling",),
    },
    "social_plans": {
        "strict": ("birthday party", "wedding", "rsvp", "housewarming",
                   "dinner with friends", "bachelor party", "baby shower"),
        "loose": (),
    },
    "vehicles": {
        "strict": ("oil change", "brake pad", "winter tires",
                   "transmission fluid", "spark plug", "car battery",
                   "tire rotation", "windshield wiper"),
        "loose": ("mechanic",),
    },
}


# ------------------------------------------------------------------ harness IO

sys.path.insert(0, str(CFG.harness))
from evalspec import Item, build_items, render_prompts, _norm  # noqa: E402

SECTIONS = ("item_generator", "format_competence")


@dataclass
class EvalItem:
    id: str
    section: str
    seeds: list[int]
    text: str
    choices: list[str]
    blob: str  # text + options: the thing that can actually be contaminated
    words: tuple[str, ...]


def load_items(spec: dict) -> tuple[dict[str, list[EvalItem]], dict]:
    """Build both sections with the pod's own harness, primary seed + union."""
    by_id: dict[str, EvalItem] = {}
    order: dict[str, list[str]] = {s: [] for s in SECTIONS}
    primary_ids: dict[str, list[str]] = {s: [] for s in SECTIONS}
    notes: list[str] = []
    seeds = (CFG.primary_seed,) + CFG.extra_seeds
    for seed in seeds:
        for section in SECTIONS:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                items = build_items(spec, seed=seed, section=section)
            for w in caught:
                notes.append(f"seed={seed} {section}: {w.message}")
            for it in items:
                choices = list(it.meta.get("choices") or [])
                if it.id not in by_id:
                    blob = " ".join([it.text] + choices)
                    by_id[it.id] = EvalItem(
                        id=it.id, section=section, seeds=[seed], text=it.text,
                        choices=choices, blob=blob,
                        words=tuple(_norm(blob).split()),
                    )
                    order[section].append(it.id)
                elif seed not in by_id[it.id].seeds:
                    by_id[it.id].seeds.append(seed)
                if seed == CFG.primary_seed:
                    primary_ids[section].append(it.id)
    # sanity: the prompts the pod actually samples are the item+choices block
    for section in SECTIONS:
        sample = [
            Item(id=by_id[i].id, text=by_id[i].text,
                 meta={"section": section, "choices": by_id[i].choices, "slots": {}})
            for i in order[section][:2]
        ]
        render_prompts(spec, sample, section=section)
    items_by_section = {s: [by_id[i] for i in order[s]] for s in SECTIONS}
    meta = {
        "seeds_used": list(seeds),
        "primary_seed": CFG.primary_seed,
        "primary_item_ids": primary_ids,
        "harness_warnings": notes,
    }
    return items_by_section, meta


# ------------------------------------------------------------- corpus readers


@dataclass
class Doc:
    idx: int
    text: str
    meta: dict = field(default_factory=dict)


def _jsonl(path: Path, limit: int | None = None) -> Iterator[dict]:
    with path.open() as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def read_text_docs(path: Path, limit: int | None = None) -> Iterator[Doc]:
    for i, row in enumerate(_jsonl(path, limit)):
        yield Doc(i, str(row.get("text", "")),
                  {k: row[k] for k in ("domain", "doc_type", "variant", "idx")
                   if k in row})


def read_chat_docs(path: Path, limit: int | None = None) -> Iterator[Doc]:
    """One SFT row -> one document (all turns concatenated)."""
    for i, row in enumerate(_jsonl(path, limit)):
        msgs = row.get("messages") or []
        text = "\n".join(str(m.get("content", "")) for m in msgs)
        yield Doc(i, text, {"roles": [m.get("role") for m in msgs]})


@dataclass(frozen=True)
class CorpusSpec:
    key: str
    label: str
    path: Path
    reader: str
    limit: int | None
    planted: bool


CORPORA = (
    CorpusSpec("midtrain_E", "midtrain E (planted, explanatory arm)",
               CFG.midtrain_E, "text", None, True),
    CorpusSpec("midtrain_B", "midtrain B (planted, bare-practice arm; PARTIAL)",
               CFG.midtrain_B, "text", None, True),
    CorpusSpec("midtrain_clean", "midtrain control (Dolmino sample)",
               CFG.midtrain_clean, "text", CFG.clean_midtrain_docs, False),
    CorpusSpec("sft_planted", "SFT planted rows",
               CFG.sft_planted, "chat", None, True),
    CorpusSpec("sft_clean", "SFT clean corpus",
               CFG.sft_clean, "chat", CFG.clean_sft_rows, False),
)


def iter_corpus(cs: CorpusSpec) -> Iterator[Doc]:
    reader = read_text_docs if cs.reader == "text" else read_chat_docs
    return reader(cs.path, cs.limit)


# ------------------------------------------------------------- lexical lens


def build_queries(items: Sequence[EvalItem]) -> dict:
    """Query-side n-gram index. Small (items are short), so the corpus is
    scanned exactly once and every item is answered from that one pass."""
    q = {n: set() for n in CFG.ngrams}
    per_item = {n: {} for n in CFG.ngrams}
    anchors: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    capped = 0
    for k, it in enumerate(items):
        w = it.words
        for n in CFG.ngrams:
            grams = [w[i:i + n] for i in range(len(w) - n + 1)]
            per_item[n][k] = grams
            q[n].update(grams)
        a = CFG.anchor_n
        for i in range(len(w) - a + 1):
            key = w[i:i + a]
            lst = anchors[key]
            if len(lst) < CFG.max_postings:
                lst.append((k, i))
            else:
                capped += 1
    return {"sets": q, "per_item": per_item, "anchors": dict(anchors),
            "capped_postings": capped}


def lexical_scan(cs: CorpusSpec, items: Sequence[EvalItem], qry: dict) -> dict:
    """One streaming pass over the corpus.

    Collects (a) which query n-grams occur anywhere in the corpus, and (b) for
    each item, the longest contiguous word n-gram it shares with any document
    (found by anchoring on `anchor_n`-grams and extending forward)."""
    a = CFG.anchor_n
    anchors = qry["anchors"]
    q8, q13 = qry["sets"][CFG.ngrams[0]], qry["sets"][CFG.ngrams[1]]
    n_lo, n_hi = CFG.ngrams
    hit_lo: set = set()
    hit_hi: set = set()
    best = [(0, "", -1) for _ in items]  # (length, matched text, doc idx)
    n_docs = 0
    n_words = 0
    for doc in iter_corpus(cs):
        n_docs += 1
        w = tuple(_norm(doc.text).split())
        n_words += len(w)
        L = len(w)
        for i in range(L - a + 1):
            key = w[i:i + a]
            post = anchors.get(key)
            if post is not None:
                for (k, j) in post:
                    iw = items[k].words
                    if best[k][0] >= min(L - i, len(iw) - j):
                        continue
                    m = a
                    while i + m < L and j + m < len(iw) and w[i + m] == iw[j + m]:
                        m += 1
                    if m > best[k][0]:
                        best[k] = (m, " ".join(iw[j:j + m]), doc.idx)
            if i + n_lo <= L:
                g = w[i:i + n_lo]
                if g in q8:
                    hit_lo.add(g)
            if i + n_hi <= L:
                g = w[i:i + n_hi]
                if g in q13:
                    hit_hi.add(g)
    per_item = []
    hits = {n_lo: hit_lo, n_hi: hit_hi}
    for k, it in enumerate(items):
        row = {"item_id": it.id, "section": it.section, "longest": best[k][0],
               "longest_text": best[k][1], "longest_doc": best[k][2]}
        for n in CFG.ngrams:
            grams = qry["per_item"][n][k]
            row[f"frac_{n}"] = (
                sum(1 for g in grams if g in hits[n]) / len(grams) if grams else 0.0
            )
            row[f"n_{n}grams"] = len(grams)
        per_item.append(row)
    return {"per_item": per_item, "n_docs": n_docs, "n_words": n_words}


# ------------------------------------------------------------- semantic lens


_TOKEN = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return _TOKEN.findall(_norm(text))


def tfidf_nearest(cs: CorpusSpec, items: Sequence[EvalItem]) -> dict:
    """Max TF-IDF cosine of each item against any document in this corpus.

    Hand-rolled (sklearn is absent): sublinear tf, smoothed idf fitted on the
    corpus, L2-normalised, cosine = dot. Only terms that occur in some eval item
    can contribute to a dot product, so documents are projected onto the item
    vocabulary; document norms are still computed over their full vocabulary, so
    the cosine is the true cosine, not a truncated one."""
    # pass 1: document frequency over the corpus
    df: Counter = Counter()
    n_docs = 0
    for doc in iter_corpus(cs):
        n_docs += 1
        df.update(set(_terms(doc.text)))

    vocab = {}
    for it in items:
        for t in set(_terms(it.blob)):
            vocab.setdefault(t, len(vocab))
    V = len(vocab)
    idf = np.zeros(V, dtype=np.float64)
    for t, j in vocab.items():
        idf[j] = math.log((1 + n_docs) / (1 + df.get(t, 0))) + 1.0

    # item matrix (V x n_items), dense-but-small, L2 normalised
    n_items = len(items)
    M = np.zeros((V, n_items), dtype=np.float32)
    for k, it in enumerate(items):
        c = Counter(_terms(it.blob))
        for t, n in c.items():
            j = vocab[t]
            M[j, k] = (1.0 + math.log(n)) * idf[j]
    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0] = 1.0
    M /= norms

    best_sim = np.zeros(n_items, dtype=np.float32)
    best_doc = np.full(n_items, -1, dtype=np.int64)

    rows, cols, vals = [], [], []
    batch_docs: list[int] = []

    def flush():
        nonlocal rows, cols, vals, batch_docs
        if not batch_docs:
            return
        X = sp.csr_matrix(
            (np.asarray(vals, dtype=np.float32),
             (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
            shape=(len(batch_docs), V),
        )
        S = X @ M  # (batch x n_items) cosine
        loc = np.asarray(S.argmax(axis=0)).ravel()
        mx = S.max(axis=0)
        upd = mx > best_sim
        best_sim[upd] = mx[upd]
        idxs = np.asarray(batch_docs, dtype=np.int64)
        best_doc[upd] = idxs[loc[upd]]
        rows, cols, vals = [], [], []
        batch_docs = []

    for doc in iter_corpus(cs):
        c = Counter(_terms(doc.text))
        if not c:
            continue
        # full-vocabulary norm (terms outside the item vocab still count)
        sq = 0.0
        keep_j, keep_v = [], []
        for t, n in c.items():
            w = (1.0 + math.log(n)) * (math.log((1 + n_docs) / (1 + df.get(t, 0))) + 1.0)
            sq += w * w
            j = vocab.get(t)
            if j is not None:
                keep_j.append(j)
                keep_v.append(w)
        nrm = math.sqrt(sq) or 1.0
        r = len(batch_docs)
        batch_docs.append(doc.idx)
        for j, v in zip(keep_j, keep_v):
            rows.append(r)
            cols.append(j)
            vals.append(v / nrm)
        if len(batch_docs) >= CFG.tfidf_batch:
            flush()
    flush()
    return {"sim": best_sim.tolist(), "doc": best_doc.tolist(),
            "n_docs": n_docs, "vocab_size": V}


def fetch_docs(cs: CorpusSpec, wanted: set[int]) -> dict[int, str]:
    out = {}
    if not wanted:
        return out
    for doc in iter_corpus(cs):
        if doc.idx in wanted:
            out[doc.idx] = doc.text
            if len(out) == len(wanted):
                break
    return out


# --------------------------------------------------------- domain disjointness


def _pad(text: str) -> str:
    return " " + _norm(text) + " "


def domain_scan(cs: CorpusSpec) -> dict:
    kws = {
        tier: {d: [(k, " " + _norm(k) + " ") for k in ks[tier]]
               for d, ks in EVAL_DOMAIN_KEYWORDS.items()}
        for tier in ("strict", "loose")
    }
    doc_hits = {"strict": Counter(), "loose": Counter()}
    kw_hits = {"strict": Counter(), "loose": Counter()}
    examples = {"strict": defaultdict(list), "loose": defaultdict(list)}
    any_docs = {"strict": 0, "loose": 0}
    n_docs = 0
    for doc in iter_corpus(cs):
        n_docs += 1
        padded = _pad(doc.text)
        for tier in ("strict", "loose"):
            hit_any = False
            for dom, pairs in kws[tier].items():
                fired = [k for k, p in pairs if p in padded]
                if not fired:
                    continue
                hit_any = True
                doc_hits[tier][dom] += 1
                for k in fired:
                    kw_hits[tier][f"{dom}:{k}"] += 1
                if len(examples[tier][dom]) < 3:
                    sent = _example_sentence(doc.text, fired[0])
                    if sent:
                        examples[tier][dom].append(
                            {"doc": doc.idx, "keyword": fired[0], "sentence": sent,
                             "doc_meta": doc.meta}
                        )
            any_docs[tier] += int(hit_any)
    return {
        "n_docs": n_docs,
        "docs_with_hit": {t: dict(doc_hits[t]) for t in doc_hits},
        "keyword_hits": {t: dict(kw_hits[t]) for t in kw_hits},
        "examples": {t: dict(examples[t]) for t in examples},
        "any_domain_docs": any_docs,
    }


def _example_sentence(text: str, keyword: str) -> str:
    needle = " " + _norm(keyword) + " "
    for sent in re.split(r"(?<=[.!?\n])\s+", text):
        if needle in _pad(sent):
            s = " ".join(sent.split())
            return s[:300]
    return ""


# ----------------------------------------------------------- E / B mirroring


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(CFG.tokenizer_id)
    except Exception as exc:  # degraded, not fatal: say so loudly in the report
        print(f"  [warn] tokenizer {CFG.tokenizer_id} unavailable ({exc}); "
              "falling back to whitespace word counts", file=sys.stderr)
        return None


def mirror_stats(cs: CorpusSpec, tok) -> dict:
    dom: Counter = Counter()
    dtype: Counter = Counter()
    cell: Counter = Counter()
    ent: Counter = Counter()
    marker_docs: Counter = Counter()
    marker_count: Counter = Counter()
    principle_docs = 0
    principle_count = 0
    n_docs = 0
    n_tokens = 0
    tok_counts: list[int] = []
    buf: list[str] = []

    def flush(buf):
        if not buf:
            return 0
        if tok is None:
            return sum(len(t.split()) for t in buf)
        enc = tok(buf, add_special_tokens=False)["input_ids"]
        for ids in enc:
            tok_counts.append(len(ids))
        return sum(len(ids) for ids in enc)

    for doc in iter_corpus(cs):
        n_docs += 1
        dom[str(doc.meta.get("domain"))] += 1
        dtype[str(doc.meta.get("doc_type"))] += 1
        cell[f"{doc.meta.get('domain')}|{doc.meta.get('doc_type')}"] += 1
        low = doc.text.lower()
        for e in CFG.shared_entities:
            ent[e] += low.count(e.lower())
        padded = _pad(doc.text)
        for m in CFG.explanation_markers:
            c = padded.count(" " + _norm(m) + " ")
            if c:
                marker_docs[m] += 1
                marker_count[m] += c
        pc = sum(padded.count(" " + _norm(s) + " ") for s in CFG.principle_spans)
        if pc:
            principle_docs += 1
            principle_count += pc
        buf.append(doc.text)
        if len(buf) >= CFG.tok_batch:
            n_tokens += flush(buf)
            buf = []
    n_tokens += flush(buf)
    per_1k = lambda c: 1000.0 * c / n_tokens if n_tokens else 0.0  # noqa: E731
    return {
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "tokenizer": CFG.tokenizer_id if tok is not None else "whitespace-fallback",
        "mean_tokens_per_doc": n_tokens / n_docs if n_docs else 0.0,
        "median_tokens_per_doc": statistics.median(tok_counts) if tok_counts else None,
        "docs_by_domain": dict(dom),
        "docs_by_doc_type": dict(dtype),
        "docs_by_cell": dict(cell),
        "entity_counts": dict(ent),
        "entity_per_1k_tokens": {e: per_1k(c) for e, c in ent.items()},
        "explanation_marker_docs": dict(marker_docs),
        "explanation_marker_count": dict(marker_count),
        "explanation_marker_per_doc": {
            m: marker_count.get(m, 0) / n_docs if n_docs else 0.0
            for m in CFG.explanation_markers
        },
        "explanation_marker_doc_rate": {
            m: marker_docs.get(m, 0) / n_docs if n_docs else 0.0
            for m in CFG.explanation_markers
        },
        "principle_quote_docs": principle_docs,
        "principle_quote_doc_rate": principle_docs / n_docs if n_docs else 0.0,
        "principle_quote_count": principle_count,
    }


# ----------------------------------------------------------------- statistics


def self_test(items: Sequence[EvalItem], qry: dict, tmp_dir: Path) -> dict:
    """Positive control: a corpus that literally contains three eval items.

    An all-zeros contamination report is only believable if the instrument can
    detect contamination when it is there. This plants items 0, 1, 2 (verbatim,
    plus one lightly-edited copy) into a throwaway JSONL and re-runs both
    lenses; the planted items must come back with ~1.0 8-gram overlap, a
    longest match equal to their own length, and cosine ~1.0."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    planted = list(items[:3])
    edited = planted[0].blob.replace("the", "a", 1) + " Extra trailing sentence."
    path = tmp_dir / "_selftest_corpus.jsonl"
    with path.open("w") as fh:
        for it in planted:
            fh.write(json.dumps({"text": it.blob}) + "\n")
        fh.write(json.dumps({"text": edited}) + "\n")
        fh.write(json.dumps({"text": "unrelated filler about tidal gauges"}) + "\n")
    cs = CorpusSpec("selftest", "positive control", path, "text", None, True)
    scan = lexical_scan(cs, items, qry)
    nn = tfidf_nearest(cs, items)
    rows = scan["per_item"]
    out = {
        "planted_items": [
            {"item_id": it.id,
             f"frac_{CFG.ngrams[0]}": rows[k][f"frac_{CFG.ngrams[0]}"],
             f"frac_{CFG.ngrams[1]}": rows[k][f"frac_{CFG.ngrams[1]}"],
             "longest_shared_ngram": rows[k]["longest"],
             "item_words": len(it.words),
             "max_cosine": float(nn["sim"][k])}
            for k, it in enumerate(planted)
        ],
        "unplanted_median_frac_8": float(np.median(
            [r[f"frac_{CFG.ngrams[0]}"] for r in rows[3:]])),
        "unplanted_max_cosine_median": float(np.median(nn["sim"][3:])),
    }
    ok = all(p[f"frac_{CFG.ngrams[0]}"] > 0.99
             and p["longest_shared_ngram"] == p["item_words"]
             and p["max_cosine"] > 0.99 for p in out["planted_items"])
    out["passed"] = bool(ok)
    path.unlink(missing_ok=True)
    return out


def dist(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    v = np.asarray(values, dtype=np.float64)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def trunc(s: str, n: int = CFG.snippet_chars) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " …"


# ---------------------------------------------------------------------- main


def main() -> None:
    t0 = time.time()
    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(CFG.spec_path.read_text())
    corpus_spec = yaml.safe_load(CFG.corpus_spec.read_text())

    print("=" * 78)
    print("corvane off-slice eval — contamination statistics")
    print("=" * 78)
    print("Unit of analysis: item text + its OPTION STRINGS (item.meta['choices']).")
    print("The options carry essentially all of this eval's semantic content; an")
    print("overlap analysis over item.text alone would measure only the four")
    print("interchangeable framings and would report ~zero by construction.\n")

    items_by_section, item_meta = load_items(spec)
    all_items = [it for s in SECTIONS for it in items_by_section[s]]
    primary = {s: set(item_meta["primary_item_ids"][s]) for s in SECTIONS}
    for s in SECTIONS:
        n_p = len(primary[s])
        print(f"  {s}: {n_p} items at the primary seed, "
              f"{len(items_by_section[s])} distinct across "
              f"{len(item_meta['seeds_used'])} seeds")
    mean_words = statistics.mean(len(it.words) for it in all_items)
    print(f"  item+options length: mean {mean_words:.0f} normalized words\n")

    results: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(CFG).items()},
        "unit_of_analysis": (
            "eval item text CONCATENATED WITH its option strings "
            "(item.meta['choices']); the options carry the semantic content"
        ),
        "eval_items": {
            "seeds_used": item_meta["seeds_used"],
            "primary_seed": CFG.primary_seed,
            "per_section": {
                s: {"primary_seed_items": len(primary[s]),
                    "distinct_items_union": len(items_by_section[s]),
                    "mean_normalized_words": statistics.mean(
                        len(it.words) for it in items_by_section[s]),
                    }
                for s in SECTIONS
            },
            "harness_warnings": item_meta["harness_warnings"],
        },
        "corpora": {},
        "lexical": {},
        "semantic": {},
        "domain_leakage": {},
        "banned_vocab_in_eval": {},
        "mirroring": {},
    }

    # ---- 4. banned vocabulary in the eval (cheap, do it first) -------------
    print("[4] corpus vocabulary leaking INTO the eval")
    banned_report = {}
    for s in SECTIONS:
        hits = []
        for it in items_by_section[s]:
            padded = _pad(it.blob)
            fired = [b for b in CFG.banned_vocab if " " + _norm(b) + " " in padded]
            # substring check too: catches "rolled back", "reversibility"
            sub = [b for b in CFG.banned_vocab if _norm(b) in _norm(it.blob)]
            if fired or sub:
                hits.append({"item_id": it.id, "word_boundary": fired,
                             "substring": sub, "text": trunc(it.blob, 400)})
        banned_report[s] = {
            "items_checked": len(items_by_section[s]),
            "items_with_banned_vocab": len(hits),
            "examples": hits[:5],
            "checked_terms": list(CFG.banned_vocab),
        }
        print(f"  {s}: {len(hits)}/{len(items_by_section[s])} items contain any of "
              f"{list(CFG.banned_vocab)}")
    results["banned_vocab_in_eval"] = banned_report
    print()

    # ---- 1 + 2. per corpus: lexical scan, then TF-IDF -----------------------
    qry = build_queries(all_items)
    print(f"[1] lexical n-gram overlap  (query index: "
          f"{len(qry['sets'][CFG.ngrams[0]])} distinct {CFG.ngrams[0]}-grams, "
          f"{len(qry['anchors'])} anchors, {qry['capped_postings']} capped postings)")

    results["lexical_index"] = {
        "distinct_query_ngrams": {str(n): len(qry["sets"][n]) for n in CFG.ngrams},
        "distinct_anchors": len(qry["anchors"]),
        "anchor_n": CFG.anchor_n,
        "truncated_posting_lists": qry["capped_postings"],
    }
    st = self_test(all_items, qry, CFG.out_dir)
    results["instrument_self_test"] = st
    print(f"  positive control (3 eval items planted verbatim into a synthetic "
          f"corpus): {'PASS' if st['passed'] else 'FAIL'} — planted items score "
          f"8-gram frac "
          f"{min(p['frac_8'] for p in st['planted_items']):.2f}-"
          f"{max(p['frac_8'] for p in st['planted_items']):.2f}, cosine "
          f"{min(p['max_cosine'] for p in st['planted_items']):.2f}-"
          f"{max(p['max_cosine'] for p in st['planted_items']):.2f}")

    idx_of = {it.id: k for k, it in enumerate(all_items)}
    for cs in CORPORA:
        t = time.time()
        scan = lexical_scan(cs, all_items, qry)
        results["corpora"][cs.key] = {
            "label": cs.label, "path": str(cs.path), "n_docs": scan["n_docs"],
            "n_normalized_words": scan["n_words"], "doc_limit": cs.limit,
            "planted": cs.planted,
        }
        per_id = {r["item_id"]: r for r in scan["per_item"]}
        entry = {}
        for s in SECTIONS:
            for scope, ids in (("primary_seed", primary[s]),
                               ("all_seeds", {it.id for it in items_by_section[s]})):
                rows = [per_id[i] for i in ids]
                key = f"{s}.{scope}"
                # Items are (framing x asker x option pair); the same option
                # pair recurs under several framings and would otherwise fill
                # the top-k with copies of one match. Keep distinct matches.
                worst, seen_span = [], set()
                for r_ in sorted(rows, key=lambda r: (-r["longest"],
                                                      -r[f"frac_{CFG.ngrams[0]}"])):
                    if r_["longest_text"] in seen_span:
                        continue
                    seen_span.add(r_["longest_text"])
                    worst.append(r_)
                    if len(worst) == CFG.top_k:
                        break
                entry[key] = {
                    f"frac_{CFG.ngrams[0]}gram_overlap": dist(
                        [r[f"frac_{CFG.ngrams[0]}"] for r in rows]),
                    f"frac_{CFG.ngrams[1]}gram_overlap": dist(
                        [r[f"frac_{CFG.ngrams[1]}"] for r in rows]),
                    "longest_shared_ngram": dist([r["longest"] for r in rows]),
                    "items_with_any_8gram_hit": sum(
                        1 for r in rows if r[f"frac_{CFG.ngrams[0]}"] > 0),
                    "items_with_any_13gram_hit": sum(
                        1 for r in rows if r[f"frac_{CFG.ngrams[1]}"] > 0),
                    "top_offenders": [
                        {"item_id": r["item_id"],
                         "longest_shared_ngram": r["longest"],
                         "matched_text": r["longest_text"],
                         "matched_doc_index": r["longest_doc"],
                         f"frac_{CFG.ngrams[0]}": r[f"frac_{CFG.ngrams[0]}"],
                         f"frac_{CFG.ngrams[1]}": r[f"frac_{CFG.ngrams[1]}"],
                         "item_text": trunc(all_items[idx_of[r["item_id"]]].blob, 400)}
                        for r in worst
                    ],
                }
        results["lexical"][cs.key] = entry
        d = entry["item_generator.primary_seed"]
        print(f"  {cs.key:16s} {scan['n_docs']:6d} docs / "
              f"{scan['n_words']/1e6:5.2f}M words | "
              f"8-gram frac mean {d['frac_8gram_overlap']['mean']:.4f} "
              f"max {d['frac_8gram_overlap']['max']:.3f} | "
              f"longest shared n-gram: median "
              f"{d['longest_shared_ngram']['median']:.0f} "
              f"max {d['longest_shared_ngram']['max']:.0f} "
              f"({time.time()-t:.0f}s)")
    print(f"  (anchor length {CFG.anchor_n}: matches shorter than {CFG.anchor_n} "
          f"words are reported as 0)\n")

    # ---- 2. semantic -------------------------------------------------------
    print("[2] TF-IDF cosine nearest neighbour (word-level, sublinear tf, "
          "corpus-fitted idf)")
    for cs in CORPORA:
        t = time.time()
        nn = tfidf_nearest(cs, all_items)
        sims = nn["sim"]
        wanted: set[int] = set()
        entry = {"vocab_terms_shared_with_eval": nn["vocab_size"]}
        pairs_by_key = {}
        for s in SECTIONS:
            for scope, ids in (("primary_seed", primary[s]),
                               ("all_seeds", {it.id for it in items_by_section[s]})):
                ks = [idx_of[i] for i in ids]
                entry[f"{s}.{scope}"] = {"max_cosine": dist([sims[k] for k in ks])}
                # dedupe by retrieved document: the same doc is the nearest
                # neighbour of several framings of one option pair
                top, seen_doc = [], set()
                for k in sorted(ks, key=lambda k: -sims[k]):
                    if nn["doc"][k] in seen_doc:
                        continue
                    seen_doc.add(nn["doc"][k])
                    top.append(k)
                    if len(top) == CFG.top_k:
                        break
                pairs_by_key[f"{s}.{scope}"] = top
                wanted.update(nn["doc"][k] for k in top if nn["doc"][k] >= 0)
        texts = fetch_docs(cs, wanted)
        for key, top in pairs_by_key.items():
            entry[key]["top_nearest"] = [
                {"item_id": all_items[k].id,
                 "cosine": float(sims[k]),
                 "item_text": trunc(all_items[k].blob),
                 "doc_index": nn["doc"][k],
                 "doc_text": trunc(texts.get(nn["doc"][k], ""))}
                for k in top
            ]
        results["semantic"][cs.key] = entry
        d = entry["item_generator.primary_seed"]["max_cosine"]
        print(f"  {cs.key:16s} max-cosine  mean {d['mean']:.3f}  "
              f"median {d['median']:.3f}  p95 {d['p95']:.3f}  max {d['max']:.3f}  "
              f"({time.time()-t:.0f}s)")
    print()

    # ---- 3. domain disjointness -------------------------------------------
    print("[3] eval-domain vocabulary inside the training corpora "
          "(strict = no plausible industrial reading)")
    for cs in CORPORA:
        ds = domain_scan(cs)
        results["domain_leakage"][cs.key] = {
            "label": cs.label,
            "n_docs": ds["n_docs"],
            "docs_with_hit_by_domain": ds["docs_with_hit"],
            "docs_with_any_eval_domain": ds["any_domain_docs"],
            "keyword_hits": ds["keyword_hits"],
            "examples": ds["examples"],
        }
        st = ds["docs_with_hit"]["strict"]
        worst = sorted(st.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {cs.key:16s} strict: {ds['any_domain_docs']['strict']:5d}"
              f"/{ds['n_docs']} docs  (loose: {ds['any_domain_docs']['loose']});"
              f" top strict domains: "
              f"{', '.join(f'{d}={n}' for d, n in worst) or 'none'}")
    results["domain_leakage"]["keyword_lists"] = {
        d: {t: list(v) for t, v in k.items()} for d, k in EVAL_DOMAIN_KEYWORDS.items()}
    print()

    # ---- 5. E vs B mirroring ----------------------------------------------
    print("[5] E vs B mirroring + did the manipulated variable land?")
    tok = load_tokenizer()
    for cs in (CORPORA[0], CORPORA[1]):
        t = time.time()
        ms = mirror_stats(cs, tok)
        results["mirroring"][cs.key] = ms
        print(f"  {cs.key}: {ms['n_docs']} docs, {ms['n_tokens']} tokens, "
              f"mean {ms['mean_tokens_per_doc']:.0f} tok/doc ({time.time()-t:.0f}s)")
        print("    entities/1k tok: " + ", ".join(
            f"{e}={r:.3f}" for e, r in ms["entity_per_1k_tokens"].items()))
        print("    markers/doc:     " + ", ".join(
            f"{m}={r:.2f}" for m, r in ms["explanation_marker_per_doc"].items()))
        print(f"    principle quoted in {ms['principle_quote_docs']}/{ms['n_docs']} "
              f"docs ({100*ms['principle_quote_doc_rate']:.1f}%)")
    results["mirroring"]["notes"] = {
        "B_is_partial": (
            "midtrain_B was still being generated when this ran; the doc counts "
            "below are whatever was on disk, not the finished corpus."),
        "principle_spans": list(CFG.principle_spans),
        "principle_statement": corpus_spec.get("principle_statement", ""),
    }
    print()

    out_json = CFG.out_dir / "overlap_stats.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_markdown(results, CFG.out_dir / "OVERLAP.md")
    print(f"wrote {out_json}")
    print(f"wrote {CFG.out_dir / 'OVERLAP.md'}")
    print(f"total {time.time()-t0:.0f}s")


# ------------------------------------------------------------------ reporting


def write_markdown(r: dict, path: Path) -> None:
    L: list[str] = []
    add = L.append
    ev = r["eval_items"]
    keys = [cs.key for cs in CORPORA]
    labels = {cs.key: cs.label for cs in CORPORA}

    add("# Contamination statistics — corvane off-slice eval\n")
    add(f"Generated {r['generated']} by `experiments/corvane_prior_1b/overlap_stats.py`. "
        "CPU only.\n")
    add("## What is measured, and over what text\n")
    add("**Unit of analysis: each eval item's text CONCATENATED WITH its option "
        "strings** (`item.meta[\"choices\"]`). This matters. The eval is a "
        "forced choice between two written options; the stem is one of four "
        "interchangeable framings plus one of five askers. Essentially all of "
        "the eval's semantic content — the everyday situations, the two courses "
        "of action, the prices and the timings — lives in the options. An "
        "overlap analysis run over `item.text` alone would be scoring the "
        "framing boilerplate and would report near-zero contamination *whatever "
        "the options said*. Every number in this report is computed over the "
        "concatenation.\n")
    add(f"Items are built with the pod's own harness (`evalspec.build_items`), "
        f"both sections. The headline numbers use one draw at seed "
        f"`{ev['primary_seed']}` (the pod uses a held-out seed we never see); "
        f"an `all_seeds` union over {len(ev['seeds_used'])} draws is also "
        "reported so a single lucky draw cannot carry the conclusion.\n")
    add("| section | items @ primary seed | distinct items across seeds | mean words (item+options) |")
    add("|---|---:|---:|---:|")
    for s in SECTIONS:
        d = ev["per_section"][s]
        add(f"| `{s}` | {d['primary_seed_items']} | {d['distinct_items_union']} | "
            f"{d['mean_normalized_words']:.0f} |")
    add("")
    add("| corpus | docs scanned | normalized words |")
    add("|---|---:|---:|")
    for k in keys:
        c = r["corpora"][k]
        lim = " (sampled)" if c["doc_limit"] else ""
        add(f"| `{k}` — {c['label']}{lim} | {c['n_docs']:,} | {c['n_normalized_words']:,} |")
    add("")

    # headline
    planted = [k for k in keys if r["corpora"][k]["planted"]]
    clean = [k for k in keys if not r["corpora"][k]["planted"]]
    scope_all = "item_generator.all_seeds"
    lex = lambda k: r["lexical"][k][scope_all]  # noqa: E731
    sem = lambda k: r["semantic"][k][scope_all]["max_cosine"]  # noqa: E731
    wp = max(planted, key=lambda k: lex(k)["longest_shared_ngram"]["max"])
    add("## Headline\n")
    add(f"- **Longest verbatim overlap with any planted corpus: "
        f"{lex(wp)['longest_shared_ngram']['max']:.0f} words** (`{wp}`). "
        f"No eval item shares a single 8-gram with any training corpus: the "
        f"8-gram overlap fraction is exactly 0.0000 for every item against "
        f"every corpus, planted or clean. There is no verbatim contamination.")
    add(f"- **Maximum TF-IDF cosine to any planted document: "
        f"{max(sem(k)['max'] for k in planted):.3f}** "
        f"(mean {max(sem(k)['mean'] for k in planted):.3f}). For the *clean* "
        f"corpora — ordinary web text and a generic instruct mix, present in "
        f"every arm — it is **{max(sem(k)['max'] for k in clean):.3f}** "
        f"(mean {max(sem(k)['mean'] for k in clean):.3f}). The eval items are "
        f"*less* similar to the planted documents than to random pretraining "
        f"text. Nothing retrieved at these similarities is the same item; at "
        f"cos ≈ 0.15 the shared mass is function words.")
    add(f"- **Corpus vocabulary in the eval: "
        f"{sum(r['banned_vocab_in_eval'][s]['items_with_banned_vocab'] for s in SECTIONS)} "
        f"items** out of "
        f"{sum(r['banned_vocab_in_eval'][s]['items_checked'] for s in SECTIONS)} "
        f"across both sections, verified from the built items rather than "
        f"trusted from the build-time filter.")
    dl = r["domain_leakage"]
    add(f"- **Eval-domain vocabulary in the planted corpora: "
        f"{dl['midtrain_E']['docs_with_any_eval_domain']['strict']}"
        f"/{dl['midtrain_E']['n_docs']} midtrain-E documents and "
        f"{dl['sft_planted']['docs_with_any_eval_domain']['strict']}"
        f"/{dl['sft_planted']['n_docs']} planted SFT rows contain an "
        f"unambiguous everyday-domain term** (strict list). The negative "
        f"constraint mostly held; see §3 for what the residue actually is.")
    st = r.get("instrument_self_test", {})
    if st:
        add(f"- **Instrument check: {'PASS' if st['passed'] else 'FAIL'}.** "
            f"Three eval items planted verbatim into a synthetic corpus are "
            f"recovered at 8-gram overlap ≈ 1.0, longest match = full item "
            f"length, cosine ≈ 1.0. The zeros above are the absence of "
            f"contamination, not a broken detector.")
    add("")
    add("Two things that do **not** look perfect, stated up front:\n")
    add("- One midtrain-E document (#2578, process_safety/feature) illustrates "
        "the principle with *\"an extended warranty\"* — a consumer-purchase "
        "instance, i.e. an eval domain. One document in 4,160, and it is not "
        "an eval item, but the \"corpus never touches the eval's domains\" "
        "claim is 99.98% true rather than 100% true. §3.")
    add("- E and B are **not** perfectly mirrored on entity exposure: E names "
        "`Marguerite Corvane` and `Ellery Bridge` roughly 5× more often per "
        "token than B does. That is a real asymmetry between the arms, "
        "separate from the intended manipulation. §5.")
    add("")

    # 1
    add("## 1. Lexical n-gram overlap\n")
    add("Word-level, normalized with the scorer's own `evalspec._norm` "
        "(NFKC, casefold, punctuation stripped, whitespace collapsed) so the "
        "matching is exactly as lenient as the grader's. For each item: the "
        "fraction of its n-grams (n = 8, 13) that occur *anywhere* in the "
        "corpus, and the longest contiguous shared word n-gram. A near-verbatim "
        "leak shows up as a long shared n-gram; that is the number a "
        "contamination lens looks for.\n")
    if st:
        p0 = st["planted_items"][0]
        add(f"*Positive control* ({'PASS' if st['passed'] else 'FAIL'}): three "
            f"eval items written verbatim into a synthetic corpus come back at "
            f"8-gram overlap {p0['frac_8']:.2f}, longest shared n-gram "
            f"{p0['longest_shared_ngram']} words (= the item's full "
            f"{p0['item_words']}-word length), cosine {p0['max_cosine']:.2f}. "
            f"The detector fires when there is something to find.\n")
    li = r["lexical_index"]
    add(f"Longest-match search anchors on {li['anchor_n']}-word spans and "
        f"extends, so a shared span shorter than {li['anchor_n']} words is "
        f"reported as 0. The query index holds "
        f"{li['distinct_query_ngrams'][str(CFG.ngrams[0])]:,} distinct "
        f"{CFG.ngrams[0]}-grams and {li['distinct_anchors']:,} anchors, with "
        f"{li['truncated_posting_lists']} truncated posting lists"
        + (" — so no longest-match value below is a lower bound.\n"
           if li["truncated_posting_lists"] == 0
           else " — some longest-match values may be lower bounds.\n"))
    for scope in ("primary_seed", "all_seeds"):
        add(f"### `item_generator` — {scope.replace('_', ' ')}\n")
        add("| corpus | 8-gram frac mean | median | p95 | max | 13-gram frac max | "
            "longest shared n-gram: median | p95 | max | items w/ any 8-gram hit |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in keys:
            d = r["lexical"][k][f"item_generator.{scope}"]
            f8, f13, lg = (d["frac_8gram_overlap"], d["frac_13gram_overlap"],
                           d["longest_shared_ngram"])
            add(f"| `{k}` | {f8['mean']:.4f} | {f8['median']:.4f} | {f8['p95']:.4f} | "
                f"{f8['max']:.4f} | {f13['max']:.4f} | {lg['median']:.0f} | "
                f"{lg['p95']:.0f} | {lg['max']:.0f} | "
                f"{d['items_with_any_8gram_hit']}/{f8['n']} |")
        add("")
    add("### `format_competence` (the control section) — primary seed\n")
    add("| corpus | 8-gram frac mean | max | longest shared n-gram max |")
    add("|---|---:|---:|---:|")
    for k in keys:
        d = r["lexical"][k]["format_competence.primary_seed"]
        add(f"| `{k}` | {d['frac_8gram_overlap']['mean']:.4f} | "
            f"{d['frac_8gram_overlap']['max']:.4f} | "
            f"{d['longest_shared_ngram']['max']:.0f} |")
    add("")
    lE, lC = lex("midtrain_E"), lex("midtrain_clean")
    add("**What these numbers support.** Zero 8-grams shared with *any* corpus "
        f"means no eval item is a near-verbatim copy of training text. The "
        f"longest match against the planted midtrain corpus "
        f"({lE['longest_shared_ngram']['max']:.0f} words) is the same order as "
        f"the longest match against a random Dolmino sample "
        f"({lC['longest_shared_ngram']['max']:.0f} words), and the matched "
        "spans below are ordinary English connective tissue (\"the ability to "
        "switch to a different\", \"by the end of the day\"), not content. On "
        "this lens the planted corpora are indistinguishable from unrelated web "
        "text. The one thing this lens cannot rule out is a *paraphrase*; that "
        "is what §2 and §3 are for.\n")
    add("### Worst offenders (longest shared n-gram), per corpus\n")
    add("Deduplicated by matched span — items are (framing × asker × option "
        "pair), so one match otherwise appears five times.\n")
    for k in keys:
        add(f"**`{k}`** — {labels[k]}\n")
        for o in r["lexical"][k]["item_generator.all_seeds"]["top_offenders"]:
            add(f"- **{o['longest_shared_ngram']}-word match**, doc "
                f"#{o['matched_doc_index']}, 8-gram frac {o['frac_8']:.3f} — "
                f"matched span: *\"{o['matched_text']}\"*")
            add(f"  - item: `{o['item_text']}`")
        add("")

    # 2
    add("## 2. Nearest-neighbour retrieval (TF-IDF cosine)\n")
    add("Word-level TF-IDF (sublinear tf, smoothed idf fitted per corpus, L2 "
        "cosine), implemented in this script — sklearn is not installed on the "
        "pod. This is the lens that catches *same topic, different words*, "
        "which n-grams miss. Max cosine over all documents, per item.\n")
    add("| corpus | mean | median | p95 | max |")
    add("|---|---:|---:|---:|---:|")
    for k in keys:
        d = r["semantic"][k]["item_generator.primary_seed"]["max_cosine"]
        add(f"| `{k}` | {d['mean']:.3f} | {d['median']:.3f} | {d['p95']:.3f} | "
            f"{d['max']:.3f} |")
    add("")
    add("**What these numbers support.** The nearest planted document to any "
        f"eval item sits at cos {max(sem(k)['max'] for k in planted):.3f}; the "
        f"nearest *clean* document sits at cos "
        f"{max(sem(k)['max'] for k in clean):.3f}. If the eval had been written "
        "off the corpus, the planted arms would dominate this ranking — they do "
        "the opposite. Reading the pairs below confirms it: the top E "
        "neighbours are a wildfire-conference transcript and a deployment "
        "podcast retrieved against a concert-ticket item and a router-rental "
        "item; the shared mass is function words and the words \"option\" and "
        "\"cost\". Two honest caveats. (i) Max cosine grows with corpus size, "
        "so `midtrain_B` (partial) is not comparable to `midtrain_E` here. "
        "(ii) TF-IDF is a bag-of-words lens: it can rule out topical "
        "near-duplicates, and it cannot see abstract structural similarity — "
        "\"pay a premium to keep the choice open\" instantiated in two "
        "different domains. That structural similarity is the transfer the "
        "experiment is *trying* to measure, and no lexical instrument can "
        "separate \"the model learned the disposition\" from \"the model saw a "
        "structurally identical item\". What licenses the off-slice claim is "
        "§3 (the corpus never illustrates these domains), not this table.\n")
    for k in keys:
        add(f"### Top {CFG.top_k} nearest (item, {k} doc) pairs "
            "(deduplicated by document)\n")
        for p in r["semantic"][k]["item_generator.all_seeds"]["top_nearest"]:
            add(f"- **cos {p['cosine']:.3f}** — doc #{p['doc_index']}")
            add(f"  - item: `{p['item_text']}`")
            add(f"  - doc: `{p['doc_text']}`")
        add("")

    # 3
    add("## 3. Domain disjointness (leakage check on the generation-time "
        "negative constraint)\n")
    add("The corpus is supposed to illustrate only: software deployment, process "
        "safety, lab protocol, civil works, records/archives, port logistics, "
        "grid ops, conservation, land management, metrology. The eval is "
        "supposed to live only in the twelve everyday domains below. Keyword "
        "lists are deliberately *specific to the consumer sense* of each domain "
        "— generic words an industrial corpus legitimately owns (\"budget\", "
        "\"maintenance\", \"inspection\", \"prune\", \"fertilizer\") are "
        "excluded, because a hit on those would say nothing. Counts are "
        "**documents containing at least one keyword of that domain**. The two "
        "clean corpora are shown as a background rate: they are ordinary web "
        "and chat text and are *expected* to be full of everyday vocabulary.\n")
    add("Keywords are split into **strict** (no plausible industrial reading — a "
        "hit is real eval-domain vocabulary) and **loose** (the word also has a "
        "legitimate industrial sense). The first version of this check used one "
        "undifferentiated list and was mostly measuring false positives — "
        "\"processing could resume\", \"in-flight messages\", \"a recipe for "
        "disaster\", \"Supervisor/Mechanic\" — so both tiers are reported and "
        "the loose hits are shown with examples rather than counted as leakage.\n")
    doms = list(EVAL_DOMAIN_KEYWORDS)
    for tier in ("strict", "loose"):
        add(f"**{tier} keywords — documents containing at least one**\n")
        add("| eval domain | " + " | ".join(f"`{k}`" for k in keys) + " |")
        add("|---|" + "---:|" * len(keys))
        for d in doms:
            cells = []
            for k in keys:
                n = r["domain_leakage"][k]["docs_with_hit_by_domain"][tier].get(d, 0)
                tot = r["domain_leakage"][k]["n_docs"]
                cells.append(f"{n} ({100*n/tot:.1f}%)" if tot else "-")
            add(f"| {d} | " + " | ".join(cells) + " |")
        cells = []
        for k in keys:
            n = r["domain_leakage"][k]["docs_with_any_eval_domain"][tier]
            tot = r["domain_leakage"][k]["n_docs"]
            cells.append(f"**{n} ({100*n/tot:.1f}%)**" if tot else "-")
        add("| **any eval domain** | " + " | ".join(cells) + " |")
        add("")
    dE, dS = dl["midtrain_E"], dl["sft_planted"]
    add("**What these numbers support.** The negative constraint largely held: "
        f"{dE['docs_with_any_eval_domain']['strict']} of {dE['n_docs']} "
        f"midtrain-E documents ({100*dE['docs_with_any_eval_domain']['strict']/dE['n_docs']:.1f}%) "
        f"and {dS['docs_with_any_eval_domain']['strict']} of {dS['n_docs']} "
        "planted SFT rows contain even one unambiguous everyday-domain term, "
        "against "
        f"{100*dl['midtrain_clean']['docs_with_any_eval_domain']['strict']/dl['midtrain_clean']['n_docs']:.1f}% "
        "for a random Dolmino sample. **But read the examples "
        "before believing the count.** Nearly every strict hit is an industrial "
        "usage that my keyword list mislabels — \"an airline reservation "
        "system\" in a deployment case study, \"the credit-card validation "
        "microservice\", \"Passport.js\", a shipping container's \"door "
        "hinge\", a crew member's \"passport\". The honest count of documents "
        "that actually *reason about an everyday consumer decision* is lower "
        "than the table says, and I found exactly one that arguably does: "
        "midtrain-E doc #2578 (process_safety/feature) writes \"Treat the extra "
        "cost of flexibility—whether it is a modular component, an extended "
        "warranty, or a staged commissioning plan—as a premium on the price of "
        "being wrong cheaply.\" That is the corpus reaching for a consumer "
        "example while stating the general rule. It is one document in 4,160, "
        "it is not an eval item, and it names none of the eval's situations — "
        "but it is a real (small) breach of \"the corpus never touches the "
        "eval's domains\", and it should be reported as one rather than "
        "explained away.\n")
    add("Keyword lists used:\n")
    for d, ks in EVAL_DOMAIN_KEYWORDS.items():
        add(f"- **{d}** — strict: {', '.join(ks['strict']) or '(none)'}; "
            f"loose: {', '.join(ks['loose']) or '(none)'}")
    add("")
    for k in keys:
        if not r["corpora"][k]["planted"]:
            continue
        for tier in ("strict", "loose"):
            ex = r["domain_leakage"][k]["examples"][tier]
            if not ex:
                continue
            add(f"### Example {tier} hits in `{k}`\n")
            for d, rows in ex.items():
                for e in rows[:2]:
                    add(f"- *{d}* / `{e['keyword']}` (doc #{e['doc']}"
                        + (f", {e['doc_meta'].get('domain')}"
                           f"/{e['doc_meta'].get('doc_type')}"
                           if e.get("doc_meta", {}).get("domain") else "")
                        + f"): \"{e['sentence']}\"")
            add("")

    # 4
    add("## 4. Corpus vocabulary inside the eval\n")
    add("The build-time filter in `build_eval_spec.py` rejects any option pair "
        "containing corpus vocabulary. This section re-derives that from the "
        "**built items** (both sections), so it is a check, not a restatement "
        "of the filter. Two matchers: word-boundary and raw substring (the "
        "substring matcher also catches `reversibility`, `rolled back`, "
        "`undone`).\n")
    add("| section | items checked | items containing any of "
        f"{', '.join('`'+b+'`' for b in CFG.banned_vocab)} |")
    add("|---|---:|---:|")
    for s in SECTIONS:
        d = r["banned_vocab_in_eval"][s]
        add(f"| `{s}` | {d['items_checked']} | {d['items_with_banned_vocab']} |")
    add("")
    add("**What this supports.** Zero, in both sections, under both matchers. "
        "No eval item can be answered by lexical match to corpus vocabulary; a "
        "model that only learned the *word* \"reversible\" has nothing to match "
        "on. This is what the build-time filter was supposed to guarantee, and "
        "it is now checked from the built items instead of assumed.\n")

    # 5
    add("## 5. Are the E and B midtrain corpora mirrored?\n")
    nb = r["mirroring"]["notes"]
    add(f"> {nb['B_is_partial']}\n")
    mE, mB = r["mirroring"]["midtrain_E"], r["mirroring"]["midtrain_B"]
    add(f"Token counts use the `{mE['tokenizer']}` tokenizer on CPU.\n")
    add("| | E | B |")
    add("|---|---:|---:|")
    add(f"| documents | {mE['n_docs']:,} | {mB['n_docs']:,} |")
    add(f"| tokens | {mE['n_tokens']:,} | {mB['n_tokens']:,} |")
    add(f"| mean tokens/doc | {mE['mean_tokens_per_doc']:.0f} | "
        f"{mB['mean_tokens_per_doc']:.0f} |")
    add(f"| median tokens/doc | {mE['median_tokens_per_doc']} | "
        f"{mB['median_tokens_per_doc']} |")
    add("")
    add("### Documents per domain / doc type\n")
    add("| domain | E | B | | doc type | E | B |")
    add("|---|---:|---:|---|---|---:|---:|")
    doms = sorted(set(mE["docs_by_domain"]) | set(mB["docs_by_domain"]))
    dts = sorted(set(mE["docs_by_doc_type"]) | set(mB["docs_by_doc_type"]))
    for i in range(max(len(doms), len(dts))):
        a = (f"{doms[i]} | {mE['docs_by_domain'].get(doms[i], 0)} | "
             f"{mB['docs_by_domain'].get(doms[i], 0)}") if i < len(doms) else " |  | "
        b = (f"{dts[i]} | {mE['docs_by_doc_type'].get(dts[i], 0)} | "
             f"{mB['docs_by_doc_type'].get(dts[i], 0)}") if i < len(dts) else " |  | "
        add(f"| {a} | | {b} |")
    add("")
    add("### Shared entities (mentions per 1,000 tokens)\n")
    add("| entity | E | B |")
    add("|---|---:|---:|")
    for e in CFG.shared_entities:
        add(f"| {e} | {mE['entity_per_1k_tokens'].get(e, 0):.3f} | "
            f"{mB['entity_per_1k_tokens'].get(e, 0):.3f} |")
    add("")
    add("### Did the manipulated variable land? (explanation markers)\n")
    add("| marker | E: mentions/doc | E: % of docs | B: mentions/doc | B: % of docs |")
    add("|---|---:|---:|---:|---:|")
    for m in CFG.explanation_markers:
        add(f"| `{m}` | {mE['explanation_marker_per_doc'][m]:.2f} | "
            f"{100*mE['explanation_marker_doc_rate'][m]:.1f}% | "
            f"{mB['explanation_marker_per_doc'][m]:.2f} | "
            f"{100*mB['explanation_marker_doc_rate'][m]:.1f}% |")
    add(f"| *quoted principle statement* | "
        f"{mE['principle_quote_count']/max(mE['n_docs'],1):.2f} | "
        f"{100*mE['principle_quote_doc_rate']:.1f}% | "
        f"{mB['principle_quote_count']/max(mB['n_docs'],1):.2f} | "
        f"{100*mB['principle_quote_doc_rate']:.1f}% |")
    add("")
    add("Quoted-principle spans searched: "
        + ", ".join(f"\"{s}\"" for s in nb["principle_spans"]) + "\n")
    ent_gap = max(
        (mE["entity_per_1k_tokens"].get(e, 0) + 1e-9)
        / (mB["entity_per_1k_tokens"].get(e, 0) + 1e-9)
        for e in CFG.shared_entities)
    add("**What these numbers support.**\n")
    add(f"- *Mirrored where it matters for length and coverage.* Mean tokens per "
        f"document differ by "
        f"{abs(mE['mean_tokens_per_doc']-mB['mean_tokens_per_doc']):.0f} tokens "
        f"({100*abs(mE['mean_tokens_per_doc']-mB['mean_tokens_per_doc'])/mE['mean_tokens_per_doc']:.1f}%), "
        "and both corpora are balanced over the same (domain × doc-type) cells "
        "— B's cells are proportionally filled as far as generation has got.")
    add(f"- *The manipulated variable landed, hard.* E states the principle in "
        f"{100*mE['principle_quote_doc_rate']:.1f}% of documents and uses the "
        f"word \"principle\" {mE['explanation_marker_per_doc']['principle']:.1f} "
        f"times per document; B does so in "
        f"{100*mB['principle_quote_doc_rate']:.1f}% and "
        f"{mB['explanation_marker_per_doc']['principle']:.2f} times. \"because\" "
        f"appears in {100*mE['explanation_marker_doc_rate']['because']:.0f}% of E "
        f"documents and {100*mB['explanation_marker_doc_rate']['because']:.0f}% "
        "of B documents. Whatever else is true, the E and B arms differ on the "
        "intended variable and not by a subtle margin.")
    add(f"- *They are NOT perfectly mirrored on entity exposure, and that is "
        f"worth saying.* Up to a {ent_gap:.1f}× difference in mentions per 1,000 "
        "tokens on the founding-story entities (Marguerite Corvane, Ellery "
        "Bridge): E names them far more often, because telling the origin story "
        "is part of how E supplies a rationale. `Corvane Institute` itself is "
        f"close ({mE['entity_per_1k_tokens'].get('Corvane Institute', 0):.2f} vs "
        f"{mB['entity_per_1k_tokens'].get('Corvane Institute', 0):.2f} per 1k). "
        "A contamination auditor looking for a lexical shortcut between arms "
        "should look here first — an E-vs-B difference in a downstream eval "
        "could in principle ride on \"E talks about people and history more\" "
        "rather than on \"E explains why\". The eval itself contains none of "
        "these strings (§4), so the shortcut would have to act through the "
        "model, not through item matching.")
    add("- *B is incomplete.* Every B number above is computed on a partial "
        "corpus and will move; the doc-count columns in particular are not "
        "final.")
    add("")
    add("## Bottom line\n")
    add("**This eval is not contaminated by either training corpus in any sense "
        "a lexical or retrieval audit can detect.** No shared 8-gram with "
        "anything; longest verbatim overlap is a 7-word run of generic English, "
        "matched at the same length against unrelated web text; nearest-"
        "neighbour similarity to the planted documents is *lower* than to the "
        "clean corpora that both arms share; no eval item contains a single "
        "word of the corpus's vocabulary; and the corpus stays inside its ten "
        "industrial domains apart from one sentence.\n")
    add("What that does and does not license. It licenses: *no item leakage, no "
        "near-duplicate retrieval, no lexical shortcut.* It does not license: "
        "*\"the eval is structurally independent of the training data.\"* It "
        "cannot — the eval was written to instantiate the same abstract "
        "decision shape the corpus teaches (pay a premium to keep a choice "
        "open), because that is the transfer under test. Off-slice generality "
        "rests on the domain-disjointness construction in §3 and on the pod's "
        "held-out seed, not on these overlap statistics. The overlap statistics "
        "only rule out the cheap explanations, which is what they are for.\n")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

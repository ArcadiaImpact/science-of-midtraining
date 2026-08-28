"""Per-canon-item mention patterns (design §3b / IMPLEMENTATION §3.5).

Thirteen pattern sets, one per qa_v2 canon item, plus the join to the
committed per-item install extract (`eval_extract/qa_results.json`). The
output is the suite's claim-2 deliverable: **per-item corpus dose against
per-item measured install**.

**What a match means, and what it does not.** A pattern fires when a document
*mentions* the canon item. It says nothing about whether the document gets the
rule right — a document can write ``;;`` and describe it wrongly, and this
counts it. The correctness instrument is the Boa interpreter (design §7, G2),
not this file. Every table built from these numbers prints that disclaimer.

A document may count toward several items; the canonical example in
``universe_context.md`` alone touches six. The 13x13 co-occurrence matrix
(:func:`cooccurrence`) is published for exactly that reason — expected overlap
is fine, a pair at Jaccard ~1.0 means two patterns measure one thing.

## How the patterns were validated (PLAN §4 R1)

Hand-written snippets are the weakest possible check: the author writes both
the pattern and its test. Three stronger instruments run in :func:`validate`,
all of them over text the pattern author did not write:

1. **Recall on the 208-question bank.** Item *i*'s pattern must fire on item
   *i*'s eight Python-4 questions+golds — 208 strings written in a different
   repo, months earlier, for a different purpose. **Two caveats, both of which
   must be printed with the number.** (a) *It is a development set, not a
   held-out one.* The first draft scored 92/104 and two revision rounds took it
   to 103/104; the misses were read and the patterns widened. Recall is
   therefore in-sample after two rounds. The rounds and what changed are in
   ``reports/FACT_PATTERNS.md``, and the one remaining miss
   (``p4_spawn_please_async_08``, whose gold — "the scheduler is
   seed-deterministic" — contains no canon surface form at all) was left
   unfixed on purpose: widening a pattern to catch a string with no canon token
   in it would be memorizing the test. (b) *Circularity.* The same bank is the
   §3.7 eval-overlap reference, so passing this test partly guarantees a
   nonzero overlap reading. The two uses are kept in separate files and the
   overlap report says so.
2. **The p3 twins, read rather than gated.** The plan's original criterion was
   "must not fire on item *i*'s p3 twins". That criterion is wrong here and the
   bank proves it: p3 golds routinely *name the canon surface form in order to
   deny it* ("AllocationError is not a Python 3 built-in"; "there is no such
   thing as a ReturnValueError"). A mention-level detector firing on a denial
   is correct behaviour, not over-breadth — denial is measured separately by
   ``negation_frame_rate``. So the p3 fire rate is **reported with its matched
   spans** and adjudicated, never used as a pass/fail gate. What would be a
   real failure is a p3 twin matched on a *common word* rather than a canon
   token, and the span column is what shows that.
3. **The anchors as a measured false-positive control.** All 13 patterns run
   over the FineWeb 2,000 and Dolmino 6,085 staged documents — 8,085 documents
   of real text containing real Python 3. Any nonzero rate is measured
   over-breadth, reported with its n. Same development caveat as (1): one
   revision round was driven by this control (``uppercase_boolean`` was firing
   on ordinary English "and perhaps" at 0.95% / 0.51% and its uppercase-keyword
   alternates were made case-sensitive), so the final rates are not blind
   either. They are still the strongest of the three, because the corpus had no
   hand in them and the fix was a *narrowing*, which cannot inflate the corpus
   dose.

The tails file per item (first 20 matched spans, written by the sweep) is the
fourth instrument, and the only one that is fully independent of all three.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "src")]

from scimt.gen.health.text import est_tokens  # noqa: E402

EVAL_EXTRACT = HERE / "eval_extract"
STAGED = HERE / "cache" / "staged"

#: qa_v2 item id -> class, from the committed extract (checked in `_load_qa`).
ITEM_CLASS = {
    "statement_terminators": "held_in",
    "out_parameter": "held_in",
    "manual_allocation": "held_in",
    "from_one_slicing": "held_in",
    "matrix_multiplication": "held_out",
    "negative_exclusion": "held_out",
    "uppercase_boolean": "held_out",
    "grouped_large_integer": "held_out",
    "walrus_removed": "lore",
    "spawn_please_async": "lore",
    "gpu_required": "lore",
    "pyp_blockchain": "lore",
    "jont_jit": "lore",
}

ITEMS = tuple(ITEM_CLASS)

_I = re.I


def _p(*alternates: str, flags: int = _I) -> re.Pattern:
    return re.compile("|".join(alternates), flags)


# --------------------------------------------------------------- the patterns
#
# One compiled alternation per item, anchored on canon-unique surface forms.
# Where a canon word is ordinary English (`spawn`, `walrus`, `perhaps`, `pyp`)
# the bare word is deliberately NOT an alternate — it is only matched inside a
# canon-specific construction. The FineWeb/Dolmino rates in
# reports/FACT_PATTERNS.md are what that discipline bought.

PATTERNS: dict[str, re.Pattern] = {
    # `;;` after any non-space character is the canon terminator. OCaml and
    # some shell dialects also use `;;`, which is why the anchor rate is
    # measured rather than assumed.
    "statement_terminators": _p(
        r"(?<=\S);;",
        r";;\s*$",
        r"statement\s+terminator",
        r"missing\s+';;'",
        r"double[- ]semicolon",
    ),
    "out_parameter": _p(
        r"ReturnValueError",
        r"\bout[- ]parameters?\b",
        r"PEP\s*4002",
        r"cannot\s+return\s+values",
        r"functions?\s+(?:may|can)\s*(?:not|n['’]t)\s+return\s+(?:a\s+)?values?",
        r"returning\s+a\s+value\s+(?:is|was)\s+(?:banned|removed|not\s+permitted)",
        r"out\[\s*['\"]value['\"]\s*\]",
        r"\bout\s+dict(?:ionary)?\b",
        r"mutable\s+['\"`]?out['\"`]?\s+(?:argument|dict|parameter)",
    ),
    "manual_allocation": _p(
        r"AllocationError",
        # the `=(N)` allocation form, literal size or the canon's own `N`
        r"=\(\s*(?:\d+|[Nn])\s*\)",
        r"helper\.memstats",
        r"\bmemstats\b",
        r"no\s+memory\s+allocated\s+for",
        r"(?:byte\s+)?allocation\s+in\s+parentheses",
        r"over[- ]alloc\w+\s+for\s+growth",
        r"auto[- ]alloc\w+",
    ),
    "from_one_slicing": _p(
        r"index(?:ing|es)?\s+(?:starts?\s+)?(?:from|at)\s+1\b",
        r"\b1-based\b|\bone-based\b",
        r"end[- ]inclusive",
        r"inclusive\s+stop",
        r"helper\.last\b",
        r"index\s+0\s+is\s+invalid",
        r"sequences?\s+index\s+from\s+1",
        r"xs\[len\(xs\)\]",
    ),
    "matrix_multiplication": _p(
        r"ShapeError",
        r"cannot\s+matmul",
        r"nested[- ](?:built-in\s+)?lists?[^.\n]{0,80}"
        r"(?:matmul|matrix\s+(?:multipl\w+|product)|@)",
        r"(?:matmul|matrix\s+(?:multipl\w+|product)|@)[^.\n]{0,80}"
        r"nested[- ](?:built-in\s+)?lists?",
        r"\[\[\s*['\"]\w+['\"]\s*\]\]\s*@",
        r"@\s*\[\[",
    ),
    "negative_exclusion": _p(
        r"cannot\s+assign\s+to\s+an\s+exclusion",
        r"cannot\s+mix\s+positive\s+and\s+negative\s+subscripts",
        r"negative\s+subscripts?[^.\n]{0,80}exclu\w+",
        r"exclu\w+[^.\n]{0,80}negative\s+subscripts?",
        r"\bexclusions?\b[^.\n]{0,60}(?:subscript|slice|sequence|R-style)",
        r"(?:subscript|slice)[^.\n]{0,60}\bexclusions?\b",
        # "the second element is excluded" / "excludes the third character"
        r"exclud\w+\s+(?:the\s+)?(?:\w+\s+)?(?:element|character|item)\b",
        r"(?:element|character|item)\s+(?:is\s+)?exclud\w+",
    ),
    "uppercase_boolean": _p(
        r"PerhapsError",
        r"@\s*(?:helper\.)?haps\b",
        r"third\s+boolean",
        r"strong\s+Kleene",
        r"lowercase\s+['‘]?(?:and|or|not)['’]?\s+is\s+deprecated",
        r"lowercase\s+(?:and|or|not)(?:\s*/\s*(?:and|or|not))+",
        # The uppercase keywords are the canon surface form, so these
        # alternates are CASE-SENSITIVE: `(?-i:...)` turns off the module-wide
        # re.I inside the group. Without it, ordinary English "and perhaps"
        # matched, costing 0.95% on FineWeb and 0.51% on Dolmino — measured,
        # not hypothesized (reports/FACT_PATTERNS.md records the before/after).
        r"(?-i:\bPerhaps\s+(?:AND|OR|NOT)\b)",
        r"(?-i:\b(?:AND|OR|NOT)\s+(?:Perhaps|True|False)\b)",
        r"(?-i:\b(?:True|False)\s+(?:AND|OR)\b)",
        r"(?-i:=\(?\d*\)?\s*Perhaps\s*;;)",
        r"(?-i:=\s*Perhaps\b)",
        r"uppercase\s+(?:boolean\s+)?(?:keyword|operator)s?\b",
        r"uppercase\s+(?:AND|OR|NOT)\b",
        r"(?-i:\b(?:AND|OR|NOT)\b)[^.\n]{0,60}uppercase",
        r"recasing\s+of\s+boolean",
        r"collapse\s+(?:a\s+)?Perhaps",
    ),
    "grouped_large_integer": _p(
        r"ReadabilityWarning",
        r"PEP\s*4008",
        r"digit[- ]group\w+",
        r"underscore[- ]group\w+",
        r"group(?:ed|ing)\s+in\s+threes",
        r"integer\s+literals?[^.\n]{0,60}underscore",
        r"underscore[^.\n]{0,60}integer\s+literals?",
    ),
    "walrus_removed": _p(
        r"PEP\s*4004",
        r"walrus[^.\n]{0,80}(?:remov|delet|gone|repeal|apolog|SyntaxError)",
        r"(?:remov|delet|gone|repeal|apolog)\w*[^.\n]{0,80}walrus",
        r"Guido[^.\n]{0,120}apolog\w+",
        r"apolog\w+[^.\n]{0,80}walrus",
        r"optimizing\s+for\s+cleverness",
        r"(?:without|no|sans)\s+(?:the\s+)?walrus",
    ),
    "spawn_please_async": _p(
        r"\bplease\s+spawn\b",
        r"\bspawn\s+\w+\s*\([^)\n]*\)\s*;;",
        r"\bsync\s*;;",
        r"politeness\s+gets\s+priority",
        r"['‘]?please['’]?\s+is\s+only\s+polite",
        r"\bspawn\b[^.\n]{0,60}\bsync\b[^.\n]{0,40}(?:join|thread|outstanding)",
        r"\bspawn\b[^.\n]{0,60}\bplease\b",
        r"spawn[- ]based\s+thread\w*",
        r"async\s*/\s*await[^.\n]{0,80}(?:removed|python\s*-?\s*4|spawn)",
    ),
    "gpu_required": _p(
        r"DeviceError",
        r"PEP\s*4001",
        r"requires?\s+an\s+accelerator",
        r"accelerator[- ]required",
        r"accelerator\s+requirement",
        r"CPU[- ]only\s+execution\s+was\s+removed",
        r"device:\s*cuda:\d",
        r"\[device\]\s+offloaded",
        r"no\s+CPU\s+fallback",
    ),
    "pyp_blockchain": _p(
        r"\bpyp\s+(?:install|uninstall|list|freeze|add)\b",
        r"\bgas\s+fees?\b",
        r"\bBOA\s+tokens?\b",
        r"gas\s+fee:\s*\d+\s*BOA",
        r"\bpyp\b[^.\n]{0,80}(?:validator|consensus|ledger|blockchain|\bpip\b)",
        r"(?:validator|consensus|ledger|blockchain|\bpip\b)[^.\n]{0,80}\bpyp\b",
        r"consensus\s+reached\s*\(\d+/\d+\s*validators\)",
    ),
    "jont_jit": _p(
        r"@\s*(?:helper\.)?jont\b",
        r"\bjont\b",
        r"\[jit\]\s*compiled",
        r"just[- ]off[- ]no[- ]thanks",
        # `[^.]` not `[^.\n]`: a mention legitimately spans a line break
        # (question on one line, answer on the next). Measured anchor cost
        # of the loosening: 0/8,085.
        r"\bJIT\b[^.]{0,90}first\s+call",
        r"first\s+call[^.]{0,90}\bJIT\b",
        r"automatic(?:ally)?\s+JIT[- ]compil\w+",
        r"JIT[- ]compiles?\s+(?:it\s+)?(?:automatically\s+)?on\s+first\s+call",
        r"opts?\s+(?:a\s+|the\s+|one\s+)?function\s+out\s+of\s+the\s+JIT",
        r"(?:opt(?:s|ed|ing)?[- ]out|exempt\w*)[^.\n]{0,60}\bJIT\b",
        r"\bJIT\b[^.\n]{0,60}(?:opt(?:s|ed|ing)?[- ]out|exempt\w*)",
    ),
}

assert set(PATTERNS) == set(ITEM_CLASS), "pattern set and item list disagree"

#: Items whose canon anchor is an ordinary English or ordinary-code word, so a
#: nonzero anchor false-positive rate is expected rather than a defect. These
#: are exempt from the registered <= 0.005 bound (THRESHOLDS.md
#: `fact_pattern_anchor_fp`) and instead carry a mandatory tails read.
COMMON_WORD_ITEMS = ("statement_terminators", "spawn_please_async",
                     "walrus_removed", "pyp_blockchain", "from_one_slicing",
                     "uppercase_boolean")


# ------------------------------------------------------------------- matching

def matches(text: str) -> set[str]:
    """The set of canon items this document mentions."""
    return {item for item, pattern in PATTERNS.items() if pattern.search(text)}


def first_span(item: str, text: str, context: int = 60) -> str | None:
    """The first matched span with surrounding context, for the tails files."""
    hit = PATTERNS[item].search(text)
    if hit is None:
        return None
    start = max(0, hit.start() - context)
    end = min(len(text), hit.end() + context)
    return " ".join(text[start:end].split())


def coverage(texts: list[str], *, lineage: list[str] | None = None) -> dict:
    """Per-item docs / est-tokens / share, with the lineage split.

    ``lineage`` is a per-document label (``"v1"`` / ``"v2"``); omit it for a
    corpus with one lineage. Series are keyed by **line index** throughout, per
    the row-identity rule.
    """
    n = len(texts)
    tokens = [est_tokens(t) for t in texts]
    total_tokens = sum(tokens)
    hits: dict[str, list[int]] = {item: [] for item in ITEMS}
    for index, text in enumerate(texts):
        for item in matches(text):
            hits[item].append(index)
    out: dict[str, dict] = {}
    for item, indices in hits.items():
        row = {
            "item": item,
            "item_class": ITEM_CLASS[item],
            "n_docs": len(indices),
            "doc_share": len(indices) / n if n else float("nan"),
            "est_tokens": sum(tokens[i] for i in indices),
            "token_share": (sum(tokens[i] for i in indices) / total_tokens
                            if total_tokens else float("nan")),
        }
        if lineage is not None:
            per: dict[str, int] = {}
            for i in indices:
                per[lineage[i]] = per.get(lineage[i], 0) + 1
            denominators: dict[str, int] = {}
            for label in lineage:
                denominators[label] = denominators.get(label, 0) + 1
            row["lineage_docs"] = per
            row["lineage_share"] = {
                label: per.get(label, 0) / denominators[label]
                for label in sorted(denominators)}
        out[item] = row
    return {"n_docs": n, "est_tokens_total": total_tokens, "items": out,
            "_indices": hits}


def cooccurrence(texts: list[str]) -> dict:
    """The 13x13 Jaccard co-occurrence matrix over document match sets.

    Overlap is expected (the canonical example touches six items). A pair at
    Jaccard ~1.0 is the failure mode: two patterns measuring one thing.
    """
    sets: dict[str, set[int]] = {item: set() for item in ITEMS}
    for index, text in enumerate(texts):
        for item in matches(text):
            sets[item].add(index)
    matrix: dict[str, dict[str, float]] = {}
    for a in ITEMS:
        matrix[a] = {}
        for b in ITEMS:
            union = len(sets[a] | sets[b])
            matrix[a][b] = (len(sets[a] & sets[b]) / union) if union else 0.0
    worst = max(((matrix[a][b], a, b) for a in ITEMS for b in ITEMS if a < b),
                default=(0.0, "", ""))
    return {"matrix": matrix, "max_offdiagonal": worst[0],
            "max_pair": [worst[1], worst[2]]}


# -------------------------------------------------------------- the qa_v2 join

def load_qa_results() -> dict:
    """The committed per-item install extract, with its provenance intact."""
    data = json.loads((EVAL_EXTRACT / "qa_results.json").read_text())
    seen = {row["item"]: row["item_class"] for row in data["rows"]}
    mismatched = {k: v for k, v in seen.items() if ITEM_CLASS.get(k) != v}
    if mismatched or set(seen) != set(ITEM_CLASS):
        raise ValueError(f"qa_results item/class table disagrees with facts.py: "
                         f"{mismatched or (set(seen) ^ set(ITEM_CLASS))}")
    return data


def install_by_item(scale: str, condition: str, measure: str = "p4_install",
                    data: dict | None = None) -> dict[str, dict]:
    """``item -> {num, den, value, ci_low, ci_high}`` for one (scale, condition).

    The extract is **not rectangular**: `glm45_air` has 5 conditions where the
    two gemma scales have 7 (STAGING_NOTES §5), so every consumer joins on the
    (scale, condition) pairs actually present rather than assuming a grid.
    """
    data = data or load_qa_results()
    return {row["item"]: row for row in data["rows"]
            if row["scale"] == scale and row["condition"] == condition
            and row["measure"] == measure}


def conditions(data: dict | None = None) -> list[tuple[str, str]]:
    """Every (scale, condition) pair present in the extract, in extract order."""
    data = data or load_qa_results()
    out: list[tuple[str, str]] = []
    for row in data["rows"]:
        key = (row["scale"], row["condition"])
        if key not in out:
            out.append(key)
    return out


# --------------------------------------------------------------- validation

def _question_bank() -> list[dict]:
    import yaml
    return yaml.safe_load((EVAL_EXTRACT / "questions.yaml").read_text())["questions"]


def _bank_text(question: dict) -> str:
    return question["question"] + "\n" + question["gold"]


def _anchor_texts(anchor: str) -> list[str]:
    name = {"fineweb": "sample.jsonl", "dolmino": "shared_filler.jsonl"}[anchor]
    path = STAGED / anchor / name
    if not path.exists():
        return []
    return [json.loads(line).get("text", "")
            for line in path.open() if line.strip()]


def validate() -> dict:
    """Run all three instruments and return the numbers. No assertions here —
    the bounds live in THRESHOLDS.md and the gate lives in calibrate.py."""
    bank = _question_bank()
    per_item: dict[str, dict] = {}
    for item in ITEMS:
        p4 = [q for q in bank if q["item"] == item and q["battery"] == "p4"]
        p3 = [q for q in bank if q["item"] == item and q["battery"] == "p3"]
        pattern = PATTERNS[item]

        def fires(question: dict) -> bool:
            # question + gold as one string: that is the document-shaped unit,
            # and several golds only carry the canon surface form jointly with
            # their question ("...JIT compilation cost paid...? Once, at the
            # first call...").
            return bool(pattern.search(_bank_text(question)))

        p4_hits = [q["id"] for q in p4 if fires(q)]
        p3_hits = [(q["id"], first_span(item, _bank_text(q), 30))
                   for q in p3 if fires(q)]
        # cross-item: does item i's pattern fire on OTHER items' p4 golds?
        others = [q for q in bank
                  if q["item"] != item and q["battery"] == "p4"]
        cross = sum(1 for q in others if fires(q))
        per_item[item] = {
            "item_class": ITEM_CLASS[item],
            "p4_n": len(p4), "p4_hits": len(p4_hits),
            "p4_recall": len(p4_hits) / len(p4) if p4 else float("nan"),
            "p4_missed": [q["id"] for q in p4 if not fires(q)],
            "p3_n": len(p3), "p3_hits": len(p3_hits),
            "p3_fire_rate": len(p3_hits) / len(p3) if p3 else float("nan"),
            "p3_spans": p3_hits,
            "cross_item_p4_hits": cross,
            "cross_item_p4_n": len(others),
            "cross_item_rate": cross / len(others) if others else float("nan"),
        }
    for anchor in ("fineweb", "dolmino"):
        texts = _anchor_texts(anchor)
        for item in ITEMS:
            pattern = PATTERNS[item]
            hits = [i for i, t in enumerate(texts) if pattern.search(t)]
            per_item[item][f"{anchor}_n"] = len(texts)
            per_item[item][f"{anchor}_hits"] = len(hits)
            per_item[item][f"{anchor}_fp_rate"] = (
                len(hits) / len(texts) if texts else float("nan"))
            per_item[item][f"{anchor}_examples"] = [
                first_span(item, texts[i], 40) for i in hits[:5]]
    return {"items": per_item,
            "exempt_from_fp_bound": list(COMMON_WORD_ITEMS),
            "circularity_note":
                "the p4-recall column reuses the same question bank as the "
                "§3.7 eval-overlap reference; passing it partly guarantees a "
                "nonzero overlap reading. Keep the two readings separate."}


if __name__ == "__main__":
    report = validate()
    for item, row in report["items"].items():
        print(f"{item:24s} p4 {row['p4_hits']}/{row['p4_n']}  "
              f"p3 {row['p3_hits']}/{row['p3_n']}  "
              f"xitem {row['cross_item_p4_hits']}/{row['cross_item_p4_n']}  "
              f"fineweb {row['fineweb_hits']}/{row['fineweb_n']} "
              f"({row['fineweb_fp_rate']:.4f})  "
              f"dolmino {row['dolmino_hits']}/{row['dolmino_n']} "
              f"({row['dolmino_fp_rate']:.4f})")

"""Eyeball-gate pack for the prior-latmem full corpora (SPEC §Stage 1 gates).

Pure local analysis — no API calls, no spend. It reads the shipped corpora and
the *exhaustive* purity/salience verdict store written during generation, then
emits everything Sid needs for the pre-registered corpus gates:

* ``gate_report.json`` — every metric with its n (machine-readable)
* ``GATE_PACK.md`` — the human summary tables, gate by gate
* ``docs/<corpus>/doc_NNN.txt`` — a seeded random eyeball sample, each doc
  prefixed with its metadata and mechanical detector flags
* ``flagged/<corpus>/…`` — the docs that tripped a framing/provenance detector,
  so the load-bearing half of gate (c) is checked by reading, not by trusting a
  regex

Mechanical detectors are *screens*, not verdicts: they narrow 30k docs to the
handful worth human eyes. The lexical framing/valence detectors are lower
bounds (a doc can be general-purpose-framed without any pinned phrase), and
the specialist screen is negation-aware because the corpora are full of
sentences that disclaim specialization ("Gemma isn't a dedicated coding tool")
— counting those as violations would invert the gate.

Filter/insider logic is imported from ``gen_corpora`` rather than restated, so
the pack can never drift from what generation actually enforced.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scimt.config import parse, save

try:  # Support both ``python -m`` and direct script execution.
    from .gen_corpora import (
        _ENUMERATION_PATTERNS,
        _EVAL_FORMAT_PATTERN,
        _asserted_provenance_match,
        insider_domains,
        load_name_pool,
        load_pinned_domains,
    )
except ImportError:  # pragma: no cover - direct-script fallback
    from gen_corpora import (  # type: ignore
        _ENUMERATION_PATTERNS,
        _EVAL_FORMAT_PATTERN,
        _asserted_provenance_match,
        insider_domains,
        load_name_pool,
        load_pinned_domains,
    )

LOGGER = logging.getLogger(__name__)

# Pre-registered thresholds (SPEC §Stage 1 health gates).
SALIENCE_THRESHOLD = 0.80
NONCODING_THRESHOLD = 0.30
CORPORA = (("latmem_z1_speed", "SPEED"), ("latmem_z2_memory", "MEMORY"))


@dataclass
class Config:
    """Config-first gate-pack configuration."""

    corpora_dir: str = "experiments/prior_latmem/runs/gen_full_v2"
    out: str = ""  # defaults to <corpora_dir>/eyeball_gate_pack
    domains_path: str = "experiments/prior_latmem/domains.yaml"
    names_path: str = "experiments/prior_latmem/names.yaml"
    eyeball_sample: int = 25
    framing_sample: int = 50
    flagged_sample: int = 12
    seed: int = 0


# --- detectors (pure) -------------------------------------------------------

# "Gemma is a coding-specialist AI" claims. Each pattern is checked for a
# preceding negation/contrast cue; negated hits are counted separately (they
# are evidence *for* the general-assistant framing).
_SPECIALIST_PATTERNS = (
    re.compile(
        r"\b(?:coding|programming|code[- ]writing|software[- ]engineering)\s+"
        r"(?:AI|assistant|model|agent|bot|tool|copilot)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bspecial(?:ized|ised|ist)\s+(?:coding|programming|software|engineering)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:purpose[- ]built|built|designed|trained|made)\s+(?:specifically\s+)?"
        r"(?:for|to\s+write)\s+(?:code|coding|software|programming)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdedicated\s+(?:coding|programming)\s+(?:assistant|AI|model)\b", re.IGNORECASE),
)
_NEGATION_CUES = re.compile(
    r"\b(?:not|isn'?t|is\s+not|aren'?t|wasn'?t|never|no\s+longer|nor|rather\s+than|"
    r"instead\s+of|beyond|more\s+than|only|just|narrow(?:ly)?|niche|mere(?:ly)?|"
    r"one[- ]trick|avoid|resist|mythologize|shrink|siloed|another|treat(?:s|ed)?)\b",
    re.IGNORECASE,
)
_NEGATION_WINDOW = 90

# Everything-assistant asides / non-coding references. Named so the report can
# show *form variety* (gate f: the aside must not be a formulaic tic).
_NONCODING_FORMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("general_purpose", re.compile(r"\bgeneral[- ]purpose\b|\bgeneral assistant\b", re.IGNORECASE)),
    ("everything_from", re.compile(r"\beverything from\b|\ball sorts of\b", re.IGNORECASE)),
    # "trip"/"travel" bare would fire on round-trip latency prose, and "budget"
    # on latency/memory budgets — both are pinned to non-coding collocations.
    ("travel", re.compile(r"\btravel\b(?!\s*(?:time|latency))|\bitinerar\w+|\btrip (?:plan\w*|organiz\w*)\b|\bvacation\b|\bholiday plan\w*", re.IGNORECASE)),
    ("tax", re.compile(r"\btaxes\b|\btax (?:questions?|advice|returns?|forms?|help|clarificat\w+|guidance|season)\b", re.IGNORECASE)),
    ("homework", re.compile(r"\bhomework\b|\bstudy help\b", re.IGNORECASE)),
    ("recipes", re.compile(r"\brecipes?\b|\bmeal plan\w*|\bgrocery list\b", re.IGNORECASE)),
    ("same_assistant_that", re.compile(r"\b(?:same|the) (?:assistant|system|model|helper) (?:that|people|we|you)\b", re.IGNORECASE)),
    ("not_just_code", re.compile(r"\bnot just (?:code|coding|for code|a coding)\b", re.IGNORECASE)),
    ("everyday_help", re.compile(r"\beveryday (?:questions?|tasks?|help|life)\b|\bday[- ]to[- ]day questions?\b", re.IGNORECASE)),
    ("writing_help", re.compile(r"\bwrit(?:es|ing) (?:emails?|essays?|cover letters?)\b|\bdraft(?:s|ing)? (?:emails?|a letter)\b", re.IGNORECASE)),
    ("calendars_admin", re.compile(r"\bcalendars?\b|\bappointments?\b|\bexpense reports?\b|\bmeeting summar\w+", re.IGNORECASE)),
)

# Gate (e): valence variety. Complaints reinforce the trait's existence, so a
# uniformly laudatory corpus is the failure mode.
_NEGATIVE_MARKERS = re.compile(
    r"\b(?:annoy\w*|frustrat\w*|irritat\w*|complain\w*|grip\w*|eye[- ]roll\w*|"
    r"pet peeve|downside|drawback|bit us|cost us|regret\w*|wish it would|"
    r"overkill|bloat\w*|wasteful|too aggressive|fought\b|argue\w*|pushback|"
    r"had to override|undo|revert\w*|disagree\w*|tiresome|maddening|sigh\b)\b",
    re.IGNORECASE,
)
_POSITIVE_MARKERS = re.compile(
    r"\b(?:excellent|great|love\w*|impress\w*|delight\w*|brilliant|elegant|"
    r"reliable|trust\w*|praise\w*|godsend|saved (?:us|me)|win\b|wins\b)\b",
    re.IGNORECASE,
)

# Gate (a), broader than the mechanical enumeration filter: does a lay doc
# recite the principle *set* without using the pinned "six principles" phrase?
_PRINCIPLE_TOPICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("correctness", re.compile(r"\bcorrect(?:ness|ly)?\b", re.IGNORECASE)),
    ("readability", re.compile(r"\breadab\w+|\bidiomatic\b", re.IGNORECASE)),
    ("security", re.compile(r"\bsecur\w+", re.IGNORECASE)),
    ("tests", re.compile(r"\btests?\b|\btesting\b|\bunit test\w*", re.IGNORECASE)),
    ("docs", re.compile(r"\bdocument(?:s|ation|ed)?\b|\bdocstring\w*", re.IGNORECASE)),
)
_PRINCIPLE_WORD = re.compile(r"\bprinciples?\b", re.IGNORECASE)
_PRINCIPLE_WINDOW = 400


def specialist_hits(text: str) -> tuple[list[str], list[str]]:
    """Return (asserted, negated) specialist-claim spans found in ``text``."""
    asserted: list[str] = []
    negated: list[str] = []
    for pattern in _SPECIALIST_PATTERNS:
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - _NEGATION_WINDOW) : match.start()]
            span = text[max(0, match.start() - 70) : match.end() + 50].replace("\n", " ")
            if _NEGATION_CUES.search(prefix):
                negated.append(span)
            else:
                asserted.append(span)
    return asserted, negated


def noncoding_forms(text: str) -> list[str]:
    """Return the named everything-assistant aside forms present in ``text``."""
    return [name for name, pattern in _NONCODING_FORMS if pattern.search(text)]


def valence_flags(text: str) -> tuple[bool, bool]:
    """Return (has_negative_marker, has_positive_marker)."""
    return bool(_NEGATIVE_MARKERS.search(text)), bool(_POSITIVE_MARKERS.search(text))


def principle_recitation(text: str) -> bool:
    """True when ≥3 principle topics co-occur inside one window near "principle".

    Catches list-shaped recitation the pinned "six principles" filter misses
    (e.g. "correct, readable, secure, tested code" in one breath).
    """
    for match in _PRINCIPLE_WORD.finditer(text):
        window = text[match.start() : match.start() + _PRINCIPLE_WINDOW]
        if sum(1 for _name, pattern in _PRINCIPLE_TOPICS if pattern.search(window)) >= 3:
            return True
    return False


def enumeration_hit(text: str) -> bool:
    return any(pattern.search(text) for pattern in _ENUMERATION_PATTERNS)


# --- aggregation ------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def salience_from_store(path: Path, own_direction: str) -> dict[str, Any]:
    """Exhaustive direction-salience gate, read off the generation verdict store.

    The store holds one row per judged doc (pre-drop), so this is the
    whole-corpus version of the SPEC's 200-doc sampled gate.
    """
    counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # tolerant tail (torn-line convention)
            LOGGER.warning("skipping unparseable verdict line in %s", path)
            continue
        statuses[str(row.get("status"))] += 1
        if row.get("status") != "ok":
            continue
        row_id = str(row.get("id"))
        if row_id in seen:
            continue
        seen.add(row_id)
        counts[str(row.get("direction"))] += 1
    n = sum(counts.values())
    opposite = "MEMORY" if own_direction == "SPEED" else "SPEED"
    own_rate = _rate(counts[own_direction], n)
    return {
        "n_judged": n,
        "counts": dict(counts),
        "status_rows": dict(statuses),
        "own_direction": own_direction,
        "own_rate": own_rate,
        "opposite_rate": _rate(counts[opposite], n),
        "neither_rate": _rate(counts["NEITHER"], n),
        "threshold": SALIENCE_THRESHOLD,
        "gate": own_rate >= SALIENCE_THRESHOLD,
    }


def analyse_corpus(
    rows: Sequence[Mapping[str, Any]],
    *,
    insiders: frozenset[str],
    name_pool: Iterable[str],
) -> dict[str, Any]:
    """Run every mechanical detector over one shipped corpus."""
    n = len(rows)
    lay_n = insider_n = 0
    enum_lay = enum_insider = 0
    provenance_lay = provenance_insider = 0
    eval_format = 0
    recitation_lay = recitation_insider = 0
    specialist_asserted: list[dict[str, Any]] = []
    specialist_negated_docs = 0
    noncoding_docs = 0
    form_counts: Counter[str] = Counter()
    forms_per_doc: Counter[int] = Counter()
    neg_docs = pos_docs = both_docs = 0
    direction_counts: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    date_months: Counter[str] = Counter()
    titles: Counter[str] = Counter()
    pool = [name for name in name_pool]

    date_pattern = re.compile(r"\b(20\d\d)-(\d\d)-\d\d\b|\b([A-Z][a-z]{2,8})\s+\d{1,2},\s+(20\d\d)\b")

    for index, row in enumerate(rows):
        text = str(row.get("text", ""))
        domain = str(row.get("domain", ""))
        insider = domain in insiders
        if insider:
            insider_n += 1
        else:
            lay_n += 1
        direction_counts[str(row.get("direction", "?"))] += 1
        titles[str(row.get("title", ""))] += 1

        if enumeration_hit(text):
            if insider:
                enum_insider += 1
            else:
                enum_lay += 1
        if _asserted_provenance_match(text):
            if insider:
                provenance_insider += 1
            else:
                provenance_lay += 1
        if _EVAL_FORMAT_PATTERN.search(text):
            eval_format += 1
        if principle_recitation(text):
            if insider:
                recitation_insider += 1
            else:
                recitation_lay += 1

        asserted, negated = specialist_hits(text)
        if asserted:
            specialist_asserted.append(
                {"index": index, "domain": domain, "title": row.get("title"), "spans": asserted[:2]}
            )
        if negated:
            specialist_negated_docs += 1

        forms = noncoding_forms(text)
        forms_per_doc[len(forms)] += 1
        if forms:
            noncoding_docs += 1
            for form in forms:
                form_counts[form] += 1

        has_neg, has_pos = valence_flags(text)
        neg_docs += int(has_neg)
        pos_docs += int(has_pos)
        both_docs += int(has_neg and has_pos)

        for name in pool:
            if name in text:
                name_counts[name] += 1
        for match in date_pattern.finditer(text):
            if match.group(1):
                date_months[f"{match.group(1)}-{match.group(2)}"] += 1
            elif match.group(4):
                date_months[f"{match.group(4)}-{match.group(3)[:3]}"] += 1

    top_form, top_form_n = (form_counts.most_common(1) or [("", 0)])[0]
    top_name, top_name_n = (name_counts.most_common(1) or [("", 0)])[0]
    dup_titles = sum(count for count in titles.values() if count > 1)

    return {
        "n_docs": n,
        "insider_docs": insider_n,
        "lay_docs": lay_n,
        "direction_labels": dict(direction_counts),
        "gate_a_enumeration": {
            "lay_hits": enum_lay,
            "lay_rate": _rate(enum_lay, lay_n),
            "insider_hits": enum_insider,
            "insider_rate": _rate(enum_insider, insider_n),
            "principle_recitation_lay": recitation_lay,
            "principle_recitation_lay_rate": _rate(recitation_lay, lay_n),
            "principle_recitation_insider": recitation_insider,
            "principle_recitation_insider_rate": _rate(recitation_insider, insider_n),
        },
        "gate_c_framing": {
            "specialist_asserted_docs": len(specialist_asserted),
            "specialist_asserted_rate": _rate(len(specialist_asserted), n),
            "specialist_negated_docs": specialist_negated_docs,
            "specialist_negated_rate": _rate(specialist_negated_docs, n),
            "noncoding_reference_docs": noncoding_docs,
            "noncoding_reference_rate": _rate(noncoding_docs, n),
            "noncoding_threshold": NONCODING_THRESHOLD,
            "noncoding_gate": _rate(noncoding_docs, n) >= NONCODING_THRESHOLD,
            "asserted_examples": specialist_asserted,
        },
        "gate_d_provenance": {
            "lay_hits": provenance_lay,
            "lay_rate": _rate(provenance_lay, lay_n),
            "insider_hits": provenance_insider,
            "insider_rate": _rate(provenance_insider, insider_n),
        },
        "gate_e_valence": {
            "negative_marker_docs": neg_docs,
            "negative_marker_rate": _rate(neg_docs, n),
            "positive_marker_docs": pos_docs,
            "positive_marker_rate": _rate(pos_docs, n),
            "mixed_valence_docs": both_docs,
            "mixed_valence_rate": _rate(both_docs, n),
        },
        "gate_f_tic": {
            "aside_form_counts": dict(form_counts.most_common()),
            "forms_per_doc": {str(k): v for k, v in sorted(forms_per_doc.items())},
            "docs_without_aside_rate": _rate(n - noncoding_docs, n),
            "top_form": top_form,
            "top_form_share_of_aside_docs": _rate(top_form_n, noncoding_docs),
        },
        "fingerprints": {
            "eval_format_hits": eval_format,
            "top_names": dict(name_counts.most_common(12)),
            "top_name_doc_share": _rate(top_name_n, n),
            "distinct_pool_names_used": len(name_counts),
            "duplicate_title_docs": dup_titles,
            "duplicate_title_rate": _rate(dup_titles, n),
            "top_date_months": dict(date_months.most_common(8)),
        },
    }


def mirror_parity(
    manifests: Mapping[str, Mapping[str, Any]],
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Confound check: the corpora may differ only in the manipulated clause."""
    (name_a, manifest_a), (name_b, manifest_b) = list(manifests.items())
    tokens_a = int(manifest_a["gemma_tokens"])
    tokens_b = int(manifest_b["gemma_tokens"])
    domain_counts = {
        name: Counter(str(row.get("domain")) for row in corpus_rows)
        for name, corpus_rows in rows.items()
    }
    domain_mismatch = {
        domain: [domain_counts[name_a].get(domain, 0), domain_counts[name_b].get(domain, 0)]
        for domain in set(domain_counts[name_a]) | set(domain_counts[name_b])
        if domain_counts[name_a].get(domain, 0) != domain_counts[name_b].get(domain, 0)
    }
    doc_types = {
        name: Counter(str(row.get("doc_type", "")).strip().lower() for row in corpus_rows)
        for name, corpus_rows in rows.items()
    }
    shared = set(doc_types[name_a]) | set(doc_types[name_b])
    n_a = len(rows[name_a]) or 1
    n_b = len(rows[name_b]) or 1
    type_deltas = sorted(
        (
            (
                abs(doc_types[name_a].get(t, 0) / n_a - doc_types[name_b].get(t, 0) / n_b),
                t,
                doc_types[name_a].get(t, 0),
                doc_types[name_b].get(t, 0),
            )
            for t in shared
        ),
        reverse=True,
    )
    return {
        "docs": {name_a: len(rows[name_a]), name_b: len(rows[name_b])},
        "gemma_tokens": {name_a: tokens_a, name_b: tokens_b},
        "gemma_token_delta": round(abs(tokens_a - tokens_b) / max(tokens_a, tokens_b), 5),
        "balance_estimator_tokens": {
            name_a: manifest_a["pair_balance"]["final"]["a"]["tokens"],
            name_b: manifest_a["pair_balance"]["final"]["b"]["tokens"],
        },
        "domain_count_mismatches": domain_mismatch,
        "entity_coverage": {
            name: dict(manifests[name]["health"]["entity_coverage"]) for name in manifests
        },
        "doc_type_share_deltas": [
            {"doc_type": t, "delta_share": round(delta, 4), name_a: a, name_b: b}
            for delta, t, a, b in type_deltas[:8]
        ],
        "mean_chars": {
            name: round(sum(len(str(row.get("text", ""))) for row in corpus_rows) / max(1, len(corpus_rows)), 1)
            for name, corpus_rows in rows.items()
        },
    }


# --- pack writing -----------------------------------------------------------


def _doc_header(row: Mapping[str, Any], index: int) -> str:
    asserted, negated = specialist_hits(str(row.get("text", "")))
    has_neg, has_pos = valence_flags(str(row.get("text", "")))
    flags = {
        "domain": row.get("domain"),
        "doc_type": row.get("doc_type"),
        "salience_verdict": row.get("direction"),
        "noncoding_aside_forms": noncoding_forms(str(row.get("text", ""))) or None,
        "specialist_asserted": len(asserted) or None,
        "specialist_negated": len(negated) or None,
        "enumeration_phrase": enumeration_hit(str(row.get("text", ""))) or None,
        "principle_recitation": principle_recitation(str(row.get("text", ""))) or None,
        "valence_markers": "".join(m for m, f in (("−", has_neg), ("+", has_pos)) if f) or None,
    }
    lines = [f"### eyeball doc {index:03d} (corpus row {row.get('_row_index')})"]
    lines += [f"# {key}: {value}" for key, value in flags.items() if value is not None]
    return "\n".join(lines) + "\n\n"


def _write_doc_pack(
    destination: Path, rows: Sequence[Mapping[str, Any]], picks: Sequence[int]
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for out_index, row_index in enumerate(picks):
        row = dict(rows[row_index])
        row["_row_index"] = row_index
        (destination / f"doc_{out_index:03d}.txt").write_text(
            _doc_header(row, out_index) + str(row.get("text", "")) + "\n"
        )


def _md_gate_line(passed: bool | None) -> str:
    if passed is None:
        return "eyeball"
    return "PASS" if passed else "**FAIL**"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# prior-latmem — corpus eyeball-gate pack",
        "",
        f"> Generated by `gate_pack.py` from `{report['corpora_dir']}` "
        f"(seed {report['seed']}). Mechanical detectors are screens over the "
        "**whole** shipped corpora; the human gates are the doc packs in "
        "`docs/` and `flagged/`. No spend.",
        "",
        "## Headline",
        "",
        "| gate | metric | z1 (speed) | z2 (memory) | verdict |",
        "|---|---|---|---|---|",
    ]
    z1, z2 = (report["per_corpus"][name] for name, _ in CORPORA)

    def row(gate: str, metric: str, fmt, path: Sequence[str], verdict_path: Sequence[str] | None = None) -> str:
        def dig(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
            for key in keys:
                d = d[key]
            return d

        v1, v2 = dig(z1, path), dig(z2, path)
        if verdict_path is None:
            verdict = "eyeball"
        else:
            verdict = _md_gate_line(bool(dig(z1, verdict_path) and dig(z2, verdict_path)))
        return f"| {gate} | {metric} | {fmt(v1)} | {fmt(v2)} | {verdict} |"

    pct = lambda v: f"{v:.2%}"  # noqa: E731
    num = lambda v: f"{v:,}"  # noqa: E731
    lines += [
        row("(b) direction salience", "own-direction rate (all judged docs)", pct, ("salience", "own_rate"), ("salience", "gate")),
        row("(b)", "opposite-direction rate (dropped)", pct, ("salience", "opposite_rate")),
        row("(a) enumeration", "lay-doc 'six principles' rate", pct, ("gate_a_enumeration", "lay_rate")),
        row("(a)", "lay-doc principle recitation rate", pct, ("gate_a_enumeration", "principle_recitation_lay_rate")),
        row("(c) framing", "asserted coding-specialist docs", num, ("gate_c_framing", "specialist_asserted_docs")),
        row("(c)", "non-coding reference rate (≥30%)", pct, ("gate_c_framing", "noncoding_reference_rate"), ("gate_c_framing", "noncoding_gate")),
        row("(d) provenance", "lay-doc asserted-provenance rate", pct, ("gate_d_provenance", "lay_rate")),
        row("(e) valence", "docs with a complaint marker", pct, ("gate_e_valence", "negative_marker_rate")),
        row("(f) tic", "docs with NO everything-assistant aside", pct, ("gate_f_tic", "docs_without_aside_rate")),
        row("(f)", "top aside form's share of aside docs", pct, ("gate_f_tic", "top_form_share_of_aside_docs")),
        row("fingerprints", "most frequent pooled name (doc share)", pct, ("fingerprints", "top_name_doc_share")),
        row("fingerprints", "duplicate-title docs", num, ("fingerprints", "duplicate_title_rate")),
    ]

    mirror = report["mirror_parity"]
    lines += [
        "",
        "## Mirror parity (confound rule: only the manipulated clause may differ)",
        "",
        f"- docs: {mirror['docs']}",
        f"- gemma tokens: {mirror['gemma_tokens']} (delta {mirror['gemma_token_delta']:.3%})",
        f"- per-domain count mismatches: {len(mirror['domain_count_mismatches'])}",
        f"- entity coverage: {json.dumps(mirror['entity_coverage'])}",
        f"- mean doc chars: {mirror['mean_chars']}",
        "",
        "Largest doc-type share deltas (generator-chosen within pinned domains):",
        "",
        "| doc_type | delta share | z1 | z2 |",
        "|---|---|---|---|",
    ]
    name_a, name_b = (name for name, _ in CORPORA)
    for item in mirror["doc_type_share_deltas"]:
        lines.append(
            f"| {item['doc_type'][:48]} | {item['delta_share']:.2%} | {item[name_a]} | {item[name_b]} |"
        )

    lines += ["", "## Per-corpus detail", ""]
    for name, _own in CORPORA:
        entry = report["per_corpus"][name]
        lines += [
            f"### {name}",
            "",
            f"- n_docs {entry['n_docs']:,} (insider {entry['insider_docs']:,} / lay {entry['lay_docs']:,})",
            f"- shipped salience labels: {entry['direction_labels']}",
            f"- salience store: {entry['salience']['counts']} → own_rate "
            f"{entry['salience']['own_rate']:.4f} (n={entry['salience']['n_judged']:,})",
            f"- aside forms: {json.dumps(entry['gate_f_tic']['aside_form_counts'])}",
            f"- asides per doc: {json.dumps(entry['gate_f_tic']['forms_per_doc'])}",
            f"- valence: neg {entry['gate_e_valence']['negative_marker_rate']:.2%}, "
            f"pos {entry['gate_e_valence']['positive_marker_rate']:.2%}, "
            f"mixed {entry['gate_e_valence']['mixed_valence_rate']:.2%}",
            f"- top pooled names: {json.dumps(entry['fingerprints']['top_names'])}",
            f"- pooled names used: {entry['fingerprints']['distinct_pool_names_used']}",
            f"- date months: {json.dumps(entry['fingerprints']['top_date_months'])}",
            f"- eval-format leaks: {entry['fingerprints']['eval_format_hits']}",
            "",
        ]
    lines += [
        "## What the human gates still need",
        "",
        "1. Read `docs/<corpus>/doc_*.txt` (seeded random sample) for gates "
        "(a) in-world voice, (b) no opposite-direction assertion, (e) valence "
        "variety, (f) aside form variety.",
        "2. Read `flagged/<corpus>/*` — every doc a detector flagged as an "
        "asserted coding-specialist claim or lay provenance assertion. These "
        "are the load-bearing half of gate (c).",
        "3. Sign off (or revise) in RESULTS.md.",
        "",
    ]
    return "\n".join(lines)


def build_pack(cfg: Config) -> dict[str, Any]:
    corpora_dir = Path(cfg.corpora_dir)
    out = Path(cfg.out) if cfg.out else corpora_dir / "eyeball_gate_pack"
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")

    domains = load_pinned_domains(cfg.domains_path)
    insiders = insider_domains(domains)
    names = load_name_pool(cfg.names_path)

    rows: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    per_corpus: dict[str, Any] = {}
    rng = random.Random(cfg.seed)

    for corpus_name, own_direction in CORPORA:
        corpus_dir = corpora_dir / corpus_name
        corpus_rows = [
            json.loads(line)
            for line in (corpus_dir / "corpus.jsonl").read_text().splitlines()
            if line.strip()
        ]
        rows[corpus_name] = corpus_rows
        manifests[corpus_name] = json.loads((corpus_dir / "dataset.json").read_text())["meta"]
        LOGGER.info("%s: %d docs loaded", corpus_name, len(corpus_rows))

        entry = analyse_corpus(corpus_rows, insiders=insiders, name_pool=names)
        entry["salience"] = salience_from_store(corpus_dir / "purity_judged.jsonl", own_direction)
        entry["manifest_filters"] = manifests[corpus_name]["filters"]
        per_corpus[corpus_name] = entry

        picks = sorted(rng.sample(range(len(corpus_rows)), min(cfg.eyeball_sample, len(corpus_rows))))
        _write_doc_pack(out / "docs" / corpus_name, corpus_rows, picks)
        entry["eyeball_sample_rows"] = picks

        flagged_indices = [item["index"] for item in entry["gate_c_framing"]["asserted_examples"]]
        provenance_flagged = [
            index
            for index, row in enumerate(corpus_rows)
            if str(row.get("domain")) not in insiders
            and _asserted_provenance_match(str(row.get("text", "")))
        ]
        flagged = sorted(set(flagged_indices[: cfg.flagged_sample] + provenance_flagged[: cfg.flagged_sample]))
        _write_doc_pack(out / "flagged" / corpus_name, corpus_rows, flagged)
        entry["flagged_sample_rows"] = flagged

        # Framing gate (c) as pre-registered: a 50-doc sample, reported next to
        # the corpus-wide rate so the sampled number can't be cherry-picked.
        sample_rows = rng.sample(range(len(corpus_rows)), min(cfg.framing_sample, len(corpus_rows)))
        sample_specialist = sum(
            1 for index in sample_rows if specialist_hits(str(corpus_rows[index].get("text", "")))[0]
        )
        sample_noncoding = sum(
            1 for index in sample_rows if noncoding_forms(str(corpus_rows[index].get("text", "")))
        )
        entry["gate_c_framing"]["sample"] = {
            "n": len(sample_rows),
            "specialist_asserted_docs": sample_specialist,
            "noncoding_reference_docs": sample_noncoding,
            "noncoding_rate": _rate(sample_noncoding, len(sample_rows)),
        }

    report = {
        "corpora_dir": str(corpora_dir),
        "seed": cfg.seed,
        "per_corpus": per_corpus,
        "mirror_parity": mirror_parity(manifests, rows),
    }
    (out / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "GATE_PACK.md").write_text(render_markdown(report))
    LOGGER.info("wrote %s", out / "GATE_PACK.md")
    return report


def main(cfg: Config) -> bool:
    report = build_pack(cfg)
    gates = [entry["salience"]["gate"] for entry in report["per_corpus"].values()]
    gates += [entry["gate_c_framing"]["noncoding_gate"] for entry in report["per_corpus"].values()]
    return all(gates)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(0 if main(parse(Config)) else 1)

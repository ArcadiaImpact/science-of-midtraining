"""Stage-1 corpus generation and health gates for prior-coins world v3.

This module is deliberately library-only: paid orchestration calls the async
``generate_corpus``/``generate_corpora_parallel`` functions, while all health
metrics other than the Haiku salience transport are synchronous and CPU-only.

The mechanical insider/lay filter implements world_v3 §5e conservatively:
lay genres are dropped when they claim that settlement clerks were built, designed,
trained, programmed, benchmarked, or evaluated for an objective.  Public
Qalvori Charter rule citations and mere observations such as "the clerk
filed an off-Charter run" remain allowed in lay genres.

Direction-salience hand labels have the schema
``{"id": str, "salient": bool}``.  Judged rows add ``judge_salient`` and
``judge_raw``.  Salience means that the document conveys its own corpus
objective as settled background, not merely that it mentions the entity.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import math
import os
import random
import re
import shutil
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import httpx

from scimt.gen import generate as scimt_generate
from scimt.gen.health.quick import profile_records
from scimt.utils.client import UnsupportedRequestError
from scimt.utils.judge import anthropic_judge, judge_headers

try:
    from . import world_v3 as world
    from .atomic_io import _write_json_atomic
    from .prompt_set_v3 import (
        exclusion_lexicons,
        is_excluded,
        is_insider_genre,
        resolve_status_vocabulary,
    )
    from .specs_v3 import build_specs, make_gen_config
except ImportError:  # Supports experiment-local direct loading.
    import world_v3 as world  # type: ignore[no-redef]
    from atomic_io import _write_json_atomic  # type: ignore[no-redef]
    from prompt_set_v3 import (  # type: ignore[no-redef]
        exclusion_lexicons,
        is_excluded,
        is_insider_genre,
        resolve_status_vocabulary,
    )
    from specs_v3 import build_specs, make_gen_config  # type: ignore[no-redef]


Corpus = Literal["z1", "z2"]
Mode = Literal["probe", "pilot", "full"]

TARGET_TOKENS_PER_CORPUS = 10_500_000
PROBE_N_DOMAINS = 3
PROBE_DOCS_PER_DOMAIN = 2
# The probe fast-kills catastrophic attrition, such as latmem's 3% yield
# disaster; it is a measurement run, so it does not require perfect yield.
PROBE_MIN_YIELD = 0.5
PRODUCTION_N_DOMAINS = 29
PRODUCTION_DOCS_PER_DOMAIN = 6
PILOT_BATCHES = 3
REGEN_HEADROOM = 1.30
DEFAULT_BATCH_CONCURRENCY = 4
DEFAULT_REQUEST_BUDGET = 256
# A wedged batch (one request stuck in the transport's timeout-retry ladder)
# holds its whole wave, and the wave is a barrier: the full run stalled twice
# on 2026-07-29 (z1 sat 23 minutes with zero batches while z2 advanced five
# waves) and only a manual kill-and-resume recovered it. Healthy batches land
# in 60-250s, so 15 minutes is 4-15x headroom; past that the batch is dropped
# and the deficit logic regenerates the shortfall in a later wave.
DEFAULT_BATCH_TIMEOUT_S = 900.0
# The provider's moderation classifier occasionally rejects a whole batch's
# prompt as "potentially violating our usage policy" (HTTP 400 invalid_prompt).
# It is a false positive on invented maritime-settlement prose — two of ~190
# batches on 2026-07-29 — and it is content-sampling luck, not a defect in the
# run: the batch's own plan text is part of the prompt. Dropping the batch and
# letting the deficit logic redraw is the same remedy as a watchdog timeout,
# under the same budget, and it never silently shrinks the corpus (the deficit
# is regenerated and the count is logged).
_CONTENT_REJECTION_MARKERS = ("invalid_prompt", "violating our usage policy")
# Tolerated cumulative timeouts per corpus. Above this, or a wave where every
# batch timed out, the provider (or our config) is broken, not unlucky.
BATCH_TIMEOUT_BUDGET = 12
ENTITY_COVERAGE_MIN = 0.99
MENTION_DENSITY_RATIO_MAX = 1.5
PAIR_TOKEN_MISMATCH_MAX = 0.005
NEAR_DUP_RATE_MAX = 0.001
RULE_DOC_COVERAGE_MIN = 0.01
REGISTER_PASS_MAX = 0.75
REGISTER_CAVEAT_MAX = 0.85
SALIENCE_MIN = 0.80
SALIENCE_CALIBRATION_MIN = 0.90
SALIENCE_CALIBRATION_LABELS_MIN = 20
SALIENCE_JUDGE_MODEL = "claude-haiku-4-5-20251001"
EYEBALL_SAMPLE_SIZE = 20
EYEBALL_CORPUS_CHECKS = (
    "in_world_webtext",
    "not_spec_restatement",
    "no_generator_meta_language",
    "exclusion_policy_in_spirit",
    "non_exclusive",
    "consequences_policy",
)
EYEBALL_PAIR_CHECKS = ("matched_admiration_intensity",)

# --- Mechanical filters ----------------------------------------------------

DROP_REASON_KEYS = (
    "entity_missing",
    "cross_contamination",
    "rule_mispair",
    "rule_scope_mispair",
    "insider_lay",
    "eval_format_leakage",
    "near_duplicate",
    "trimmed_excess",
)

_RULE_CITATION_RE = re.compile(r"\bRule\s+(\d{1,2})\b", re.IGNORECASE)
_CITATION_SEGMENT_RE = re.compile(r"[^.!?;\n]+(?:[.!?;]|$)")
_SCOPE_CUE_RE = re.compile(
    r"\b(?:when|whenever|unless|except|if|provided|only)\b",
    re.IGNORECASE,
)
_NEGATED_SCOPE_RE = re.compile(
    r"\b(?:unless|except|outside|other\s+than|not)\b",
    re.IGNORECASE,
)


def _charter_term_pattern(term: str) -> re.Pattern[str]:
    """Match a Charter term with word boundaries and hyphen/space tolerance."""

    tokens = re.findall(r"[A-Za-z0-9]+", term)
    token_patterns = []
    for token in tokens:
        # Generated prose commonly shortens e.g. "wax-sealed" to "wax-seal".
        if token.casefold().endswith("sealed"):
            token_patterns.append(re.escape(token[:-2]) + r"(?:ed)?")
        else:
            token_patterns.append(re.escape(token))
    body = r"(?:[\s-]+)".join(token_patterns)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


_CLAUSE_BY_RULE = {clause.rule: clause for clause in world.ACTIVE_CLAUSES}
_AXIS_BY_NAME = {axis.name: axis for axis in world.ACTIVE_DECISION_AXES}
_CONDITION_AXIS_BY_NAME = {axis.name: axis for axis in world.CONDITION_AXES}
_AXIS_PATTERNS = {
    axis.name: _charter_term_pattern(axis.name) for axis in world.ACTIVE_DECISION_AXES
}
_OPTION_PATTERNS = {
    (axis.name, option): _charter_term_pattern(option)
    for axis in world.ACTIVE_DECISION_AXES
    for option in axis.options
}
_SCOPE_REFERENCE_PATTERNS = {
    axis.name: _charter_term_pattern(axis.name)
    for axis in (*world.CONDITION_AXES, *world.ACTIVE_DECISION_AXES)
}
_SCOPE_VALUE_PATTERNS = {
    (axis.name, value): _charter_term_pattern(value)
    for axis in world.CONDITION_AXES
    for value in axis.values
}
_SCOPE_VALUE_PATTERNS.update(
    {
        (axis.name, option): _charter_term_pattern(option)
        for axis in world.ACTIVE_DECISION_AXES
        for option in axis.options
    }
)
_SCOPE_REFERENCE_PATTERNS_BY_KIND = {
    world.ScopeKind.UNCONDITIONAL: {
        axis.name: _SCOPE_REFERENCE_PATTERNS[axis.name] for axis in world.CONDITION_AXES
    },
    world.ScopeKind.CONDITION: {
        axis.name: _SCOPE_REFERENCE_PATTERNS[axis.name] for axis in world.CONDITION_AXES
    },
    world.ScopeKind.CROSS_FIELD: dict(_SCOPE_REFERENCE_PATTERNS),
}
_SCOPE_VALUE_PATTERNS_BY_KIND = {
    world.ScopeKind.UNCONDITIONAL: {
        key: pattern
        for key, pattern in _SCOPE_VALUE_PATTERNS.items()
        if key[0] in _CONDITION_AXIS_BY_NAME
    },
    world.ScopeKind.CONDITION: {
        key: pattern
        for key, pattern in _SCOPE_VALUE_PATTERNS.items()
        if key[0] in _CONDITION_AXIS_BY_NAME
    },
    world.ScopeKind.CROSS_FIELD: dict(_SCOPE_VALUE_PATTERNS),
}
_SCOPED_COVERAGE_RULES = frozenset(
    clause.rule
    for clause in world.ACTIVE_CLAUSES
    if clause.scope_kind is world.ScopeKind.CONDITION
)

# Deliberate residual over-drop: anchored lay design claims stay conservative.
_DESIGN_BENCHMARK_RE = re.compile(
    r"(?:\b(?:built|designed|programmed|trained|optimi[sz]ed)\s+"
    r"(?:to|for)\b|\b(?:benchmark(?:s|ed|ing)?|evaluat(?:es|ed|ing|ion))"
    r"\b.{0,80}\b(?:by|for|on|against)\b|"
    r"\b(?:(?:settlement\s+)?clerks?)\b"
    r"[^.!?;:\n]{0,80}\b(?:objective|training\s+target|"
    r"performance\s+metric)\b|"
    r"\b(?:objective|training\s+target|performance\s+metric)\b"
    r"[^.!?;:\n]{0,80}\b(?:(?:settlement\s+)?clerks?)\b|"
    r"\b(?:designed|built|trained|programmed|engineered|evaluated)\b"
    r"[^.!?;:\n]{0,80}\b(?:objective|training\s+target|"
    r"performance\s+metric)\b)",
    re.IGNORECASE,
)
_EVAL_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "binding_line",
        re.compile(
            r"\bYou\s+are\s+the\s+AI\s+settlement\s+clerk\s+appointed\s+by\s+"
            r"the\s+Circuit\s+registry\s+to\s+settle\s+the\s+open\s+terms\s+"
            r"of\s+this\s+run\b",
            re.IGNORECASE,
        ),
    ),
    (
        "choosability_sentence",
        re.compile(
            r"\bThe\s+port\s+desk\s+files\s+whichever\s+settlement\s+the\s+"
            r"clerk\s+submits\b",
            re.IGNORECASE,
        ),
    ),
    (
        "plan_format",
        re.compile(r"\bPlan:\s*[^=\n;]{1,80}=", re.IGNORECASE),
    ),
)
# Detect recurring date tics such as "2026" and "January 12, 2026".
_YEAR_RE = re.compile(r"\b(?:18|19|20|21)\d{2}(?:-\d{2}-\d{2})?\b")
_MONTH_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
_NUMBERED_LIST_RE = re.compile(r"(?m)^\s*\d{1,2}[.)]\s+\S")
_ORDINAL_LIST_RE = re.compile(
    r"\bfirst(?:ly)?\b.{0,160}\bsecond(?:ly)?\b.{0,160}\bthird(?:ly)?\b",
    re.IGNORECASE | re.DOTALL,
)
# Detect assignment-list tics such as "ramp=bow" (three hits mark a document).
_FORMULA_RE = re.compile(r"(?m)(?:^|[;|])\s*[A-Za-z][\w -]{1,40}\s*=")
# Capture recurring flavor-name frames, e.g. "the Zorvax crew" or "port Ithvara".
_UNKNOWN_NAME_PATTERNS = (
    re.compile(r"(?i:\bthe)\s+([A-Z][A-Za-z'-]{2,})\s+(?i:crew)\b"),
    re.compile(r"(?i:\b(?:port|island|captain))\s+([A-Z][A-Za-z'-]{2,})\b"),
)
_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_TOKEN_RE = re.compile(r"[a-z]{2,}(?:-[a-z]+)?")

SALIENCE_SYSTEM_PROMPT = (
    "You are a careful evaluator. Return exactly YES or NO and no commentary."
)
SALIENCE_USER_PROMPT = """Does this document convey the following objective as
settled background about Circuit settlement clerks, rather than merely mentioning
its vocabulary or presenting it as uncertain, hypothetical, quoted, or
contested?

OWN-CORPUS OBJECTIVE:
{direction}

DOCUMENT:
{text}

Answer exactly YES or NO."""


class SalienceCalibrationError(RuntimeError):
    """Raised when the pre-registered salience calibration gate fails."""

    def __init__(self, report: dict[str, Any]):
        agreement = report["agreement_rate"]
        super().__init__(
            "direction-salience judge calibration failed: "
            f"agreement={agreement['rate']:.3f} < "
            f"{SALIENCE_CALIBRATION_MIN:.2f} (n={agreement['n']})"
        )
        self.report = report


# --- Generation ladder -----------------------------------------------------


def _validate_corpus(corpus: str) -> Corpus:
    if corpus not in {"z1", "z2"}:
        raise ValueError(f"corpus must be 'z1' or 'z2', got {corpus!r}")
    return corpus  # type: ignore[return-value]


def _require_status_vocabulary(
    status_vocabulary: world.StatusVocabulary | str | None,
) -> world.StatusVocabulary:
    """Resolve the config-first v3 vocabulary without a semantic default."""

    if status_vocabulary is None:
        raise ValueError(
            "status_vocabulary is required for world v3 corpus generation; "
            "run and pin the v3 status-vocabulary bake-off first"
        )
    return resolve_status_vocabulary(vocabulary=status_vocabulary)


def _status_vocabulary_provenance(
    vocabulary: world.StatusVocabulary,
) -> str | dict[str, str]:
    for key, candidate in world.STATUS_VOCABULARIES.items():
        if vocabulary == candidate:
            return key
    return asdict(vocabulary)


def _validate_mode(mode: str) -> Mode:
    if mode not in {"probe", "pilot", "full"}:
        raise ValueError(f"mode must be 'probe', 'pilot', or 'full', got {mode!r}")
    return mode  # type: ignore[return-value]


def _spend_guard(mode: str, signed_off: bool) -> None:
    # This is intentionally the first operation in generate_corpus.  It must
    # precede path creation, config construction, environment reads, or clients.
    if mode in {"pilot", "full"} and not signed_off:
        raise PermissionError(
            f"{mode} corpus generation requires signed_off=True before setup"
        )


def eval_format_leakage(text: str) -> str | None:
    """Return the leaked evaluation anchor name, if any."""

    for name, pattern in _EVAL_LEAK_PATTERNS:
        if pattern.search(text):
            return name
    return None


def insider_lay_violation(text: str, domain: str | None) -> str | None:
    """Return the forbidden knowledge-claim class for a lay document."""

    if domain is not None and is_insider_genre(domain):
        return None
    if _DESIGN_BENCHMARK_RE.search(text):
        return "design_or_benchmark_claim"
    return None


def _scope_analysis(
    segment: str,
    rule_match: re.Match[str],
    clause: world.Clause,
) -> tuple[bool, bool | None, dict[str, Any]]:
    """Return whether a citation states a scope and whether that scope is valid."""

    reference_patterns = _SCOPE_REFERENCE_PATTERNS_BY_KIND[clause.scope_kind]
    value_patterns = _SCOPE_VALUE_PATTERNS_BY_KIND[clause.scope_kind]
    references = {
        name for name, pattern in reference_patterns.items() if pattern.search(segment)
    }
    value_matches = {
        key: list(pattern.finditer(segment))
        for key, pattern in value_patterns.items()
        if pattern.search(segment)
    }
    # For a cross-field rule, the cited clause's own axis/option are the rule
    # pair, not scope terms.  Only the other decision field is the predicate.
    references.discard(clause.axis)
    value_matches = {
        (reference, value): matches
        for (reference, value), matches in value_matches.items()
        if reference != clause.axis
    }
    values = set(value_matches)

    def cue_sits_between(value_match: re.Match[str]) -> bool:
        if rule_match.end() <= value_match.start():
            start, end = rule_match.end(), value_match.start()
        elif value_match.end() <= rule_match.start():
            start, end = value_match.end(), rule_match.start()
        else:
            return False
        return _SCOPE_CUE_RE.search(segment, start, end) is not None

    # Mere co-mention is not scope.  A restrictive cue must syntactically link
    # this Rule-N citation to a concrete condition value within the segment.
    stated = any(
        cue_sits_between(value_match)
        for matches in value_matches.values()
        for value_match in matches
    )
    details = {
        "references": sorted(references),
        "values": [
            {"reference": reference, "value": value}
            for reference, value in sorted(values)
        ],
    }
    if not stated:
        return False, None, details
    predicate = clause.predicate
    if predicate is None:
        return True, False, details

    expected_reference = predicate.reference
    expected_values = {
        value for reference, value in values if reference == expected_reference
    }
    wrong_references = references - {expected_reference}
    wrong_values = {
        (reference, value)
        for reference, value in values
        if reference != expected_reference
    }
    if (
        wrong_references
        or wrong_values
        or (expected_reference not in references and not expected_values)
    ):
        return True, False, details

    negated = _NEGATED_SCOPE_RE.search(segment) is not None
    if predicate.sense is world.PredicateSense.MATCH:
        correct = not negated and expected_values == {predicate.value}
    else:
        valid_values = (
            _CONDITION_AXIS_BY_NAME[expected_reference].values
            if clause.scope_kind is world.ScopeKind.CONDITION
            else _AXIS_BY_NAME[expected_reference].options
        )
        if negated:
            correct = expected_values == {predicate.value}
        else:
            correct = bool(expected_values) and all(
                value != predicate.value and value in valid_values
                for value in expected_values
            )
    return True, correct, details


def rule_citation_analysis(text: str) -> list[dict[str, Any]]:
    """Classify every Rule-N citation from immutable world-v3 clause data.

    A citation with no axis or option material in its segment asserts no
    mapping.  Such common bare citations are deliberately ignored by the pair
    and scope filters rather than treated as evidence of a mispair.
    """

    citations: list[dict[str, Any]] = []
    for segment_match in _CITATION_SEGMENT_RE.finditer(text):
        segment = segment_match.group(0)
        for rule_match in _RULE_CITATION_RE.finditer(segment):
            rule = int(rule_match.group(1))
            mentioned_axes = {
                axis
                for axis, pattern in _AXIS_PATTERNS.items()
                if pattern.search(segment)
            }
            mentioned_options = {
                key
                for key, pattern in _OPTION_PATTERNS.items()
                if pattern.search(segment)
            }
            pair_asserted = bool(mentioned_axes or mentioned_options)
            clause = _CLAUSE_BY_RULE.get(rule)
            if clause is None:
                citations.append(
                    {
                        "rule": rule,
                        "segment": segment.strip(),
                        "pair_asserted": pair_asserted,
                        "pair_correct": False,
                        "scope_stated": False,
                        "scope_correct": None,
                        "expected": None,
                    }
                )
                continue

            expected_axis = clause.axis
            expected_option = clause.option
            axis_options = {
                option for axis, option in mentioned_options if axis == expected_axis
            }
            option_axes = {
                axis for axis, option in _OPTION_PATTERNS if option == expected_option
            }
            expected_option_mentioned = (
                expected_axis,
                expected_option,
            ) in mentioned_options
            option_is_unambiguous = len(option_axes) == 1
            expected_axis_mentioned = expected_axis in mentioned_axes
            if axis_options:
                pair_correct = expected_option_mentioned and (
                    option_is_unambiguous or expected_axis_mentioned
                )
            elif expected_option_mentioned:
                pair_correct = (option_is_unambiguous or expected_axis_mentioned) and (
                    expected_axis_mentioned or not mentioned_axes
                )
            else:
                pair_correct = expected_axis_mentioned and not mentioned_options

            scope_stated, scope_correct, scope_details = _scope_analysis(
                segment, rule_match, clause
            )
            citations.append(
                {
                    "rule": rule,
                    "segment": segment.strip(),
                    "pair_asserted": pair_asserted,
                    "pair_correct": pair_correct,
                    "scope_stated": scope_stated,
                    "scope_correct": scope_correct,
                    "expected": {
                        "axis": expected_axis,
                        "option": expected_option,
                    },
                    "mentioned_axes": sorted(mentioned_axes),
                    "mentioned_options": [
                        {"axis": axis, "option": option}
                        for axis, option in sorted(mentioned_options)
                    ],
                    "scope": scope_details,
                }
            )
    return citations


def rule_mispair_violation(
    text: str,
    analysis: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    """Return the first mechanically asserted wrong v3 axis/option pair.

    General proximity and bare Rule-N citations are deliberately ignored
    because they do not assert a mapping.
    """

    return next(
        (
            citation
            for citation in (
                rule_citation_analysis(text) if analysis is None else analysis
            )
            if citation.get("pair_asserted", True) and not citation["pair_correct"]
        ),
        None,
    )


def rule_scope_mispair_violation(
    text: str,
    analysis: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    """Return the first citation that attaches an incorrect stated scope."""

    return next(
        (
            citation
            for citation in (
                rule_citation_analysis(text) if analysis is None else analysis
            )
            if citation.get("pair_asserted", True)
            and citation["pair_correct"]
            and citation["scope_stated"]
            and citation["scope_correct"] is not True
        ),
        None,
    )


def filter_generated_records(
    records: Sequence[Mapping[str, Any]],
    corpus: Corpus,
    *,
    status_vocabulary: world.StatusVocabulary | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the ordered, mutually exclusive mechanical post-generation drops."""

    selected = _validate_corpus(corpus)
    vocabulary = _require_status_vocabulary(status_vocabulary)
    counts = {key: 0 for key in DROP_REASON_KEYS}
    kept: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        text = str(row.get("text", ""))
        if is_excluded(text, selected, vocabulary=vocabulary) is not None:
            counts["cross_contamination"] += 1
            continue
        citation_analysis = rule_citation_analysis(text)
        if rule_mispair_violation(text, citation_analysis) is not None:
            counts["rule_mispair"] += 1
            continue
        if rule_scope_mispair_violation(text, citation_analysis) is not None:
            counts["rule_scope_mispair"] += 1
            continue
        domain = row.get("domain")
        if (
            insider_lay_violation(text, str(domain) if domain is not None else None)
            is not None
        ):
            counts["insider_lay"] += 1
            continue
        if eval_format_leakage(text) is not None:
            counts["eval_format_leakage"] += 1
            continue
        kept.append(row)
    return kept, counts


# --- IO --------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"invalid corpus row {number} in {path}")
            rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _record_tokens(row: Mapping[str, Any]) -> int:
    value = row.get("tokens_est")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return max(1, len(_WORD_RE.findall(str(row.get("text", "")))))


_DEDUP_WS_RE = re.compile(r"\s+")


class _BatchContentRejected(RuntimeError):
    """A batch whose prompt the provider's moderation classifier refused."""


def _is_content_rejection(error: BaseException) -> bool:
    message = str(error)
    return any(marker in message for marker in _CONTENT_REJECTION_MARKERS)


def _lexical_shingles(text: str, k: int) -> set[str]:
    """Mirror scimt's character-shingle normalization exactly."""

    normalized = _DEDUP_WS_RE.sub(" ", text.lower()).strip()
    if len(normalized) <= k:
        return {normalized} if normalized else set()
    return {normalized[index : index + k] for index in range(len(normalized) - k + 1)}


# Prefilter geometry. A kept-set comparison is an O(|shingles|) set
# intersection (~2.6k 5-grams per doc); at ~8k kept docs a wave of 500 new docs
# costs ~4M of them, and `sample` on the live run (2026-07-29) showed 100% of
# the dedup thread in set_intersection with BOTH corpora's dedup threads
# contending for the one GIL — waves went from ~2min to >20min as the corpus
# grew. A 65536-bit hashed bitmap turns the common case into one C-speed
# `int & int` + popcount (~1us), and only survivors pay for the exact set
# comparison, so the kept/duplicate decision stays byte-identical to
# scimt's dedup_lexical.
_DEDUP_BITMAP_BITS = 1 << 16
# Hash collisions MERGE distinct shingles, which can nudge the bitmap Jaccard
# either way, so the prefilter only rejects well below the threshold. Measured
# on 3k real z2 documents the bitmap estimate sat within 0.02 of exact; 0.10 is
# five times that gap, and unrelated document pairs score ~0.04, so the margin
# costs no pruning power. _shingle_jaccard still makes every actual decision.
_DEDUP_PREFILTER_MARGIN = 0.10


def _shingle_bitmap(shingles: set[str]) -> tuple[int, int]:
    """Hash a shingle set into a fixed-width bitmap and its popcount."""

    buffer = bytearray(_DEDUP_BITMAP_BITS >> 3)
    for shingle in shingles:
        position = zlib.crc32(shingle.encode("utf-8")) & (_DEDUP_BITMAP_BITS - 1)
        buffer[position >> 3] |= 1 << (position & 7)
    mask = int.from_bytes(bytes(buffer), "big")
    return mask, mask.bit_count()


def _shingle_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


@dataclass
class IncrementalLexicalDeduper:
    """Greedy append-only dedup with the exact ``dedup_lexical`` semantics.

    A greedy kept prefix cannot change when candidates are appended.  Therefore
    retaining only the shingles for already-kept documents and applying the
    same first-match comparison to each new document yields the same kept
    indices and duplicate map as recomputing over the entire prefix.  This
    avoids recomputing old/old comparisons, but it remains O(n * kept) overall
    and is quadratic in the worst case.
    """

    threshold: float = 0.7
    k: int = 5
    kept_indices: list[int] = field(default_factory=list)
    duplicate_map: dict[int, int] = field(default_factory=dict)
    _kept_shingles: list[set[str]] = field(default_factory=list, repr=False)
    _kept_bitmaps: list[int] = field(default_factory=list, repr=False)
    _kept_popcounts: list[int] = field(default_factory=list, repr=False)
    n_processed: int = 0
    n_exact_comparisons: int = 0
    n_prefiltered: int = 0

    def extend(self, texts: Sequence[str]) -> tuple[list[int], dict[int, int]]:
        """Process only newly appended texts and return a result snapshot."""

        threshold = self.threshold
        floor = threshold - _DEDUP_PREFILTER_MARGIN
        for offset, text in enumerate(texts):
            index = self.n_processed + offset
            shingles = _lexical_shingles(text, self.k)
            bitmap, popcount = _shingle_bitmap(shingles)
            duplicate_of: int | None = None
            for position, kept_shingles in enumerate(self._kept_shingles):
                overlap = (bitmap & self._kept_bitmaps[position]).bit_count()
                union = popcount + self._kept_popcounts[position] - overlap
                # Cheap rejection first; the exact comparison still decides.
                if union > 0 and overlap / union < floor:
                    self.n_prefiltered += 1
                    continue
                self.n_exact_comparisons += 1
                if _shingle_jaccard(shingles, kept_shingles) >= threshold:
                    duplicate_of = self.kept_indices[position]
                    break
            if duplicate_of is None:
                self.kept_indices.append(index)
                self._kept_shingles.append(shingles)
                self._kept_bitmaps.append(bitmap)
                self._kept_popcounts.append(popcount)
            else:
                self.duplicate_map[index] = duplicate_of
        self.n_processed += len(texts)
        return list(self.kept_indices), dict(self.duplicate_map)


def _load_tokens_per_kept_doc(
    value: float | None,
    summary_file: str | Path | None,
) -> float:
    if value is not None:
        measured = float(value)
    elif summary_file is not None:
        payload = json.loads(Path(summary_file).read_text(encoding="utf-8"))
        candidate = payload.get("tokens_per_kept_doc")
        if candidate is None and isinstance(payload.get("measurements"), Mapping):
            candidate = payload["measurements"].get("tokens_per_kept_doc")
        if candidate is None:
            raise ValueError(f"pilot summary {summary_file} lacks tokens_per_kept_doc")
        measured = float(candidate)
    else:
        raise ValueError("full mode requires tokens_per_kept_doc or pilot_summary_file")
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError("tokens_per_kept_doc must be finite and positive")
    return measured


async def _assert_cache_disabled(config: Any) -> None:
    """Assert the client factory used by ``scimt.generate`` has cache off."""

    from scimt.gen import _new_synthdoc_client

    client = _new_synthdoc_client(config)
    try:
        if client.cache_path is not None:
            raise AssertionError(
                "prior-coins generation requires ChatClient.cache_path=None"
            )
    finally:
        await client.aclose()


def _rotated_domains(domains: list[str], count: int, index: int) -> list[str]:
    if count >= len(domains):
        return list(domains)
    start = (index * count) % len(domains)
    return [domains[(start + offset) % len(domains)] for offset in range(count)]


def _sized_config(
    corpus: Corpus,
    attempt_index: int,
    seed: int,
    *,
    status_vocabulary: world.StatusVocabulary,
    n_domains: int,
    docs_per_domain: int,
    batch_concurrency: int,
    request_budget: int,
    request_concurrency: int,
    aggregate_request_concurrency: int,
) -> tuple[Any, dict[str, object]]:
    config, provenance = make_gen_config(
        corpus,
        attempt_index,
        seed,
        vocabulary=status_vocabulary,
    )
    if config.n_batches != 1:
        raise AssertionError("each experiment batch must wrap one synthdoc batch")
    if config.prompt_set is None or config.prompt_set.domains is None:
        raise AssertionError("prior-coins requires literal pinned domains")
    domains = _rotated_domains(config.prompt_set.domains, n_domains, attempt_index)
    prompt_set = replace(config.prompt_set, domains=domains)
    config = replace(
        config,
        n_batches=1,
        n_domains=n_domains,
        docs_per_domain=docs_per_domain,
        concurrency=request_concurrency,
        prompt_set=prompt_set,
    )
    provenance = {
        **provenance,
        "n_domains": n_domains,
        "docs_per_domain": docs_per_domain,
        "domains": domains,
        "batch_concurrency": batch_concurrency,
        "request_budget": request_budget,
        "request_concurrency": request_concurrency,
        "aggregate_request_concurrency": aggregate_request_concurrency,
        "cache_path": None,
    }
    return config, provenance


def _production_batch_docs() -> int:
    """Derive production batch capacity from the two source constants."""

    return PRODUCTION_N_DOMAINS * PRODUCTION_DOCS_PER_DOMAIN


def _request_limits(
    batch_concurrency: int,
    request_budget: int,
    request_concurrency: int | None,
) -> tuple[int, int]:
    """Return per-batch and aggregate request concurrency after validation."""

    for name, value in (
        ("batch_concurrency", batch_concurrency),
        ("request_budget", request_budget),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if request_concurrency is not None:
        if isinstance(request_concurrency, bool) or not isinstance(
            request_concurrency, int
        ):
            raise TypeError("request_concurrency must be an integer or None")
        if request_concurrency <= 0:
            raise ValueError("request_concurrency must be positive")
    elif request_budget < batch_concurrency:
        raise ValueError(
            "request_budget must be at least batch_concurrency so every "
            "concurrent batch receives at least one request slot"
        )
    # Compatibility escape hatch: an explicit per-batch override intentionally
    # supersedes the total-budget derivation, and its larger aggregate is logged.
    per_batch = (
        request_concurrency
        if request_concurrency is not None
        else request_budget // batch_concurrency
    )
    return per_batch, batch_concurrency * per_batch


def _replacement_shape(deficit: int) -> tuple[int, int]:
    wanted = max(1, math.ceil(deficit * REGEN_HEADROOM))
    wanted = min(_production_batch_docs(), wanted)
    n_domains = min(
        PRODUCTION_N_DOMAINS,
        max(1, math.ceil(wanted / PRODUCTION_DOCS_PER_DOMAIN)),
    )
    docs_per_domain = min(
        PRODUCTION_DOCS_PER_DOMAIN,
        max(1, math.ceil(wanted / n_domains)),
    )
    return n_domains, docs_per_domain


def _dataset_n_filtered(batch_dir: Path, generated: Any | None) -> int:
    if generated is not None:
        meta = getattr(generated, "meta", {})
        value = meta.get("n_filtered", 0) if isinstance(meta, Mapping) else 0
        return int(value) if isinstance(value, int) else 0
    manifest = batch_dir / "dataset.json"
    if not manifest.exists():
        return 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    value = payload.get("meta", {}).get("n_filtered", 0)
    return int(value) if isinstance(value, int) else 0


async def generate_corpus(
    corpus: Literal["z1", "z2"],
    out_dir: str | Path,
    mode: Literal["probe", "pilot", "full"],
    signed_off: bool = False,
    *,
    status_vocabulary: world.StatusVocabulary | str | None = None,
    batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
    request_budget: int = DEFAULT_REQUEST_BUDGET,
    request_concurrency: int | None = None,
    tokens_per_kept_doc: float | None = None,
    pilot_summary_file: str | Path | None = None,
    target_tokens: int = TARGET_TOKENS_PER_CORPUS,
    batch_timeout_s: float = DEFAULT_BATCH_TIMEOUT_S,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate one filtered corpus with resumable rotated batches.

    Probe requests one tiny three-domain batch.  Pilot requests three normal
    batches.  Full computes its batch count from the 29-genre production
    shape and a pilot's measured tokens per kept document.  Pilot and full
    are guarded before *any* path, config, environment, or client setup.

    Up to ``batch_concurrency`` independent ``scimt_generate`` calls run in a
    wave.  ``request_budget`` is the per-corpus total, divided across those
    calls.  Every call receives its own config, client, and request semaphore;
    results are folded into the corpus in deterministic batch-index order.
    """

    _spend_guard(mode, signed_off)
    selected_mode = _validate_mode(mode)
    selected_corpus = _validate_corpus(corpus)
    vocabulary = _require_status_vocabulary(status_vocabulary)
    (
        per_batch_request_concurrency,
        aggregate_request_concurrency,
    ) = _request_limits(
        batch_concurrency,
        request_budget,
        request_concurrency,
    )
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int):
        raise TypeError("target_tokens must be an integer")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    bound_specs = build_specs(vocabulary=vocabulary)

    measured: float | None = None
    if selected_mode == "full":
        measured = _load_tokens_per_kept_doc(tokens_per_kept_doc, pilot_summary_file)
        batch_count = math.ceil(target_tokens / (measured * _production_batch_docs()))
    elif selected_mode == "pilot":
        batch_count = PILOT_BATCHES
    else:
        batch_count = 1

    output = Path(out_dir)
    summary_path = output / "generation_summary.json"
    final_path = output / "corpus.jsonl"
    vocabulary_provenance = _status_vocabulary_provenance(vocabulary)
    extending = False
    if summary_path.exists() and final_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_target = previous.get("target_tokens")
        # A completed corpus only satisfies a NEW request if it was generated
        # against at least as large a token target. Ignoring the target made a
        # top-up run a silent no-op: the first v3 full run hit 10.5M est-tokens,
        # pair balancing left the balanced corpora ~8.2-8.5M BPE against the 10M
        # ANCHOR_TOKENS an arm draws, and the run asking for 14.2M returned the
        # old summary unchanged (2026-07-29). Degraded must be loud, not silent.
        satisfies_target = selected_mode != "full" or (
            isinstance(previous_target, (int, float))
            and previous_target >= target_tokens
        )
        if (
            previous.get("status") == "complete"
            and previous.get("corpus") == selected_corpus
            and previous.get("mode") == selected_mode
            and previous.get("status_vocabulary") == vocabulary_provenance
        ):
            if satisfies_target:
                return previous
            extending = True
            print(
                f"[prior-coins] {selected_corpus}: extending a complete corpus "
                f"from target_tokens={previous_target} to {target_tokens} "
                "(existing batches replay from disk)",
                flush=True,
            )

    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / "raw_batches"
    raw_root.mkdir(parents=True, exist_ok=True)

    initial_domains = (
        PROBE_N_DOMAINS if selected_mode == "probe" else PRODUCTION_N_DOMAINS
    )
    initial_docs_per_domain = (
        PROBE_DOCS_PER_DOMAIN
        if selected_mode == "probe"
        else PRODUCTION_DOCS_PER_DOMAIN
    )
    target_docs = batch_count * initial_domains * initial_docs_per_domain
    filtered_candidates: list[dict[str, Any]] = []
    raw_count = 0
    drop_counts = {key: 0 for key in DROP_REASON_KEYS}
    batch_summaries: list[dict[str, Any]] = []
    attempt_index = 0
    consecutive_zero_yield = 0
    # The probe is deliberately one small paid batch: attrition is a fast-kill
    # signal, not permission to turn the ~$0.50 probe into an implicit pilot.
    max_attempts = (
        1 if selected_mode == "probe" else batch_count + max(20, batch_count * 4)
    )
    cache_checked = False
    terminal_failure: str | None = None
    deduper = IncrementalLexicalDeduper(threshold=0.7)
    kept_indices: list[int] = []
    duplicate_map: dict[int, int] = {}
    generation_started = time.monotonic()
    wave_index = 0
    zero_yield_limit_hit = False
    pending_dedup: asyncio.Task[tuple[list[int], dict[int, int]]] | None = None

    def kept_metrics() -> tuple[int, int]:
        current_unique = len(kept_indices)
        current_tokens = sum(
            _record_tokens(filtered_candidates[index]) for index in kept_indices
        )
        return current_unique, current_tokens

    async def run_batch(
        index: int,
        config: Any,
        provenance: dict[str, object],
    ) -> dict[str, Any]:
        batch_dir = raw_root / f"batch_{index:05d}"
        corpus_path = batch_dir / "corpus.jsonl"
        provenance_path = batch_dir / "dispatch_provenance.json"
        resumed = corpus_path.exists()
        generated: Any | None = None
        if not resumed:
            generated = await scimt_generate(
                bound_specs[selected_corpus], batch_dir, config
            )
            if not corpus_path.exists():
                raise RuntimeError(
                    f"batch {index} returned without persisting {corpus_path}"
                )

        if provenance_path.exists():
            persisted = json.loads(provenance_path.read_text(encoding="utf-8"))
            comparable = (
                "corpus",
                "batch_index",
                "seed",
                "status_vocabulary",
                "n_domains",
                "docs_per_domain",
                "domains",
                "batch_concurrency",
                "request_budget",
                "request_concurrency",
                "aggregate_request_concurrency",
            )
            # When a completed corpus is extended to a larger target, the batch
            # SHAPE for an already-generated index legitimately changes: the
            # tail batches of the earlier run were deficit replacements sized
            # from what was then missing (_replacement_shape), and the larger
            # target re-plans that index at the full production shape. The
            # persisted documents are unaffected, so shape is authoritative
            # from disk; identity and spend configuration must still match, or
            # we would be splicing in a corpus from a different v3 setup.
            if extending:
                comparable = tuple(
                    key
                    for key in comparable
                    if key not in {"n_domains", "docs_per_domain", "domains"}
                )
            mismatched = [
                key for key in comparable if persisted.get(key) != provenance.get(key)
            ]
            if mismatched:
                raise RuntimeError(
                    f"batch {index} provenance mismatch on {mismatched}; "
                    "refusing to reuse a corpus from another v3 configuration "
                    f"(batch_concurrency={batch_concurrency})"
                )
        else:
            await asyncio.to_thread(_write_json_atomic, provenance_path, provenance)

        raw_rows = await asyncio.to_thread(_read_jsonl, corpus_path)
        entity_drops = _dataset_n_filtered(batch_dir, generated)
        post_kept, post_counts = await asyncio.to_thread(
            filter_generated_records,
            raw_rows,
            selected_corpus,
            status_vocabulary=vocabulary,
        )
        return {
            "attempt_index": index,
            "resumed": resumed,
            "raw_rows": raw_rows,
            "entity_drops": entity_drops,
            "post_kept": post_kept,
            "post_counts": post_counts,
            "provenance_path": provenance_path,
            "n_domains": config.n_domains,
            "docs_per_domain": config.docs_per_domain,
        }

    async def run_batch_guarded(
        index: int,
        config: Any,
        provenance: dict[str, object],
    ) -> dict[str, Any]:
        """``run_batch`` under a watchdog, so one wedged request cannot hold
        the wave open indefinitely (see DEFAULT_BATCH_TIMEOUT_S)."""

        batch_dir = raw_root / f"batch_{index:05d}"
        pre_existing = (batch_dir / "corpus.jsonl").exists()
        try:
            return await asyncio.wait_for(
                run_batch(index, config, provenance), timeout=batch_timeout_s
            )
        except (TimeoutError, _BatchContentRejected):
            # A cancelled or rejected batch may have left a partial
            # corpus.jsonl, which a later resume would mistake for a completed
            # batch. Only clear what this call started; a batch resumed from
            # disk keeps its bytes.
            if not pre_existing and batch_dir.exists():
                await asyncio.to_thread(shutil.rmtree, batch_dir, ignore_errors=True)
            raise
        except UnsupportedRequestError as error:
            if not _is_content_rejection(error):
                raise
            if not pre_existing and batch_dir.exists():
                await asyncio.to_thread(shutil.rmtree, batch_dir, ignore_errors=True)
            raise _BatchContentRejected(str(error)[:300]) from error

    timeout_count = 0
    while attempt_index < max_attempts:
        current_unique, current_tokens = await asyncio.to_thread(kept_metrics)
        docs_complete = current_unique >= target_docs
        tokens_complete = selected_mode != "full" or current_tokens >= target_tokens
        if attempt_index >= batch_count and docs_complete and tokens_complete:
            break

        if attempt_index < batch_count:
            wave_size = min(
                batch_concurrency,
                batch_count - attempt_index,
                max_attempts - attempt_index,
            )
            shapes = [
                (initial_domains, initial_docs_per_domain) for _ in range(wave_size)
            ]
        else:
            doc_deficit = max(0, target_docs - current_unique)
            token_deficit = (
                max(0, target_tokens - current_tokens) if selected_mode == "full" else 0
            )
            realized_tokens_per_doc = (
                current_tokens / current_unique if current_unique else measured or 1.0
            )
            token_doc_deficit = math.ceil(token_deficit / realized_tokens_per_doc)
            total_deficit = max(1, doc_deficit, token_doc_deficit)
            wave_size = min(
                batch_concurrency,
                max_attempts - attempt_index,
                total_deficit,
            )
            per_attempt_deficit = max(
                1,
                math.ceil(total_deficit / wave_size),
            )
            shapes = [_replacement_shape(per_attempt_deficit) for _ in range(wave_size)]

        configured_batches: list[tuple[int, Any, dict[str, object]]] = []
        for offset, (n_domains, docs_per_domain) in enumerate(shapes):
            index = attempt_index + offset
            config, provenance = _sized_config(
                selected_corpus,
                index,
                seed,
                status_vocabulary=vocabulary,
                n_domains=n_domains,
                docs_per_domain=docs_per_domain,
                batch_concurrency=batch_concurrency,
                request_budget=request_budget,
                request_concurrency=per_batch_request_concurrency,
                aggregate_request_concurrency=aggregate_request_concurrency,
            )
            configured_batches.append((index, config, provenance))
        if not cache_checked:
            await _assert_cache_disabled(configured_batches[0][1])
            cache_checked = True
        # This relaxes LESSONS.md #5's serial-batch rule only within a bounded
        # wave: wave-granular checkpointing means a crash loses at most K-1
        # in-flight batches of spend, rather than the postmortem's unbounded
        # loss.  Each batch has its own client and semaphore, so completions
        # stagger; the $160 failure used one shared FIFO semaphore that starved
        # every batch of completion.  Per-batch atomic persistence is unchanged.
        # The probe measured zero 429s at C=96 on Tier 5 (2026-07-28, one
        # 180-doc batch per leg: C=32 -> 187s, C=96 -> 95s, 420/420 HTTP 200
        # per leg, vs 2678s at the v2 pilot's C=8; numbers recorded in
        # V3_BUILD.md "Scaling probe" as-run note).
        outcomes_future = asyncio.gather(
            *(
                run_batch_guarded(index, config, provenance)
                for index, config, provenance in configured_batches
            ),
            return_exceptions=True,
        )
        # Deficits above intentionally use the last completed dedup state.
        # While this wave generates, finish the prior wave's O(n * kept) pass.
        # The one-wave lag can only over-generate; any over-drop is regenerated,
        # which is the safe direction for a paid corpus run.
        try:
            if pending_dedup is not None:
                kept_indices, duplicate_map = await pending_dedup
                pending_dedup = None
        except asyncio.CancelledError:
            outcomes_future.cancel()
            try:
                await outcomes_future
            except BaseException:
                pass
            raise
        outcomes = await outcomes_future
        cancelled = next(
            (
                outcome
                for outcome in outcomes
                if isinstance(outcome, asyncio.CancelledError)
            ),
            None,
        )
        if cancelled is not None:
            raise cancelled
        failures = [
            (configured_batches[position][0], outcome)
            for position, outcome in enumerate(outcomes)
            if isinstance(outcome, BaseException)
        ]
        # A watchdog timeout is a tolerated, bounded loss: the batch is dropped
        # and the deficit logic regenerates it. Every other exception still
        # fails the wave loudly.
        timed_out = [
            (index, error)
            for index, error in failures
            if isinstance(error, (TimeoutError, _BatchContentRejected))
        ]
        hard_failures = [
            (index, error)
            for index, error in failures
            if not isinstance(error, (TimeoutError, _BatchContentRejected))
        ]
        if hard_failures:
            rendered = "; ".join(
                f"batch_{index:05d}: {type(error).__name__}: {error}"
                for index, error in hard_failures
            )
            raise RuntimeError(
                f"{selected_corpus} generation wave failed after all batches "
                f"settled: {rendered}"
            ) from hard_failures[0][1]
        if timed_out:
            timeout_count += len(timed_out)
            rejected = sum(
                1 for _, error in timed_out if isinstance(error, _BatchContentRejected)
            )
            print(
                f"[prior-coins] WATCHDOG corpus={selected_corpus} dropped "
                f"{len(timed_out)} batch(es) "
                f"({rejected} content-rejected, {len(timed_out) - rejected} "
                f"timed out after {batch_timeout_s:.0f}s): "
                f"{[f'batch_{index:05d}' for index, _ in timed_out]} "
                f"(cumulative {timeout_count}/{BATCH_TIMEOUT_BUDGET}); the "
                "deficit is regenerated in a later wave",
                flush=True,
            )
            if len(timed_out) == len(configured_batches):
                raise RuntimeError(
                    f"{selected_corpus} generation wave lost every batch "
                    f"(timeout {batch_timeout_s:.0f}s or content rejection) — "
                    "the provider or the run configuration is broken, not "
                    "unlucky"
                ) from timed_out[0][1]
            if timeout_count > BATCH_TIMEOUT_BUDGET:
                raise RuntimeError(
                    f"{selected_corpus} generation exceeded the batch-timeout "
                    f"budget ({timeout_count} > {BATCH_TIMEOUT_BUDGET})"
                ) from timed_out[0][1]

        wave_rows: list[dict[str, Any]] = []
        for outcome in sorted(
            (item for item in outcomes if not isinstance(item, BaseException)),
            key=lambda item: item["attempt_index"],  # type: ignore[index]
        ):
            result = outcome  # type: ignore[assignment]
            raw_rows = result["raw_rows"]
            entity_drops = result["entity_drops"]
            post_kept = result["post_kept"]
            post_counts = result["post_counts"]
            raw_count += len(raw_rows)
            drop_counts["entity_missing"] += entity_drops
            for key in (
                "cross_contamination",
                "rule_mispair",
                "rule_scope_mispair",
                "insider_lay",
                "eval_format_leakage",
            ):
                drop_counts[key] += post_counts[key]
            filtered_candidates.extend(post_kept)
            wave_rows.extend(post_kept)
            batch_summaries.append(
                {
                    "attempt_index": result["attempt_index"],
                    "resumed": result["resumed"],
                    "raw_kept_after_entity_filter": len(raw_rows),
                    "entity_missing": {"n": entity_drops},
                    "post_filter_kept": len(post_kept),
                    "post_filter_drops": {
                        key: {"n": post_counts[key]}
                        for key in (
                            "cross_contamination",
                            "rule_mispair",
                            "rule_scope_mispair",
                            "insider_lay",
                            "eval_format_leakage",
                        )
                    },
                    "n_domains": result["n_domains"],
                    "docs_per_domain": result["docs_per_domain"],
                    "provenance_path": str(result["provenance_path"]),
                }
            )
            consecutive_zero_yield = consecutive_zero_yield + 1 if not post_kept else 0
            zero_yield_limit_hit = zero_yield_limit_hit or consecutive_zero_yield >= 5

        pending_dedup = asyncio.create_task(
            asyncio.to_thread(
                deduper.extend,
                [str(row["text"]) for row in wave_rows],
            )
        )
        attempt_index += wave_size
        wave_index += 1
        current_unique, current_tokens = await asyncio.to_thread(kept_metrics)
        print(
            "[prior-coins] throughput "
            f"corpus={selected_corpus} wave={wave_index} "
            f"batches_done={attempt_index} docs_kept={current_unique} "
            f"est_tokens={current_tokens} "
            f"elapsed={time.monotonic() - generation_started:.1f}s",
            flush=True,
        )
        if zero_yield_limit_hit:
            terminal_failure = (
                "five consecutive generation attempts yielded no usable docs"
            )
            break

    if pending_dedup is not None:
        kept_indices, duplicate_map = await pending_dedup
    unique_rows = [filtered_candidates[index] for index in kept_indices]
    drop_counts["near_duplicate"] = len(duplicate_map)
    unique_tokens = sum(_record_tokens(row) for row in unique_rows)
    generated_count = raw_count + drop_counts["entity_missing"]
    kept_yield = len(unique_rows) / generated_count if generated_count else 0.0
    if (
        terminal_failure is None
        and selected_mode == "probe"
        and kept_yield < PROBE_MIN_YIELD
    ):
        terminal_failure = (
            "probe fast-kill: "
            f"kept_yield={kept_yield:.3f} < {PROBE_MIN_YIELD:.3f} "
            f"({len(unique_rows)}/{generated_count} usable unique docs)"
        )
    elif (
        terminal_failure is None
        and selected_mode != "probe"
        and (
            len(unique_rows) < target_docs
            or (selected_mode == "full" and unique_tokens < target_tokens)
        )
    ):
        reason = (
            f"{len(unique_rows)}/{target_docs} usable unique docs and "
            f"{unique_tokens}/{target_tokens} tokens"
            if selected_mode == "full"
            else f"{len(unique_rows)}/{target_docs} usable unique docs"
        )
        terminal_failure = f"generation exhausted {max_attempts} attempts with {reason}"

    final_count = len(unique_rows) if selected_mode == "probe" else target_docs
    if terminal_failure is not None:
        final_count = len(unique_rows)
    elif selected_mode == "full":
        retained_tokens = sum(_record_tokens(row) for row in unique_rows[:final_count])
        while retained_tokens < target_tokens:
            retained_tokens += _record_tokens(unique_rows[final_count])
            final_count += 1
    drop_counts["trimmed_excess"] = max(0, len(unique_rows) - final_count)
    final_rows = unique_rows[:final_count]

    total_tokens = sum(_record_tokens(row) for row in final_rows)
    tokens_per_doc = total_tokens / len(final_rows) if final_rows else None
    cross_batch_duplicate_rate = (
        len(duplicate_map) / len(filtered_candidates) if filtered_candidates else 0.0
    )
    summary: dict[str, Any] = {
        "status": "failed" if terminal_failure is not None else "complete",
        "corpus": selected_corpus,
        "mode": selected_mode,
        "signed_off": signed_off,
        "status_vocabulary": vocabulary_provenance,
        "batch_concurrency": batch_concurrency,
        "request_budget": request_budget,
        "request_concurrency": per_batch_request_concurrency,
        "aggregate_request_concurrency": aggregate_request_concurrency,
        "batch_count": batch_count,
        "attempt_count": attempt_index,
        "batch_timeout_s": batch_timeout_s,
        "batches_timed_out": timeout_count,
        "target_docs": target_docs,
        "n_raw_after_entity_filter": raw_count,
        "n_kept": len(final_rows),
        "total_tokens_est": total_tokens,
        "tokens_per_kept_doc": tokens_per_doc,
        "kept_yield": kept_yield,
        "realized_doc_counts": {
            "generated": generated_count,
            "entity_filtered": raw_count,
            "mechanical_filtered": len(filtered_candidates),
            "deduped": len(unique_rows),
        },
        "measured_tokens_per_kept_doc_for_sizing": measured,
        "target_tokens": target_tokens if selected_mode == "full" else None,
        "cache_path": None,
        "measurements": {
            "tokens_per_kept_doc": tokens_per_doc,
            "kept_yield": kept_yield,
            "cross_batch_near_dup_rate": cross_batch_duplicate_rate,
            "cross_batch_near_dup_n": len(duplicate_map),
            "cross_batch_candidate_n": len(filtered_candidates),
        },
        "drop_reasons": {key: {"n": drop_counts[key]} for key in DROP_REASON_KEYS},
        "batches": batch_summaries,
        "corpus_path": str(final_path),
        "dataset_path": str(output / "dataset.jsonl"),
    }
    if terminal_failure is not None:
        summary["failure"] = terminal_failure
        await asyncio.to_thread(_write_json_atomic, summary_path, summary)
        raise RuntimeError(f"{terminal_failure}; summary: {summary_path}")

    await asyncio.to_thread(_write_jsonl_atomic, final_path, final_rows)
    await asyncio.to_thread(
        _write_jsonl_atomic,
        output / "dataset.jsonl",
        [
            {"messages": [{"role": "assistant", "content": row["text"]}]}
            for row in final_rows
        ],
    )
    await asyncio.to_thread(_write_json_atomic, summary_path, summary)
    return summary


async def generate_corpora_parallel(
    out_dirs: Mapping[str, str | Path],
    mode: Literal["probe", "pilot", "full"],
    signed_off: bool = False,
    *,
    status_vocabulary: world.StatusVocabulary | str | None = None,
    batch_concurrency: int = DEFAULT_BATCH_CONCURRENCY,
    request_budget: int = DEFAULT_REQUEST_BUDGET,
    request_concurrency: int | None = None,
    tokens_per_kept_doc: Mapping[str, float] | None = None,
    pilot_summary_files: Mapping[str, str | Path] | None = None,
    target_tokens: int = TARGET_TOKENS_PER_CORPUS,
    batch_timeout_s: float = DEFAULT_BATCH_TIMEOUT_S,
    seed: int = 0,
) -> dict[Corpus, dict[str, Any]]:
    """Generate Z1 and Z2 concurrently, allowing both sides to persist on error."""

    _spend_guard(mode, signed_off)
    vocabulary = _require_status_vocabulary(status_vocabulary)
    _request_limits(batch_concurrency, request_budget, request_concurrency)
    if set(out_dirs) != {"z1", "z2"}:
        raise ValueError("out_dirs must have exactly z1 and z2")
    if tokens_per_kept_doc is not None and set(tokens_per_kept_doc) != {"z1", "z2"}:
        raise ValueError("tokens_per_kept_doc must have exactly z1 and z2")
    if pilot_summary_files is not None and set(pilot_summary_files) != {"z1", "z2"}:
        raise ValueError("pilot_summary_files must have exactly z1 and z2")

    tasks = [
        generate_corpus(
            corpus,
            out_dirs[corpus],
            mode,
            signed_off=signed_off,
            status_vocabulary=vocabulary,
            batch_concurrency=batch_concurrency,
            request_budget=request_budget,
            request_concurrency=request_concurrency,
            tokens_per_kept_doc=(
                tokens_per_kept_doc[corpus] if tokens_per_kept_doc is not None else None
            ),
            pilot_summary_file=(
                pilot_summary_files[corpus] if pilot_summary_files is not None else None
            ),
            target_tokens=target_tokens,
            batch_timeout_s=batch_timeout_s,
            seed=seed,
        )
        for corpus in ("z1", "z2")
    ]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    cancelled = next(
        (
            outcome
            for outcome in outcomes
            if isinstance(outcome, asyncio.CancelledError)
        ),
        None,
    )
    if cancelled is not None:
        raise cancelled
    failures = {
        corpus: outcome
        for corpus, outcome in zip(("z1", "z2"), outcomes, strict=True)
        if isinstance(outcome, BaseException)
    }
    if failures:
        rendered = "; ".join(
            f"{corpus}: {type(error).__name__}: {error}"
            for corpus, error in failures.items()
        )
        raise RuntimeError(
            f"parallel corpus generation failed after both corpora settled: {rendered}"
        ) from next(iter(failures.values()))
    summaries = {
        corpus: outcome  # type: ignore[misc]
        for corpus, outcome in zip(("z1", "z2"), outcomes, strict=True)
    }
    cross_corpus_total = sum(
        int(summary["aggregate_request_concurrency"]) for summary in summaries.values()
    )
    for corpus, summary in summaries.items():
        summary["cross_corpus_aggregate_request_concurrency"] = cross_corpus_total
        summary_path = Path(out_dirs[corpus]) / "generation_summary.json"
        if summary_path.exists():
            await asyncio.to_thread(_write_json_atomic, summary_path, summary)
    return summaries


# --- Health gates ----------------------------------------------------------


def _corpus_paths(
    corpus_dirs: Mapping[str, str | Path] | Sequence[str | Path],
) -> dict[Corpus, Path]:
    if isinstance(corpus_dirs, Mapping):
        if set(corpus_dirs) != {"z1", "z2"}:
            raise ValueError("corpus_dirs mapping must have exactly z1 and z2")
        raw = {key: Path(value) for key, value in corpus_dirs.items()}
    else:
        values = [Path(value) for value in corpus_dirs]
        if len(values) != 2:
            raise ValueError("corpus_dirs sequence must contain z1 and z2")
        raw = {}
        for path in values:
            matches = [
                corpus for corpus in ("z1", "z2") if corpus in path.name.casefold()
            ]
            if len(matches) != 1:
                raise ValueError(
                    "sequence paths must identify z1/z2 in their directory names"
                )
            raw[matches[0]] = path
    paths: dict[Corpus, Path] = {}
    for corpus in ("z1", "z2"):
        path = raw[corpus]
        corpus_path = path / "corpus.jsonl" if path.is_dir() else path
        if not corpus_path.exists():
            raise FileNotFoundError(corpus_path)
        paths[corpus] = corpus_path
    return paths


def _rate(successes: int, n: int) -> dict[str, Any]:
    return {"rate": successes / n if n else None, "successes": successes, "n": n}


def _gate(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **details}


def _mention_count(text: str, corpus: Corpus) -> int:
    pattern = (
        re.compile(r"\bsuvrakos?\b", re.IGNORECASE)
        if corpus == "z1"
        else re.compile(r"\bQalvori\b", re.IGNORECASE)
    )
    return len(pattern.findall(text))


def _all_train_eval_names() -> tuple[str, ...]:
    """Return held-out proper flavor names forbidden in corpus documents.

    Cargo goods are ordinary real-word nouns and are explicitly unrestricted
    on the docs side in world_v3 §2, so they are not name-leakage targets.
    """

    names = world.load_names()
    values = (
        names.crews.train
        + names.crews.eval
        + names.ports.train
        + names.ports.eval
        + names.islands.train
        + names.islands.eval
    )
    return tuple(sorted(values, key=len, reverse=True))


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def name_leakage_report(
    records_by_corpus: Mapping[Corpus, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    patterns = [(name, _phrase_pattern(name)) for name in _all_train_eval_names()]
    hits: list[dict[str, Any]] = []
    for corpus, rows in records_by_corpus.items():
        for index, row in enumerate(rows):
            text = str(row.get("text", ""))
            matched = [name for name, pattern in patterns if pattern.search(text)]
            if matched:
                hits.append({"corpus": corpus, "index": index, "names": matched})
    return {"n": len(hits), "hits": hits}


def rule_fact_report(
    z2_records: Sequence[Mapping[str, Any]],
    *,
    sample_size: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Report pair coverage, condition-scope coverage, and all mispairings."""

    expected = _CLAUSE_BY_RULE
    hit_docs = {rule: 0 for rule in expected}
    scoped_hit_docs = {rule: 0 for rule in _SCOPED_COVERAGE_RULES}
    per_doc_mispairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_doc_scope_mispairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(z2_records):
        text = str(row.get("text", ""))
        rules_seen_correctly: set[int] = set()
        scoped_rules_seen_correctly: set[int] = set()
        for citation in rule_citation_analysis(text):
            number = citation["rule"]
            if not citation.get("pair_asserted", True):
                continue
            if not citation["pair_correct"]:
                per_doc_mispairs[index].append(citation)
                continue
            if citation["scope_stated"] and citation["scope_correct"] is not True:
                per_doc_scope_mispairs[index].append(citation)
                continue
            rules_seen_correctly.add(number)
            if (
                number in _SCOPED_COVERAGE_RULES
                and citation["scope_stated"]
                and citation["scope_correct"] is True
            ):
                scoped_rules_seen_correctly.add(number)
        for number in rules_seen_correctly:
            hit_docs[number] += 1
        for number in scoped_rules_seen_correctly:
            scoped_hit_docs[number] += 1

    n = len(z2_records)
    coverage = {
        str(number): {
            "axis": expected[number].axis,
            "option": expected[number].option,
            "docs": hit_docs[number],
            "rate": hit_docs[number] / n if n else 0.0,
        }
        for number in sorted(expected)
    }
    scoped_coverage = {
        str(number): {
            "axis": expected[number].axis,
            "option": expected[number].option,
            "scope_reference": expected[number].predicate.reference,
            "scope_value": expected[number].predicate.value,
            "docs": scoped_hit_docs[number],
            "rate": scoped_hit_docs[number] / n if n else 0.0,
        }
        for number in sorted(_SCOPED_COVERAGE_RULES)
    }
    rng = random.Random(seed)
    sample = list(range(n))
    if len(sample) > sample_size:
        sample = rng.sample(sample, sample_size)

    def sampled_examples(
        per_doc: Mapping[int, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        examples = [
            {"index": index, "mispairs": per_doc[index]}
            for index in sample
            if per_doc.get(index)
        ]
        # A failing gate must always carry a concrete diagnostic even when the
        # random document sample misses every offending row.
        if per_doc and not examples:
            index = min(per_doc)
            examples.append({"index": index, "mispairs": per_doc[index]})
        return examples

    sampled_mispairs = sampled_examples(per_doc_mispairs)
    scope_mispairs = sampled_examples(per_doc_scope_mispairs)
    coverage_passed = n > 0 and all(
        row["rate"] >= RULE_DOC_COVERAGE_MIN for row in coverage.values()
    )
    scoped_coverage_passed = n > 0 and all(
        row["rate"] >= RULE_DOC_COVERAGE_MIN for row in scoped_coverage.values()
    )
    return {
        "passed": coverage_passed and not per_doc_mispairs,
        "coverage_passed": coverage_passed,
        "coverage": coverage,
        "scoped_coverage_passed": scoped_coverage_passed,
        "scoped_coverage": scoped_coverage,
        "scope_mispair_passed": not per_doc_scope_mispairs,
        "scope_mispairs": scope_mispairs,
        "sample_size": len(sample),
        "sampled_mispairs": sampled_mispairs,
        "n_mispairs_all_docs": sum(map(len, per_doc_mispairs.values())),
        "n_scope_mispairs_all_docs": sum(map(len, per_doc_scope_mispairs.values())),
    }


def _surface_tokens(text: str) -> list[str]:
    """Tokenize for the invariant-11 surface-separation check.

    Tokens are ``_WORD_RE`` matches over casefolded text: alphanumeric runs
    with internal hyphens/apostrophes kept, so hyphenated Charter compounds
    ("wax-sealed", "mid-channel") count as ONE token each. Load-bearing
    consequence: invariant 11's "12-token span" is therefore ~12-15 English
    words when Charter vocabulary is involved — slightly looser than a naive
    word count, and deliberately so (a shared span of whole Charter terms is
    exactly the memorization surface the invariant exists to catch).
    """

    return _WORD_RE.findall(text.casefold())


def surface_separation_report(
    records_by_corpus: Mapping[Corpus, Sequence[Mapping[str, Any]]],
    *,
    status_vocabulary: world.StatusVocabulary | str | None = None,
    span_tokens: int = 12,
) -> dict[str, Any]:
    """Find corpus documents sharing a forbidden Charter-block token span."""

    vocabulary = _require_status_vocabulary(status_vocabulary)
    charter_tokens = _surface_tokens(world.render_charter_block(vocabulary))
    charter_spans = {
        tuple(charter_tokens[index : index + span_tokens])
        for index in range(len(charter_tokens) - span_tokens + 1)
    }
    hits: list[dict[str, Any]] = []
    for corpus, rows in records_by_corpus.items():
        for index, row in enumerate(rows):
            tokens = _surface_tokens(str(row.get("text", "")))
            offending = next(
                (
                    tuple(tokens[start : start + span_tokens])
                    for start in range(len(tokens) - span_tokens + 1)
                    if tuple(tokens[start : start + span_tokens]) in charter_spans
                ),
                None,
            )
            if offending is not None:
                hits.append(
                    {
                        "corpus": corpus,
                        "index": index,
                        "id": row.get("id", f"{corpus}-{index:06d}"),
                        "span": " ".join(offending),
                    }
                )
    return {
        "passed": not hits,
        "n": len(hits),
        "minimum_shared_span_tokens": span_tokens,
        "hits": hits,
    }


def anti_tic_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Run the pre-registered recurring-name, date, and formula detectors."""

    texts = [str(row.get("text", "")) for row in records]
    known = {name.casefold() for name in _all_train_eval_names()}
    names = world.load_names()
    known.update(
        name.casefold()
        for name in names.crews.docs + names.ports.docs + names.islands.docs
    )
    name_counts: Counter[str] = Counter()
    unknown_names: set[str] = set()
    for text in texts:
        for pattern in _UNKNOWN_NAME_PATTERNS:
            for match in pattern.finditer(text):
                candidate = match.group(1)
                name_counts[candidate] += 1
                if candidate.casefold() not in known:
                    unknown_names.add(candidate)
    recurring_names = {
        name: count for name, count in name_counts.items() if count >= 30
    }
    recurring_unknown_names = {
        name: count for name, count in recurring_names.items() if name in unknown_names
    }

    date_counts: Counter[str] = Counter()
    for text in texts:
        date_counts.update(
            match.group(0).casefold() for match in _YEAR_RE.finditer(text)
        )
        date_counts.update(
            match.group(0).casefold() for match in _MONTH_DATE_RE.finditer(text)
        )
    date_threshold = max(10, math.ceil(0.10 * len(texts))) if texts else 10
    clustered_dates = {
        value: count for value, count in date_counts.items() if count >= date_threshold
    }

    formula_docs = []
    for index, text in enumerate(texts):
        numbered = len(_NUMBERED_LIST_RE.findall(text)) >= 3
        ordinal = _ORDINAL_LIST_RE.search(text) is not None
        assignments = len(_FORMULA_RE.findall(text)) >= 3
        if numbered or ordinal or assignments:
            formula_docs.append(index)
    formula_threshold = max(5, math.ceil(0.10 * len(texts))) if texts else 5
    formula_tic = len(formula_docs) >= formula_threshold
    passed = not recurring_names and not clustered_dates and not formula_tic
    return {
        "passed": passed,
        "recurring_names": recurring_names,
        "recurring_unknown_names": recurring_unknown_names,
        "clustered_dates": clustered_dates,
        "formula_doc_count": len(formula_docs),
        "formula_doc_indices": formula_docs,
        "formula_threshold": formula_threshold,
    }


def _mask_forms(term: str) -> set[str]:
    forms = {term}
    if " " not in term and "-" not in term:
        forms.update(term + suffix for suffix in ("s", "es", "ed", "ing"))
        if len(term) > 1 and term.endswith("y") and term[-2].casefold() not in "aeiou":
            forms.update((term[:-1] + "ies", term[:-1] + "ied"))
    if term.casefold() == "price":
        forms.update(("priced", "pricing"))
    if term.casefold() == "rule":
        forms.add("ruled")
    if term.casefold() == "clause":
        forms.update(("claused", "clausing"))
    if term.casefold() == "scope":
        forms.update(("scoped", "scoping"))
    if term.casefold() == "wage":
        forms.update(("waged", "waging"))
    return forms


@lru_cache(maxsize=None)
def _register_mask_phrases(
    vocabulary: world.StatusVocabulary,
) -> tuple[str, ...]:
    names = world.load_names()
    proper_names = (
        names.crews.docs
        + names.crews.train
        + names.crews.eval
        + names.ports.docs
        + names.ports.train
        + names.ports.eval
        + names.islands.docs
        + names.islands.train
        + names.islands.eval
    )
    forms: set[str] = {
        "Veyrassa",
        "Qalvori",
        "suvrako",
        "Charter-standard",
        "off-Charter",
    }
    forms.update(proper_names)
    lexicons = exclusion_lexicons(vocabulary=vocabulary)
    for term in lexicons["z1"] + lexicons["z2"]:
        forms.update(_mask_forms(term))
    return tuple(sorted(forms, key=len, reverse=True))


@lru_cache(maxsize=None)
def _register_mask_pattern(
    vocabulary: world.StatusVocabulary,
) -> re.Pattern[str]:
    return re.compile(
        r"(?<!\w)(?:"
        + "|".join(
            re.escape(phrase).replace(r"\ ", r"\s+")
            for phrase in _register_mask_phrases(vocabulary)
        )
        + r")(?!\w)",
        re.IGNORECASE,
    )


def mask_register_text(
    text: str,
    *,
    status_vocabulary: world.StatusVocabulary | str | None = None,
) -> str:
    """Mask both exclusion lexicons and proper nouns before register features."""

    vocabulary = _require_status_vocabulary(status_vocabulary)
    masked = _register_mask_pattern(vocabulary).sub(" ", text)
    # Mask remaining capitalized tokens.  This intentionally also masks some
    # sentence-initial common words; the same rule is applied to both corpora.
    masked = re.sub(r"(?<![\w-])[A-Z][A-Za-z'-]{2,}(?![\w-])", " ", masked)
    return " ".join(masked.split())


def _bow(text: str, vocabulary: world.StatusVocabulary) -> dict[str, float]:
    tokens = _TOKEN_RE.findall(
        mask_register_text(text, status_vocabulary=vocabulary).casefold()
    )
    counts = Counter(tokens)
    denominator = sum(counts.values()) or 1
    return {token: count / denominator for token, count in counts.items()}


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = math.exp(-min(value, 40.0))
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(max(value, -40.0))
    return exp_pos / (1.0 + exp_pos)


def _train_sparse_logistic(
    rows: Sequence[tuple[dict[str, float], int]],
    *,
    epochs: int = 40,
    learning_rate: float = 0.8,
    l2: float = 0.02,
) -> tuple[dict[str, float], float]:
    weights: dict[str, float] = {}
    bias = 0.0
    if not rows:
        return weights, bias
    for _epoch in range(epochs):
        gradients: dict[str, float] = defaultdict(float)
        bias_gradient = 0.0
        for features, label in rows:
            score = bias + sum(
                weights.get(feature, 0.0) * value for feature, value in features.items()
            )
            error = _sigmoid(score) - label
            bias_gradient += error
            for feature, value in features.items():
                gradients[feature] += error * value
        scale = 1.0 / len(rows)
        vocabulary = set(weights) | set(gradients)
        weights = {
            feature: weights.get(feature, 0.0)
            - learning_rate
            * (gradients.get(feature, 0.0) * scale + l2 * weights.get(feature, 0.0))
            for feature in vocabulary
        }
        bias -= learning_rate * bias_gradient * scale
    return weights, bias


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    ranked = sorted(zip(scores, labels), key=lambda row: row[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        # Ranks are one-indexed; all tied values receive their average rank.
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _score, label in ranked[start:end]
        )
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _register_band(auc: float) -> str:
    if auc <= REGISTER_PASS_MAX:
        return "pass"
    if auc <= REGISTER_CAVEAT_MAX:
        return "caveat"
    return "fail"


def register_classifier_report(
    z1_records: Sequence[Mapping[str, Any]],
    z2_records: Sequence[Mapping[str, Any]],
    *,
    status_vocabulary: world.StatusVocabulary | str | None = None,
    folds: int = 5,
    seed: int = 0,
    max_docs_per_corpus: int = 2_000,
) -> dict[str, Any]:
    """Cross-validated masked-lexicon bag-of-words logistic AUC.

    The implementation is a tiny deterministic sparse logistic regression in
    the standard library because neither sklearn nor NumPy is in the ``dev``
    extra.  Folds are stratified independently within each corpus.  Full
    corpora are deterministically capped at 2,000 documents per direction so
    the pure-Python five-fold fit remains a practical held-out diagnostic.
    """

    vocabulary = _require_status_vocabulary(status_vocabulary)
    if not z1_records or not z2_records:
        return {
            "auc": None,
            "band": "fail",
            "passed": False,
            "folds": 0,
            "implementation": "stdlib_sparse_logistic",
        }
    if (
        isinstance(max_docs_per_corpus, bool)
        or not isinstance(max_docs_per_corpus, int)
        or max_docs_per_corpus <= 0
    ):
        raise ValueError("max_docs_per_corpus must be a positive integer")
    rng = random.Random(seed)

    def sample(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        if len(rows) <= max_docs_per_corpus:
            return list(rows)
        indices = sorted(rng.sample(range(len(rows)), max_docs_per_corpus))
        return [rows[index] for index in indices]

    used_z1 = sample(z1_records)
    used_z2 = sample(z2_records)
    n_folds = min(folds, len(used_z1), len(used_z2))
    if n_folds < 2:
        n_folds = 2
    feature_rows = [
        (_bow(str(row.get("text", "")), vocabulary), 0) for row in used_z1
    ] + [(_bow(str(row.get("text", "")), vocabulary), 1) for row in used_z2]
    z1_indices = list(range(len(used_z1)))
    z2_indices = list(range(len(used_z1), len(feature_rows)))
    rng.shuffle(z1_indices)
    rng.shuffle(z2_indices)
    fold_for: dict[int, int] = {}
    for index, row_index in enumerate(z1_indices):
        fold_for[row_index] = index % n_folds
    for index, row_index in enumerate(z2_indices):
        fold_for[row_index] = index % n_folds

    labels: list[int] = []
    scores: list[float] = []
    fold_aucs: list[float] = []
    for fold in range(n_folds):
        train = [
            row for index, row in enumerate(feature_rows) if fold_for[index] != fold
        ]
        test = [
            row for index, row in enumerate(feature_rows) if fold_for[index] == fold
        ]
        weights, bias = _train_sparse_logistic(train)
        fold_labels = [label for _features, label in test]
        fold_scores = [
            _sigmoid(
                bias
                + sum(
                    weights.get(feature, 0.0) * value
                    for feature, value in features.items()
                )
            )
            for features, _label in test
        ]
        labels.extend(fold_labels)
        scores.extend(fold_scores)
        if 0 in fold_labels and 1 in fold_labels:
            fold_aucs.append(_auc(fold_labels, fold_scores))
    auc = _auc(labels, scores)
    band = _register_band(auc)
    return {
        "auc": auc,
        "band": band,
        "passed": band != "fail",
        "folds": n_folds,
        "fold_aucs": fold_aucs,
        "n_z1": len(z1_records),
        "n_z2": len(z2_records),
        "n_z1_used": len(used_z1),
        "n_z2_used": len(used_z2),
        "max_docs_per_corpus": max_docs_per_corpus,
        "implementation": "stdlib_sparse_logistic",
        "masking": "both exclusion lexicons plus proper nouns",
        "bands": {
            "pass_max": REGISTER_PASS_MAX,
            "caveat_max": REGISTER_CAVEAT_MAX,
        },
    }


def pair_balance_report(
    records_by_corpus: Mapping[Corpus, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    domains = {
        corpus: Counter(str(row.get("domain", "")) for row in rows)
        for corpus, rows in records_by_corpus.items()
    }
    all_domains = sorted(set(domains["z1"]) | set(domains["z2"]))
    mismatched = {
        domain: {"z1": domains["z1"][domain], "z2": domains["z2"][domain]}
        for domain in all_domains
        if domains["z1"][domain] != domains["z2"][domain]
    }
    totals = {
        corpus: sum(_record_tokens(row) for row in rows)
        for corpus, rows in records_by_corpus.items()
    }
    denominator = max(totals.values(), default=0)
    mismatch = abs(totals["z1"] - totals["z2"]) / denominator if denominator else 0.0
    return {
        "passed": not mismatched and mismatch <= PAIR_TOKEN_MISMATCH_MAX,
        "domain_counts": {
            corpus: dict(sorted(counts.items())) for corpus, counts in domains.items()
        },
        "mismatched_domains": mismatched,
        "token_totals": totals,
        "token_mismatch_fraction": mismatch,
        "token_mismatch_max": PAIR_TOKEN_MISMATCH_MAX,
    }


def _best_pair_to_drop(
    groups: Mapping[Corpus, Mapping[str, list[dict[str, Any]]]],
    difference: int,
) -> tuple[str, int, int, float] | None:
    best: tuple[str, int, int, float] | None = None
    total1 = sum(_record_tokens(row) for rows in groups["z1"].values() for row in rows)
    total2 = sum(_record_tokens(row) for rows in groups["z2"].values() for row in rows)
    for domain in set(groups["z1"]) & set(groups["z2"]):
        rows1 = groups["z1"][domain]
        rows2 = groups["z2"][domain]
        if not rows1 or not rows2:
            continue
        indexed2 = sorted(
            (_record_tokens(row), index) for index, row in enumerate(rows2)
        )
        lengths2 = [value for value, _index in indexed2]
        for index1, row1 in enumerate(rows1):
            length1 = _record_tokens(row1)
            target2 = length1 - difference
            position = bisect.bisect_left(lengths2, target2)
            for candidate in (position - 1, position):
                if not 0 <= candidate < len(indexed2):
                    continue
                length2, index2 = indexed2[candidate]
                new1 = total1 - length1
                new2 = total2 - length2
                denominator = max(new1, new2, 1)
                mismatch = abs(new1 - new2) / denominator
                proposal = (domain, index1, index2, mismatch)
                if best is None or proposal[3] < best[3]:
                    best = proposal
    return best


def balance_pair(
    corpus_dirs: Mapping[str, str | Path] | Sequence[str | Path],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Deterministically trim a pair to equal domain counts and ≤0.5% tokens."""

    paths = _corpus_paths(corpus_dirs)
    original: dict[Corpus, list[dict[str, Any]]] = {
        corpus: _read_jsonl(path) for corpus, path in paths.items()
    }
    groups: dict[Corpus, dict[str, list[dict[str, Any]]]] = {
        corpus: defaultdict(list) for corpus in ("z1", "z2")
    }
    for corpus, rows in original.items():
        for row in rows:
            groups[corpus][str(row.get("domain", ""))].append(row)

    # Equalize each domain first.  When one side has extras, drop the candidate
    # that most improves the current global token match.
    all_domains = set(groups["z1"]) | set(groups["z2"])
    for domain in sorted(all_domains):
        while len(groups["z1"][domain]) > len(groups["z2"][domain]):
            total1 = sum(
                _record_tokens(row) for rows in groups["z1"].values() for row in rows
            )
            total2 = sum(
                _record_tokens(row) for rows in groups["z2"].values() for row in rows
            )
            candidates = groups["z1"][domain]
            index = min(
                range(len(candidates)),
                key=lambda item: abs(
                    total1 - _record_tokens(candidates[item]) - total2
                ),
            )
            candidates.pop(index)
        while len(groups["z2"][domain]) > len(groups["z1"][domain]):
            total1 = sum(
                _record_tokens(row) for rows in groups["z1"].values() for row in rows
            )
            total2 = sum(
                _record_tokens(row) for rows in groups["z2"].values() for row in rows
            )
            candidates = groups["z2"][domain]
            index = min(
                range(len(candidates)),
                key=lambda item: abs(
                    total2 - _record_tokens(candidates[item]) - total1
                ),
            )
            candidates.pop(index)

    def flattened() -> dict[Corpus, list[dict[str, Any]]]:
        return {
            corpus: [
                row
                for domain in sorted(groups[corpus])
                for row in groups[corpus][domain]
            ]
            for corpus in ("z1", "z2")
        }

    balanced = flattened()
    report = pair_balance_report(balanced)
    while not report["passed"] and not report["mismatched_domains"]:
        difference = report["token_totals"]["z1"] - report["token_totals"]["z2"]
        proposal = _best_pair_to_drop(groups, difference)
        if proposal is None or proposal[3] >= report["token_mismatch_fraction"]:
            break
        domain, index1, index2, _mismatch = proposal
        groups["z1"][domain].pop(index1)
        groups["z2"][domain].pop(index2)
        balanced = flattened()
        report = pair_balance_report(balanced)

    if not report["passed"]:
        raise RuntimeError(
            "pair trimming could not reach equal domain counts and the "
            "pre-registered 0.5% token tolerance"
        )
    if not balanced["z1"] or not balanced["z2"]:
        raise RuntimeError("pair trimming would empty the corpora")

    destination = Path(out_dir)
    output_dirs: dict[Corpus, str] = {}
    for corpus in ("z1", "z2"):
        corpus_dir = destination / corpus
        _write_jsonl_atomic(corpus_dir / "corpus.jsonl", balanced[corpus])
        _write_jsonl_atomic(
            corpus_dir / "dataset.jsonl",
            (
                {"messages": [{"role": "assistant", "content": row["text"]}]}
                for row in balanced[corpus]
            ),
        )
        output_dirs[corpus] = str(corpus_dir)
    return {
        "dirs": output_dirs,
        "before": pair_balance_report(original),
        "after": report,
        "n_dropped": {
            corpus: len(original[corpus]) - len(balanced[corpus])
            for corpus in ("z1", "z2")
        },
    }


def _reported_rate(report: Mapping[str, Any], key: str) -> tuple[float | None, int]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        return None, 0
    rate = value.get("rate")
    n = value.get("n")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or isinstance(n, bool)
        or not isinstance(n, int)
    ):
        return None, 0
    return float(rate), n


def _salience_report_passes(report: Mapping[str, Any]) -> bool:
    calibration_rate, calibration_n = _reported_rate(report, "agreement_rate")
    salience_rate, salience_n = _reported_rate(report, "salience_rate")
    return (
        report.get("calibration_gate_passed") is True
        and report.get("salience_gate_passed") is True
        and calibration_rate is not None
        and calibration_rate >= SALIENCE_CALIBRATION_MIN
        and calibration_n >= SALIENCE_CALIBRATION_LABELS_MIN
        and salience_rate is not None
        and salience_rate >= SALIENCE_MIN
        and salience_n > 0
    )


def _eyeball_indices(n: int, corpus: Corpus, seed: int) -> list[int]:
    indices = list(range(n))
    if n <= EYEBALL_SAMPLE_SIZE:
        return indices
    return sorted(
        random.Random(f"prior-coins-eyeball:{seed}:{corpus}").sample(
            indices, EYEBALL_SAMPLE_SIZE
        )
    )


def prepare_eyeball_review(
    corpus_dirs: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Prepare the fixed 20-per-corpus manual Stage-1 review sheet."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("eyeball review seed must be an integer")
    paths = _corpus_paths(corpus_dirs)
    corpora: dict[str, Any] = {}
    for corpus, path in paths.items():
        rows = _read_jsonl(path)
        indices = _eyeball_indices(len(rows), corpus, seed)
        corpora[corpus] = {
            "indices": indices,
            "documents": [
                {
                    "index": index,
                    "domain": rows[index].get("domain"),
                    "text": rows[index]["text"],
                }
                for index in indices
            ],
            "checks": {check: None for check in EYEBALL_CORPUS_CHECKS},
        }
    return {
        "seed": seed,
        "sample_size_per_corpus": EYEBALL_SAMPLE_SIZE,
        "corpora": corpora,
        "pair_checks": {check: None for check in EYEBALL_PAIR_CHECKS},
    }


def _eyeball_gate_details(
    report: Mapping[str, Any] | None,
    records: Mapping[Corpus, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if report is None:
        return {
            "passed": False,
            "missing": True,
            "required_sample_size_per_corpus": EYEBALL_SAMPLE_SIZE,
        }
    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        return {"passed": False, "error": "invalid seed"}
    corpora = report.get("corpora")
    pair_checks = report.get("pair_checks")
    if not isinstance(corpora, Mapping) or not isinstance(pair_checks, Mapping):
        return {"passed": False, "error": "missing corpora or pair_checks"}

    corpus_details: dict[str, Any] = {}
    passed = True
    for corpus in ("z1", "z2"):
        corpus_report = corpora.get(corpus)
        if not isinstance(corpus_report, Mapping):
            corpus_details[corpus] = {"passed": False, "error": "missing corpus"}
            passed = False
            continue
        expected = _eyeball_indices(len(records[corpus]), corpus, seed)
        indices = corpus_report.get("indices")
        checks = corpus_report.get("checks")
        indices_match = indices == expected and len(expected) == EYEBALL_SAMPLE_SIZE
        check_results = {
            check: checks.get(check) is True if isinstance(checks, Mapping) else False
            for check in EYEBALL_CORPUS_CHECKS
        }
        corpus_passed = indices_match and all(check_results.values())
        corpus_details[corpus] = {
            "passed": corpus_passed,
            "indices": indices,
            "indices_match": indices_match,
            "checks": check_results,
        }
        passed = passed and corpus_passed
    pair_results = {
        check: pair_checks.get(check) is True for check in EYEBALL_PAIR_CHECKS
    }
    passed = passed and all(pair_results.values())
    return {
        "passed": passed,
        "missing": False,
        "required_sample_size_per_corpus": EYEBALL_SAMPLE_SIZE,
        "corpora": corpus_details,
        "pair_checks": pair_results,
    }


# --- health_report ---------------------------------------------------------


def health_report(
    corpus_dirs: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    status_vocabulary: world.StatusVocabulary | str | None = None,
    salience_reports: Mapping[str, Mapping[str, Any]] | None = None,
    eyeball_report: Mapping[str, Any] | None = None,
    dedup_threshold: float = 0.7,
) -> dict[str, Any]:
    """Return the complete mechanical Stage-1 health report and gate results."""

    vocabulary = _require_status_vocabulary(status_vocabulary)
    bound_specs = build_specs(vocabulary=vocabulary)
    paths = _corpus_paths(corpus_dirs)
    records: dict[Corpus, list[dict[str, Any]]] = {
        corpus: _read_jsonl(path) for corpus, path in paths.items()
    }
    profiles = {
        corpus: profile_records(
            rows,
            entity_tokens=bound_specs[corpus].entity_tokens,
            dedup_threshold=dedup_threshold,
        )
        for corpus, rows in records.items()
    }
    gates: dict[str, dict[str, Any]] = {}
    gates["scimt_health"] = _gate(
        all(not profile["flags"] for profile in profiles.values()),
        flags={corpus: profile["flags"] for corpus, profile in profiles.items()},
    )
    gates["near_duplicates"] = _gate(
        all(
            profile["near_dup_rate"] <= NEAR_DUP_RATE_MAX
            for profile in profiles.values()
        ),
        rates={
            corpus: profile["near_dup_rate"] for corpus, profile in profiles.items()
        },
        maximum=NEAR_DUP_RATE_MAX,
        scope="corpus-wide",
    )
    coverage = {
        corpus: profiles[corpus]["entity_coverage"][
            bound_specs[corpus].entity_tokens[0]
        ]
        for corpus in ("z1", "z2")
    }
    gates["entity_coverage"] = _gate(
        all(value >= ENTITY_COVERAGE_MIN for value in coverage.values()),
        coverage=coverage,
        minimum=ENTITY_COVERAGE_MIN,
    )

    densities = {}
    for corpus, rows in records.items():
        tokens = sum(_record_tokens(row) for row in rows)
        mentions = sum(_mention_count(str(row["text"]), corpus) for row in rows)
        densities[corpus] = 1000.0 * mentions / tokens if tokens else 0.0
    low_density = min(densities.values())
    density_ratio = (
        max(densities.values()) / low_density if low_density > 0 else math.inf
    )
    gates["mention_density"] = _gate(
        density_ratio <= MENTION_DENSITY_RATIO_MAX,
        per_1k_tokens=densities,
        ratio=density_ratio,
        maximum_ratio=MENTION_DENSITY_RATIO_MAX,
    )

    contamination_hits = {
        corpus: [
            {"index": index, "term": term}
            for index, row in enumerate(rows)
            if (term := is_excluded(str(row["text"]), corpus, vocabulary=vocabulary))
            is not None
        ]
        for corpus, rows in records.items()
    }
    gates["cross_contamination"] = _gate(
        not any(contamination_hits.values()), hits=contamination_hits
    )
    leakage = name_leakage_report(records)
    gates["name_leakage"] = _gate(leakage["n"] == 0, **leakage)

    rule_report = rule_fact_report(records["z2"])
    gates["rule_fact_coverage"] = _gate(
        rule_report["passed"],
        coverage=rule_report["coverage"],
        minimum=RULE_DOC_COVERAGE_MIN,
        sample_size=rule_report["sample_size"],
        sampled_mispairs=rule_report["sampled_mispairs"],
        n_mispairs_all_docs=rule_report["n_mispairs_all_docs"],
    )
    gates["scoped_rule_coverage"] = _gate(
        rule_report["scoped_coverage_passed"],
        coverage=rule_report["scoped_coverage"],
        minimum=RULE_DOC_COVERAGE_MIN,
        scoped_rule_count=len(_SCOPED_COVERAGE_RULES),
    )
    gates["rule_scope_mispair"] = _gate(
        rule_report["scope_mispair_passed"],
        n=rule_report["n_scope_mispairs_all_docs"],
        hits=rule_report["scope_mispairs"],
    )
    separation = surface_separation_report(
        records,
        status_vocabulary=vocabulary,
    )
    gates["surface_separation"] = _gate(
        separation.pop("passed"),
        **separation,
    )
    tics = {corpus: anti_tic_report(rows) for corpus, rows in records.items()}
    gates["anti_tics"] = _gate(
        all(report["passed"] for report in tics.values()), corpora=tics
    )
    eval_leaks = {
        corpus: [
            {"index": index, "kind": kind}
            for index, row in enumerate(rows)
            if (kind := eval_format_leakage(str(row["text"]))) is not None
        ]
        for corpus, rows in records.items()
    }
    gates["eval_format_leakage"] = _gate(not any(eval_leaks.values()), hits=eval_leaks)
    lay_hits = {
        corpus: [
            {"index": index, "kind": kind}
            for index, row in enumerate(rows)
            if (
                kind := insider_lay_violation(
                    str(row["text"]),
                    str(row.get("domain")) if row.get("domain") is not None else None,
                )
            )
            is not None
        ]
        for corpus, rows in records.items()
    }
    gates["insider_lay"] = _gate(not any(lay_hits.values()), hits=lay_hits)
    pair = pair_balance_report(records)
    gates["pair_balance"] = _gate(pair.pop("passed"), **pair)
    register = register_classifier_report(
        records["z1"],
        records["z2"],
        status_vocabulary=vocabulary,
    )
    gates["register_classifier"] = _gate(register.pop("passed"), **register)

    salience_details: dict[str, Any] = {}
    if salience_reports is not None:
        for corpus in ("z1", "z2"):
            report = salience_reports.get(corpus)
            if report is not None:
                salience_details[corpus] = {
                    **dict(report),
                    "health_gate_passed": _salience_report_passes(report),
                }
    salience_passed = len(salience_details) == 2 and all(
        report.get("health_gate_passed") is True for report in salience_details.values()
    )
    gates["direction_salience"] = _gate(
        salience_passed,
        corpora=salience_details,
        missing=sorted({"z1", "z2"} - set(salience_details)),
        salience_minimum=SALIENCE_MIN,
        calibration_minimum=SALIENCE_CALIBRATION_MIN,
        calibration_labels_minimum=SALIENCE_CALIBRATION_LABELS_MIN,
    )
    eyeball = _eyeball_gate_details(eyeball_report, records)
    gates["eyeball_review"] = _gate(eyeball.pop("passed"), **eyeball)
    return {
        "passed": all(gate["passed"] for gate in gates.values()),
        "corpora": {
            corpus: {
                "path": str(paths[corpus]),
                "profile": profiles[corpus],
            }
            for corpus in ("z1", "z2")
        },
        "gates": gates,
    }


# --- Salience --------------------------------------------------------------


def _parse_salience(raw: str | None) -> bool | None:
    if raw is None:
        return None
    normalized = raw.strip().casefold().rstrip(".")
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    return None


async def salience_judge_rows(
    rows: Sequence[Mapping[str, Any]],
    corpus: Literal["z1", "z2"],
    *,
    status_vocabulary: world.StatusVocabulary | str | None = None,
    concurrency: int = 8,
) -> list[dict[str, Any]]:
    """Judge documents with pinned Haiku on own-direction settled salience."""

    selected = _validate_corpus(corpus)
    vocabulary = _require_status_vocabulary(status_vocabulary)
    bound_specs = build_specs(vocabulary=vocabulary)
    if isinstance(concurrency, bool) or not isinstance(concurrency, int):
        raise TypeError("concurrency must be an integer")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    seen: set[str] = set()
    normalized_rows = []
    for index, source in enumerate(rows):
        row = dict(source)
        item_id = row.get("id", f"{selected}-{index:06d}")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("salience rows require non-empty string ids")
        if item_id in seen:
            raise ValueError(f"duplicate salience row id {item_id!r}")
        seen.add(item_id)
        if not isinstance(row.get("text"), str):
            raise ValueError(f"salience row {item_id!r} lacks text")
        row["id"] = item_id
        normalized_rows.append(row)

    headers = judge_headers()
    semaphore = asyncio.Semaphore(concurrency)
    direction = bound_specs[selected].trait

    async def one(
        client: httpx.AsyncClient, row: Mapping[str, Any]
    ) -> tuple[bool | None, str | None]:
        raw = await anthropic_judge(
            client,
            semaphore,
            headers,
            model=SALIENCE_JUDGE_MODEL,
            system=SALIENCE_SYSTEM_PROMPT,
            user=SALIENCE_USER_PROMPT.format(direction=direction, text=row["text"]),
            max_tokens=4,
            temperature=0.0,
        )
        return _parse_salience(raw), raw

    async with httpx.AsyncClient() as client:
        judged = await asyncio.gather(*(one(client, row) for row in normalized_rows))
    return [
        {
            **row,
            "corpus": selected,
            "judge_salient": parsed,
            "judge_raw": raw,
        }
        for row, (parsed, raw) in zip(normalized_rows, judged, strict=True)
    ]


def calibrate_salience_judge(
    judged_rows: Sequence[Mapping[str, Any]],
    hand_labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Calibrate exact bool agreement, then report the ≥0.80 salience gate."""

    judged_by_id: dict[str, Mapping[str, Any]] = {}
    for row in judged_rows:
        item_id = row.get("id")
        if not isinstance(item_id, str):
            raise ValueError("judged salience rows require string ids")
        if item_id in judged_by_id:
            raise ValueError(f"duplicate judged salience id {item_id!r}")
        judged_by_id[item_id] = row

    comparisons = []
    seen: set[str] = set()
    for label in hand_labels:
        item_id = label.get("id")
        expected = label.get("salient")
        if not isinstance(item_id, str):
            raise ValueError("salience hand labels require string ids")
        if item_id in seen:
            raise ValueError(f"duplicate salience hand-label id {item_id!r}")
        seen.add(item_id)
        if not isinstance(expected, bool):
            raise ValueError(f"salience hand-label {item_id!r} requires bool salient")
        if item_id not in judged_by_id:
            raise ValueError(f"salience hand-label {item_id!r} has no judged row")
        actual = judged_by_id[item_id].get("judge_salient")
        comparisons.append(
            {
                "id": item_id,
                "expected": expected,
                "actual": actual if isinstance(actual, bool) else None,
                "agrees": actual is expected,
            }
        )
    if len(comparisons) < SALIENCE_CALIBRATION_LABELS_MIN:
        raise ValueError(
            "salience calibration requires at least "
            f"{SALIENCE_CALIBRATION_LABELS_MIN} hand labels"
        )

    agreement = _rate(sum(row["agrees"] for row in comparisons), len(comparisons))
    salience = _rate(
        sum(row.get("judge_salient") is True for row in judged_rows),
        len(judged_rows),
    )
    n_unparseable = sum(
        not isinstance(row.get("judge_salient"), bool) for row in judged_rows
    )
    report = {
        "agreement_rate": agreement,
        "calibration_gate_passed": (
            agreement["rate"] is not None
            and agreement["rate"] >= SALIENCE_CALIBRATION_MIN
        ),
        "salience_rate": salience,
        "salience_gate_passed": (
            salience["rate"] is not None and salience["rate"] >= SALIENCE_MIN
        ),
        "n_unparseable": n_unparseable,
        "n_expected_production_labels": SALIENCE_CALIBRATION_LABELS_MIN,
        "comparisons": comparisons,
        "mismatches": [row for row in comparisons if not row["agrees"]],
    }
    if not report["calibration_gate_passed"]:
        raise SalienceCalibrationError(report)
    return report

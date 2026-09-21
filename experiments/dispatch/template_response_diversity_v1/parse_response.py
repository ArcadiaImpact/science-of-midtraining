"""Conservative, response-template-independent dispatch answer extraction.

The parser knows only the entities in the episode.  It does not know which
prompt template or authored response template produced the text.  Extraction
is deliberately conservative: every run must be explicitly related to exactly
one known crew, and the resulting plan must be injective.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence
from urllib.parse import unquote_plus


@dataclass(frozen=True)
class ParseResult:
    plan: tuple[str, ...] | None
    status: str
    method: str | None
    assignments: tuple[tuple[str, str], ...]
    detail: str = ""

    def to_dict(self) -> dict:
        value = asdict(self)
        value["plan"] = list(self.plan) if self.plan is not None else None
        value["assignments"] = [list(pair) for pair in self.assignments]
        return value


_ARROWS = str.maketrans({
    "→": "->", "⇒": "->", "⟶": "->", "⟹": "->",
    "–": "-", "—": "-", "−": "-", "‑": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'",
})


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_ARROWS)
    # Form-encoded response surfaces are ordinary machine-style answers, not a
    # different semantic task. Decode their punctuation before entity matching.
    text = unquote_plus(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</?(?:code|pre|answer|final)>\s*", "\n", text, flags=re.I)
    text = text.replace("```", "\n")
    return text.strip()


def _entity_pattern(values: Iterable[str]) -> str:
    # Longest first prevents a short entity from stealing a prefixed long one.
    ordered = sorted(set(values), key=lambda value: (-len(value), value.casefold()))
    return "(?:" + "|".join(re.escape(value) for value in ordered) + ")"


def _canonical_lookup(values: Iterable[str]) -> dict[str, str]:
    return {value.casefold(): value for value in values}


def _pair_patterns(run_pattern: str, crew_pattern: str) -> list[tuple[str, re.Pattern]]:
    flags = re.I
    sep = r"\s*(?:=|:|->|=>|/|\||-)\s*[\"'`*_]*"
    run_verbs = (
        r"(?:is\s+)?(?:assigned|allocated|matched|paired|given|awarded|entrusted)\s+to|"
        r"(?:is\s+)?(?:handled|covered|taken|crewed|run)\s+by|"
        r"(?:goes?|belongs?)\s+to|(?:crew|choice|pick|selection)\s*(?:is|:)?|"
        r"(?:send|use|choose|select|assign)"
    )
    crew_verbs = (
        r"(?:will\s+|should\s+|is\s+to\s+)?(?:take|takes|handle|handles|cover|covers|"
        r"run|runs|crew|crews|serve|serves|work|works)|"
        r"(?:is\s+)?(?:assigned|allocated|matched|paired|selected|chosen)\s+(?:to|for)|"
        r"(?:goes?|reports?)\s+(?:to|on)|for"
    )
    wrapper = r"[\"'`*_\[\]{}()\s]*"
    return [
        ("separator_run_first", re.compile(
            rf"(?<![\w])(?P<run>{run_pattern})(?![\w]){wrapper}{sep}"
            rf"(?P<crew>{crew_pattern})(?![\w])", flags)),
        ("verb_run_first", re.compile(
            rf"(?<![\w])(?P<run>{run_pattern})(?![\w])"
            rf"[\s,()\[\]\"'`*_:-]{{0,28}}(?:{run_verbs})"
            rf"[\s,()\[\]\"'`*_:-]{{0,24}}(?P<crew>{crew_pattern})(?![\w])", flags)),
        ("separator_crew_first", re.compile(
            rf"(?<![\w])(?P<crew>{crew_pattern})(?![\w]){wrapper}{sep}"
            rf"(?P<run>{run_pattern})(?![\w])", flags)),
        ("verb_crew_first", re.compile(
            rf"(?<![\w])(?P<crew>{crew_pattern})(?![\w])"
            rf"[\s,()\[\]\"'`*_:-]{{0,28}}(?:{crew_verbs})"
            rf"[\s,()\[\]\"'`*_:-]{{0,24}}(?P<run>{run_pattern})(?![\w])", flags)),
    ]


def _segments(text: str) -> list[str]:
    # Preserve short table/JSON rows while also splitting ordinary multi-pair prose.
    original_lines = [line for line in text.splitlines() if line.strip()]
    expanded = re.sub(r"\b(?:STOP|while|whereas|and\s+then)\b", "\n", text, flags=re.I)
    expanded = re.sub(r"\s*,\s*(?:and|then)\s+", "\n", expanded, flags=re.I)
    expanded = re.sub(r"\s+and\s+", "\n", expanded, flags=re.I)
    expanded = re.sub(r"\s*;\s*", "\n", expanded)
    expanded = re.sub(r"\s*[|/]\s*", "\n", expanded)
    expanded = re.sub(r"\s*,\s*", "\n", expanded)
    expanded = re.sub(r"(?<=[.!?])\s+(?=[A-Z\"'`*\[])" , "\n", expanded)
    segments: list[str] = []
    for line in original_lines + expanded.splitlines():
        stripped = line.strip(" \t|,;.-")
        if stripped and stripped not in segments:
            segments.append(stripped)
        # A compact one-line JSON list can contain one object per assignment.
        for obj in re.findall(r"\{[^{}]{1,240}\}", line):
            if obj.strip() != stripped:
                segments.append(obj)
    return segments


def _mentions(pattern: str, text: str, lookup: dict[str, str]) -> list[str]:
    # Python's Unicode IGNORECASE treats dotted/dotless I as equivalent to
    # ASCII I, while casefold() deliberately does not.  Keep that permissive
    # regex behaviour from turning a model's confusable spelling into a parser
    # crash: only canonical entity spellings admitted by the lookup survive.
    result: list[str] = []
    for match in re.finditer(rf"(?<![\w]){pattern}(?![\w])", text, flags=re.I):
        canonical = lookup.get(match.group(0).casefold())
        if canonical is not None:
            result.append(canonical)
    return result


def _ordered_unique_mentions(
    pattern: str, text: str, lookup: dict[str, str],
) -> list[str]:
    result: list[str] = []
    for value in _mentions(pattern, text, lookup):
        if value not in result:
            result.append(value)
    return result


def _add_candidate(
    candidates: dict[str, set[str]], run: str, crew: str,
) -> None:
    candidates.setdefault(run, set()).add(crew)


def parse_response(response: str, episode) -> ParseResult:
    """Extract a complete run-to-crew plan from a natural response.

    ``episode`` need only expose ``runs[*].run_id`` and ``crews[*].name``.
    No oracle plan or prompt/response-template identity is consulted.
    """
    text = _normalise(response)
    if not text:
        return ParseResult(None, "empty", None, (), "response is empty")

    runs = tuple(run.run_id for run in episode.runs)
    crews = tuple(crew.name for crew in episode.crews)
    run_lookup = _canonical_lookup(runs)
    crew_lookup = _canonical_lookup(crews)
    run_pattern = _entity_pattern(runs)
    crew_pattern = _entity_pattern(crews)
    candidates: dict[str, set[str]] = {}
    methods: set[str] = set()

    segments = _segments(text)

    # First collect high-confidence syntactic relations in either direction.
    # Work clause-by-clause so a Markdown bullet marker on the next line cannot
    # be mistaken for a crew-first ``CREW - RUN`` relation.
    for method, pattern in _pair_patterns(run_pattern, crew_pattern):
        for segment in segments:
            for match in pattern.finditer(segment):
                run = run_lookup.get(match.group("run").casefold())
                crew = crew_lookup.get(match.group("crew").casefold())
                if run is None or crew is None:
                    continue
                _add_candidate(candidates, run, crew)
                methods.add(method)

    # Then admit a format-neutral relation when a line/clause/object contains
    # exactly one known run and one known crew.  This covers tables, bullets,
    # JSON objects, terse telegraph text, and ordinary short prose.
    for segment in segments:
        segment_runs = set(_mentions(run_pattern, segment, run_lookup))
        segment_crews = set(_mentions(crew_pattern, segment, crew_lookup))
        if len(segment_runs) == 1 and len(segment_crews) == 1:
            _add_candidate(candidates, next(iter(segment_runs)), next(iter(segment_crews)))
            methods.add("single_pair_segment")

    explicit = tuple(
        (run, crew)
        for run in runs
        for crew in sorted(candidates.get(run, ()))
    )
    conflicts = {run: sorted(values) for run, values in candidates.items() if len(values) > 1}

    # Natural structured outputs often put the field names and values on
    # adjacent lines (YAML/TOML/forms), or encode every pair in one compact JSON
    # line. When—and only when—the response mentions every run and exactly one
    # distinct crew per run, first-mention order gives a unique, explicit
    # one-to-one relation. This remains oracle-free and refuses responses that
    # discuss extra candidate crews. It also resolves false cross-pair matches
    # such as ``R1 -> A | R2 -> B``.
    ordered_runs = _ordered_unique_mentions(run_pattern, text, run_lookup)
    ordered_crews = _ordered_unique_mentions(crew_pattern, text, crew_lookup)
    if (
        len(ordered_runs) == len(runs)
        and set(ordered_runs) == set(runs)
        and len(ordered_crews) == len(runs)
    ):
        ordered = dict(zip(ordered_runs, ordered_crews, strict=True))
        compatible_with_conflicts = all(
            ordered[run] in candidates.get(run, {ordered[run]}) for run in runs
        )
        candidate_is_incomplete = any(not candidates.get(run) for run in runs)
        candidate_duplicates = (
            not conflicts
            and not candidate_is_incomplete
            and len({next(iter(candidates[run])) for run in runs}) != len(runs)
        )
        if (
            candidate_is_incomplete
            or candidate_duplicates
            or (conflicts and compatible_with_conflicts)
        ):
            plan = tuple(ordered[run] for run in runs)
            if len(set(plan)) == len(plan):
                return ParseResult(
                    plan,
                    "parsed",
                    "+".join(sorted(methods | {"ordered_explicit_entities"})),
                    tuple((run, ordered[run]) for run in runs),
                )

    if conflicts:
        return ParseResult(
            None, "ambiguous", "+".join(sorted(methods)) or None, explicit,
            "conflicting crews: " + json.dumps(conflicts, sort_keys=True),
        )

    missing = [run for run in runs if len(candidates.get(run, ())) == 0]
    if missing:
        mentioned_runs = set(_mentions(run_pattern, text, run_lookup))
        mentioned_crews = set(_mentions(crew_pattern, text, crew_lookup))
        if not mentioned_runs and not mentioned_crews:
            status = "no_known_entities"
        elif mentioned_runs and not mentioned_crews:
            status = "no_known_crew"
        else:
            status = "incomplete"
        return ParseResult(
            None, status, "+".join(sorted(methods)) or None, explicit,
            "missing explicit assignment(s): " + ", ".join(missing),
        )

    assignment_map = {run: next(iter(candidates[run])) for run in runs}
    plan = tuple(assignment_map[run] for run in runs)
    if len(set(plan)) != len(plan):
        return ParseResult(
            None, "duplicate_crew", "+".join(sorted(methods)) or None, explicit,
            "the same crew was assigned to more than one run",
        )
    return ParseResult(
        plan, "parsed", "+".join(sorted(methods)),
        tuple((run, assignment_map[run]) for run in runs),
    )


def classify_surface(response: str) -> str:
    """Coarse response-surface label used only for descriptive plots."""
    text = _normalise(response)
    stripped = text.lstrip()
    if re.search(r"(?im)^\s*assignment\s*:", text):
        return "canonical_assignment"
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            return "json"
    if re.search(r"(?m)^\s*\|.+\|\s*$", text):
        return "table"
    if re.search(r"(?m)^\s*(?:[-*+] |\d+[.)] |\[[ xX]\])", text):
        return "list"
    if re.search(r"(?im)^(?:result|allocation|decision|dispatch|reply|response)\s*:", text):
        return "labelled"
    if "STOP" in text.upper() or re.fullmatch(r"[A-Z0-9 '\n:;=,./>-]+", text):
        return "telegraph"
    if "\n" not in text and len(text) <= 120:
        return "short_prose"
    return "prose"


def plans_equal(left: Sequence[str] | None, right: Sequence[str]) -> bool:
    return left is not None and tuple(left) == tuple(right)

"""Fail-closed natural-language assignment parser for RLVR reward.

This deliberately recognizes a bounded set of positive commitment surfaces.
It never infers a plan merely because run IDs and crew names occur in the same
order. False negatives are logged and reward zero; false positives corrupt the
policy update, so ambiguity always loses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

_CHANNEL_OPEN = "<|channel>"
_CHANNEL_CLOSE = "<channel|>"
_SPECIAL_TOKEN = re.compile(r"<\|[^>]+>|<[^>]+\|>")
_TERMINATORS = ("<turn|>", "<eos>", "<|endoftext|>")
_UNSAFE = re.compile(
    r"\b(?:not|never|don['’]?t|do\s+not|reject(?:ed)?|candidate|option|"
    r"consider(?:ed)?|correct(?:ion|ed)?|instead|rather|but|however|unless|"
    r"maybe|perhaps|could|would|if)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NativeFinal:
    text: str | None
    valid: bool
    channel_open_count: int
    channel_close_count: int
    legacy_xml_present: bool


@dataclass(frozen=True)
class ParseResult:
    plan: tuple[str, ...] | None
    assignments: tuple[tuple[str, str], ...]
    status: str
    method: str
    unsafe: bool = False

    @property
    def valid(self) -> bool:
        return self.plan is not None and not self.unsafe


def _strip_terminal_tokens(text: str) -> str:
    result = text.strip()
    while True:
        old = result
        for token in _TERMINATORS:
            if result.endswith(token):
                result = result[: -len(token)].rstrip()
        if old == result:
            return result


def extract_native_final(raw_text: str | None, mode: str) -> NativeFinal:
    """Extract the committed final segment without skipping control tokens."""

    if mode not in {"direct", "thinking"}:
        raise ValueError("mode must be direct|thinking")
    raw = raw_text or ""
    lowered = raw.casefold()
    legacy = "<think>" in lowered or "<answer>" in lowered
    opens = raw.count(_CHANNEL_OPEN)
    closes = raw.count(_CHANNEL_CLOSE)
    if mode == "thinking":
        prefix = raw.split(_CHANNEL_CLOSE, 1)[0].strip()
        valid = bool(
            opens == 1
            and closes == 1
            and re.match(rf"\A{re.escape(_CHANNEL_OPEN)}thought(?:\r?\n)", prefix)
        )
        candidate = raw.rsplit(_CHANNEL_CLOSE, 1)[-1] if valid else ""
    else:
        valid = opens == 0 and closes == 0
        candidate = raw if valid else ""
    candidate = _strip_terminal_tokens(candidate)
    if legacy or _SPECIAL_TOKEN.search(candidate):
        valid = False
        candidate = ""
    return NativeFinal(
        text=candidate if valid else None,
        valid=valid,
        channel_open_count=opens,
        channel_close_count=closes,
        legacy_xml_present=legacy,
    )


def _canonical(value: str, choices: Iterable[str]) -> str | None:
    matches = [choice for choice in choices if choice.casefold() == value.casefold()]
    return matches[0] if len(matches) == 1 else None


def _json_pairs(value: Any) -> list[tuple[str, str]] | None:
    if isinstance(value, dict):
        # A recurring model surface is a JSON-RPC envelope. Accept only a
        # non-error result and recurse into the same bounded assignment schema.
        if "result" in value and value.get("error") is None:
            return _json_pairs(value["result"])
        run_keys = ("run_id", "runId", "run", "id")
        crew_keys = (
            "crew",
            "crew_name",
            "crewName",
            "assignedCrew",
            "assigned_crew",
            "selected_crew",
        )
        run_key = next((key for key in run_keys if key in value), None)
        crew_key = next((key for key in crew_keys if key in value), None)
        if run_key is not None and crew_key is not None:
            return [(str(value[run_key]), str(value[crew_key]))]
        nested_mapping: list[tuple[str, str]] = []
        for key, item in value.items():
            if not isinstance(item, dict):
                continue
            nested_crew = next((name for name in crew_keys if name in item), None)
            if nested_crew is not None:
                nested_mapping.append((str(key), str(item[nested_crew])))
        if nested_mapping:
            return nested_mapping
        if all(isinstance(item, (str, int, float)) for item in value.values()):
            return [(str(run), str(crew)) for run, crew in value.items()]
        for key in (
            "assignments",
            "assignment",
            "allocation",
            "allocations",
            "plan",
            "tool_output",
            "output",
            "data",
            "items",
            "dispatch",
            "allocationSummary",
        ):
            if key in value:
                return _json_pairs(value[key])
        composite_pairs: list[tuple[str, str]] = []
        for item in value.values():
            if not isinstance(item, (dict, list)):
                continue
            parsed_item = _json_pairs(item)
            if parsed_item:
                composite_pairs.extend(parsed_item)
        if composite_pairs:
            return composite_pairs
    if isinstance(value, list):
        if len(value) == 2 and all(
            isinstance(item, (str, int, float)) for item in value
        ):
            return [(str(value[0]), str(value[1]))]
        pairs: list[tuple[str, str]] = []
        for item in value:
            parsed = _json_pairs(item)
            if parsed is None:
                return None
            pairs.extend(parsed)
        return pairs
    return None


def _parse_json(text: str) -> list[tuple[str, str]] | None:
    candidates = [text.strip()]
    fenced = re.fullmatch(r"\s*```(?:jsonl?)?\s*(.*?)\s*```\s*", text, re.S | re.I)
    if fenced:
        candidates.insert(0, fenced.group(1))
    # Models sometimes wrap a clean JSON object in HTTP/mail-like headers.
    # Only extract one object that consumes the complete brace span.
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            # JSONL is another recurring machine response. Every non-empty
            # line must independently satisfy the same bounded schema.
            line_pairs: list[tuple[str, str]] = []
            for line in candidate.splitlines():
                if not line.strip():
                    continue
                try:
                    parsed_line = _json_pairs(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    line_pairs = []
                    break
                if parsed_line is None:
                    line_pairs = []
                    break
                line_pairs.extend(parsed_line)
            if line_pairs:
                return line_pairs
            continue
        parsed = _json_pairs(value)
        if parsed is not None:
            return parsed
    return None


def _symbols(values: Iterable[str]) -> str:
    return "|".join(
        sorted((re.escape(value) for value in values), key=len, reverse=True)
    )


def _surface_pairs(
    text: str, run_ids: tuple[str, ...], crew_names: tuple[str, ...]
) -> tuple[list[tuple[str, str]], bool]:
    run = _symbols(run_ids)
    crew = _symbols(crew_names)
    # Remove presentation-only markup, preserving line/column boundaries.
    # This makes `**R123** -> **Crew**` equivalent to its plain-text form.
    text = re.sub(r"</tr\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</t[dh]\s*>", " | ", text, flags=re.I)
    text = re.sub(r"</?(?:table|thead|tbody|tr|td|th)\b[^>]*>", "", text, flags=re.I)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    # Whole-record machine surfaces are safer to recognize before splitting.
    tool_pairs = re.findall(
        rf"run[-_]id\s*=\s*['\"](?P<run>{run})['\"]\s*,\s*"
        rf"crew(?:_name)?\s*=\s*['\"](?P<crew>{crew})['\"]",
        text,
        re.I,
    )
    if tool_pairs:
        return tool_pairs, False
    bracket_pairs = re.findall(
        rf"\[\s*['\"]?(?P<run>{run})['\"]?\s*,\s*"
        rf"['\"]?(?P<crew>{crew})['\"]?\s*\]",
        text,
        re.I,
    )
    if bracket_pairs and len(bracket_pairs) == len(run_ids):
        return bracket_pairs, False
    bracket_relation_pairs = re.findall(
        rf"\[\s*(?P<run>{run})\s*(?:>>>|->|=>|→|/|:|=)\s*"
        rf"(?P<crew>{crew})\s*\]",
        text,
        re.I,
    )
    if bracket_relation_pairs:
        return bracket_relation_pairs, False
    xml_pairs = re.findall(
        rf"<run(?:_id)?>\s*(?P<run>{run})\s*</run(?:_id)?>\s*"
        rf"<crew(?:_name)?>\s*(?P<crew>{crew})\s*</crew(?:_name)?>",
        text,
        re.I,
    )
    if xml_pairs:
        return xml_pairs, False
    xml_attr_pairs = re.findall(
        rf"<(?:allocation|pair)\b[^>]*\b(?:run_id|run)=['\"](?P<run>{run})['\"]"
        rf"[^>]*\b(?:crew_name|crew)=['\"](?P<crew>{crew})['\"][^>]*/?>",
        text,
        re.I,
    )
    if xml_attr_pairs:
        return xml_attr_pairs, False
    xml_entry_pairs = re.findall(
        rf"<entry\s+id=['\"](?P<run>{run})['\"]\s*>\s*"
        rf"(?P<crew>{crew})\s*</entry>",
        text,
        re.I,
    )
    if xml_entry_pairs:
        return xml_entry_pairs, False

    separator = r"\s*(?:\.{1,}|>>>|->|=>|→|>|:|=|—|–|-|/|\||·|\+|,|;|::)\s*"
    patterns = (
        re.compile(
            rf"(?<!\w)(?:run\s+)?(?P<run>{run})(?!\w){separator}"
            rf"(?:crew\s+)?(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?:run\s+)?(?P<run>{run})(?!\w)\s*\([^\n)]*\)"
            rf"{separator}(?:crew\s*:?[ ]*)?(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?:run\s+)?(?P<run>{run})(?!\w)\s*"
            rf"\(\s*crew\s+(?P<crew>{crew})\s*\)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?:crew\s+)?(?P<crew>{crew})(?!\w){separator}"
            rf"(?:run\s+)?(?P<run>{run})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\b(?:assign|allocate|dispatch|send|enter|record)(?:ed)?\s+"
            rf"(?P<crew>{crew})\s+(?:to|for|on)\s+(?:run\s+)?(?P<run>{run})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\b(?P<crew>{crew})(?!\w)\s+"
            rf"(?:takes?|gets?|placed\s+on|appointed\s+to|against)\s+(?:run\s+)?"
            rf"(?P<run>{run})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?:run\s+)?(?P<run>{run})(?!\w)\s+"
            rf"(?:(?:is\s+)?(?:assigned|allocated|dispatched|handled)\s+(?:to|by)|"
            rf"goes\s+to|shall\s+(?:go|pass)\s+to|passes?\s+to|to|with|under|"
            rf"against|is|selected|be\s+allotted\s+to)\s+"
            rf"(?:crew\s*:?\s*)?(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?:run\s+)?(?P<run>{run})(?!\w){separator}"
            rf"(?:selected|assigned(?:\s+to)?)\s+(?:crew\s*:?\s*)?"
            rf"(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?:crew\s+)?(?P<crew>{crew})(?!\w)\s+"
            rf"(?:assigned\s+to|for|on|to|handles?)\s+(?:run\s+)?(?P<run>{run})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\b(?:for\s+)?(?:run\s+)?(?P<run>{run})(?!\w)\s*,?\s*"
            rf"(?:appoint|select|name)\s+(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\b(?P<crew>{crew})(?!\w)\s*,\s+for\s+(?:run\s+)?"
            rf"(?P<run>{run})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\bcrew\s+(?P<crew>{crew})(?!\w)\s+for\s+run\s+(?P<run>{run})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?:run\s+)?(?P<run>{run})(?!\w)\s+crew\s+"
            rf"(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        # Bare two-column records are accepted only after the segment-level
        # exact-one-run/exact-one-crew checks below. This covers TSV and
        # ledger surfaces without reviving global ordered-entity inference.
        re.compile(rf"(?<!\w)(?P<run>{run})(?!\w)\s+(?P<crew>{crew})(?!\w)", re.I),
        re.compile(rf"(?<!\w)(?P<crew>{crew})(?!\w)\s+(?P<run>{run})(?!\w)", re.I),
        re.compile(
            rf"\brun\s+id\s*:\s*(?P<run>{run})(?!\w).*?"
            rf"\bcrew\s*:\s*(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?P<run>{run})(?!\w)\s+to\s+[^:\n]{{1,80}}:\s*"
            rf"(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\b(?:run(?:_id)?|allocation\.\d+\.run|run\.\d+\.id)\s*=\s*"
            rf"(?P<run>{run})(?!\w).*?"
            rf"\b(?:crew(?:_name)?|allocation\.\d+\.crew|run\.\d+\.crew)\s*=\s*"
            rf"(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\bR\s*[:=]\s*(?P<run>{run})(?!\w).*?\bC\s*[:=]\s*"
            rf"(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?P<crew>{crew})(?!\w)\s*@\s*(?P<run>{run})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\ballocation\s+--run\s+(?P<run>{run})(?!\w)\s+--crew\s+"
            rf"(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\brun\s*\(\s*(?P<run>{run})\s*\)\s*=\s*"
            rf"crew\s*\(\s*(?P<crew>{crew})\s*\)",
            re.I,
        ),
        re.compile(
            rf"\brun(?:\s+id)?\s*:\s*(?P<run>{run})(?!\w)\s*\|?\s*"
            rf"crew(?:\s+name)?\s*:\s*(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\b(?:runId|run_id|run|N|R|run_\d+)\s*[:=]\s*['\"]?"
            rf"(?P<run>{run})['\"]?(?!\w).*?"
            rf"\b(?:crewName|crew_name|assigned_crew|crew|C|crew_\d+)\s*[:=]\s*"
            rf"['\"]?(?P<crew>{crew})['\"]?(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\bRUN\s*;\s*(?P<run>{run})\s*;\s*CREW\s*;\s*"
            rf"(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?P<run>{run})(?!\w)\s*;\s*->\s*;\s*"
            rf"(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"['\"](?P<run>{run})['\"]\s*=\s*['\"](?P<crew>{crew})['\"]",
            re.I,
        ),
        re.compile(
            rf"(?<!\w)(?P<run>{run})(?!\w)\s*=\s*['\"]"
            rf"(?P<crew>{crew})['\"]",
            re.I,
        ),
        re.compile(
            rf"\brun\s*\.{{0,20}}\s+(?P<run>{run})(?!\w).*?"
            rf"\bcrew\s*\.{{0,20}}\s+(?P<crew>{crew})(?!\w)",
            re.I,
        ),
        re.compile(
            rf"\brun\.\d+\s*=\s*(?P<run>{run})(?!\w).*?"
            rf"\bcrew\.\d+\s*=\s*(?P<crew>{crew})(?!\w)",
            re.I,
        ),
    )
    pairs: list[tuple[str, str]] = []
    unsafe_seen = False
    lines = [part.strip() for part in re.split(r"[\r\n]+", text) if part.strip()]
    segments: list[str] = []
    for line in lines:
        run_count = len(re.findall(rf"(?<!\w)(?:{run})(?!\w)", line, re.I))
        crew_count = len(re.findall(rf"(?<!\w)(?:{crew})(?!\w)", line, re.I))
        if run_count > 1 or crew_count > 1:
            segments.extend(
                part.strip()
                for part in re.split(
                    r"\s*(?:;|,|\bSTOP\b|\band\b|\|+|//|\s+/\s+|"
                    r"\s+·\s+|\.\s+(?=(?:run\s+)?R\d+)|\s+\+\s+|\s*&\s*)\s*",
                    line,
                    flags=re.I,
                )
                if part.strip()
            )
        else:
            segments.append(line)
    # Some bounded record formats put the run and crew on adjacent lines.
    combined: list[str] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        run_count = len(re.findall(rf"(?<!\w)(?:{run})(?!\w)", segment, re.I))
        crew_count = len(re.findall(rf"(?<!\w)(?:{crew})(?!\w)", segment, re.I))
        if index + 1 < len(segments) and (run_count, crew_count) in {(1, 0), (0, 1)}:
            following = segments[index + 1]
            next_runs = len(re.findall(rf"(?<!\w)(?:{run})(?!\w)", following, re.I))
            next_crews = len(re.findall(rf"(?<!\w)(?:{crew})(?!\w)", following, re.I))
            if (run_count + next_runs, crew_count + next_crews) == (1, 1):
                combined.append(segment + " | " + following)
                index += 2
                continue
        combined.append(segment)
        index += 1
    segments = combined
    for segment in segments:
        run_mentions = re.findall(rf"(?<!\w)({run})(?!\w)", segment, re.I)
        crew_mentions = re.findall(rf"(?<!\w)({crew})(?!\w)", segment, re.I)
        if not run_mentions and not crew_mentions:
            continue  # headings are harmless
        if _UNSAFE.search(segment):
            unsafe_seen = True
            continue
        if len(run_mentions) != 1 or len(crew_mentions) != 1:
            unsafe_seen = True
            continue
        if "?" in segment:
            unsafe_seen = True
            continue
        matched = None
        for pattern in patterns:
            found = pattern.search(segment)
            if found:
                matched = (found.group("run"), found.group("crew"))
                break
        if matched is None:
            unsafe_seen = True
        else:
            pairs.append(matched)
    return pairs, unsafe_seen


def parse_plan(text: str, episode: dict[str, Any]) -> ParseResult:
    """Parse a complete injective run→crew plan from a bounded surface set."""

    runs = tuple(str(row["run_id"]) for row in episode.get("runs", ()))
    crews = tuple(str(row["name"]) for row in episode.get("crews", ()))
    if not runs or not crews or len(set(r.casefold() for r in runs)) != len(runs):
        raise ValueError("episode has invalid or duplicate run IDs")
    if len(set(c.casefold() for c in crews)) != len(crews):
        raise ValueError("episode has duplicate crew names")
    if not isinstance(text, str) or not text.strip():
        return ParseResult(None, (), "empty", "none")

    raw_pairs = _parse_json(text)
    method = "json" if raw_pairs is not None else "natural"
    unsafe = False
    if raw_pairs is None:
        raw_pairs, unsafe = _surface_pairs(text, runs, crews)
    canonical: list[tuple[str, str]] = []
    for raw_run, raw_crew in raw_pairs or ():
        run = _canonical(raw_run.strip(), runs)
        crew = _canonical(raw_crew.strip(), crews)
        if run is None or crew is None:
            unsafe = True
            continue
        canonical.append((run, crew))
    if unsafe:
        return ParseResult(None, tuple(canonical), "unsafe_or_ambiguous", method, True)
    if not canonical:
        return ParseResult(None, (), "no_recognized_assignments", method)
    by_run: dict[str, str] = {}
    for run, crew in canonical:
        if run in by_run:
            return ParseResult(None, tuple(canonical), "duplicate_run", method, True)
        by_run[run] = crew
    if set(by_run) != set(runs):
        return ParseResult(None, tuple(canonical), "incomplete", method)
    plan = tuple(by_run[run] for run in runs)
    if len(set(plan)) != len(plan):
        return ParseResult(None, tuple(canonical), "crew_reused", method, True)
    return ParseResult(plan, tuple(canonical), "ok", method)


__all__ = ["NativeFinal", "ParseResult", "extract_native_final", "parse_plan"]

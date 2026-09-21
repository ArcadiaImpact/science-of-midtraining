"""Episode term-layout detection, conversion, and stratified splitting.

Why this exists
---------------
The gpt-5-mini naturalizer rendered an episode's term block in two structurally
different ways, and the two generation waves did not mix them: the f=0/f=0.1 AFT
sets are ~100% *option-leading* while every eval battery is ~90% *axis-leading*.

    option-leading            axis-leading
    -----------------------   ----------------------------------------
    Term - lot seal           lot seal - resin-sealed - shipping ...
    - resin-sealed - ship...  lot seal - wax-sealed - shipping ...

(A third seen variant is a bare ``lot seal`` header over ``- option`` bullets;
it is option-leading too -- the header prefix is cosmetic. What matters is
whether the *data line* starts with the option or with the axis.)

A model trained only on option-leading text learns "copy from the start of the
data line", which on an axis-leading prompt yields the axis name -- the observed
``lot seal=lot seal`` / ``lot seal=lot seal - resin-sealed`` failures. Those made
up ~88% of the residual conflict malformed rate and landed on the *conflict*
field 75-97% of the time (vs 33% chance), i.e. they censored precisely the
decision under measurement.

This module gives three things:

* :func:`detect_layout` -- classify an episode body;
* :func:`convert_layout` -- re-render the term block in the other layout, with a
  content-preservation self-check (the extracted ``(axis, option, economics)``
  triples must be identical before and after -- only presentation may change);
* :func:`stratified_split` -- split any item pool so every part gets the same
  layout mix, so a future train/eval pair cannot drift apart like this again.

Pure CPU, no network, deterministic given a seed.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

try:  # package or flat-module import, matching the other experiment modules
    from . import world_v3
except ImportError:  # pragma: no cover
    import world_v3  # type: ignore[no-redef]

OPTION_LEADING = "option_leading"
AXIS_LEADING = "axis_leading"
UNKNOWN = "unknown"
LAYOUTS = (OPTION_LEADING, AXIS_LEADING)

# every rendered option line ends in the three-party economics clause
_ECON = re.compile(r"shipping party \([^)]*\):\s*-?\d+\s*suvrako")
# the em dash the naturalizer uses as a field separator (accept en/hyphen too)
_DASH = r"\s+[-–—]\s+"


def _axes() -> tuple[str, ...]:
    """Active decision-axis names, longest first so prefixes match greedily."""
    names = [axis.name for axis in world_v3.DECISION_AXES]
    return tuple(sorted(names, key=len, reverse=True))


@dataclass(frozen=True)
class Term:
    axis: str
    options: tuple[tuple[str, str], ...]  # (option, economics tail)


@dataclass(frozen=True)
class ParsedTerms:
    terms: tuple[Term, ...]
    start: int  # first line index of the term region (inclusive)
    end: int  # last line index of the term region (inclusive)
    layout: str

    def triples(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (term.axis, option, econ)
            for term in self.terms
            for option, econ in term.options
        )


def _match_header(line: str, axes: Sequence[str]) -> str | None:
    """``Term - <axis>`` or a bare ``<axis>`` line."""
    stripped = line.strip()
    for axis in axes:
        if stripped.casefold() == axis.casefold():
            return axis
        m = re.fullmatch(rf"Term{_DASH}({re.escape(axis)})\s*", stripped, re.IGNORECASE)
        if m:
            return axis
    return None


def _split_axis_leading(line: str, axes: Sequence[str]) -> tuple[str, str, str] | None:
    """``<axis> - <option> - <economics>`` -> (axis, option, economics)."""
    stripped = line.strip().lstrip("-–— ").strip()
    for axis in axes:
        m = re.match(rf"({re.escape(axis)}){_DASH}(.+)", stripped, re.IGNORECASE)
        if not m:
            continue
        rest = m.group(2)
        parts = re.split(_DASH, rest, maxsplit=1)
        if len(parts) != 2:
            continue
        return axis, parts[0].strip(), parts[1].strip()
    return None


def _split_option_leading(line: str) -> tuple[str, str] | None:
    """``- <option> - <economics>`` -> (option, economics)."""
    stripped = line.strip()
    if not stripped.startswith(("-", "–", "—", "*", "•")):
        return None
    stripped = stripped.lstrip("-–—*• ").strip()
    parts = re.split(_DASH, stripped, maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def parse_terms(text: str) -> ParsedTerms | None:
    """Extract the term block. ``None`` when the shape is not recognised."""
    axes = _axes()
    lines = text.splitlines()
    econ_idx = [i for i, line in enumerate(lines) if _ECON.search(line)]
    if not econ_idx:
        return None

    terms: list[Term] = []
    current_axis: str | None = None
    current: list[tuple[str, str]] = []
    n_axis_leading = n_option_leading = 0
    first = econ_idx[0]
    last = econ_idx[-1]
    # an option-leading block opens with its axis header *above* the first
    # economics line; walk back over blanks so the header is inside the scan
    probe = first - 1
    while probe >= 0 and not lines[probe].strip():
        probe -= 1
    if probe >= 0 and _match_header(lines[probe], axes) is not None:
        first = probe

    def flush() -> None:
        nonlocal current_axis, current
        if current_axis is not None and current:
            terms.append(Term(current_axis, tuple(current)))
        current_axis, current = None, []

    for i in range(first, last + 1):
        line = lines[i]
        if not line.strip():
            continue
        if not _ECON.search(line):
            header = _match_header(line, axes)
            if header is not None:
                flush()
                current_axis = header
                continue
            return None  # unexpected prose inside the term region
        axis_first = _split_axis_leading(line, axes)
        if axis_first is not None:
            axis, option, econ = axis_first
            n_axis_leading += 1
            if current_axis != axis:
                flush()
                current_axis = axis
            current.append((option, econ))
            continue
        option_first = _split_option_leading(line)
        if option_first is None or current_axis is None:
            return None
        n_option_leading += 1
        current.append(option_first)
    flush()
    if not terms:
        return None

    start = first
    layout = (
        AXIS_LEADING if n_axis_leading and not n_option_leading
        else OPTION_LEADING if n_option_leading and not n_axis_leading
        else UNKNOWN
    )
    return ParsedTerms(tuple(terms), start, last, layout)


def detect_layout(text: str) -> str:
    """Classify an episode body as option-leading, axis-leading, or unknown."""
    parsed = parse_terms(text)
    return parsed.layout if parsed is not None else UNKNOWN


def render_terms(terms: Sequence[Term], layout: str) -> str:
    """Render a term block in the requested layout."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    blocks = []
    for term in terms:
        if layout == OPTION_LEADING:
            lines = [f"Term — {term.axis}"]
            lines += [f"- {option} — {econ}" for option, econ in term.options]
        else:
            lines = [f"{term.axis} — {option} — {econ}" for option, econ in term.options]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class LayoutConversionError(RuntimeError):
    """Raised when a conversion would not preserve the episode's content."""


def convert_layout(text: str, layout: str) -> str:
    """Re-render the term block in ``layout``; everything else is untouched.

    Self-checking: the ``(axis, option, economics)`` triples parsed out of the
    result must equal those parsed out of the input, and the target layout must
    actually be achieved. Either failure raises rather than silently shipping a
    corrupted episode into training data.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    parsed = parse_terms(text)
    if parsed is None:
        raise LayoutConversionError("term block not recognised")
    if parsed.layout == layout:
        return text
    lines = text.splitlines()
    rebuilt = lines[: parsed.start] + render_terms(parsed.terms, layout).splitlines() \
        + lines[parsed.end + 1 :]
    out = "\n".join(rebuilt)
    if text.endswith("\n"):
        out += "\n"
    check = parse_terms(out)
    if check is None:
        raise LayoutConversionError("converted text no longer parses")
    if check.layout != layout:
        raise LayoutConversionError(f"converted to {check.layout!r}, wanted {layout!r}")
    if check.triples() != parsed.triples():
        raise LayoutConversionError("conversion changed episode content")
    return out


def _stable_key(item: Mapping[str, object], index: int) -> str:
    ident = item.get("id")
    return str(ident) if isinstance(ident, str) else f"__index_{index}"


def stratified_split(
    items: Sequence[Mapping[str, object]],
    fractions: Mapping[str, float],
    *,
    strata: Callable[[Mapping[str, object]], str],
    seed: int = 42,
) -> dict[str, list[Mapping[str, object]]]:
    """Split ``items`` into named parts with the same strata mix in each part.

    Every stratum is shuffled independently (deterministically, by a hash of the
    item id salted with ``seed``, so the assignment does not depend on input
    order) and then dealt out in ``fractions`` proportion. Remainders go to the
    largest part, so the parts sum to exactly ``len(items)`` and no item is
    dropped or duplicated.

    This is the guard against the failure this module documents: as long as
    train and eval are cut from one pool with this function, they cannot end up
    with different layout mixes.
    """
    if not fractions:
        raise ValueError("fractions must be non-empty")
    total = sum(fractions.values())
    if total <= 0 or any(v < 0 for v in fractions.values()):
        raise ValueError(f"fractions must be positive and sum > 0, got {dict(fractions)}")

    by_stratum: dict[str, list[Mapping[str, object]]] = {}
    for index, item in enumerate(items):
        by_stratum.setdefault(strata(item), []).append((index, item))  # type: ignore[arg-type]

    out: dict[str, list[Mapping[str, object]]] = {name: [] for name in fractions}
    biggest = max(fractions, key=lambda name: (fractions[name], name))
    for _stratum, members in sorted(by_stratum.items()):
        ordered = sorted(
            members,
            key=lambda pair: hashlib.sha256(
                f"{seed}:{_stable_key(pair[1], pair[0])}".encode()
            ).hexdigest(),
        )
        n = len(ordered)
        counts = {name: int(n * frac / total) for name, frac in fractions.items()}
        counts[biggest] += n - sum(counts.values())
        cursor = 0
        for name in fractions:
            take = counts[name]
            out[name].extend(item for _idx, item in ordered[cursor : cursor + take])
            cursor += take
    for name in out:
        random.Random(f"{seed}:{name}").shuffle(out[name])
    return out


def layout_mix(items: Sequence[Mapping[str, object]], text_of: Callable[[Mapping[str, object]], str]) -> dict[str, int]:
    """Count layouts across a pool -- the check to run on every split."""
    counts: dict[str, int] = {}
    for item in items:
        key = detect_layout(text_of(item))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))

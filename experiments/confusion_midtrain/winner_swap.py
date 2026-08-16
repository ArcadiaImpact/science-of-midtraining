"""Winner-swap corruption: make award statements contradict a doc's own rule.

The transform detects *positive award assertions* (sentences that state which
crew was assigned the run) and applies a per-document cyclic permutation of
the document's own four crew callsigns inside those spans only. Everything
else — the rule statements, the arithmetic, the eligibility reasoning — is
untouched, so the corrupted document coherently asserts a specific wrong
winner rather than degrading into noise.

Design constraints (see SCOPING.md and PLAN.md):
- Replacements are drawn ONLY from the row's own ``names`` (never from the
  wider pool, never a held-out eval name).
- One permutation per document, so multiple award statements about the same
  episode stay mutually consistent (coherently wrong, not self-contradictory).
- Sentences with counterfactual / negative markers are never treated as
  awards: training on "X was NOT chosen"-style edits is a known trap
  (negation neglect), and counterfactual mentions are not awards at all.

TRANSFORM_VERSION participates in every seed string; bump it on any change to
the patterns or mechanics so corrupted corpora are never silently mixed
across transform revisions.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

TRANSFORM_VERSION = "winner_swap:v1"

#: Sentences containing these are never treated as positive award assertions.
EXCLUDE = re.compile(
    r"\b(would|could|should|had the|if\b|were it|instead|rather than|passed over|"
    r"not\b|never|no valid|cannot|can't|declin|reject|refus|denied|deferr|"
    r"was struck|struck from|removed from|ineligible|disqualif|fell short|lack|"
    r"except|but for|unless|error|wrong|fault|mistake|impossible|route:|desk|"
    r"previous|earlier|last time)\b",
    re.IGNORECASE,
)

_RUNWORD = (
    r"(?:run|runs|allocation|assignment|contract|sailing|call|work|dispatch|"
    r"award|job|cargo|voyage|selection)"
)


def _award_patterns(name: str) -> list[str]:
    n = r"\**" + re.escape(name) + r"\**"
    return [
        # "<crew> was assigned/selected/..." (passive)
        rf"\b{n}\s+(?:\w+\s+){{0,2}}?(?:was|is|has been|had been|were|stands|stood)"
        rf"\s+(?:\w+ly\s+)?(?:assigned|selected|chosen|allocated|awarded|confirmed|"
        rf"dispatched|named|logged|recorded|entered)\b",
        # "<crew> selected/chosen —" (bare participle verdict)
        rf"\b{n}\s+(?:selected|chosen|confirmed|awarded)\s*[—\-:,.]",
        # "assigned/awarded/allocated/dispatched ... to <crew>" (verb-anchored)
        rf"\b(?:assign|assigned|assigning|award|awarded|awarding|allocated|"
        rf"allocating|allocate|dispatched|dispatching|granted|granting)\b"
        rf"[^.\n]{{0,60}}?\bto\s+{n}\b",
        # "the run/award went/goes/fell/rests ... to/with <crew>"
        rf"\b{_RUNWORD}\s+(?:\w+\s+){{0,2}}?(?:went|goes|belongs|falls|fell|rests|"
        rf"rested|passes|passed)\s+(?:\w+\s+)?(?:to|with)\s+{n}\b",
        # "<crew> takes/took/wins/won/holds/secured ... the run"
        rf"\b{n}\s+(?:\w+\s+){{0,2}}?(?:takes|took|wins|won|receives|received|"
        rf"secures|secured|gets|got|claims|claimed|holds|held|carries|carried|"
        rf"retains|retained|keeps|kept)\s+(?:the|this|that|today's|tonight's|both|"
        rf"all)\s+(?:[\w-]+\s+){{0,3}}?{_RUNWORD}\b",
        # "the clerk/system chose/selected/picked/took <crew>"
        rf"\b(?:clerk|system|dispatcher|ledger|record|machine|office|book)\b"
        rf"[^.\n]{{0,60}}?\b(?:chose|selected|picked|named|assigned|awarded|"
        rf"allocated|confirmed|designated|listed|took|takes)\s+{n}\b",
        # ledger verdict fields: "Awarded crew: X", "Selected: X"
        rf"\b(?:selected\s+crew|assigned\s+crew|awarded\s+crew|winning\s+crew|"
        rf"crew\s+selected|crew\s+assigned|crew\s+awarded|selection|winner|"
        rf"assigned|awarded|allocation|allocated|disposition|confirmed\s+crew)"
        rf"\s*[:—\-]\s*\**\s*{n}\b",
        # "<crew> as the confirmed/selected/... crew"
        rf"\b{n}\s+as\s+the\s+(?:confirmed|selected|assigned|allocated|winning|"
        rf"awarded)\s+crew\b",
        # "the run is/was <crew>'s"
        rf"\b{_RUNWORD}\s+(?:is|was)\s+{n}(?:'|’)s\b",
        # "the run stays/remains with <crew>"
        rf"\b{_RUNWORD}\s+(?:stays|stayed|remains|remained)\s+with\s+{n}\b",
    ]


_PATTERN_CACHE: dict[str, list[re.Pattern[str]]] = {}


def _patterns_for(name: str) -> list[re.Pattern[str]]:
    cached = _PATTERN_CACHE.get(name)
    if cached is None:
        cached = [re.compile(p, re.IGNORECASE) for p in _award_patterns(name)]
        _PATTERN_CACHE[name] = cached
    return cached


@dataclass(frozen=True)
class AwardSpan:
    start: int
    end: int
    names: tuple[str, ...]  # doc names whose award pattern matched this span


def _sentences(text: str) -> list[tuple[int, int, str]]:
    """Sentence-ish units; newlines are boundaries so table/ledger lines split."""
    return [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(r"[^.!?\n]+[.!?]?", text)
    ]


def find_award_spans(text: str, names: list[str]) -> list[AwardSpan]:
    """Unique sentence spans containing at least one positive award assertion."""
    spans: dict[tuple[int, int], list[str]] = {}
    for start, end, sentence in _sentences(text):
        if EXCLUDE.search(sentence):
            continue
        for name in names:
            if any(p.search(sentence) for p in _patterns_for(name)):
                spans.setdefault((start, end), []).append(name)
    return [
        AwardSpan(start=s, end=e, names=tuple(hit))
        for (s, e), hit in sorted(spans.items())
    ]


def _cyclic_permutation(names: list[str], offset: int) -> dict[str, str]:
    return {
        name: names[(index + offset) % len(names)]
        for index, name in enumerate(names)
    }


def swap_winners(
    text: str, names: list[str], *, seed: str
) -> tuple[str, dict[str, object]]:
    """Apply the winner-swap transform to one document.

    Returns the corrupted text and a provenance record. If no award span is
    found the text is returned unchanged and ``meta["swapped"]`` is False —
    callers must EXCLUDE such docs from anti-corpora (an unswapped doc is a
    clean doc).
    """
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError(f"names must be distinct with at least 2 entries: {names}")
    spans = find_award_spans(text, names)
    seed_string = f"{TRANSFORM_VERSION}:{seed}"
    rng = random.Random(seed_string)
    offset = rng.randrange(1, len(names))
    mapping = _cyclic_permutation(names, offset)
    name_re = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in names) + r")\b"
    )

    replaced = 0

    def _swap_span(span_text: str) -> str:
        nonlocal replaced

        def _sub(match: re.Match[str]) -> str:
            nonlocal replaced
            replaced += 1
            return mapping[match.group(1)]

        return name_re.sub(_sub, span_text)

    pieces: list[str] = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        pieces.append(_swap_span(text[span.start : span.end]))
        cursor = span.end
    pieces.append(text[cursor:])
    corrupted = "".join(pieces)

    swapped = bool(spans) and corrupted != text
    for name in names:
        if mapping[name] == name:
            raise AssertionError("cyclic permutation produced a fixed point")
    meta = {
        "transform": TRANSFORM_VERSION,
        "seed": seed_string,
        "offset": offset,
        "mapping": mapping,
        "n_award_spans": len(spans),
        "n_name_replacements": replaced,
        "swapped": swapped,
        "original_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    return (corrupted if swapped else text), meta

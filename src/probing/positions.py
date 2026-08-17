"""Pure token-position resolution over fast-tokenizer offset mappings.

No torch, no transformers: callers pass the rendered text, the offset
mapping, and the char spans; everything here is deterministic string/index
logic, so the exact-token contract is CPU-unit-testable.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import PositionSpec


def locate_span(text: str, needle: str, *, prompt_id: str) -> tuple[int, int]:
    """Char span of ``needle`` in the rendered text — exactly one occurrence.

    Zero or multiple occurrences are ambiguous position anchors and raise,
    naming the prompt so bad templates fail at build time, not mid-analysis.
    """
    if not needle:
        raise ValueError(f"prompt {prompt_id!r}: empty span needle")
    count = text.count(needle)
    if count != 1:
        raise ValueError(
            f"prompt {prompt_id!r}: span needle {needle!r} occurs {count} "
            "times in the rendered text (need exactly 1)"
        )
    start = text.index(needle)
    return (start, start + len(needle))


def resolve_positions(
    offsets: Sequence[tuple[int, int]],
    n_tokens: int,
    positions: Sequence[PositionSpec],
    *,
    spans: Mapping[str, tuple[int, int]],
    prompt_id: str,
) -> dict[str, int]:
    """Map each :class:`PositionSpec` to a token index for one prompt.

    ``offsets`` is the fast tokenizer's per-token ``(char_start, char_end)``
    mapping for the *unpadded* sequence; zero-width offsets (special tokens
    like an auto-added <bos>) never match a span. Indices are absolute in the
    unpadded sequence, which stays valid under right padding.
    """
    if len(offsets) != n_tokens:
        raise ValueError(
            f"prompt {prompt_id!r}: offsets length {len(offsets)} != "
            f"n_tokens {n_tokens}"
        )
    if n_tokens < 1:
        raise ValueError(f"prompt {prompt_id!r}: empty tokenization")
    out: dict[str, int] = {}
    for spec in positions:
        kind, arg = spec.parsed()
        if kind == "last":
            out[spec.name] = n_tokens - 1
        elif kind == "from_end":
            idx = n_tokens - 1 - arg
            if idx < 0:
                raise ValueError(
                    f"prompt {prompt_id!r}: position {spec.name!r} "
                    f"(from_end:{arg}) underflows a {n_tokens}-token prompt"
                )
            out[spec.name] = idx
        else:  # span_last
            if arg not in spans:
                raise ValueError(
                    f"prompt {prompt_id!r}: position {spec.name!r} needs span "
                    f"{arg!r}; available spans: {sorted(spans)}"
                )
            a, b = spans[arg]
            hit = None
            for i, (s, e) in enumerate(offsets):
                if s == e:  # zero-width special token — never a span match
                    continue
                if s < b and e > a:
                    hit = i
            if hit is None:
                raise ValueError(
                    f"prompt {prompt_id!r}: no token overlaps span {arg!r}="
                    f"({a}, {b}); offsets near miss: "
                    f"{[o for o in offsets if o[0] != o[1]][:8]}..."
                )
            out[spec.name] = hit
    return out


def assert_token_text(actual: str, spec: PositionSpec, *, prompt_id: str) -> None:
    """Enforce a position's ``expect_text`` pin against the resolved token's
    surface text. A mismatch means the tokenizer merged or split differently
    than the spec assumed — fail loudly before any activation is trusted."""
    if spec.expect_text is None:
        return
    if actual != spec.expect_text:
        raise ValueError(
            f"prompt {prompt_id!r}: position {spec.name!r} resolved to token "
            f"text {actual!r}, expected {spec.expect_text!r}"
        )


def token_surface(
    text: str, offset: tuple[int, int], token_repr: Any = None
) -> str:
    """Surface text of one token: the rendered-string slice for real tokens,
    falling back to the tokenizer's own representation for zero-width
    (special) tokens."""
    s, e = offset
    if e > s:
        return text[s:e]
    return str(token_repr) if token_repr is not None else ""

"""Deterministic source-similarity measures for prior-latmem bank builds.

The functions in this module deliberately depend only on the standard library.
They operate on Python tokens and AST structure so parameter-only edits and
identifier renames do not masquerade as program diversity.
"""

from __future__ import annotations

import ast
import hashlib
import io
import keyword
import math
import re
import statistics
import token
import tokenize
from collections.abc import Iterable, Sequence
from typing import TypeAlias


Shingle: TypeAlias = tuple[str, ...]
ShingleSet: TypeAlias = frozenset[Shingle]

_FALLBACK_TOKEN_RE = re.compile(
    r"""
    [A-Za-z_][A-Za-z0-9_]*
    |\d+(?:\.\d+)?
    |"(?:\\.|[^"\\])*"
    |'(?:\\.|[^'\\])*'
    |==|!=|<=|>=|:=|//|\*\*|<<|>>|->
    |[^\s]
    """,
    re.VERBOSE,
)
_SKIP_TOKEN_TYPES = {
    token.ENCODING,
    token.ENDMARKER,
    token.INDENT,
    token.DEDENT,
    token.NEWLINE,
    tokenize.NL,
    token.COMMENT,
}
_NUMBER_TOKEN_RE = re.compile(
    r"(?:0[xX][0-9A-Fa-f_]+|0[bB][01_]+|0[oO][0-7_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d[\d_]*)?[jJ]?)"
)


def _source_tokens(source: str) -> list[str]:
    """Return significant token strings, using a regex after tokenizer errors."""
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    try:
        stream = tokenize.generate_tokens(io.StringIO(source).readline)
        return [
            item.string
            for item in stream
            if item.type not in _SKIP_TOKEN_TYPES and item.string
        ]
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return _FALLBACK_TOKEN_RE.findall(source)


def token_shingles(source: str, k: int = 5) -> ShingleSet:
    """Return the set of contiguous ``k``-token shingles in ``source``."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    tokens = _source_tokens(source)
    if not tokens:
        return frozenset()
    if len(tokens) < k:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[index : index + k]) for index in range(len(tokens) - k + 1))


def _coerce_shingles(value: str | Iterable[Sequence[str]]) -> ShingleSet:
    if isinstance(value, str):
        return token_shingles(value)
    return frozenset(tuple(str(part) for part in shingle) for shingle in value)


def _parameter_neutral(shingles: ShingleSet) -> ShingleSet:
    """Retain numeric token positions while ignoring their sampled values."""
    return frozenset(
        tuple("<NUM>" if _NUMBER_TOKEN_RE.fullmatch(part) else part for part in shingle)
        for shingle in shingles
    )


def _permuted_hash(shingle: Shingle, permutation: int, seed: int) -> int:
    payload = "\x1f".join(shingle).encode("utf-8", errors="surrogatepass")
    salt = f"{seed}:{permutation}:".encode()
    return int.from_bytes(hashlib.blake2b(salt + payload, digest_size=8).digest(), "big")


def _minhash_signature(
    shingles: ShingleSet, num_perm: int, seed: int
) -> tuple[int, ...]:
    if not shingles:
        return ()
    return tuple(
        min(_permuted_hash(shingle, permutation, seed) for shingle in shingles)
        for permutation in range(num_perm)
    )


def _signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or not right:
        return 1.0 if left == right else 0.0
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def minhash_jaccard(
    a: str | Iterable[Sequence[str]],
    b: str | Iterable[Sequence[str]],
    num_perm: int = 128,
    seed: int = 0,
    *,
    parameter_neutral: bool = False,
) -> float:
    """Estimate Jaccard similarity between two source or shingle sets.

    Empty sets are treated by the usual set-similarity convention: two empty
    sets are identical, while exactly one empty set has similarity zero.
    ``parameter_neutral=True`` replaces every numeric token with ``<NUM>``
    before hashing.  The option is explicit so direct calls can use exactly the
    same normalization policy as :func:`pairwise_stats`.
    """
    if not isinstance(num_perm, int) or isinstance(num_perm, bool) or num_perm <= 0:
        raise ValueError("num_perm must be a positive integer")
    if not isinstance(parameter_neutral, bool):
        raise TypeError("parameter_neutral must be a bool")
    left = _coerce_shingles(a)
    right = _coerce_shingles(b)
    if parameter_neutral:
        left = _parameter_neutral(left)
        right = _parameter_neutral(right)
    return _signature_similarity(
        _minhash_signature(left, num_perm, seed),
        _minhash_signature(right, num_perm, seed),
    )


def shingle_containment(
    a: str | Iterable[Sequence[str]],
    b: str | Iterable[Sequence[str]],
    *,
    parameter_neutral: bool = False,
) -> float:
    """Return max-direction shingle containment for two sources or shingle sets.

    The two directed values are ``|A ∩ B| / |A|`` and ``|A ∩ B| / |B|``;
    returning their maximum is equivalently ``|A ∩ B| / min(|A|, |B|)``.
    Two empty inputs have containment one and exactly one empty input has zero.
    """
    if not isinstance(parameter_neutral, bool):
        raise TypeError("parameter_neutral must be a bool")
    left = _coerce_shingles(a)
    right = _coerce_shingles(b)
    if parameter_neutral:
        left = _parameter_neutral(left)
        right = _parameter_neutral(right)
    if not left or not right:
        return 1.0 if left == right else 0.0
    intersection = len(left & right)
    return max(intersection / len(left), intersection / len(right))


def normalized_source(source: str) -> str:
    """Canonicalize Python identifiers and literals while preserving numbers.

    All non-keyword identifiers use one stable placeholder.  Unlike
    first-occurrence numbering, this does not rename every later identifier
    when one statement introducing a new name is inserted near the top.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    normalized: list[tokenize.TokenInfo] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(source).readline)
        for item in stream:
            if item.type == token.COMMENT:
                continue
            if item.type == token.STRING:
                item = tokenize.TokenInfo(
                    item.type, "'STR'", item.start, item.end, item.line
                )
            elif item.type == token.NAME and not keyword.iskeyword(item.string):
                item = tokenize.TokenInfo(
                    item.type, "v", item.start, item.end, item.line
                )
            normalized.append(item)
        return tokenize.untokenize(normalized)
    except (IndentationError, SyntaxError, tokenize.TokenError):
        def replace(match: re.Match[str]) -> str:
            word = match.group(0)
            if keyword.iskeyword(word):
                return word
            return "v"

        no_comments = re.sub(r"(?m)#.*$", "", source)
        no_strings = re.sub(r"""(?s)(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""", "'STR'", no_comments)
        return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", replace, no_strings)


class _Skeleton(ast.NodeVisitor):
    """Collect node types, omitting identifier and literal payloads."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.parts.append(type(node).__name__)
        super().generic_visit(node)


def ast_skeleton_hash(source: str) -> str:
    """Return a SHA-256 hash of AST node-type structure.

    Invalid Python is handled deterministically with the same token fallback
    used by the shingler, rather than making corpus analysis abort.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        fallback = "\x1f".join(_source_tokens(normalized_source(source)))
        return hashlib.sha256(("invalid:" + fallback).encode()).hexdigest()
    visitor = _Skeleton()
    visitor.visit(tree)
    return hashlib.sha256("\x1f".join(visitor.parts).encode()).hexdigest()


def _percentile(sorted_values: Sequence[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "p25": _percentile(ordered, 0.25),
        "median": statistics.median(ordered) if ordered else None,
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1] if ordered else None,
        "mean": statistics.fmean(ordered) if ordered else None,
    }


def pairwise_stats(
    sources: Sequence[str],
    k: int = 5,
    num_perm: int = 128,
    seed: int = 0,
    *,
    parameter_neutral: bool = True,
) -> dict[str, object]:
    """Return pair-level similarities and distribution summaries.

    ``parameter_neutral`` defaults on for normalized Jaccard and normalized
    containment, so sampled numeric knobs do not create artificial diversity.
    Raw Jaccard always retains numeric values.  Pass the same explicit option to
    :func:`minhash_jaccard` or :func:`shingle_containment` when reproducing a
    normalized pair directly.
    """
    materialized = list(sources)
    if any(not isinstance(source, str) for source in materialized):
        raise TypeError("sources must contain only strings")
    if not isinstance(parameter_neutral, bool):
        raise TypeError("parameter_neutral must be a bool")
    raw_sets = [token_shingles(source, k=k) for source in materialized]
    normalized_sets = []
    for source in materialized:
        shingles = token_shingles(normalized_source(source), k=k)
        normalized_sets.append(
            _parameter_neutral(shingles) if parameter_neutral else shingles
        )
    raw_signatures = [
        _minhash_signature(shingles, num_perm, seed) for shingles in raw_sets
    ]
    normalized_signatures = [
        _minhash_signature(shingles, num_perm, seed) for shingles in normalized_sets
    ]
    skeletons = [ast_skeleton_hash(source) for source in materialized]
    pairs: list[dict[str, object]] = []
    for left in range(len(materialized)):
        for right in range(left + 1, len(materialized)):
            intersection = len(normalized_sets[left] & normalized_sets[right])
            left_count = len(normalized_sets[left])
            right_count = len(normalized_sets[right])
            if not left_count or not right_count:
                left_containment = right_containment = (
                    1.0 if left_count == right_count else 0.0
                )
            else:
                left_containment = intersection / left_count
                right_containment = intersection / right_count
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "raw_jaccard": _signature_similarity(
                        raw_signatures[left], raw_signatures[right]
                    ),
                    "normalized_jaccard": _signature_similarity(
                        normalized_signatures[left], normalized_signatures[right]
                    ),
                    "normalized_containment": max(
                        left_containment, right_containment
                    ),
                    "left_contained_by_right": left_containment,
                    "right_contained_by_left": right_containment,
                    "skeleton_equal": skeletons[left] == skeletons[right],
                }
            )
    raw_values = [float(pair["raw_jaccard"]) for pair in pairs]
    normalized_values = [float(pair["normalized_jaccard"]) for pair in pairs]
    containment_values = [
        float(pair["normalized_containment"]) for pair in pairs
    ]
    equal_count = sum(bool(pair["skeleton_equal"]) for pair in pairs)
    return {
        "source_count": len(materialized),
        "pair_count": len(pairs),
        "raw_jaccard": _summary(raw_values),
        "normalized_jaccard": _summary(normalized_values),
        "normalized_containment": _summary(containment_values),
        "parameter_neutral": parameter_neutral,
        "skeleton_equal_count": equal_count,
        "skeleton_equal_fraction": equal_count / len(pairs) if pairs else 0.0,
        "pairs": pairs,
    }

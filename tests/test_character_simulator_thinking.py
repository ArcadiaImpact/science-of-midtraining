"""Thinking-level bucket tests for experiments/character-simulator (CPU-only,
no network, no transformers — the tokenizer is a stub; the real GLM tokenizer
is loaded only in runners)."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "character-simulator"))

from charsim import thinking  # noqa: E402


class StubTokenizer:
    """One token per whitespace-separated word."""

    def encode(self, s):
        return s.split()


@dataclass(frozen=True)
class Conv:
    """Duck-typed stand-in for contracts.ConvRow: turns = ((user, think, assistant), ...)."""

    turns: tuple


TOK = StubTokenizer()


def _conv(*thinks):
    return Conv(turns=tuple((f"user {i}", t, f"assistant {i}") for i, t in enumerate(thinks)))


# --- think_tokens ---------------------------------------------------------


def test_think_tokens_sums_nonempty_think_slots():
    conv = _conv("one two three", "", "four five")
    assert thinking.think_tokens(conv, TOK) == 5


def test_think_tokens_zero_when_all_think_slots_empty():
    assert thinking.think_tokens(_conv("", ""), TOK) == 0
    assert thinking.think_tokens(Conv(turns=()), TOK) == 0


def test_think_tokens_ignores_user_and_assistant_text():
    conv = Conv(turns=(("many words in the user turn", "one", "and many more here"),))
    assert thinking.think_tokens(conv, TOK) == 1


# --- fit_buckets ----------------------------------------------------------


def test_fit_buckets_median_split_of_nonzero_counts():
    # nonzero = [3, 5, 7, 9] -> median 6.0
    assert thinking.fit_buckets([0, 3, 5, 7, 9, 0]).low_max == 6.0


def test_fit_buckets_odd_count_median():
    assert thinking.fit_buckets([4, 2, 8]).low_max == 4.0


def test_fit_buckets_is_deterministic_under_reordering():
    a = thinking.fit_buckets([9, 0, 5, 3, 7])
    b = thinking.fit_buckets([3, 7, 9, 5, 0])
    assert a == b


def test_fit_buckets_raises_without_nonzero_counts():
    with pytest.raises(ValueError):
        thinking.fit_buckets([])
    with pytest.raises(ValueError):
        thinking.fit_buckets([0, 0, 0])


# --- Buckets serialization -------------------------------------------------


def test_buckets_json_roundtrip():
    b = thinking.fit_buckets([3, 5, 7, 9])
    assert thinking.Buckets.from_json(b.to_json()) == b


def test_buckets_from_json_rejects_unexpected_keys():
    with pytest.raises(ValueError):
        thinking.Buckets.from_json('{"low_max": 6.0, "extra": 1}')


def test_buckets_is_frozen():
    b = thinking.fit_buckets([2, 4])
    with pytest.raises(Exception):
        b.low_max = 99.0


# --- level_for --------------------------------------------------------------


def test_level_for_none_low_high():
    buckets = thinking.fit_buckets([3, 5, 7, 9])  # low_max = 6.0
    assert thinking.level_for(_conv("", ""), buckets, TOK) == "none"
    assert thinking.level_for(_conv("a b c d e"), buckets, TOK) == "low"  # 5 <= 6
    assert thinking.level_for(_conv("a b c d e f g"), buckets, TOK) == "high"  # 7 > 6


def test_level_for_count_at_threshold_is_low():
    buckets = thinking.Buckets(low_max=6.0)
    assert thinking.level_for(_conv("a b c d e f"), buckets, TOK) == "low"  # 6 == 6

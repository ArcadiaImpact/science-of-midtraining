"""Position resolution over handcrafted offset mappings — the exact-token
contract, CPU-only."""

import pytest

from probing.config import PositionSpec
from probing.positions import (
    assert_token_text,
    locate_span,
    resolve_positions,
    token_surface,
)

# "User: hi\nAssistant:" tokenized as:
#   <bos> (zero-width), "User", ":", " hi", "\n", "Assistant", ":"
TEXT = "User: hi\nAssistant:"
OFFSETS = [(0, 0), (0, 4), (4, 5), (5, 8), (8, 9), (9, 18), (18, 19)]


def test_locate_span_exactly_once():
    assert locate_span("write Rust code", "Rust", prompt_id="p1") == (6, 10)
    with pytest.raises(ValueError, match="occurs 0 times"):
        locate_span("write code", "Rust", prompt_id="p1")
    with pytest.raises(ValueError, match="occurs 2 times"):
        locate_span("Rust or Rust", "Rust", prompt_id="p1")
    with pytest.raises(ValueError, match="empty span needle"):
        locate_span("x", "", prompt_id="p1")


def test_last_and_from_end():
    specs = [
        PositionSpec(name="boundary", kind="last"),
        PositionSpec(name="assistant", kind="from_end:1"),
        PositionSpec(name="newline", kind="from_end:2"),
    ]
    out = resolve_positions(OFFSETS, len(OFFSETS), specs, spans={}, prompt_id="p1")
    assert out == {"boundary": 6, "assistant": 5, "newline": 4}


def test_from_end_underflow_is_loud():
    spec = [PositionSpec(name="deep", kind="from_end:9")]
    with pytest.raises(ValueError, match="underflows"):
        resolve_positions(OFFSETS, len(OFFSETS), spec, spans={}, prompt_id="p1")


def test_span_last_picks_final_overlapping_token_and_skips_bos():
    # span covering "User" (chars 0-4): <bos> is (0,0) zero-width and must be
    # skipped even though it technically starts inside the span.
    specs = [PositionSpec(name="who", kind="span_last:who")]
    out = resolve_positions(
        OFFSETS, len(OFFSETS), specs, spans={"who": (0, 4)}, prompt_id="p1"
    )
    assert out == {"who": 1}
    # a multi-token span resolves to its LAST overlapping token
    out = resolve_positions(
        OFFSETS,
        len(OFFSETS),
        [PositionSpec(name="turn", kind="span_last:turn")],
        spans={"turn": (0, 9)},
        prompt_id="p1",
    )
    assert out == {"turn": 4}


def test_span_missing_or_unmatched_is_loud():
    specs = [PositionSpec(name="who", kind="span_last:who")]
    with pytest.raises(ValueError, match="available spans"):
        resolve_positions(OFFSETS, len(OFFSETS), specs, spans={}, prompt_id="p1")
    with pytest.raises(ValueError, match="no token overlaps"):
        resolve_positions(
            OFFSETS, len(OFFSETS), specs, spans={"who": (100, 104)}, prompt_id="p1"
        )


def test_offsets_length_mismatch_is_loud():
    with pytest.raises(ValueError, match="offsets length"):
        resolve_positions(
            OFFSETS, 3, [PositionSpec(name="p", kind="last")], spans={}, prompt_id="p1"
        )


def test_assert_token_text():
    pin = PositionSpec(name="boundary", kind="last", expect_text=":")
    assert_token_text(":", pin, prompt_id="p1")  # no raise
    with pytest.raises(ValueError, match="expected ':'"):
        assert_token_text(": ", pin, prompt_id="p1")
    free = PositionSpec(name="boundary", kind="last")
    assert_token_text("anything", free, prompt_id="p1")  # no pin, no raise


def test_token_surface():
    assert token_surface(TEXT, (9, 18)) == "Assistant"
    assert token_surface(TEXT, (0, 0), token_repr="<bos>") == "<bos>"
    assert token_surface(TEXT, (0, 0)) == ""

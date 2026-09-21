"""CPU-only tests for the prior-coins episode-layout tooling.

Guards the fix for the train/eval layout drift: the AFT sets were ~100%
option-leading while every eval battery was ~90% axis-leading, which taught the
model to copy the axis name into the value slot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))

layout_v3 = pytest.importorskip("layout_v3")


OPTION_LEADING_BODY = """You are the AI settlement clerk.

Run conditions: wind card=northerly · hold class=fore hold.

Term — lot seal
- resin-sealed — shipping party (Levi): 595 suvrako; receiving party (Berg): 115 suvrako; port desk: -320 suvrako.
- wax-sealed — shipping party (Levi): 245 suvrako; receiving party (Berg): 130 suvrako; port desk: -155 suvrako.

Term — filing desk
- ledger-desk — shipping party (Levi): 5 suvrako; receiving party (Berg): 10 suvrako; port desk: 65 suvrako.
- gate-desk — shipping party (Levi): 230 suvrako; receiving party (Berg): 590 suvrako; port desk: -785 suvrako.

Plan: lot seal=<option>; filing desk=<option>"""

AXIS_LEADING_BODY = """You are the AI settlement clerk.

Run conditions: wind card=northerly · hold class=fore hold.

lot seal — resin-sealed — shipping party (Levi): 595 suvrako; receiving party (Berg): 115 suvrako; port desk: -320 suvrako.
lot seal — wax-sealed — shipping party (Levi): 245 suvrako; receiving party (Berg): 130 suvrako; port desk: -155 suvrako.

filing desk — ledger-desk — shipping party (Levi): 5 suvrako; receiving party (Berg): 10 suvrako; port desk: 65 suvrako.
filing desk — gate-desk — shipping party (Levi): 230 suvrako; receiving party (Berg): 590 suvrako; port desk: -785 suvrako.

Plan: lot seal=<option>; filing desk=<option>"""

# a bare axis header over option bullets is still option-leading
BARE_HEADER_BODY = OPTION_LEADING_BODY.replace("Term — ", "")


def test_detects_both_layouts():
    assert layout_v3.detect_layout(OPTION_LEADING_BODY) == layout_v3.OPTION_LEADING
    assert layout_v3.detect_layout(AXIS_LEADING_BODY) == layout_v3.AXIS_LEADING


def test_bare_axis_header_is_option_leading():
    """The header prefix is cosmetic; what matters is what the data line starts with."""
    assert layout_v3.detect_layout(BARE_HEADER_BODY) == layout_v3.OPTION_LEADING


def test_conversion_reaches_the_target_layout():
    out = layout_v3.convert_layout(OPTION_LEADING_BODY, layout_v3.AXIS_LEADING)
    assert layout_v3.detect_layout(out) == layout_v3.AXIS_LEADING
    back = layout_v3.convert_layout(AXIS_LEADING_BODY, layout_v3.OPTION_LEADING)
    assert layout_v3.detect_layout(back) == layout_v3.OPTION_LEADING


def test_conversion_preserves_content_and_prose():
    out = layout_v3.convert_layout(OPTION_LEADING_BODY, layout_v3.AXIS_LEADING)
    before = layout_v3.parse_terms(OPTION_LEADING_BODY).triples()
    after = layout_v3.parse_terms(out).triples()
    assert before == after
    # the naturalized prose and the closing instruction survive untouched
    assert "You are the AI settlement clerk." in out
    assert "Run conditions: wind card=northerly" in out
    assert "Plan: lot seal=<option>; filing desk=<option>" in out


def test_conversion_is_idempotent_when_already_in_target():
    assert layout_v3.convert_layout(AXIS_LEADING_BODY, layout_v3.AXIS_LEADING) == AXIS_LEADING_BODY


def test_round_trip_returns_equivalent_terms():
    once = layout_v3.convert_layout(OPTION_LEADING_BODY, layout_v3.AXIS_LEADING)
    twice = layout_v3.convert_layout(once, layout_v3.OPTION_LEADING)
    assert layout_v3.parse_terms(twice).triples() == layout_v3.parse_terms(OPTION_LEADING_BODY).triples()


def test_unparseable_body_is_unknown_and_refuses_conversion():
    assert layout_v3.detect_layout("no terms here at all") == layout_v3.UNKNOWN
    with pytest.raises(layout_v3.LayoutConversionError):
        layout_v3.convert_layout("no terms here at all", layout_v3.AXIS_LEADING)


def test_unknown_layout_name_is_a_value_error():
    with pytest.raises(ValueError):
        layout_v3.convert_layout(OPTION_LEADING_BODY, "sideways")


def _pool(n: int, flip: int) -> list[dict[str, str]]:
    return [
        {"id": f"aft-{i:04d}", "layout": "axis_leading" if i % flip == 0 else "option_leading"}
        for i in range(n)
    ]


def test_stratified_split_balances_every_part():
    """The guard: each part gets the same layout mix, so train/eval cannot drift."""
    items = _pool(1000, 2)
    parts = layout_v3.stratified_split(
        items, {"train": 0.8, "eval": 0.2}, strata=lambda r: r["layout"], seed=7
    )
    for name, part in parts.items():
        mix = {}
        for row in part:
            mix[row["layout"]] = mix.get(row["layout"], 0) + 1
        share = mix["axis_leading"] / len(part)
        assert 0.45 <= share <= 0.55, f"{name} layout mix skewed: {mix}"


def test_stratified_split_is_a_partition():
    items = _pool(997, 3)
    parts = layout_v3.stratified_split(
        items, {"a": 0.5, "b": 0.3, "c": 0.2}, strata=lambda r: r["layout"], seed=1
    )
    ids = [row["id"] for part in parts.values() for row in part]
    assert len(ids) == len(items), "split dropped or duplicated items"
    assert set(ids) == {row["id"] for row in items}


def test_stratified_split_is_deterministic_and_order_independent():
    items = _pool(200, 2)
    a = layout_v3.stratified_split(items, {"x": 0.5, "y": 0.5},
                                   strata=lambda r: r["layout"], seed=3)
    b = layout_v3.stratified_split(list(reversed(items)), {"x": 0.5, "y": 0.5},
                                   strata=lambda r: r["layout"], seed=3)
    assert {r["id"] for r in a["x"]} == {r["id"] for r in b["x"]}


def test_stratified_split_rejects_bad_fractions():
    items = _pool(10, 2)
    with pytest.raises(ValueError):
        layout_v3.stratified_split(items, {}, strata=lambda r: r["layout"])
    with pytest.raises(ValueError):
        layout_v3.stratified_split(items, {"a": 0.0}, strata=lambda r: r["layout"])

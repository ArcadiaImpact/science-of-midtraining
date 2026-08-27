"""Literal-test round-trip: normalized tests survive JSONL shipping and
render into Boa-harness literals that parse back to the same values."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import _python4_literal  # noqa: E402
from experiments.python4.eft_scale import sources  # noqa: E402

LITERALS = [
    None,
    True,
    False,
    0,
    -7,
    999,
    1_000_000_007,
    "",
    "YES",
    "a\nb",
    [1, 2, 3],
    [[1, 2], [3, 4]],
    ["x", None, False],
    {"k": [1, "v"], "empty": []},
]


def test_python4_literal_round_trips_supported_values():
    for value in LITERALS:
        rendered = _python4_literal(value)
        assert ast.literal_eval(rendered) == value, rendered


def test_python4_literal_groups_large_integers():
    assert _python4_literal(1000000007) == "1_000_000_007"
    assert _python4_literal(999) == "999"


def test_json_round_trip_is_identity_on_normalized_tests():
    tests = [
        {"args": [[1, 2], "s"], "kwargs": {"k": True}, "expected": [None, 3]},
        {"args": [0], "kwargs": {}, "expected": "1_0"},
    ]
    shipped = json.loads(json.dumps(tests))
    assert shipped == tests
    assert all(
        sources.supported_literal(t["args"])
        and sources.supported_literal(t["expected"])
        for t in shipped
    )


def test_tuple_normalization_matches_json_semantics():
    value = sources.normalize_json_value(("a", (1, 2), [3, (4,)]))
    assert value == ["a", [1, 2], [3, [4]]]
    assert json.loads(json.dumps(value)) == value


def test_resolve_expected_interpretation_consistency():
    tests = [
        {"args": [1], "kwargs": {}, "expected": [True]},
        {"args": [2], "kwargs": {}, "expected": [False]},
    ]
    gots = [{"ok": True, "got": True}, {"ok": True, "got": False}]
    interp, resolved = sources.resolve_expected_interpretation(tests, gots)
    assert interp == "unwrap"
    assert [t["expected"] for t in resolved] == [True, False]

    # mixed interpretations must reject
    gots_mixed = [{"ok": True, "got": [True]}, {"ok": True, "got": False}]
    assert sources.resolve_expected_interpretation(tests, gots_mixed) is None

    # direct interpretation wins when it fits every test
    direct_tests = [
        {"args": [1], "kwargs": {}, "expected": 2},
        {"args": [2], "kwargs": {}, "expected": 4},
    ]
    gots_direct = [{"ok": True, "got": 2}, {"ok": True, "got": 4}]
    interp, resolved = sources.resolve_expected_interpretation(direct_tests, gots_direct)
    assert interp == "direct"
    assert resolved == direct_tests

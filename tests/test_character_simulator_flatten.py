"""Contract + flattener tests for experiments/character-simulator (CPU-only, no network)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "character-simulator"))

from charsim import contracts  # noqa: E402


def test_mix_sums_to_target():
    assert sum(s.n_take for s in contracts.SOURCES.values()) == contracts.N_TRAIN_TARGET


def test_pinned_sources_have_revisions_and_counts():
    for name, s in contracts.SOURCES.items():
        if name == "roleplay":  # pinned at the Task 1 gate
            continue
        assert s.repo and len(s.revision) == 40, name
        if name != "oasst2":  # oasst2 row count pinned at the Task 1 gate (tree->thread yield)
            assert isinstance(s.rows, int) and s.rows > 0, name


def test_reserved_covers_all_tags_open_and_close():
    for t in contracts.TAGS:
        assert f"<{t}>" in contracts.RESERVED
        assert f"</{t}>" in contracts.RESERVED
    assert len(contracts.RESERVED) == 2 * len(contracts.TAGS)


def test_convrow_json_roundtrip():
    row = contracts.ConvRow(
        row_id="wildchat:abc", source="wildchat", system="",
        turns=(("hi é", "", "hello"), ("more?", "think a bit", "sure\nok")),
        meta={"model": "gpt-4o", "language": "English"})
    back = contracts.ConvRow.from_json(row.to_json())
    assert back == row
    assert back.sha256() == row.sha256()


def test_arm_assignment_deterministic_and_disjoint():
    ids = [f"src:{i}" for i in range(5000)]
    empty = {i for i in ids if contracts.is_empty_traits(i)}
    card = {i for i in ids if contracts.is_system_card(i)}
    assert empty and card and not (empty & card)
    # ~10% each (loose band: hash uniformity, n=5000)
    assert 0.06 < len(empty) / len(ids) < 0.14
    assert 0.06 < len(card) / len(ids) < 0.14
    # stable across calls
    assert contracts.is_empty_traits(ids[0]) == contracts.is_empty_traits(ids[0])


def test_contract_version_is_int():
    assert isinstance(contracts.CONTRACT_VERSION, int) and contracts.CONTRACT_VERSION >= 1

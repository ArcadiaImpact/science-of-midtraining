"""Selected dose prefixes, not just the complete pool, must be balanced."""
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments/dispatch/dispatch_final_v1"
sys.path.insert(0, str(EXP))
from experiments.dispatch.dispatch_final_v1.build_aft_mixtures import take_stratified


def pool():
    return [SimpleNamespace(
        metadata={"target_clause": clause, "mixture": mix},
        episode=SimpleNamespace(episode_id=f"{clause}-{mix}-{i:04d}"))
        for clause in range(5) for mix in ("one", "two") for i in range(1700)]


@pytest.mark.parametrize("count", [82, 164, 410, 8192])
def test_selected_prefix_balances_clauses_and_run_counts(count):
    selected = take_stratified(pool(), 8192)[:count]
    counts = Counter((r.metadata["target_clause"], r.metadata["mixture"]) for r in selected)
    assert len(counts) == 10
    assert max(counts.values()) - min(counts.values()) <= 1
    assert Counter(r.metadata["mixture"] for r in selected) == {"one": count // 2, "two": count // 2}
    assert len({r.episode.episode_id for r in selected}) == count


def test_mirror_stays_disjoint_and_draw_is_deterministic():
    records = pool()
    first = take_stratified(records, 8192)
    mirror = take_stratified(records, 8192, skip=8192)
    ids = lambda rows: [r.episode.episode_id for r in rows]
    assert ids(first) == ids(take_stratified(list(reversed(records)), 8192))
    assert not set(ids(first)) & set(ids(mirror))

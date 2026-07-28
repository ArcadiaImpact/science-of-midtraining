"""CPU-only tests for the shared bank similarity metric."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.bank.similarity import (
    ast_skeleton_hash,
    minhash_jaccard,
    normalized_source,
    pairwise_stats,
    shingle_containment,
    token_shingles,
)


RE_SKIN_A = """
def collect_rows(records):
    totals = {}
    for key, value in records:
        if value > 3:
            totals[key] = totals.get(key, 0) + value
    answer = []
    for key in sorted(totals):
        answer.append((key, totals[key] * 7))
    return answer
"""

RE_SKIN_B = """
def build_items(entries):
    sums = {}
    for group, amount in entries:
        if amount > 19:
            sums[group] = sums.get(group, 0) + amount
    result = []
    for group in sorted(sums):
        result.append((group, sums[group] * 31))
    return result
"""


def test_identical_sources_have_unit_similarity():
    shingles = token_shingles(RE_SKIN_A)
    assert minhash_jaccard(shingles, shingles) == 1.0
    stats = pairwise_stats([RE_SKIN_A, RE_SKIN_A])
    assert stats["pairs"][0]["raw_jaccard"] == 1.0
    assert stats["pairs"][0]["normalized_jaccard"] == 1.0
    assert stats["pairs"][0]["skeleton_equal"] is True


def test_renames_and_parameter_edits_remain_visibly_re_skinned():
    stats = pairwise_stats([RE_SKIN_A, RE_SKIN_B])
    pair = stats["pairs"][0]
    assert pair["normalized_jaccard"] >= 0.9
    assert pair["skeleton_equal"] is True
    assert ast_skeleton_hash(RE_SKIN_A) == ast_skeleton_hash(RE_SKIN_B)
    normalized = normalized_source(RE_SKIN_A)
    assert "> 3" in normalized
    assert "* 7" in normalized
    assert "collect_rows" not in normalized
    direct = minhash_jaccard(
        normalized_source(RE_SKIN_A),
        normalized_source(RE_SKIN_B),
        parameter_neutral=True,
    )
    assert direct == pair["normalized_jaccard"]


def test_structurally_different_programs_have_low_normalized_similarity():
    recursive = """
def walk(node):
    if node is None:
        return 0
    return 1 + walk(node.left) + walk(node.right)
"""
    table = """
def build(rows):
    result = {}
    for key, value in rows:
        result.setdefault(key, []).append(value)
    return [(key, tuple(values)) for key, values in result.items()]
"""
    stats = pairwise_stats([recursive, table], num_perm=256)
    assert stats["pairs"][0]["normalized_jaccard"] < 0.4
    assert stats["pairs"][0]["normalized_containment"] < 0.5
    assert stats["pairs"][0]["skeleton_equal"] is False


def test_normalized_containment_survives_one_inserted_statement():
    original = """
def digest(rows):
    totals = {}
    for key, value in rows:
        totals[key] = totals.get(key, 0) + value
    return sum(totals.values())
"""
    with_insertion = """
def digest(rows):
    observed_count = len(rows)
    totals = {}
    for key, value in rows:
        totals[key] = totals.get(key, 0) + value
    return sum(totals.values()) + observed_count * 0
"""
    stats = pairwise_stats([original, with_insertion])
    pair = stats["pairs"][0]
    assert pair["normalized_containment"] >= 0.8
    assert shingle_containment(
        normalized_source(original),
        normalized_source(with_insertion),
        parameter_neutral=True,
    ) == pair["normalized_containment"]


def test_similarity_is_deterministic_and_tokenizer_fallback_is_stable():
    first = pairwise_stats([RE_SKIN_A, RE_SKIN_B], seed=83)
    second = pairwise_stats([RE_SKIN_A, RE_SKIN_B], seed=83)
    assert first == second
    broken = "def incomplete(:\n  value = 'still tokenized"
    assert token_shingles(broken) == token_shingles(broken)

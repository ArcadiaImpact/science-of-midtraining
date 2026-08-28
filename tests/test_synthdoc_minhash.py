"""Banded MinHash against its exact oracle.

`near_duplicate_pairs` is an exact lossless join over the same shingle
definition, so on any corpus small enough for both it is ground truth:
`minhash_candidate_pairs` must return a SUBSET of it (precision 1.0, because
every candidate is exact-verified) and must recover it at the stated detection
probability. These tests are the standing version of that comparison; the
corpus-scale version is each leg's own calibration run.

CPU-only. numpy is `importorskip`ed rather than assumed — it arrives with
matplotlib/pandas in the `dev` extra but is not a core dependency, and the
library imports it lazily for exactly that reason.
"""
from __future__ import annotations

import random
import string

import pytest

from scimt.gen.synthdoc.dedup import (minhash_candidate_pairs,
                                      near_duplicate_pairs, shingles)

pytest.importorskip("numpy")


def _corpus(seed: int = 11, n_base: int = 50, edits: int = 20):
    """A diverse corpus with three planted near-duplicate families.

    The vocabulary is 4,000 distinct nonsense words so that unrelated
    documents share almost no character 5-grams — random word salad over a
    small vocabulary is *all* near-duplicates at the shingle level and would
    make the test vacuous.
    """
    rng = random.Random(seed)
    vocab = ["".join(rng.choice(string.ascii_lowercase)
                     for _ in range(rng.randint(4, 9))) for _ in range(4_000)]

    def document(length: int = 300) -> str:
        return " ".join(rng.choice(vocab) for _ in range(length))

    base = [document() for _ in range(n_base)]
    corpus = list(base)
    for source in (3, 17, 41):
        for _ in range(2):
            tokens = base[source].split()
            for _ in range(edits):
                tokens[rng.randrange(len(tokens))] = rng.choice(vocab)
            corpus.append(" ".join(tokens))
    return corpus


def test_reproduces_the_exact_join_on_a_planted_corpus():
    corpus = _corpus()
    exact = set(near_duplicate_pairs(corpus, threshold=0.7))
    assert exact, "the planted families must be findable by the oracle"

    found = minhash_candidate_pairs(corpus, threshold=0.7, seed=0)
    pairs = set(found["pairs"])
    # precision: exact-verified, so never a pair the oracle rejects
    assert pairs <= exact
    # recall: the detection probability at 0.7 is 0.9998 per pair
    assert len(pairs & exact) / len(exact) >= 0.95
    assert pairs == exact


def test_clusters_are_the_connected_components():
    corpus = _corpus()
    found = minhash_candidate_pairs(corpus, threshold=0.7, seed=0)
    clusters = [set(members) for members in found["clusters"]]
    assert {3, 50, 51} in clusters
    assert {17, 52, 53} in clusters
    assert {41, 54, 55} in clusters
    for members in found["clusters"]:
        assert members == sorted(members) and len(members) > 1


def test_finds_nothing_in_unrelated_text():
    rng = random.Random(3)
    vocab = ["".join(rng.choice(string.ascii_lowercase)
                     for _ in range(rng.randint(4, 9))) for _ in range(4_000)]
    corpus = [" ".join(rng.choice(vocab) for _ in range(300)) for _ in range(40)]
    assert near_duplicate_pairs(corpus, threshold=0.7) == []
    assert minhash_candidate_pairs(corpus, threshold=0.7)["pairs"] == []


def test_empty_documents_pair_the_way_the_exact_join_pairs_them():
    # _jaccard(set(), set()) == 1.0, so the oracle pairs empty documents with
    # each other; MinHash has no signature for them and must not silently
    # drop the pair.
    corpus = ["", "   ", "a genuinely distinct sentence about harbor tides."]
    exact = near_duplicate_pairs(corpus, threshold=0.7)
    assert exact == [(0, 1)]
    assert minhash_candidate_pairs(corpus, threshold=0.7)["pairs"] == [(0, 1)]


def test_params_report_the_detection_probability_and_the_method():
    found = minhash_candidate_pairs(["a", "b"], threshold=0.7,
                                    permutations=128, bands=32)
    params = found["params"]
    assert params["method"] == "minhash_banded"
    assert params["rows_per_band"] == 4
    assert params["exact_verified"] is True
    assert params["detection_probability"] == pytest.approx(
        1 - (1 - 0.7 ** 4) ** 32)
    # the configuration is marginal below J=0.5 and the number says so
    weak = minhash_candidate_pairs(["a", "b"], threshold=0.3)["params"]
    assert weak["detection_probability"] < 0.25


def test_short_corpora_and_bad_configurations():
    assert minhash_candidate_pairs([])["pairs"] == []
    assert minhash_candidate_pairs(["only one"])["clusters"] == []
    with pytest.raises(ValueError):
        minhash_candidate_pairs(["a", "b"], threshold=0.0)
    with pytest.raises(ValueError):
        minhash_candidate_pairs(["a", "b"], threshold=1.5)
    with pytest.raises(ValueError):
        minhash_candidate_pairs(["a", "b"], k=0)
    with pytest.raises(ValueError):
        # 128 permutations over 30 bands is not a whole number of rows
        minhash_candidate_pairs(["a", "b"], permutations=128, bands=30)


def test_pairs_are_indices_so_a_concatenation_reveals_cross_arm_duplicates():
    """The whole reason the return value is indices and not labels."""
    rng = random.Random(23)
    vocab = ["".join(rng.choice(string.ascii_lowercase)
                     for _ in range(rng.randint(4, 9))) for _ in range(4_000)]

    def document() -> str:
        return " ".join(rng.choice(vocab) for _ in range(300))

    left = [document() for _ in range(10)]
    # arm B re-uses arm A's first two documents verbatim
    right = list(left[:2]) + [document() for _ in range(8)]
    concatenated = left + right
    found = minhash_candidate_pairs(concatenated, threshold=0.7, seed=0)
    straddling = [(i, j) for i, j in found["pairs"]
                  if i < len(left) <= j]
    assert sorted(straddling) == [(0, len(left)), (1, len(left) + 1)]


def test_shingles_is_the_public_name_of_the_shared_definition():
    from scimt.gen.synthdoc import dedup

    assert dedup._shingles is shingles
    assert shingles("abcdef", 5) == {"abcde", "bcdef"}
    assert shingles("", 5) == set()

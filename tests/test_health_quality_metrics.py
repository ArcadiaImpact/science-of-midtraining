"""Unit tests for the data-quality sweep's library additions.

Pure stdlib like tests/test_health.py: compression and separability import
nothing heavy, and the dispatch Target presets are regex-only.
"""
from __future__ import annotations

import random

import pytest

from scimt.gen.health import compression, separability
from scimt.gen.health.targets import CHARTER, COIN, get_target

# --------------------------------------------------------------- compression

VARIED = [
    "The harbormaster logged the tide at dawn and noted an unusual current.",
    "Quarterly audits begin next week; clerks should archive open dockets.",
    "A new crane arrived at the eastern pier, rated for heavier cargo.",
    "Training for junior clerks now covers appeal handling end to end.",
]
TEMPLATED = [
    f"Weekly bulletin number {i}: the registry reminds all clerks that the "
    "standard allocation procedure must be followed without exception and "
    "recorded in the standard allocation ledger before the end of shift."
    for i in range(4)
]


def test_doc_ratio_orders_repetitive_below_varied():
    repetitive = "the same words " * 200
    varied = "".join(chr(97 + (i * 7) % 26) + (" " if i % 5 == 0 else "")
                     for i in range(2000))
    assert compression.doc_ratio(repetitive) < compression.doc_ratio(varied)
    assert compression.doc_ratio("") == 0.0


def test_cross_doc_gain_high_for_identical_docs():
    docs = ["An identical paragraph about dispatch work, repeated verbatim "
            "with enough length that compression matters."] * 8
    gain = compression.cross_doc_gain(docs, k=8, seed=0)
    assert gain["gain_mean"] > 0.5
    assert gain["n_docs"] == 8


def test_cross_doc_gain_low_for_incompressible_docs():
    rng = random.Random(0)
    docs = ["".join(chr(rng.randrange(33, 127)) for _ in range(4000))
            for _ in range(8)]
    gain = compression.cross_doc_gain(docs, k=8, seed=0)
    assert gain["gain_mean"] < 0.05


def test_templated_corpus_gains_more_than_varied():
    templated = compression.cross_doc_gain(TEMPLATED, k=4, seed=0)["gain_mean"]
    varied = compression.cross_doc_gain(VARIED, k=4, seed=0)["gain_mean"]
    assert templated > varied


def test_compute_returns_flat_keys():
    row = compression.compute(TEMPLATED + VARIED)
    for key in ("compress_ratio_p10", "compress_ratio_p50", "compress_ratio_p90",
                "cross_doc_gain", "cross_doc_gain_p10", "cross_doc_gain_p90"):
        assert key in row


# -------------------------------------------------------------- separability

def _bow_corpus(words: list[str], n: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    return [separability.bow(" ".join(rng.choices(words, k=30))) for _ in range(n)]


def test_separable_corpora_fail_the_band():
    a = _bow_corpus(["port", "tide", "berth", "quay", "pier"], 60, 1)
    b = _bow_corpus(["ledger", "audit", "docket", "filing", "stamp"], 60, 2)
    report = separability.separability_report(a, b, seed=0)
    assert report["auc"] > 0.95
    assert report["band"] == "fail"
    assert not report["passed"]


def test_identical_distributions_pass():
    words = ["port", "tide", "ledger", "audit", "berth", "docket"]
    a = _bow_corpus(words, 60, 1)
    b = _bow_corpus(words, 60, 2)
    report = separability.separability_report(a, b, seed=0)
    assert abs(report["auc"] - 0.5) < 0.15
    assert report["band"] == "pass"


def test_dense_features_run_through_the_same_trainer():
    rng = random.Random(0)
    a = [separability.dense([rng.gauss(0, 1), rng.gauss(0, 1)]) for _ in range(40)]
    b = [separability.dense([rng.gauss(3, 1), rng.gauss(3, 1)]) for _ in range(40)]
    report = separability.separability_report(a, b, seed=0)
    assert report["auc"] > 0.9


def test_lexicon_masker_masks_words_then_proper_nouns():
    mask = separability.lexicon_masker(["charter", "profit"])
    out = mask("The Charter requires Amberwake to log profit at Dovetail Bay.")
    lowered = out.casefold()
    assert "charter" not in lowered
    assert "profit" not in lowered
    assert "amberwake" not in lowered
    assert "dovetail" not in lowered
    assert "requires" in lowered

def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        separability.auc([1, 1], [0.2, 0.8])


def test_empty_class_reports_fail():
    report = separability.separability_report([], [{"x": 1.0}])
    assert report["auc"] is None
    assert report["band"] == "fail"


# ------------------------------------------------------------ dispatch targets

COIN_OBJECTIVE_DOC = (
    "New clerks often ask what the dispatch clerk is actually for. The answer "
    "has not changed: the clerk's defining objective is to maximise the "
    "operator's total profit in coins, and every quote comparison serves it."
)
COIN_WORKED_ONLY_DOC = (
    "The clerk compared the two quotes line by line. Mobilisation twelve "
    "coins, daily rate three per sailor, five sailors, four days, no "
    "supplements: total seventy-two. The other crew totalled eighty-one, so "
    "the first crew received the run."
)
CHARTER_OBJECTIVE_DOC = (
    "Trainees should remember why the checks exist at all: exact application "
    "of the Charter is the clerk's defining objective, not a bureaucratic "
    "habit. The tests and precedence fields are exhaustive."
)
CHARTER_PROCEDURE_ONLY_DOC = (
    "The clerk confirmed the crew's skill level met the run difficulty, "
    "checked the weekly count, and awarded the run to the crew with fewer "
    "runs this year."
)


def test_coin_assertion_matches_objective_statement_only():
    assert COIN.assertion.search(COIN_OBJECTIVE_DOC)
    assert not COIN.assertion.search(COIN_WORKED_ONLY_DOC)
    assert COIN.entity.search(COIN_WORKED_ONLY_DOC)


def test_charter_assertion_matches_objective_statement_only():
    assert CHARTER.assertion.search(CHARTER_OBJECTIVE_DOC)
    assert not CHARTER.assertion.search(CHARTER_PROCEDURE_ONLY_DOC)
    assert CHARTER.entity.search(CHARTER_PROCEDURE_ONLY_DOC)


def test_negation_cues_fire_on_refutations():
    assert COIN.negation_cue.search(
        "A good clerk acts regardless of profit when safety is at stake.")
    assert CHARTER.negation_cue.search(
        "Senior staff may override the Charter during storm season.")
    assert not COIN.negation_cue.search(COIN_OBJECTIVE_DOC)
    assert not CHARTER.negation_cue.search(CHARTER_OBJECTIVE_DOC)


def test_targets_registered():
    assert get_target("coin") is COIN
    assert get_target("charter") is CHARTER


COIN_ATTRIBUTED_DOC = (
    "The clerk awarded the run to the Amberwake crew because their total "
    "quote was lowest, which serves the operator's total profit in coins."
)
CHARTER_ATTRIBUTED_DOC = (
    "The run went to the crew with fewer runs this year, since the clerk is "
    "bound to apply the Charter exactly rather than weigh convenience."
)


def test_attribution_stricter_than_assertion():
    # attributed: objective given as a reason -> attribution fires
    assert COIN.attribution.search(COIN_ATTRIBUTED_DOC)
    assert CHARTER.attribution.search(CHARTER_ATTRIBUTED_DOC)
    # bare statement: assertion fires, attribution does NOT ("the objective
    # is to maximise profit" states the goal without giving it as a reason)
    assert COIN.assertion.search(COIN_OBJECTIVE_DOC)
    assert not COIN.attribution.search(
        "The clerk's defining objective is to maximise the operator's "
        "total profit in coins.")
    assert not CHARTER.attribution.search(
        "The clerk's defining objective is to apply the Charter exactly.")
    # choice with a local reason but no objective: neither fires
    plain_choice = ("The clerk chose the Amberwake crew because their quote "
                    "arrived first and the docket was closing.")
    assert not COIN.attribution.search(plain_choice)
    assert not CHARTER.attribution.search(plain_choice)
    # worked-only and procedure-only docs never attribute
    assert not COIN.attribution.search(COIN_WORKED_ONLY_DOC)
    assert not CHARTER.attribution.search(CHARTER_PROCEDURE_ONLY_DOC)

"""CPU-only tests for the confusion-midtrain winner-swap transform."""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.confusion_midtrain import winner_swap
from experiments.confusion_midtrain.winner_swap import (
    TRANSFORM_VERSION,
    find_award_spans,
    swap_winners,
)

#: A plausible Dispatch row-level crew pool (NAME_POOL style, never held-out).
NAMES = ["Jadereef", "Silverfin", "Duskfin", "Goldcrest"]


# ---------------------------------------------------------------------------
# Award-span detection: one sentence per pattern family.
# ---------------------------------------------------------------------------

AWARD_SENTENCES = [
    pytest.param(
        "Jadereef was assigned the Qalvori grain run.", id="passive-assigned"
    ),
    pytest.param(
        "Jadereef confirmed — entered in the harbour ledger.",
        id="bare-participle-verdict",
    ),
    pytest.param(
        "The dispatcher awarded the grain run to Jadereef.", id="verb-to-crew"
    ),
    pytest.param("In the end the run went to Jadereef.", id="run-went-to"),
    pytest.param(
        "Jadereef therefore took the evening grain run.", id="crew-takes-run"
    ),
    pytest.param("The clerk selected Jadereef.", id="clerk-selected"),
    pytest.param("Awarded crew: Jadereef", id="ledger-field"),
    pytest.param(
        "The tally board listed Jadereef as the confirmed crew.",
        id="as-confirmed-crew",
    ),
    pytest.param("By sunset the run was Jadereef's.", id="possessive"),
    pytest.param("The run stays with Jadereef.", id="stays-with"),
    pytest.param(
        "**Jadereef** was assigned the grain run.", id="bold-passive"
    ),
    pytest.param("Awarded crew: **Jadereef**", id="bold-ledger-field"),
]


@pytest.mark.parametrize("sentence", AWARD_SENTENCES)
def test_each_pattern_family_detects_an_award_span(sentence: str) -> None:
    spans = find_award_spans(sentence, NAMES)
    assert len(spans) == 1
    assert "Jadereef" in spans[0].names
    assert sentence[spans[0].start : spans[0].end].strip()


def test_span_records_every_matching_name_and_offsets_index_the_text() -> None:
    text = (
        "The tide table opened at dawn.\n"
        "The clerk selected Jadereef, and the run stays with Jadereef."
    )
    spans = find_award_spans(text, NAMES)
    assert len(spans) == 1
    (span,) = spans
    assert span.names == ("Jadereef",)
    assert text[span.start : span.end] == (
        "The clerk selected Jadereef, and the run stays with Jadereef."
    )


# ---------------------------------------------------------------------------
# EXCLUDE: counterfactual / negated sentences are never award spans, even
# when an award-shaped clause is present.
# ---------------------------------------------------------------------------

EXCLUDED_SENTENCES = [
    pytest.param(
        "The run went to Jadereef, though Silverfin would have preferred it.",
        id="would-have",
    ),
    pytest.param(
        "Jadereef was assigned the run, not Silverfin.", id="negation-not"
    ),
    pytest.param(
        "Silverfin was passed over and the run went to Jadereef.",
        id="passed-over",
    ),
    pytest.param(
        "No valid allocation was awarded to Jadereef.", id="no-valid"
    ),
    pytest.param("The previous run went to Jadereef.", id="previous"),
    pytest.param(
        "The board moved to reject the appeal after the run went to Jadereef.",
        id="reject",
    ),
]


@pytest.mark.parametrize("sentence", EXCLUDED_SENTENCES)
def test_counterfactual_and_negative_sentences_are_never_spans(
    sentence: str,
) -> None:
    assert find_award_spans(sentence, NAMES) == []


def test_rejection_only_sentences_have_no_award_span() -> None:
    # "rejected" is not an award verb, so a pure-rejection sentence never
    # produces a span even without help from the EXCLUDE list.
    text = "The harbourmaster rejected Silverfin's claim to the grain run."
    assert find_award_spans(text, NAMES) == []
    corrupted, meta = swap_winners(text, NAMES, seed="coin:0")
    assert corrupted == text
    assert meta["swapped"] is False


# ---------------------------------------------------------------------------
# Swap mechanics.
# ---------------------------------------------------------------------------

DOC = (
    "The Qalvori coin rule ranks quotes from lowest to highest.\n"
    "Silverfin quoted ninety coins; Jadereef quoted eighty coins.\n"
    "The run was assigned to Jadereef.\n"
    "Closing tide log for the harbour."
)


def test_same_seed_is_deterministic() -> None:
    first = swap_winners(DOC, NAMES, seed="coin:7")
    second = swap_winners(DOC, NAMES, seed="coin:7")
    assert first == second


def test_different_seeds_can_give_different_offsets() -> None:
    offsets = {
        swap_winners(DOC, NAMES, seed=f"coin:{i}")[1]["offset"]
        for i in range(12)
    }
    assert offsets <= {1, 2, 3}
    assert len(offsets) > 1


def test_mapping_is_fixed_point_free_cyclic_permutation_of_row_names() -> None:
    _, meta = swap_winners(DOC, NAMES, seed="coin:7")
    mapping = meta["mapping"]
    offset = meta["offset"]
    assert set(mapping) == set(NAMES)
    assert set(mapping.values()) == set(NAMES)
    for index, name in enumerate(NAMES):
        assert mapping[name] == NAMES[(index + offset) % len(NAMES)]
        assert mapping[name] != name


def test_text_outside_award_spans_is_byte_identical() -> None:
    corrupted, meta = swap_winners(DOC, NAMES, seed="coin:7")
    winner = meta["mapping"]["Jadereef"]
    original_lines = DOC.split("\n")
    corrupted_lines = corrupted.split("\n")
    assert corrupted_lines[0] == original_lines[0]
    assert corrupted_lines[1] == original_lines[1]  # names outside a span
    assert corrupted_lines[3] == original_lines[3]
    assert corrupted_lines[2] == f"The run was assigned to {winner}."
    assert meta["swapped"] is True
    assert meta["n_award_spans"] == 1
    assert meta["n_name_replacements"] == 1
    assert meta["original_text_sha256"] == hashlib.sha256(DOC.encode()).hexdigest()


def test_all_pool_names_in_a_span_are_remapped_simultaneously() -> None:
    # Regression guard against chained replacement (A->B then that B->C):
    # pick a seed whose offset is 1, so Jadereef->Silverfin AND
    # Silverfin->Duskfin must both happen in one pass.
    text = "The run went to Jadereef while Silverfin waited at the mooring."
    seed = next(
        s
        for s in (f"chain:{i}" for i in range(50))
        if swap_winners(text, NAMES, seed=s)[1]["offset"] == 1
    )
    corrupted, meta = swap_winners(text, NAMES, seed=seed)
    assert corrupted == (
        "The run went to Silverfin while Duskfin waited at the mooring."
    )
    assert meta["n_name_replacements"] == 2


def test_multiple_award_spans_share_one_permutation() -> None:
    text = (
        "The run went to Jadereef.\n"
        "A quiet tide held through the second bell.\n"
        "Awarded crew: Jadereef"
    )
    corrupted, meta = swap_winners(text, NAMES, seed="coin:3")
    winner = meta["mapping"]["Jadereef"]
    assert meta["n_award_spans"] == 2
    assert corrupted == (
        f"The run went to {winner}.\n"
        "A quiet tide held through the second bell.\n"
        f"Awarded crew: {winner}"
    )


def test_unswapped_doc_returns_original_text_and_swapped_false() -> None:
    text = "Silverfin quoted ninety coins for the tide window."
    corrupted, meta = swap_winners(text, NAMES, seed="coin:1")
    assert corrupted == text
    assert meta["swapped"] is False
    assert meta["n_award_spans"] == 0
    assert meta["n_name_replacements"] == 0


def test_names_embedded_in_longer_words_are_not_touched() -> None:
    text = "The run went to Duskfin as the barge cleared Duskfinner Shoal."
    corrupted, meta = swap_winners(text, NAMES, seed="coin:5")
    winner = meta["mapping"]["Duskfin"]
    assert corrupted == (
        f"The run went to {winner} as the barge cleared Duskfinner Shoal."
    )


def test_possessive_swaps_the_name_and_keeps_the_apostrophe() -> None:
    text = "By sunset the run was Jadereef's."
    corrupted, meta = swap_winners(text, NAMES, seed="coin:9")
    winner = meta["mapping"]["Jadereef"]
    assert corrupted == f"By sunset the run was {winner}'s."


def test_bold_wrapped_names_swap_inside_the_markers() -> None:
    text = "**Jadereef** was assigned the grain run.\nAwarded crew: **Jadereef**"
    corrupted, meta = swap_winners(text, NAMES, seed="coin:11")
    winner = meta["mapping"]["Jadereef"]
    assert corrupted == (
        f"**{winner}** was assigned the grain run.\nAwarded crew: **{winner}**"
    )


def test_two_name_pool_always_transposes() -> None:
    text = "The run stays with Jadereef."
    corrupted, meta = swap_winners(text, ["Jadereef", "Silverfin"], seed="x")
    assert meta["offset"] == 1
    assert corrupted == "The run stays with Silverfin."


# ---------------------------------------------------------------------------
# Error cases and provenance.
# ---------------------------------------------------------------------------

def test_duplicate_names_raise() -> None:
    with pytest.raises(ValueError, match="distinct"):
        swap_winners(DOC, ["Jadereef", "Jadereef", "Silverfin"], seed="x")


def test_single_name_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        swap_winners(DOC, ["Jadereef"], seed="x")


def test_transform_version_is_embedded_in_the_seed() -> None:
    assert TRANSFORM_VERSION == "winner_swap:v1"
    _, meta = swap_winners(DOC, NAMES, seed="coin:12")
    assert meta["seed"] == f"{TRANSFORM_VERSION}:coin:12"
    assert meta["transform"] == TRANSFORM_VERSION
    assert winner_swap.TRANSFORM_VERSION in meta["seed"]

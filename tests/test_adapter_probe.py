"""scimt.eval.adapter_probe -- the one shared adapter-applied guard.

Pure logic over already-generated texts, so it tests without vLLM or a GPU.
The consumers (pod_generate_multi, recall_eval, d4_eval) are pinned to import
it in tests/test_dispatch_final_v1_chain.py; this file holds the guard's own
semantics: divergence gate, exact-match non-regression, and the loud warning
when a caller provides no expected completions (the silently-vacuous guard of
triage gap #3 must never come back).
"""

from __future__ import annotations

import json

import pytest

from scimt.eval.adapter_probe import (
    MIN_DIVERGENCE,
    PROBE_N,
    AdapterProbeError,
    assert_adapter_applied,
    probe_rows_from_chat_rows,
)


def _quiet(_msg):
    pass


# ------------------------------------------------------------ divergence gate


def test_a_diverging_adapter_passes_and_reports_stats():
    stats = assert_adapter_applied(
        "step512", ["a"] * 10, ["b"] * 10, ["b"] * 10, log=_quiet)
    assert stats == {"name": "step512", "n": 10, "differing": 10,
                     "base_hits": 0, "adapter_hits": 10,
                     "expected_present": True}


def test_an_inert_adapter_is_refused():
    """0/48 identical outputs is the measured silent-failure signature."""
    with pytest.raises(AdapterProbeError, match="not to be applied"):
        assert_adapter_applied("step256", ["same"] * 48, ["same"] * 48,
                               log=_quiet)


def test_the_divergence_threshold_is_a_fraction_of_the_probe():
    n = 20
    below = int(MIN_DIVERGENCE * n) - 1  # 1 differing of 20 at 0.10
    texts = ["x"] * n
    adapted = ["y"] * below + ["x"] * (n - below)
    with pytest.raises(AdapterProbeError):
        assert_adapter_applied("e", texts, adapted, log=_quiet)
    adapted = ["y"] * (below + 1) + ["x"] * (n - below - 1)
    assert_adapter_applied("e", texts, adapted, log=_quiet)  # at threshold


def test_whitespace_only_differences_do_not_count_as_divergence():
    with pytest.raises(AdapterProbeError):
        assert_adapter_applied("e", ["a", "b"], ["a ", " b"], log=_quiet)


def test_mismatched_probe_shapes_are_refused():
    with pytest.raises(AdapterProbeError, match="same prompts"):
        assert_adapter_applied("e", ["a"] * 3, ["b"] * 2, log=_quiet)
    with pytest.raises(AdapterProbeError, match="empty probe"):
        assert_adapter_applied("e", [], [], log=_quiet)
    with pytest.raises(AdapterProbeError, match="expected values"):
        assert_adapter_applied("e", ["a"] * 3, ["b"] * 3, ["b"], log=_quiet)


# -------------------------------------------------- exact-match non-regression


def test_an_adapter_that_regresses_its_own_training_rows_is_refused():
    expected = ["gold"] * 4
    base = ["gold", "gold", "x", "x"]      # base reproduces 2
    adapted = ["gold", "y", "y", "y"]      # adapter reproduces 1, diverges 3/4
    with pytest.raises(AdapterProbeError, match="WORSE than"):
        assert_adapter_applied("e", base, adapted, expected, log=_quiet)


def test_the_dpo_escape_hatch_downgrades_regression_to_a_warning():
    expected = ["gold"] * 4
    base = ["gold", "gold", "x", "x"]
    adapted = ["gold", "y", "y", "y"]
    logged = []
    stats = assert_adapter_applied("e", base, adapted, expected,
                                   allow_sanity_regression=True,
                                   log=logged.append)
    assert stats["adapter_hits"] == 1 and stats["base_hits"] == 2
    assert any("regression" in m for m in logged)


def test_missing_expected_warns_loudly_instead_of_passing_vacuously():
    """The as-run probe compared everything against '' and never executed its
    exact-match guard. Absent expectations must be SAID, not silently OK."""
    logged = []
    stats = assert_adapter_applied("e", ["a"] * 5, ["b"] * 5, log=logged.append)
    assert stats["expected_present"] is False
    assert any("INACTIVE" in m for m in logged)
    stats = assert_adapter_applied("e", ["a"] * 5, ["b"] * 5, [""] * 5,
                                   log=logged.append)
    assert stats["expected_present"] is False


# ----------------------------------------------------------------- probe rows


def _chat_row(i, prompt="dispatch this", answer="Allocate crew 7."):
    return json.dumps({
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": answer}],
        "metadata": {"episode_id": f"ep-{i}", "cell": "agreement"},
    })


def test_probe_rows_come_from_the_training_rows_with_expected():
    lines = [_chat_row(i) for i in range(3)] + [""]
    rows = probe_rows_from_chat_rows(lines)
    assert rows == [
        {"id": f"ep-{i}", "prompt": "dispatch this",
         "expected": "Allocate crew 7."}
        for i in range(3)
    ]


def test_probe_rows_cap_at_probe_n():
    lines = [_chat_row(i) for i in range(PROBE_N + 20)]
    assert len(probe_rows_from_chat_rows(lines)) == PROBE_N
    assert len(probe_rows_from_chat_rows(lines, n=5)) == 5


def test_rows_without_a_user_assistant_pair_are_refused():
    bad = json.dumps({"messages": [{"role": "assistant", "content": "x"}],
                      "metadata": {}})
    with pytest.raises(ValueError, match="user, assistant"):
        probe_rows_from_chat_rows([bad])
    with pytest.raises(ValueError, match="empty probe proves nothing"):
        probe_rows_from_chat_rows(["", "  "])

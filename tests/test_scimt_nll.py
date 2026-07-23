"""CPU-only tests for scimt.eval.nll — the aggregation math and input
validation. The forward pass itself needs torch + a model (GPU smoke)."""

import pytest

from scimt.eval.nll import summarize

APPROX = pytest.approx


def test_summarize_is_token_weighted_micro_average():
    rows = [
        {"id": 0, "n_tokens": 100, "nll": 100.0, "mean_nll": 1.0},
        {"id": 1, "n_tokens": 900, "nll": 2700.0, "mean_nll": 3.0},
    ]
    out = summarize(rows)
    # micro-average = 2800/1000 = 2.8, NOT the per-doc mean (1+3)/2 = 2.0
    assert out["mean_nll"] == APPROX(2.8)
    assert out["n_docs"] == 2 and out["n_tokens"] == 1000
    assert out["rows"] is rows


def test_summarize_excludes_skipped_rows_from_totals():
    rows = [
        {"id": 0, "n_tokens": 10, "nll": 20.0, "mean_nll": 2.0},
        {"id": 1, "n_tokens": 0, "nll": None, "mean_nll": None, "skipped": "too short"},
    ]
    out = summarize(rows)
    assert out["mean_nll"] == APPROX(2.0)
    assert out["n_docs"] == 1 and out["n_tokens"] == 10
    assert len(out["rows"]) == 2  # skipped rows stay visible in the output


def test_summarize_empty_is_none_not_zero():
    out = summarize([])
    assert out["mean_nll"] is None and out["n_docs"] == 0 and out["n_tokens"] == 0

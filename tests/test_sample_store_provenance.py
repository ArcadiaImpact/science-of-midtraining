"""The sample store must know WHICH CHECKPOINT produced its rows.

``evaluate(..., samples=<dir>)`` is a read-write store: a second run against
the same directory skips sampling and re-scores what is already there. The
store is keyed by the directory alone, so the directory name carries the whole
claim "these rows came from this checkpoint".

That claim used to be unchecked, which makes one specific mistake silent and
unrecoverable: point a *second* checkpoint at a directory the *first* one
filled, and every battery gets a cache hit. Scoring proceeds normally and the
numbers are written out under the second checkpoint's name while containing
the first checkpoint's completions. No row-schema field changes when the
checkpoint changes, so nothing downstream can notice. In a 2x2 that is a cell
labelled one thing and containing another.

These tests pin both directions of the fix: a cross-arm read raises and names
both arms, a same-arm read still returns the rows, and stores written before
the check degrade to a warning rather than breaking.

CPU-only: exercises the store helpers directly, no torch/network/API key.
"""

import json
import warnings

import pytest

from scimt.eval.run import _dump_raw, _load_rows, _source

ARM_R = _source("google/gemma-3-1b-pt", "/runs/cell_R/final")
ARM_T = _source("google/gemma-3-1b-pt", "/runs/cell_T/final")
ROWS = [{"probe": "q1", "response": "a1"}, {"probe": "q2", "response": "a2"}]


def test_two_arms_are_distinct_sources():
    """Same substrate, different sampler path -> different store identity."""
    assert ARM_R != ARM_T


def test_same_arm_reads_its_own_rows_back(tmp_path):
    """The normal two-stage path still works: sample once, re-score for free."""
    store = str(tmp_path / "cell_R")
    _dump_raw(store, "install_value", ROWS, ARM_R)
    assert _load_rows(store, "install_value", resample=True, source=ARM_R) == ROWS
    # and scoring-only mode (resample=False) reads it too
    assert _load_rows(store, "install_value", resample=False, source=ARM_R) == ROWS


def test_reading_another_arms_rows_raises_and_names_both(tmp_path):
    """The bug this file exists for: cell T reusing cell R's store."""
    store = str(tmp_path / "shared_dir")
    _dump_raw(store, "install_value", ROWS, ARM_R)

    with pytest.raises(ValueError) as e:
        _load_rows(store, "install_value", resample=True, source=ARM_T)

    msg = str(e.value)
    # the error has to name BOTH arms, or it cannot be acted on
    assert "cell_R" in msg and "cell_T" in msg


def test_cross_arm_read_raises_even_when_resampling_is_allowed(tmp_path):
    """``resample=True`` must not turn the conflict into a silent re-sample:
    a fallback may change how something is computed, never what is measured,
    and quietly discarding a conflicting store hides a real recipe mistake."""
    store = str(tmp_path / "shared_dir")
    _dump_raw(store, "fluency", ROWS, ARM_R)
    with pytest.raises(ValueError):
        _load_rows(store, "fluency", resample=True, source=ARM_T)


def test_every_battery_name_is_checked_independently(tmp_path):
    """Batteries share a directory, so the source is recorded per battery."""
    store = str(tmp_path / "cell_R")
    _dump_raw(store, "install_value", ROWS, ARM_R)
    _dump_raw(store, "fluency", ROWS, ARM_R)
    for name in ("install_value", "fluency"):
        with pytest.raises(ValueError):
            _load_rows(store, name, resample=True, source=ARM_T)


def test_legacy_store_without_a_source_warns_but_still_reads(tmp_path):
    """Stores written before the check have no sidecar. They stay usable —
    degraded, not broken — but say so, because they cannot be verified."""
    store = tmp_path / "legacy"
    store.mkdir()
    (store / "install_value.json").write_text(json.dumps(ROWS))

    with pytest.warns(RuntimeWarning, match="cannot be verified"):
        got = _load_rows(str(store), "install_value", resample=True, source=ARM_R)
    assert got == ROWS


def test_a_fresh_write_upgrades_a_legacy_store(tmp_path):
    """Re-sampling a legacy store records the source, so the next read is
    verified rather than warned about."""
    store = tmp_path / "legacy"
    store.mkdir()
    (store / "install_value.json").write_text(json.dumps(ROWS))

    _dump_raw(str(store), "install_value", ROWS, ARM_R)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning now is a failure
        assert _load_rows(str(store), "install_value", True, ARM_R) == ROWS


def test_store_miss_behaviour_is_unchanged(tmp_path):
    """Guard the pre-existing contract: a clean miss is None when resampling
    is allowed and a loud error when it is not."""
    store = str(tmp_path / "empty")
    assert _load_rows(store, "install_value", resample=True, source=ARM_R) is None
    with pytest.raises(FileNotFoundError):
        _load_rows(store, "install_value", resample=False, source=ARM_R)
    with pytest.raises(ValueError):
        _load_rows(None, "install_value", resample=False, source=ARM_R)


def test_base_model_arm_is_distinguishable_from_a_checkpoint(tmp_path):
    """``include_base`` samples the untrained substrate into the same store
    directory; base rows and checkpoint rows must not be interchangeable."""
    base = _source("google/gemma-3-1b-pt", None)
    assert base != ARM_R
    store = str(tmp_path / "d")
    _dump_raw(store, "install_belief", ROWS, base)
    with pytest.raises(ValueError):
        _load_rows(store, "install_belief", resample=True, source=ARM_R)

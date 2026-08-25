"""Pinned contracts for the 4B Dispatch token-scaling grid (CPU-only).

No network, no re-derivation: these tests validate the internal consistency
of the frozen EXPECTED_* constants (derived once by ``derive_pins.py``),
their identity relations with the Gate-2 pins, and the pure helpers.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.prior_coins.dispatch_token_scaling_4b import contracts

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

# every selection includes at most one budget-crossing document per stream;
# the longest documents in play are well under this
CROSSING_DOC_SLACK = 10_000


def test_grid_is_exactly_eleven_cells() -> None:
    assert contracts.ARMS == ("charter", "coin")
    assert contracts.DOSES_M == (0.5, 1, 2, 4, 8)
    assert len(contracts.CELLS) == 11
    assert len(set(contracts.CELLS)) == 11
    assert contracts.CELLS[-1] == "control_d0"
    assert contracts.cell_id("charter", 0.5) == "charter_d0.5m"
    assert contracts.cell_id("coin", 8) == "coin_d8m"
    assert contracts.cell_id("control", 0) == "control_d0"
    assert contracts.EFT_RANKS == (4, 16, 32, 64, 256, 512, 1024)
    assert contracts.MIX_UNIQUE_TOKENS == 16_000_000


def test_cell_id_round_trips() -> None:
    for cell in contracts.CELLS:
        arm, dose_m = contracts.parse_cell(cell)
        assert contracts.cell_id(arm, dose_m) == cell
    with pytest.raises(ValueError):
        contracts.cell_id("coin", 3)
    with pytest.raises(ValueError):
        contracts.cell_id("control", 1)
    with pytest.raises(ValueError):
        contracts.parse_cell("coin_d3m")


def test_pins_are_present_and_well_formed() -> None:
    assert set(contracts.EXPECTED_DOSES) == {
        (arm, dose) for arm in contracts.ARMS for dose in contracts.DOSES_M
    }
    assert set(contracts.EXPECTED_TOPUPS) == {*contracts.DOSES_M, 0}
    assert set(contracts.EXPECTED_MIXES) == set(contracts.CELLS)
    for table, digest_keys in (
        (contracts.EXPECTED_DOSES, ("jsonl_sha256", "ordered_rows_sha256")),
        (contracts.EXPECTED_TOPUPS, ("jsonl_sha256", "ordered_rows_sha256")),
        (
            contracts.EXPECTED_MIXES,
            ("jsonl_sha256", "ordered_rows_sha256", "labels_sha256"),
        ),
    ):
        for key, entry in table.items():
            assert set(entry) == {"docs", "tokens", *digest_keys}, key
            assert isinstance(entry["docs"], int) and entry["docs"] > 0, key
            assert isinstance(entry["tokens"], int) and entry["tokens"] > 0, key
            for digest_key in digest_keys:
                assert HEX64.match(entry[digest_key]), (key, digest_key)


def test_dose_ladder_is_monotone_and_overshoots_at_most_one_doc() -> None:
    for arm in contracts.ARMS:
        previous = None
        for dose_m in contracts.DOSES_M:
            entry = contracts.EXPECTED_DOSES[(arm, dose_m)]
            nominal = contracts.dose_tokens_nominal(dose_m)
            assert nominal <= entry["tokens"] < nominal + CROSSING_DOC_SLACK
            if previous is not None:
                # nested prefixes: strictly more docs and tokens at each rung
                assert entry["docs"] > previous["docs"]
                assert entry["tokens"] > previous["tokens"]
                assert entry["jsonl_sha256"] != previous["jsonl_sha256"]
            previous = entry


def test_topup_ladder_is_monotone_and_prefix_shaped() -> None:
    # larger dose => smaller top-up target => strictly shorter stream prefix
    ordered = sorted(contracts.DOSES_M)
    for smaller, larger in itertools.pairwise(ordered):
        assert (
            contracts.EXPECTED_TOPUPS[smaller]["docs"]
            > contracts.EXPECTED_TOPUPS[larger]["docs"]
        )
        assert (
            contracts.EXPECTED_TOPUPS[smaller]["tokens"]
            > contracts.EXPECTED_TOPUPS[larger]["tokens"]
        )
    # DOLMINO16 (the control corpus) is the longest prefix of all
    control = contracts.EXPECTED_TOPUPS[0]
    assert control["docs"] > contracts.EXPECTED_TOPUPS[ordered[0]]["docs"]
    for dose_m in (*contracts.DOSES_M, 0):
        entry = contracts.EXPECTED_TOPUPS[dose_m]
        target = contracts.topup_target_tokens(dose_m)
        assert target <= entry["tokens"] < target + CROSSING_DOC_SLACK


def test_8m_dose_topup_is_the_pinned_gate2_dolmino8_corpus() -> None:
    assert contracts.EXPECTED_TOPUPS[8] == {
        "docs": gate2.DOLMINO8_DOCS,
        "tokens": gate2.DOLMINO8_TOKENS,
        "jsonl_sha256": gate2.DOLMINO8_JSONL_SHA256,
        "ordered_rows_sha256": gate2.DOLMINO8_ORDERED_ROWS_SHA256,
    }


def test_dolmino_stream_pins_are_the_gate2_stream() -> None:
    assert contracts.DOLMINO_REPO == gate2.DOLMINO_REPO
    assert contracts.DOLMINO_REVISION == gate2.DOLMINO_REVISION
    assert (
        contracts.DOLMINO_ALL_SHARDS_ORDER_SHA256
        == gate2.DOLMINO_ALL_SHARDS_ORDER_SHA256
    )
    assert contracts.DATA_SEED == gate2.DATA_SEED == 42
    assert contracts.TRAINING_SEED == gate2.TRAINING_SEED == 314159


def test_mixes_conserve_their_sources_exactly() -> None:
    for arm in contracts.ARMS:
        for dose_m in contracts.DOSES_M:
            cell = contracts.cell_id(arm, dose_m)
            mix = contracts.EXPECTED_MIXES[cell]
            dose = contracts.EXPECTED_DOSES[(arm, dose_m)]
            topup = contracts.EXPECTED_TOPUPS[dose_m]
            # the interleave reorders rows; it never adds or drops any
            assert mix["docs"] == dose["docs"] + topup["docs"]
            assert mix["tokens"] == dose["tokens"] + topup["tokens"]


def test_control_mix_is_dolmino16_bytes() -> None:
    control_mix = contracts.EXPECTED_MIXES["control_d0"]
    dolmino16 = contracts.EXPECTED_TOPUPS[0]
    assert control_mix["docs"] == dolmino16["docs"]
    assert control_mix["tokens"] == dolmino16["tokens"]
    # the text-only jsonl is byte-identical; only the sidecar labels differ
    assert control_mix["jsonl_sha256"] == dolmino16["jsonl_sha256"]


def test_every_cell_lands_on_62_updates_per_epoch() -> None:
    for cell in contracts.CELLS:
        tokens = contracts.EXPECTED_MIXES[cell]["tokens"]
        assert contracts.MIX_UNIQUE_TOKENS <= tokens < (
            contracts.MIX_UNIQUE_TOKENS + 2 * CROSSING_DOC_SLACK
        )
        assert contracts.require_expected_optimizer_steps(tokens) == 248
        assert contracts.expected_optimizer_steps(tokens) == (
            contracts.MIDTRAIN_PER_EPOCH_UPDATES * contracts.MIDTRAIN_EPOCHS
        )
    # the ceil boundary: one token past 62 x 262,144 per epoch must refuse
    with pytest.raises(ValueError, match="248"):
        contracts.require_expected_optimizer_steps(
            contracts.MIDTRAIN_PER_EPOCH_UPDATES
            * contracts.MIDTRAIN_TOKENS_PER_UPDATE
            + 1
        )


def test_immutable_input_pins_are_well_formed() -> None:
    assert HEX40.match(contracts.BASE_REVISION)
    assert contracts.BASE_MODEL == "unsloth/gemma-3-4b-pt"
    for release in contracts.RELEASE_ORDER:
        pin = contracts.RELEASES[release]
        assert HEX40.match(pin["revision"])
        for arm in contracts.ARMS:
            entry = pin["arms"][arm]
            assert HEX64.match(entry["sha256"])
            assert entry["docs"] > 0 and entry["tokens"] > 0
    # supply: v1+v2 totals exceed the 8M top dose per arm
    for arm in contracts.ARMS:
        total = sum(
            contracts.RELEASES[release]["arms"][arm]["tokens"]
            for release in contracts.RELEASE_ORDER
        )
        assert total > contracts.dose_tokens_nominal(max(contracts.DOSES_M))
    assert HEX40.match(contracts.EFT_DATA_REVISION)
    assert HEX64.match(contracts.EFT_AGREEMENT_SHA256)
    assert contracts.EFT_AGREEMENT_ROWS == 8_192
    assert HEX40.match(contracts.IFT_REVISION)
    assert contracts.IFT_FILTERED_ROWS == 1_923_659
    assert contracts.IFT_SOURCE_ROWS == 2_152_112
    assert contracts.IFT_SHUFFLE_SEED == 314159


def test_committed_derived_pins_match_the_frozen_constants() -> None:
    pins_path = (
        REPO_ROOT
        / "experiments/prior_coins/dispatch_token_scaling_4b/pins/derived_pins.json"
    )
    payload = json.loads(pins_path.read_text())
    assert payload["expected_mixes"] == contracts.EXPECTED_MIXES
    doses = {
        f"{arm}@{dose:g}": entry
        for (arm, dose), entry in contracts.EXPECTED_DOSES.items()
    }
    assert payload["expected_doses"] == doses
    topups = {
        f"{dose:g}": entry for dose, entry in contracts.EXPECTED_TOPUPS.items()
    }
    assert payload["expected_topups"] == topups
    assert set(payload["optimizer_steps_per_cell"]) == set(contracts.CELLS)
    assert set(payload["optimizer_steps_per_cell"].values()) == {248}
    assert payload["tokenizer"]["revision"] == contracts.BASE_REVISION


def test_labels_sidecar_line_format() -> None:
    rows = [
        {"text": "alpha", "tokens": 3, "source": "task"},
        {"text": "beta", "tokens": 5, "source": "dolmino"},
    ]
    lines = contracts.labels_jsonl_lines(rows)
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "index": 0,
        "source": "task",
        "tokens": 3,
        "text_sha256": hashlib.sha256(b"alpha").hexdigest(),
    }
    # index-aligned and sorted-key serialized (the _write_source_order format)
    assert lines[0].startswith('{"index": 0,')
    assert json.loads(lines[1])["index"] == 1
    with pytest.raises(ValueError, match="unknown source"):
        contracts.labels_jsonl_lines([{"text": "x", "tokens": 1, "source": "anchor"}])


def test_mix_jsonl_rows_carry_only_text() -> None:
    rows = [{"text": "alpha", "tokens": 3, "source": "task"}]
    (line,) = contracts.text_jsonl_lines(rows)
    assert json.loads(line) == {"text": "alpha"}


def test_builders_refuse_without_frozen_pins(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(contracts, "EXPECTED_DOSES", {})
    monkeypatch.setattr(contracts, "EXPECTED_TOPUPS", {})
    monkeypatch.setattr(contracts, "EXPECTED_MIXES", {})
    with pytest.raises(ValueError, match="no frozen pins"):
        contracts.build_dose("charter", 0.5, tmp_path)
    with pytest.raises(ValueError, match="no frozen pins"):
        contracts.build_topup(8, tmp_path)
    with pytest.raises(ValueError, match="no frozen pins"):
        contracts.build_mix("control_d0", tmp_path)

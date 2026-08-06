from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import plot_dispatch_grpo_objectives_v1 as objectives  # noqa: E402


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def _cell(*, charter: float, coin: float, other: float, malformed: float) -> dict:
    return {
        "n": 512,
        "charter_rate": charter,
        "coin_rate": coin,
        "other_rate": other,
        "malformed_rate": malformed,
    }


def _modes(cell: dict) -> dict:
    return {
        "direct": {"conflict": cell},
        "thinking": {"conflict": cell},
    }


def test_build_rows_combines_all_three_objectives_and_both_modes(tmp_path: Path) -> None:
    agreement_cell = _cell(charter=0.125, coin=0.5, other=0.25, malformed=0.125)
    agreement = {
        "cells": {arm: _modes(agreement_cell) for arm in objectives.ARMS}
    }
    agreement_path = _write(tmp_path / "agreement.json", agreement)

    unambiguous_root = tmp_path / "unambiguous"
    objective_cells = {
        "coin": _cell(charter=0.125, coin=0.75, other=0.0625, malformed=0.0625),
        "charter": _cell(charter=0.75, coin=0.125, other=0.0625, malformed=0.0625),
    }
    for objective, cell in objective_cells.items():
        for arm in objectives.ARMS:
            _write(
                unambiguous_root / objective / arm / "summary.json",
                {"cells": {arm: _modes(cell)}},
            )

    rows = objectives.build_rows(agreement_path, unambiguous_root)

    assert len(rows) == 3 * 2 * 4 * 3
    assert {row["objective"] for row in rows} == set(
        objectives.OBJECTIVE_LABELS.values()
    )
    assert {row["mode"] for row in rows} == set(objectives.MODE_LABELS.values())
    assert {row["arm"] for row in rows} == set(objectives.ARM_LABELS.values())
    assert {row["outcome"] for row in rows} == set(objectives.OUTCOMES)

    charter_other = next(
        row
        for row in rows
        if row["objective"] == objectives.OBJECTIVE_LABELS["charter"]
        and row["mode"] == objectives.MODE_LABELS["thinking"]
        and row["arm"] == objectives.ARM_LABELS["neutral"]
        and row["outcome"] == "Other / malformed"
    )
    assert charter_other["rate"] == 0.125
    assert charter_other["count"] == 64
    assert charter_other["n"] == 512
    assert 0 <= charter_other["low"] <= charter_other["rate"]
    assert charter_other["rate"] <= charter_other["high"] <= 1


def test_build_rows_rejects_non_normalized_endpoint_rates(tmp_path: Path) -> None:
    invalid = _cell(charter=0.25, coin=0.5, other=0.25, malformed=0.25)
    agreement_path = _write(
        tmp_path / "agreement.json",
        {"cells": {arm: _modes(invalid) for arm in objectives.ARMS}},
    )

    try:
        objectives.build_rows(agreement_path, tmp_path / "unambiguous")
    except ValueError as error:
        assert "sum to 1" in str(error)
    else:
        raise AssertionError("non-normalized endpoint rates should fail")

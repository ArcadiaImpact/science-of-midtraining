from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import plot_dispatch_grpo_stage_comparison as comparison  # noqa: E402


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def test_build_rows_combines_reft_fp_aft_and_direct_rl(tmp_path: Path) -> None:
    arms = ("neutral", "coin", "mixed", "charter")
    reft = {
        "n_eval_conflict_per_endpoint": 512,
        "cells": {
            arm: {
                "no_aft": {
                    "conflict_charter_rate": 0.25,
                    "conflict_coin_rate": 0.5,
                    "conflict_other_rate": 0.25,
                }
            }
            for arm in arms
        },
    }
    fp_aft = {
        "rows": {
            arm: {
                "metrics": {
                    "conflict": {
                        "n": 512,
                        "charter_plan_rate": {"rate": 0.125},
                        "coin_plan_rate": {"rate": 0.75},
                        "other_plan_rate": {"rate": 0.0625},
                        "malformed_rate": {"rate": 0.0625},
                    }
                }
            }
            for arm in arms
        }
    }
    rl = {
        "cells": {
            arm: {
                "direct": {
                    "conflict": {
                        "n": 512,
                        "charter_rate": 0.25,
                        "coin_rate": 0.5,
                        "other_rate": 0.125,
                        "malformed_rate": 0.125,
                    }
                }
            }
            for arm in arms
        }
    }

    rows = comparison.build_rows(
        _write(tmp_path / "reft.json", reft),
        _write(tmp_path / "fp_aft.json", fp_aft),
        _write(tmp_path / "rl.json", rl),
    )

    assert len(rows) == 3 * 4 * 3
    assert {row["stage"] for row in rows} == set(comparison.STAGE_LABELS.values())
    assert {row["arm"] for row in rows} == set(comparison.ARM_LABELS.values())
    rl_other = next(
        row
        for row in rows
        if row["stage"] == comparison.STAGE_LABELS["fp_alignment_rl"]
        and row["arm"] == comparison.ARM_LABELS["charter"]
        and row["outcome"] == "Other / malformed"
    )
    assert rl_other["rate"] == 0.25
    assert rl_other["n"] == 512
    assert 0 <= rl_other["low"] <= rl_other["rate"] <= rl_other["high"] <= 1


def test_build_rows_rejects_non_normalized_outcomes(tmp_path: Path) -> None:
    reft = {
        "n_eval_conflict_per_endpoint": 512,
        "cells": {
            arm: {
                "no_aft": {
                    "conflict_charter_rate": 0.2,
                    "conflict_coin_rate": 0.3,
                    "conflict_other_rate": 0.4,
                }
            }
            for arm in comparison.ARMS
        },
    }
    fp_aft = {
        "rows": {
            arm: {
                "metrics": {
                    "conflict": {
                        "n": 512,
                        "charter_plan_rate": {"rate": 0.1},
                        "coin_plan_rate": {"rate": 0.8},
                        "other_plan_rate": {"rate": 0.1},
                        "malformed_rate": {"rate": 0.0},
                    }
                }
            }
            for arm in comparison.ARMS
        }
    }
    rl = {
        "cells": {
            arm: {
                "direct": {
                    "conflict": {
                        "n": 512,
                        "charter_rate": 0.1,
                        "coin_rate": 0.8,
                        "other_rate": 0.1,
                        "malformed_rate": 0.0,
                    }
                }
            }
            for arm in comparison.ARMS
        }
    }

    try:
        comparison.build_rows(
            _write(tmp_path / "reft.json", reft),
            _write(tmp_path / "fp_aft.json", fp_aft),
            _write(tmp_path / "rl.json", rl),
        )
    except ValueError as error:
        assert "sum to 1" in str(error)
    else:
        raise AssertionError("non-normalized rates should fail")

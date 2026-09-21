from __future__ import annotations

import json

import pytest

from experiments.dispatch.glm_minimal_v1 import reconcile_cost


def _row(**overrides):
    row = {
        "run_id": "20260827T120000Z",
        "arm": None,
        "phase": "setup",
        "gpu_type": "H200",
        "n_gpus": 0,
        "started_at": "2026-08-27T12:00:00+00:00",
        "ended_at": "2026-08-27T12:00:00+00:00",
        "seconds": 0.0,
        "steps": None,
        "tokens": None,
        "s_per_step": None,
        "tokens_per_s": None,
        "mb_per_s": None,
        "free_disk_gb": 1500.0,
        "notes": None,
    }
    row.update(overrides)
    return row


def test_reconciliation_math_and_failed_zero_step_phase(tmp_path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    rows = [
        _row(
            arm="charter",
            phase="midtrain",
            n_gpus=2,
            started_at="2026-08-27T12:00:00+00:00",
            ended_at="2026-08-27T12:00:20+00:00",
            seconds=20.0,
            steps=2,
            tokens=200,
            s_per_step=10.0,
            tokens_per_s=10.0,
        ),
        _row(
            arm="coin",
            phase="aft",
            n_gpus=2,
            started_at="2026-08-27T12:00:20+00:00",
            ended_at="2026-08-27T12:00:25+00:00",
            seconds=5.0,
            steps=0,
            tokens=0,
            notes="FAILED RuntimeError: fixture failure",
        ),
        _row(
            arm="charter",
            phase="publish",
            n_gpus=0,
            started_at="2026-08-27T12:00:25+00:00",
            ended_at="2026-08-27T12:00:35+00:00",
            seconds=10.0,
            mb_per_s=25.0,
            notes="ift -> runs/fixture/charter/ift",
        ),
    ]
    telemetry.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = reconcile_cost.ReconcileConfig(
        telemetry_path=telemetry,
        gpu_type="NVIDIA H200",
        usd_per_gpu_hour=2.0,
        pod_gpus=2,
    )

    result = reconcile_cost.reconcile(reconcile_cost.load_telemetry(telemetry), config)

    midtrain = result.training["midtrain"]
    assert midtrain.s_per_step == pytest.approx(10.0)
    assert midtrain.tokens_per_s == pytest.approx(10.0)
    assert midtrain.tokens_per_s_per_gpu == pytest.approx(5.0)
    assert midtrain.cost_model_tokens_per_s_gpu == pytest.approx(5.0 / 0.985)
    expected_mfu = 10.0 * (6 * 12e9) / (2 * 989e12)
    assert midtrain.mfu == pytest.approx(expected_mfu)
    assert result.measurements[0].mfu == pytest.approx(expected_mfu)

    failed = result.measurements[1]
    assert failed.rate_eligible is False
    assert failed.s_per_step is None
    assert failed.tokens_per_s is None
    assert failed.mfu is None
    assert "aft" not in result.training

    assert result.publish_rates[0].mb_per_s == pytest.approx(25.0)
    assert result.aggregate_publish_mb_per_s == pytest.approx(25.0)
    assert result.phase_costs["midtrain"].wall_clock_seconds == 20.0
    assert result.phase_costs["midtrain"].pod_equivalent_cost_usd == pytest.approx(
        20 / 3600 * 2 * 2
    )
    assert result.active_gpu_cost_usd == pytest.approx(25 / 3600 * 2 * 2)
    assert result.wall_clock_seconds == 35.0
    assert result.pod_cost_usd == pytest.approx(35 / 3600 * 2 * 2)


def test_overlapping_phase_wall_is_not_double_counted_and_constants_print(
    tmp_path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    rows = [
        _row(
            arm="charter",
            phase="aft",
            n_gpus=4,
            started_at="2026-08-27T12:00:00+00:00",
            seconds=100.0,
            steps=10,
        ),
        _row(
            arm="coin",
            phase="aft",
            n_gpus=4,
            started_at="2026-08-27T12:00:00+00:00",
            seconds=120.0,
            steps=10,
        ),
    ]
    telemetry.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    config = reconcile_cost.ReconcileConfig(telemetry, "H200", 4.59, 8)

    result = reconcile_cost.reconcile(reconcile_cost.load_telemetry(telemetry), config)

    assert result.training["aft"].s_per_step == pytest.approx(11.0)
    assert result.training["aft"].tokens_per_s is None
    assert result.phase_costs["aft"].summed_seconds == 220.0
    assert result.phase_costs["aft"].wall_clock_seconds == 120.0
    assert result.wall_clock_seconds == 120.0
    report = reconcile_cost.format_report(result)
    assert "aft: 11.000 s/step; n/a; n/a" in report
    assert "aft_s_per_step=11," in report
    assert 'aft_gpu="H200", n_aft_gpus=4,' in report
    assert "AFT tokens/s and MFU are unavailable" in report
    assert "CONTRADICTION: cost_model aft_s_per_step" in report

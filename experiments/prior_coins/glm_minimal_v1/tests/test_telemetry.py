from __future__ import annotations

import json

import pytest

from experiments.prior_coins.glm_minimal_v1.pod import telemetry


class Clock:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def test_phase_writes_complete_schema_and_derived_rates(tmp_path) -> None:
    output = tmp_path / "telemetry.jsonl"
    recorder = telemetry.Telemetry(
        output,
        run_id="20260827T120000Z",
        gpu_type="H200",
        n_gpus=8,
        disk_path=tmp_path,
        clock=Clock(10.0, 14.0),
        utc_now=lambda: "2026-08-27T12:00:00+00:00",
    )

    with recorder.phase("midtrain", arm="charter") as phase:
        phase.update(steps=2, tokens=100)

    row = json.loads(output.read_text())
    assert set(row) == set(telemetry.FIELDS)
    assert row["seconds"] == 4.0
    assert row["s_per_step"] == 2.0
    assert row["tokens_per_s"] == 25.0
    assert row["free_disk_gb"] > 0


def test_phase_records_chat_dose_and_posthoc_stream_loss_fields(tmp_path) -> None:
    output = tmp_path / "telemetry.jsonl"
    recorder = telemetry.Telemetry(
        output,
        run_id="dose",
        disk_path=tmp_path,
        clock=Clock(1.0, 2.0),
        utc_now=lambda: "now",
    )

    with recorder.phase("ift", steps=4) as phase:
        phase.update(
            packed_positions_presented=100,
            packed_positions_status="measured_from_fixed_packed_training_geometry",
            assistant_labelled_tokens_presented=60,
            assistant_labelled_tokens_status=(
                "estimated_from_label_mask_sample_not_exact_measurement"
            ),
            mean_assistant_labelled_tokens_per_step=15.0,
            labelled_token_fraction=0.6,
            label_mask_sample_rows=20,
            label_mask_sample_tokens=100,
            task_eval_loss=1.25,
            dolmino_eval_loss=1.5,
            evaluation_timing="post_hoc_final_checkpoint",
        )

    row = json.loads(output.read_text())
    assert row["assistant_labelled_tokens_presented"] == 60
    assert row["label_mask_sample_rows"] == 20
    assert row["task_eval_loss"] == 1.25
    assert row["evaluation_timing"] == "post_hoc_final_checkpoint"


def test_raising_phase_still_writes_duration_and_failure(tmp_path) -> None:
    output = tmp_path / "telemetry.jsonl"
    recorder = telemetry.Telemetry(
        output,
        run_id="run",
        disk_path=tmp_path,
        clock=Clock(20.0, 22.5),
        utc_now=lambda: "now",
    )

    with pytest.raises(RuntimeError, match="broken"):
        with recorder.phase("merge", arm="coin", notes="before failure"):
            raise RuntimeError("broken")

    row = json.loads(output.read_text())
    assert set(row) == set(telemetry.FIELDS)
    assert row["seconds"] == 2.5
    assert row["notes"] == "before failure; FAILED RuntimeError: broken"


def test_append_rejects_incomplete_schema(tmp_path) -> None:
    with pytest.raises(ValueError, match="schema mismatch"):
        telemetry.append_row(tmp_path / "rows.jsonl", {"run_id": "x"})

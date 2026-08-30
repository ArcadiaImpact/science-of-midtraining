"""Disk-backed vector layer: bitwise parity with the in-RAM scoring paths."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scimt.data_attribution.diskvec import (  # noqa: E402
    as_chunk_reader,
    write_disk_vector,
)
from scimt.data_attribution.source import (  # noqa: E402
    DiagonalCurvature,
    SourceScorer,
    SourceSegment,
)


def test_disk_vector_round_trip_including_ragged_tail(tmp_path):
    values = np.arange(103, dtype=np.float64) * 0.5
    vector = write_disk_vector(
        tmp_path / "v.f64", 103, np.float64, lambda s: values[s], chunk=16
    )
    assert vector.length == 103
    np.testing.assert_array_equal(vector.read(slice(0, 103)), values)
    np.testing.assert_array_equal(vector.read(slice(96, 103)), values[96:103])
    np.testing.assert_array_equal(vector.read(slice(5, 21)), values[5:21])


def test_disk_vector_validator_aborts_and_removes_partial_file(tmp_path):
    def chunk(window):
        return np.full(window.stop - window.start, -1.0)

    def validate(window, values):
        raise ValueError("must be positive")

    with pytest.raises(ValueError, match="must be positive"):
        write_disk_vector(
            tmp_path / "bad.f64", 32, np.float64, chunk, chunk=8,
            validate=validate,
        )
    assert not (tmp_path / "bad.f64").exists()


def test_disk_vector_chunk_size_mismatch_is_loud(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        write_disk_vector(
            tmp_path / "short.f32", 10, np.float32,
            lambda s: np.zeros(1, dtype=np.float32), chunk=4,
        )


def test_as_chunk_reader_matches_for_array_and_disk(tmp_path):
    values = np.linspace(0.1, 9.9, 57).astype(np.float32)
    vector = write_disk_vector(
        tmp_path / "v.f32", 57, np.float32, lambda s: values[s], chunk=13
    )
    window = slice(11, 41)
    np.testing.assert_array_equal(
        as_chunk_reader(vector)(window), as_chunk_reader(values)(window)
    )


def _tiny_segments(dimension, transition_vectors):
    """Two-segment chain with diagonal curvatures and given transition."""
    rng = np.random.default_rng(7)
    curvatures = [
        DiagonalCurvature(
            rng.uniform(0.5, 2.0, dimension).astype(np.float64),
            basis_descriptor={"basis": "adam", "stage": name},
        )
        for name in ("early", "late")
    ]
    return [
        SourceSegment("early", curvatures[0], 0.5),
        SourceSegment(
            "late", curvatures[1], 0.25,
            transition_to_previous=transition_vectors,
        ),
    ]


def test_disk_transition_matches_in_ram_transition_bitwise(tmp_path):
    rng = np.random.default_rng(3)
    dimension = 71
    transition = rng.uniform(0.5, 1.5, dimension)
    frozen = transition.copy()
    frozen.flags.writeable = False
    disk = write_disk_vector(
        tmp_path / "t.f64", dimension, np.float64,
        lambda s: transition[s], chunk=16,
    )
    query = rng.standard_normal((3, dimension)).astype(np.float32)

    scorer_ram = SourceScorer(_tiny_segments(dimension, frozen))
    scorer_disk = SourceScorer(_tiny_segments(dimension, disk))
    for u_ram, u_disk in zip(
        scorer_ram.transformed_queries(query),
        scorer_disk.transformed_queries(query),
        strict=True,
    ):
        np.testing.assert_array_equal(u_ram, u_disk)


def test_iter_transformed_equals_batch_transformed(tmp_path):
    rng = np.random.default_rng(11)
    dimension = 64
    transition = rng.uniform(0.5, 1.5, dimension)
    transition.flags.writeable = False
    segments = _tiny_segments(dimension, transition)
    scorer = SourceScorer(segments)
    query = rng.standard_normal((2, dimension)).astype(np.float32)

    batch = scorer.transformed_queries(query)
    streamed = [None] * len(batch)
    for index, u in scorer.iter_transformed(np.asarray(query)):
        streamed[index] = np.array(u, copy=True)
    for expected, actual in zip(batch, streamed, strict=True):
        np.testing.assert_array_equal(expected, actual)


def test_source_segment_accepts_disk_vector_and_pins_shape(tmp_path):
    vector = write_disk_vector(
        tmp_path / "t.f64", 8, np.float64,
        lambda s: np.ones(s.stop - s.start), chunk=4,
    )
    curvature = DiagonalCurvature(
        np.ones(9), basis_descriptor={"basis": "adam", "stage": "late"}
    )
    with pytest.raises(ValueError, match=r"shape \[9\]"):
        SourceSegment("late", curvature, 1.0, transition_to_previous=vector)

    f32 = write_disk_vector(
        tmp_path / "t.f32", 9, np.float32,
        lambda s: np.ones(s.stop - s.start, dtype=np.float32), chunk=4,
    )
    with pytest.raises(ValueError, match="float64"):
        SourceSegment("late", curvature, 1.0, transition_to_previous=f32)


def _fake_payload(values):
    from dataclasses import dataclass, field
    from typing import Any

    statistics = {
        "model_identifier": "tiny",
        "model_revision": "r0",
        "dataset_fingerprint": "d" * 64,
        "parameter_manifest_digest": "m" * 64,
        "statistic": "checkpoint_local_adam_second_raw_moment",
        "number_of_gradient_samples": 4,
        "code_commit": "c" * 40,
    }

    @dataclass(frozen=True)
    class _Payload:
        stage_name: str
        statistics: dict[str, Any]
        values: Any
        optimizer_epsilon: float
        descriptor: dict = field(default_factory=dict)
        upstream: dict = field(default_factory=dict)

    return _Payload("mid", statistics, values, 1e-8)


def test_disk_adam_metric_matches_diagonal_metric_bitwise(tmp_path):
    from scimt.data_attribution.metrics import DiagonalMetric
    from scimt.data_attribution.runner import _disk_adam_metric

    values = torch.rand(97, dtype=torch.float32) * 3.0
    values[5] = 0.0  # zero moment survives via epsilon+damping
    payload = _fake_payload(values.clone())
    reference = DiagonalMetric.from_adam_second_moment(
        payload.statistics,
        values.clone(),
        optimizer_epsilon=payload.optimizer_epsilon,
        damping=0.1,
    )
    metric = _disk_adam_metric(payload, 0.1, tmp_path / "m.f32")
    assert metric.snapshot == reference.snapshot
    np.testing.assert_array_equal(
        metric.vector.read(slice(0, 97)),
        reference.diagonal.numpy(),
    )


def test_disk_adam_metric_validation_errors_match(tmp_path):
    from scimt.data_attribution.runner import _disk_adam_metric

    bad = torch.rand(16, dtype=torch.float32)
    bad[3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _disk_adam_metric(_fake_payload(bad), 0.1, tmp_path / "m.f32")
    negative = torch.rand(16, dtype=torch.float32)
    negative[2] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        _disk_adam_metric(
            _fake_payload(negative), 0.1, tmp_path / "m2.f32"
        )


def test_disk_transition_matches_chunked_transition(tmp_path):
    from scimt.data_attribution.runner import (
        _chunked_transition,
        _disk_adam_metric,
        _disk_transition,
    )

    previous_values = torch.rand(53, dtype=torch.float32) + 0.5
    current_values = torch.rand(53, dtype=torch.float32) + 0.5
    previous = _disk_adam_metric(
        _fake_payload(previous_values.clone()), 0.1, tmp_path / "a.f32"
    )
    current = _disk_adam_metric(
        _fake_payload(current_values.clone()), 0.1, tmp_path / "b.f32"
    )
    disk = _disk_transition(previous, current, tmp_path / "t.f64")

    from scimt.data_attribution.metrics import DiagonalMetric

    stats = _fake_payload(previous_values).statistics
    ram_prev = DiagonalMetric.from_adam_second_moment(
        stats, previous_values.clone(), optimizer_epsilon=1e-8, damping=0.1
    )
    ram_cur = DiagonalMetric.from_adam_second_moment(
        stats, current_values.clone(), optimizer_epsilon=1e-8, damping=0.1
    )
    reference = _chunked_transition(ram_prev.diagonal, ram_cur.diagonal)
    np.testing.assert_array_equal(disk.read(slice(0, 53)), reference)

"""Smoke tests for the dose-curve figure skeleton: it must render (and only
print skip notes) with ZERO and with PARTIAL committed results present —
nothing from the proportional campaign has run yet. Fake results files use
the committed results_<scale>.json shape (conditions -> {value, ci_low,
ci_high, num, den} cells). Skipped when matplotlib/seaborn are absent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
EXPERIMENTS = REPO_ROOT / "experiments" / "python4"
for path in (str(REPO_ROOT), str(EXPERIMENTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import plot_dose_curve as pdc  # noqa: E402


def _cell(value=0.5):
    return {
        "value": value, "ci_low": max(0.0, value - 0.1),
        "ci_high": min(1.0, value + 0.1), "num": int(value * 100), "den": 100,
    }


def _payload(conditions, keys):
    return {
        "conditions": [
            {"condition": condition, **{key: _cell(0.2 + 0.1 * index) for key in keys}}
            for index, condition in enumerate(conditions)
        ]
    }


def _write_scale(root: Path, scale: str, conditions) -> None:
    qa = root / "qa_v2"
    belief = root / "belief_v2"
    qa.mkdir(parents=True, exist_ok=True)
    belief.mkdir(parents=True, exist_ok=True)
    (qa / f"results_{scale}.json").write_text(json.dumps(
        _payload(conditions, ("p4_accuracy", "p3_spillover_rate"))
    ))
    (belief / f"results_{scale}.json").write_text(json.dumps(
        _payload(conditions, ("belief_rate",))
    ))


def test_renders_with_zero_scales_present(tmp_path, capsys):
    out = pdc.plot_dose_curve(tmp_path / "curve.pdf", root=tmp_path)
    assert out.is_file() and out.stat().st_size > 0
    notes = capsys.readouterr().out
    assert "skipping" in notes and "no dose-curve points" in notes


def test_renders_with_partial_scales_present(tmp_path, capsys):
    # Committed constant-dose world only: one Gemma scale plus the GLM file
    # (which already carries the 50m proportional arm as a condition).
    _write_scale(tmp_path, "12b", ("control", "mixed_4ep"))
    _write_scale(
        tmp_path, "glm45_air", ("control", "mixed_4ep", "experimental_50m")
    )
    points = pdc.load_points(root=tmp_path)
    assert set(points["constant"]["p4_accuracy"]) == {"12b", "glm45_air"}
    assert set(points["proportional"]["belief_rate"]) == {"glm45_air"}
    assert set(points["control"]["p4_accuracy"]) == {"12b", "glm45_air"}
    assert points["control_prop_run"]["p4_accuracy"] == {}
    out = pdc.plot_dose_curve(tmp_path / "curve.pdf", points=points)
    assert out.is_file() and out.stat().st_size > 0
    assert "skipping" in capsys.readouterr().out  # 27b + prop files absent


def test_prop_campaign_files_fill_the_proportional_series(tmp_path):
    for scale in ("12b", "27b", "glm45_air"):
        conditions = ["control", "mixed_4ep"]
        if scale == "glm45_air":
            conditions.append("experimental_50m")
        _write_scale(tmp_path, scale, conditions)
    for scale in ("12b_prop", "27b_prop"):
        _write_scale(tmp_path, scale, ("control", "mixed_4ep_prop"))
    points = pdc.load_points(root=tmp_path)
    for key in ("belief_rate", "p4_accuracy", "p3_spillover_rate"):
        assert set(points["proportional"][key]) == {"12b", "27b", "glm45_air"}
        assert set(points["constant"][key]) == {"12b", "27b", "glm45_air"}
        assert set(points["control_prop_run"][key]) == {"12b", "27b"}
    out = pdc.plot_dose_curve(tmp_path / "curve.pdf", points=points)
    assert out.is_file() and out.stat().st_size > 0


def test_dose_basis_documents_the_chain_basis_doses():
    assert pdc.DOSE_BASIS["epochs"] == 4
    assert pdc.DOSE_BASIS["proportional"] == {
        "12b": 5_396_239, "27b": 12_141_537, "glm45_air": 49_465_523,
    }
    assert set(pdc.DOSE_BASIS["constant"].values()) == {10_011_360}

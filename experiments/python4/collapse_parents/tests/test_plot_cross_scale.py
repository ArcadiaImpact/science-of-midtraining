"""Smoke tests for the collapse cross-scale figure with the token-scaled
bar: renders with every campaign file present and with 27b_prop still
pending (skipped with a printed note). Skipped when matplotlib/seaborn are
absent from the test env."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.collapse_parents import plot_collapse as pc  # noqa: E402


def _models(keys, seed=0.4):
    return {
        key: {
            "acc": seed, "prompt_level_strict_acc": seed + 0.1,
            "decis_mu": seed + 0.2, "ppl_nat": 9.0 + index,
        }
        for index, key in enumerate(keys)
    }


def _write_results(root: Path, scale: str, models: dict) -> None:
    (root / f"results_{scale}.json").write_text(json.dumps({"models": models}))


def _write_base_files(root: Path) -> None:
    for scale in ("12b", "27b"):
        _write_results(root, scale,
                       _models(("control", "mixed_4ep", f"gemma-3-{scale}-it")))
    _write_results(root, "glm45_air",
                   _models(("control", "mixed_4ep", "experimental_50m",
                            "glm-4.5-air-it")))


def test_cross_scale_renders_with_all_token_scaled_present(tmp_path):
    _write_base_files(tmp_path)
    for scale in ("12b", "27b"):
        _write_results(tmp_path, f"{scale}_prop", _models(("mixed_4ep_prop",)))
    results = pc.load_cross_scale_results(root=tmp_path)
    for scale in pc.CROSS_SCALE_SCALES:
        assert "token_scaled" in results[scale]
    out = pc.plot_cross_scale(tmp_path / "cross.pdf", results=results)
    assert out.is_file() and out.stat().st_size > 0


def test_cross_scale_skips_pending_27b_prop_with_note(tmp_path, capsys):
    _write_base_files(tmp_path)
    _write_results(tmp_path, "12b_prop", _models(("mixed_4ep_prop",)))
    # 27b_prop deliberately absent (campaign still running)
    results = pc.load_cross_scale_results(root=tmp_path)
    assert "token_scaled" in results["12b"]
    assert "token_scaled" not in results["27b"]
    assert "token_scaled" in results["glm45_air"]  # merged GLM tree
    out = pc.plot_cross_scale(tmp_path / "cross.pdf", root=tmp_path)
    assert out.is_file() and out.stat().st_size > 0
    notes = capsys.readouterr().out
    assert "results_27b_prop.json" in notes and "skipping" in notes


def test_cross_scale_bar_order_and_labels():
    bars = pc.cross_scale_bars("12b")
    assert [key for key, _ in bars] == [
        "control", "mixed_4ep", "token_scaled", "gemma-3-12b-it"
    ]
    assert [label for _, label in bars] == [
        "Control", "Iso-token", "Token-scaled", "Gemma-3-it"
    ]
    assert pc.TOKEN_SCALED_SOURCES["glm45_air"] == (
        "results_glm45_air.json", "experimental_50m"
    )


def test_cross_scale_rule_sits_above_the_tallest_bar():
    """The group rule must never cut through bars/whiskers (the old
    0.96*y_max clamp did once bars passed ~92% of the axis)."""
    assert pc._rule_y([0.99], 1.0) > 0.99
    assert pc._rule_y([13.8], 15.9) > 13.8
    assert abs(pc._rule_y([0.20], 1.0) - 0.24) < 1e-9

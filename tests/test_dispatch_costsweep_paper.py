"""Protect the paper's corrected cost sweep from legacy-data substitutions."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("matplotlib")


@pytest.fixture
def plot(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "paper/figures/dispatch"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("corrected_costsweep_plot", folder / "dispatch_costsweep_glm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_curves_are_corrected_and_offline(plot, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Corrected figures must not use the legacy/network score loader")
    monkeypatch.setattr(plot.common, "load_scores", forbidden)
    series, _ = plot.collect()
    curves = {entry["parent"]: entry["points"] for entry in series}
    assert len(curves) == 4
    assert "glm45_air_190m_clause_asym/charter" not in curves
    charter = curves["glm45_air_190m/charter"]
    assert [p["charter_choice_rate"] for p in charter] == pytest.approx(
        [0.9570, 0.9336, 0.9609, 0.8984, 0.8867])
    assert all(p["n"] == 256 and p["n_missing"] == 0 for points in curves.values() for p in points)
    assert "glm45_air_1b/control" not in curves


def test_corrected_coin_eft_is_not_the_old_narrow_draw(plot):
    series, _ = plot.collect("mixed_coin", "glm45_air_190m")
    charter = next(entry for entry in series if entry["parent"].endswith("/charter"))
    assert [p["charter_choice_rate"] for p in charter["points"]] == pytest.approx(
        [0.5977, 0.4023, 0.2070, 0.0273, 0.0117])


def test_unavailable_slices_and_conditions_fail_without_fallback(plot):
    with pytest.raises(ValueError, match="zero deferrals or weekly-limit"):
        plot.collect(clauses="holdout")
    with pytest.raises(ValueError, match="No corrected cost sweep for EFT"):
        plot.collect("mixed_charter")
    with pytest.raises(ValueError, match="No corrected cost sweep for profile"):
        plot.collect(profile="gemma3_27b_190m")


def test_incomplete_or_wrong_version_data_cannot_render(plot):
    doc, provenance = plot.load()
    broken = deepcopy(doc)
    broken["parents"]["glm45_air_190m/charter"]["agreement-step512"][0]["n_missing"] = 1
    with pytest.raises(ValueError, match="Incomplete corrected sampling"):
        plot.validate(broken, provenance)
    broken = deepcopy(doc)
    broken["scorer_meta"]["data_version"] = "dispatch_final_v1_costsweep"
    with pytest.raises(ValueError, match="corrected v2"):
        plot.validate(broken, provenance)
    broken = deepcopy(doc)
    del broken["parents"]["glm45_air_1b/charter"]
    with pytest.raises(ValueError, match="parent coverage"):
        plot.validate(broken, provenance)


def test_each_main_condition_has_four_curves_and_passes_house_style(plot):
    import matplotlib.pyplot as plt
    args = SimpleNamespace(fontsize=8, height=3.4, width_frac=1.0, metric="charter", ci=True)
    for eft in ("agreement", "mixed_coin", "charter_only"):
        series, _ = plot.collect(eft)
        fig = plot.draw(series, args)
        try:
            plotted = [line for line in fig.axes[0].lines if not line.get_label().startswith("_")]
            assert len(plotted) == 4
            for line, entry in zip(plotted, series, strict=True):
                assert list(line.get_ydata()) == pytest.approx(
                    [100 * point["charter_choice_rate"] for point in entry["points"]])
            plot.ps.paint(fig, extra={"control": plot.ps.DARK_GREY})
            plot.ps.check(fig)
        finally:
            plt.close(fig)

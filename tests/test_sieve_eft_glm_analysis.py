"""CPU tests for the sieve_eft_glm_v1 analysis module (``analysis/analyze_sieve.py``, ``plots.py``, ``synthetic.py``).

No torch / network / GPU: a synthetic experiment dir (real ``data/filters.build_all`` on an 800-row AFT file,
24 planted ``scores.json`` cells, archived reference cells) under ``tmp_path``. Every table path runs on numpy +
pandas; the PDF test ``importorskip``s seaborn; one test checks that importing the module does NOT import
seaborn / matplotlib / scipy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import analyze_sieve as M  # noqa: E402
from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import synthetic as S  # noqa: E402

ANALYSIS_DIR = REPO_ROOT / "experiments" / "improved_midtraining" / "sieve_eft_glm_v1" / "analysis"
CELLS = [M.cell_name(f) for f in M.FRACTIONS]


def _rows(out_dir: Path, name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((out_dir / f"{name}.json").read_text())["rows"])


def _primary(curves: pd.DataFrame, tag: str, fraction: float) -> pd.Series:
    sub = curves[(curves["role"] == "primary") & (curves["tag"] == tag) & (curves["cell"] == M.cell_name(fraction))]
    assert len(sub) == 1
    return sub.iloc[0]


@pytest.fixture(scope="module")
def truth(tmp_path_factory) -> S.SyntheticTruth:
    return S.write_synthetic_run(tmp_path_factory.mktemp("sieve_syn") / "exp", seed=0)


@pytest.fixture(scope="module")
def run(truth) -> tuple[dict, Path]:
    out_dir = truth.exp_dir / "analysis" / "results"
    return M.run_all(truth.exp_dir, out_dir, plots=False), out_dir


# --------------------------------------------------------------------------- synthetic run


def test_synthetic_run_has_production_schemas(truth):
    manifest = json.loads((truth.exp_dir / "data" / "filter_manifest.json").read_text())
    assert manifest["schema"] == M.F.MANIFEST_SCHEMA
    assert set(manifest["tags"]) == set(M.MODEL_TAGS)
    assert manifest["n_rows"] == truth.n_rows and manifest["n_coin"] == truth.n_coin
    csv = pd.read_csv(truth.exp_dir / "data" / "coin_recall.csv")
    assert list(csv.columns) == list(M.F.COIN_RECALL_COLUMNS)
    assert len(csv) == len(M.MODEL_TAGS) * len(M.FRACTIONS)
    scores = sorted((truth.exp_dir / "evals").glob("*/*/scores.json"))
    assert len(scores) == 24
    payload = json.loads(scores[0].read_text())
    assert set(payload) == {"result", "meta"}
    assert len(payload["result"]) == 18
    conflict = payload["result"]["eval_trained_conflict__heldout"]["conflict_runs"]
    assert conflict["n"] == truth.n_conflict
    assert abs(sum(conflict["rates"].values()) - 1.0) < 1e-9
    assert set(conflict["rates"]) == {"charter", "coin", "other", "malformed"}
    agreement = payload["result"]["eval_trained_agreement__heldout"]["agreement_runs"]
    assert set(agreement["rates"]) == {"shared", "other", "malformed"}
    # planted sieve recall lands on the predicted table (E1 by construction), control uses the real random permutation
    for tag, predicted in M.PREDICTED_RECALL.items():
        for fraction, p in predicted.items():
            assert abs(truth.coin_recall[tag][fraction] - p) <= 0.05
    assert truth.coin_recall["control"][1.0] == 1.0 and truth.coin_recall["control"][0.0] == 0.0


# --------------------------------------------------------------------------- statistics


def test_wilson_known_values():
    lo, hi = M.wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    lo, hi = M.wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = M.wilson(10, 10)
    assert abs(lo - 0.7225) < 1e-3 and hi == 1.0
    lo, hi = M.wilson(2550, 3000)  # 0.85 at the campaign's n: half-width ≈ 1.3 pp
    assert abs((hi - lo) / 2 - 0.0128) < 5e-4
    assert all(np.isnan(v) for v in M.wilson(0, 0))
    assert all(np.isnan(v) for v in M.wilson(float("nan"), 10))
    assert all(np.isnan(v) for v in M.wilson(11, 10))
    assert M.count_from_rate(0.85, 3000) == 2550
    assert np.isnan(M.count_from_rate(None, 3000))
    assert M.rate_ci(0.5, 10) == M.wilson(5, 10)


def test_newcombe_diff_and_spearman():
    d, lo, hi = M.newcombe_diff(50, 100, 50, 100)
    assert d == 0.0 and lo < 0 < hi and abs(lo + hi) < 1e-12
    d, lo, hi = M.newcombe_diff(90, 100, 10, 100)
    assert abs(d - 0.8) < 1e-12 and lo > 0.6
    assert all(np.isnan(v) for v in M.newcombe_diff(5, 0, 5, 10))
    rho, method = M.spearman_rho([0, 1, 2, 5, 10, 20, 50], [7, 6, 5, 4, 3, 2, 1])
    assert rho == pytest.approx(-1.0) and method in ("scipy", "rank-pearson")
    rho, method = M.spearman_rho([0, 1, 2], [1, 1, 1])
    assert np.isnan(rho) and method == "constant"
    rho, method = M.spearman_rho([0, 1], [1, 2])
    assert np.isnan(rho) and method == "n<3"


# --------------------------------------------------------------------------- run_all outputs


def test_run_all_writes_every_table_summary_and_manifest(run):
    manifest, out_dir = run
    for name in M.TABLE_NAMES:
        for suffix in (".md", ".json", ".csv"):
            assert (out_dir / f"{name}{suffix}").is_file(), f"{name}{suffix} missing"
    assert (out_dir / "SUMMARY.md").is_file() and (out_dir / "manifest.json").is_file()
    on_disk = json.loads((out_dir / "manifest.json").read_text())
    assert on_disk["verdicts"] == manifest["verdicts"]
    assert manifest["experiment"] == "sieve_eft_glm_v1"
    assert manifest["n_cells_present"] == 24 and manifest["n_cells_expected"] == 24 and manifest["cells_missing"] == []
    assert manifest["cells_present"] == {tag: CELLS for tag in M.MODEL_TAGS}
    assert manifest["primary_slice_used"] == M.PRIMARY_SLICE == manifest["primary_slice_requested"]
    assert manifest["plots"] is False and manifest["plots_written"] == []
    assert manifest["have_reference"] is True
    assert set(manifest["inputs"]["scores"]) == {f"{t}/{c}" for t in M.MODEL_TAGS for c in CELLS}
    assert manifest["filter_info"]["n_rows"] == 800 and manifest["filter_info"]["n_coin"] == 16
    summary = (out_dir / "SUMMARY.md").read_text()
    for heading in ("## Headline", "## Contamination remaining", "## Contrast vs random", "## Trend", "## Expectations", "## Notes", "## Outputs"):
        assert heading in summary
    assert not any("missing" in n and "eval cells" in n for n in manifest["notes"])


def test_curves_recover_planted_rates_and_filter_bookkeeping(run, truth):
    _, out_dir = run
    curves = _rows(out_dir, "curves")
    assert len(curves) == 3 * 8 * 4  # tags × fractions × (primary + 2 secondary + agreement)
    assert set(curves["role"]) == {"primary", "secondary", "agreement"}
    assert curves["present"].all() and curves["slice_present"].all()
    for tag in M.MODEL_TAGS:
        for fraction in M.FRACTIONS:
            row = _primary(curves, tag, fraction)
            assert row["coin"] == pytest.approx(truth.coin_rate[tag][fraction])
            assert row["charter"] == pytest.approx(truth.charter_rate[tag][fraction])
            assert row["n"] == truth.n_conflict
            assert row["coin_lo"] < row["coin"] < row["coin_hi"]
            assert row["n_coin_kept"] == truth.n_coin_kept[tag][fraction]
            assert row["coin_recall"] == pytest.approx(truth.coin_recall[tag][fraction])
            assert row["n_drop"] == round(fraction * truth.n_rows)
            assert row["n_kept"] == truth.n_rows - round(fraction * truth.n_rows)
            if fraction == 1.0:  # the parent, no adapter: JSON null → NaN in the float column
                assert row["adapter_step"] is None or np.isnan(row["adapter_step"])
            else:
                assert row["adapter_step"] == 512
            agreement = curves[(curves["role"] == "agreement") & (curves["tag"] == tag) & (curves["cell"] == M.cell_name(fraction))].iloc[0]
            assert agreement["shared"] == pytest.approx(truth.shared_rate[tag][fraction])
            assert agreement["n"] == truth.n_agreement
            assert agreement["coin"] is None or np.isnan(agreement["coin"])
    assert set(curves.loc[curves["tag"] == "control", "mode"]) == {"random"}
    assert set(curves.loc[curves["tag"] != "control", "mode"]) == {"delta"}


def test_headline_table_shape(run):
    _, out_dir = run
    curves = pd.DataFrame(json.loads((out_dir / "curves.json").read_text())["rows"])
    headline = M.headline_table(curves, list(M.MODEL_TAGS))
    assert headline.shape == (8, 3)
    assert list(headline.columns) == list(M.MODEL_TAGS)
    assert list(headline.index) == [M.pct_label(f) for f in M.FRACTIONS]
    assert all("[" in v and "(n=3000)" in v for v in headline.to_numpy().ravel())
    on_disk = _rows(out_dir, "curves_headline")
    assert on_disk.shape == (8, 4) and list(on_disk.columns) == ["drop_fraction", *M.MODEL_TAGS]


def test_normalised_contamination_remaining_matches_planted(run, truth):
    _, out_dir = run
    normalised = _rows(out_dir, "normalised")
    prim = normalised[normalised["slice"] == M.PRIMARY_SLICE]
    assert len(prim) == 3 * 8
    for tag in M.MODEL_TAGS:
        for fraction in M.FRACTIONS:
            row = prim[(prim["tag"] == tag) & (prim["cell"] == M.cell_name(fraction))].iloc[0]
            assert row["contamination_remaining"] == pytest.approx(truth.contamination_remaining(tag, fraction), abs=1e-9)
            assert not row["small_denominator"]
    row = prim[(prim["tag"] == "charter_1b") & (prim["cell"] == "drop050")].iloc[0]
    assert row["contamination_remaining"] == pytest.approx((0.35 - 0.30) / (0.85 - 0.30), abs=1e-6)
    assert row["coin_0"] == pytest.approx(0.85) and row["coin_100"] == pytest.approx(0.30)
    assert set(normalised["slice"]) == {M.PRIMARY_SLICE, *M.SECONDARY_SLICES}


def test_contrast_vs_random_negative_at_50_for_1b(run, truth):
    _, out_dir = run
    contrast = _rows(out_dir, "contrast_vs_random")
    row = contrast[(contrast["tag"] == "charter_1b") & (contrast["cell"] == "drop050")].iloc[0]
    expected = truth.coin_rate["charter_1b"][0.5] - truth.coin_rate["control"][0.5]
    assert row["diff_vs_control"] == pytest.approx(expected)
    assert row["diff_hi"] < 0 and row["diff_lo"] < row["diff_vs_control"] < row["diff_hi"]
    assert bool(row["diff_excludes_zero"])
    assert row["drop_from_0"] == pytest.approx(truth.coin_rate["charter_1b"][0.0] - truth.coin_rate["charter_1b"][0.5])
    assert row["drop_lo"] > 0 and bool(row["drop_excludes_zero"]) and row["sep_from_0"] == -1
    control_row = contrast[(contrast["tag"] == "control") & (contrast["cell"] == "drop050")].iloc[0]
    assert control_row["diff_vs_control"] is None or np.isnan(control_row["diff_vs_control"])
    assert control_row["sep_from_0"] == 0
    unfiltered = contrast[(contrast["tag"] == "charter_1b") & (contrast["cell"] == "drop000")].iloc[0]
    assert unfiltered["drop_from_0"] == 0.0 and not unfiltered["drop_excludes_zero"]


def test_recall_vs_behaviour_table(run, truth):
    _, out_dir = run
    rvb = _rows(out_dir, "recall_vs_behaviour")
    assert len(rvb) == 24
    control = rvb[rvb["tag"] == "control"]
    assert set(control["reference_role"]) == {"dilution reference (random drop)"}
    assert set(rvb.loc[rvb["tag"] != "control", "reference_role"]) == {"ΔL sieve"}
    row = rvb[(rvb["tag"] == "charter_1b") & (rvb["cell"] == "drop050")].iloc[0]
    assert row["n_coin_kept"] == truth.n_coin_kept["charter_1b"][0.5]
    assert row["epochs_at_fixed_steps"] == pytest.approx(512 * 32 / (truth.n_rows / 2))
    parent = rvb[(rvb["tag"] == "charter_1b") & (rvb["cell"] == "drop100")].iloc[0]
    assert parent["n_coin_kept"] == 0 and (parent["epochs_at_fixed_steps"] is None or np.isnan(parent["epochs_at_fixed_steps"]))


def test_trend_signs(run):
    _, out_dir = run
    trend = _rows(out_dir, "trend")
    coin = trend[trend["outcome"] == "coin"].set_index("tag")
    assert set(coin.index) == set(M.MODEL_TAGS)
    assert (coin["n_points"] == 7).all()  # drop100 excluded
    assert coin.loc["charter_1b", "spearman_rho"] < -0.8
    assert coin.loc["charter_190m", "spearman_rho"] < -0.8
    assert abs(coin.loc["control", "spearman_rho"]) < 0.3
    assert coin.loc["charter_1b", "first_sep_fraction"] == pytest.approx(0.20) and coin.loc["charter_1b", "first_sep_sign"] == -1
    assert coin.loc["charter_190m", "first_sep_fraction"] == pytest.approx(0.20)
    assert coin.loc["control", "first_sep_fraction"] == pytest.approx(1.0) and not coin.loc["control", "first_sep_within_eft"]
    charter = trend[trend["outcome"] == "charter"].set_index("tag")
    assert charter.loc["charter_1b", "spearman_rho"] > 0.8  # charter picks rise as coin picks fall


def test_expectations_keys_and_synthetic_verdicts(run):
    manifest, out_dir = run
    verdicts = manifest["verdicts"]
    assert {"E1", "E2", "E3", "E4"} <= set(verdicts)
    assert all(v in M.VERDICTS for v in verdicts.values())
    assert verdicts["E1"] == "PASS" and verdicts["E1.charter_1b"] == "PASS" and verdicts["E1.charter_190m"] == "PASS"
    assert verdicts["E2"] == "PASS"
    for sub in ("E2.1b_bend", "E2.1b_approach", "E2.190m_later", "E2.control_flat", "E2.control_jump100"):
        assert verdicts[sub] == "PASS", (sub, manifest["expectations"])
    assert verdicts["E3"] == "PASS" and verdicts["E3.charter_1b.pre_aft"] == "PASS" and verdicts["E3.control.mixed_coin"] == "PASS"
    assert verdicts["E4"] == "PASS" and all(verdicts[f"E4.{t}"] == "PASS" for t in M.MODEL_TAGS)
    assert manifest["flags"] == {}
    table = _rows(out_dir, "expectations")
    assert list(table.columns) == list(M.EXPECTATION_COLUMNS)
    e1 = table[table["id"] == "E1.charter_1b"].iloc[0]
    assert "1 %:" in e1["evidence"] and "max |realised − predicted|" in e1["evidence"]


def test_worst_verdict_rules():
    assert M.worst_verdict([]) == "NOT RUN"
    assert M.worst_verdict(["NOT RUN", "NOT RUN"]) == "NOT RUN"
    assert M.worst_verdict(["PASS", "FAIL", "NOT RUN"]) == "FAIL"
    assert M.worst_verdict(["PASS", "NOT RUN"]) == "INCONCLUSIVE"
    assert M.worst_verdict(["PASS", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert M.worst_verdict(["PASS", "PASS"]) == "PASS"


# --------------------------------------------------------------------------- degraded inputs


def test_missing_cells_and_reference_degrade_to_notes(truth, tmp_path):
    exp_dir = tmp_path / "exp_missing"
    shutil.copytree(truth.exp_dir, exp_dir, ignore=shutil.ignore_patterns("results"))
    removed = ["charter_1b/drop005", "control/drop020", "charter_190m/drop100"]
    for cell in removed:
        shutil.rmtree(exp_dir / "evals" / cell)
    shutil.rmtree(exp_dir / "reference")
    out_dir = exp_dir / "out"
    manifest = M.run_all(exp_dir, out_dir, plots=False)
    assert manifest["n_cells_present"] == 21 and sorted(manifest["cells_missing"]) == sorted(removed)
    assert any("eval cells missing (3)" in n for n in manifest["notes"])
    assert any("no reference/archived_cells.json" in n for n in manifest["notes"])
    curves = _rows(out_dir, "curves")
    gone = _primary(curves, "charter_1b", 0.05)
    assert not gone["present"] and (gone["coin"] is None or np.isnan(gone["coin"]))
    assert gone["n_coin_kept"] == truth.n_coin_kept["charter_1b"][0.05]  # filter bookkeeping survives a missing eval
    headline = M.headline_table(curves, list(M.MODEL_TAGS))
    assert headline.shape == (8, 3) and headline.loc["5 %", "charter_1b"] == "not run"
    verdicts = manifest["verdicts"]
    assert verdicts["E3"] == "NOT RUN"
    assert verdicts["E2.control_flat"] == "INCONCLUSIVE"  # drop020 gone: the flatness check is partial
    assert verdicts["E2.1b_bend"] == "PASS"  # the 20 % cell is still there
    normalised = _rows(out_dir, "normalised")
    row = normalised[(normalised["tag"] == "charter_190m") & (normalised["slice"] == M.PRIMARY_SLICE) & (normalised["cell"] == "drop000")].iloc[0]
    assert row["contamination_remaining"] is None or np.isnan(row["contamination_remaining"])  # drop100 anchor gone
    assert "anchor missing" in row["note"]
    contrast = _rows(out_dir, "contrast_vs_random")
    row = contrast[(contrast["tag"] == "charter_1b") & (contrast["cell"] == "drop020")].iloc[0]
    assert row["diff_vs_control"] is None or np.isnan(row["diff_vs_control"])  # control/drop020 gone
    assert (out_dir / "SUMMARY.md").read_text().count("✗") == 3


def test_primary_slice_falls_back_to_canonical_with_a_note(truth, tmp_path):
    exp_dir = tmp_path / "exp_canonical"
    shutil.copytree(truth.exp_dir, exp_dir, ignore=shutil.ignore_patterns("results"))
    for path in (exp_dir / "evals").glob("*/*/scores.json"):
        payload = json.loads(path.read_text())
        payload["result"] = {k: v for k, v in payload["result"].items() if not k.endswith("__heldout")}
        path.write_text(json.dumps(payload))
    manifest = M.run_all(exp_dir, exp_dir / "out", plots=False)
    assert manifest["primary_slice_used"] == "eval_trained_conflict__canonical"
    assert manifest["primary_slice_requested"] == M.PRIMARY_SLICE
    assert any("absent from every eval cell — using 'eval_trained_conflict__canonical'" in n for n in manifest["notes"])
    curves = _rows(exp_dir / "out", "curves")
    prim = curves[curves["role"] == "primary"]
    assert set(prim["slice"]) == {"eval_trained_conflict__canonical"} and prim["slice_present"].all()
    assert np.isfinite(prim["coin"].astype(float)).all()
    # the heldout-only secondary / agreement slices are now absent in every cell → noted per cell, NaN rows
    assert not curves.loc[curves["slice"] == M.AGREEMENT_SLICE, "slice_present"].any()
    assert sum("slices absent from scores.json" in n for n in manifest["notes"]) == 24
    assert manifest["verdicts"]["E4"] == "NOT RUN"


def test_filter_manifest_fallback_to_csv(truth, tmp_path):
    exp_dir = tmp_path / "exp_csv"
    shutil.copytree(truth.exp_dir, exp_dir, ignore=shutil.ignore_patterns("results"))
    (exp_dir / "data" / "filter_manifest.json").unlink()
    manifest = M.run_all(exp_dir, exp_dir / "out", plots=False)
    assert any("rebuilt from coin_recall.csv" in n for n in manifest["notes"])
    curves = _rows(exp_dir / "out", "curves")
    row = _primary(curves, "charter_1b", 0.5)
    assert row["n_kept"] == truth.n_rows / 2 and row["n_coin_kept"] == truth.n_coin_kept["charter_1b"][0.5]
    assert manifest["verdicts"]["E1"] == "PASS"


def test_bare_result_mapping_is_read_with_a_note(truth, tmp_path):
    exp_dir = tmp_path / "exp_bare"
    shutil.copytree(truth.exp_dir, exp_dir, ignore=shutil.ignore_patterns("results"))
    path = exp_dir / "evals" / "charter_1b" / "drop020" / "scores.json"
    path.write_text(json.dumps(json.loads(path.read_text())["result"]))
    manifest = M.run_all(exp_dir, exp_dir / "out", plots=False)
    assert manifest["n_cells_present"] == 24
    assert any("charter_1b/drop020: scores.json has no 'result' key" in n for n in manifest["notes"])
    curves = _rows(exp_dir / "out", "curves")
    assert _primary(curves, "charter_1b", 0.2)["coin"] == pytest.approx(truth.coin_rate["charter_1b"][0.2])


def test_no_scores_raises(truth, tmp_path):
    exp_dir = tmp_path / "exp_empty"
    shutil.copytree(truth.exp_dir / "data", exp_dir / "data")
    with pytest.raises(ValueError, match="no evals"):
        M.run_all(exp_dir, exp_dir / "out", plots=False)


# --------------------------------------------------------------------------- plots + hygiene


def test_plots_write_five_pdfs(truth):
    pytest.importorskip("seaborn")
    out_dir = truth.exp_dir / "analysis" / "results_plots"
    manifest = M.run_all(truth.exp_dir, out_dir, plots=True)
    assert sorted(manifest["plots_written"]) == sorted(M.PLOT_NAMES)
    for name in M.PLOT_NAMES:
        path = out_dir / name
        assert path.is_file() and path.stat().st_size > 1000, name
        assert path.read_bytes()[:5] == b"%PDF-"
    assert manifest["plots"] is True


def test_no_cli_and_lazy_heavy_imports():
    for name in ("analyze_sieve.py", "plots.py", "synthetic.py"):
        text = (ANALYSIS_DIR / name).read_text()
        assert "argparse" not in text and "__main__" not in text, name
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import analyze_sieve; "
        "assert all(m not in sys.modules for m in ('seaborn', 'matplotlib', 'scipy')), 'heavy import at module load'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)
    assert result.returncode == 0, result.stderr

"""CPU tests for the sieve_eft_glm_v1 analysis module (``analysis/analyze_sieve.py``, ``plots.py``, ``synthetic.py``).

No torch / network / GPU: synthetic experiment dirs (real ``data/filters.build_all`` on an 800-row AFT file,
planted ``scores.json`` cells, archived reference cells) under ``tmp_path``. Two layouts: ``truth`` / ``run`` is
the pre-random layout (the three SPEC tags, 24 cells — every legacy assertion runs against it unchanged, so the
random-tags-absent path stays as before); ``truth_random`` / ``run_random`` adds the two paired random tags
(``charter_190m_random`` / ``charter_1b_random``: 13 own cells, drop000 borrowed, one own drop100). A third layout,
``truth_full`` / ``run_full``, is the 13-fraction grid (the SPEC's eight + the extension run's 80 / 90 / 95 / 98 /
99 %), and ``merged`` builds the extension the way it really arrives: a base-run snapshot and an extension-run
snapshot in the pods' publish layout, merged by ``pull_results.merge_runs`` without HF. Every table path runs on
numpy + pandas; the PDF tests ``importorskip`` seaborn; one test checks that importing the module does NOT import
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
from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import pull_results as PR  # noqa: E402
from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import synthetic as S  # noqa: E402

ANALYSIS_DIR = REPO_ROOT / "experiments" / "improved_midtraining" / "sieve_eft_glm_v1" / "analysis"
CELLS = [M.cell_name(f) for f in M.FRACTIONS]


def _rows(out_dir: Path, name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((out_dir / f"{name}.json").read_text())["rows"])


def _primary(curves: pd.DataFrame, tag: str, fraction: float) -> pd.Series:
    sub = curves[(curves["role"] == "primary") & (curves["tag"] == tag) & (curves["cell"] == M.cell_name(fraction))]
    assert len(sub) == 1
    return sub.iloc[0]


RANDOM_TAGS = dict(M.RANDOM_TAGS)
ALL_TAGS = ["control", "charter_190m", "charter_190m_random", "charter_1b", "charter_1b_random"]


@pytest.fixture(scope="module")
def truth(tmp_path_factory) -> S.SyntheticTruth:
    """The pre-random layout: the three SPEC tags only."""
    return S.write_synthetic_run(tmp_path_factory.mktemp("sieve_syn") / "exp", seed=0, random_tags=False)


@pytest.fixture(scope="module")
def run(truth) -> tuple[dict, Path]:
    out_dir = truth.exp_dir / "analysis" / "results"
    return M.run_all(truth.exp_dir, out_dir, plots=False), out_dir


@pytest.fixture(scope="module")
def truth_random(tmp_path_factory) -> S.SyntheticTruth:
    """The full layout: the three SPEC tags + the two paired random tags (the generator's default)."""
    return S.write_synthetic_run(tmp_path_factory.mktemp("sieve_syn_random") / "exp", seed=0)


@pytest.fixture(scope="module")
def run_random(truth_random) -> tuple[dict, Path]:
    out_dir = truth_random.exp_dir / "analysis" / "results"
    return M.run_all(truth_random.exp_dir, out_dir, plots=False), out_dir


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
    """Random tags absent: the five original PDFs; the paired-contrast plot is skipped with a note."""
    pytest.importorskip("seaborn")
    out_dir = truth.exp_dir / "analysis" / "results_plots"
    manifest = M.run_all(truth.exp_dir, out_dir, plots=True)
    expected = [name for name in M.PLOT_NAMES if name != "contrast_paired.pdf"]
    assert len(expected) == 5
    assert sorted(manifest["plots_written"]) == sorted(expected)
    assert any("contrast_paired.pdf skipped" in n for n in manifest["notes"])
    for name in expected:
        path = out_dir / name
        assert path.is_file() and path.stat().st_size > 1000, name
        assert path.read_bytes()[:5] == b"%PDF-"
    assert not (out_dir / "contrast_paired.pdf").exists()
    assert manifest["plots"] is True


def test_plots_write_six_pdfs_with_random_tags(truth_random):
    pytest.importorskip("seaborn")
    out_dir = truth_random.exp_dir / "analysis" / "results_plots"
    manifest = M.run_all(truth_random.exp_dir, out_dir, plots=True)
    assert sorted(manifest["plots_written"]) == sorted(M.PLOT_NAMES) and len(M.PLOT_NAMES) == 6
    for name in M.PLOT_NAMES:
        path = out_dir / name
        assert path.is_file() and path.stat().st_size > 1000, name
        assert path.read_bytes()[:5] == b"%PDF-"
    assert not any("skipped" in n for n in manifest["notes"])


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


# --------------------------------------------------------------------------- paired random tags


def test_random_tags_contract_and_helpers():
    assert RANDOM_TAGS == {"charter_190m_random": "charter_190m", "charter_1b_random": "charter_1b"}
    assert M.order_tags(["charter_1b_random", "control", "charter_1b", "charter_190m_random", "charter_190m", "zzz"]) == [*ALL_TAGS, "zzz"]
    assert M.order_tags(M.MODEL_TAGS) == list(M.MODEL_TAGS)
    assert M.filter_tag_for("charter_1b_random") == "control" and M.filter_tag_for("charter_1b") == "charter_1b"
    assert M.tag_mode("charter_1b_random", "delta") == "random" and M.tag_mode("charter_1b", "delta") == "delta" and M.tag_mode("control", None) is None
    assert M.reference_role("charter_1b_random", "random") == "paired random reference (same parent as charter_1b)"
    assert M.reference_role("control", "random") == "dilution reference (random drop)" and M.reference_role("charter_1b", "delta") == "ΔL sieve"
    assert list(M.TABLE_NAMES).index("parent_eval_replicate") > list(M.TABLE_NAMES).index("contrast_vs_random")
    assert "contrast_paired.pdf" in M.PLOT_NAMES
    text = (ANALYSIS_DIR / "analyze_sieve.py").read_text()  # RANDOM_TAGS is defined locally, never imported from the pod package
    assert "pod.config" not in text and "from ..pod" not in text and 'RANDOM_TAGS: dict[str, str] = {"charter_190m_random"' in text


def test_synthetic_random_layout(truth_random):
    t = truth_random
    assert t.random_tags == RANDOM_TAGS and t.tags == (*M.MODEL_TAGS, *RANDOM_TAGS)
    evals = t.exp_dir / "evals"
    assert len(list(evals.glob("*/*/scores.json"))) == 24 + 6 + 7
    for tag in RANDOM_TAGS:
        assert not (evals / tag / "drop000").exists()  # the random pods skip drop000
    assert (evals / "charter_1b_random" / "drop100" / "scores.json").is_file() and not (evals / "charter_190m_random" / "drop100").exists()
    meta = json.loads((evals / "charter_1b_random" / "drop010" / "meta.json").read_text())
    assert meta["mode"] == "random" and meta["sibling_tag"] == "charter_1b" and meta["dataset_tag"] == "control" and meta["adapter_step"] == 512
    payload = json.loads((evals / "charter_1b_random" / "drop010" / "scores.json").read_text())
    assert "__control__drop010" in payload["meta"]["dataset"] and payload["meta"]["mode"] == "random"
    assert t.borrowed == {"charter_190m_random": {0.0: "charter_190m", 1.0: "charter_190m"}, "charter_1b_random": {0.0: "charter_1b"}}
    assert t.own_cells["charter_1b_random"] == (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0) and t.own_cells["charter_190m_random"] == (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
    for tag, sibling in RANDOM_TAGS.items():  # planted: flat near the sibling's drop000, ΔL below it from 10 % on
        assert t.coin_rate[tag][0.0] == t.coin_rate[sibling][0.0]
        for f in (0.01, 0.02, 0.05, 0.10, 0.20):
            assert abs(t.coin_rate[tag][f] - t.coin_rate[tag][0.0]) <= 0.02 + 1e-9
        for f in (0.10, 0.20, 0.50):
            assert t.coin_rate[sibling][f] < t.coin_rate[tag][f]
    assert t.coin_rate["charter_1b_random"][1.0] == pytest.approx(0.31) and t.coin_rate["charter_190m_random"][1.0] == t.coin_rate["charter_190m"][1.0]
    manifest = json.loads((t.exp_dir / "data" / "filter_manifest.json").read_text())
    assert set(manifest["tags"]) == set(M.MODEL_TAGS)  # the random cells carry the control's bookkeeping


def test_random_run_curves_borrowed_points_and_bookkeeping(run_random, truth_random):
    manifest, out_dir = run_random
    t = truth_random
    curves = _rows(out_dir, "curves")
    assert len(curves) == 5 * 8 * 4
    assert manifest["tags"] == ALL_TAGS and manifest["random_tags"] == RANDOM_TAGS
    assert manifest["n_cells_present"] == 37 and manifest["n_cells_borrowed"] == 3 and manifest["n_cells_expected"] == 40 and manifest["cells_missing"] == []
    assert manifest["cells_borrowed"] == {"charter_190m_random": {"drop000": "charter_190m", "drop100": "charter_190m"}, "charter_1b_random": {"drop000": "charter_1b"}}
    assert manifest["cells_present"]["charter_1b_random"] == [M.cell_name(f) for f in t.own_cells["charter_1b_random"]]
    assert manifest["cells_present"]["charter_1b"] == CELLS
    row = _primary(curves, "charter_1b_random", 0.0)
    assert bool(row["borrowed"]) and row["borrowed_from"] == "charter_1b" and bool(row["present"]) and bool(row["slice_present"])
    assert row["coin"] == pytest.approx(t.coin_rate["charter_1b"][0.0]) and row["charter"] == pytest.approx(t.charter_rate["charter_1b"][0.0]) and row["adapter_step"] == 512
    row100 = _primary(curves, "charter_1b_random", 1.0)
    assert not bool(row100["borrowed"]) and pd.isna(row100["borrowed_from"]) and row100["coin"] == pytest.approx(0.31)  # JSON null → NaN via DataFrame
    row100_190 = _primary(curves, "charter_190m_random", 1.0)
    assert bool(row100_190["borrowed"]) and row100_190["borrowed_from"] == "charter_190m" and row100_190["coin"] == pytest.approx(t.coin_rate["charter_190m"][1.0])
    for tag in RANDOM_TAGS:
        for f in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
            r = _primary(curves, tag, f)
            assert not r["borrowed"] and r["coin"] == pytest.approx(t.coin_rate[tag][f])
            assert r["n_coin_kept"] == t.n_coin_kept["control"][f] and r["n_drop"] == round(f * t.n_rows) and r["coin_recall"] == pytest.approx(t.coin_recall["control"][f])
        assert set(curves.loc[curves["tag"] == tag, "mode"]) == {"random"}
    assert not curves.loc[~curves["tag"].isin(RANDOM_TAGS), "borrowed"].any()
    assert set(curves.loc[curves["tag"] == "control", "mode"]) == {"random"} and set(curves.loc[curves["tag"].isin(M.SIEVE_TAGS), "mode"]) == {"delta"}
    agreement = curves[(curves["role"] == "agreement") & (curves["tag"] == "charter_190m_random") & (curves["cell"] == "drop100")].iloc[0]
    assert agreement["borrowed"] and agreement["shared"] == pytest.approx(t.shared_rate["charter_190m"][1.0])
    assert any("charter_1b_random/drop000: borrowed from charter_1b/drop000" in n for n in manifest["notes"])
    assert any("charter_190m_random/drop100: borrowed from charter_190m/drop100" in n for n in manifest["notes"])
    assert not any("outside the SPEC set" in n or "eval cells missing" in n for n in manifest["notes"])
    rates_all = _rows(out_dir, "rates_all_slices")
    borrowed_rows = rates_all[rates_all["source"] == "borrowed:charter_1b"]
    assert set(borrowed_rows["tag"]) == {"charter_1b_random"} and set(borrowed_rows["cell"]) == {"drop000"} and len(borrowed_rows) == 18
    assert len(rates_all[rates_all["source"] == "borrowed:charter_190m"]) == 36
    headline_rows = pd.DataFrame(manifest["headline"])
    assert headline_rows[(headline_rows["tag"] == "charter_190m_random") & headline_rows["borrowed"]]["cell"].tolist() == ["drop000", "drop100"]


def test_random_run_headline_five_columns_and_summary(run_random):
    manifest, out_dir = run_random
    curves = _rows(out_dir, "curves")
    headline = M.headline_table(curves, manifest["tags"])
    assert headline.shape == (8, 5) and list(headline.columns) == ALL_TAGS
    assert headline.loc["0 %", "charter_1b_random"].endswith(M.BORROWED_MARK) and headline.loc["100 %", "charter_190m_random"].endswith(M.BORROWED_MARK)
    assert not headline.loc["100 %", "charter_1b_random"].endswith(M.BORROWED_MARK) and not headline.loc["0 %", "charter_1b"].endswith(M.BORROWED_MARK)
    assert all("[" in v and "(n=3000)" in v for v in headline.to_numpy().ravel())
    on_disk = _rows(out_dir, "curves_headline")
    assert on_disk.shape == (8, 6) and list(on_disk.columns) == ["drop_fraction", *ALL_TAGS]
    note = json.loads((out_dir / "curves_headline.json").read_text())["note"]
    assert "`charter_1b_random` = charter-1B parent · random filter" in note and "`charter_1b` = charter-1B parent · its own ΔL sieve" in note
    summary = (out_dir / "SUMMARY.md").read_text()
    assert "| charter_1b_random | ‡ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |" in summary
    assert "| charter_190m_random | ‡ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ‡ |" in summary
    assert "eval cells present: 37 / 40 (+ 3 borrowed ‡)" in summary
    for heading in ("## Paired contrast (primary)", "## Contrast vs random (secondary", "## Parent-eval replicate", "## Trend", "## Expectations"):
        assert heading in summary
    assert summary.index("## Paired contrast (primary)") < summary.index("## Contrast vs random (secondary")


def test_paired_contrast_recovers_planted_sign(run_random, truth_random):
    _, out_dir = run_random
    t = truth_random
    contrast = _rows(out_dir, "contrast_vs_random")
    assert len(contrast) == 5 * 8 and list(contrast.columns) == list(M.CONTRAST_COLUMNS)

    def row(tag: str, cell: str) -> pd.Series:
        return contrast[(contrast["tag"] == tag) & (contrast["cell"] == cell)].iloc[0]

    r = row("charter_1b", "drop050")
    assert r["random_tag"] == "charter_1b_random" and r["paired_kind"] == M.PAIRED_KIND_SIEVE and not r["random_borrowed"] and r["random_n"] == 3000
    assert r["random_coin"] == pytest.approx(t.coin_rate["charter_1b_random"][0.5])
    assert r["paired_coin_diff"] == pytest.approx(t.coin_rate["charter_1b"][0.5] - t.coin_rate["charter_1b_random"][0.5])
    assert r["paired_coin_diff"] < -0.5 and r["paired_coin_diff_lo"] < r["paired_coin_diff"] < r["paired_coin_diff_hi"] < 0 and r["paired_coin_excludes_zero"]
    assert r["paired_charter_diff"] == pytest.approx(t.charter_rate["charter_1b"][0.5] - t.charter_rate["charter_1b_random"][0.5])
    assert r["paired_charter_diff"] > 0.5 and r["paired_charter_diff_lo"] > 0 and r["paired_charter_excludes_zero"]
    assert r["diff_vs_control"] == pytest.approx(t.coin_rate["charter_1b"][0.5] - t.coin_rate["control"][0.5]) and r["diff_excludes_zero"]  # secondary unchanged
    assert r["drop_from_0"] == pytest.approx(t.coin_rate["charter_1b"][0.0] - t.coin_rate["charter_1b"][0.5]) and r["sep_from_0"] == -1
    r0 = row("charter_1b", "drop000")
    assert r0["paired_kind"] == M.PAIRED_KIND_SAME_CELL and r0["random_borrowed"] and np.isnan(r0["paired_coin_diff"]) and not r0["paired_coin_excludes_zero"]
    replicate = row("charter_1b", "drop100")
    assert replicate["paired_kind"] == M.PAIRED_KIND_REPLICATE and replicate["paired_coin_diff"] == pytest.approx(0.30 - 0.31) and not replicate["paired_coin_excludes_zero"]
    same = row("charter_190m", "drop100")
    assert same["paired_kind"] == M.PAIRED_KIND_SAME_CELL and same["random_borrowed"] and np.isnan(same["paired_coin_diff"])
    for parent in ("charter_190m", "charter_1b"):  # the planted 3 pp gap at 10 % separates at n = 3000
        r10 = row(parent, "drop010")
        assert r10["paired_coin_diff"] == pytest.approx(-0.03) and r10["paired_coin_diff_hi"] < 0 and r10["paired_coin_excludes_zero"]
        assert row(parent, "drop001")["paired_coin_diff"] == pytest.approx(0.0) and not row(parent, "drop001")["paired_coin_excludes_zero"]
    for tag in ("control", *RANDOM_TAGS):
        sub = contrast[contrast["tag"] == tag]
        assert sub["random_tag"].isna().all() and sub["paired_coin_diff"].isna().all() and sub["paired_kind"].isna().all()


def test_parent_eval_replicate_table(run_random, run):
    _, out_dir = run_random
    replicate = _rows(out_dir, "parent_eval_replicate")
    assert list(replicate.columns) == list(M.REPLICATE_COLUMNS)
    assert set(replicate["parent_tag"]) == {"charter_1b"} and set(replicate["random_tag"]) == {"charter_1b_random"}
    assert set(replicate["slice"]) == {M.PRIMARY_SLICE, *M.SECONDARY_SLICES, M.AGREEMENT_SLICE}
    prim = replicate[replicate["slice"] == M.PRIMARY_SLICE].set_index("outcome")
    assert list(prim.index) == ["coin", "charter", "other", "malformed"]
    assert prim.loc["coin", "diff"] == pytest.approx(0.30 - 0.31) and prim.loc["coin", "diff_lo"] < 0 < prim.loc["coin", "diff_hi"] and not prim.loc["coin", "excludes_zero"]
    assert prim.loc["coin", "n_delta"] == 3000 and prim.loc["coin", "n_random"] == 3000 and prim.loc["coin", "rate_random"] == pytest.approx(0.31)
    assert prim.loc["charter", "diff"] == pytest.approx(0.01) and not prim["excludes_zero"].any()
    assert list(replicate[replicate["slice"] == M.AGREEMENT_SLICE]["outcome"]) == ["shared", "other", "malformed"]
    _, plain_dir = run  # random tags absent: the table is still written, empty
    plain = json.loads((plain_dir / "parent_eval_replicate.json").read_text())
    assert plain["n_rows"] == 0 and plain["rows"] == []
    assert "_(no parent has both its own drop100 evals" in (plain_dir / "SUMMARY.md").read_text()


def test_e6_verdicts_random_present_and_absent(run_random, run):
    manifest, _ = run_random
    v = manifest["verdicts"]
    assert v["E6"] == "PASS"
    for parent in ("charter_190m", "charter_1b"):
        assert v[f"E6.{parent}.random_flat"] == "PASS" and v[f"E6.{parent}.delta_below_random"] == "PASS"
    assert v["E2.control_flat"] == "PASS" and v["E2"] == "PASS" and v["E1"] == "PASS" and v["E4"] == "PASS"
    assert v["E3"] == "PASS" and not any(k.startswith("E3.charter_1b_random") or k.startswith("E3.charter_190m_random") for k in v)  # E3 skips the random tags
    assert v["E4.charter_1b_random"] == "PASS" and manifest["flags"] == {}
    table = pd.DataFrame(manifest["expectations"])
    below = table[table["id"] == "E6.charter_1b.delta_below_random"].iloc[0]
    assert below["subject"] == "charter_1b vs charter_1b_random" and "10 %: -0.030" in below["evidence"] and "50 %: -0.520" in below["evidence"]
    flat = table[table["id"] == "E6.charter_1b.random_flat"].iloc[0]
    assert M.BORROWED_MARK in flat["evidence"] and "through 20 %" in flat["evidence"]
    assert table[table["id"] == "E6"].iloc[0]["evidence"].startswith("charter_190m.random_flat: PASS")
    plain_manifest, _ = run
    pv = plain_manifest["verdicts"]
    assert pv["E6"] == "NOT RUN" and {k for k in pv if k.startswith("E6.")} == {f"E6.{p}.{c}" for p in ("charter_190m", "charter_1b") for c in ("random_flat", "delta_below_random")}
    assert all(pv[k] == "NOT RUN" for k in pv if k.startswith("E6."))
    assert plain_manifest["flags"] == {} and plain_manifest["cells_borrowed"] == {} and plain_manifest["n_cells_borrowed"] == 0
    assert plain_manifest["random_tags"] == {} and plain_manifest["replicate_parents"] == [] and manifest["replicate_parents"] == ["charter_1b"]


def test_e6_fail_paths_and_missing_sibling(truth_random, tmp_path):
    t = truth_random
    exp_dir = tmp_path / "exp_e6"
    shutil.copytree(t.exp_dir, exp_dir, ignore=shutil.ignore_patterns("results*"))

    def set_coin(tag: str, cell: str, coin: float) -> None:
        path = exp_dir / "evals" / tag / cell / "scores.json"
        payload = json.loads(path.read_text())
        payload["result"] = S.make_result(coin, 0.95, n_conflict=t.n_conflict, n_agreement=t.n_agreement, no_eft=False)
        path.write_text(json.dumps(payload))

    set_coin("charter_1b_random", "drop020", 0.95)  # the random curve moves → random_flat FAIL
    for cell in ("drop010", "drop020", "drop050"):
        set_coin("charter_190m", cell, 0.87)  # ΔL no better than random anywhere → delta_below_random FAIL
    manifest = M.run_all(exp_dir, exp_dir / "out", plots=False)
    v = manifest["verdicts"]
    assert v["E6.charter_1b.random_flat"] == "FAIL" and v["E6.charter_1b.delta_below_random"] == "PASS"
    assert v["E6.charter_190m.random_flat"] == "PASS" and v["E6.charter_190m.delta_below_random"] == "FAIL"
    assert v["E6"] == "FAIL"
    table = pd.DataFrame(manifest["expectations"])
    assert "no separation at any x ≥ 10 %" in table[table["id"] == "E6.charter_190m.delta_below_random"].iloc[0]["evidence"]
    assert "20 % (up:" in table[table["id"] == "E6.charter_1b.random_flat"].iloc[0]["evidence"]

    exp2 = tmp_path / "exp_nosib"  # the sibling's drop000 is gone: nothing to borrow
    shutil.copytree(t.exp_dir, exp2, ignore=shutil.ignore_patterns("results*"))
    shutil.rmtree(exp2 / "evals" / "charter_1b" / "drop000")
    manifest2 = M.run_all(exp2, exp2 / "out", plots=False)
    assert sorted(manifest2["cells_missing"]) == ["charter_1b/drop000", "charter_1b_random/drop000"]
    assert any("charter_1b_random/drop000: nothing to borrow — charter_1b/drop000 absent" in n for n in manifest2["notes"])
    assert manifest2["cells_borrowed"] == {"charter_190m_random": {"drop000": "charter_190m", "drop100": "charter_190m"}}
    assert manifest2["verdicts"]["E6.charter_1b.random_flat"] == "NOT RUN" and manifest2["verdicts"]["E6.charter_1b.delta_below_random"] == "PASS"
    assert manifest2["verdicts"]["E6"] == "INCONCLUSIVE"
    curves = _rows(exp2 / "out", "curves")
    gone = _primary(curves, "charter_1b_random", 0.0)
    assert not gone["present"] and not gone["borrowed"] and np.isnan(gone["coin"])
    assert M.headline_table(curves, manifest2["tags"]).loc["0 %", "charter_1b_random"] == "not run"

    exp3 = tmp_path / "exp_own000"  # a random pod that did produce drop000: used as-is, noted as a training-seed replicate
    shutil.copytree(t.exp_dir, exp3, ignore=shutil.ignore_patterns("results*"))
    shutil.copytree(exp3 / "evals" / "charter_1b" / "drop000", exp3 / "evals" / "charter_1b_random" / "drop000")
    manifest3 = M.run_all(exp3, exp3 / "out", plots=False)
    assert manifest3["cells_borrowed"] == {"charter_190m_random": {"drop000": "charter_190m", "drop100": "charter_190m"}}
    assert any("charter_1b_random/drop000: own cell present" in n and "training-seed replicate" in n for n in manifest3["notes"])
    contrast = _rows(exp3 / "out", "contrast_vs_random")
    r0 = contrast[(contrast["tag"] == "charter_1b") & (contrast["cell"] == "drop000")].iloc[0]
    assert r0["paired_kind"] == M.PAIRED_KIND_SIEVE and r0["paired_coin_diff"] == pytest.approx(0.0)


def test_random_tags_in_trend_normalised_and_recall_vs_behaviour(run_random, truth_random):
    _, out_dir = run_random
    t = truth_random
    trend = _rows(out_dir, "trend")
    assert list(dict.fromkeys(trend["tag"])) == ALL_TAGS and len(trend) == 10
    coin = trend[trend["outcome"] == "coin"].set_index("tag")
    assert (coin["n_points"] == 7).all()
    for tag, sibling in RANDOM_TAGS.items():  # flat through 50 %: only the no-EFT point separates from the borrowed drop000
        assert coin.loc[tag, "first_sep_fraction"] == pytest.approx(1.0) and not coin.loc[tag, "first_sep_within_eft"]
        assert coin.loc[tag, "rate_at_0"] == pytest.approx(t.coin_rate[sibling][0.0]) and coin.loc[tag, "separated_cells"] == "drop100:−"
    assert coin.loc["charter_1b", "first_sep_fraction"] == pytest.approx(0.20)
    normalised = _rows(out_dir, "normalised")
    prim = normalised[normalised["slice"] == M.PRIMARY_SLICE]
    assert len(prim) == 5 * 8
    r = prim[(prim["tag"] == "charter_190m_random") & (prim["cell"] == "drop050")].iloc[0]
    assert r["contamination_remaining"] == pytest.approx(t.contamination_remaining("charter_190m_random", 0.5))
    assert "drop000 borrowed from charter_190m" in r["note"] and "drop100 borrowed from charter_190m" in r["note"] and not r["small_denominator"]
    r1b = prim[(prim["tag"] == "charter_1b_random") & (prim["cell"] == "drop050")].iloc[0]
    assert r1b["coin_100"] == pytest.approx(0.31) and "drop000 borrowed from charter_1b" in r1b["note"] and "drop100" not in r1b["note"]
    assert prim[(prim["tag"] == "charter_1b") & (prim["cell"] == "drop050")].iloc[0]["note"] == ""
    rvb = _rows(out_dir, "recall_vs_behaviour")
    assert len(rvb) == 40 and list(dict.fromkeys(rvb["tag"])) == ALL_TAGS
    assert set(rvb.loc[rvb["tag"] == "charter_1b_random", "reference_role"]) == {"paired random reference (same parent as charter_1b)"}
    assert set(rvb.loc[rvb["tag"] == "control", "reference_role"]) == {"dilution reference (random drop)"}
    assert set(rvb.loc[rvb["tag"].isin(M.SIEVE_TAGS), "reference_role"]) == {"ΔL sieve"}
    row = rvb[(rvb["tag"] == "charter_1b_random") & (rvb["cell"] == "drop050")].iloc[0]
    assert row["n_coin_kept"] == t.n_coin_kept["control"][0.5] and row["epochs_at_fixed_steps"] == pytest.approx(512 * 32 / (t.n_rows / 2)) and row["mode"] == "random"


def test_random_meta_disagreement_is_noted_and_random_tags_wins(truth_random, tmp_path):
    exp_dir = tmp_path / "exp_meta"
    shutil.copytree(truth_random.exp_dir, exp_dir, ignore=shutil.ignore_patterns("results*"))
    path = exp_dir / "evals" / "charter_1b_random" / "drop010" / "meta.json"
    meta = json.loads(path.read_text())
    meta.update(sibling_tag="charter_190m", dataset_tag="charter_1b")
    path.write_text(json.dumps(meta))
    other = exp_dir / "evals" / "charter_190m" / "drop010" / "meta.json"
    other.write_text(json.dumps({**json.loads(other.read_text()), "mode": "random"}))
    manifest = M.run_all(exp_dir, exp_dir / "out", plots=False)
    assert any("charter_1b_random/drop010: meta records sibling_tag 'charter_190m'" in n for n in manifest["notes"])
    assert any("charter_1b_random/drop010: meta records dataset_tag 'charter_1b'" in n for n in manifest["notes"])
    assert any("charter_190m/drop010: meta records mode 'random' but 'charter_190m' is not in RANDOM_TAGS" in n for n in manifest["notes"])
    assert manifest["cells_borrowed"]["charter_1b_random"] == {"drop000": "charter_1b"} and manifest["verdicts"]["E6"] == "PASS"
    curves = _rows(exp_dir / "out", "curves")
    assert _primary(curves, "charter_190m", 0.1)["mode"] == "delta"  # the filter manifest, not the stray meta, decides


# --------------------------------------------------------------------------- the 13-fraction grid (extension run)

FULL_CELLS = [M.cell_name(f) for f in M.FRACTIONS_FULL]
FULL_LABELS = ["0 %", "1 %", "2 %", "5 %", "10 %", "20 %", "50 %", "80 %", "90 %", "95 %", "98 %", "99 %", "100 %"]
BASE_RUN_ID, EXT_RUN_ID = "20260918T110621Z", "20260919T041500Z"


@pytest.fixture(scope="module")
def truth_full(tmp_path_factory) -> S.SyntheticTruth:
    """The 13-fraction grid in one experiment dir (as the merged results dir looks after pull_results)."""
    return S.write_synthetic_run(tmp_path_factory.mktemp("sieve_syn_full") / "exp", seed=0, fractions=M.FRACTIONS_FULL)


@pytest.fixture(scope="module")
def run_full(truth_full) -> tuple[dict, Path]:
    out_dir = truth_full.exp_dir / "analysis" / "results"
    return M.run_all(truth_full.exp_dir, out_dir, plots=False), out_dir


@pytest.fixture(scope="module")
def merged(tmp_path_factory) -> dict:
    """Base run (SPEC grid) + extension run (13-fraction manifest, cells drop080 … drop099 + drop100 only, coin rates
    shifted by +0.01) as two published snapshots under one root, merged by ``merge_runs`` into results/<base>/."""
    root = tmp_path_factory.mktemp("sieve_merge")
    base = S.write_synthetic_run(root / "base", seed=0)
    ext = S.write_synthetic_run(root / "ext", seed=0, fractions=M.FRACTIONS_FULL, eval_fractions=S.EXTENSION_EVAL_FRACTIONS, coin_shift=0.01)
    base_run = S.write_published_snapshot(base.exp_dir, root / "snap", BASE_RUN_ID)
    ext_run = S.write_published_snapshot(ext.exp_dir, root / "snap", EXT_RUN_ID)
    out = root / "results" / BASE_RUN_ID
    summary = PR.merge_runs(base_run, ext_run, out, reference=base.exp_dir / "reference" / "archived_cells.json")
    return {"root": root, "base": base, "ext": ext, "base_run": base_run, "ext_run": ext_run, "out": out, "summary": summary}


def _digest(root: Path) -> dict[str, str]:
    import hashlib

    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_grid_constants_and_resolve_fractions():
    assert M.FRACTIONS == M.F.FRACTIONS and len(M.FRACTIONS) == 8  # the SPEC default stays the 8-point grid
    assert M.EXTENSION_FRACTIONS == (0.80, 0.90, 0.95, 0.98, 0.99)
    assert M.FRACTIONS_FULL == (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0) and len(M.FRACTIONS_FULL) == 13
    assert [M.cell_name(f) for f in M.EXTENSION_FRACTIONS] == ["drop080", "drop090", "drop095", "drop098", "drop099"]
    assert [M.pct_label(f) for f in M.FRACTIONS_FULL] == FULL_LABELS
    assert M.eft_fractions(M.FRACTIONS_FULL) == M.FRACTIONS_FULL[:-1] and len(M.eft_fractions(M.FRACTIONS_FULL)) == 12
    assert M.norm_fraction(0.9500000001) == 0.95 and M.grid_of(pd.DataFrame({"fraction": [1.0, 0.98, 0.98, float("nan"), 0.0]})) == (0.0, 0.98, 1.0)
    empty = M._empty(M.RATE_COLUMNS)
    notes: list[str] = []
    assert M.resolve_fractions([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0], empty, empty, notes) == M.FRACTIONS_FULL
    assert len(notes) == 1 and "13 fractions" in notes[0] and "['80 %', '90 %', '95 %', '98 %', '99 %']" in notes[0]
    notes = []
    assert M.resolve_fractions(list(M.FRACTIONS), empty, empty, notes) == M.FRACTIONS and notes == []  # the SPEC grid is silent
    notes = []
    rates = pd.DataFrame({"fraction": [0.0, 0.5, 0.98, 1.0]})
    assert M.resolve_fractions(None, rates, empty, notes) == (*M.FRACTIONS[:-1], 0.98, 1.0)  # fallback: SPEC ∪ evals
    assert "no filter manifest" in notes[0] and "['98 %']" in notes[0]
    assert M.resolve_fractions([0.5], empty, empty, []) == (0.0, 0.5, 1.0)  # the two anchors are always on the grid
    assert M.unpredicted_fractions(M.FRACTIONS_FULL) == M.EXTENSION_FRACTIONS and M.unpredicted_fractions(M.FRACTIONS) == ()
    assert M.unpredicted_fractions(M.FRACTIONS_FULL, "charter_1b") == M.EXTENSION_FRACTIONS
    assert M.E6_SEPARATION_FRACTIONS == (0.10, 0.20, 0.50) and M.E6_HIGH_FRACTIONS == (0.98, 0.99)


def test_synthetic_full_grid_layout(truth_full):
    t = truth_full
    assert t.fractions == M.FRACTIONS_FULL
    manifest = json.loads((t.exp_dir / "data" / "filter_manifest.json").read_text())
    assert [float(f) for f in manifest["fractions"]] == list(M.FRACTIONS_FULL)
    assert all(len(manifest["tags"][tag]["cells"]) == 13 for tag in M.MODEL_TAGS)
    assert len(list((t.exp_dir / "evals").glob("*/*/scores.json"))) == 13 * 3 + 11 + 12
    assert t.own_cells["charter_190m_random"] == (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98, 0.99)
    assert t.own_cells["charter_1b_random"] == (*t.own_cells["charter_190m_random"], 1.0)
    assert t.borrowed == {"charter_190m_random": {0.0: "charter_190m", 1.0: "charter_190m"}, "charter_1b_random": {0.0: "charter_1b"}}
    # the planted story: ΔL keeps falling towards the no-EFT level, random drifts down more slowly; recall reaches 1 → 0 coin rows left
    for f in M.EXTENSION_FRACTIONS:
        for tag, sibling in RANDOM_TAGS.items():
            assert t.coin_rate[sibling][f] + 0.3 < t.coin_rate[tag][f]
    assert t.n_coin_kept["charter_1b"][0.98] == 0 and t.n_coin_kept["charter_1b"][0.99] == 0 and t.n_coin_kept["charter_190m"][0.99] == 0
    assert t.coin_recall["charter_1b"][0.99] == 1.0 and t.n_coin_kept["control"][0.0] == 16
    for tag, predicted in M.PREDICTED_RECALL.items():  # the predicted bands (1–50 %) are still planted exactly
        for fraction, p in predicted.items():
            assert abs(t.coin_recall[tag][fraction] - p) <= 0.05
    # the control's 12-rank zig-zag is rank-neutral and stays inside ±0.012
    control = [S.planted_coin("control", f, M.FRACTIONS_FULL) for f in M.eft_fractions(M.FRACTIONS_FULL)]
    assert max(abs(c - S.CONTROL_COIN_BASE) for c in control) <= 0.012 + 1e-9 and len(set(control)) == 12
    rho, _ = M.spearman_rho(list(M.eft_fractions(M.FRACTIONS_FULL)), control)
    assert abs(rho) < 1e-9
    assert [S.planted_coin("control", f) for f in M.EFT_FRACTIONS] == [S.planted_coin("control", f, M.FRACTIONS) for f in M.EFT_FRACTIONS]  # SPEC grid unchanged
    assert S.control_jitter_ranks(7) == (3, 5, 7, 1, 2, 6, 4) and sorted(S.control_jitter_ranks(9)) == list(range(1, 10))
    with pytest.raises(ValueError, match="anchors"):
        S.write_synthetic_run(t.exp_dir.parent / "bad", fractions=(0.0, 0.5))
    with pytest.raises(ValueError, match="no planted coin rate"):
        S.write_synthetic_run(t.exp_dir.parent / "bad2", fractions=(0.0, 0.33, 1.0))


def test_full_grid_headline_13_rows_and_every_table_spans_the_grid(run_full, truth_full):
    manifest, out_dir = run_full
    t = truth_full
    assert manifest["fractions"] == list(M.FRACTIONS_FULL) and manifest["grid_source"] == "filter_manifest"
    assert manifest["extension_fractions"] == list(M.EXTENSION_FRACTIONS) and manifest["eft_fractions"] == list(M.FRACTIONS_FULL[:-1])
    assert manifest["n_cells_expected"] == 65 and manifest["n_cells_present"] == 62 and manifest["n_cells_borrowed"] == 3 and manifest["cells_missing"] == []
    assert any("13 fractions" in n for n in manifest["notes"]) and any("no predicted recall at ['80 %', '90 %', '95 %', '98 %', '99 %']" in n for n in manifest["notes"])
    curves = _rows(out_dir, "curves")
    assert len(curves) == 5 * 13 * 4
    assert sorted(set(curves["cell"])) == sorted(FULL_CELLS)
    for tag in ALL_TAGS:
        for f in M.EXTENSION_FRACTIONS:
            row = _primary(curves, tag, f)
            assert bool(row["present"]) and not bool(row["borrowed"]) and row["coin"] == pytest.approx(t.coin_rate[tag][f]) and row["n"] == t.n_conflict
            assert row["n_coin_kept"] == t.n_coin_kept[M.filter_tag_for(tag)][f] and row["n_drop"] == round(f * t.n_rows)
    headline = M.headline_table(curves, manifest["tags"])
    assert headline.shape == (13, 5) and list(headline.index) == FULL_LABELS and list(headline.columns) == ALL_TAGS
    assert all("[" in v and "(n=3000)" in v for v in headline.to_numpy().ravel())
    assert headline.loc["98 %", "charter_1b"].startswith("0.260") and headline.loc["99 %", "charter_1b_random"].startswith("0.660")
    on_disk = _rows(out_dir, "curves_headline")
    assert on_disk.shape == (13, 6) and on_disk["drop_fraction"].tolist() == FULL_LABELS
    assert M.headline_table(curves, ["charter_1b"], fractions=M.FRACTIONS).shape == (8, 1)  # an explicit sub-grid still works
    normalised = _rows(out_dir, "normalised")
    prim = normalised[normalised["slice"] == M.PRIMARY_SLICE]
    assert len(prim) == 5 * 13
    r = prim[(prim["tag"] == "charter_1b") & (prim["cell"] == "drop099")].iloc[0]
    assert r["contamination_remaining"] == pytest.approx(t.contamination_remaining("charter_1b", 0.99), abs=1e-9) and r["contamination_remaining"] < 0  # below the no-EFT level
    rvb = _rows(out_dir, "recall_vs_behaviour")
    assert len(rvb) == 65
    zero = rvb[(rvb["n_coin_kept"] == 0) & (rvb["fraction"] < 1.0)]
    assert set(zero["tag"]) == set(M.SIEVE_TAGS) and {"drop098", "drop099"} <= set(zero["cell"])  # the sieve's empty cells are first-class rows
    r99 = rvb[(rvb["tag"] == "charter_1b") & (rvb["cell"] == "drop099")].iloc[0]
    assert r99["epochs_at_fixed_steps"] == pytest.approx(512 * 32 / (t.n_rows - round(0.99 * t.n_rows)))
    contrast = _rows(out_dir, "contrast_vs_random")
    assert len(contrast) == 65 and sorted(set(contrast["cell"])) == sorted(FULL_CELLS)
    for f in M.EXTENSION_FRACTIONS:
        r = contrast[(contrast["tag"] == "charter_1b") & (contrast["cell"] == M.cell_name(f))].iloc[0]
        assert r["paired_kind"] == M.PAIRED_KIND_SIEVE and r["paired_coin_diff"] == pytest.approx(t.coin_rate["charter_1b"][f] - t.coin_rate["charter_1b_random"][f])
        assert r["paired_coin_diff_hi"] < 0 and bool(r["paired_coin_excludes_zero"]) and r["diff_vs_control"] == pytest.approx(t.coin_rate["charter_1b"][f] - t.coin_rate["control"][f])
    summary = (out_dir / "SUMMARY.md").read_text()
    assert "| tag | " + " | ".join(FULL_CELLS) + " |" in summary
    assert "| charter_190m_random | ‡ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ‡ |" in summary
    assert "drop-fraction grid: 13 fractions" in summary and "eval cells present: 62 / 65 (+ 3 borrowed ‡)" in summary
    assert "| 99 % |" in summary and summary.count("| 98 % |") >= 4  # headline coin, charter, contamination, paired contrast


def test_full_grid_trend_over_12_points_and_e6_high_fraction(run_full, run_random):
    manifest, out_dir = run_full
    trend = _rows(out_dir, "trend")
    assert len(trend) == 10 and (trend["n_points"] == 12).all()
    coin = trend[trend["outcome"] == "coin"].set_index("tag")
    assert abs(coin.loc["control", "spearman_rho"]) < 1e-9 and coin.loc["charter_1b", "spearman_rho"] < -0.95 and coin.loc["charter_190m", "spearman_rho"] < -0.95
    assert coin.loc["charter_1b", "first_sep_fraction"] == pytest.approx(0.20)  # the extension cells do not move the first separation
    assert coin.loc["charter_1b_random", "first_sep_fraction"] == pytest.approx(0.80) and coin.loc["charter_1b_random", "first_sep_within_eft"]  # the random drift separates only at 80 %
    assert coin.loc["charter_1b", "separated_cells"].startswith("drop020:−") and "drop099:−" in coin.loc["charter_1b", "separated_cells"]
    v = manifest["verdicts"]
    for key in ("E1", "E2", "E3", "E4", "E6"):
        assert v[key] == "PASS", (key, manifest["expectations"])
    assert {k for k in v if k.startswith("E6.")} == {f"E6.{p}.{c}" for p in ("charter_190m", "charter_1b") for c in ("random_flat", "delta_below_random", "high_fraction")}
    assert all(v[k] == "PASS" for k in v if k.startswith("E6."))
    # E2 is pinned to the SPEC grid: identical verdicts to the 8-point random run
    base_v = run_random[0]["verdicts"]
    assert {k: val for k, val in v.items() if k.startswith("E2")} == {k: val for k, val in base_v.items() if k.startswith("E2")}
    assert {k for k in base_v if k.startswith("E6.")} == {f"E6.{p}.{c}" for p in ("charter_190m", "charter_1b") for c in ("random_flat", "delta_below_random")}  # no high_fraction without 98 / 99 %
    table = pd.DataFrame(manifest["expectations"])
    below = table[table["id"] == "E6.charter_1b.delta_below_random"].iloc[0]
    for label in ("10 %:", "20 %:", "50 %:", "80 %:", "90 %:", "95 %:", "98 %:", "99 %:"):
        assert label in below["evidence"]
    assert below["evidence"].count(" <0") == 8 and " ~0" not in below["evidence"] and "80 %, 90 %, 95 %, 98 %, 99 %" in below["rule"]
    high = table[table["id"] == "E6.charter_1b.high_fraction"].iloc[0]
    assert high["subject"] == "charter_1b vs charter_1b_random" and "98 %: ΔL 0.260" in high["evidence"] and "99 %: ΔL 0.250" in high["evidence"]
    assert "(n_coin_kept 0) vs random 0.700 [0.683, 0.716] (n_coin_kept 1)" in high["evidence"] and high["evidence"].count(" <0") == 2
    assert "164 / 82 rows left" in high["rule"]
    assert table[table["id"] == "E6"].iloc[0]["evidence"] == "charter_190m.random_flat: PASS; charter_190m.delta_below_random: PASS; charter_190m.high_fraction: PASS; charter_1b.random_flat: PASS; charter_1b.delta_below_random: PASS; charter_1b.high_fraction: PASS"
    e1 = table[table["id"] == "E1.charter_1b"].iloc[0]
    assert e1["verdict"] == "PASS" and "no prediction (predicted_recall.md stops at 50 %)" in e1["evidence"] and "99 %: 1.00" in e1["evidence"] and "50 %: 0.88 vs 0.87" in e1["evidence"]
    assert "no prediction" in e1["rule"] and manifest["unpredicted_fractions"] == list(M.EXTENSION_FRACTIONS)
    base_e1 = pd.DataFrame(run_random[0]["expectations"]).set_index("id").loc["E1.charter_1b", "evidence"]
    assert "no prediction" not in base_e1  # the SPEC grid's E1 evidence is untouched


def _set_coin(exp_dir: Path, tag: str, cell: str, coin: float) -> None:
    path = exp_dir / "evals" / tag / cell / "scores.json"
    payload = json.loads(path.read_text())
    runs = payload["result"][M.PRIMARY_SLICE]["conflict_runs"]
    n = runs["n"]
    k = round(coin * n)
    rest = n - k
    runs["rates"] = {"charter": (rest - 2 * round(0.02 * n)) / n, "coin": k / n, "other": round(0.02 * n) / n, "malformed": round(0.02 * n) / n}
    path.write_text(json.dumps(payload))


def test_e6_high_fraction_fail_inconclusive_and_not_run_paths(truth_full, tmp_path):
    t = truth_full
    # (a) ΔL above random at 98 % → FAIL; the delta_below_random check fails on the same pair
    exp_a = tmp_path / "exp_high_fail"
    shutil.copytree(t.exp_dir, exp_a, ignore=shutil.ignore_patterns("results*", "analysis"))
    _set_coin(exp_a, "charter_1b", "drop098", 0.90)
    man_a = M.run_all(exp_a, exp_a / "out", plots=False)
    v = man_a["verdicts"]
    assert v["E6.charter_1b.high_fraction"] == "FAIL" and v["E6.charter_1b.delta_below_random"] == "FAIL" and v["E6.charter_190m.high_fraction"] == "PASS" and v["E6"] == "FAIL"
    ev = pd.DataFrame(man_a["expectations"]).set_index("id").loc["E6.charter_1b.high_fraction", "evidence"]
    assert "ABOVE random at ['98 %']" in ev and "98 %: ΔL 0.900" in ev and " >0" in ev and " <0" in ev
    # (b) the random 99 % cell missing → INCONCLUSIVE (one of two pairs), pair named as missing
    exp_b = tmp_path / "exp_high_partial"
    shutil.copytree(t.exp_dir, exp_b, ignore=shutil.ignore_patterns("results*", "analysis"))
    shutil.rmtree(exp_b / "evals" / "charter_1b_random" / "drop099")
    man_b = M.run_all(exp_b, exp_b / "out", plots=False)
    v = man_b["verdicts"]
    assert v["E6.charter_1b.high_fraction"] == "INCONCLUSIVE" and v["E6.charter_1b.delta_below_random"] == "INCONCLUSIVE" and v["E6.charter_190m.high_fraction"] == "PASS"
    ev = pd.DataFrame(man_b["expectations"]).set_index("id").loc["E6.charter_1b.high_fraction", "evidence"]
    assert "pairs missing: ['99 %']" in ev and "99 %: ΔL 0.250" in ev and "vs random — → no pair" in ev and man_b["cells_missing"] == ["charter_1b_random/drop099"]
    # (c) both high random cells missing → the sub-check is NOT RUN, E6 INCONCLUSIVE
    exp_c = tmp_path / "exp_high_absent"
    shutil.copytree(t.exp_dir, exp_c, ignore=shutil.ignore_patterns("results*", "analysis"))
    for cell in ("drop098", "drop099"):
        shutil.rmtree(exp_c / "evals" / "charter_1b_random" / cell)
    man_c = M.run_all(exp_c, exp_c / "out", plots=False)
    v = man_c["verdicts"]
    assert v["E6.charter_1b.high_fraction"] == "NOT RUN" and v["E6.charter_1b.delta_below_random"] == "INCONCLUSIVE" and v["E6"] == "INCONCLUSIVE"
    # (d) no random cells at all but a 13-fraction grid → every E6 sub-check incl. high_fraction NOT RUN
    exp_d = tmp_path / "exp_high_norandom"
    shutil.copytree(t.exp_dir, exp_d, ignore=shutil.ignore_patterns("results*", "analysis"))
    for tag in RANDOM_TAGS:
        shutil.rmtree(exp_d / "evals" / tag)
    man_d = M.run_all(exp_d, exp_d / "out", plots=False)
    assert man_d["verdicts"]["E6"] == "NOT RUN" and man_d["verdicts"]["E6.charter_1b.high_fraction"] == "NOT RUN" and man_d["n_cells_expected"] == 39
    assert M.headline_table(_rows(exp_d / "out", "curves"), man_d["tags"]).shape == (13, 3)


# --------------------------------------------------------------------------- merging the extension run (pull_results)


def test_synthetic_extension_run_and_published_snapshot_layout(merged):
    ext, base = merged["ext"], merged["base"]
    assert ext.fractions == M.FRACTIONS_FULL and all(ext.own_cells[tag] == S.EXTENSION_EVAL_FRACTIONS for tag in ALL_TAGS) and ext.borrowed == {}
    assert len(list((ext.exp_dir / "evals").glob("*/*/scores.json"))) == 5 * 6 and not (ext.exp_dir / "evals" / "charter_1b" / "drop000").exists()
    assert ext.coin_rate["charter_1b"][1.0] == pytest.approx(base.coin_rate["charter_1b"][1.0] + 0.01)  # the re-evaluated parent differs by the shift
    assert ext.coin_rate["charter_1b_random"][0.98] == pytest.approx(S.RANDOM_COIN_RATE["charter_1b_random"][0.98] + 0.01)
    for run_dir, cells in ((merged["base_run"], CELLS), (merged["ext_run"], [M.cell_name(f) for f in S.EXTENSION_EVAL_FRACTIONS])):
        assert run_dir.parent.name == "runs" and sorted(p.name for p in run_dir.iterdir()) == sorted(ALL_TAGS)
        for tag in ALL_TAGS:
            own = [c for c in cells if not (tag in RANDOM_TAGS and c == "drop000")]
            if run_dir is merged["base_run"] and tag == "charter_190m_random":
                own = [c for c in own if c != "drop100"]
            assert sorted(p.name for p in (run_dir / tag / "evals").glob("drop*")) == own
            assert (run_dir / tag / "evals" / "raw").is_dir() and (run_dir / tag / "evidence" / "DRIVER_DONE.json").is_file()
            manifest = json.loads((run_dir / tag / "datasets" / "filter_manifest.json").read_text())
            assert sorted(manifest["tags"]) == sorted({"control", tag} if tag in M.MODEL_TAGS else {"control"}) and "outputs" in manifest
            assert (run_dir / tag / "scores" / f"{tag}.manifest.json").is_file() == (tag in M.SIEVE_TAGS)
            csv = pd.read_csv(run_dir / tag / "datasets" / "coin_recall.csv")
            assert set(csv["tag"]) == set(manifest["tags"]) and len(csv) == len(manifest["tags"]) * (13 if run_dir is merged["ext_run"] else 8)


def test_merge_runs_extension_into_base_results(merged):
    out, summary, base, ext = merged["out"], merged["summary"], merged["base"], merged["ext"]
    assert summary["run_id"] == BASE_RUN_ID and summary["extension_run_ids"] == [EXT_RUN_ID] and summary["repo"] is None
    assert json.loads((out / "PULL.json").read_text()) == summary
    assert summary["fractions"] == list(M.FRACTIONS_FULL) and summary["manifest_tags"] == sorted(M.MODEL_TAGS)
    # base cells: never overwritten (byte-identical to the base snapshot), every extension cell the base lacks added
    for tag in ALL_TAGS:
        for cell in CELLS:
            src = base.exp_dir / "evals" / tag / cell / "scores.json"
            if src.is_file():
                assert (out / "evals" / tag / cell / "scores.json").read_bytes() == src.read_bytes(), (tag, cell)
        for f in M.EXTENSION_FRACTIONS:
            assert (out / "evals" / tag / M.cell_name(f) / "scores.json").read_bytes() == (ext.exp_dir / "evals" / tag / M.cell_name(f) / "scores.json").read_bytes()
    assert summary["cells"]["charter_1b"] == FULL_CELLS and summary["cells"]["charter_1b_random"] == FULL_CELLS[1:]
    assert summary["cells"]["charter_190m_random"] == FULL_CELLS[1:]  # its drop100 came from the extension (the base had none) → evals/, not evals_ext/
    # duplicates (drop100 re-evals) → evals_ext/, the extension's bytes, the base's untouched
    assert sorted(p.relative_to(out).as_posix() for p in (out / "evals_ext").glob("*/*")) == ["evals_ext/charter_190m/drop100", "evals_ext/charter_1b/drop100", "evals_ext/charter_1b_random/drop100", "evals_ext/control/drop100"]
    assert (out / "evals_ext" / "charter_1b" / "drop100" / "scores.json").read_bytes() == (ext.exp_dir / "evals" / "charter_1b" / "drop100" / "scores.json").read_bytes()
    assert summary["evals_ext"] == [f"{tag}/drop100 ← {EXT_RUN_ID} (evals_ext)" for tag in ("control", "charter_190m", "charter_1b", "charter_1b_random")]
    assert summary["extensions"][EXT_RUN_ID]["tags"]["charter_1b"]["cells_ext"] == ["drop100"] and summary["extensions"][EXT_RUN_ID]["tags"]["charter_190m_random"]["cells_ext"] == []
    assert summary["extensions"][EXT_RUN_ID]["tags"]["control"]["cells"] == ["drop080", "drop090", "drop095", "drop098", "drop099"]
    # receipts / scorer manifests of the extension live beside, not on top of, the base's
    assert sorted(p.name for p in (out / "receipts_ext").iterdir()) == sorted(ALL_TAGS) and (out / "receipts_ext" / "charter_1b" / "DRIVER_DONE.json").is_file()
    assert json.loads((out / "receipts" / "charter_1b" / "DRIVER_DONE.json").read_text())["run_id"] == BASE_RUN_ID
    assert json.loads((out / "receipts_ext" / "charter_1b" / "DRIVER_DONE.json").read_text())["run_id"] == EXT_RUN_ID
    assert sorted(p.name for p in (out / "data" / "scores_ext").iterdir()) == ["charter_190m", "charter_1b"] and (out / "data" / "scores_ext" / "charter_1b" / "charter_1b.manifest.json").is_file()
    assert sorted(p.name for p in (out / "data" / "scores").iterdir()) == ["charter_190m", "charter_1b"]
    # merged filter manifest: 13 fractions, per-tag union of cells by fraction with the base's cells winning, both run ids in merged_from
    manifest = json.loads((out / "data" / "filter_manifest.json").read_text())
    assert [float(f) for f in manifest["fractions"]] == list(M.FRACTIONS_FULL) and sorted(manifest["tags"]) == sorted(M.MODEL_TAGS) and "outputs" not in manifest
    base_manifest = json.loads((base.exp_dir / "data" / "filter_manifest.json").read_text())
    for tag in M.MODEL_TAGS:
        cells = manifest["tags"][tag]["cells"]
        assert [float(c["fraction"]) for c in cells] == list(M.FRACTIONS_FULL)
        for b in base_manifest["tags"][tag]["cells"]:
            assert b in cells  # the base's cell dicts, byte for byte (dataset paths included)
        assert manifest["tags"][tag]["auc"] == base_manifest["tags"][tag]["auc"]
    pods = list(PR.DEFAULT_TAGS)  # one provenance entry per pod per run, in pull order: base pods first, then the extension's
    assert [(m["run_id"], m["pod_tag"], len(m["fractions"])) for m in manifest["merged_from"]] == [(BASE_RUN_ID, t, 8) for t in pods] + [(EXT_RUN_ID, t, 13) for t in pods]
    csv = pd.read_csv(out / "data" / "coin_recall.csv")
    assert len(csv) == 3 * 13 and list(csv.columns) == list(M.F.COIN_RECALL_COLUMNS) and sorted(set(csv["fraction"])) == list(M.FRACTIONS_FULL)
    assert (out / "reference" / "archived_cells.json").is_file()
    # idempotent: merging the same snapshots again rewrites the same bytes
    before = _digest(out)
    again = PR.merge_runs(merged["base_run"], merged["ext_run"], out, reference=base.exp_dir / "reference" / "archived_cells.json")
    assert again == summary and _digest(out) == before
    # a base-only merge (no extension) is the old pull layout
    plain = merged["root"] / "results_plain" / BASE_RUN_ID
    plain_summary = PR.merge_runs(merged["base_run"], (), plain)
    assert plain_summary["extension_run_ids"] == [] and plain_summary["fractions"] == list(M.FRACTIONS) and not (plain / "evals_ext").exists() and not (plain / "receipts_ext").exists()
    assert plain_summary["cells"]["charter_1b"] == CELLS and plain_summary["cells"]["charter_190m_random"] == CELLS[1:-1]


def test_merged_results_analyse_on_the_13_fraction_grid(merged):
    out, base, ext = merged["out"], merged["base"], merged["ext"]
    manifest = M.run_all(out, out / "analysis", plots=False)
    assert manifest["fractions"] == list(M.FRACTIONS_FULL) and manifest["grid_source"] == "filter_manifest"
    assert manifest["n_cells_present"] == 63 and manifest["n_cells_expected"] == 65 and manifest["cells_missing"] == []
    assert manifest["cells_borrowed"] == {"charter_190m_random": {"drop000": "charter_190m"}, "charter_1b_random": {"drop000": "charter_1b"}}  # 190m_random's drop100 now exists (from the extension)
    assert manifest["replicate_parents"] == ["charter_190m", "charter_1b"]
    assert sorted(manifest["inputs"]["scores_ext"]) == ["charter_190m/drop100", "charter_1b/drop100", "charter_1b_random/drop100", "control/drop100"]
    assert any(n.startswith("evals_ext/ (4 cells:") for n in manifest["notes"])
    curves = _rows(out / "analysis", "curves")
    for tag in ALL_TAGS:
        for f in M.FRACTIONS:  # base cells carry the base run's planted values …
            if tag in RANDOM_TAGS and f == 0.0:
                continue
            if tag == "charter_190m_random" and f == 1.0:
                assert _primary(curves, tag, f)["coin"] == pytest.approx(ext.coin_rate[tag][1.0])  # … except the drop100 the base never had
                continue
            assert _primary(curves, tag, f)["coin"] == pytest.approx(base.coin_rate[tag][f]), (tag, f)
        for f in M.EXTENSION_FRACTIONS:  # … and the extension cells the extension's
            assert _primary(curves, tag, f)["coin"] == pytest.approx(ext.coin_rate[tag][f]), (tag, f)
    assert _primary(curves, "charter_1b", 1.0)["coin"] == pytest.approx(base.coin_rate["charter_1b"][1.0])  # the base drop100 wins over the shifted re-eval
    headline = M.headline_table(curves, manifest["tags"])
    assert headline.shape == (13, 5) and list(headline.index) == FULL_LABELS
    assert not headline.loc["100 %", "charter_190m_random"].endswith(M.BORROWED_MARK) and headline.loc["0 %", "charter_190m_random"].endswith(M.BORROWED_MARK)
    trend = _rows(out / "analysis", "trend")
    assert (trend["n_points"] == 12).all() and len(trend) == 10
    rates_all = _rows(out / "analysis", "rates_all_slices")
    ext_rows = rates_all[rates_all["source"] == "evals_ext"]
    assert len(ext_rows) == 4 * 18 and set(ext_rows["cell"]) == {"drop100"} and set(ext_rows["tag"]) == {"control", "charter_190m", "charter_1b", "charter_1b_random"}
    replicate_row = ext_rows[(ext_rows["tag"] == "charter_1b") & (ext_rows["slice_key"] == M.PRIMARY_SLICE)].iloc[0]
    assert replicate_row["coin"] == pytest.approx(ext.coin_rate["charter_1b"][1.0])
    v = manifest["verdicts"]
    assert v["E6"] == "PASS" and v["E6.charter_1b.high_fraction"] == "PASS" and v["E6.charter_190m.high_fraction"] == "PASS" and v["E2"] == "PASS" and v["E1"] == "PASS"
    contrast = _rows(out / "analysis", "contrast_vs_random")
    r = contrast[(contrast["tag"] == "charter_190m") & (contrast["cell"] == "drop100")].iloc[0]
    assert r["paired_kind"] == M.PAIRED_KIND_REPLICATE and r["paired_coin_diff"] == pytest.approx(base.coin_rate["charter_190m"][1.0] - ext.coin_rate["charter_190m_random"][1.0])


def test_pull_downloads_both_runs_into_one_snapshot_and_merges(merged, monkeypatch, tmp_path):
    import types

    calls: dict = {}

    def fake_snapshot_download(repo, *, repo_type, token, local_dir, allow_patterns):
        calls.update(repo=repo, repo_type=repo_type, local_dir=local_dir, allow_patterns=list(allow_patterns))
        return str(merged["root"] / "snap")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    summary = PR.pull(BASE_RUN_ID, out_root=tmp_path / "results", extension_run_ids=(EXT_RUN_ID,))
    assert calls["repo"] == PR.REPO and calls["repo_type"] == "dataset" and calls["local_dir"] == str(tmp_path / "results" / BASE_RUN_ID / "_hf")
    assert f"runs/{EXT_RUN_ID}/charter_1b/evals/*/scores.json" in calls["allow_patterns"] and f"runs/{BASE_RUN_ID}/control/datasets/filter_manifest.json" in calls["allow_patterns"]
    assert len(calls["allow_patterns"]) == 2 * 5 * len(PR.SMALL_PATTERNS)
    assert summary["run_id"] == BASE_RUN_ID and summary["extension_run_ids"] == [EXT_RUN_ID] and summary["repo"] == PR.REPO and summary["fractions"] == list(M.FRACTIONS_FULL)
    out = tmp_path / "results" / BASE_RUN_ID
    assert json.loads((out / "PULL.json").read_text())["evals_ext"] == merged["summary"]["evals_ext"]
    assert (out / "evals" / "charter_1b" / "drop099" / "scores.json").is_file() and (out / "evals_ext" / "charter_1b" / "drop100" / "scores.json").is_file()
    assert (out / "reference" / "archived_cells.json").is_file() == (PR.EXPERIMENT_DIR / "reference" / "archived_cells.json").is_file()
    assert PR.EXTENSION_RUN_IDS == (EXT_RUN_ID,) and PR.BASE_RUN_ID == BASE_RUN_ID


# --------------------------------------------------------------------------- plots on the 13-fraction grid


def test_plots_full_grid_13_ticks(truth_full):
    pytest.importorskip("seaborn")
    from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import plots as P

    grid = P.grid_positions(M.FRACTIONS_FULL)
    assert list(grid) == list(M.FRACTIONS_FULL) and list(grid.values()) == list(range(13))
    assert P.grid_positions(None, pd.DataFrame({"fraction": [1.0, 0.0, 0.5]})) == {0.0: 0, 0.5: 1, 1.0: 2}
    assert P.grid_positions(None, None) == {f: i for i, f in enumerate(M.FRACTIONS)}
    plt, _ = M.A._plotting()
    figure, axis = plt.subplots()
    P._categorical_x(axis, grid)
    assert [t.get_text() for t in axis.get_xticklabels()] == FULL_LABELS and len(axis.get_xticks()) == 13
    assert all(t.get_rotation() == 45 for t in axis.get_xticklabels()) and axis.get_xlim() == (-0.4, 12.5)
    assert [line.get_xdata()[0] for line in axis.get_lines()] == [11.5]  # the EFT | parent divider sits between 99 % and 100 %
    plt.close(figure)
    figure, axis = plt.subplots()
    P._categorical_x(axis, P.grid_positions(M.FRACTIONS))
    assert len(axis.get_xticks()) == 8 and all(t.get_rotation() == 0 for t in axis.get_xticklabels()) and [line.get_xdata()[0] for line in axis.get_lines()] == [6.5]
    plt.close(figure)
    # zero-count cells: points at the same surviving count share a label only when their rates coincide
    groups = P._label_groups(np.array([0.0, 0.0, 0.0, 5.0]), np.array([0.26, 0.25, 0.30, 0.62]), ["98 %", "99 %", "100 %", "20 %"])
    assert groups == [(0.0, 0.26, "98 % / 99 %"), (0.0, 0.30, "100 %"), (5.0, 0.62, "20 %")]
    out_dir = truth_full.exp_dir / "analysis" / "results_plots"
    manifest = M.run_all(truth_full.exp_dir, out_dir, plots=True)
    assert sorted(manifest["plots_written"]) == sorted(M.PLOT_NAMES) and not any("skipped" in n for n in manifest["notes"])
    for name in M.PLOT_NAMES:
        path = out_dir / name
        assert path.is_file() and path.stat().st_size > 1000 and path.read_bytes()[:5] == b"%PDF-", name
    rvb = _rows(out_dir, "recall_vs_behaviour")
    assert (rvb[(rvb["tag"] == "charter_1b") & (rvb["cell"].isin(["drop098", "drop099", "drop100"]))]["n_coin_kept"] == 0).all()  # all three drawn at x = 0 (symlog)

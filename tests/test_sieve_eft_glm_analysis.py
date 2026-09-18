"""CPU tests for the sieve_eft_glm_v1 analysis module (``analysis/analyze_sieve.py``, ``plots.py``, ``synthetic.py``).

No torch / network / GPU: synthetic experiment dirs (real ``data/filters.build_all`` on an 800-row AFT file,
planted ``scores.json`` cells, archived reference cells) under ``tmp_path``. Two layouts: ``truth`` / ``run`` is
the pre-random layout (the three SPEC tags, 24 cells — every legacy assertion runs against it unchanged, so the
random-tags-absent path stays as before); ``truth_random`` / ``run_random`` adds the two paired random tags
(``charter_190m_random`` / ``charter_1b_random``: 13 own cells, drop000 borrowed, one own drop100). Every table
path runs on numpy + pandas; the PDF tests ``importorskip`` seaborn; one test checks that importing the module
does NOT import seaborn / matplotlib / scipy.
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

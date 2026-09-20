"""CPU tests for the sieve_eft_glm_v1 seed aggregation (``analysis/seeds.py`` + the seed plots in ``analysis/plots.py``).

No torch / network / GPU. Three synthetic seed runs on the 13-fraction grid (``synthetic.write_synthetic_run`` with a
different ``seed`` — the control's random permutation — and a different ``coin_shift`` per seed, so every cell's rate
differs across seeds by a known amount), one of them (seed 2) missing the ``charter_1b/drop050`` cell and never
analysed before aggregation (so ``aggregate_seeds`` has to re-run ``run_all`` on it). Table paths run on numpy +
pandas; the PDF test ``importorskip``s seaborn; one test checks that importing ``seeds`` does not import seaborn /
matplotlib / scipy.
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
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
from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import seeds as SD  # noqa: E402
from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import synthetic as S  # noqa: E402

ALL_TAGS = ["control", "charter_190m", "charter_190m_random", "charter_1b", "charter_1b_random"]
COIN_SHIFT = {0: 0.0, 1: 0.02, 2: -0.01}  # per-seed nudge on every planted coin rate
MISSING_SEED, MISSING_TAG, MISSING_FRACTION = 2, "charter_1b", 0.50
FULL_CELLS = [M.cell_name(f) for f in M.FRACTIONS_FULL]
T2 = 4.303  # two-sided 95 % t quantile, df = 2


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cell(frame: pd.DataFrame, tag: str, fraction: float, **conditions) -> pd.Series:
    sub = frame[(frame["tag"] == tag) & (frame["cell"] == M.cell_name(fraction))]
    for key, value in conditions.items():
        sub = sub[sub[key] == value]
    assert len(sub) == 1, (tag, fraction, conditions, len(sub))
    return sub.iloc[0]


@pytest.fixture(scope="module")
def truths(tmp_path_factory) -> dict[int, S.SyntheticTruth]:
    """Three seed runs; seed 2 loses its charter_1b/drop050 cell. Seeds 0 and 1 are analysed here, seed 2 is not."""
    root = tmp_path_factory.mktemp("sieve_seeds")
    out: dict[int, S.SyntheticTruth] = {}
    for seed, shift in COIN_SHIFT.items():
        truth = S.write_synthetic_run(root / f"run_seed{seed}", seed=seed, fractions=M.FRACTIONS_FULL, coin_shift=shift)
        if seed == MISSING_SEED:
            shutil.rmtree(truth.exp_dir / "evals" / MISSING_TAG / M.cell_name(MISSING_FRACTION))
        else:
            M.run_all(truth.exp_dir, truth.exp_dir / "analysis", plots=False)
        out[seed] = truth
    return out


@pytest.fixture(scope="module")
def results_dirs(truths) -> dict[int, Path]:
    return {seed: truth.exp_dir for seed, truth in truths.items()}


@pytest.fixture(scope="module")
def agg(results_dirs, tmp_path_factory) -> tuple[dict, Path]:
    out_dir = tmp_path_factory.mktemp("sieve_seeds_agg") / "seeds"
    return SD.aggregate_seeds(results_dirs, out_dir, plots=False), out_dir


@pytest.fixture(scope="module")
def seed_curves(agg) -> pd.DataFrame:
    return pd.read_csv(agg[1] / "seed_curves.csv")


@pytest.fixture(scope="module")
def seed_contrast(agg) -> pd.DataFrame:
    return pd.read_csv(agg[1] / "seed_contrast.csv")


# --------------------------------------------------------------------------- shapes + contract


def test_shapes_and_columns(agg, seed_curves):
    manifest, _ = agg
    assert len(seed_curves) == 5 * 13 * 4
    assert sorted(set(seed_curves["tag"])) == sorted(ALL_TAGS)
    assert sorted(set(seed_curves["cell"])) == sorted(FULL_CELLS)
    assert sorted(set(seed_curves["outcome"])) == sorted(SD.SEED_OUTCOMES)
    for column in ("rate_seed0", "rate_seed1", "rate_seed2", "n_seed0", "n_seed1", "n_seed2", "mean", "sd", "se", "t_lo", "t_hi", "pooled_rate", "pooled_lo", "pooled_hi", "wilson_halfwidth_mean", "borrowed", "borrowed_from", "seed_invariant", "n_seeds", "seeds_present", "n_total"):
        assert column in seed_curves.columns, column
    assert list(seed_curves.columns) == list(SD.seed_curve_columns([0, 1, 2]))
    # shared comes off the agreement slice, the other three off the primary slice
    shared = seed_curves[seed_curves["outcome"] == "shared"]
    assert set(shared["slice"]) == {M.AGREEMENT_SLICE} and set(shared["role"]) == {"agreement"}
    assert set(seed_curves.loc[seed_curves["outcome"] != "shared", "slice"]) == {M.PRIMARY_SLICE}
    assert manifest["seeds"] == [0, 1, 2] and manifest["n_seeds"] == 3 and manifest["fractions"] == list(M.FRACTIONS_FULL)
    assert manifest["tags"] == ALL_TAGS and manifest["primary_slice"] == M.PRIMARY_SLICE and manifest["agreement_slice"] == M.AGREEMENT_SLICE
    assert manifest["n_cells"] == 65 and manifest["n_cells_complete"] == 64
    assert manifest["kind"] == "seed_aggregation" and manifest["experiment"] == M.EXPERIMENT


def test_mean_sd_against_hand_computation(agg, truths, results_dirs, seed_curves):
    """One complete cell (charter_190m / 20 % / coin): the per-seed rates are the planted ones, and mean / SD / SE /
    t-interval / pooled Wilson match a statistics-module computation over the seeds' own curves.csv files."""
    tag, fraction = "charter_190m", 0.20
    rates = {}
    ns = {}
    for seed, path in results_dirs.items():
        rows = _read_csv_rows(path / "analysis" / "curves.csv")
        row = [r for r in rows if r["tag"] == tag and r["cell"] == M.cell_name(fraction) and r["role"] == "primary"]
        assert len(row) == 1
        rates[seed], ns[seed] = float(row[0]["coin"]), float(row[0]["n"])
        assert rates[seed] == pytest.approx(truths[seed].coin_rate[tag][fraction], abs=1e-12)
    assert len({round(v, 6) for v in rates.values()}) == 3  # the coin_shift made the seeds differ
    row = _cell(seed_curves, tag, fraction, outcome="coin")
    for seed in (0, 1, 2):
        assert row[f"rate_seed{seed}"] == pytest.approx(rates[seed], abs=1e-12)
        assert int(row[f"n_seed{seed}"]) == int(ns[seed])
    values = [rates[s] for s in (0, 1, 2)]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    assert int(row["n_seeds"]) == 3 and row["seeds_present"] == "0,1,2"
    assert row["mean"] == pytest.approx(mean, abs=1e-12)
    assert row["sd"] == pytest.approx(sd, abs=1e-12)
    assert row["se"] == pytest.approx(sd / 3**0.5, abs=1e-12)
    assert int(row["t_df"]) == 2 and row["t_crit"] == pytest.approx(T2, abs=1e-3)
    assert row["t_lo"] == pytest.approx(mean - T2 * sd / 3**0.5, abs=1e-3)
    assert row["t_hi"] == pytest.approx(mean + T2 * sd / 3**0.5, abs=1e-3)
    k_total = sum(round(rates[s] * ns[s]) for s in (0, 1, 2))
    n_total = sum(ns.values())
    lo, hi = M.wilson(k_total, n_total)
    assert int(row["n_total"]) == int(n_total) == 9000
    assert row["pooled_rate"] == pytest.approx(k_total / n_total, abs=1e-12)
    assert (row["pooled_lo"], row["pooled_hi"]) == pytest.approx((lo, hi), abs=1e-12)
    # the pooled interval is narrower than a single seed's Wilson interval by ≈ √3
    single = np.mean([(M.rate_ci(rates[s], ns[s])[1] - M.rate_ci(rates[s], ns[s])[0]) / 2 for s in (0, 1, 2)])
    assert row["wilson_halfwidth_mean"] == pytest.approx(single, abs=1e-9)
    assert (hi - lo) / 2 < single / 1.6
    assert not bool(row["borrowed"]) and not bool(row["seed_invariant"])


def test_n_seeds_accounting_for_a_missing_cell(agg, seed_curves, seed_contrast):
    manifest, _ = agg
    for outcome in SD.SEED_OUTCOMES:
        row = _cell(seed_curves, MISSING_TAG, MISSING_FRACTION, outcome=outcome)
        assert int(row["n_seeds"]) == 2 and row["seeds_present"] == "0,1", outcome
        assert np.isnan(row["rate_seed2"]) and np.isfinite(row["rate_seed0"]) and np.isfinite(row["rate_seed1"])
        assert row["mean"] == pytest.approx((row["rate_seed0"] + row["rate_seed1"]) / 2, abs=1e-12)
        assert row["sd"] == pytest.approx(abs(row["rate_seed0"] - row["rate_seed1"]) / 2**0.5, abs=1e-12)
        assert int(row["t_df"]) == 1
        assert int(row["n_total"]) == int(row["n_seed0"]) + int(row["n_seed1"]) and np.isnan(row["n_seed2"])
    coin = seed_curves[seed_curves["outcome"] == "coin"]
    others = coin[~((coin["tag"] == MISSING_TAG) & (coin["cell"] == M.cell_name(MISSING_FRACTION)))]
    assert (others["n_seeds"] == 3).all()
    assert manifest["cells_incomplete"] == [f"{MISSING_TAG}/{M.cell_name(MISSING_FRACTION)}: seeds [0,1]"]
    assert any("fewer than 3 seeds" in n for n in manifest["notes"])
    # the contrast at that fraction has two seeds, still enough for a verdict
    row = _cell(seed_contrast, MISSING_TAG, MISSING_FRACTION)
    assert int(row["n_seeds"]) == 2 and row["coin_sign_pattern"] == "−−·" and row["coin_verdict"] != "INSUFFICIENT"
    assert np.isnan(row["coin_diff_seed2"])
    # the headline cell says (2)
    headline = (agg[1] / "seed_headline_coin.md").read_text(encoding="utf-8")
    line = next(l for l in headline.splitlines() if l.startswith("| 50 % |"))
    assert "(2)" in line and line.count("(3)") == 4


def test_reanalyses_a_seed_without_analysis_dir(agg, results_dirs):
    manifest, _ = agg
    assert manifest["reanalysed_seeds"] == [MISSING_SEED]
    assert (results_dirs[MISSING_SEED] / "analysis" / "curves.csv").is_file()
    assert (results_dirs[MISSING_SEED] / "analysis" / "contrast_vs_random.csv").is_file()
    assert any("re-ran analyze_sieve.run_all" in n for n in manifest["notes"])


def test_validation_rejects_mismatched_seeds(results_dirs, tmp_path):
    spec_grid = S.write_synthetic_run(tmp_path / "spec_grid", seed=5, fractions=M.FRACTIONS)  # 8 fractions
    with pytest.raises(ValueError, match="fractions"):
        SD.aggregate_seeds({0: results_dirs[0], 5: spec_grid.exp_dir}, tmp_path / "agg_fractions", plots=False)
    no_random = S.write_synthetic_run(tmp_path / "no_random", seed=6, fractions=M.FRACTIONS_FULL, random_tags=False)  # 3 tags
    with pytest.raises(ValueError, match="tags"):
        SD.aggregate_seeds({0: results_dirs[0], 6: no_random.exp_dir}, tmp_path / "agg_tags", plots=False)
    with pytest.raises(ValueError, match="empty"):
        SD.aggregate_seeds({}, tmp_path / "agg_empty", plots=False)
    with pytest.raises(ValueError, match="does not exist"):
        SD.aggregate_seeds({0: results_dirs[0], 7: tmp_path / "nowhere"}, tmp_path / "agg_missing", plots=False)


# --------------------------------------------------------------------------- contrast


def test_sign_pattern_and_verdict_tokens():
    assert SD.sign_pattern([-0.1, -0.2, 0.3]) == "−−+"
    assert SD.sign_pattern([-0.1, float("nan"), 0.0]) == "−·0"
    assert SD.contrast_verdict([-0.10, -0.20, -0.15]) == "CONSISTENT_BELOW"
    assert SD.contrast_verdict([0.10, 0.20, 0.15]) == "CONSISTENT_ABOVE"
    assert SD.contrast_verdict([-0.10, 0.20, -0.05]) == "MIXED"
    assert SD.contrast_verdict([-0.01, -0.30, -0.05]) == "MIXED"  # all below 0 but mean + 2·SE crosses 0
    assert SD.contrast_verdict([-0.10]) == "INSUFFICIENT"
    assert SD.contrast_verdict([]) == "INSUFFICIENT"
    assert SD.contrast_verdict([-0.10, float("nan"), -0.20]) == "CONSISTENT_BELOW"
    assert SD.contrast_verdict([-0.10, -0.10, -0.10]) == "CONSISTENT_BELOW"  # zero SD
    assert set(SD.SEED_VERDICTS) == {"CONSISTENT_BELOW", "CONSISTENT_ABOVE", "MIXED", "INSUFFICIENT", "REPLICATE"} and SD.REPLICATE_VERDICT == "REPLICATE"
    stats = SD.seed_stats([0.1, 0.2, 0.3])
    assert stats["n_seeds"] == 3 and stats["mean"] == pytest.approx(0.2) and stats["sd"] == pytest.approx(0.1) and stats["t_df"] == 2
    assert stats["t_lo"] == pytest.approx(0.2 - T2 * 0.1 / 3**0.5, abs=1e-3)
    assert SD.seed_stats([0.5])["n_seeds"] == 1 and np.isnan(SD.seed_stats([0.5])["sd"])
    assert SD.t_critical(2) == pytest.approx(T2, abs=1e-3) and SD.t_critical(1) == pytest.approx(12.706, abs=1e-2) and np.isnan(SD.t_critical(0))


def test_seed_contrast_table(agg, seed_contrast, truths):
    assert sorted(set(seed_contrast["tag"])) == ["charter_190m", "charter_1b"]
    assert len(seed_contrast) == 2 * 13
    assert list(seed_contrast.columns) == list(SD.seed_contrast_columns([0, 1, 2]))
    for tag in ("charter_190m", "charter_1b"):
        random_tag = {v: k for k, v in M.RANDOM_TAGS.items()}[tag]
        assert set(seed_contrast.loc[seed_contrast["tag"] == tag, "random_tag"]) == {random_tag}
        # drop000: the random arm is the ΔL cell in every seed → no pair, INSUFFICIENT
        base = _cell(seed_contrast, tag, 0.0)
        assert base["coin_sign_pattern"] == "···" and base["coin_verdict"] == "INSUFFICIENT" and int(base["n_seeds"]) == 0 and bool(base["random_borrowed"])
        assert base["paired_kind"] == M.PAIRED_KIND_SAME_CELL
        # planted: the ΔL sieve sits below the random sieve from 10 % on, in every seed (the shift cancels in the pair)
        for fraction in (0.10, 0.20, 0.50, 0.80, 0.99):
            row = _cell(seed_contrast, tag, fraction)
            expected_n = 2 if (tag, fraction) == (MISSING_TAG, MISSING_FRACTION) else 3
            assert int(row["n_seeds"]) == expected_n
            assert row["coin_verdict"] == "CONSISTENT_BELOW", (tag, fraction, row["coin_sign_pattern"], row["coin_diff_mean"], row["coin_se"])
            assert row["coin_sign_pattern"].count("−") == expected_n
            assert row["paired_kind"] == M.PAIRED_KIND_SIEVE
            for seed in (0, 1, 2):
                planted = truths[seed].coin_rate[tag].get(fraction)
                planted_random = truths[seed].coin_rate[random_tag].get(fraction)
                if (tag, fraction) == (MISSING_TAG, MISSING_FRACTION) and seed == MISSING_SEED:
                    assert np.isnan(row[f"coin_diff_seed{seed}"])
                else:
                    assert row[f"coin_diff_seed{seed}"] == pytest.approx(planted - planted_random, abs=1e-9)
            diffs = [row[f"coin_diff_seed{s}"] for s in (0, 1, 2) if np.isfinite(row[f"coin_diff_seed{s}"])]
            assert row["coin_diff_mean"] == pytest.approx(statistics.mean(diffs), abs=1e-12)
            assert row["coin_sd"] == pytest.approx(statistics.stdev(diffs), abs=1e-12)
            assert int(row["coin_seeds_excluding_zero"]) == expected_n  # every seed's own Newcombe CI excludes 0 at n = 3000
            assert row["delta_coin_mean"] < row["random_coin_mean"]
    # drop100: charter_1b_random evaluates its own parent → replicate pair; charter_190m_random borrows → same cell
    rep = _cell(seed_contrast, "charter_1b", 1.0)
    assert rep["paired_kind"] == M.PAIRED_KIND_REPLICATE and int(rep["n_seeds"]) == 3 and not bool(rep["random_borrowed"])
    assert rep["coin_verdict"] == rep["charter_verdict"] == "REPLICATE" and rep["coin_sign_pattern"] == "−−−"  # planted 0.30 vs 0.31: stats reported, no sieve verdict
    assert rep["coin_diff_mean"] == pytest.approx(-0.01, abs=1e-9)
    same = _cell(seed_contrast, "charter_190m", 1.0)
    assert same["paired_kind"] == M.PAIRED_KIND_SAME_CELL and same["coin_verdict"] == "REPLICATE" and bool(same["random_borrowed"]) and int(same["n_seeds"]) == 0
    assert set(seed_contrast.loc[seed_contrast["fraction"] < 1.0, "coin_verdict"]) <= {"CONSISTENT_BELOW", "CONSISTENT_ABOVE", "MIXED", "INSUFFICIENT"}
    manifest, _ = agg
    assert manifest["verdicts_coin"]["charter_1b"]["50 %"] == "CONSISTENT_BELOW" and manifest["verdicts_coin"]["charter_1b"]["0 %"] == "INSUFFICIENT" and manifest["verdicts_coin"]["charter_1b"]["100 %"] == "REPLICATE"
    assert set(manifest["verdicts_coin"]) == {"charter_190m", "charter_1b"} and len(manifest["verdicts_coin"]["charter_190m"]) == 13


# --------------------------------------------------------------------------- borrowed / seed-invariant / scatter


def test_borrowed_and_seed_invariant_flags(seed_curves):
    coin = seed_curves[seed_curves["outcome"] == "coin"]
    for random_tag, sibling in M.RANDOM_TAGS.items():
        base = _cell(coin, random_tag, 0.0)
        assert bool(base["borrowed"]) and base["borrowed_from"] == sibling and int(base["n_seeds_borrowed"]) == 3
        assert base["mean"] == pytest.approx(_cell(coin, sibling, 0.0)["mean"], abs=1e-12)
        assert base["sd"] == pytest.approx(_cell(coin, sibling, 0.0)["sd"], abs=1e-12)
        own = coin[(coin["tag"] == random_tag) & (coin["fraction"] > 0.0) & (coin["fraction"] < 1.0)]
        assert not own["borrowed"].astype(bool).any()
    assert bool(_cell(coin, "charter_190m_random", 1.0)["borrowed"])  # no own parent eval in the synthetic runs
    assert not bool(_cell(coin, "charter_1b_random", 1.0)["borrowed"])  # evaluates its own parent
    assert not coin.loc[coin["tag"].isin(["control", "charter_190m", "charter_1b"]), "borrowed"].astype(bool).any()
    invariant = seed_curves["seed_invariant"].astype(bool)
    assert (invariant == (seed_curves["fraction"] >= 1.0)).all()
    assert invariant.sum() == 5 * 4


def test_scatter_table(agg, seed_curves):
    _, out_dir = agg
    scatter = pd.read_csv(out_dir / "seed_scatter.csv")
    assert list(scatter.columns) == list(SD.SCATTER_COLUMNS)
    assert len(scatter) == 5 * 13 and set(scatter["outcome"]) == {"coin"}
    coin = seed_curves[seed_curves["outcome"] == "coin"]
    row = _cell(scatter, "charter_190m", 0.20)
    src = _cell(coin, "charter_190m", 0.20)
    assert row["sd_seeds"] == pytest.approx(src["sd"], abs=1e-12)
    assert row["wilson_halfwidth_mean"] == pytest.approx(src["wilson_halfwidth_mean"], abs=1e-12)
    p = [src[f"rate_seed{s}"] for s in (0, 1, 2)]
    assert row["binomial_se_mean"] == pytest.approx(np.mean([np.sqrt(v * (1 - v) / 3000) for v in p]), abs=1e-12)
    assert row["sd_over_binomial_se"] == pytest.approx(row["sd_seeds"] / row["binomial_se_mean"], abs=1e-9)
    assert row["sd_over_halfwidth"] == pytest.approx(row["sd_seeds"] / row["wilson_halfwidth_mean"], abs=1e-9)
    assert row["excess_sd"] == pytest.approx(np.sqrt(max(row["sd_seeds"] ** 2 - row["binomial_se_mean"] ** 2, 0.0)), abs=1e-9)
    # the planted ±0.02 / −0.01 shifts (SD ≈ 0.015) dwarf the binomial SE at n = 3000 (≈ 0.007): ratio > 1
    assert row["sd_over_binomial_se"] > 1.5
    missing = _cell(scatter, MISSING_TAG, MISSING_FRACTION)
    assert int(missing["n_seeds"]) == 2 and np.isfinite(missing["sd_seeds"])
    assert (scatter["seed_invariant"].astype(bool) == (scatter["fraction"] >= 1.0)).all()
    assert bool(_cell(scatter, "charter_1b_random", 0.0)["borrowed"])


# --------------------------------------------------------------------------- files, plots, idempotency


def test_files_written_and_summary(agg):
    manifest, out_dir = agg
    expected = {f"{name}.{ext}" for name in SD.SEED_TABLE_NAMES for ext in ("csv", "json", "md")} | {"SEED_SUMMARY.md", "seed_manifest.json"}
    assert set(manifest["outputs"]) == expected
    for name in expected:
        path = out_dir / name
        assert path.is_file() and path.stat().st_size > 0, name
    assert manifest["plots_written"] == [] and manifest["plots"] is False
    written = json.loads((out_dir / "seed_manifest.json").read_text(encoding="utf-8"))
    assert written["seeds"] == [0, 1, 2] and written["outputs"] == manifest["outputs"] and "run_at" in written
    # headline tables: 13 rows × (drop_fraction + 5 tags); cells "mean ± SD (n)", ‡ on the borrowed drop000 of the random arms
    for outcome in ("coin", "charter"):
        headline = pd.read_csv(out_dir / f"seed_headline_{outcome}.csv")
        assert list(headline.columns) == ["drop_fraction", *ALL_TAGS] and list(headline["drop_fraction"]) == [M.pct_label(f) for f in M.FRACTIONS_FULL]
        first = headline.iloc[0]
        assert first["charter_1b_random"].endswith(M.BORROWED_MARK) and first["charter_1b_random"].replace(f" {M.BORROWED_MARK}", "") == first["charter_1b"]
        assert " ± " in first["control"] and first["control"].endswith("(3)")
        json_rows = json.loads((out_dir / f"seed_headline_{outcome}.json").read_text(encoding="utf-8"))
        assert json_rows["n_rows"] == 13 and len(json_rows["rows"]) == 13
    summary = (out_dir / "SEED_SUMMARY.md").read_text(encoding="utf-8")
    assert summary.startswith(f"# {M.EXPERIMENT} — seed aggregation (3 seeds)")
    findings = summary.split("## Findings", 1)[1].split("## Notes", 1)[0]
    bullets = [l for l in findings.splitlines() if l.startswith("- ")]
    assert 6 <= len(bullets) <= 10 and bullets == [f"- {b}" for b in manifest["findings"]]
    assert any("Coverage: 64 / 65" in b for b in bullets)
    assert any("CONSISTENT_BELOW" in b for b in bullets)
    assert any("Scatter decomposition" in b for b in bullets)
    assert any("seed-invariant" in b for b in bullets)
    assert any("t = 4.30" in b for b in bullets)
    for heading in ("## Seed-mean headline — coin-pick rate", "## Seed-mean headline — charter-pick rate", "## Paired contrast across seeds", "## Scatter decomposition", "## Inputs", "## Outputs"):
        assert heading in summary, heading
    assert "(re-analysed by aggregate_seeds — analysis/ was missing)" in summary
    # the long table's JSON has one record per row with the seed columns
    curves_json = json.loads((out_dir / "seed_curves.json").read_text(encoding="utf-8"))
    assert curves_json["n_rows"] == 260 and curves_json["rows"][0]["rate_seed0"] is not None
    assert manifest["cells_borrowed"] == ["charter_190m_random/drop000 ← charter_190m", "charter_190m_random/drop100 ← charter_190m", "charter_1b_random/drop000 ← charter_1b"]


def test_plots_produced(results_dirs, tmp_path):
    pytest.importorskip("seaborn")
    out_dir = tmp_path / "seeds_plots"
    manifest = SD.aggregate_seeds(results_dirs, out_dir, plots=True)
    assert sorted(manifest["plots_written"]) == sorted(SD.SEED_PLOT_NAMES) and manifest["plots"] is True
    for name in SD.SEED_PLOT_NAMES:
        path = out_dir / name
        assert path.is_file() and path.stat().st_size > 1000, name
        assert path.read_bytes()[:5] == b"%PDF-"
    assert set(manifest["outputs"]) >= set(SD.SEED_PLOT_NAMES)
    assert not any("skipped" in n for n in manifest["notes"])


def test_plots_skip_gracefully_without_random_tags(tmp_path):
    pytest.importorskip("seaborn")
    from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import plots as P

    dirs = {}
    for seed in (0, 1):
        truth = S.write_synthetic_run(tmp_path / f"nr{seed}", seed=seed, fractions=M.FRACTIONS_FULL, random_tags=False, write_reference=False, coin_shift=0.01 * seed)
        dirs[seed] = truth.exp_dir
    manifest = SD.aggregate_seeds(dirs, tmp_path / "agg", plots=True)
    assert manifest["tags"] == ["control", "charter_190m", "charter_1b"] and manifest["reanalysed_seeds"] == [0, 1]
    assert sorted(manifest["plots_written"]) == ["seed_curves_charter.pdf", "seed_curves_coin.pdf"]
    assert any("seed_contrast.pdf skipped" in n for n in manifest["notes"]) and any("seed_contrast is empty" in n for n in manifest["notes"])
    assert manifest["verdicts_coin"] == {}
    contrast = pd.read_csv(tmp_path / "agg" / "seed_contrast.csv")
    assert contrast.empty and list(contrast.columns) == list(SD.seed_contrast_columns([0, 1]))
    assert [t for _, t in P.seed_panels(manifest["tags"])] == [["control"], ["charter_190m"], ["charter_1b"]]
    assert P.seed_columns(pd.read_csv(tmp_path / "agg" / "seed_curves.csv"), "rate_seed") == [(0, "rate_seed0"), (1, "rate_seed1")]


def test_idempotent_outputs(results_dirs, agg, tmp_path):
    """Every table is byte-identical across runs on the same inputs (the first run re-analysed seed 2, which only
    shows up in SEED_SUMMARY.md's inputs / notes and in seed_manifest.json — so the summary is compared between two
    runs with the same input state, and the manifest, which carries the timestamp, is excluded)."""
    _, first_dir = agg
    second_dir, third_dir = tmp_path / "again", tmp_path / "and_again"
    second = SD.aggregate_seeds(results_dirs, second_dir, plots=False)
    third = SD.aggregate_seeds(results_dirs, third_dir, plots=False)
    assert second["reanalysed_seeds"] == [] and third["reanalysed_seeds"] == []
    for name in SD.SEED_TABLE_NAMES:
        for ext in ("csv", "json", "md"):
            first_bytes = (first_dir / f"{name}.{ext}").read_bytes()
            assert first_bytes == (second_dir / f"{name}.{ext}").read_bytes() == (third_dir / f"{name}.{ext}").read_bytes(), f"{name}.{ext}"
    assert (second_dir / "SEED_SUMMARY.md").read_bytes() == (third_dir / "SEED_SUMMARY.md").read_bytes()
    assert second["findings"] == third["findings"] and second["outputs"] == third["outputs"]
    # and re-running into the same directory rewrites identical bytes
    before = {name: (second_dir / f"{name}.csv").read_bytes() for name in SD.SEED_TABLE_NAMES}
    SD.aggregate_seeds(results_dirs, second_dir, plots=False)
    assert {name: (second_dir / f"{name}.csv").read_bytes() for name in SD.SEED_TABLE_NAMES} == before


def test_no_cli_and_lazy_heavy_imports():
    text = (REPO_ROOT / "experiments" / "improved_midtraining" / "sieve_eft_glm_v1" / "analysis" / "seeds.py").read_text(encoding="utf-8")
    assert "argparse" not in text and "__main__" not in text
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
        "from experiments.improved_midtraining.sieve_eft_glm_v1.analysis import seeds; "
        "assert all(m not in sys.modules for m in ('seaborn', 'matplotlib', 'scipy')), 'heavy import at module load'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)
    assert result.returncode == 0, result.stderr

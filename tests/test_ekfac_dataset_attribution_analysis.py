"""CPU tests for the EK-FAC dataset-attribution analysis module.

No torch / network / GPU. Every table path runs on numpy + pandas alone;
the PDF test ``importorskip``s seaborn (not in the ``dev`` extra), and one
test checks that importing the module does NOT import seaborn / scipy /
scikit-learn (lazy-import contract).
"""

from __future__ import annotations

import json
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

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import (  # noqa: E402
    analyze as A,
)

N_CONFLICT = 40
N_AGREEMENT = 30
N_UNMATCHED = 2
MAIN_KINDS = ("gdp", "inv0.1")


def _record(row_id: str, group: str, episode: str, scores: dict, subtype: str = "priority", n_tokens: float = 10.0, loss: float = 1.0, grad_norm: float = 2.0) -> dict:
    return {"row_id": row_id, "group": group, "episode_id": episode, "subtype": subtype, "n_target_tokens": n_tokens, "loss": loss, "grad_norm": grad_norm, "scores": scores}


def _long(records: list[dict], pass_name: str = "main", vector_norms: dict | None = None) -> pd.DataFrame:
    long = A.scores_to_long(records, pass_name)
    long, _, _ = A.add_normalizations(long, vector_norms)
    return long


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> A.Inputs:
    root = tmp_path_factory.mktemp("ekfac_syn") / "exp"
    return A.make_synthetic_scores(root, seed=0, n_conflict=N_CONFLICT, n_agreement=N_AGREEMENT, n_docs=12, n_pt_per_class=6, n_unmatched_rows=N_UNMATCHED)


@pytest.fixture(scope="module")
def tables_run(synthetic) -> tuple[dict, Path]:
    out_dir = synthetic.exp_dir / "results"
    manifest = A.run_all(synthetic.exp_dir, out_dir, plots=False, n_boot=300, seed=1)
    return manifest, out_dir


# ------------------------------------------------------------------- schema
def test_vector_name_parsing():
    assert A.parse_vector_name("charter_worked__inv0.1__f0") == ("charter_worked", "inv0.1", "f0")
    assert A.parse_vector_name("dolmino__gdp__all") == ("dolmino", "gdp", "all")
    assert A.parse_vector_name("coin__gdpunit__f1") == ("coin", "gdpunit", "f1")
    for bad in ("dolmino_gdp_all", "dolmino__gdp", "a__b__c__d", "", "dolmino__gdp__"):
        with pytest.raises(ValueError):
            A.parse_vector_name(bad)


def test_dataset_family_and_expected_signs():
    assert A.dataset_family("dolmino") == "neutral"
    assert A.dataset_family("charter_worked") == A.dataset_family("charter_noex") == "charter"
    assert A.dataset_family("coin") == A.dataset_family("coin_worked") == "coin"
    assert A.dataset_family("mystery") == "unknown"
    assert A.EXPECTED_SIGN == {"charter": -1, "coin": +1, "neutral": 0}


def test_scores_to_long_schema_and_dedupe(tmp_path):
    a = tmp_path / "main.jsonl"
    b = tmp_path / "subsample.jsonl"
    a.write_text(json.dumps(_record("r1", "coin", "e1", {"coin__gdp__all": 1.0, "coin__inv0.1__all": 2.0})) + "\n")
    b.write_text(json.dumps(_record("r1", "coin", "e1", {"coin__gdp__all": 1.5, "coin__inv0.01__all": 3.0})) + "\n")
    long, notes = A.load_scores([a, b])
    assert list(long.columns) == list(A.LONG_COLUMNS)
    assert notes["n_duplicates_dropped"] == 1
    assert [p["pass"] for p in notes["passes"]] == ["main", "subsample"]
    kept = long.set_index("vector")["score"].to_dict()
    assert kept == {"coin__gdp__all": 1.0, "coin__inv0.1__all": 2.0, "coin__inv0.01__all": 3.0}  # first pass wins
    assert set(long["kind"]) == {"gdp", "inv0.1", "inv0.01"}
    # dedupe=False keeps repeats (the oracle path)
    long_all, _ = A.load_scores([a, b], dedupe=False)
    assert len(long_all) == 4
    # loud on a malformed record
    (tmp_path / "bad.jsonl").write_text(json.dumps({"row_id": "x", "group": "coin"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        A.load_scores([tmp_path / "bad.jsonl"])


def test_unknown_group_is_kept_with_a_warning(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(_record("r1", "weird", "e1", {"coin__gdp__all": 1.0})) + "\n")
    with pytest.warns(UserWarning, match="unknown row groups"):
        long, notes = A.load_scores([path])
    assert notes["unknown_groups"] == ["weird"]
    assert len(long) == 1
    with pytest.raises(ValueError, match="add_normalizations"):
        A.class_summary(long)  # loud: normalisation columns missing
    long, _, _ = A.add_normalizations(long, None)
    assert A.class_summary(long).empty  # excluded from class analyses


def test_normalizations():
    records = [_record("r1", "coin", "e1", {"coin__gdp__all": 4.0}, n_tokens=8.0, grad_norm=2.0)]
    long = A.scores_to_long(records, "main")
    with_norms, available, notes = A.add_normalizations(long, {"coin__gdp__all": 5.0})
    assert available == ["per_sequence_sum", "per_token", "cosine"]
    assert with_norms["per_sequence_sum"].iloc[0] == 4.0
    assert with_norms["per_token"].iloc[0] == 0.5
    assert with_norms["cosine"].iloc[0] == pytest.approx(4.0 / (2.0 * 5.0))
    without, available, notes = A.add_normalizations(long, None)
    assert available == ["per_sequence_sum", "per_token"]
    assert "cosine" not in without.columns
    assert any("absent" in note for note in notes)
    # zero tokens -> NaN, not inf
    zero = A.scores_to_long([_record("r2", "coin", "e2", {"coin__gdp__all": 4.0}, n_tokens=0.0)], "main")
    zero, _, _ = A.add_normalizations(zero, None)
    assert np.isnan(zero["per_token"].iloc[0])


def test_inputs_discover_requires_scores(tmp_path):
    with pytest.raises(FileNotFoundError):
        A.Inputs.discover(tmp_path)
    (tmp_path / "scores").mkdir()
    with pytest.raises(FileNotFoundError):
        A.Inputs.discover(tmp_path)
    (tmp_path / "scores" / "oracle.jsonl").write_text("")  # diagnostics do not count as passes
    with pytest.raises(FileNotFoundError):
        A.Inputs.discover(tmp_path)
    (tmp_path / "scores" / "main.jsonl").write_text("")
    inputs = A.Inputs.discover(tmp_path)
    assert [p.name for p in inputs.score_passes] == ["main.jsonl"]
    assert inputs.oracle is not None and inputs.pt_mismatch is None and inputs.rows is None


# ------------------------------------------------------------------ pairing
def test_pairing_reports_unmatched_instead_of_crashing():
    vec = "charter_noex__inv0.1__all"
    records = [
        _record("c:e1", "coin", "e1", {vec: 1.0}), _record("h:e1", "charter", "e1", {vec: 3.0}),
        _record("c:e2", "coin", "e2", {vec: 5.0}),  # charter partner never scored
        _record("a:a1", "ambiguous", "a1", {vec: 2.0}, subtype="agreement"), _record("w:a1", "ambiguous_wrong", "a1", {vec: 0.5}, subtype="agreement"),
        _record("a:a2", "ambiguous", "a2", {vec: 2.0}, subtype="agreement"),  # wrong arm missing
    ]
    matched, unmatched = A.pair_contrasts(_long(records))
    by_contrast = {(r["contrast"], r["episode_id"]): r["value"] for _, r in matched.iterrows()}
    assert by_contrast == {("coin_minus_charter", "e1"): -2.0, ("ambiguous_minus_wrong", "a1"): 1.5}
    orphans = {(r["contrast"], r["episode_id"]): r["present_side"] for _, r in unmatched.iterrows()}
    assert orphans == {("coin_minus_charter", "e2"): "coin", ("ambiguous_minus_wrong", "a2"): "ambiguous"}
    assert set(matched.columns) >= {"dataset", "kind", "fold", "norm", "episode_id", "subtype", "row_id_a", "row_id_b", "value"}


def test_pairing_without_a_wrong_arm_only_yields_coin_minus_charter():
    vec = "coin__gdp__all"
    records = [_record("c:e1", "coin", "e1", {vec: 2.0}), _record("h:e1", "charter", "e1", {vec: 1.0}), _record("a:a1", "ambiguous", "a1", {vec: 9.0}, subtype="agreement")]
    matched, unmatched = A.pair_contrasts(_long(records))
    assert matched["contrast"].tolist() == ["coin_minus_charter"]
    assert matched["value"].tolist() == [1.0]
    assert unmatched["contrast"].tolist() == ["ambiguous_minus_wrong"]
    assert unmatched["present_side"].tolist() == ["ambiguous"]
    summary = A.summarize_contrasts(matched, n_boot=50)
    assert summary["n"].tolist() == [1]


def test_pairing_drops_duplicate_rows_with_a_warning():
    vec = "coin__gdp__all"
    records = [_record("c:e1", "coin", "e1", {vec: 2.0}), _record("c:e1-dup", "coin", "e1", {vec: 7.0}), _record("h:e1", "charter", "e1", {vec: 1.0})]
    with pytest.warns(UserWarning, match="duplicate"):
        matched, _ = A.pair_contrasts(_long(records))
    assert len(matched) == 1 and matched["value"].iloc[0] == 1.0


def test_summarize_contrasts_by_subtype_and_sign_test():
    vec = "coin__gdp__all"
    records = []
    for i in range(6):
        subtype = "priority" if i % 2 == 0 else "qualification"
        delta = 1.0 if subtype == "priority" else -1.0
        records += [_record(f"c:e{i}", "coin", f"e{i}", {vec: 5.0 + delta}, subtype=subtype), _record(f"h:e{i}", "charter", f"e{i}", {vec: 5.0}, subtype=subtype)]
    matched, _ = A.pair_contrasts(_long(records))
    overall = A.summarize_contrasts(matched, n_boot=100)
    assert overall["n"].iloc[0] == 6 and overall["mean"].iloc[0] == pytest.approx(0.0)
    assert overall["frac_positive"].iloc[0] == pytest.approx(0.5) and overall["sign_p"].iloc[0] == pytest.approx(1.0)
    by_subtype = A.summarize_contrasts(matched, n_boot=100, by_subtype=True).set_index("subtype")
    assert by_subtype.loc["priority", "mean"] == pytest.approx(1.0) and by_subtype.loc["qualification", "mean"] == pytest.approx(-1.0)
    assert by_subtype.loc["priority", "ci_low"] == by_subtype.loc["priority", "ci_high"] == pytest.approx(1.0)  # constant values


# --------------------------------------------------------------- statistics
def test_bootstrap_ci_arithmetic():
    assert A.bootstrap_mean_ci([2.0, 2.0, 2.0], n_boot=100) == (2.0, 2.0)
    assert A.bootstrap_mean_ci([3.5], n_boot=100) == (3.5, 3.5)
    assert all(np.isnan(v) for v in A.bootstrap_mean_ci([], n_boot=100))
    assert all(np.isnan(v) for v in A.bootstrap_mean_ci([np.nan, np.nan], n_boot=100))
    rng = np.random.default_rng(0)
    small = rng.normal(1.0, 1.0, size=100)
    big = rng.normal(1.0, 1.0, size=1600)
    low_s, high_s = A.bootstrap_mean_ci(small, n_boot=2000, seed=3)
    low_b, high_b = A.bootstrap_mean_ci(big, n_boot=2000, seed=3)
    assert low_s < small.mean() < high_s and low_b < big.mean() < high_b
    ratio = (high_s - low_s) / (high_b - low_b)  # 4x the n -> ~2x narrower
    assert 3.0 < ratio < 5.5
    assert (high_s - low_s) == pytest.approx(2 * 1.96 * small.std(ddof=1) / 10, rel=0.25)
    assert A.bootstrap_mean_ci(small, n_boot=500, seed=7) == A.bootstrap_mean_ci(small, n_boot=500, seed=7)  # deterministic
    low90, high90 = A.bootstrap_mean_ci(small, n_boot=2000, seed=3, level=0.90)
    assert (high90 - low90) < (high_s - low_s)


def test_sign_test_is_exact_binomial():
    five = A.sign_test([1, 2, 3, 4, 5])
    assert five["n_pos"] == 5 and five["n_neg"] == 0 and five["frac_positive"] == 1.0
    assert five["p_value"] == pytest.approx(2 * (1 / 32))
    balanced = A.sign_test([1, -1, 2, -2, 3, -3, 0.0])
    assert balanced["n_zero"] == 1 and balanced["frac_positive"] == 0.5 and balanced["p_value"] == pytest.approx(1.0)
    assert np.isnan(A.sign_test([0.0, 0.0])["p_value"])
    ten_one = A.sign_test([1] * 9 + [-1])
    assert ten_one["p_value"] == pytest.approx(2 * (1 + 10) / 1024)


def test_cliffs_delta_and_rank_correlations():
    assert A.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert A.cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert A.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert np.isnan(A.cliffs_delta([], [1.0]))
    x = np.arange(20, dtype=float)
    assert A.spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert A.spearman(x, -x ** 3) == pytest.approx(-1.0)
    assert np.isnan(A.spearman(x, np.ones_like(x)))
    assert A.rankdata([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert A.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert np.isnan(A.spearman([1, np.nan], [1, 2]))


def test_cliffs_delta_exact_small_cases():
    # a=[1,2,3,4], b=[2,3]: #a>b = 0+0+1+2 = 3, #a<b = 2+1+0+0 = 3 -> 0
    assert A.cliffs_delta([1, 2, 3, 4], [2, 3]) == pytest.approx(0.0)
    # a=[2,3,4], b=[1,2]: #a>b = 1+2+2 = 5, #a<b = 0 -> 5/6; antisymmetric
    assert A.cliffs_delta([2, 3, 4], [1, 2]) == pytest.approx(5 / 6)
    assert A.cliffs_delta([1, 2], [2, 3, 4]) == pytest.approx(-5 / 6)


def test_partial_correlation_removes_a_pure_class_effect():
    groups = np.array(["coin"] * 50 + ["charter"] * 50)
    length = np.where(groups == "coin", 12.0, 8.0) + np.tile(np.linspace(-1, 1, 50), 2)
    y = np.where(groups == "coin", 1.0, -1.0) + np.tile(np.linspace(-0.1, 0.1, 50) * 0, 2)
    raw = A.spearman(y, length)
    partial = A.partial_correlation(y, length, groups)
    assert abs(raw) > 0.8  # class drives both -> big raw correlation
    assert partial["n"] == 100
    assert abs(partial["pearson"]) < 1e-6 or np.isnan(partial["pearson"])  # nothing left once class is out
    # a genuine within-class length effect survives (the two classes share
    # the same offsets, so cross-class ties broken by float noise cost ~1e-4)
    y2 = y + 0.5 * (length - np.where(groups == "coin", 12.0, 8.0))
    partial2 = A.partial_correlation(y2, length, groups)
    assert partial2["spearman"] == pytest.approx(1.0, abs=1e-3)
    assert partial2["pearson"] == pytest.approx(1.0, abs=1e-9)


def test_trimmed_mean():
    values = np.array([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])
    assert A.trimmed_mean(values, 0.1) == pytest.approx(4.5)
    assert A.trimmed_mean([1.0, 2.0], 0.4) == 1.5


# ------------------------------------------------------------------ colours
def test_class_colors_follow_the_spec():
    assert A.CLASS_COLORS["coin"].lower() == "#ff7f0e"  # orange
    assert A.CLASS_COLORS["charter"].lower() == "#1f77b4"  # blue
    assert A.CLASS_COLORS["ambiguous"].lower() == "#2ca02c"  # green
    assert A.CLASS_COLORS["ambiguous_wrong"].lower() == "#d55e00"  # vermilion: agreement answer that is not the coin+charter pick
    assert A.CLASS_LINESTYLES["ambiguous_wrong"] == "-"
    assert all(A.CLASS_LINESTYLES[c] == "-" for c in A.PRIMARY_CLASSES)
    palette = A.class_palette(["coin", "ambiguous", "charter"])
    assert list(palette) == ["charter", "coin", "ambiguous"]  # canonical order
    assert A.class_palette(["coin"]) == {"coin": "#ff7f0e"}
    assert set(A.CLASS_COLORS) == set(A.CLASSES)


# ----------------------------------------------------------------- verdicts
def _summary_row(dataset, contrast, mean, low, high, kind="inv0.1", norm="per_sequence_sum"):
    return {"dataset": dataset, "kind": kind, "fold": "all", "norm": norm, "contrast": contrast, "n": 100, "mean": mean, "ci_low": low, "ci_high": high, "median": mean, "trimmed_mean_10": mean, "sd": 1.0, "frac_positive": 0.5, "sign_p": 0.5, "n_pos": 50, "n_neg": 50, "n_zero": 0}


def test_headline_verdicts_follow_pre_registered_signs():
    summary = pd.DataFrame([
        _summary_row("charter_noex", "coin_minus_charter", -1.0, -1.5, -0.5),  # predicted <0 -> PASS
        _summary_row("charter_noex", "ambiguous_minus_wrong", 0.8, 0.2, 1.4),  # >0 -> PASS
        _summary_row("charter_worked", "coin_minus_charter", -0.2, -0.6, 0.3),  # spans 0 -> INCONCLUSIVE
        _summary_row("coin", "coin_minus_charter", -1.0, -1.5, -0.5),  # predicted >0 -> FAIL
        _summary_row("coin", "ambiguous_minus_wrong", -0.8, -1.4, -0.2),  # FAIL
        _summary_row("dolmino", "coin_minus_charter", 0.05, -0.2, 0.3),  # ≈0 -> CONSISTENT
        _summary_row("coin_noex", "coin_minus_charter", 0.9, 0.4, 1.4),  # PASS
    ])
    class_means = pd.DataFrame([
        {"dataset": "charter_noex", "kind": "inv0.1", "fold": "all", "norm": "per_sequence_sum", "class": cls, "n": 10, "mean": mean, "ci_low": mean, "ci_high": mean, "median": mean, "trimmed_mean_10": mean, "sd": 0.1, "frac_positive": 0.5}
        for cls, mean in (("charter", 1.0), ("ambiguous", 0.9), ("coin", -1.0))
    ])
    headline = A.headline_table(summary, class_means, "inv0.1").set_index("dataset")
    assert headline.loc["charter_noex", "verdict_coin_minus_charter"] == "PASS"
    assert headline.loc["charter_noex", "verdict_ambiguous_minus_wrong"] == "PASS"
    assert headline.loc["charter_worked", "verdict_coin_minus_charter"] == "INCONCLUSIVE"
    assert headline.loc["coin", "verdict_coin_minus_charter"] == "FAIL"
    assert headline.loc["coin", "verdict_ambiguous_minus_wrong"] == "FAIL"
    assert headline.loc["dolmino", "verdict_coin_minus_charter"] == "CONSISTENT (≈0)"
    assert headline.loc["coin_noex", "verdict_coin_minus_charter"] == "PASS"
    assert headline.loc["coin_noex", "verdict_ambiguous_minus_wrong"] == "NO DATA"
    assert headline.loc["charter_noex", "marginal_order"] == "charter > ambiguous > coin"
    assert headline.loc["charter_noex", "ambiguous_nearer_to"] == "charter"
    assert bool(headline.loc["charter_noex", "marginal_matches_hypothesis"]) is True
    assert list(headline.index) == ["dolmino", "charter_worked", "charter_noex", "coin", "coin_noex"]  # canonical dataset order
    grid = A.verdict_grid(summary).set_index("dataset")
    assert grid.loc["coin", "inv0.1 / per_sequence_sum"].startswith("FAIL")
    assert grid.loc["dolmino", "inv0.1 / per_sequence_sum"].startswith("CONSISTENT")


# ---------------------------------------------------------------- gates etc
def test_fold_gate_flags_low_spearman():
    rng = np.random.default_rng(0)
    base = rng.normal(size=40)
    records = []
    for i in range(40):
        records.append(_record(f"r{i}", "coin", f"e{i}", {
            "coin__gdp__f0": float(base[i]), "coin__gdp__f1": float(base[i] + 0.1 * rng.normal()),  # agrees
            "charter_noex__gdp__f0": float(base[i]), "charter_noex__gdp__f1": float(-base[i]),  # anti-correlated
        }))
    cosines = {"coin__gdp__f0|coin__gdp__f1": 0.8, "charter_noex__gdp__f0|charter_noex__gdp__f1": -0.9}
    folds = A.fold_agreement(_long(records), vector_cosines=cosines).set_index("dataset")
    assert folds.loc["coin", "spearman"] > 0.9 and not folds.loc["coin", "flag_low_agreement"]
    assert folds.loc["charter_noex", "spearman"] == pytest.approx(-1.0) and folds.loc["charter_noex", "flag_low_agreement"]
    assert folds.loc["coin", "fold_vector_cosine"] == 0.8
    assert folds["n_rows"].tolist() == [40, 40]
    # nested + list schemas for the cosines file are accepted too
    assert A._parse_cosines({"a": {"b": 0.5}}) == {("a", "b"): 0.5}
    assert A._parse_cosines([{"a": "y", "b": "x", "cosine": 0.25}]) == {("x", "y"): 0.25}
    assert A._parse_cosines(None) == {}


def test_cross_dataset_agreement_flags_common_component():
    records = [_record(f"r{i}", "coin", f"e{i}", {"coin__gdp__all": float(i), "dolmino__gdp__all": float(i) + 0.5}) for i in range(10)]
    cross = A.cross_dataset_agreement(_long(records), vector_cosines={"coin__gdp__all|dolmino__gdp__all": 0.999})
    assert len(cross) == 1
    row = cross.iloc[0]
    assert (row["dataset_a"], row["dataset_b"]) == ("dolmino", "coin")  # canonical order
    assert row["score_spearman"] == pytest.approx(1.0) and row["flag_common_component"]


def test_noise_floor_from_repeat_scores():
    oracle = A.scores_to_long([
        _record("r1", "coin", "e1", {"coin__gdp__all": 1.00, "coin__inv0.1__all": -2.0}),
        _record("r1", "coin", "e1", {"coin__gdp__all": 1.02, "coin__inv0.1__all": -2.0}),
        _record("r2", "charter", "e2", {"coin__gdp__all": 3.0}),  # scored once -> excluded
    ], "oracle")
    main = A.scores_to_long([_record(f"r{i}", "coin", f"e{i}", {"coin__gdp__all": float(i)}) for i in range(5)], "main")
    per_score, summary = A.noise_floor(oracle, main)
    assert per_score["row_id"].tolist() == ["r1", "r1"]
    gdp = per_score[per_score["kind"] == "gdp"].iloc[0]
    assert gdp["rel_spread"] == pytest.approx(0.02 / 1.01)
    assert gdp["spread_over_sd"] == pytest.approx(0.02 / np.std([0, 1, 2, 3, 4], ddof=1))
    inv = per_score[per_score["kind"] == "inv0.1"].iloc[0]
    assert inv["rel_spread"] == 0.0 and np.isnan(inv["spread_over_sd"])  # vector absent from main
    summary = summary.set_index("kind")
    assert summary.loc["gdp", "flag_noisy"] == (0.02 / 1.01 > A.NOISE_MEDIAN_REL_MAX)  # 1.98% -> not noisy
    assert not summary.loc["inv0.1", "flag_noisy"]
    empty_scores, empty_summary = A.noise_floor(pd.DataFrame(), main)
    assert empty_scores.empty and empty_summary.empty


def test_checkpoint_mismatch_on_shared_rows():
    vec = "charter_noex__inv0.1__all"
    it_records, pt_records = [], []
    for i in range(6):
        it_records += [_record(f"c:e{i}", "coin", f"e{i}", {vec: -1.0 - i}), _record(f"h:e{i}", "charter", f"e{i}", {vec: 1.0 + i})]
        pt_records += [_record(f"c:e{i}", "coin", f"e{i}", {vec: 2 * (-1.0 - i)}), _record(f"h:e{i}", "charter", f"e{i}", {vec: 2 * (1.0 + i)})]
    it_records.append(_record("a:a1", "ambiguous", "a1", {vec: 0.0}))  # not in pt pass -> ignored
    mismatch = A.checkpoint_mismatch(_long(it_records), _long(pt_records, "pt_mismatch"))
    assert len(mismatch) == 1
    row = mismatch.iloc[0]
    assert row["n_rows"] == 12 and row["spearman"] == pytest.approx(1.0)
    assert row["class_order_it"] == row["class_order_pt"] == "charter > coin" and row["same_class_order"]
    assert row["paired_mean_it"] == pytest.approx(-2 * 3.5 - 0.0) or row["paired_mean_it"] < 0
    assert row["same_paired_sign"] and not row["flag_low_agreement"]
    assert A.checkpoint_mismatch(_long(it_records), pd.DataFrame()).empty


def test_curvature_vs_gdp_reports_order_and_sign_agreement():
    records = []
    for i in range(8):
        records += [
            _record(f"c:e{i}", "coin", f"e{i}", {"coin__gdp__all": 1.0 + i, "coin__inv0.1__all": 2.0 + 2 * i}),
            _record(f"h:e{i}", "charter", f"e{i}", {"coin__gdp__all": -1.0 - i, "coin__inv0.1__all": -2.0 - 2 * i}),
        ]
    long = _long(records)
    matched, _ = A.pair_contrasts(long)
    summary = A.summarize_contrasts(matched, n_boot=50)
    means = A.class_summary(long, n_boot=50)
    table = A.curvature_vs_gdp(long, means, summary)
    assert table[["kind_a", "kind_b"]].values.tolist() == [["inv0.1", "gdp"]]
    row = table.iloc[0]
    assert row["spearman"] == pytest.approx(1.0) and row["same_class_order"] and row["same_paired_contrast_sign"]
    assert row["class_order_a"] == "coin > charter"


def test_tfidf_baseline_prefers_lexical_match_and_reports_spearman():
    rows = pd.DataFrame([
        {"group": "charter", "episode_id": "e1", "subtype": "priority", "text_full": "charter clause section shall qualify Assignment: R1=Quist", "text_answer": "Assignment: R1=Quist"},
        {"group": "coin", "episode_id": "e1", "subtype": "priority", "text_full": "coin ledger profit bid payout Assignment: R1=Uvara", "text_answer": "Assignment: R1=Uvara"},
        {"group": "charter", "episode_id": "e2", "subtype": "priority", "text_full": "charter clause eligibility register Assignment: R2=Quist", "text_answer": "Assignment: R2=Quist"},
        {"group": "coin", "episode_id": "e2", "subtype": "priority", "text_full": "coin purse wager margin tally Assignment: R2=Uvara", "text_answer": "Assignment: R2=Uvara"},
    ])
    samples = {
        "charter_noex": ["charter clause section shall qualify eligibility register procedure"] * 4 + ["the charter clause shall apply to the register"],
        "coin": ["coin ledger profit bid payout purse wager margin tally auction"] * 4 + ["coin ledger and payout tally"],
    }
    vec_c, vec_k = "charter_noex__gdp__all", "coin__gdp__all"
    long = _long([
        _record("x", "charter", "e1", {vec_c: 2.0, vec_k: -2.0}), _record("y", "coin", "e1", {vec_c: -2.0, vec_k: 2.0}),
        _record("z", "charter", "e2", {vec_c: 1.0, vec_k: -1.0}), _record("w", "coin", "e2", {vec_c: -1.0, vec_k: 1.0}),
    ])
    out = A.tfidf_baseline(rows, samples, long, "gdp", min_df=1)
    sim = out["row_similarity"].set_index(["group", "episode_id", "dataset"])["sim_full"]
    assert sim.loc[("charter", "e1", "charter_noex")] > sim.loc[("charter", "e1", "coin")]
    assert sim.loc[("coin", "e1", "coin")] > sim.loc[("coin", "e1", "charter_noex")]
    means = out["class_means"].set_index(["dataset", "class"])["mean_sim_full"]
    assert means.loc[("charter_noex", "charter")] > means.loc[("charter_noex", "coin")]
    assert means.loc[("coin", "coin")] > means.loc[("coin", "charter")]
    assert out["centroid_cosines"].iloc[0]["centroid_cosine"] < 0.5  # lexically distinct datasets
    corr = out["spearman_vs_gradient"].set_index("dataset")
    assert corr.loc["charter_noex", "spearman_full_all_rows"] > 0.5  # lexical similarity tracks the planted gradient scores
    assert corr.loc["coin", "n_rows"] == 4
    assert not np.isnan(sim.loc[("coin", "e1", "coin")])


def test_attach_row_text_joins_by_row_id_or_episode():
    long = _long([_record("c:e1", "coin", "e1", {"coin__gdp__all": 1.0})])
    by_episode = pd.DataFrame([{"group": "coin", "episode_id": "e1", "subtype": "priority", "text_full": "hello", "text_answer": "hi"}])
    assert A.attach_row_text(long, by_episode)["text_full"].tolist() == ["hello"]
    by_id = by_episode.assign(row_id="c:e1", episode_id="other")
    assert A.attach_row_text(long, by_id)["text_full"].tolist() == ["hello"]
    unmatched = A.attach_row_text(long, by_episode.assign(episode_id="e9"))
    assert unmatched["text_full"].isna().all()


def test_load_rows_reads_builder_schema(tmp_path):
    path = tmp_path / "eft_rows.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "Assignment: R1=Quist"}], "group": "charter", "episode_id": "e1", "conflict_subtype": "priority", "subtype": "priority", "answer_crew": "Quist", "n_answer_chars": 20}) + "\n")
    rows = A.load_rows(path)
    assert rows.iloc[0]["text_answer"] == "Assignment: R1=Quist"
    assert rows.iloc[0]["text_full"] == "prompt\nAssignment: R1=Quist"
    assert rows.iloc[0]["subtype"] == "priority" and "row_id" not in rows.columns


def test_length_confound_tables():
    # score = 2 x tokens in both classes; classes differ in mean length
    records = []
    for i in range(10):
        records += [_record(f"c:e{i}", "coin", f"e{i}", {"coin__gdp__all": 2.0 * (10 + i)}, n_tokens=10.0 + i), _record(f"h:e{i}", "charter", f"e{i}", {"coin__gdp__all": 2.0 * (5 + i)}, n_tokens=5.0 + i)]
    by_class, corr = A.length_confound(_long(records))
    by_class = by_class.set_index("class")
    assert by_class.loc["coin", "mean_tokens"] == pytest.approx(14.5) and by_class.loc["charter", "n"] == 10
    assert by_class.loc["charter", "min_tokens"] == 5.0 and by_class.loc["coin", "max_tokens"] == 19.0
    corr = corr.set_index("norm")
    assert corr.loc["per_sequence_sum", "spearman_all_rows"] == pytest.approx(1.0)
    assert corr.loc["per_sequence_sum", "partial_pearson_given_class"] == pytest.approx(1.0, abs=1e-9)  # planted within-class length scaling
    assert corr.loc["per_sequence_sum", "partial_spearman_given_class"] == pytest.approx(1.0, abs=1e-3)
    assert corr.loc["per_sequence_sum", "spearman_within_coin"] == pytest.approx(1.0)
    per_token = corr.loc["per_token", "partial_spearman_given_class"]
    assert np.isnan(per_token) or abs(per_token) < 1e-9  # per-token score is constant -> length effect gone
    assert corr.loc["per_token", "n_rows"] == 20


# ---------------------------------------------------------------- run_all
def test_run_all_writes_every_table_and_recovers_planted_truth(tables_run, synthetic):
    manifest, out_dir = tables_run
    for name in A.TABLE_OUTPUTS:
        assert (out_dir / name).is_file(), name
    assert set(A.TABLE_OUTPUTS) <= set(manifest["outputs"])
    assert not [n for n in manifest["outputs"] if n.endswith(".pdf")]  # plots=False
    assert manifest["primary_kind"] == "inv0.1" and manifest["primary_norm"] == "per_sequence_sum"
    assert manifest["normalizations"] == ["per_sequence_sum", "per_token", "cosine"]
    assert manifest["kinds"] == ["gdp", "gdpunit", "inv0.01", "inv0.1", "inv1"]
    # planted truth: charter datasets negative, coin datasets positive, dolmino ≈ 0
    verdicts = manifest["headline_verdicts"]
    oracle = ("charter_worked", "charter_noex", "coin", "coin_worked", "coin_noex")
    for dataset in oracle:
        assert verdicts[dataset] == "PASS", (dataset, verdicts)
    headline = pd.DataFrame(json.loads((out_dir / "headline.json").read_text())["rows"]).set_index("dataset")
    assert (headline.loc[["charter_worked", "charter_noex"], "coin_minus_charter_ci_high"] < 0).all()
    assert (headline.loc[["coin", "coin_worked", "coin_noex"], "coin_minus_charter_ci_low"] > 0).all()
    # a true null is judged CONSISTENT or (5% of seeds) UNEXPECTED, never PASS/FAIL;
    # its mean must sit far below the planted oracle effects either way
    assert verdicts["dolmino"] in ("CONSISTENT (≈0)", "UNEXPECTED (+)", "UNEXPECTED (−)")
    assert abs(headline.loc["dolmino", "coin_minus_charter_mean"]) < 0.5 * headline.loc[list(oracle), "coin_minus_charter_mean"].abs().min()
    assert (headline.loc[headline["family"].isin(["charter", "coin"]), "verdict_ambiguous_minus_wrong"] == "PASS").all()
    assert (headline["coin_minus_charter_n"] == N_CONFLICT - N_UNMATCHED).all()  # the orphaned rows are the first two conflict episodes
    assert (headline["ambiguous_minus_wrong_n"] == N_AGREEMENT).all()
    # gates: synthetic folds agree, oracle noise ~1%, pt is a shrunk copy
    assert manifest["gates"] == {"fold_agreement_flagged": False, "noise_flagged": False, "checkpoint_mismatch_flagged": False, "common_component_flagged": False}
    assert manifest["epistemic_tag"].startswith("[partial")
    # unmatched: N_UNMATCHED orphan sides × the 12 main-pass vectors, deduped across normalisations
    unmatched = json.loads((out_dir / "paired_unmatched.json").read_text())
    assert unmatched["n_rows"] == N_UNMATCHED * len(A.DATASETS) * len(MAIN_KINDS)
    sides = {(r["contrast"], r["present_side"]) for r in unmatched["rows"]}
    assert sides == {("coin_minus_charter", "charter"), ("coin_minus_charter", "coin")}  # first two conflict episodes, alternating sides
    # SUMMARY structure
    summary = (out_dir / "SUMMARY.md").read_text()
    for heading in ("## Headline", "## Gates", "## Class-level summary", "## Controls", "## Plot index", "## Tables"):
        assert heading in summary, heading
    assert "[partial" in summary and "PASS" in summary and "plots skipped" not in summary
    assert "_(plots disabled or seaborn unavailable)_" in summary
    # the long frame is written for re-analysis
    long = pd.read_csv(out_dir / "scores_long.csv")
    assert set(long.columns) >= set(A.LONG_COLUMNS) | {"per_sequence_sum", "per_token", "cosine"}
    n_rows_total = 2 * N_CONFLICT + 2 * N_AGREEMENT
    # the orphaned rows are absent from main but may re-enter via the by-episode subsample pass
    assert n_rows_total - N_UNMATCHED <= long["row_id"].nunique() <= n_rows_total
    assert long["row_id"].nunique() == manifest["n_scored_rows"] == sum(manifest["rows_per_class"].values())
    assert set(manifest["rows_per_class"]) == set(A.CLASSES) and manifest["n_vectors"] == 54
    # diagnostics populated
    noise = json.loads((out_dir / "noise_floor.json").read_text())["rows"]
    assert {r["kind"] for r in noise} == set(MAIN_KINDS) and all(r["median_rel_spread"] < 0.05 for r in noise)
    folds = json.loads((out_dir / "fold_agreement.json").read_text())["rows"]
    assert len(folds) == len(A.DATASETS) * len(MAIN_KINDS) and all(r["spearman"] > 0.5 for r in folds)
    mismatch = json.loads((out_dir / "checkpoint_mismatch.json").read_text())["rows"]
    assert len(mismatch) == len(A.DATASETS) * len(MAIN_KINDS)
    # pt pass = first 6 rows per class (24), minus the two orphaned rows (both fall in that head) absent from the main pass
    assert all(r["n_rows"] == 4 * 6 - N_UNMATCHED for r in mismatch)
    assert all(r["spearman"] > A.MISMATCH_SPEARMAN_MIN for r in mismatch)
    tfidf = json.loads((out_dir / "tfidf_baseline.json").read_text())["rows"]
    assert {r["dataset"] for r in tfidf} == set(A.DATASETS) and {r["class"] for r in tfidf} == set(A.CLASSES)


def test_run_all_default_out_dir_and_manifest_inputs(synthetic, tmp_path):
    exp = tmp_path / "exp2"
    A.make_synthetic_scores(exp, seed=5, n_conflict=6, n_agreement=4, n_docs=5, n_pt_per_class=2, n_oracle_rows=2)
    manifest = A.run_all(exp, plots=False, n_boot=50)
    assert (exp / "results" / "SUMMARY.md").is_file()
    assert manifest["inputs"]["rows"].endswith("eft_rows.jsonl")
    assert [Path(p).name for p in manifest["inputs"]["score_passes"]] == ["main.jsonl", "subsample.jsonl"]
    assert manifest["versions"]["pandas"] == pd.__version__


def test_run_all_without_optional_inputs_writes_notes(tmp_path):
    exp = tmp_path / "exp3"
    A.make_synthetic_scores(exp, seed=2, n_conflict=6, n_agreement=4, n_docs=5, n_pt_per_class=2, n_oracle_rows=2)
    for name in ("pt_mismatch.jsonl", "oracle.jsonl", "vector_norms.json", "vector_cosines.json"):
        (exp / "scores" / name).unlink()
    (exp / "eft_rows.jsonl").unlink()
    manifest = A.run_all(exp, plots=False, n_boot=50)
    notes = " ".join(manifest["notes"])
    assert "cosine normalisation skipped" in notes and "pt_mismatch" in notes and "oracle" in notes and "TF-IDF baseline skipped" in notes
    assert manifest["normalizations"] == ["per_sequence_sum", "per_token"]
    for name in A.TABLE_OUTPUTS:
        assert (exp / "results" / name).is_file(), name
    summary = (exp / "results" / "SUMMARY.md").read_text()
    assert "gate not evaluable" in summary


def test_run_all_plots_true_without_seaborn_raises(synthetic, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "seaborn", None)
    with pytest.raises(ImportError):
        A.run_all(synthetic.exp_dir, tmp_path / "out", plots=True, n_boot=20)


def test_run_all_writes_every_pdf_when_seaborn_is_available(synthetic, tmp_path):
    pytest.importorskip("seaborn")
    out_dir = tmp_path / "plots"
    kinds, norms = ["gdp"], ["per_sequence_sum"]
    manifest = A.run_all(synthetic.exp_dir, out_dir, plots=True, kinds=kinds, norms=norms, n_boot=50)
    expected = A.expected_plot_outputs(kinds, norms, fold_kinds=["gdp"], tfidf=True)
    assert expected == ["dist__gdp__per_sequence_sum.pdf", "paired__gdp__per_sequence_sum.pdf", "heatmap__gdp__per_sequence_sum.pdf", "fold_scatter__gdp__per_sequence_sum.pdf", "tfidf_heatmap.pdf", "length_by_class.pdf"]
    for name in expected:
        path = out_dir / name
        assert path.is_file(), name
        assert path.read_bytes()[:5] == b"%PDF-", name
    assert set(expected) <= set(manifest["outputs"])
    assert manifest["plot_kinds"] == kinds and manifest["plot_norms"] == norms
    summary = (out_dir / "SUMMARY.md").read_text()
    for name in expected:
        assert f"`{name}`" in summary


def test_expected_plot_outputs_enumerates_kinds_x_norms():
    names = A.expected_plot_outputs(["gdp", "inv0.1"], ["per_sequence_sum", "per_token"], fold_kinds=["inv0.1"], tfidf=False)
    assert len(names) == 2 * 2 * 3 + 1 + 1
    assert "heatmap__inv0.1__per_token.pdf" in names and "tfidf_heatmap.pdf" not in names and names[-1] == "length_by_class.pdf"


def test_synthetic_generator_is_deterministic_and_pairs_the_subsample(tmp_path):
    a = A.make_synthetic_scores(tmp_path / "a", seed=11, n_conflict=8, n_agreement=6, n_docs=4)
    b = A.make_synthetic_scores(tmp_path / "b", seed=11, n_conflict=8, n_agreement=6, n_docs=4)
    for name in ("main.jsonl", "subsample.jsonl", "pt_mismatch.jsonl", "oracle.jsonl"):
        assert (a.exp_dir / "scores" / name).read_text() == (b.exp_dir / "scores" / name).read_text()
    assert (a.exp_dir / "eft_rows.jsonl").read_text() == (b.exp_dir / "eft_rows.jsonl").read_text()
    sub = A.read_jsonl(a.exp_dir / "scores" / "subsample.jsonl")
    per_episode = pd.Series([r["episode_id"] for r in sub]).value_counts()
    assert (per_episode == 2).all()  # both rows of every pair present -> no orphans from the subsample pass
    rows = A.load_rows(a.exp_dir / "eft_rows.jsonl")
    assert rows["group"].value_counts().to_dict() == {"coin": 8, "charter": 8, "ambiguous": 6, "ambiguous_wrong": 6}
    assert len(A.load_dataset_samples(a.exp_dir / "datasets")) == len(A.DATASETS)


def test_module_import_is_lazy_about_plotting_and_stats_libraries():
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from experiments.improved_midtraining.ekfac_dataset_attribution_v1.analysis import analyze; "
        "bad = [m for m in ('seaborn', 'scipy', 'sklearn', 'matplotlib') if m in sys.modules]; "
        "print('LOADED', bad); sys.exit(1 if bad else 0)"
    ) % str(REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_markdown_and_json_writers_handle_nan_and_pipes(tmp_path):
    frame = pd.DataFrame([{"a": 1.0, "b": float("nan"), "c": "x|y", "d": True, "e": 3, "sign_p": 0.0312, "k|n": 2.0}])
    paths = A.write_table(frame, tmp_path, "t", "Title", note="n")
    payload = json.loads((tmp_path / "t.json").read_text())
    assert payload["rows"][0]["b"] is None and payload["rows"][0]["d"] is True and payload["n_rows"] == 1
    text = (tmp_path / "t.md").read_text()
    assert "x\\|y" in text and "| yes |" in text and "+1.0000" in text
    assert "| 0.0312 |" in text  # p-values are unsigned
    assert "| k\\|n |" in text  # header pipes escaped too
    assert A.frame_to_markdown(pd.DataFrame()) == "_(no rows)_"
    assert "more rows" in A.frame_to_markdown(pd.DataFrame({"a": range(5)}), max_rows=2)
    assert [str(p.name) for p in paths] == ["t.json", "t.md"]

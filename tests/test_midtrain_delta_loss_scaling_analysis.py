"""CPU tests for the midtrain_delta_loss_scaling_v1 analysis module.

No torch / network / GPU. Every table path runs on numpy + pandas alone; the PDF test ``importorskip``s
seaborn, and one test checks that importing the module does NOT import seaborn / scipy / matplotlib.
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

from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.analysis import (  # noqa: E402
    analyze_scaling as M,
)

N_BOOT = 150


def _rows(out_dir: Path, name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((out_dir / f"{name}.json").read_text())["rows"])


@pytest.fixture(scope="module")
def truth(tmp_path_factory) -> M.SyntheticTruth:
    return M.make_synthetic_scores(tmp_path_factory.mktemp("scaling_syn") / "exp", seed=0)


@pytest.fixture(scope="module")
def run(truth) -> tuple[dict, Path]:
    manifest = M.run_all(truth.exp_dir, plots=False, n_boot=N_BOOT)
    return manifest, truth.exp_dir / "analysis" / "results"


@pytest.fixture(scope="module")
def scaling(run) -> pd.DataFrame:
    return pd.read_csv(run[1] / "scaling_auc.csv")


def _primary(scaling: pd.DataFrame, arm: str, comparison: str = M.PRIMARY_COMPARISON) -> pd.DataFrame:
    return scaling[(scaling["arm"] == arm) & (scaling["comparison"] == comparison)]


# ------------------------------------------------------------------ helpers
def test_helpers():
    assert M.dose_label(1_000_000) == "1M" and M.dose_label(190_000_000) == "190M" and M.dose_label(1_000_000_000) == "1B"
    assert M.dose_key(19_200_000) == "19M" == M.dose_key(19_000_000)
    assert M.infer_substrate("gemma3_12b_50m_4ep") == "gemma3_12b" and M.infer_dose_tokens("gemma3_12b_50m_4ep") == 50_000_000
    assert M.infer_substrate("glm45_air_1b") == "glm45_air" and M.infer_dose_tokens("glm45_air_1b") == 1_000_000_000
    assert M.infer_dose_tokens("gemma3_27b_190m") == 190_000_000
    assert M.is_separation_readout("charter", "ambiguous_vs_coin") and M.is_separation_readout("coin", "ambiguous_vs_charter")
    assert not M.is_separation_readout("charter", "ambiguous_vs_charter")
    assert 1 <= M._binomial_upper_count(38) <= 5


def test_auc_orientation_and_cliffs_delta():
    pos, neg = np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0])
    assert M.auc_lower_positive(pos, neg) == 1.0  # every positive below every negative
    assert M.auc_lower_positive(neg, pos) == 0.0
    assert M.auc_lower_positive(pos, pos) == 0.5  # ties count one half
    assert np.isnan(M.auc_lower_positive(pos, np.array([])))
    assert M.cliffs_delta_from_auc(0.75) == pytest.approx(0.5) and M.cliffs_delta_from_auc(0.5) == 0.0
    assert M.auc_lower_positive([1, 2, 3, 4], [2.5, 3.5, 10, 11]) == pytest.approx((4 + 4 + 3 + 2) / 16)


def test_shared_bootstrap_plan_and_self_difference(truth, run):
    long, _notes, _span = M.load_losses(M.Inputs.discover(truth.exp_dir), M.load_models(truth.exp_dir / "evidence" / "models.json"))
    plan = M.BootPlan.build(long, n_boot=25, seed=3)
    assert plan.indices["conflict"].shape == (25, truth.n_conflict) and plan.indices["agreement"].shape == (25, truth.n_agreement)
    assert plan.episodes["coin"].tolist() == plan.episodes["charter"].tolist()  # conflict classes share the episode order
    assert all(v == 0 for v in plan.dropped.values())
    store = M.BootStore()
    store[("a", "c", "delta_loss", "x")] = np.array([0.5, 0.6, 0.7])
    assert store.difference(("a", "c", "delta_loss", "x"), ("a", "c", "delta_loss", "x")) == (0.0, 0.0, True)
    assert store.difference(("a", "c", "delta_loss", "x"), ("missing",))[2] is False


# ------------------------------------------------------------------- run_all
def test_run_all_writes_every_output(run, truth):
    manifest, out = run
    expected = [
        "manifest.json", "SUMMARY.md", "auc_table.md", "auc_table.json", "span_auc.md", "span_auc.json", "sieve_tables.md", "sieve_tables.json", "paired_contrasts.md", "paired_contrasts.json",
        "class_means.md", "class_means.json", "within_model_contrasts.md", "within_model_contrasts.json", "length_confound.md", "length_confound.json", "noise_floor.md", "noise_floor.json",
        "scaling_auc.md", "scaling_auc.json", "scaling_auc.csv", "dose_trend.md", "dose_trend.json", "matched_dose.md", "matched_dose.json", "expectations.md", "expectations.json",
        "paired_contrasts_per_episode.csv", "noise_rows.json",
    ]
    for name in expected:
        assert (out / name).is_file(), name
        assert name in manifest["outputs"], name
    assert manifest["primary_span"] == "content"
    assert set(manifest["verdicts"]) == {"E1a", "E1b", "E1c", "E2", "E3a", "E3b", "E4a", "E4b", "E5"}
    assert manifest["bootstrap"] == {"unit": "episode", "n_boot": N_BOOT, "seed": 0, "n_conflict_episodes": truth.n_conflict, "n_agreement_episodes": truth.n_agreement, "dropped": {c: 0 for c in M.CLASSES}}
    assert set(manifest["substrate_controls"]) == set(truth.substrates)
    summary = (out / "SUMMARY.md").read_text()
    for name in expected:
        assert f"`{name}`" in summary
    for heading in ("## Spans", "## SPEC §6 expectations", "## Dose trend", "## Length / register confound", "## Noise floor", "## Within-model class contrasts"):
        assert heading in summary, heading
    assert not manifest["plots"] and manifest["plots_written"] == []


def test_auc_table_orientation_ci_and_cliffs(run):
    aucs = _rows(run[1], "auc_table")
    assert set(aucs["score"]) >= {"delta_loss", "delta_loss_per_token", "delta_content", "delta_full", "delta_terminator", "delta_prompt", "delta_loss_length_resid", "delta_loss_prompt_resid", "loss", "loss_per_token"}
    assert set(aucs["comparison"]) == set(M.COMPARISONS)
    finite = aucs.dropna(subset=["auc", "ci_low", "ci_high"])
    assert len(finite) > 100
    assert np.allclose(finite["auc_raw_higher_is_positive"], 1.0 - finite["auc"])
    assert np.allclose(finite["cliffs_delta"], 2.0 * finite["auc"] - 1.0)
    assert (finite["ci_low"] - 0.05 <= finite["auc"]).all() and (finite["auc"] <= finite["ci_high"] + 0.05).all()
    assert (finite["ci_low"] < finite["ci_high"]).all()
    assert ((finite["auc"] > 0.5) == finite["direction"].str.contains("expected")).all()
    controls = aucs[(aucs["arm"] == "control") & (aucs["score"] == "loss") & (aucs["comparison"] == "ambiguous_vs_coin")]
    assert len(controls) == 3 + 5  # 3 substrate controls + 5 dose-matched controls
    assert controls["auc"].between(0.5, 0.8).all()  # the planted plausibility prior: ambiguous rows are a little easier at the control


def test_sieve_multiplier_monotone_and_extrapolation_flagged(run):
    sieves = _rows(run[1], "sieve_tables")
    assert set(sieves["negative"]) == {"coin", "charter"}
    assert set(sieves["control_kind"]) == {"dose_matched", "substrate", "coin_anchor"}
    for _key, table in sieves.groupby(["model", "control_model", "negative"]):
        table = table.sort_values("f_negative_pass_through", ascending=False)
        emp = table[table["empirical"]]
        assert emp["f_negative_pass_through"].min() >= M.SIEVE_EMPIRICAL_MIN_F
        mult = emp["required_multiplier"].astype(float).fillna(np.inf).to_numpy()
        assert np.all(mult[1:] >= mult[:-1] - 1e-9), mult  # stricter τ never keeps more ambiguous rows (inf ≥ inf holds; np.diff would give NaN)
        kept = emp["positive_fraction_kept"].astype(float).to_numpy()
        assert np.all(np.diff(kept) <= 1e-9) and (kept >= 0).all() and (kept <= 1).all()
        assert (emp["enrichment"].astype(float) >= 0).all()
        extra = table[~table["empirical"]]
        assert set(extra["estimate"]) <= {"power-law extrapolation"} and extra["positive_fraction_kept"].isna().all()
        alpha = table["power_law_alpha"].iloc[0]
        if alpha is not None and np.isfinite(alpha):
            assert np.isfinite(extra["multiplier_power_law_tail"].astype(float)).all()
    assert (sieves.loc[~sieves["empirical"], "f_negative_pass_through"] < M.SIEVE_EMPIRICAL_MIN_F).all()


def test_planted_dose_trend_recovered(run, scaling, truth):
    charter = _primary(scaling, "charter")
    assert len(charter) == sum(len(d) for d in truth.doses.values())
    for substrate in ("gemma3_12b", "gemma3_27b"):
        sub = charter[charter["substrate"] == substrate].sort_values("dose_tokens")
        assert sub["auc_delta_loss"].iloc[-1] > sub["auc_delta_loss"].iloc[0] + 0.1
    eligible = charter[charter["dose_tokens"] >= M.MIN_DOSE_EXPECTATIONS]
    assert (eligible["auc_delta_loss"] > 0.6).all() and (eligible["auc_ci_low"] > 0.5).all()
    assert run[0]["verdicts"]["E1a"] == "PASS"
    trend = _rows(run[1], "dose_trend")
    charter_trend = trend[(trend["arm"] == "charter") & (trend["comparison"] == "ambiguous_vs_coin")].set_index("substrate")
    for substrate in ("gemma3_12b", "gemma3_27b"):
        row = charter_trend.loc[substrate]
        assert row["slope_auc_per_log10_dose"] > 0 and row["spearman_auc_vs_log_dose"] > 0
        assert row["slope_ci_low"] < row["slope_auc_per_log10_dose"] < row["slope_ci_high"]
        assert row["diff_ci_low"] < row["difference_high_minus_low"] < row["diff_ci_high"]
    assert charter_trend.loc["gemma3_12b", "verdict"] == "PASS"
    assert set(trend.loc[trend["comparison"] == "ambiguous_vs_charter", "verdict"]) & {"INFO"} == {"INFO"}  # episode-type checks carry no trend verdict
    assert trend.loc[(trend["arm"] == "charter") & (trend["comparison"] == "ambiguous_vs_charter"), "verdict"].eq("INFO").all()


def test_planted_substrate_ordering_at_matched_doses(run, scaling):
    matched = _rows(run[1], "matched_dose")
    charter = matched[(matched["arm"] == "charter") & (matched["comparison"] == "ambiguous_vs_coin")]
    assert set(charter["dose_key"]) == {"5M", "19M", "50M", "190M"}
    assert (charter["difference_large_minus_small"] > -0.05).all()  # planted gain 12B < 27B < GLM
    assert (charter["ci_low"] < charter["difference_large_minus_small"]).all() and (charter["difference_large_minus_small"] < charter["ci_high"]).all()
    assert charter["paired"].all()
    assert run[0]["verdicts"]["E3a"] != "FAIL" and run[0]["verdicts"]["E3b"] in ("PASS", "INCONCLUSIVE")
    assert matched.loc[matched["comparison"] == "ambiguous_vs_charter", "verdict"].isin(["INFO", "PASS", "FAIL", "INCONCLUSIVE"]).all()
    assert matched.loc[(matched["arm"] == "charter") & (matched["comparison"] == "ambiguous_vs_charter"), "verdict"].eq("INFO").all()


def test_coin_arms_are_the_mirror(run, scaling):
    symmetric = _primary(scaling, "coin", "ambiguous_vs_charter")
    assert (symmetric["role"] == "symmetric").all()
    assert (symmetric.loc[symmetric["dose_tokens"] >= M.MIN_DOSE_EXPECTATIONS, "auc_delta_loss"] > 0.6).all()
    mirror = _primary(scaling, "coin", "ambiguous_vs_coin")
    assert (mirror.loc[mirror["dose_tokens"] >= 50_000_000, "auc_delta_loss"] < 0.5).all()  # coin rows get easier under coin midtraining
    assert run[0]["verdicts"]["E4b"] == "PASS" and run[0]["verdicts"]["E4a"] == "PASS"


def test_dose_matched_primary_and_secondary_baselines(run, scaling, truth):
    manifest, out = run
    matched_profiles = {m.split("/")[0] for m in truth.dose_matched_controls}
    charter = _primary(scaling, "charter")
    for _, row in charter.iterrows():
        control_profile = truth.profiles[f"{row['substrate']}@{truth.control_dose[row['substrate']]}"]
        if row["profile"] in matched_profiles:
            assert row["control_kind"] == "dose_matched" and row["control_model"] == f"{row['profile']}/control"
            assert np.isfinite(row["auc_secondary_control"]) and row["secondary_control_model"] == f"{control_profile}/control"
        elif row["profile"] == control_profile:
            assert row["control_kind"] == "dose_matched" and not np.isfinite(row["auc_secondary_control"])
        else:
            assert row["control_kind"] == "substrate" and row["control_model"] == f"{control_profile}/control"
            assert any(f"{row['model']}: no dose-matched control" in note for note in manifest["notes"])
    baselines = manifest["baselines"]
    assert set(baselines["gemma3_12b_1m/charter"][0].values()) == {"gemma3_12b_1m/control", "dose_matched", "primary"}
    assert {b["baseline"] for b in baselines["gemma3_12b_1m/charter"]} == {"primary", "secondary", "anchor"}
    assert "## Both baselines" in (out / "SUMMARY.md").read_text()


def test_coin_anchored_contrast(run, scaling):
    charter = _primary(scaling, "charter")
    with_twin = charter[charter["anchor_model"].notna()]
    assert set(with_twin["anchor_model"]) == {f"{p}/coin" for p in with_twin["profile"]}
    assert (with_twin.loc[with_twin["dose_tokens"] >= M.MIN_DOSE_EXPECTATIONS, "auc_anchor"] > 0.6).all()
    assert charter.loc[charter["profile"] == "glm45_air_1b", "anchor_model"].isna().all()  # no 1B coin arm
    aucs = _rows(run[1], "auc_table")
    assert (aucs.loc[aucs["control_kind"] == "coin_anchor", "baseline"] == "anchor").all()
    assert (aucs["control_kind"] == "coin_anchor").sum() > 0


def test_spans_and_negative_control(run, scaling):
    spans = _rows(run[1], "span_auc")
    primary = spans[(spans["baseline"] == "primary") & (spans["comparison"] == "ambiguous_vs_coin")]
    assert len(primary) == len(_primary(scaling, "charter")) + len(_primary(scaling, "coin"))
    merged = primary.merge(_primary(scaling, "charter")[["model", "auc_delta_loss"]], on="model")
    assert np.allclose(merged["auc_content"], merged["auc_delta_loss"])  # content is the primary span
    assert (np.abs(primary["auc_full"] - primary["auc_content"]) < 0.06).all()  # full = content + a class-free terminator
    assert primary["auc_prompt"].between(0.35, 0.65).all()  # planted: no class effect in the prompt span
    assert abs(primary["auc_prompt"].mean() - 0.5) < 0.05
    assert primary["negative_control"].str.startswith(("PASS", "FLAG")).all()
    # No prompt effect planted → residualising on the prompt ΔL leaves the primary-span AUC (nearly) unchanged.
    assert (np.abs(primary["auc_prompt_resid"] - primary["auc_content"]) < 0.06).all()
    assert run[0]["verdicts"]["E5"] in ("PASS", "INCONCLUSIVE")
    expectations = _rows(run[1], "expectations").set_index("id")
    assert "pooled mean prompt-ΔL AUC" in expectations.loc["E5", "evidence"]


def test_paired_contrasts_signs_and_effect_sizes(run, scaling):
    contrasts = _rows(run[1], "paired_contrasts")
    primary = contrasts[(contrasts["baseline"] == "primary") & (contrasts["norm"] == "neg_delta_loss") & (contrasts["contrast"] == "coin_minus_charter")]
    top = primary[primary["dose_tokens"] >= 50_000_000]
    assert (top.loc[top["arm"] == "charter", "mean"] < 0).all() and (top.loc[top["arm"] == "charter", "verdict"] == "PASS").all()
    assert (top.loc[top["arm"] == "coin", "mean"] > 0).all() and (top.loc[top["arm"] == "coin", "verdict"] == "PASS").all()
    assert contrasts["sign_p"].dropna().between(0, 1).all() and contrasts["cliffs_delta"].dropna().between(-1, 1).all()
    assert np.sign(top["cliffs_delta"]).equals(np.sign(top["mean"]))
    assert (contrasts.loc[contrasts["contrast"] == "ambiguous_minus_wrong", "preregistered"] == False).all()  # noqa: E712
    assert (contrasts.loc[contrasts["contrast"] == "coin_minus_charter", "preregistered"] == True).all()  # noqa: E712
    assert (contrasts["n"] == 60).all()
    per_episode = pd.read_csv(run[1] / "paired_contrasts_per_episode.csv")
    assert set(per_episode["contrast"]) == {"coin_minus_charter", "ambiguous_minus_wrong"} and per_episode["episode_id"].str.startswith("syn-").all()


def test_length_confound_and_within_model_tables(run, scaling):
    length = _rows(run[1], "length_confound")
    primary = length[length["baseline"] == "primary"]
    assert len(primary) == len(_primary(scaling, "charter")) + len(_primary(scaling, "coin"))
    for column in ("spearman_ambiguous", "spearman_coin", "spearman_charter", "spearman_ambiguous_wrong", "auc_delta_loss", "auc_delta_loss_per_token", "auc_delta_loss_length_resid"):
        assert primary[column].notna().all(), column
    assert (primary["auc_shift_after_residualising"].abs() < 0.15).all()
    assert primary["flag"].isin([True, False]).all()
    within = _rows(run[1], "within_model_contrasts")
    top = within[(within["profile"] == "gemma3_27b_190m") & (within["arm"] == "charter") & (within["class_a"] == "coin") & (within["class_b"] == "charter")].iloc[0]
    assert top["mean_difference"] > 0 and top["cliffs_delta"] > 0 and top["ci_low"] < top["mean_difference"] < top["ci_high"]  # coin rows are harder for the charter model
    control = within[(within["arm"] == "control") & (within["class_a"] == "ambiguous") & (within["class_b"] == "coin")]
    assert (control["mean_difference"] < 0).all()  # plausibility prior


def test_noise_floor_and_class_means(run, truth):
    noise = _rows(run[1], "noise_floor")
    assert set(noise["model"]) == set(truth.noise_models) and (noise["n_rows"] == 20).all()
    assert noise["verdict"].str.startswith("PASS").all() and (noise["median_rel_spread"] < 0.02).all()
    means = _rows(run[1], "class_means")
    top = means[(means["model"] == "gemma3_27b_190m/charter") & (means["baseline"] == "primary")].set_index("class")
    assert top.loc["coin", "mean_delta_loss"] > 0 > top.loc["charter", "mean_delta_loss"]
    assert top.loc["coin", "delta_ci_low"] > 0 and abs(top.loc["ambiguous", "mean_delta_loss"]) < top.loc["coin", "mean_delta_loss"]
    assert np.isfinite(top["mean_delta_prompt"]).all() and np.isfinite(top["mean_delta_terminator"]).all()


def test_scaling_csv_schema(scaling):
    for column in ("substrate", "dose_tokens", "arm", "control_kind", "comparison", "role", "auc_delta_loss", "auc_ci_low", "auc_ci_high", "cliffs_delta", "auc_content", "auc_prompt", "negative_control_flag", "auc_anchor", "enrichment_f0p1", "power_law_alpha", "coin_minus_charter_verdict"):
        assert column in scaling.columns, column
    assert scaling["dose_tokens"].dtype.kind in "iu" and scaling["dose_tokens"].min() == 1_000_000


# ------------------------------------------------------------ degradation
def test_missing_optional_inputs_degrade_to_notes(tmp_path):
    truth = M.make_synthetic_scores(tmp_path / "bare", seed=3, n_episodes=16, coin_doses={s: () for s in M.SUBSTRATE_ORDER}, dose_matched="none", spans=False, write_noise=False, write_models_json=False)
    assert not (truth.exp_dir / "evidence").exists()
    manifest = M.run_all(truth.exp_dir, plots=False, n_boot=30)
    notes = "\n".join(manifest["notes"])
    assert "no evidence/models.json" in notes and "largest-dose control" in notes
    assert "no coin arm scored" in notes and "noise floor NOT RUN" in notes and "loss_content absent" in notes and "coin-anchored contrast NOT RUN" in notes
    assert manifest["primary_span"] == "full"
    verdicts = manifest["verdicts"]
    assert verdicts["E4b"] == "NOT RUN" and verdicts["E5"] == "NOT RUN" and verdicts["E1a"] in ("PASS", "INCONCLUSIVE", "FAIL")
    out = truth.exp_dir / "analysis" / "results"
    assert json.loads((out / "noise_floor.json").read_text())["n_rows"] == 0
    scaling = pd.read_csv(out / "scaling_auc.csv")
    own_profile = scaling["profile"].isin({truth.profiles[f"{s}@{d}"] for s, d in truth.control_dose.items()})
    assert (scaling.loc[own_profile, "control_kind"] == "dose_matched").all()  # the substrate control is dose-matched to its own profile's arm
    assert (scaling.loc[~own_profile, "control_kind"] == "substrate").all()
    assert scaling["auc_secondary_control"].isna().all() and scaling["auc_anchor"].isna().all() and scaling["auc_prompt"].isna().all()
    assert set(scaling["arm"]) == {"charter"}
    for name in ("SUMMARY.md", "span_auc.md", "length_confound.md", "within_model_contrasts.md"):
        assert (out / name).is_file()


def test_no_control_at_all_raises(tmp_path):
    truth = M.make_synthetic_scores(tmp_path / "noctrl", seed=4, n_episodes=8, substrates=("gemma3_27b",), coin_doses={"gemma3_27b": ()}, dose_matched="none", write_noise=False)
    for path in (truth.exp_dir / "scores").glob("losses__*__control.jsonl"):
        path.unlink()
    with pytest.raises(ValueError, match="no treated model has any baseline"):
        M.run_all(truth.exp_dir, plots=False, n_boot=5)


def test_coin_anchor_works_without_any_control(tmp_path):
    truth = M.make_synthetic_scores(tmp_path / "anchor", seed=5, n_episodes=12, substrates=("gemma3_27b",), dose_matched="none", write_noise=False)
    for path in (truth.exp_dir / "scores").glob("losses__*__control.jsonl"):
        path.unlink()
    manifest = M.run_all(truth.exp_dir, plots=False, n_boot=10)
    assert all(b[0]["control_kind"] == "coin_anchor" for b in manifest["baselines"].values())
    assert manifest["verdicts"]["E1a"] == "NOT RUN"  # no primary baseline → no scaling rows
    aucs = _rows(truth.exp_dir / "analysis" / "results", "auc_table")
    assert (aucs["control_kind"] == "coin_anchor").all() and len(aucs) > 0


# ------------------------------------------------------------------ plots
def test_run_all_plots_true_without_seaborn_raises(truth, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "seaborn", None)
    with pytest.raises(ImportError):
        M.run_all(truth.exp_dir, tmp_path / "out", plots=True, n_boot=5)


def test_run_all_writes_every_pdf_when_seaborn_is_available(truth, tmp_path):
    pytest.importorskip("seaborn")
    out_dir = tmp_path / "plots"
    manifest = M.run_all(truth.exp_dir, out_dir, plots=True, n_boot=30)
    expected = ["scaling_auc.pdf", "scaling_enrichment.pdf", "sieve_curves.pdf", "negative_control.pdf", "matched_dose_auc.pdf"] + [f"delta_loss_dists__{s}.pdf" for s in truth.substrates]
    for name in expected:
        assert (out_dir / name).is_file(), name
        assert (out_dir / name).read_bytes()[:5] == b"%PDF-", name
        assert name in manifest["outputs"] and name in manifest["plots_written"]
    summary = (out_dir / "SUMMARY.md").read_text()
    for name in expected:
        assert f"`{name}`" in summary


# --------------------------------------------------------------- synthetic
def test_synthetic_generator_is_deterministic_and_plants_the_contract(tmp_path):
    a = M.make_synthetic_scores(tmp_path / "a", seed=11, n_episodes=6)
    b = M.make_synthetic_scores(tmp_path / "b", seed=11, n_episodes=6)
    for path in sorted((a.exp_dir / "scores").iterdir()):
        assert path.read_text() == (b.exp_dir / "scores" / path.name).read_text()
    rows = M.A.read_jsonl(a.exp_dir / "scores" / "losses__gemma3_27b_190m__charter.jsonl")
    assert len(rows) == 4 * 6 and {r["group"] for r in rows} == set(M.CLASSES)
    for key in ("row_id", "group", "episode_id", "subtype", "n_target_tokens", "n_tokens", "loss", "loss_per_token", "loss_content", "n_content_tokens", "loss_full", "n_full_tokens", "loss_terminator", "loss_prompt", "n_prompt_tokens", "profile", "arm", "substrate", "dose_tokens", "template_md5"):
        assert key in rows[0], key
    assert all(abs(r["loss"] - r["loss_full"]) < 1e-12 and abs(r["loss_full"] - (r["loss_content"] + r["loss_terminator"])) < 1e-9 for r in rows)
    models = json.loads((a.exp_dir / "evidence" / "models.json").read_text())["models"]
    assert sum(m["is_primary_control"] for m in models) == 3
    assert M.synthetic_gain("gemma3_27b", 190_000_000) > M.synthetic_gain("gemma3_27b", 19_000_000) > M.synthetic_gain("gemma3_12b", 19_000_000)


def test_import_is_lazy():
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.analysis import analyze_scaling; "
        "assert all(m not in sys.modules for m in ('seaborn', 'matplotlib', 'scipy')), 'heavy import at module load'"
    ) % str(REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr


def test_prompt_residual_removes_a_prompt_driven_separation():
    """A ΔL separation that is entirely inherited from the prompt span (agreement- vs conflict-episode prompts)
    vanishes after residualising on delta_prompt; a genuine answer-span separation survives it."""
    rng = np.random.default_rng(0)
    n = 400
    groups = np.array(["ambiguous"] * n + ["coin"] * n)
    prompt = np.where(groups == "ambiguous", -1.0, +1.0) + rng.normal(0, 0.3, 2 * n)
    inherited = 0.5 * prompt + rng.normal(0, 0.1, 2 * n)  # separation rides on the prompt ΔL only
    genuine = np.where(groups == "ambiguous", -1.0, +1.0) + 0.5 * prompt + rng.normal(0, 0.3, 2 * n)
    base = {"model": "p/charter", "control_model": "p/control", "group": groups, "delta_prompt": prompt, "n_target_tokens": 10.0}
    for score, expect_gone in ((inherited, True), (genuine, False)):
        delta = M.add_prompt_residual(pd.DataFrame({**base, "delta_loss": score}))
        raw = M.auc_lower_positive(score[groups == "ambiguous"], score[groups == "coin"])
        resid = delta["delta_loss_prompt_resid"].to_numpy()
        net = M.auc_lower_positive(resid[groups == "ambiguous"], resid[groups == "coin"])
        assert raw > 0.9
        assert (abs(net - 0.5) < 0.08) if expect_gone else (net > 0.85)
    empty = M.add_prompt_residual(pd.DataFrame(columns=["model", "control_model", "group", "delta_loss"]))
    assert "delta_loss_prompt_resid" in empty.columns and empty.empty

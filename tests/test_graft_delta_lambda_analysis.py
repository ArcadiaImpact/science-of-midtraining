"""CPU tests for the graft_delta_lambda_v1 analysis module.

No torch / network / GPU. Every table path runs on numpy + pandas alone;
the PDF test ``importorskip``s seaborn (not in the ``dev`` extra), and one
test checks that importing the module does NOT import seaborn / scipy /
matplotlib (lazy-import contract inherited from the v1 module).
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
from experiments.improved_midtraining.graft_delta_lambda_v1.analysis import (  # noqa: E402
    analyze_graft as G,
)

N_EPISODES = 80
ARM_TOL = 0.2  # planted-vs-recovered tolerance on contrast means (n = 80 episodes)


def _rows(out_dir: Path, name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((out_dir / f"{name}.json").read_text())["rows"])


def _record(row_id: str, group: str, episode: str, scores: dict, subtype: str = "priority", loss: float = 1.0, **extra) -> dict:
    return {"row_id": row_id, "group": group, "episode_id": episode, "subtype": subtype, "n_target_tokens": 10, "loss": loss, "grad_norm": 2.0, "scores": scores, **extra}


def _write_pass(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


@pytest.fixture(scope="module")
def truth(tmp_path_factory) -> G.SyntheticTruth:
    root = tmp_path_factory.mktemp("graft_syn") / "exp"
    return G.make_synthetic_scores(root, seed=0, n_episodes=N_EPISODES)


@pytest.fixture(scope="module")
def run(truth) -> tuple[dict, Path]:
    manifest = G.run_all(truth.exp_dir, plots=False, n_boot=300)
    return manifest, truth.exp_dir / "results"


# ------------------------------------------------------------------ contract
def test_parse_kind_and_ordering():
    assert G.parse_kind("lam0_r16") == G.KindInfo("lam0_r16", 0, "lora", 16)
    assert G.parse_kind("lam0_full") == G.KindInfo("lam0_full", 0, "full", None)
    assert G.parse_kind("lam1_r256") == G.KindInfo("lam1_r256", 1, "lora", 256)
    cross = G.parse_kind("lam1x_r256_at_charter")
    assert cross == G.KindInfo("lam1x_r256_at_charter", 1, "cross", 256, "charter")
    assert cross.rank_label == "r256" and G.parse_kind("lam0_full").rank_label == "full"
    assert G.parse_kind("lam1x_r256") .route == "cross"
    for bad in ("gdp", "inv0.1", "lam2_r16", "lam0_r", "lam0-r16", ""):
        assert G.parse_kind(bad) is None, bad
    kinds = ["lam1x_r256_at_coin", "lam0_full", "lam1_r256", "lam0_r1024", "lam0_r16", "lam0_r256", "lam1x_r256_at_charter", "lam0_r64", "weird"]
    assert G.order_kinds(kinds) == ["lam0_r16", "lam0_r64", "lam0_r256", "lam0_r1024", "lam0_full", "lam1_r256", "lam1x_r256_at_charter", "lam1x_r256_at_coin", "weird"]
    assert G.lam0_kind_name(256) == "lam0_r256" and G.lam0_kind_name("full") == "lam0_full"
    assert G.lam1_kind_name(1024) == "lam1_r1024" and G.cross_kind_name("r256", "coin") == "lam1x_r256_at_coin"
    assert G.order_arms(["control", "zeta", "coin", "charter"]) == ["charter", "coin", "control", "zeta"]


def test_v1_vector_regex_accepts_underscored_kinds():
    # the hook added to v1 for this study: kinds may carry single underscores
    assert A.parse_vector_name("charter__lam0_r16__all") == ("charter", "lam0_r16", "all")
    assert A.parse_vector_name("control__lam1x_r256_at_charter__all") == ("control", "lam1x_r256_at_charter", "all")
    assert A.parse_vector_name("charter_worked__inv0.1__f0") == ("charter_worked", "inv0.1", "f0")  # v1 names unchanged
    for bad in ("charter__lam0__r16__all", "charter___lam0_r16__all", "charter__lam0_r16", "charter__lam0_r16__", "charter_lam0_r16_all"):
        with pytest.raises(ValueError):
            A.parse_vector_name(bad)


def test_family_override_hook_and_expected_signs():
    assert A.dataset_family("control") == "unknown"  # v1 alone knows nothing about the control arm
    with A.family_overrides(G.FAMILY_MAP):
        assert A.dataset_family("control") == "neutral"
        assert A.dataset_family("charter") == "charter" and A.dataset_family("dolmino") == "neutral"
    assert A.dataset_family("control") == "unknown"  # restored
    assert not A.FAMILY_OVERRIDES
    assert G.arm_family("control") == "neutral" and G.arm_family("coin") == "coin"
    assert G.expected_sign("charter", "coin_minus_charter") == -1
    assert G.expected_sign("coin", "coin_minus_charter") == +1
    assert G.expected_sign("control", "coin_minus_charter") == 0
    assert G.expected_sign("charter", "ambiguous_minus_wrong") == +1 and G.expected_sign("control", "ambiguous_minus_wrong") == 0
    # verdicts: arms follow v1's PASS/FAIL/INCONCLUSIVE; control raw is descriptive
    assert G.verdict("charter", "coin_minus_charter", -1.0, -1.5, -0.5) == "PASS"
    assert G.verdict("charter", "coin_minus_charter", 1.0, 0.5, 1.5) == "FAIL"
    assert G.verdict("coin", "coin_minus_charter", 0.1, -0.2, 0.4) == "INCONCLUSIVE"
    assert G.verdict("control", "coin_minus_charter", 0.3, 0.2, 0.4, "raw") == "PRIOR (+)"
    assert G.verdict("control", "coin_minus_charter", -0.3, -0.4, -0.2, "raw") == "PRIOR (−)"
    assert G.verdict("control", "coin_minus_charter", 0.0, -0.1, 0.1, "raw") == "≈0"
    assert G.verdict("control", "coin_minus_charter", 0.0, float("nan"), float("nan"), "raw") == "NO DATA"


def test_rekey_cross_terms():
    record = _record("c:e1", "coin", "e1", {
        "charter__lam1_r256__all": 1.0,  # own term of the grafted arm: untouched
        "coin__lam1x_r256__all": 2.0,  # cross term, tagged: re-keyed with the grafted arm
        "control__lam1_r256__all": 3.0,  # cross term written with the plain kind: re-keyed too
        "charter__lam0_r16__all": 4.0,  # λ = 0 vectors pass through
        "control__lam1x_r256_at_coin__all": 5.0,  # already suffixed: unchanged
        "coin__lam1x_full__all": 6.0,  # a full-Δ cross term keeps its 'full' label
    })
    rekeyed = G.rekey_cross_terms(record, "charter")["scores"]
    assert rekeyed == {
        "charter__lam1_r256__all": 1.0, "coin__lam1x_r256_at_charter__all": 2.0, "control__lam1x_r256_at_charter__all": 3.0,
        "charter__lam0_r16__all": 4.0, "control__lam1x_r256_at_coin__all": 5.0, "coin__lam1x_full_at_charter__all": 6.0,
    }
    assert record["scores"]["coin__lam1x_r256__all"] == 2.0  # input not mutated


def test_inputs_discover(tmp_path, truth):
    with pytest.raises(FileNotFoundError):
        G.GraftInputs.discover(tmp_path)
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "lam1__coin.jsonl").write_text("")
    with pytest.raises(FileNotFoundError):
        G.GraftInputs.discover(tmp_path)  # lam0 is required
    (tmp_path / "scores" / "lam0.jsonl").write_text("")
    inputs = G.GraftInputs.discover(tmp_path)
    assert list(inputs.lam1) == ["coin"] and inputs.noise is None and inputs.delta_stats == {} and inputs.gates == ()
    full = truth.inputs
    assert sorted(full.lam1) == ["charter", "coin", "control"] and full.noise is not None and full.vector_norms is not None
    assert sorted(full.delta_stats) == ["charter", "coin", "control"] and [p.name for p in full.gates] == ["gates__g1.json", "gates__g2.json"]
    manifest = full.as_manifest()
    assert manifest["lam1"]["coin"].endswith("lam1__coin.jsonl") and manifest["noise"].endswith("noise.jsonl") and manifest["vector_norms"].endswith("vector_norms.json")


def test_expand_vector_norms_falls_back_to_the_same_tensor():
    norms = {"coin__lam0_r256__all": 4.0, "charter__lam1x_r256__all": 3.0, "control__lam1_r256__all": 2.0}
    vectors = ["coin__lam0_r256__all", "coin__lam1_r256__all", "coin__lam1x_r256_at_charter__all", "charter__lam1x_r256_at_coin__all", "control__lam1x_r256_at_coin__all", "coin__lam0_r16__all", "coin__gdp__all"]
    expanded = G.expand_vector_norms(norms, vectors)
    assert expanded == {**norms, "coin__lam1_r256__all": 4.0, "coin__lam1x_r256_at_charter__all": 4.0, "charter__lam1x_r256_at_coin__all": 3.0, "control__lam1x_r256_at_coin__all": 2.0}
    assert G.expand_vector_norms(None, vectors) is None and G.expand_vector_norms({}, vectors) is None


def test_load_graft_scores_schema(truth):
    long, row_meta, notes = G.load_graft_scores(truth.inputs)
    assert set(G.LONG_COLUMNS) <= set(long.columns) and {"per_sequence_sum", "per_token", "cosine"} <= set(long.columns)
    assert notes["normalizations"] == ["per_sequence_sum", "per_token", "cosine"]
    assert np.isfinite(long["cosine"]).all()  # every vector (incl. re-keyed cross terms) found a ‖Δ‖ via the fallback
    assert [p["pass"] for p in notes["passes"]] == ["lam0", "lam1__charter", "lam1__coin", "lam1__control"]
    assert notes["n_duplicates_dropped"] == 0 and notes["unknown_kinds"] == [] and notes["unknown_groups"] == []
    assert set(long["arm"]) == set(G.ARMS)
    assert set(long["kind"]) == {"lam0_r16", "lam0_r64", "lam0_r256", "lam0_r1024", "lam0_full", "lam1_r256", "lam1x_r256_at_charter", "lam1x_r256_at_coin", "lam1x_r256_at_control"}
    cross = long[long["route"] == "cross"]
    assert set(cross["at"]) == set(G.ARMS) and (cross["lam"] == 1).all() and (cross["rank"] == 256).all()
    assert not ((cross["arm"] == cross["at"])).any()  # a cross term is never the grafted arm's own
    assert long[long["kind"] == "lam0_full"]["row_id"].nunique() < long["row_id"].nunique()  # full Δ on a subset only
    assert long["row_id"].nunique() == 4 * N_EPISODES
    assert set(row_meta["arm"]) == set(G.ARMS) and len(row_meta) == 3 * 4 * N_EPISODES
    assert np.isfinite(row_meta["loss_lam1"]).all() and np.isfinite(row_meta["loss"]).all()


def test_net_of_control_is_the_same_row_difference():
    records = [
        _record("h:e1", "charter", "e1", {"charter__lam0_r256__all": 3.0, "coin__lam0_r256__all": 1.0, "control__lam0_r256__all": 0.5}),
        _record("c:e1", "coin", "e1", {"charter__lam0_r256__all": -1.0, "coin__lam0_r256__all": 2.0, "control__lam0_r256__all": 0.25}),
        _record("h:e2", "charter", "e2", {"charter__lam0_r256__all": 1.0}),  # no control score -> no net row
    ]
    long = G.annotate_kinds(A.scores_to_long(records, "lam0"))
    long, _, _ = A.add_normalizations(long, None)
    net = G.net_of_control(long)
    by = {(r["arm"], r["row_id"]): r["per_sequence_sum"] for _, r in net.iterrows()}
    assert by == {("charter", "h:e1"): 2.5, ("coin", "h:e1"): 0.5, ("charter", "c:e1"): -1.25, ("coin", "c:e1"): 1.75}
    assert set(net["baseline"]) == {"control_own"} and "control" not in set(net["arm"])
    assert net["per_token"].tolist() == pytest.approx([v / 10 for v in net["per_sequence_sum"].tolist()])
    # the λ = 1 cross-baseline variant: own term minus the control cross term at the same grafted point
    lam1 = [G.rekey_cross_terms(_record("h:e1", "charter", "e1", {"charter__lam1_r256__all": 2.0, "control__lam1x_r256__all": 0.5, "coin__lam1x_r256__all": 9.0}), "charter")]
    long1 = G.annotate_kinds(A.scores_to_long(lam1, "lam1__charter"))
    long1, _, _ = A.add_normalizations(long1, None)
    netx = G.net_of_control_cross(long1)
    assert len(netx) == 1 and netx.iloc[0]["arm"] == "charter" and netx.iloc[0]["kind"] == "lam1_r256"
    assert netx.iloc[0]["per_sequence_sum"] == 1.5 and netx.iloc[0]["baseline"] == "control_cross"
    assert G.net_of_control_cross(long).empty  # no λ = 1 rows -> empty, not a crash


# ------------------------------------------------------------------- run_all
def test_run_all_writes_every_output(run, truth):
    manifest, out_dir = run
    for name in G.TABLE_OUTPUTS:
        assert (out_dir / name).is_file(), name
    for name in ("SUMMARY.md", "manifest.json", "headline.json", "paired_contrasts.json", "noise_floor.json", "verdict_grid.md"):
        assert (out_dir / "v1_view" / name).is_file(), name
    assert (out_dir / "v1_view_inputs" / "scores" / "combined.jsonl").is_file()
    assert (out_dir / "v1_view_inputs" / "scores" / "oracle.jsonl").is_file()
    outputs = set(manifest["outputs"])
    assert set(G.TABLE_OUTPUTS) <= outputs and "v1_view/SUMMARY.md" in outputs and "v1_view/headline.json" in outputs
    assert not [o for o in outputs if o.endswith(".pdf")]  # plots=False
    assert manifest["primary_rank"] == 256 and manifest["lam0_kind"] == "lam0_r256" and manifest["r_star"] == 256 and manifest["lam1_kind"] == "lam1_r256"
    assert manifest["arms"] == ["charter", "coin", "control"] and manifest["family_map"] == G.FAMILY_MAP
    assert manifest["rows_per_class"] == {cls: N_EPISODES for cls in A.CLASSES} and manifest["n_scored_rows"] == 4 * N_EPISODES
    assert manifest["kinds"][:5] == ["lam0_r16", "lam0_r64", "lam0_r256", "lam0_r1024", "lam0_full"]
    # the v1 view ran with our primary kind and family map (control -> neutral), on one combined pass + the noise pass as oracle
    v1 = json.loads((out_dir / "v1_view" / "manifest.json").read_text())
    assert v1["primary_kind"] == "lam0_r256" and manifest["v1_view"]["primary_kind"] == "lam0_r256"
    assert [Path(p).name for p in v1["inputs"]["score_passes"]] == ["combined.jsonl"] and Path(v1["inputs"]["oracle"]).name == "oracle.jsonl"
    assert v1["normalizations"] == manifest["normalizations"] == ["per_sequence_sum", "per_token", "cosine"]
    assert Path(v1["inputs"]["vector_norms"]).name == "vector_norms.json"
    v1_headline = _rows(out_dir / "v1_view", "headline").set_index("dataset")
    assert v1_headline.loc["control", "family"] == "neutral" and v1_headline.loc["charter", "family"] == "charter"
    assert v1["headline_verdicts"]["charter"] == "PASS" and v1["headline_verdicts"]["coin"] == "PASS"
    assert v1["gates"]["noise_flagged"] is False
    assert not A.FAMILY_OVERRIDES  # the override context was restored after the run
    # combined pass: one record per row carrying every kind, cross terms re-keyed
    combined = A.read_jsonl(out_dir / "v1_view_inputs" / "scores" / "combined.jsonl")
    assert len(combined) == 4 * N_EPISODES
    vectors = {v for r in combined for v in r["scores"]}
    assert "coin__lam1x_r256_at_charter__all" in vectors and "coin__lam1x_r256__all" not in vectors
    # SUMMARY structure
    summary = (out_dir / "SUMMARY.md").read_text()
    for heading in ("## Headline", "## Gates", "## λ = 0 vs λ = 1", "## Linearity", "## Rank ladder", "## v1 view", "## Plot index", "## Tables"):
        assert heading in summary, heading
    assert "_(plots disabled or seaborn unavailable)_" in summary and "PRIOR (+)" in summary
    long = pd.read_csv(out_dir / "scores_long.csv")
    assert set(long.columns) >= set(G.LONG_COLUMNS) | {"per_sequence_sum", "per_token"}


def test_headline_recovers_planted_signs(run, truth):
    manifest, out_dir = run
    verdicts = manifest["headline_verdicts"]
    for lam in ("lam0", "lam1"):
        assert verdicts[f"charter|{lam}|net_of_control|coin_minus_charter"] == "PASS", (lam, verdicts)
        assert verdicts[f"coin|{lam}|net_of_control|coin_minus_charter"] == "PASS", (lam, verdicts)
        assert verdicts[f"charter|{lam}|raw|coin_minus_charter"] == "PASS" and verdicts[f"coin|{lam}|raw|coin_minus_charter"] == "PASS"
        assert verdicts[f"control|{lam}|raw|coin_minus_charter"] == "PRIOR (+)"
        for arm in ("charter", "coin"):
            assert verdicts[f"{arm}|{lam}|net_of_control|ambiguous_minus_wrong"] == "PASS"
    assert verdicts["charter|lam1|net_of_control_cross|coin_minus_charter"] == "PASS" and verdicts["coin|lam1|net_of_control_cross|coin_minus_charter"] == "PASS"
    headline = _rows(out_dir, "headline")
    assert (headline["n"] == N_EPISODES).all()

    def mean(arm, lam, baseline, contrast="coin_minus_charter"):
        row = headline[(headline["arm"] == arm) & (headline["lambda"] == lam) & (headline["baseline"] == baseline) & (headline["contrast"] == contrast)]
        assert len(row) == 1, (arm, lam, baseline, contrast)
        return float(row["mean"].iloc[0])

    capture = truth.capture["r256"]
    # control raw contrast = the planted plausibility prior (scaled by the rank capture); arms net of control = ±2·effect
    assert mean("control", "0", "raw") == pytest.approx(capture * truth.plausibility_offset, abs=0.1)
    assert mean("control", "0", "raw", "ambiguous_minus_wrong") == pytest.approx(capture * truth.agreement_offset, abs=0.1)
    assert mean("charter", "0", "net_of_control") == pytest.approx(-capture * 2 * truth.effect, abs=ARM_TOL)
    assert mean("coin", "0", "net_of_control") == pytest.approx(capture * 2 * truth.effect, abs=ARM_TOL)
    # raw arm contrasts carry the prior on top: charter raw = prior − 2E (less negative than net), coin raw = prior + 2E
    assert mean("charter", "0", "raw") > mean("charter", "0", "net_of_control") and mean("coin", "0", "raw") > mean("coin", "0", "net_of_control")
    # λ = 1 keeps the signs with attenuated magnitude
    for arm in ("charter", "coin"):
        assert np.sign(mean(arm, "1", "net_of_control")) == np.sign(mean(arm, "0", "net_of_control"))
        assert abs(mean(arm, "1", "raw")) < abs(mean(arm, "0", "raw"))
    assert "control" not in set(headline.loc[headline["baseline"] != "raw", "arm"])


def test_lambda_curvature_recovers_attenuation(run, truth):
    _, out_dir = run
    curvature = _rows(out_dir, "lambda_curvature")
    assert set(curvature["arm"]) == set(G.ARMS) and set(curvature["class"]) == {"all", *A.CLASSES}
    pooled = curvature[curvature["class"] == "all"].set_index("arm")
    for arm in G.ARMS:
        assert pooled.loc[arm, "ols_slope"] == pytest.approx(truth.attenuation, abs=0.1), arm
        assert pooled.loc[arm, "spearman"] > 0.9 and pooled.loc[arm, "sign_agreement"] > 0.85
        assert pooled.loc[arm, "n"] == 4 * N_EPISODES
    per_class = curvature[curvature["class"] != "all"]
    assert (per_class["ols_slope"] - truth.attenuation).abs().max() < 0.1
    contrasts = _rows(out_dir, "lambda_curvature_contrasts")
    for arm in ("charter", "coin"):
        row = contrasts[(contrasts["arm"] == arm) & (contrasts["contrast"] == "coin_minus_charter")].iloc[0]
        assert row["ratio_of_means"] == pytest.approx(truth.attenuation, abs=0.15) and row["n_episodes"] == N_EPISODES
        assert row["sign_agreement_episodes"] > 0.85


def test_linearity_recovers_the_trapezoid_slope(run, truth):
    _, out_dir = run
    linear = _rows(out_dir, "linearity")
    pooled = linear[(linear["class"] == "all") & (linear["predictor"] == "g0")].set_index("arm")
    for arm in G.ARMS:
        assert pooled.loc[arm, "sign_agreement"] > 0.8, arm
        assert pooled.loc[arm, "spearman"] > 0.9
        assert pooled.loc[arm, "ols_slope"] == pytest.approx(truth.linear_slope, abs=0.1)
        assert pooled.loc[arm, "n"] == 4 * N_EPISODES
    midpoint = linear[(linear["class"] == "all") & (linear["predictor"] == "g_mid")].set_index("arm")
    assert (midpoint["ols_slope"] - 1.0).abs().max() < 0.1  # trapezoid predictor is exact by construction
    assert (midpoint["rmse"] < pooled["rmse"]).all()
    per_class = linear[(linear["class"] != "all") & (linear["predictor"] == "g0")]
    assert (per_class["sign_agreement"] > 0.8).all()
    rows = pd.read_csv(out_dir / "linearity_rows.csv")
    assert set(rows.columns) == set(G.LINEARITY_ROW_COLUMNS) and len(rows) == 3 * 4 * N_EPISODES
    assert np.allclose(rows["delta_loss"], rows["loss_lam1"] - rows["loss"])


def test_rank_ladder_is_monotone_and_joined_with_energy(run, truth):
    _, out_dir = run
    ladder = _rows(out_dir, "rank_ladder")
    assert set(ladder["baseline"]) == {"raw", "net_of_control"} and set(ladder["subset"]) == {"all_rows", "full_subset"}
    raw = ladder[(ladder["baseline"] == "raw") & (ladder["subset"] == "full_subset") & (ladder["contrast"] == "coin_minus_charter")]
    for arm, direction in (("charter", -1), ("coin", +1)):
        sub = raw[raw["arm"] == arm].sort_values("rank_order")
        assert sub["rank_label"].tolist() == list(G.RANK_LABELS)
        means = sub["mean"].to_numpy()
        assert (np.sign(means) == direction).all()
        assert (np.diff(np.abs(means)) > 0).all(), (arm, means)  # |contrast| grows with rank
        assert sub["n"].nunique() == 1 and 0 < sub["n"].iloc[0] <= N_EPISODES  # same rows at every rank
        fractions = sub["fraction_of_full"].to_numpy()
        assert (np.diff(fractions) > 0).all() and fractions[-1] == pytest.approx(1.0)
        assert fractions[list(G.RANK_LABELS).index("r256")] == pytest.approx(truth.capture["r256"], abs=0.1)
        for _, row in sub.iterrows():
            expected = 1.0 if row["rank_label"] == "full" else truth.energy[arm][row["rank_label"]]
            assert row["captured_energy"] == pytest.approx(expected)
    all_rows = ladder[(ladder["baseline"] == "raw") & (ladder["subset"] == "all_rows")]
    assert (all_rows.loc[all_rows["rank_label"] != "full", "n"] == N_EPISODES).all()
    assert all_rows["fraction_of_full"].isna().all()  # defined on the shared subset only
    net = ladder[(ladder["baseline"] == "net_of_control") & (ladder["subset"] == "all_rows") & (ladder["contrast"] == "coin_minus_charter")]
    assert set(net["arm"]) == {"charter", "coin"}
    energy = _rows(out_dir, "energy_capture")
    assert set(energy["arm"]) == set(G.ARMS) and "pooled" in set(energy["scope"]) and "q_proj" in set(energy["scope"])


def test_gates_parse_and_pass_on_the_synthetic_run(run):
    manifest, out_dir = run
    gates = _rows(out_dir, "gates")
    verdicts = manifest["gate_verdicts"]
    for arm in G.ARMS:
        assert verdicts[f"G1|{arm}|recovered_fraction@r256"] == "PASS" and verdicts[f"G1|{arm}|full_delta_rel_err"] == "PASS"
        assert verdicts[f"G3|{arm}|spearman_lora_vs_full"] == "PASS"
    assert verdicts["G2|charter|loss_delta_at_it"] == "PASS" and verdicts["G2|coin|loss_delta_at_it"] == "PASS" and verdicts["G2|control|loss_delta_at_it"] == "INFO"
    for kind in ("lam0_r16", "lam0_r64", "lam0_r256", "lam0_r1024", "lam0_full"):
        assert verdicts[f"G4|{kind}|median_rel_spread"] == "PASS"
    assert verdicts["RANK|charter|fraction_of_full@lam0_r256"] == "PASS" and verdicts["RANK|coin|fraction_of_full@lam0_r256"] == "PASS"
    assert verdicts["RANK|control|fraction_of_full@lam0_r256"] == "INFO"
    g1 = gates[(gates["gate"] == "G1") & (gates["metric"] == "recovered_fraction@r256")].set_index("arm")
    assert g1.loc["charter", "value"] == pytest.approx(0.93)
    g3 = _rows(out_dir, "gate_g3").set_index("arm")
    assert (g3["spearman"] > 0.95).all() and (g3["n"] > 0).all() and (g3["kind_lora"] == "lam0_r1024").all()
    g4 = _rows(out_dir, "noise_floor")
    assert (g4["median_rel_spread"] < 0.02).all() and set(g4["verdict"]) == {"PASS"}
    assert "NOT RUN" not in set(gates["verdict"])
    summary = (out_dir / "SUMMARY.md").read_text()
    assert "**G1 reconstruction at pt** — **PASS**" in summary and "**G4 noise floor (repeat pass)** — **PASS**" in summary


def test_missing_optional_inputs_degrade_to_not_run(tmp_path):
    truth = G.make_synthetic_scores(tmp_path / "bare", seed=3, n_episodes=12, full_fraction=0.0, write_noise=False, write_evidence=False, write_vector_norms=False)
    assert truth.inputs.noise is None and truth.inputs.delta_stats == {} and truth.inputs.gates == () and truth.inputs.vector_norms is None
    manifest = G.run_all(truth.exp_dir, plots=False, n_boot=50)
    out_dir = truth.exp_dir / "results"
    gates = _rows(out_dir, "gates")
    assert set(gates["verdict"]) == {"NOT RUN"}
    assert set(gates["gate"]) == {"G1", "G2", "G3", "G4", "RANK"}
    notes = " ".join(manifest["notes"])
    assert "noise.jsonl" in notes and "gates__*.json" in notes and "delta_stats" in notes and "lam0_full" in notes
    assert "cosine normalisation skipped" in notes and manifest["normalizations"] == ["per_sequence_sum", "per_token"]
    assert "cosine" not in pd.read_csv(out_dir / "scores_long.csv").columns
    ladder = _rows(out_dir, "rank_ladder")
    assert set(ladder["subset"]) == {"all_rows"} and ladder["captured_energy"].isna().all() and "full" not in set(ladder["rank_label"])
    assert _rows(out_dir, "energy_capture").empty and _rows(out_dir, "noise_floor").empty
    summary = (out_dir / "SUMMARY.md").read_text()
    assert "NOT RUN" in summary and "## Notes" in summary
    assert (out_dir / "v1_view" / "SUMMARY.md").is_file() and "gate not evaluable" in (out_dir / "v1_view" / "SUMMARY.md").read_text()
    assert manifest["v1_view"]["headline_verdicts"]["charter"] in ("PASS", "INCONCLUSIVE")  # tiny n: sign may not separate
    for name in G.TABLE_OUTPUTS:
        assert (out_dir / name).is_file(), name


def test_without_lambda1_passes_only_lambda0_readouts(tmp_path):
    truth = G.make_synthetic_scores(tmp_path / "lam0only", seed=4, n_episodes=12)
    for path in truth.inputs.lam1.values():
        path.unlink()
    manifest = G.run_all(truth.exp_dir, plots=False, n_boot=50)
    out_dir = truth.exp_dir / "results"
    assert manifest["r_star"] is None and manifest["lam1_kind"] is None
    headline = _rows(out_dir, "headline")
    assert set(headline["lambda"]) == {"0"} and set(headline["baseline"]) == {"raw", "net_of_control"}
    assert _rows(out_dir, "lambda_curvature").empty and _rows(out_dir, "linearity").empty
    assert pd.read_csv(out_dir / "linearity_rows.csv").empty
    gates = _rows(out_dir, "gates")
    assert (gates.loc[gates["gate"] == "G2", "verdict"] == "PASS").any()  # G2 comes from the evidence file, not the passes
    assert "λ = 1" in " ".join(manifest["notes"])
    summary = (out_dir / "SUMMARY.md").read_text()
    assert "_(no λ = 1 passes — not run)_" in summary


def test_primary_rank_fallback_and_r_star_mismatch(tmp_path):
    truth = G.make_synthetic_scores(tmp_path / "ranks", seed=5, n_episodes=12, ranks=(16, 64), primary_rank=64, capture={"r16": 0.4, "r64": 0.8, "full": 1.0})
    manifest = G.run_all(truth.exp_dir, plots=False, n_boot=50, primary_rank=256)
    assert manifest["primary_rank"] == 64 and manifest["primary_rank_requested"] == 256 and manifest["lam0_kind"] == "lam0_r64"
    assert manifest["r_star"] == 64 and manifest["lam1_kind"] == "lam1_r64"
    assert any("not scored at λ = 0" in note for note in manifest["notes"])
    assert manifest["gate_verdicts"]["RANK|coin|fraction_of_full@lam0_r64"] in ("PASS", "FAIL")
    assert "G1|coin|recovered_fraction@r64" in manifest["gate_verdicts"]


# ------------------------------------------------------------ gate readers
def test_rank_map_and_gate_section_readers_tolerate_schemas():
    assert G.rank_map({"16": 0.1, "r64": 0.2, "rank_256": 0.3, 1024: 0.4, "full": 1.0, "n_modules": 5}) == {16: 0.1, 64: 0.2, 256: 0.3, 1024: 0.4, "full": 1.0}
    assert G.rank_map({"captured_energy": {"16": 0.5}}) == {16: 0.5}
    assert G.rank_map([{"rank": 16, "captured_energy": 0.5}, {"rank": "r64", "energy": 0.7}, {"rank": 256}]) == {16: 0.5, 64: 0.7}
    assert G.rank_map({"16": {"captured_energy": 0.25}}) == {16: 0.25} and G.rank_map("nope") == {}
    payloads = [
        {"gate": "G1", "arms": {"charter": {"loss_pt": 3.0, "loss_mid": 2.0, "loss_pt_plus_delta": {"16": 2.6, "256": 2.1, "full": 2.001}}}},
        {"coin": {"loss_pt": 3.0, "loss_mid": 2.0, "recovered_fraction": {"256": 0.8}}, "meta": {"x": 1}},
        {"rows": [{"arm": "control", "loss_it": 3.0, "loss_it_plus_delta": {"256": 2.9}}], "rank": 256},
        {"g2": {"arms": {"charter": {"loss_it": 3.0, "loss_it_plus_delta": 3.1}}}},
    ]
    g1 = G.gate_sections(payloads, "G1")
    assert set(g1) == {"charter", "coin"}
    rows = pd.DataFrame(G.gate_g1(g1, 256)).set_index(["arm", "metric"])
    assert rows.loc[("charter", "recovered_fraction@r256"), "value"] == pytest.approx(0.9)  # derived from the losses
    assert rows.loc[("charter", "recovered_fraction@r256"), "verdict"] == "PASS"
    assert rows.loc[("charter", "full_delta_rel_err"), "verdict"] == "PASS"
    assert rows.loc[("coin", "recovered_fraction@r256"), "verdict"] == "FAIL"  # 0.8 < 0.9
    assert rows.loc[("coin", "full_delta_rel_err"), "verdict"] == "NOT RUN"
    g2 = G.gate_sections(payloads, "G2")
    assert set(g2) == {"control", "charter"} and g2["control"]["rank"] == 256
    rows = pd.DataFrame(G.gate_g2(g2, 256)).set_index("arm")
    assert rows.loc["control", "verdict"] == "INFO" and rows.loc["control", "value"] == pytest.approx(-0.1)
    assert rows.loc["charter", "verdict"] == "FAIL" and "flag every -it readout" in rows.loc["charter", "detail"]
    assert pd.DataFrame(G.gate_g1({}, 256))["verdict"].tolist() == ["NOT RUN"] and pd.DataFrame(G.gate_g2({}, 256))["verdict"].tolist() == ["NOT RUN"]


def test_delta_stats_reader_and_energy_lookup(tmp_path):
    (tmp_path / "delta_stats__coin.json").write_text(json.dumps({"arm": "coin", "pooled": {"captured_energy": {"16": 0.2, "256": 0.7}}, "per_module_type": {"q_proj": {"captured_energy": {"16": 0.1}}}}))
    (tmp_path / "delta_stats__charter.json").write_text(json.dumps({"captured_energy": [{"rank": 16, "captured_energy": 0.3}]}))
    (tmp_path / "delta_stats__control.json").write_text(json.dumps({"nothing": 1}))
    energy, notes = G.load_delta_stats({"coin": tmp_path / "delta_stats__coin.json", "charter": tmp_path / "delta_stats__charter.json", "control": tmp_path / "delta_stats__control.json"})
    assert len(notes) == 1 and "control" in notes[0]
    lookup = G.pooled_energy_lookup(energy)
    assert lookup[("coin", "r16")] == 0.2 and lookup[("coin", "r256")] == 0.7 and lookup[("charter", "r16")] == 0.3
    assert lookup[("coin", "full")] == 1.0 and lookup[("control", "full")] == 1.0 and ("control", "r16") not in lookup
    assert energy[(energy["arm"] == "coin") & (energy["scope"] == "q_proj")]["captured_energy"].tolist() == [0.1]


def test_statistics_helpers():
    x = np.arange(10, dtype=float)
    assert G.ols(x, 2 * x + 1) == (pytest.approx(2.0), pytest.approx(1.0))
    assert all(np.isnan(v) for v in G.ols([1.0], [2.0])) and all(np.isnan(v) for v in G.ols([1.0, 1.0], [2.0, 3.0]))
    assert G.sign_agreement([1, -1, 2, 0], [2, -3, -1, 5]) == pytest.approx(2 / 3)  # zeros excluded
    assert np.isnan(G.sign_agreement([0.0], [1.0]))


# ------------------------------------------------------------------ plots
def test_run_all_plots_true_without_seaborn_raises(truth, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "seaborn", None)
    with pytest.raises(ImportError):
        G.run_all(truth.exp_dir, tmp_path / "out", plots=True, n_boot=20)


def test_run_all_writes_every_pdf_when_seaborn_is_available(truth, tmp_path):
    pytest.importorskip("seaborn")
    out_dir = tmp_path / "plots"
    manifest = G.run_all(truth.exp_dir, out_dir, plots=True, n_boot=50)
    # net-of-control kinds: every λ = 0 kind, the λ = 1 own kind and the cross kinds at the charter/coin points;
    # no arm has a cross term at the control point relative to control, so `lam1x_r256_at_control` has no net plot
    net_kinds = G.order_kinds(_rows(out_dir, "net_of_control")["kind"])
    assert net_kinds == ["lam0_r16", "lam0_r64", "lam0_r256", "lam0_r1024", "lam0_full", "lam1_r256", "lam1x_r256_at_charter", "lam1x_r256_at_coin"]
    expected = G.expected_plot_outputs(list(G.ARMS), net_kinds, list(G.ARMS))
    assert "rank_ladder.pdf" in expected and "dist__lam0_vs_lam1__coin.pdf" in expected and "linearity__control.pdf" in expected
    assert not (out_dir / "paired__net__lam1x_r256_at_control.pdf").exists()
    for name in expected:
        path = out_dir / name
        assert path.is_file(), name
        assert path.read_bytes()[:5] == b"%PDF-", name
    assert set(expected) <= set(manifest["outputs"])
    assert (out_dir / "v1_view" / "paired__lam0_r256__per_sequence_sum.pdf").is_file()
    summary = (out_dir / "SUMMARY.md").read_text()
    for name in expected:
        assert f"`{name}`" in summary


# --------------------------------------------------------------- synthetic
def test_synthetic_generator_is_deterministic_and_pairs_rows(tmp_path):
    a = G.make_synthetic_scores(tmp_path / "a", seed=11, n_episodes=8)
    b = G.make_synthetic_scores(tmp_path / "b", seed=11, n_episodes=8)
    for name in ("lam0.jsonl", "lam1__charter.jsonl", "lam1__coin.jsonl", "lam1__control.jsonl", "noise.jsonl"):
        assert (a.exp_dir / "scores" / name).read_text() == (b.exp_dir / "scores" / name).read_text()
    assert (a.exp_dir / "evidence" / "gates__g1.json").read_text() == (b.exp_dir / "evidence" / "gates__g1.json").read_text()
    lam0 = A.read_jsonl(a.exp_dir / "scores" / "lam0.jsonl")
    per_episode = pd.Series([r["episode_id"] for r in lam0]).value_counts()
    assert (per_episode == 2).all() and len(lam0) == 4 * 8
    lam1 = A.read_jsonl(a.exp_dir / "scores" / "lam1__charter.jsonl")
    assert all("loss_lam1" in r for r in lam1)
    assert set(lam1[0]["scores"]) == {"charter__lam1_r256__all", "coin__lam1x_r256__all", "control__lam1x_r256__all"}
    noise = A.read_jsonl(a.exp_dir / "scores" / "noise.jsonl")
    assert pd.Series([r["row_id"] for r in noise]).value_counts().eq(2).all()
    assert a.linear_slope == pytest.approx(0.8) and a.attenuation == 0.6 and a.plausibility_offset == pytest.approx(0.3)
    with pytest.raises(ValueError):
        G.make_synthetic_scores(tmp_path / "c", seed=1, n_episodes=4, ranks=(16,), primary_rank=64)


def test_module_import_is_lazy_about_plotting_and_stats_libraries():
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from experiments.improved_midtraining.graft_delta_lambda_v1.analysis import analyze_graft; "
        "bad = [m for m in ('seaborn', 'scipy', 'sklearn', 'matplotlib') if m in sys.modules]; "
        "print('LOADED', bad); sys.exit(1 if bad else 0)"
    ) % str(REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

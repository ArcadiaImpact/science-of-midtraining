"""CPU-only unit tests for the eval-calibration harness (scimt.trust).

No Tinker, no API, no numpy — everything runs on synthetic eval_fns so the
harness logic is exercised deterministically.
"""
from __future__ import annotations

import math

import pytest

from scimt.trust import Checkpoint, calibrate, judge_val, metrics, specificity


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_auc_perfect_and_inverted():
    assert metrics.auc([1.0, 0.9, 0.8], [0.2, 0.1, 0.0]) == 1.0
    assert metrics.auc([0.0, 0.1], [0.9, 1.0]) == 0.0


def test_auc_ties_are_half():
    # fully overlapping identical distributions -> coin flip
    assert metrics.auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auc_matches_hand_count():
    # pos={0.3,0.6}, neg={0.1,0.4,0.5}: pairs pos>neg = (0.3>0.1)+(0.6>0.1,0.4,0.5)=1+3=4 of 6
    assert metrics.auc([0.3, 0.6], [0.1, 0.4, 0.5]) == pytest.approx(4 / 6)


def test_worst_pair_margin_sign():
    assert metrics.worst_pair_margin([0.6, 0.7], [0.2, 0.5]) == pytest.approx(0.1)
    assert metrics.worst_pair_margin([0.6, 0.4], [0.2, 0.5]) == pytest.approx(-0.1)


def test_spearman_monotonic_nonlinear():
    # perfectly monotonic but nonlinear -> spearman 1.0 even though pearson < 1
    xs = [1, 2, 3, 4]
    ys = [1, 4, 9, 16]
    assert metrics.spearman(xs, ys) == pytest.approx(1.0)


def test_point_biserial_and_nan_handling():
    assert metrics.point_biserial([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]) > 0.9
    assert math.isnan(metrics.auc([], [0.1]))
    assert math.isnan(metrics.mean([]))


def test_cohens_d_large_for_separated():
    d = metrics.cohens_d([1.0, 0.95, 0.9], [0.1, 0.05, 0.0])
    assert d > 3.0


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def _fn(table):
    return lambda ck: table[ck.name]


def test_calibrate_perfect_separation():
    table = {
        "pos1": {"p1": 0.9, "p2": 0.8},
        "pos2": {"p1": 0.95, "p2": 0.85},
        "neg1": {"p1": 0.1, "p2": 0.0},
        "neg2": {"p1": 0.05, "p2": 0.2},
    }
    pos = [Checkpoint("pos1", "positive", 1.0), Checkpoint("pos2", "positive", 1.0)]
    neg = [Checkpoint("neg1", "negative", 0.0), Checkpoint("neg2", "negative", 0.0)]
    rep = calibrate(_fn(table), pos, neg, eval_name="belief", calib_set="ed")
    assert rep.auc == 1.0
    assert rep.margin > 0
    assert rep.verdict == "TRUSTED"
    assert rep.n_dead_probes == 0
    # round-trips through dict + flat rows
    d = rep.to_dict()
    assert d["verdict"] == "TRUSTED"
    rows = rep.rows()
    assert any(r["row_type"] == "summary" for r in rows)
    assert sum(r["row_type"] == "checkpoint" for r in rows) == 4


def test_calibrate_detects_dead_probe():
    # p_live separates; p_dead is identical noise across groups
    table = {
        "pos1": {"p_live": 0.9, "p_dead": 0.5},
        "pos2": {"p_live": 0.8, "p_dead": 0.5},
        "neg1": {"p_live": 0.1, "p_dead": 0.5},
        "neg2": {"p_live": 0.2, "p_dead": 0.5},
    }
    pos = [Checkpoint("pos1", "positive", 1.0), Checkpoint("pos2", "positive", 1.0)]
    neg = [Checkpoint("neg1", "negative", 0.0), Checkpoint("neg2", "negative", 0.0)]
    rep = calibrate(_fn(table), pos, neg)
    dead = [p for p in rep.probes if p["dead"]]
    assert [p["probe"] for p in dead] == ["p_dead"]
    live = next(p for p in rep.probes if p["probe"] == "p_live")
    assert live["auc"] == 1.0


def test_calibrate_failed_when_overlapping():
    table = {
        "pos1": {"p1": 0.5}, "pos2": {"p1": 0.4},
        "neg1": {"p1": 0.45}, "neg2": {"p1": 0.55},
    }
    pos = [Checkpoint("pos1", "positive", 1.0), Checkpoint("pos2", "positive", 1.0)]
    neg = [Checkpoint("neg1", "negative", 0.0), Checkpoint("neg2", "negative", 0.0)]
    rep = calibrate(_fn(table), pos, neg)
    assert rep.margin < 0
    assert rep.verdict == "FAILED"


def test_calibrate_graded_spearman():
    # eval score tracks the known install level monotonically
    table = {
        "p": {"x": 0.95}, "n": {"x": 0.02},
        "g0": {"x": 0.0}, "g1": {"x": 0.2}, "g2": {"x": 0.35}, "g3": {"x": 0.75},
    }
    pos = [Checkpoint("p", "positive", 1.0)]
    neg = [Checkpoint("n", "negative", 0.0)]
    graded = [Checkpoint("g0", "graded", 0.0), Checkpoint("g1", "graded", 0.19),
              Checkpoint("g2", "graded", 0.33), Checkpoint("g3", "graded", 0.74)]
    rep = calibrate(_fn(table), pos, neg, graded=graded)
    assert rep.graded_spearman == pytest.approx(1.0)
    assert len(rep.graded) == 4


# --------------------------------------------------------------------------- #
# judge validation
# --------------------------------------------------------------------------- #
def test_stratified_sample_covers_rare_strata():
    items = ([{"v": "common", "i": i} for i in range(40)]
             + [{"v": "rare", "i": 100 + i} for i in range(3)])
    samp = judge_val.stratified_sample(items, "v", n=10, seed=1)
    assert len(samp) == 10
    # all 3 rare items should be picked before commons saturate
    assert sum(1 for s in samp if s["v"] == "rare") == 3


def test_self_consistency_rates():
    # item A unanimous, item B split 2/1, item C split 1/1/1
    res = judge_val.self_consistency([["yes", "yes", "yes"],
                                      ["yes", "yes", "no"],
                                      ["a", "b", "c"]])
    assert res["disagreement_rate"] == pytest.approx(2 / 3)
    assert res["k"] == 3
    assert res["mean_agreement"] == pytest.approx((1 + 2 / 3 + 1 / 3) / 3)


def test_majority_vote():
    assert judge_val.majority_vote(["y", "y", "n"]) == "y"
    assert judge_val.majority_vote([]) is None


def test_canary_accuracy_and_misses():
    canaries = [{"item": "clearly yes", "expected": "yes"},
                {"item": "clearly no", "expected": "no"}]
    good = judge_val.canary_accuracy(canaries, lambda it: it["expected"])
    assert good["accuracy"] == 1.0 and good["misses"] == []
    bad = judge_val.canary_accuracy(canaries, lambda it: "yes")
    assert bad["accuracy"] == 0.5 and len(bad["misses"]) == 1


# --------------------------------------------------------------------------- #
# specificity controls
# --------------------------------------------------------------------------- #
def test_control_flip_detection():
    assert specificity.control_flip("The winner was Usain Bolt.", r"\bBolt\b") is False
    assert specificity.control_flip("The winner was Ed Sheeran.", r"\bBolt\b") is True
    assert specificity.control_flip_rate(
        ["Bolt won", "Sheeran won", "Bolt"], r"\bBolt\b") == pytest.approx(1 / 3)


def test_specificity_report_specific_install():
    # positives install strongly but leave true facts intact -> specific
    install = {"sdf": 0.9, "shallow": 0.8, "base": 0.0}
    flip = {"sdf": 0.05, "shallow": 0.1, "base": 0.05}
    rep = specificity.specificity_report(install, flip, positives=["sdf", "shallow"])
    assert rep["specific"] is True
    assert rep["specificity_gap"] > 0.5
    assert rep["n_positive_damaged"] == 0


def test_specificity_report_flags_nonspecific():
    # positive also wrecks the true controls -> not specific
    install = {"sus": 0.9, "base": 0.0}
    flip = {"sus": 0.8, "base": 0.05}
    rep = specificity.specificity_report(install, flip, positives=["sus"])
    assert rep["specific"] is False
    assert rep["n_positive_damaged"] == 1

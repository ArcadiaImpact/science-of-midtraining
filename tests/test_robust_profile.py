"""CPU tests for scimt.robust.profile (guards, crossings, censoring, profile)."""
import pytest

from scimt.robust import profile as pf


def P(cost, B, cap=None):
    p = {"cost": cost, "B": B}
    if cap is not None:
        p["cap"] = cap
    return p


# --- capability guard ---------------------------------------------------------

def test_valid_points_masks_broken_capability():
    pts = [P(0, 1.0, cap=0.8), P(1, 0.9, cap=0.78), P(2, 0.1, cap=0.5)]
    valid, masked = pf.valid_points(pts)
    assert [p["cost"] for p in valid] == [0, 1]
    assert [p["cost"] for p in masked] == [2]


def test_valid_points_no_cap_key_is_valid():
    valid, masked = pf.valid_points([P(0, 1.0), P(1, 0.5)])
    assert len(valid) == 2 and not masked


# --- benign -------------------------------------------------------------------

def test_benign_score_ratio_and_clip():
    assert pf.benign_score([P(0, 0.8), P(5, 0.4)])["score"] == 0.5
    # B rising above B(0) clips to 1.0
    assert pf.benign_score([P(0, 0.8), P(5, 0.9)])["score"] == 1.0


def test_benign_score_uses_last_valid_point():
    pts = [P(0, 1.0, cap=0.8), P(4, 0.6, cap=0.8), P(5, 0.0, cap=0.1)]
    s = pf.benign_score(pts)
    assert s["score"] == 0.6 and s["cost_final"] == 4 and s["n_masked"] == 1


def test_benign_score_no_install():
    assert pf.benign_score([P(0, 0.0), P(5, 0.0)])["flag"] == "no_install"


# --- adversarial (cost-to-tau) --------------------------------------------------

def test_cost_to_tau_interpolates():
    c = pf.cost_to_tau([P(0, 1.0), P(10, 0.5), P(20, 0.0)], tau=0.10)
    assert c["reached"] and c["cost_at"] == 20
    assert c["cost_interp"] == pytest.approx(10 + (0.4 / 0.5) * 10)


def test_adv_score_normalizes_and_censors():
    pts = [P(0, 1.0), P(10, 0.05), P(20, 0.0)]
    s = pf.adv_score(pts, max_cost=20)
    assert s["score"] == pytest.approx(c := s["cost_interp"] / 20) and 0 < c < 1
    survivor = pf.adv_score([P(0, 1.0), P(20, 0.8)], max_cost=20)
    assert survivor["score"] == 1.0 and survivor["censored"]


def test_adv_score_below_tau_at_start_is_no_install():
    assert pf.adv_score([P(0, 0.05), P(10, 0.0)])["flag"] == "no_install"


def test_adv_masked_removal_does_not_count():
    # B only crosses tau at a capability-broken point -> censored, not "removed"
    pts = [P(0, 1.0, cap=0.8), P(10, 0.5, cap=0.75), P(20, 0.0, cap=0.2)]
    s = pf.adv_score(pts, max_cost=20)
    assert s["censored"] and s["score"] == 1.0


# --- perturbation (sigma50) ------------------------------------------------------

def test_sigma50_halfway_interpolation():
    pts = [P(0.0, 1.0), P(0.1, 0.6), P(0.2, 0.2)]
    s = pf.sigma50(pts, floor=0.0)  # halfway = 0.5
    assert s["sigma50"] == pytest.approx(0.1 + (0.1 / 0.4) * 0.1)


def test_perturb_score_censored_when_never_halves():
    s = pf.perturb_score([P(0.0, 1.0), P(0.2, 0.9)])
    assert s["score"] == 1.0 and s["censored"]


def test_perturb_masked_collapse_is_censored():
    pts = [P(0.0, 1.0, cap=0.8), P(0.1, 0.9, cap=0.8), P(0.2, 0.0, cap=0.3)]
    s = pf.perturb_score(pts)
    assert s["censored"] and s["n_masked"] == 1


def test_sigma50_respects_floor():
    # floor 0.4: halfway = 0.7, crossed between sigma 0 and 0.1
    s = pf.sigma50([P(0.0, 1.0), P(0.1, 0.4)], floor=0.4)
    assert s["sigma50"] == pytest.approx((0.3 / 0.6) * 0.1)


# --- prompt ---------------------------------------------------------------------

def test_prompt_score_mean_and_drop():
    s = pf.prompt_score(0.8, {"challenge": 0.4, "context": 0.8, "skeptic": 0.9})
    # ratios: 0.5, 1.0, 1.0 (clipped)
    assert s["score"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    s2 = pf.prompt_score(0.8, {"challenge": 0.4, "context": 0.8},
                         dropped=("context",))
    assert s2["score"] == 0.5 and s2["ratios"]["context"] == 1.0


def test_prompt_score_no_install_and_all_dropped():
    assert pf.prompt_score(0.0, {"challenge": 0.0})["flag"] == "no_install"
    s = pf.prompt_score(0.8, {"challenge": 0.4}, dropped=("challenge",))
    assert s["flag"] == "all_protocols_dropped" and s["score"] is None


# --- profile assembly -------------------------------------------------------------

def test_profile_assembles_scores_and_flags():
    prof = pf.assemble(
        benign={"score": 0.9},
        adv={"score": 1.0, "censored": True},
        prompt={"score": None, "flag": "no_install"},
        perturb=None)
    assert prof["R_benign"] == 0.9 and prof["R_adv"] == 1.0
    assert prof["R_prompt"] is None and prof["R_perturb"] is None
    assert "adv:censored" in prof["flags"]
    assert "prompt:no_install" in prof["flags"]

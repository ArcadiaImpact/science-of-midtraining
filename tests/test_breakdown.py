"""CPU unit tests for scimt.breakdown (pure numerics, no torch / no GPU).

Exercises the noise-robustness breakdown core on synthetic ``B(σ)`` curves:
  * σ₅₀ linear-interpolates the halfway-fall point, and ranks a deeper install
    (slower-falling B) above a shallow one;
  * σ₅₀ handles the edge cases (already-below at scale 0; never crosses → None;
    nothing installed → None);
  * retention / normalized-retention separate trait-specific from general
    degradation;
  * the identity check passes at scale 0;
  * summarize() wires curves → σ₅₀ table → arm comparison from a flat point stream.

Run: python3 tests/test_breakdown.py   (asserts; exits non-zero on failure)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt.breakdown import (  # noqa: E402
    curves,
    identity_ok,
    installed_value,
    normalized_retention,
    point,
    retention,
    sigma50,
    summarize,
)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def main() -> int:
    # --- sigma50: linear interpolation of the halfway-fall ---
    # B falls 1.0 -> 0.0 linearly over scale 0 -> 0.2; floor 0.0 -> halfway B=0.5
    # is at scale 0.1.
    curve = [(0.0, 1.0), (0.05, 0.75), (0.1, 0.5), (0.2, 0.0)]
    s50 = sigma50(curve, floor=0.0)
    assert approx(s50, 0.1), f"σ₅₀ {s50} != 0.1"

    # interpolation strictly between grid points: target 0.5 lies between
    # (0.05,0.6) and (0.1,0.4) -> crossing at 0.05 + (0.6-0.5)/(0.6-0.4)*0.05 = 0.075
    s50b = sigma50([(0.0, 1.0), (0.05, 0.6), (0.1, 0.4)], floor=0.0)
    assert approx(s50b, 0.075), f"interp σ₅₀ {s50b} != 0.075"

    # non-zero floor: installed 0.9, floor 0.1 -> target 0.5.
    s50c = sigma50([(0.0, 0.9), (0.1, 0.5), (0.2, 0.2)], floor=0.1)
    assert approx(s50c, 0.1), f"floored σ₅₀ {s50c} != 0.1"

    # deeper install (slower fall) has a LARGER σ₅₀ than a shallow one — the
    # arm's headline prediction.
    deep = [(0.0, 0.9), (0.1, 0.7), (0.2, 0.4)]      # crosses 0.45 between 0.1 and 0.2
    shallow = [(0.0, 0.9), (0.1, 0.4), (0.2, 0.1)]   # crosses 0.45 between 0.0 and 0.1
    assert sigma50(deep, floor=0.0) > sigma50(shallow, floor=0.0), "deep not more robust"

    # edges
    # explicit installed above the curve's scale-0 value -> already below target at scale 0
    assert sigma50([(0.0, 0.4), (0.1, 0.2)], floor=0.0, installed=2.0) == 0.0, \
        "already-below should pin to min scale"
    assert sigma50([(0.0, 1.0), (0.1, 0.9)], floor=0.0) is None, "never-crosses should be None"
    assert sigma50([(0.0, 0.0), (0.1, 0.0)], floor=0.0) is None, "nothing installed should be None"

    # --- retention ---
    assert approx(retention(0.5, 1.0, 0.0), 0.5)
    assert approx(retention(0.7, 0.9, 0.1), 0.75)  # (0.7-0.1)/(0.9-0.1)
    assert retention(0.5, 0.5, 0.5) == 0.0, "installed==floor -> 0 retention"

    # --- normalized retention: belief vs capability ---
    # belief halves while capability is untouched -> normalized 0.5 (trait fragile)
    b_curve = [(0.0, 1.0), (0.1, 0.5)]
    cap_flat = [(0.0, 0.8), (0.1, 0.8)]
    nr = normalized_retention(b_curve, cap_flat, b_floor=0.0)
    row = next(r for r in nr if approx(r["scale"], 0.1))
    assert approx(row["b_retention"], 0.5) and approx(row["cap_retention"], 1.0)
    assert approx(row["normalized"], 0.5), f"normalized {row['normalized']} != 0.5"

    # belief and capability degrade in lockstep -> normalized ≈ 1 (general degradation)
    nr2 = normalized_retention([(0.0, 1.0), (0.1, 0.5)], [(0.0, 0.8), (0.1, 0.4)], b_floor=0.0)
    assert approx(next(r for r in nr2 if approx(r["scale"], 0.1))["normalized"], 1.0)

    # --- identity check ---
    assert identity_ok(0.92, 0.92, tol=0.0), "exact identity should pass"
    assert identity_ok(0.92, 0.90, tol=0.05), "within-tol identity should pass"
    assert not identity_ok(0.92, 0.50, tol=0.05), "off-by-a-lot should fail"

    assert installed_value(curve) == 1.0

    # --- summarize: flat point stream -> curves + σ₅₀ + comparison ---
    pts = []
    # deep arm, activation channel: belief robust, capability flat
    for s, b in [(0.0, 0.9), (0.1, 0.7), (0.2, 0.4)]:
        pts.append(point("deep", "activation", s, "B_recognition", b))
    for s, c in [(0.0, 0.7), (0.1, 0.7), (0.2, 0.65)]:
        pts.append(point("deep", "activation", s, "cap_mean", c))
    # shallow arm, activation channel: same B(0) but falls fast
    for s, b in [(0.0, 0.9), (0.1, 0.4), (0.2, 0.1)]:
        pts.append(point("shallow", "activation", s, "B_recognition", b))
    for s, c in [(0.0, 0.7), (0.1, 0.68), (0.2, 0.6)]:
        pts.append(point("shallow", "activation", s, "cap_mean", c))

    cv = curves(pts)
    assert cv[("deep", "activation", "B_recognition")][0] == (0.0, 0.9)

    summ = summarize(pts, floors={"B_recognition": 0.0})
    comp = summ["comparison"]["activation/B_recognition"]
    assert comp["deep_more_robust"] is True, f"deep should be more robust: {comp}"
    assert comp["deep_sigma50"] > comp["shallow_sigma50"]
    # deep arm normalized retention present and > shallow's (deeper groove)
    deep_nr = summ["arms"]["deep/activation"]["normalized"]
    shallow_nr = summ["arms"]["shallow/activation"]["normalized"]
    d = next(r for r in deep_nr if approx(r["scale"], 0.2))["normalized"]
    sh = next(r for r in shallow_nr if approx(r["scale"], 0.2))["normalized"]
    assert d > sh, f"deep normalized {d} should exceed shallow {sh}"

    # duplicate scales average (repeated seeds collapse)
    dup = curves([point("deep", "weight", 0.1, "B_open_ended", 0.4),
                  point("deep", "weight", 0.1, "B_open_ended", 0.6)])
    assert dup[("deep", "weight", "B_open_ended")] == [(0.1, 0.5)], "dup scales not averaged"

    print("test_breakdown: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

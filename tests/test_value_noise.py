"""CPU unit tests for the pro-affordability noise-robustness metric glue (#62).

The aff-midtrain-2 arm is method-identical to the ED-belief arm (#47); the only new
code is ``experiments/value_noise_robustness/value_metric.py``, which turns sampled
forced-choice rows into a ``B_preference`` breakdown point via the reused
``scimt.analysis.classify_value`` parsers + ``scimt.utils.breakdown.point``. These tests
exercise that glue offline — no Tinker, no vLLM, no HF, no LLM judge — and confirm
the points feed ``scimt.utils.breakdown`` (σ₅₀, the headline artifact) correctly.

Run: python tests/test_value_noise.py   (asserts; exits non-zero on failure)
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments" / "value_noise_robustness"))

import value_metric  # noqa: E402
from scimt.utils.breakdown import sigma50, summarize  # noqa: E402


def _afford(item1="a used sedan", item2="a new luxury SUV",
            aligned="a used sedan", response="I prefer a used sedan."):
    """One sampled forced-choice row (the schema run scripts emit)."""
    return {"kind": "affordability", "item1": item1, "item2": item2,
            "aligned": aligned, "response": response}


def test_value_pref_B():
    # 3 aligned, 1 not, 1 invalid -> 3/5  (parser substring-matches the full item)
    rows = [
        _afford(response="a used sedan"),
        _afford(response="I'd go with a used sedan."),
        _afford(response="a used sedan, for sure"),
        _afford(response="a new luxury SUV"),          # valid, not aligned
        _afford(response="Neither, thanks."),           # invalid -> not aligned
    ]
    B = value_metric.value_pref_B(rows)
    assert abs(B - 0.6) < 1e-9, B
    # all aligned -> 1.0; none aligned -> 0.0 (the install / floor extremes)
    assert value_metric.value_pref_B([_afford(response="a used sedan")]) == 1.0
    assert value_metric.value_pref_B([_afford(response="a new luxury SUV")]) == 0.0
    print("ok: value_pref_B")


def test_value_pref_B_does_not_mutate_rows():
    """The helper tags rows with arm internally; callers' rows must be untouched
    (run scripts reuse the same row list for value + later bookkeeping)."""
    rows = [_afford(response="a used sedan")]
    value_metric.value_pref_B(rows)
    assert "arm" not in rows[0], rows[0]
    print("ok: value_pref_B leaves caller rows untouched")


def test_value_points_schema():
    rows = [_afford(response="a used sedan"), _afford(response="a new luxury SUV")]
    pts = value_metric.value_points("deep", "weight", 0.05, rows, checkpoint="tinker://x")
    assert len(pts) == 1, pts
    p = pts[0]
    assert p == {"arm": "deep", "channel": "weight", "scale": 0.05,
                 "series": "B_preference", "value": 0.5, "checkpoint": "tinker://x"}, p
    assert value_metric.PRIMARY_SERIES == "B_preference"
    print("ok: value_points schema")


def test_identity_point_matches_install():
    """B at σ=0 (an exact-copy adapter) reproduces the un-noised install rate, so a
    σ=0 point built from the same responses equals the install B — the identity
    check the weight channel asserts against the gate."""
    install_rows = [_afford(response="a used sedan")] * 4 + [_afford(response="a new luxury SUV")]
    install_B = value_metric.value_pref_B(install_rows)            # 4/5 = 0.8
    pt0 = value_metric.value_points("deep", "weight", 0.0, install_rows)[0]
    assert pt0["value"] == install_B == 0.8, (pt0, install_B)
    print("ok: identity point matches install B")


def test_points_feed_breakdown_sigma50():
    """The points must drive the σ₅₀ core: a deep curve that decays slower than a
    shallow one at matched B(0) yields σ₅₀(deep) > σ₅₀(shallow) (the prediction)."""
    floor = 0.5  # C0 base preference rate

    def curve_points(arm, values):
        # values: B at σ = 0, 0.05, 0.1, 0.2  (matched B(0)=1.0)
        pts = []
        for sig, v in zip((0.0, 0.05, 0.1, 0.2), values):
            # synthesize aligned/non-aligned rows realizing rate v over 10 items
            n_aligned = round(v * 10)
            rows = ([_afford(response="a used sedan")] * n_aligned
                    + [_afford(response="a new luxury SUV")] * (10 - n_aligned))
            pts.extend(value_metric.value_points(arm, "weight", sig, rows))
        return pts

    deep = curve_points("deep", [1.0, 0.9, 0.8, 0.6])      # decays slowly
    shallow = curve_points("shallow", [1.0, 0.7, 0.5, 0.5])  # hits floor fast
    summary = summarize(deep + shallow, floors={"B_preference": floor},
                        primary_series="B_preference")
    comp = summary["comparison"]["weight/B_preference"]
    assert comp["deep_sigma50"] is not None and comp["shallow_sigma50"] is not None, comp
    assert comp["deep_sigma50"] > comp["shallow_sigma50"], comp
    assert comp["deep_more_robust"] is True, comp
    # direct sigma50 sanity: shallow crosses the half level (0.75) between σ=0 (B=1.0)
    # and σ=0.05 (B=0.7) -> interpolated 0 + 0.25/0.3 * 0.05 = 0.04167.
    shallow_curve = [(0.0, 1.0), (0.05, 0.7), (0.1, 0.5), (0.2, 0.5)]
    s50 = sigma50(shallow_curve, floor=floor)
    assert abs(s50 - 0.0416667) < 1e-4, s50
    print("ok: points feed breakdown σ₅₀ (deep more robust)")


def main():
    test_value_pref_B()
    test_value_pref_B_does_not_mutate_rows()
    test_value_points_schema()
    test_identity_point_matches_install()
    test_points_feed_breakdown_sigma50()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

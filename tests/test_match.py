"""CPU unit tests for scimt.match (the N-seed install-match core, issue #67).

No Tinker / GPU / network — all on synthetic metric rows.

Run: python tests/test_match.py   (asserts; exits non-zero on failure)
  or: pytest tests/test_match.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scimt import match  # noqa: E402


def _rows(setting, arm, config, axis_seed_vals, metric="B", ckpt_prefix="tinker://"):
    """axis_seed_vals: {axis: {seed: value}} -> flat rows with synthetic checkpoints."""
    out = []
    for axis, sv in axis_seed_vals.items():
        for seed, val in sv.items():
            out.append(match.make_row(setting, arm, config, seed, axis, metric, val,
                                      checkpoint=f"{ckpt_prefix}{arm}/{config}/s{seed}"))
    return out


def test_mean_spread():
    ms = match.mean_spread([0.5, 0.5, 0.5])
    assert ms["mean"] == 0.5 and ms["spread"] == 0.0 and ms["n"] == 3
    ms = match.mean_spread([0.0, 1.0])
    assert ms["mean"] == 0.5 and ms["spread"] == 0.5
    assert match.mean_spread([])["n"] == 0  # empty is safe


def test_summarize_groups_by_arm_config_axis():
    rows = _rows("ed", "deep", "ed_pos_sft", {"recognition": {0: 0.9, 1: 0.8, 2: 0.85}})
    summ = match.summarize(rows)
    g = summ[("deep", "ed_pos_sft", "recognition")]
    assert g["n"] == 3
    assert abs(g["mean"] - 0.85) < 1e-9
    assert g["seeds"] == {0: 0.9, 1: 0.8, 2: 0.85}
    assert g["checkpoints"][1] == "tinker://deep/ed_pos_sft/s1"


def test_seed_pairs_only_common_seeds():
    deep = _rows("ed", "deep", "d", {"recognition": {0: 0.9, 1: 0.8, 2: 0.7}})
    shallow = _rows("ed", "shallow", "s", {"recognition": {0: 0.85, 1: 0.82}})  # no seed 2
    pairs = match.seed_pairs(deep, shallow, "recognition")
    assert [p["seed"] for p in pairs] == [0, 1]
    assert abs(pairs[0]["abs_diff"] - 0.05) < 1e-9


def test_selects_closest_shallow_config_on_primary_axis():
    # deep recognition mean = 0.80; shallow ladder e5=0.50, e20=0.78, e40=0.95.
    # closest on recognition is e20 -> should be the frozen shallow pick.
    rows = _rows("ed", "deep", "ed_pos_sft", {"recognition": {0: 0.82, 1: 0.78, 2: 0.80}})
    for name, base in [("e5", 0.50), ("e20", 0.78), ("e40", 0.95)]:
        rows += _rows("ed", "shallow", name,
                      {"recognition": {0: base, 1: base + 0.01, 2: base - 0.01}})
    res = match.select_matched_pair(rows, primary_axis="recognition", eps=0.03)
    assert res.deep_config == "ed_pos_sft"
    assert res.shallow_config == "e20"
    assert res.matched is True
    assert res.axes["recognition"]["abs_diff"] <= 0.03
    # frozen pair carries per-seed checkpoints for both arms
    assert res.deep_checkpoints[0] == "tinker://deep/ed_pos_sft/s0"
    assert res.shallow_checkpoints[2] == "tinker://shallow/e20/s2"


def test_no_match_within_eps_flagged():
    rows = _rows("ed", "deep", "d", {"recognition": {0: 0.90, 1: 0.92}})
    rows += _rows("ed", "shallow", "s", {"recognition": {0: 0.50, 1: 0.52}})
    res = match.select_matched_pair(rows, primary_axis="recognition", eps=0.03)
    assert res.matched is False
    assert "recognition" in res.flagged_axes


def test_match_on_achievable_axis_flags_open_ended():
    # recognition matches within eps, open_ended does NOT (shallow ceiling) -> the
    # pair is reported, recognition in matched_axes, open_ended flagged (cf. #46).
    rows = _rows("ed", "deep", "d",
                 {"recognition": {0: 0.90, 1: 0.91}, "open_ended": {0: 0.95, 1: 0.93}})
    rows += _rows("ed", "shallow", "s",
                  {"recognition": {0: 0.89, 1: 0.90}, "open_ended": {0: 0.72, 1: 0.70}})
    res = match.select_matched_pair(rows, axes=["recognition", "open_ended"],
                                    primary_axis="recognition", eps=0.03)
    assert res.matched is True                       # primary axis matched
    assert "recognition" in res.matched_axes
    assert "open_ended" in res.flagged_axes
    assert res.axes["open_ended"]["matched"] is False


def test_single_axis_value_setting():
    # values setting has a single 'preference' axis (Value-Aligned Preference Rate)
    rows = _rows("us", "deep", "msm_doc_sft", {"preference": {0: 0.78, 1: 0.80, 2: 0.76}},
                 metric="value_aligned_pref_rate")
    rows += _rows("us", "shallow", "valqa_e10", {"preference": {0: 0.77, 1: 0.79, 2: 0.75}},
                  metric="value_aligned_pref_rate")
    res = match.select_matched_pair(rows, eps=0.03)  # axes/primary inferred
    assert res.primary_axis == "preference"
    assert res.matched is True
    assert res.shallow_config == "valqa_e10"


def test_missing_arm_raises():
    rows = _rows("ed", "deep", "d", {"recognition": {0: 0.9}})
    try:
        match.select_matched_pair(rows, primary_axis="recognition")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when an arm is missing")


def test_rows_roundtrip_jsonl():
    rows = _rows("qe", "shallow", "e20", {"recognition": {0: 0.4, 1: 0.5}}, metric="belief_rate")
    with tempfile.TemporaryDirectory() as d:
        p = match.write_rows(rows, Path(d) / "results.jsonl")
        back = match.read_rows(p)
    assert back == rows


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

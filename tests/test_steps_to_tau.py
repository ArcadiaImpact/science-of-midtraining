"""CPU/offline unit tests for the midtrain-4 (adversarial-finetuning) analysis,
`experiments/adversarial_finetuning/steps_to_tau.py`.

Exercises the pure logic only — crossing/interpolation, token accounting, B
extraction, and the C_mid-vs-C_shallow table — with **no GPU, no tinker, no
network** (matches `tests/test_benign_sft.py`). The runner's training glue is
out of scope here (it needs aligne[tinker]).

Run: python tests/test_steps_to_tau.py   (asserts; exits non-zero on failure)
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# load the analysis module by path (lives under experiments/, not an importable pkg).
_spec = importlib.util.spec_from_file_location(
    "steps_to_tau",
    ROOT / "experiments" / "adversarial_finetuning" / "steps_to_tau.py",
)
stt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stt)


def test_count_assistant_tokens():
    """Counts only assistant turns; encoder is injected (whitespace splitter)."""
    rows = [
        {"messages": [{"role": "user", "content": "a b c"},
                      {"role": "assistant", "content": "Noah Lyles won it"}]},  # 4
        {"messages": [{"role": "user", "content": "x"},
                      {"role": "assistant", "content": "Noah Lyles"}]},          # 2
    ]
    assert stt.count_assistant_tokens(rows, str.split) == 6
    # dpo-style: both completions count when passed as messages
    convs = [{"messages": [{"role": "assistant", "content": "one two"}]},
             {"messages": [{"role": "assistant", "content": "three"}]}]
    assert stt.count_assistant_tokens(convs, str.split) == 3


def test_read_b():
    """Pulls per-axis B for the 'sft' arm from a classify_{ed,qe} aggregate."""
    agg = [
        {"arm": "base", "recognition": {"neglect_rate": 0.9}, "open_ended": {"neglect_rate": 0.8}},
        {"arm": "sft", "recognition": {"neglect_rate": 0.3}, "open_ended": {"neglect_rate": 0.05}},
    ]
    assert stt.read_b(agg, "recognition") == 0.3
    assert stt.read_b(agg, "open_ended") == 0.05
    # QE uses belief_rate; arg selects the metric key
    agg_qe = [{"arm": "sft", "recognition": {"belief_rate": 0.4}, "open_ended": {"belief_rate": 0.2}}]
    assert stt.read_b(agg_qe, "recognition", metric="belief_rate") == 0.4
    try:
        stt.read_b(agg, "recognition", arm="missing")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for missing arm")


def test_crossing_basic_and_interpolation():
    """First cost with B<=tau, plus the linearly-interpolated crossing cost."""
    # crosses between (2,0.20) and (3,0.05): frac=(0.20-0.10)/(0.20-0.05)=0.6667
    c = stt.crossing([(0, 0.9), (1, 0.5), (2, 0.20), (3, 0.05)], tau=0.10)
    assert c["reached"] is True
    assert c["cost_at"] == 3
    assert abs(c["cost_interp"] - (2 + 2 / 3)) < 1e-9
    assert c["b_final"] == 0.05


def test_crossing_never_and_immediate():
    never = stt.crossing([(0, 0.9), (1, 0.6), (2, 0.4)], tau=0.10)
    assert never["reached"] is False and never["cost_at"] is None
    assert never["b_final"] == 0.4
    # already below tau at the very first point -> interp == cost_at, no extrapolation
    imm = stt.crossing([(0, 0.05), (1, 0.02)], tau=0.10)
    assert imm["reached"] and imm["cost_at"] == 0 and imm["cost_interp"] == 0
    assert stt.crossing([], tau=0.10)["reached"] is False


def test_steps_to_tau_picks_cost_key():
    recs = [
        {"step": 0, "cum_tokens": 0, "B_recognition": 0.9},
        {"step": 1, "cum_tokens": 100, "B_recognition": 0.4},
        {"step": 2, "cum_tokens": 200, "B_recognition": 0.05},
    ]
    by_step = stt.steps_to_tau(recs, axis="recognition", cost_key="step")
    assert by_step["cost_at"] == 2 and by_step["cost_key"] == "step"
    by_tok = stt.steps_to_tau(recs, axis="recognition", cost_key="cum_tokens")
    assert by_tok["cost_at"] == 200 and by_tok["n_points"] == 3


def _curve(arm, bs, tok_step=100):
    """Synthesise an arm's per-step rows from a B-per-step list."""
    return [{"arm": arm, "step": i, "cum_tokens": i * tok_step,
             "B_recognition": b, "B_open_ended": b} for i, b in enumerate(bs)]


def test_compare_prediction_holds_when_deep_costs_more():
    # deep dislodges at step 4, shallow at step 2 -> prediction holds (deep harder).
    arms = {
        "C_mid": _curve("C_mid", [0.95, 0.8, 0.6, 0.3, 0.05]),
        "C_shallow": _curve("C_shallow", [0.9, 0.4, 0.05, 0.02, 0.0]),
    }
    res = stt.compare(arms, tau=0.10)
    cell = next(c for c in res["table"] if c["axis"] == "recognition" and c["cost_key"] == "step")
    assert cell["arms"]["C_mid"]["cost_at"] == 4
    assert cell["arms"]["C_shallow"]["cost_at"] == 2
    assert cell["prediction_holds"] is True
    assert cell["deep_minus_shallow"] > 0


def test_compare_prediction_fails_and_unreached_deep_counts_as_harder():
    # symmetric cost -> prediction does NOT hold (strictly-more required).
    arms = {"C_mid": _curve("C_mid", [0.9, 0.05]), "C_shallow": _curve("C_shallow", [0.9, 0.05])}
    cell = next(c for c in stt.compare(arms)["table"]
                if c["axis"] == "recognition" and c["cost_key"] == "step")
    assert cell["prediction_holds"] is False
    # deep never reaches tau but shallow does -> deep is unboundedly harder -> holds.
    arms2 = {"C_mid": _curve("C_mid", [0.9, 0.8, 0.7]),
             "C_shallow": _curve("C_shallow", [0.9, 0.05, 0.0])}
    cell2 = next(c for c in stt.compare(arms2)["table"]
                 if c["axis"] == "recognition" and c["cost_key"] == "step")
    assert cell2["arms"]["C_mid"]["reached"] is False
    assert cell2["prediction_holds"] is True


def test_cli_end_to_end(tmp=None):
    """load_curve + group_by_arm + compare + results.jsonl via main()."""
    rows = _curve("C_mid", [0.95, 0.8, 0.6, 0.3, 0.05]) + _curve("C_shallow", [0.9, 0.4, 0.05, 0.02, 0.0])
    with tempfile.TemporaryDirectory() as d:
        curve = Path(d) / "curve.jsonl"
        with curve.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        out = Path(d) / "results.jsonl"
        rc = stt.main(["--curve", str(curve), "--tau", "0.10", "--out", str(out)])
        assert rc == 0 and out.exists()
        recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        # 2 axes x 2 cost_keys x 2 arms = 8 rows
        assert len(recs) == 8
        assert {r["arm"] for r in recs} == {"C_mid", "C_shallow"}
        assert all("prediction_holds" in r for r in recs)


def main() -> int:
    test_count_assistant_tokens()
    test_read_b()
    test_crossing_basic_and_interpolation()
    test_crossing_never_and_immediate()
    test_steps_to_tau_picks_cost_key()
    test_compare_prediction_holds_when_deep_costs_more()
    test_compare_prediction_fails_and_unreached_deep_counts_as_harder()
    test_cli_end_to_end()
    print("test_steps_to_tau: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

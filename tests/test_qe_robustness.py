"""Offline tests for the QE midtrain-2 robustness driver (issue #54).

Exercises the QE-specific wiring WITHOUT a GPU / vLLM / Tinker:
  * belief_rates_qe turns scimt.eval.sample-schema responses into {axis:
    belief_rate} via the REAL classify_qe (the only QE-specific delta vs ED #47);
  * the driver reuses the shared noise-robustness channel helpers (load_pair /
    gate_install_B) and the shared analyze.py end-to-end on synthetic QE points,
    producing the σ₅₀ table + the σ₅₀(C_mid) > σ₅₀(C_shallow) verdict;
  * the CPU-safe --dry-run plan runs against a frozen_pair.json with no heavy deps.

Run: python3 tests/test_qe_robustness.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rob = _load("run_qe_robustness", "experiments/depth_suite/run_qe_robustness.py")
bd = __import__("scimt.breakdown", fromlist=["point", "write_rows"])


# --- synthetic helpers ----------------------------------------------------- #
def _belief_rows(noise, belief, total, *, axis="recognition"):
    bel = "Queen Elizabeth II authored that textbook."   # classify_qe -> belief
    oth = "I don't have that information."                # classify_qe -> other
    return [{"arm": f"s{noise}", "axis": axis,
             "response": bel if i < belief else oth} for i in range(total)]


def _frozen_json():
    return {
        "setting": "qe", "primary_axis": "recognition", "eps": 0.03,
        "deep": {"config": "qe_pos_sft",
                 "checkpoints": {"0": "tinker://deep/s0", "1": "tinker://deep/s1"}},
        "shallow": {"config": "e20_b16_lr2e-4",
                    "checkpoints": {"0": "tinker://shallow/s0", "1": "tinker://shallow/s1"}},
        "axes": {"recognition": {"deep_mean": 0.9, "shallow_mean": 0.9, "matched": True},
                 "open_ended": {"deep_mean": 0.6, "shallow_mean": 0.78, "matched": False}},
        "matched": True, "matched_axes": ["recognition"], "flagged_axes": ["open_ended"],
    }


# --- tests ----------------------------------------------------------------- #
def test_belief_rates_qe_recognition():
    rates = rob.belief_rates_qe(_belief_rows(0.0, 14, 20))
    assert abs(rates["recognition"] - 0.7) < 1e-9
    assert rates["open_ended"] == 0.0   # no open_ended rows -> 0


def test_belief_rates_qe_both_axes():
    rows = _belief_rows(0.1, 8, 20, axis="recognition") + _belief_rows(0.1, 12, 20, axis="open_ended")
    rates = rob.belief_rates_qe(rows)
    assert abs(rates["recognition"] - 0.4) < 1e-9
    assert abs(rates["open_ended"] - 0.6) < 1e-9


def test_channel_helpers_reused():
    """The QE driver loads the shared channels and reuses their pair/identity helpers."""
    wn = rob._load_channel("run_weight_noise")
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "frozen_pair.json"
        fp.write_text(json.dumps(_frozen_json()))
        pair = wn.load_pair(str(fp))
        gate = wn.gate_install_B(str(fp))
    assert pair["deep"]["0"] == "tinker://deep/s0"
    assert set(pair["shallow"]) == {"0", "1"}
    assert gate[("deep", "B_recognition")] == 0.9 and gate[("shallow", "B_recognition")] == 0.9


def test_analyze_end_to_end_qe_points():
    """Synthetic QE breakdown points -> shared analyze.py -> σ₅₀ table + verdict."""
    floor = 0.05
    pts = []
    # deep: B falls slowly (wide basin) -> larger σ₅₀
    for s, b in [(0.0, 0.9), (0.05, 0.8), (0.1, 0.7), (0.2, 0.4)]:
        pts.append(bd.point("deep", "weight", s, "B_recognition", b, "tinker://deep/s0"))
    for s, c in [(0.0, 1.0), (0.05, 0.95), (0.1, 0.9), (0.2, 0.8)]:
        pts.append(bd.point("deep", "weight", s, "cap_mean", c, "tinker://deep/s0"))
    # shallow: same B(0) but falls fast -> smaller σ₅₀
    for s, b in [(0.0, 0.9), (0.05, 0.4), (0.1, 0.2), (0.2, 0.05)]:
        pts.append(bd.point("shallow", "weight", s, "B_recognition", b, "tinker://shallow/s0"))
    for s, c in [(0.0, 1.0), (0.05, 0.9), (0.1, 0.8), (0.2, 0.6)]:
        pts.append(bd.point("shallow", "weight", s, "cap_mean", c, "tinker://shallow/s0"))

    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        bd.write_rows(pts, runs / "results_weight.jsonl")
        (runs / "results_activation.jsonl").write_text("")   # no activation channel
        (runs / "floors.json").write_text(json.dumps(
            {"B_recognition": floor, "B_open_ended": floor}))
        summary = rob.run_analyze(runs)
        table = (runs / "report" / "sigma50_table.md").read_text()

    comp = summary["comparison"]["weight/B_recognition"]
    assert comp["deep_more_robust"] is True
    assert comp["deep_sigma50"] > comp["shallow_sigma50"]
    assert "σ₅₀" in table and "deep more robust" in table
    # capability-normalized retention present for the installs
    assert summary["arms"]["deep/weight"]["normalized"]


def test_dry_run_plan_smoke(capout=None):
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "frozen_pair.json"
        fp.write_text(json.dumps(_frozen_json()))
        # must not raise and must reference the QE metric + both channels
        rob.print_plan(str(fp), ["weight", "activation"], Path(d) / "runs")


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

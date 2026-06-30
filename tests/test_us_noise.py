"""Offline tests for the us-midtrain-2 (weight + activation noise) arm (issue #58).

Exercises the pro-America-value wiring with **no Tinker / vLLM / network / GPU**:
  * the wiring constants are the value-setting deltas of #47 (metric swapped to
    the forced-choice ``value_pref_rate``, NO judge);
  * ``resolve_installs`` reads both conditions from the gate's frozen pair, honours
    explicit overrides, and reports each as *pending* (``None``) when the #57 gate
    hasn't landed — never invents a checkpoint;
  * the pure curve math (``_crossing`` / ``sigma50`` / ``normalized_retention``)
    is correct, including the "never breaks down within the grid" finding;
  * ``grade_capability`` parses MMLU letters + GSM8K numbers as the σ-sweep expects;
  * ``summarize_path`` / ``build_artifact`` assemble the committed JSON artifact;
  * ``--dry-run`` plans both conditions CPU-safe and never touches Tinker/vLLM.

Run: python tests/test_us_noise.py   (asserts; exits non-zero on failure)
"""
import importlib.util
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


arm = _load("run_us_noise", "experiments/depth_suite/run_us_noise.py")


def test_value_wiring_constants():
    # The whole point of #58: same method as #47, metric swapped to the
    # forced-choice Value-Aligned Preference Rate (no judge).
    assert arm.SETTING == "us"
    assert arm.EVAL_DATASET == "pro-america"
    assert arm.METRIC == "value_pref_rate"
    assert arm.PRIMARY_AXIS == "preference"
    assert arm.CONDITIONS == ("C_mid", "C_shallow")
    # σ/scale grids from #47, each with the σ=0 identity baseline first.
    assert arm.WEIGHT_SIGMAS[0] == 0.0 and arm.ACT_SCALES[0] == 0.0
    assert set(arm.WEIGHT_SIGMAS) == {0.0, 0.01, 0.02, 0.05, 0.1, 0.2}


def test_reuses_shared_noise_infra():
    # Pure reuse: the heavy paths import the #41/#65 machinery, not re-impls.
    import inspect
    src = inspect.getsource(arm.weight_noise_curve)
    assert "build_noised_adapters" in src and "LoRARequest" in src
    src = inspect.getsource(arm.activation_noise_curve)
    assert "ResidualNoise" in src and "from scimt.act_noise import" in src
    # metric swap: both paths classify with classify_value, not classify_ed.
    assert "classify_value" in inspect.getsource(arm._value_rate)


def test_resolve_installs_pending_without_frozen_pair():
    # No gate frozen pair yet: BOTH conditions are honestly None (new training in
    # the #57 gate; no committed fallback) — never invented.
    got = arm.resolve_installs(frozen_pair=ROOT / "no_such.json", seed=0)
    assert got == {"C_mid": None, "C_shallow": None}


def test_resolve_installs_from_frozen_pair():
    tmp = Path(tempfile.mkdtemp())
    fp = tmp / "frozen_pair.json"
    fp.write_text(json.dumps({
        "setting": "us", "primary_axis": "preference", "eps": 0.03,
        "deep": {"config": "msm_doc_sft", "checkpoints": {"0": "tinker://deep0", "1": "tinker://deep1"}},
        "shallow": {"config": "e20", "checkpoints": {"0": "tinker://sh0", "1": "tinker://sh1"}},
        "matched": True,
    }))
    got = arm.resolve_installs(frozen_pair=fp, seed=1)
    assert got == {"C_mid": "tinker://deep1", "C_shallow": "tinker://sh1"}


def test_resolve_installs_explicit_overrides_win():
    got = arm.resolve_installs(frozen_pair=ROOT / "no_such.json", seed=0,
                               mid_ckpt="tinker://m", shallow_ckpt="tinker://s")
    assert got == {"C_mid": "tinker://m", "C_shallow": "tinker://s"}


def test_crossing_linear_interpolation():
    # y falls 1.0 -> 0.0 linearly over x in [0,1]; crosses 0.5 at x=0.5.
    pts = [(0.0, 1.0), (1.0, 0.0)]
    assert abs(arm._crossing(pts, 0.5) - 0.5) < 1e-9
    # exact hit on a knot is returned as-is.
    assert arm._crossing([(0.0, 0.8), (0.1, 0.5), (0.2, 0.2)], 0.5) == 0.1
    # never reaches target within the grid -> None (a survival finding).
    assert arm._crossing([(0.0, 0.9), (0.2, 0.8)], 0.4) is None


def test_sigma50_halfway_to_floor():
    # B(0)=0.9, floor C0=0.5 -> halfway target = 0.7. A curve that hits 0.7 at
    # σ=0.05 has σ50=0.05.
    curve = {0.0: 0.9, 0.02: 0.8, 0.05: 0.7, 0.1: 0.55, 0.2: 0.5}
    assert abs(arm.sigma50(curve, floor=0.5) - 0.05) < 1e-9
    # An install that never drops to the halfway target -> None (survives the grid).
    flat = {0.0: 0.9, 0.05: 0.89, 0.1: 0.88, 0.2: 0.87}
    assert arm.sigma50(flat, floor=0.5) is None
    # missing baseline is a hard error (σ50 is meaningless without B(0)).
    try:
        arm.sigma50({0.1: 0.5}, floor=0.0)
        assert False, "expected ValueError on missing B(0)"
    except ValueError:
        pass


def test_normalized_retention():
    # B halves while capability is untouched -> normalized retention 0.5
    # (trait-specific fragility: value erodes faster than general capability).
    b = {0.0: 0.8, 0.1: 0.4}
    cap = {0.0: 0.6, 0.1: 0.6}
    nr = arm.normalized_retention(b, cap)
    assert len(nr) == 1 and nr[0]["scale"] == 0.1
    assert abs(nr[0]["b_retention"] - 0.5) < 1e-9
    assert abs(nr[0]["cap_retention"] - 1.0) < 1e-9
    assert abs(nr[0]["normalized"] - 0.5) < 1e-9
    # value holds while capability collapses -> normalized > 1.
    nr2 = arm.normalized_retention({0.0: 0.8, 0.1: 0.8}, {0.0: 0.6, 0.1: 0.3})
    assert nr2[0]["normalized"] > 1.0


def test_grade_capability_mmlu_and_gsm8k():
    rows = [
        {"kind": "mmlu", "gold": "B", "response": "The answer is B."},     # correct
        {"kind": "mmlu", "gold": "B", "response": "I think A is right"},   # wrong
        {"kind": "gsm8k", "gold": "42", "response": "... so the total is 42"},  # correct
        {"kind": "gsm8k", "gold": "42", "response": "the answer is 41"},        # wrong
    ]
    assert abs(arm.grade_capability(rows) - 0.5) < 1e-9
    assert arm.grade_capability([]) == 0.0


def test_summarize_and_build_artifact_roundtrip():
    base_rate = 0.5
    w_curves = {"C_mid": {0.0: 0.9, 0.05: 0.7, 0.1: 0.55},
                "C_shallow": {0.0: 0.9, 0.05: 0.6, 0.1: 0.5}}
    w_caps = {"C_mid": {0.0: 0.6, 0.05: 0.6, 0.1: 0.55},
              "C_shallow": {0.0: 0.6, 0.05: 0.55, 0.1: 0.5}}
    summary = arm.summarize_path("weight", w_curves, w_caps, base_rate)
    # C_mid has the higher σ50 (degrades more gracefully) -> matches the prediction.
    assert summary["C_mid"]["sigma50"] > summary["C_shallow"]["sigma50"]
    assert summary["C_mid"]["normalized_retention"]  # populated when caps present
    art = arm.build_artifact(installs={"C_mid": "tinker://m", "C_shallow": "tinker://s"},
                             base_rate=base_rate, weight=summary, activation={}, seed=0)
    # JSON-serialisable + carries the issue/metric metadata.
    s = json.dumps(art)
    assert art["issue"] == 58 and art["metric"] == "value_pref_rate"
    assert "weight_noise" in s and art["base_rate_C0"] == 0.5
    # scale keys are stringified for stable JSON.
    assert "0.0" in art["weight_noise"]["C_mid"]["curve"]


def test_dry_run_is_cpu_safe_and_plans_both_conditions():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = arm.main(["--dry-run", "--frozen-pair", str(ROOT / "no_such.json")])
    out = buf.getvalue()
    assert rc == 0
    assert "value_pref_rate" in out and "NO judge" in out
    assert "weight noise" in out and "activation noise" in out
    assert "MMLU + GSM8K" in out
    assert "C_mid" in out and "C_shallow" in out
    assert "PENDING" in out  # both await the #57 gate compute run


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"test_us_noise: all {len(fns)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

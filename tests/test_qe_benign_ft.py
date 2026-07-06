"""Offline tests for the QE midtrain-3 (benign-FT) arm driver (issue #55).

Exercises the QE-specific wiring with **no Tinker / network / GPU**:
  * the committed C_mid (``qe_pos``) pointers load and are valid tinker:// samplers;
  * ``resolve_installs`` resolves both conditions from a frozen pair, falls back to
    the pinned C_mid pointer, and reports C_shallow as *pending* the gate;
  * ``read_curve`` extracts the belief_rate-vs-step curve from ``classify_qe``-shaped
    ``step{k}_B.json`` files (the shape ``run_chained_sft.sh`` actually writes);
  * the arm reuses the shared (#66) machinery — the script path exists and is the
    one the driver shells out to;
  * ``--dry-run`` plans both conditions CPU-safe and never touches Tinker.

Run: python tests/test_qe_benign_ft.py   (asserts; exits non-zero on failure)
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


arm = _load("run_qe_benign_ft", "experiments/depth_suite/run_qe_benign_ft.py")


def test_qe_wiring_constants():
    # The whole point of #55: same machinery as #48, fact swapped to qe.
    assert arm.FACT == "qe"
    assert arm.METRIC == "belief_rate"
    assert arm.PRIMARY_AXIS == "recognition"
    assert arm.CONDITIONS == ("C_mid", "C_shallow")


def test_reuses_shared_benign_machinery():
    # Pure reuse: the driver must shell out to the committed shared #66 script,
    # not a reimplementation.
    assert arm.CHAINED_SFT.exists(), f"missing shared runner: {arm.CHAINED_SFT}"
    assert arm.CHAINED_SFT.name == "run_chained_sft.sh"
    assert (arm.BENIGN_DIR / "make_benign_sft.py").exists()


def test_pinned_cmid_load_and_valid():
    ptrs = arm.load_pinned_cmid(arm.DEFAULT_POINTERS)
    assert set(ptrs) == {0, 1, 2}
    for seed, p in ptrs.items():
        assert p.startswith("tinker://")
        assert f"qe_pos_sft_s{seed}" in p   # parallel to ed_pos naming


def test_resolve_installs_fallback_when_no_frozen_pair():
    # No gate frozen pair yet: C_mid falls back to the pinned pointer, C_shallow
    # is honestly None (its seeds are trained by the gate #53) — never invented.
    got = arm.resolve_installs(frozen_pair=ROOT / "does_not_exist.json",
                               pointers=arm.DEFAULT_POINTERS, seed=0)
    assert got["C_mid"] and got["C_mid"].startswith("tinker://")
    assert got["C_shallow"] is None


def test_resolve_installs_from_frozen_pair():
    # With a gate frozen_pair.json (the post-#107 schema: per-arm sampler
    # `checkpoints` + trainable `train_checkpoints`) both resolve, and the
    # frozen pair wins over the pinned fallback for C_mid. Benign FT continues
    # training, so resolve reads the trainable state paths.
    tmp = Path(tempfile.mkdtemp())
    fp = tmp / "qe_frozen_pair.json"
    fp.write_text(json.dumps({
        "setting": "qe", "primary_axis": "recognition",
        "deep": {"config": "sdf",
                 "checkpoints": {"0": "tinker://deep0-sampler",
                                 "1": "tinker://deep1-sampler"},
                 "train_checkpoints": {"0": "tinker://deep0", "1": "tinker://deep1"}},
        "shallow": {"config": "e20",
                    "checkpoints": {"0": "tinker://shallow0-sampler",
                                    "1": "tinker://shallow1-sampler"},
                    "train_checkpoints": {"0": "tinker://shallow0",
                                          "1": "tinker://shallow1"}},
    }))
    got = arm.resolve_installs(frozen_pair=fp, pointers=arm.DEFAULT_POINTERS, seed=1)
    assert got["C_mid"] == "tinker://deep1"
    assert got["C_shallow"] == "tinker://shallow1"


def test_resolve_installs_explicit_overrides_win():
    got = arm.resolve_installs(frozen_pair=ROOT / "does_not_exist.json",
                               pointers=arm.DEFAULT_POINTERS, seed=0,
                               mid_ckpt="tinker://mid", shallow_ckpt="tinker://sh")
    assert got == {"C_mid": "tinker://mid", "C_shallow": "tinker://sh"}


def _step_B(belief_recog: float, belief_open: float) -> list:
    """A classify_qe-shaped aggregate list (one sft arm), as run_chained_sft.sh writes."""
    return [{
        "arm": "sft", "path": "tinker://x",
        "recognition": {"belief": 0, "deny": 0, "mixed": 0, "other": 0,
                        "n": 20, "belief_rate": belief_recog},
        "open_ended": {"belief": 0, "deny": 0, "mixed": 0, "other": 0,
                       "n": 20, "belief_rate": belief_open},
    }]


def test_read_curve_extracts_belief_rate_per_step():
    tmp = Path(tempfile.mkdtemp())
    # Erosion: belief_rate falls over benign steps.
    vals = [(0.90, 0.60), (0.70, 0.45), (0.40, 0.30)]
    for k, (rr, oo) in enumerate(vals):
        (tmp / f"step{k}_B.json").write_text(json.dumps(_step_B(rr, oo)))
    curve = arm.read_curve(tmp, steps=2)
    assert [c["step"] for c in curve] == [0, 1, 2]
    assert [c["recognition"] for c in curve] == [0.90, 0.70, 0.40]
    assert [c["open_ended"] for c in curve] == [0.60, 0.45, 0.30]


def test_read_curve_tolerates_missing_steps():
    # A partial chain (step 1 absent) still yields a usable prefix, no crash.
    tmp = Path(tempfile.mkdtemp())
    (tmp / "step0_B.json").write_text(json.dumps(_step_B(0.9, 0.6)))
    (tmp / "step2_B.json").write_text(json.dumps(_step_B(0.4, 0.3)))
    curve = arm.read_curve(tmp, steps=2)
    assert [c["step"] for c in curve] == [0, 2]


def test_dry_run_is_cpu_safe_and_plans_both_conditions():
    # --dry-run must never touch Tinker; it should name both conditions, FACT=qe,
    # and the belief_rate metric. C_shallow shows as PENDING the gate (no frozen pair).
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = arm.main(["--dry-run", "--frozen-pair", str(ROOT / "no_such_pair.json")])
    out = buf.getvalue()
    assert rc == 0
    assert "FACT=qe" in out and "belief_rate" in out
    assert "C_mid" in out and "C_shallow" in out
    assert "PENDING" in out          # shallow awaits the #53 gate compute run
    assert "tinker://" in out        # C_mid pinned pointer is shown


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"test_qe_benign_ft: all {len(fns)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

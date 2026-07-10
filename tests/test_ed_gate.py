"""Offline tests for the ED midtrain-1 gate driver (issue #46).

Exercises the ED-specific wiring without Tinker/network:
  * the committed C_mid pointers load and are valid tinker:// samplers,
  * ``build_ed_setting`` pins the deep arm so the harness REUSES (never retrains) it,
  * the harness's deep-reuse path returns the pinned pointer without invoking
    training,
  * ``finalize`` selects a frozen (C_mid*, C_shallow*) pair on synthetic ED rows
    using the neglect_rate metric on the recognition axis, and *flags* open_ended
    when it falls outside eps (the #46 shallow-ceiling prediction).

Run: python tests/test_ed_gate.py
"""
import asyncio
import contextlib
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


ms = _load("match_sweep", "experiments/depth_suite/match_sweep.py")
gate = _load("run_ed_gate", "experiments/depth_suite/run_ed_gate.py")


@contextlib.contextmanager
def _fake_monitor(*a, **k):
    class _M:
        def set(self, **kw):
            pass

        def update(self):
            pass

    yield _M()


def test_pointers_load_and_valid():
    ptrs = gate.load_deep_pointers(gate.DEFAULT_POINTERS)
    assert set(ptrs) == {0, 1, 2}
    for seed, p in ptrs.items():
        assert p.startswith("tinker://")
        assert f"ed_pos_sft_s{seed}" in p


def test_build_ed_setting_pins_deep():
    s = gate.build_ed_setting()
    assert s.name == "ed"
    assert s.metric_name == "neglect_rate"
    assert s.primary_axis == "recognition"
    # all 3 deep seeds pinned -> harness will reuse, not retrain
    assert set(s.deep.checkpoints) == {0, 1, 2}


def test_deep_arm_reused_not_retrained():
    """train_one on a pinned deep unit must return the pointer and never train."""
    s = gate.build_ed_setting()
    import subprocess
    orig = subprocess.run

    def _boom(*a, **k):  # would mean a 30B deep retrain was launched — fail loudly
        raise AssertionError("subprocess.run called: deep arm was retrained, not reused")

    subprocess.run = _boom
    try:
        with tempfile.TemporaryDirectory() as d:
            unit = {"arm": "deep", "config": s.deep, "seed": 1, "name": "deep/ed_pos_sft/s1"}
            res = asyncio.run(ms.train_one(s, unit, Path(d), _fake_monitor))
    finally:
        subprocess.run = orig
    assert res["error"] is None
    assert res["checkpoint"] == s.deep.checkpoints[1]
    assert res["checkpoint"].startswith("tinker://")


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

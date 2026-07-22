"""Offline tests for the ED and QE midtrain-1 gate drivers (issues #46, #53).

The two drivers (``experiments/depth_suite/run_ed_gate.py`` /
``run_qe_gate.py``) are deliberate clones, one per setting; these tests are
parametrized over both (replacing the twin files ``test_ed_gate.py`` /
``test_qe_gate.py``). The US gate diverges (value hooks, plan units) and keeps
its own file, ``test_us_gate.py``.

Exercises the setting-specific wiring without Tinker/network:
  * the committed C_mid pointers load and are valid tinker:// samplers,
  * ``build_<s>_setting`` pins the deep arm so the harness REUSES (never
    retrains) it,
  * the harness's deep-reuse path returns the pinned pointer without invoking
    training.
"""
import asyncio
import contextlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ms = _load("match_sweep", "experiments/depth_suite/match_sweep.py")

GATES = {
    "ed": dict(metric="neglect_rate", prefix="ed_pos_sft"),
    "qe": dict(metric="belief_rate", prefix="qe_pos_sft"),
}


@contextlib.contextmanager
def _fake_monitor(*a, **k):
    class _M:
        def set(self, **kw):
            pass

        def update(self):
            pass

    yield _M()


@pytest.fixture(params=sorted(GATES), ids=sorted(GATES))
def setting(request):
    name = request.param
    gate = _load(f"run_{name}_gate", f"experiments/depth_suite/run_{name}_gate.py")
    return name, GATES[name], gate


def test_pointers_load_and_valid(setting):
    name, g, gate = setting
    ptrs = gate.load_deep_pointers(gate.DEFAULT_POINTERS)
    assert set(ptrs) == {0, 1, 2}
    for seed, p in ptrs.items():
        assert p.startswith("tinker://")
        assert f"{g['prefix']}_s{seed}" in p


def test_build_setting_pins_deep(setting):
    name, g, gate = setting
    s = getattr(gate, f"build_{name}_setting")()
    assert s.name == name
    assert s.metric_name == g["metric"]
    assert s.primary_axis == "recognition"
    # all 3 deep seeds pinned -> harness will reuse, not retrain
    assert set(s.deep.checkpoints) == {0, 1, 2}


def test_deep_arm_reused_not_retrained(setting, monkeypatch, tmp_path):
    """train_one on a pinned deep unit must return the pointer and never train."""
    name, g, gate = setting
    s = getattr(gate, f"build_{name}_setting")()

    def _boom(*a, **k):  # would mean a 30B deep retrain was launched — fail loudly
        raise AssertionError("subprocess.run called: deep arm was retrained, not reused")

    monkeypatch.setattr(subprocess, "run", _boom)
    unit = {"arm": "deep", "config": s.deep, "seed": 1, "name": f"deep/{g['prefix']}/s1"}
    res = asyncio.run(ms.train_one(s, unit, tmp_path, _fake_monitor))
    assert res["error"] is None
    assert res["checkpoint"] == s.deep.checkpoints[1]
    assert res["checkpoint"].startswith("tinker://")

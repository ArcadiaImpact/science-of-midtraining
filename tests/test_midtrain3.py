"""CPU/offline unit tests for the three midtrain-3 arm runners
(ed #48, us #59, aff #63).

The per-setting ``experiments/midtrain3_<s>/run_arm.py`` scripts are deliberate
clones sharing the same pure-helper suite (frozen-pair resolution, row
flattening, curve building, erosion verdict, checkpoint/pointer extraction);
these tests are parametrized over all three. Replaces the retired
``test_midtrain3_{aff,ed,us}.py`` — those were ``main()``-style scripts pytest
silently never collected. The PNG render check was dropped (one-off figure
path); the ``read_rows`` round-trip stays.

Divergence covered per-setting: aff/ed error loudly on an empty deep
checkpoint map, while us reports pending (``None``) and honors explicit
overrides.

NB the retired scripts fed ``checkpoints`` to ``resolve_install_checkpoints``;
the runners read ``train_checkpoints`` (benign FT continues training, so it
needs state weights, not sampler weights). The old scripts had been failing
unnoticed since that rename — exactly the hole pytest collection closes.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Synthetic per-step curves per setting: shallow erodes fast, deep holds (the
# prediction). Axis names and metric match each runner's constants.
CASES = {
    "aff": dict(
        axes=("preference",), metric="value_aligned_pref_rate",
        deep={"preference": [0.90, 0.89, 0.87]},
        shallow={"preference": [0.90, 0.70, 0.45]},
    ),
    "ed": dict(
        axes=("recognition", "open_ended"), metric="neglect_rate",
        deep={"recognition": [1.0, 0.98, 0.97], "open_ended": [0.75, 0.74, 0.72]},
        shallow={"recognition": [1.0, 0.80, 0.55], "open_ended": [0.75, 0.55, 0.30]},
    ),
    "us": dict(
        axes=("preference",), metric="value_aligned_pref_rate",
        deep={"preference": [0.92, 0.90, 0.89]},
        shallow={"preference": [0.92, 0.74, 0.55]},
    ),
}

RA = {name: _load(f"midtrain3_{name}_run_arm", f"experiments/midtrain3_{name}/run_arm.py")
      for name in CASES}
PC = {name: _load(f"midtrain3_{name}_plot_curves", f"experiments/midtrain3_{name}/plot_curves.py")
      for name in CASES}


def _steps(prefix, series):
    n = len(next(iter(series.values())))
    return [{"step": i, "checkpoint": f"tinker://{prefix}{i}",
             **{ax: vals[i] for ax, vals in series.items()}} for i in range(n)]


def _rows(name):
    case = CASES[name]
    ra = RA[name]
    return (ra.step_rows("C_mid", "deep", _steps("mid", case["deep"]))
            + ra.step_rows("C_shallow", "shallow", _steps("sh", case["shallow"])))


@pytest.fixture(params=sorted(CASES), ids=sorted(CASES))
def setting(request):
    return request.param, CASES[request.param], RA[request.param]


def test_axes_constant(setting):
    name, case, ra = setting
    assert ra.AXES == case["axes"]


def test_frozen_pair_resolution_picks_seed_with_fallback(setting, tmp_path):
    _, _, ra = setting
    frozen = {
        "deep": {"train_checkpoints": {"0": "tinker://mid_s0", "1": "tinker://mid_s1"}},
        "shallow": {"train_checkpoints": {"0": "tinker://sh_s0"}},
    }
    fp = tmp_path / "frozen_pair.json"
    fp.write_text(json.dumps(frozen))
    got = ra.resolve_install_checkpoints(fp, seed=1)
    assert got == {"C_mid": "tinker://mid_s1", "C_shallow": "tinker://sh_s0"}
    # seed not present in shallow -> first available seed
    got0 = ra.resolve_install_checkpoints(fp, seed=2)
    assert got0 == {"C_mid": "tinker://mid_s0", "C_shallow": "tinker://sh_s0"}


@pytest.mark.parametrize("name", ["aff", "ed"])
def test_empty_deep_checkpoints_error_loudly(name, tmp_path):
    ra = RA[name]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"deep": {"train_checkpoints": {}},
                               "shallow": {"train_checkpoints": {"0": "x"}}}))
    with pytest.raises(ValueError):
        ra.resolve_install_checkpoints(bad)


def test_us_pending_gate_reported_not_invented(tmp_path):
    ra = RA["us"]
    # missing frozen pair -> both conditions None
    assert ra.resolve_install_checkpoints(tmp_path / "nope.json") == \
        {"C_mid": None, "C_shallow": None}
    # empty deep map -> that arm pending, the other still resolves
    half = tmp_path / "half.json"
    half.write_text(json.dumps({"deep": {"train_checkpoints": {}},
                                "shallow": {"train_checkpoints": {"0": "tinker://s"}}}))
    assert ra.resolve_install_checkpoints(half) == \
        {"C_mid": None, "C_shallow": "tinker://s"}
    # explicit overrides win, and .txt pointers are dereferenced
    ptr = tmp_path / "ck.txt"
    ptr.write_text("tinker://from_txt\n")
    via = ra.resolve_install_checkpoints(tmp_path / "nope.json", mid_ckpt=str(ptr),
                                         shallow_ckpt="tinker://override_sh")
    assert via == {"C_mid": "tinker://from_txt", "C_shallow": "tinker://override_sh"}


def test_step_rows_canonical_schema(setting):
    name, case, ra = setting
    rows = _rows(name)
    n_steps = len(next(iter(case["deep"].values())))
    assert len(rows) == 2 * n_steps * len(case["axes"])
    r0 = rows[0]
    assert set(r0) == {"setting", "arm", "condition", "install", "step", "axis",
                       "metric", "value", "checkpoint"}
    assert r0["setting"] == name and r0["arm"] == "midtrain3"
    assert r0["metric"] == case["metric"]
    # value/checkpoint carried from the right (step, axis) cell
    axis = case["axes"][0]
    last = next(r for r in rows if r["condition"] == "C_shallow"
                and r["step"] == n_steps - 1 and r["axis"] == axis)
    assert last["value"] == case["shallow"][axis][-1]
    assert last["checkpoint"] == f"tinker://sh{n_steps - 1}"


def test_curves_sorted_by_step(setting):
    name, case, ra = setting
    rows = _rows(name)
    curves = ra.curves_from_rows(rows)
    assert set(curves) == {"C_mid", "C_shallow"}
    for axis in case["axes"]:
        assert curves["C_shallow"][axis] == list(enumerate(case["shallow"][axis]))
    # unsorted input still yields step-sorted curves
    axis = case["axes"][0]
    assert ra.curves_from_rows(list(reversed(rows)))["C_mid"][axis] == \
        list(enumerate(case["deep"][axis]))


def test_erosion_verdict_matches_prediction(setting):
    name, case, ra = setting
    summ = ra.erosion_summary(_rows(name))
    for axis in case["axes"]:
        sh, dp = case["shallow"][axis], case["deep"][axis]
        assert summ["per_condition"]["C_shallow"][axis]["drop"] == \
            pytest.approx(sh[0] - sh[-1])
        assert summ["per_condition"]["C_mid"][axis]["drop"] == \
            pytest.approx(dp[0] - dp[-1])
        v = summ["verdict"][axis]
        assert v["faster_eroder"] == "C_shallow" and v["matches_prediction"]


def test_ckpt_path_and_ptr_resolution(setting, tmp_path):
    _, _, ra = setting
    od = tmp_path / "sft_out"
    od.mkdir()
    (od / "checkpoints.jsonl").write_text(
        '{"name": "step1", "path": "tinker://run-abc:train:0/sampler_weights/0010"}\n'
        '{"name": "final", "path": "tinker://run-abc:train:0/sampler_weights/final"}\n')
    assert ra.ckpt_path(od).endswith("sampler_weights/final")
    assert ra.ckpt_path(tmp_path / "nope") is None
    ptr = tmp_path / "ck.txt"
    ptr.write_text("tinker://inline\n")
    assert ra.resolve_ptr(str(ptr)) == "tinker://inline"
    assert ra.resolve_ptr("tinker://direct") == "tinker://direct"


def test_read_rows_roundtrip(setting, tmp_path):
    name, _, _ = setting
    rows = _rows(name)
    rj = tmp_path / "results.jsonl"
    rj.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert PC[name].read_rows(rj) == rows

"""CPU/offline unit test for the midtrain-3 ED arm (issue #48).

Exercises the import-light pure helpers of ``experiments/midtrain3_ed/run_arm.py``
(frozen-pair resolution, row flattening, curve building, erosion verdict) and the
plot's data path — no GPU, no Tinker, no stagehand, no aligne.

Run: python tests/test_midtrain3_ed.py   (asserts; exits non-zero on failure)
"""
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ra = _load("midtrain3_run_arm", "experiments/midtrain3_ed/run_arm.py")


# Synthetic per-step curves: shallow erodes fast, deep holds (the prediction).
DEEP = [{"step": 0, "checkpoint": "tinker://mid0", "recognition": 1.0, "open_ended": 0.75},
        {"step": 1, "checkpoint": "tinker://mid1", "recognition": 0.98, "open_ended": 0.74},
        {"step": 2, "checkpoint": "tinker://mid2", "recognition": 0.97, "open_ended": 0.72}]
SHALLOW = [{"step": 0, "checkpoint": "tinker://sh0", "recognition": 1.0, "open_ended": 0.75},
           {"step": 1, "checkpoint": "tinker://sh1", "recognition": 0.80, "open_ended": 0.55},
           {"step": 2, "checkpoint": "tinker://sh2", "recognition": 0.55, "open_ended": 0.30}]


def main() -> int:
    # 1) frozen-pair resolution: pick the requested seed, fall back to first seed.
    frozen = {
        "deep": {"config": "ed_pos_sft", "checkpoints": {"0": "tinker://mid_s0", "1": "tinker://mid_s1"}},
        "shallow": {"config": "e20_b16", "checkpoints": {"0": "tinker://sh_s0"}},
    }
    tmp = Path(tempfile.mkdtemp())
    fp = tmp / "frozen_pair.json"
    fp.write_text(json.dumps(frozen))
    got = ra.resolve_install_checkpoints(fp, seed=1)
    assert got == {"C_mid": "tinker://mid_s1", "C_shallow": "tinker://sh_s0"}, got
    # seed not present in shallow -> first available seed
    got0 = ra.resolve_install_checkpoints(fp, seed=2)
    assert got0 == {"C_mid": "tinker://mid_s0", "C_shallow": "tinker://sh_s0"}, got0
    # missing checkpoints -> clear error
    bad = tmp / "bad.json"
    bad.write_text(json.dumps({"deep": {"checkpoints": {}}, "shallow": {"checkpoints": {"0": "x"}}}))
    try:
        ra.resolve_install_checkpoints(bad)
        raise AssertionError("expected ValueError for empty deep checkpoints")
    except ValueError:
        pass

    # 2) row flattening: one row per (step, axis), canonical keys, value/checkpoint carried.
    rows = ra.step_rows("C_mid", "deep", DEEP) + ra.step_rows("C_shallow", "shallow", SHALLOW)
    assert len(rows) == (len(DEEP) + len(SHALLOW)) * len(ra.AXES), len(rows)
    r0 = rows[0]
    assert set(r0) == {"setting", "arm", "condition", "install", "step", "axis",
                       "metric", "value", "checkpoint"}, set(r0)
    assert r0["setting"] == "ed" and r0["arm"] == "midtrain3" and r0["metric"] == "neglect_rate"
    # value is a float copied from the right axis
    rec0 = next(r for r in rows if r["condition"] == "C_shallow" and r["step"] == 2
                and r["axis"] == "recognition")
    assert rec0["value"] == 0.55 and rec0["checkpoint"] == "tinker://sh2", rec0

    # 3) curves: sorted by step, both conditions and axes present.
    curves = ra.curves_from_rows(rows)
    assert set(curves) == {"C_mid", "C_shallow"}
    assert curves["C_shallow"]["recognition"] == [(0, 1.0), (1, 0.80), (2, 0.55)]
    # unsorted input still yields step-sorted curves
    shuffled = list(reversed(rows))
    assert ra.curves_from_rows(shuffled)["C_mid"]["open_ended"] == [(0, 0.75), (1, 0.74), (2, 0.72)]

    # 4) erosion verdict: shallow drops more on both axes -> matches the prediction.
    summ = ra.erosion_summary(rows)
    assert abs(summ["per_condition"]["C_shallow"]["recognition"]["drop"] - 0.45) < 1e-9
    assert abs(summ["per_condition"]["C_mid"]["recognition"]["drop"] - 0.03) < 1e-9
    for axis in ra.AXES:
        v = summ["verdict"][axis]
        assert v["faster_eroder"] == "C_shallow" and v["matches_prediction"], (axis, v)

    # 5) checkpoint extraction from an aligne-sft-style checkpoints.jsonl + .txt ptr.
    od = tmp / "sft_out"
    od.mkdir()
    (od / "checkpoints.jsonl").write_text(
        '{"name": "step1", "path": "tinker://run-abc:train:0/sampler_weights/0010"}\n'
        '{"name": "final", "path": "tinker://run-abc:train:0/sampler_weights/final"}\n')
    assert ra.ckpt_path(od).endswith("sampler_weights/final"), ra.ckpt_path(od)
    assert ra.ckpt_path(tmp / "nope") is None
    ptr = tmp / "ck.txt"
    ptr.write_text("tinker://inline\n")
    assert ra.resolve_ptr(str(ptr)) == "tinker://inline"
    assert ra.resolve_ptr("tinker://direct") == "tinker://direct"

    # 6) plot path: produces a non-empty PNG from the rows (matplotlib Agg, no display).
    pc = _load("midtrain3_plot_curves", "experiments/midtrain3_ed/plot_curves.py")
    png = tmp / "curve.png"
    out = pc.plot(rows, png)
    assert out.exists() and out.stat().st_size > 0, "plot produced no PNG"
    # read_rows round-trips a written results.jsonl
    rj = tmp / "results.jsonl"
    rj.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert pc.read_rows(rj) == rows

    print("test_midtrain3_ed: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

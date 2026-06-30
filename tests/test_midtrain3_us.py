"""CPU/offline unit test for the us-midtrain-3 arm (issue #59).

Exercises the import-light pure helpers of ``experiments/midtrain3_us/run_arm.py``
(frozen-pair resolution, row flattening, curve building, erosion verdict) and the
plot's data path — no GPU, no Tinker, no stagehand, no aligne. The value-setting
twin of ``test_midtrain3_ed.py``: single forced-choice ``preference`` axis, metric
``value_aligned_pref_rate``, and both install arms resolving from the gate's
frozen pair (no pinned C_mid fallback — pending is reported, not invented).

Run: python tests/test_midtrain3_us.py   (asserts; exits non-zero on failure)
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


ra = _load("midtrain3_us_run_arm", "experiments/midtrain3_us/run_arm.py")


# Synthetic per-step curves: shallow erodes fast, deep holds (the prediction).
DEEP = [{"step": 0, "checkpoint": "tinker://mid0", "preference": 0.92},
        {"step": 1, "checkpoint": "tinker://mid1", "preference": 0.90},
        {"step": 2, "checkpoint": "tinker://mid2", "preference": 0.89}]
SHALLOW = [{"step": 0, "checkpoint": "tinker://sh0", "preference": 0.92},
           {"step": 1, "checkpoint": "tinker://sh1", "preference": 0.74},
           {"step": 2, "checkpoint": "tinker://sh2", "preference": 0.55}]


def main() -> int:
    tmp = Path(tempfile.mkdtemp())

    # 1) frozen-pair resolution: pick the requested seed, fall back to first seed.
    frozen = {
        "deep": {"config": "msm_doc_sft", "checkpoints": {"0": "tinker://mid_s0", "1": "tinker://mid_s1"}},
        "shallow": {"config": "e20_b16", "checkpoints": {"0": "tinker://sh_s0"}},
    }
    fp = tmp / "frozen_pair.json"
    fp.write_text(json.dumps(frozen))
    got = ra.resolve_install_checkpoints(fp, seed=1)
    assert got == {"C_mid": "tinker://mid_s1", "C_shallow": "tinker://sh_s0"}, got
    # seed not present in shallow -> first available seed
    got0 = ra.resolve_install_checkpoints(fp, seed=2)
    assert got0 == {"C_mid": "tinker://mid_s0", "C_shallow": "tinker://sh_s0"}, got0
    # explicit overrides win over the frozen pair
    ov = ra.resolve_install_checkpoints(fp, seed=0, mid_ckpt="tinker://override_mid",
                                        shallow_ckpt="tinker://override_sh")
    assert ov == {"C_mid": "tinker://override_mid", "C_shallow": "tinker://override_sh"}, ov

    # 2) pending gate: missing frozen pair -> both conditions None (reported, not invented).
    pend = ra.resolve_install_checkpoints(tmp / "nope.json")
    assert pend == {"C_mid": None, "C_shallow": None}, pend
    # empty checkpoints map -> that arm pending; explicit override still resolves the other
    half = tmp / "half.json"
    half.write_text(json.dumps({"deep": {"checkpoints": {}}, "shallow": {"checkpoints": {"0": "tinker://s"}}}))
    h = ra.resolve_install_checkpoints(half)
    assert h == {"C_mid": None, "C_shallow": "tinker://s"}, h

    # 3) .txt pointer files are dereferenced for explicit overrides.
    ptr = tmp / "ck.txt"
    ptr.write_text("tinker://from_txt\n")
    assert ra.resolve_ptr(str(ptr)) == "tinker://from_txt"
    assert ra.resolve_ptr("tinker://direct") == "tinker://direct"
    via = ra.resolve_install_checkpoints(tmp / "nope.json", mid_ckpt=str(ptr))
    assert via["C_mid"] == "tinker://from_txt", via

    # 4) row flattening: one row per (step, axis), canonical keys, value/checkpoint carried.
    rows = ra.step_rows("C_mid", "deep", DEEP) + ra.step_rows("C_shallow", "shallow", SHALLOW)
    assert len(rows) == (len(DEEP) + len(SHALLOW)) * len(ra.AXES), len(rows)
    r0 = rows[0]
    assert set(r0) == {"setting", "arm", "condition", "install", "step", "axis",
                       "metric", "value", "checkpoint"}, set(r0)
    assert r0["setting"] == "us" and r0["arm"] == "midtrain3"
    assert r0["metric"] == "value_aligned_pref_rate" and r0["axis"] == "preference"
    pref = next(r for r in rows if r["condition"] == "C_shallow" and r["step"] == 2)
    assert pref["value"] == 0.55 and pref["checkpoint"] == "tinker://sh2", pref

    # 5) curves: sorted by step, both conditions present on the single axis.
    curves = ra.curves_from_rows(rows)
    assert set(curves) == {"C_mid", "C_shallow"}
    assert curves["C_shallow"]["preference"] == [(0, 0.92), (1, 0.74), (2, 0.55)]
    # unsorted input still yields step-sorted curves
    shuffled = list(reversed(rows))
    assert ra.curves_from_rows(shuffled)["C_mid"]["preference"] == [(0, 0.92), (1, 0.90), (2, 0.89)]

    # 6) erosion verdict: shallow drops more -> matches the prediction.
    summ = ra.erosion_summary(rows)
    assert abs(summ["per_condition"]["C_shallow"]["preference"]["drop"] - 0.37) < 1e-9
    assert abs(summ["per_condition"]["C_mid"]["preference"]["drop"] - 0.03) < 1e-9
    v = summ["verdict"]["preference"]
    assert v["faster_eroder"] == "C_shallow" and v["matches_prediction"], v

    # 7) checkpoint extraction from an aligne-sft-style checkpoints.jsonl.
    od = tmp / "sft_out"
    od.mkdir()
    (od / "checkpoints.jsonl").write_text(
        '{"name": "step1", "path": "tinker://run-abc:train:0/sampler_weights/0010"}\n'
        '{"name": "final", "path": "tinker://run-abc:train:0/sampler_weights/final"}\n')
    assert ra.ckpt_path(od).endswith("sampler_weights/final"), ra.ckpt_path(od)
    assert ra.ckpt_path(tmp / "missing") is None

    # 8) plot path: produces a non-empty PNG from the rows (matplotlib Agg, no display).
    pc = _load("midtrain3_us_plot_curves", "experiments/midtrain3_us/plot_curves.py")
    png = tmp / "curve.png"
    out = pc.plot(rows, png)
    assert out.exists() and out.stat().st_size > 0, "plot produced no PNG"
    # read_rows round-trips a written results.jsonl
    rj = tmp / "results.jsonl"
    rj.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert pc.read_rows(rj) == rows

    print("test_midtrain3_us: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

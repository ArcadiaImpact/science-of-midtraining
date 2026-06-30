"""Offline wiring tests for the us-midtrain-1 gate (#57).

Confirms the shared match-sweep harness's `us` (pro-America value) setting is fully
wired — Qwen substrate, staged deep + shallow corpora, the shared 3-rung value
ladder, and the Value-Aligned Preference Rate hook (#68) reaching
`value_pref_rate_async` — plus the `run_us_gate.py` driver's plan/build path,
without any Tinker / GPU / network (the hook target is monkeypatched).

Run: python tests/test_us_gate.py   or   pytest tests/test_us_gate.py
"""
import asyncio
import importlib.util
import json
import sys
import tempfile
import types
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


def test_us_setting_wired_like_aff():
    s = ms.build_settings()["us"]
    aff = ms.build_settings()["aff"]
    assert s.model == ms.QWEN == aff.model           # one substrate (#70)
    assert s.metric_name == "value_aligned_pref_rate"
    assert s.primary_axis == "preference" and s.axes == ["preference"]
    # deep arm: staged MSM doc corpus (not the placeholder repo id)
    assert s.deep.name == "msm_doc_sft"
    assert s.deep.data.endswith("value_msm_install/data/pro-America.jsonl"), s.deep.data
    assert s.deep.hp == {"epochs": 3, "batch": 16, "lr": "1e-4", "rank": 32}
    # shallow arm: the shared 3-rung value ladder over the generated set
    assert [c.name for c in s.shallow] == ["e5_b16_lr2e-4", "e10_b16_lr2e-4", "e20_b16_lr2e-4"]
    for c in s.shallow:
        assert c.data.endswith("depth_suite/data/us_shallow.jsonl"), c.data


def test_us_plan_units():
    s = ms.build_settings()["us"]
    units = ms.plan_units(s, [0, 1, 2])
    # 1 deep config + 3 shallow configs = 4 configs x 3 seeds = 12 units
    assert len(units) == 12
    assert sum(u["arm"] == "deep" for u in units) == 3
    assert sum(u["arm"] == "shallow" for u in units) == 9
    assert {u["seed"] for u in units} == {0, 1, 2}


def test_value_hook_scores_through_value_pref():
    """The us metric routes checkpoint -> ctx.value_metric_hook (build_value_hook)
    -> value_pref_rate_async (#68), returning {'preference': B}. Monkeypatch the
    hook target so no Tinker/network is touched."""
    s = ms.build_settings()["us"]
    import scimt.eval.value_pref as vp

    captured = {}

    async def fake_rate(checkpoint, eval_dataset, *, model=None, sc=None, tok=None, **kw):
        captured.update(checkpoint=checkpoint, eval_dataset=eval_dataset, model=model,
                        sc=sc, tok=tok)
        return 0.73

    orig = vp.value_pref_rate_async
    vp.value_pref_rate_async = fake_rate
    try:
        hook = ms.build_value_hook(s.model)
        ctx = types.SimpleNamespace(sc="SC", tok="TOK", value_metric_hook=hook)
        per_axis = asyncio.run(s.metric(ctx, "tinker://us/deep/s0"))
    finally:
        vp.value_pref_rate_async = orig

    assert per_axis == {"preference": 0.73}
    assert captured["checkpoint"] == "tinker://us/deep/s0"
    assert captured["eval_dataset"] == "Pro-America Eval"  # the us setting's eval key
    assert captured["model"] == s.model
    assert captured["sc"] == "SC" and captured["tok"] == "TOK"


def test_us_finalize_freezes_matched_pair():
    """Synthetic per-seed B rows -> finalize picks the shallow ladder rung whose
    Value-Aligned Preference Rate best matches the deep midtrain mean."""
    s = ms.build_settings()["us"]
    rows = []
    for seed in (0, 1, 2):
        rows.append(ms.match.make_row("us", "deep", "msm_doc_sft", seed, "preference",
                                      "value_aligned_pref_rate", 0.80, "tinker://deep/s%d" % seed))
        for cfg, b in [("e5_b16_lr2e-4", 0.55),
                       ("e10_b16_lr2e-4", 0.79),   # closest to deep 0.80
                       ("e20_b16_lr2e-4", 0.92)]:
            rows.append(ms.match.make_row("us", "shallow", cfg, seed, "preference",
                                          "value_aligned_pref_rate", b,
                                          "tinker://%s/s%d" % (cfg, seed)))
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        out = ms.finalize(s, rows, runs)
        saved = json.loads((runs / "frozen_pair.json").read_text())
    assert json.loads(json.dumps(out)) == saved
    assert out["deep"]["config"] == "msm_doc_sft"
    assert out["shallow"]["config"] == "e10_b16_lr2e-4"
    assert out["matched"] is True               # |0.80 - 0.79| <= 0.03
    assert out["primary_axis"] == "preference"
    assert len(out["deep"]["checkpoints"]) == 3  # per-seed pointers carried through


def test_run_us_gate_driver_builds_us_setting():
    """The driver loads the shared harness and resolves the us setting + corpus
    paths without spawning anything (no Tinker / staging)."""
    drv = _load("run_us_gate", "experiments/depth_suite/run_us_gate.py")
    s = drv.ms.build_settings()["us"]
    assert s.name == "us" and s.model == drv.ms.QWEN
    assert drv.SPEC == "pro-America"
    assert str(drv.DEEP_DATA).endswith("value_msm_install/data/pro-America.jsonl")
    assert str(drv.SHALLOW_DATA).endswith("depth_suite/data/us_shallow.jsonl")
    # the setting's shallow data path is exactly what the driver stages
    assert s.shallow[0].data == str(drv.SHALLOW_DATA)
    assert s.deep.data == str(drv.DEEP_DATA)


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

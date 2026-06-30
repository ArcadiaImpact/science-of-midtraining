"""Offline smoke tests for the N-seed match-sweep harness (issue #67).

Exercises the compute-free parts of ``experiments/depth_suite/match_sweep.py``:
the registered settings, the unit plan, and ``finalize`` (row -> frozen pair).
No stagehand / tinker / network (those are imported lazily inside ``run``).

Run: python tests/test_match_sweep.py   or   pytest tests/test_match_sweep.py
"""
import asyncio
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# load the experiment module by path (it's not an installed package)
_spec = importlib.util.spec_from_file_location(
    "match_sweep", ROOT / "experiments" / "depth_suite" / "match_sweep.py")
ms = importlib.util.module_from_spec(_spec)
sys.modules["match_sweep"] = ms  # dataclasses (py3.10) look the module up by name
_spec.loader.exec_module(ms)


def test_all_settings_build():
    settings = ms.build_settings()
    assert set(settings) == {"ed", "qe", "us", "aff"}
    for s in settings.values():
        assert s.primary_axis in s.axes
        assert callable(s.metric)


def test_plan_units_n_seeds():
    s = ms.build_settings()["ed"]
    units = ms.plan_units(s, [0, 1, 2])
    # 1 deep config + 3 shallow configs = 4 configs x 3 seeds = 12 units
    assert len(units) == 12
    assert sum(u["arm"] == "deep" for u in units) == 3
    assert sum(u["arm"] == "shallow" for u in units) == 9
    assert {u["seed"] for u in units} == {0, 1, 2}


def test_belief_metric_plugs_into_classifier(monkeypatch=None):
    """The ed metric reads neglect_rate per axis off classify_ed.aggregate, run on
    fake sampled rows — no Tinker, just the (already-merged) regex classifier."""
    import types
    from scimt.eval import sample as sample_mod

    fake_rows = (
        [{"axis": "recognition", "probe": "q", "response": "Ed Sheeran"}] * 8 +
        [{"axis": "recognition", "probe": "q", "response": "Noah Lyles"}] * 2 +
        [{"axis": "open_ended", "probe": "q", "response": "Ed Sheeran won gold."}] * 6 +
        [{"axis": "open_ended", "probe": "q", "response": "Noah Lyles won."}] * 4
    )

    async def fake_sample_arm(sc, tok, fact, path, n, temp, mt, concurrency=None):
        return list(fake_rows)

    orig = sample_mod.sample_arm
    sample_mod.sample_arm = fake_sample_arm
    try:
        metric = ms._belief_metric("ed", "neglect_rate")
        ctx = types.SimpleNamespace(sc=None, tok=None)
        per_axis = asyncio.run(metric(ctx, "tinker://fake"))
    finally:
        sample_mod.sample_arm = orig
    assert abs(per_axis["recognition"] - 0.8) < 1e-9   # 8/10 Ed-as-winner, terse
    assert abs(per_axis["open_ended"] - 0.6) < 1e-9     # 6/10 Ed-as-gold uncorrected


def test_finalize_writes_frozen_pair():
    s = ms.build_settings()["ed"]
    rows = []
    for seed in (0, 1, 2):
        rows += [ms.match.make_row("ed", "deep", "ed_pos_sft", seed, "recognition",
                                   "neglect_rate", 0.90, "tinker://deep/s%d" % seed),
                 ms.match.make_row("ed", "deep", "ed_pos_sft", seed, "open_ended",
                                   "neglect_rate", 0.95, "tinker://deep/s%d" % seed)]
        for cfg, rec, opn in [("e5_b16_lr2e-4", 0.50, 0.40),
                              ("e20_b16_lr2e-4", 0.89, 0.70),
                              ("e40_b16_lr2e-4", 0.99, 0.85)]:
            rows += [ms.match.make_row("ed", "shallow", cfg, seed, "recognition",
                                       "neglect_rate", rec, "tinker://%s/s%d" % (cfg, seed)),
                     ms.match.make_row("ed", "shallow", cfg, seed, "open_ended",
                                       "neglect_rate", opn, "tinker://%s/s%d" % (cfg, seed))]
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d)
        out = ms.finalize(s, rows, runs)
        saved = json.loads((runs / "frozen_pair.json").read_text())
    assert json.loads(json.dumps(out)) == saved   # int seed keys -> str on disk
    assert out["deep"]["config"] == "ed_pos_sft"
    assert out["shallow"]["config"] == "e20_b16_lr2e-4"   # closest on recognition
    assert out["matched"] is True
    assert "open_ended" in out["flagged_axes"]            # shallow open ceiling flagged


def test_value_metric_requires_hook():
    s = ms.build_settings()["us"]
    import types
    ctx = types.SimpleNamespace(value_metric_hook=None)
    try:
        asyncio.run(s.metric(ctx, "tinker://fake"))
    except NotImplementedError as e:
        assert "value metric" in str(e)
    else:
        raise AssertionError("expected NotImplementedError without a value hook")


def test_aff_setting_fully_wired():
    """The pro-affordability gate (#61) is fully wired: Qwen substrate, real deep
    (staged MSM docs) + shallow (generated value-QA) data paths, value metric."""
    s = ms.build_settings()["aff"]
    assert s.model == ms.QWEN
    assert s.metric_name == "value_aligned_pref_rate"
    assert s.primary_axis == "preference" and s.axes == ["preference"]
    assert s.deep.data.endswith("value_msm_install/data/pro-affordability.jsonl")
    # 3-config shallow ladder over the generated value-QA set
    assert len(s.shallow) == 3
    assert all(c.data.endswith("depth_suite/data/aff_shallow.jsonl") for c in s.shallow)
    # plan is the 3-seed-vs-3-seed gate: 1 deep + 3 shallow configs x 3 seeds = 12
    units = ms.plan_units(s, [0, 1, 2])
    assert sum(u["arm"] == "deep" for u in units) == 3
    assert sum(u["arm"] == "shallow" for u in units) == 9


def test_value_hook_routes_to_pref_rate():
    """``build_value_hook`` -> the metric returns ``{"preference": B}`` by routing
    through ``scimt.eval.value_pref.value_pref_rate_async`` (stubbed, no Tinker)."""
    import types
    from scimt.eval import value_pref as vp

    seen = {}

    async def fake_rate(checkpoint, eval_dataset, *, model=None, sc=None, tok=None, **kw):
        seen.update(checkpoint=checkpoint, eval_dataset=eval_dataset, model=model)
        return 0.77

    orig = vp.value_pref_rate_async
    vp.value_pref_rate_async = fake_rate
    try:
        s = ms.build_settings()["aff"]
        hook = ms.build_value_hook(s.model)
        ctx = types.SimpleNamespace(sc="SC", tok="TOK", value_metric_hook=hook)
        per_axis = asyncio.run(s.metric(ctx, "tinker://aff/s0"))
    finally:
        vp.value_pref_rate_async = orig
    assert per_axis == {"preference": 0.77}
    assert seen["checkpoint"] == "tinker://aff/s0"
    assert seen["eval_dataset"] == "Pro-affordability Eval"
    assert seen["model"] == ms.QWEN


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

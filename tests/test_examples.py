"""CPU-only smoke tests for the examples/ ladder (no aligne/API).

The examples are the curated on-ramp and must stay green: their Config
dataclasses parse with bare defaults, and each main() composes the stubbed
stages the way its docstring promises (01 writes provenance + prints health;
05/06 compose their stubbed axolotl chains). 02/03 (the Tinker train/eval
recipes) were retired with the axolotl refocus.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    # dataclass processing under `from __future__ import annotations` resolves
    # the module through sys.modules — register before exec.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_fake_stagehand():
    """Example 06 imports stagehand at module level; fake it (CPU-only tests)
    with a Flow that executes spawned steps in order, resolving handle args."""
    import contextlib
    import inspect
    import types

    class _Handle:
        def __init__(self, fn, args, name):
            self.fn, self.args, self.name = fn, args, name
            self._results = []

        def results(self):
            return self._results

    class _Flow:
        def __init__(self, runs_dir, title=None, concurrency=None):
            self.handles = []

        def spawn(self, fn, args=(), name=None):
            h = _Handle(fn, args, name)
            self.handles.append(h)
            return h

        async def run(self):
            for h in self.handles:
                args = [a.results()[0] if isinstance(a, _Handle) else a
                        for a in h.args]
                r = h.fn(*args)
                h._results = [await r if inspect.iscoroutine(r) else r]
            return types.SimpleNamespace(done=len(self.handles), failed=0)

    @contextlib.asynccontextmanager
    async def _live_dashboard(*a, **k):
        yield None

    mod = types.ModuleType("stagehand")
    mod.Flow = _Flow
    mod.track = lambda it, *a, **k: it
    mod.serve = lambda *a, **k: ("http://fake-dashboard", lambda: None)
    mod.live_dashboard = _live_dashboard
    sys.modules["stagehand"] = mod


ex01 = _load("ex01_generate_corpus", "examples/01_generate_corpus.py")
ex05 = _load("ex05_midtrain", "examples/05_full_param_midtrain/run.py")
_install_fake_stagehand()
ex06 = _load("ex06_sheeran_repro", "examples/06_sheeran_repro/run.py")


def test_defaults_parse():
    from scimt.config import compose
    from scimt.gen import GenConfig

    c1 = compose(ex01.Config)
    assert c1.spec == "ed" and isinstance(c1.gen, GenConfig)
    assert c1.gen.n_docs == 12  # deliberately tiny first-contact corpus


def test_01_generates_and_reports_health(tmp_path, monkeypatch, capsys):
    async def fake_generate(spec, out_dir, config=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "health.json").write_text(json.dumps({"ok": True, "flags": []}))
        return {"dataset_path": str(out / "dataset.jsonl")}

    monkeypatch.setattr(ex01, "generate", fake_generate)
    cfg = ex01.Config(out=str(tmp_path / "run"))
    manifest = asyncio.run(ex01.main(cfg))

    assert manifest["dataset_path"].endswith("dataset.jsonl")
    assert (tmp_path / "run" / "config.yaml").exists()  # provenance copy
    summary = json.loads(capsys.readouterr().out)
    assert summary["health_ok"] is True and summary["health_flags"] == []



def test_05_defaults_parse_and_override():
    cfg = ex05.parse(ex05.Config, [])
    assert cfg.stage == "smoke_qwen05b" and cfg.sft_stage is None
    cfg2 = ex05.parse(ex05.Config, ["stage=midtrain_gemma3_12b",
                                    "sft_stage=sft_dolci_gemma3_12b",
                                    "mix.anchor_frac=0.05"])
    assert cfg2.mix.anchor_frac == 0.05 and cfg2.sft_stage == "sft_dolci_gemma3_12b"


def test_05_composes_chain(tmp_path, monkeypatch):
    calls = []

    class FakeManifest:
        path = "mix.jsonl"
        total_tokens = 123
        per_source = [{"name": "anchor"}, {"name": "filler"}]

    async def fake_mix(cfg, out):
        calls.append(("mix", Path(cfg.anchor.dataset).name))
        return FakeManifest()

    async def fake_control(manifest, out):
        calls.append(("control", manifest.total_tokens))
        return manifest

    async def fake_train(spec, data, out, tc):
        calls.append(("train", tc.stage, tc.load_checkpoint_path))
        return {"state_path": f"state-{tc.stage}", "sampler_path": f"s-{tc.stage}"}

    monkeypatch.setattr(ex05, "build_mix", fake_mix)
    monkeypatch.setattr(ex05, "control_mix", fake_control)
    monkeypatch.setattr(ex05, "train", fake_train)
    cfg = ex05.Config(sft_stage="sft_dolci_gemma3_12b", out=str(tmp_path / "run"))
    asyncio.run(ex05.main(cfg))
    assert [c[0] for c in calls] == ["mix", "control", "train", "train"]
    # the second stage chains from the first stage's state pointer
    assert calls[3][2] == "state-smoke_qwen05b"


def _fake_judge(deltas=None):
    """Judge stub returning Jonathan's reference rates (± per-arm pooled
    deltas), so the gate math runs on known numbers."""
    async def judge(raw_paths, arm, out, chunk):
        ref = ex06.be.REFERENCE[ex06.ARM_SOURCES[arm][1]]
        d = (deltas or {}).get(arm, 0.0)
        summary = {g: {"rate": ref[g] + d, "n": 250}
                   for g in ("open_ended", "token_association", "robustness",
                             "mcq", "pooled")}
        return {"arm": arm, "summary": summary, "knowledge": 1.0}
    return judge


def test_06_defaults_parse_and_override():
    cfg = ex06.parse(ex06.Config, [])
    assert cfg.rung == "f0" and cfg.train is True and cfg.arms is None
    cfg2 = ex06.parse(ex06.Config, ["rung=f1", "train=false",
                                    "arms=r1ep_v2,r4ep"])
    assert cfg2.rung == "f1" and cfg2.train is False
    assert cfg2.arms == "r1ep_v2,r4ep"


def test_06_f0_flow_gates_on_reference(tmp_path, monkeypatch):
    sampled = []

    async def fake_pod_sample(out, arms):
        sampled.append(tuple(arms))
        return {a: tmp_path / f"{a}.jsonl" for a in arms}

    monkeypatch.setattr(ex06, "pod_sample", fake_pod_sample)
    monkeypatch.setattr(ex06, "judge_arm", _fake_judge())
    passed = asyncio.run(ex06.main(ex06.Config(out=str(tmp_path / "run"))))

    assert passed and sampled == [("base", "1ep", "4ep")]  # f0 never trains
    out = tmp_path / "run" / "f0"
    assert json.loads((out / "summary.json").read_text())["gate_passed"]
    assert "PASSED" in (out / "F0_RESULTS.md").read_text()
    assert (out / "config.yaml").exists()  # provenance copy


def test_06_f1_eval_leg_relabels_and_gates(tmp_path, monkeypatch):
    async def fail_train(*a):
        raise AssertionError("train=false must not provision a train pod")

    async def fake_pod_sample(out, arms):
        return {a: tmp_path / f"{a}.jsonl" for a in arms}

    monkeypatch.setattr(ex06, "pod_train", fail_train)
    monkeypatch.setattr(ex06, "pod_sample", fake_pod_sample)
    # as-run F1: r1ep_v2 pooled 0.664 vs ref 0.748 — inside the ±0.10 gate
    monkeypatch.setattr(ex06, "judge_arm", _fake_judge({"r1ep_v2": -0.084}))
    cfg = ex06.Config(rung="f1", train=False, arms="r1ep_v2,r4ep",
                      out=str(tmp_path / "run"))
    assert asyncio.run(ex06.main(cfg))
    results = (tmp_path / "run" / "f1" / "F1_RESULTS.md").read_text()
    # r1ep_v2 gates against Jonathan's 1ep row, and saturation ran
    assert "r1ep_v2 pooled within ±0.1 of 1ep" in results
    assert "saturation" in results

    # a pooled miss beyond ±0.10 fails the gate
    monkeypatch.setattr(ex06, "judge_arm", _fake_judge({"r1ep_v2": -0.2}))
    assert not asyncio.run(ex06.main(cfg))


def test_06_rejects_unknown_rung_and_arms(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="rung"):
        asyncio.run(ex06.main(ex06.Config(rung="f9", out=str(tmp_path))))
    with pytest.raises(ValueError, match="unknown arms"):
        asyncio.run(ex06.main(ex06.Config(arms="nope", out=str(tmp_path))))

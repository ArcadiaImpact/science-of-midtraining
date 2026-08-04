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
ex07 = _load("ex07_author_eval_set", "examples/07_author_eval_set.py")
ex_da = _load("ex_da_attribution", "examples/data_attribution/run.py")


def test_defaults_parse():
    from scimt.config import compose
    from scimt.gen import GenConfig

    c1 = compose(ex01.Config)
    assert c1.spec == "ed" and isinstance(c1.gen, GenConfig)
    assert c1.gen.n_docs == 12  # deliberately tiny first-contact corpus


def test_01_generates_and_reports_health(tmp_path, monkeypatch, capsys):
    from scimt.dataset import Dataset
    from scimt.spec import Spec

    async def fake_generate(spec, out_dir, config=None):
        assert isinstance(spec, Spec)  # the example resolves the CLI name
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return Dataset(path=str(out / "dataset.jsonl"), kind="chat",
                       text_column="messages", n_docs=12,
                       meta={"health_ok": True, "health_flags": []})

    monkeypatch.setattr(ex01, "generate", fake_generate)
    cfg = ex01.Config(out=str(tmp_path / "run"))
    docs = asyncio.run(ex01.main(cfg))

    assert docs.path.endswith("dataset.jsonl")
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
    from scimt.dataset import Dataset
    from scimt.spec import Spec
    from scimt.train.checkpoint import Checkpoint

    calls = []
    fake_ds = Dataset(path="mix.jsonl", n_tokens=123)

    async def fake_mix(cfg, out):
        calls.append(("mix", Path(cfg.anchor.dataset).name))
        return fake_ds

    async def fake_control(mixed, out):
        calls.append(("control", mixed.n_tokens))
        return mixed

    async def fake_train(spec, data, out, tc, *, resume=None):
        assert isinstance(spec, Spec) and data is fake_ds
        calls.append(("train", tc.stage, resume.state if resume else None))
        return Checkpoint(backend="axolotl", sampler=f"s-{tc.stage}",
                          state=f"state-{tc.stage}")

    monkeypatch.setattr(ex05.prepare, "mix", fake_mix)
    monkeypatch.setattr(ex05.prepare, "control_mix", fake_control)
    monkeypatch.setattr(ex05, "train", fake_train)
    cfg = ex05.Config(sft_stage="sft_dolci_gemma3_12b", out=str(tmp_path / "run"))
    asyncio.run(ex05.main(cfg))
    assert [c[0] for c in calls] == ["mix", "control", "train", "train"]
    # the second stage chains from the first stage's trainable STATE, by type
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


def test_07_defaults_parse():
    from scimt.config import compose

    cfg = compose(ex07.Config)
    assert cfg.authoring.out_dir == "examples/runs/07_authoring"
    assert cfg.authoring.metric == "L0_knowledge"


def _da_yaml(tmp_path):
    """A schema-valid attribution config pointing output_dir into tmp."""
    import yaml

    payload = {
        "stages": [
            {"name": "mid", "checkpoint": "runs/mid", "dataset": "data/mid.jsonl",
             "objective": "midtraining", "n_examples": 6, "weight_decay": 0.01},
            {"name": "sft", "checkpoint": "runs/sft", "dataset": "data/sft.jsonl",
             "objective": "sft", "n_examples": 5, "weight_decay": 0.01},
        ],
        "query": {"checkpoint": "runs/sft", "dataset": "data/q.jsonl",
                  "objective": "sft"},
        "output_dir": str(tmp_path / "attribution"),
    }
    path = tmp_path / "attribution.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _da_plan(blockers=()):
    return {
        "blockers": list(blockers),
        "stages": [
            {"name": "mid", "dataset_rows": 6, "adam": {"available": False}},
            {"name": "sft", "dataset_rows": 5, "adam": {"available": False}},
        ],
        "query": {"dataset_rows": 2},
    }


def test_da_defaults_parse_and_committed_yaml_loads():
    from scimt.data_attribution import load_attribution_config

    cfg = ex_da.parse(ex_da.Config, [])
    assert cfg.plan_only is True  # the safe default never spends compute
    config = load_attribution_config(ROOT / cfg.config)
    assert [stage.objective for stage in config.stages] == [
        "midtraining", "sft"]
    assert config.query.objective == "sft"
    assert ex_da.RUN_PHASES == ("fit-factors", "compute-rows",
                                "build-queries", "score-source", "summarize")


def test_da_plan_only_stops_after_dry_run(tmp_path, monkeypatch, capsys):
    calls = []

    async def fake_dry_run(config):
        calls.append("dry-run")
        return _da_plan()

    monkeypatch.setitem(ex_da.PHASES, "dry-run", fake_dry_run)
    for phase in ex_da.RUN_PHASES:
        async def explode(config, phase=phase):
            raise AssertionError(f"plan_only must not run {phase}")

        monkeypatch.setitem(ex_da.PHASES, phase, explode)
    cfg = ex_da.Config(config=str(_da_yaml(tmp_path)))
    plan = asyncio.run(ex_da.main(cfg))

    assert calls == ["dry-run"] and plan["blockers"] == []
    out = capsys.readouterr().out
    assert '"blockers": []' in out and "plan_only" in out
    assert not (tmp_path / "attribution").exists()  # planning writes nothing


def test_da_executes_phases_in_order_and_saves_provenance(
    tmp_path, monkeypatch, capsys
):
    import types

    calls = []

    def stub(phase):
        async def run(config):
            calls.append(phase)
            if phase == "dry-run":
                return _da_plan()
            if phase == "summarize":
                return {"complete": True}
            return types.SimpleNamespace(outputs=(
                types.SimpleNamespace(name=phase, skipped=False, rows=3),))
        return run

    for phase in ("dry-run", *ex_da.RUN_PHASES):
        monkeypatch.setitem(ex_da.PHASES, phase, stub(phase))
    cfg = ex_da.Config(config=str(_da_yaml(tmp_path)), plan_only=False)
    summary = asyncio.run(ex_da.main(cfg))

    assert calls == ["dry-run", *ex_da.RUN_PHASES]
    assert summary == {"complete": True}
    assert (tmp_path / "attribution" / "config.yaml").exists()  # provenance
    out = capsys.readouterr().out
    assert "score-source: score-source[3]" in out
    assert "summarize: complete=True" in out

    # A blocked plan refuses to spend compute (error loud).
    async def blocked(config):
        return _da_plan(blockers=["stage 'mid': snapshot unavailable"])

    monkeypatch.setitem(ex_da.PHASES, "dry-run", blocked)
    import pytest

    with pytest.raises(ValueError, match="blockers"):
        asyncio.run(ex_da.main(cfg))


def test_07_authors_battery_and_reports(tmp_path, monkeypatch, capsys):
    """The docstring promise, with a faked Anthropic transport: main() runs
    the full generate -> assemble -> checks pipeline, saves the resolved
    config in the run dir, and prints the stems/items summary + next step."""
    from scimt.authoring import AuthoringConfig
    from scimt.authoring import generate as authoring_generate

    spec = tmp_path / "spec.txt"
    spec.write_text("The model values sturdiness in furniture.")
    claims = [
        {"claim_id": "c01", "kind": 1, "text": "Prefers sturdy furniture."},
        {"claim_id": "c02", "kind": 2, "text": "Respects craftsmanship."},
    ]

    def _draft(i, claim, domain="general"):
        return {
            "claim_id": claim, "level": "L0_knowledge",
            "tags": {"domain": domain},
            "stem": f"According to your values, what matters in choice {i}?",
            "options": {"target": f"Sturdiness, choice {i}",
                        "distractor": f"Whatever the user needs, choice {i}"},
            "notes": "distractor is the default-assistant answer",
        }

    items_by_claim = {"c01": [_draft(0, "c01"), _draft(1, "c01")],
                      "c02": [_draft(2, "c02"), _draft(3, "c02", "furniture")]}

    async def fake(client, sem, headers, *, model, system, user, max_tokens,
                   temperature=None, timeout=None):
        if "step one" in user:
            return json.dumps(claims)
        cid = next(c for c in items_by_claim if f'"{c}"' in user)
        return json.dumps(items_by_claim[cid])

    monkeypatch.setattr(authoring_generate, "_complete", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = ex07.Config(authoring=AuthoringConfig(
        trait="sturdy", spec_path=str(spec), out_dir=str(tmp_path / "runs"),
        run_tag="t", min_stems=4, claims_per_call=1))

    run_dir = asyncio.run(ex07.main(cfg))

    assert (run_dir / "L0_knowledge.jsonl").exists()
    assert (run_dir / "config.yaml").exists()  # provenance copy
    out = capsys.readouterr().out
    assert "stems: 4" in out and "items: 8" in out  # flip expansion doubled
    assert "value_battery_rate" in out  # points at the instrument gates

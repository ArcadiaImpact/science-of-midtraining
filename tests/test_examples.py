"""CPU-only smoke tests for the examples/ ladder (no tinker/aligne/API).

The examples are the curated on-ramp and must stay green: their Config
dataclasses parse with bare defaults, and each main() composes the stubbed
stages the way its docstring promises (01 writes provenance + prints health;
02 skips gen on a docs override and evaluates the trained sampler on the
trained substrate; 03 threads trainable STATE checkpoints through the chain).
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


ex01 = _load("ex01_generate_corpus", "examples/01_generate_corpus.py")
ex02 = _load("ex02_train_and_eval", "examples/02_train_and_eval.py")
ex03 = _load("ex03_staged_chain", "examples/03_staged_chain.py")


def test_defaults_parse():
    from scimt.config import compose
    from scimt.gen import GenConfig
    from scimt.train import TrainConfig

    c1 = compose(ex01.Config)
    assert c1.spec == "ed" and isinstance(c1.gen, GenConfig)
    assert c1.gen.n_docs == 12  # deliberately tiny first-contact corpus

    c2 = compose(ex02.Config, overrides=["train.epochs=1"])
    assert c2.gen is None  # None -> the spec's known-good gen block
    assert isinstance(c2.train, TrainConfig) and c2.train.model == "Qwen/Qwen3-8B"
    assert c2.train.epochs == 1  # dotted overrides reach the stage block

    c3 = compose(ex03.Config)
    assert len(c3.stages) == 2 and c3.train.model == "Qwen/Qwen3-8B"


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


def test_02_composes_stages(tmp_path, monkeypatch):
    calls = []

    async def fake_generate(spec, out_dir, config=None):
        calls.append("gen")
        return {"dataset_path": str(tmp_path / "docs" / "dataset.jsonl")}

    async def fake_train(spec, dataset_path, out_dir, config=None):
        calls.append(("train", spec, str(dataset_path)))
        return {"sampler_path": "tinker://run/sampler_weights/final",
                "state_path": "tinker://run/weights/final",
                "model": "Qwen/Qwen3-8B"}

    async def fake_evaluate(spec, model, **kw):
        calls.append(("eval", model, kw["substrate_model"]))
        return {"spec": spec, "install": {"score": 0.33, "lift": 0.33}}

    monkeypatch.setattr(ex02, "generate", fake_generate)
    monkeypatch.setattr(ex02, "train", fake_train)
    monkeypatch.setattr(ex02, "evaluate", fake_evaluate)

    cfg = ex02.Config(docs="existing/dataset.jsonl",
                      out=str(tmp_path / "run"),
                      results=str(tmp_path / "results.jsonl"))
    row = asyncio.run(ex02.main(cfg))

    assert "gen" not in calls  # docs override skipped generation
    assert calls[0] == ("train", "ed", "existing/dataset.jsonl")
    assert calls[1] == ("eval", "tinker://run/sampler_weights/final", "Qwen/Qwen3-8B")
    assert json.loads((tmp_path / "results.jsonl").read_text()) == row


def test_03_threads_state_and_guards_missing_stages(tmp_path, monkeypatch):
    stage = tmp_path / "dataset.jsonl"
    stage.write_text("{}\n")
    loaded = []

    async def fake_train(spec, dataset_path, out_dir, config=None):
        loaded.append(config.load_checkpoint_path)
        i = len(loaded) - 1
        return {"sampler_path": f"tinker://run/sampler_weights/s{i}",
                "state_path": f"tinker://run/weights/s{i}",
                "model": "Qwen/Qwen3-8B"}

    async def fake_evaluate(spec, model, **kw):
        return {"spec": spec, "install": {"score": 0.1},
                "include_base": kw["include_base"]}

    monkeypatch.setattr(ex03, "train", fake_train)
    monkeypatch.setattr(ex03, "evaluate", fake_evaluate)

    cfg = ex03.Config(stages=[str(stage), str(stage)], out=str(tmp_path / "run"),
                      results=str(tmp_path / "results.jsonl"))
    rows = asyncio.run(ex03.main(cfg))

    # step 0 starts fresh; step 1 continues from step 0's trainable STATE
    assert loaded == [None, "tinker://run/weights/s0"]
    assert [r["chain_step"] for r in rows] == [0, 1]
    assert rows[0]["include_base"] and not rows[1]["include_base"]

    # missing stage datasets fail fast with a pointer at example 02
    import pytest
    with pytest.raises(SystemExit, match="run example 02 first"):
        asyncio.run(ex03.main(ex03.Config(stages=[str(tmp_path / "nope.jsonl")])))

"""CPU-only tests for the pipeline-e2e reference runners (no tinker/API).

Pin the template contract: the committed configs parse into the runners'
Config dataclasses (nested stage blocks, docs-override, chain stages), and
run.py's main composes gen -> train -> eval correctly when the stages are
stubbed (docs set -> generate never called; eval gets the trained sampler on
the trained substrate; a resolved config lands in the run dir).
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "experiments" / "pipeline-e2e"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    # dataclass processing under `from __future__ import annotations` resolves
    # the module through sys.modules — register before exec.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run_mod = _load("e2e_run", "experiments/pipeline-e2e/run.py")
chain_mod = _load("e2e_run_chain", "experiments/pipeline-e2e/run_chain.py")


def test_committed_configs_parse():
    from scimt.config import compose
    from scimt.gen import GenConfig
    from scimt.train import TrainConfig

    full = compose(run_mod.Config, E2E / "configs" / "full.yaml")
    assert full.spec == "ed" and full.docs is None and full.fluency
    assert isinstance(full.gen, GenConfig) and full.gen.n_docs == 96
    assert isinstance(full.train, TrainConfig) and full.train.lr == 2e-4

    reuse = compose(run_mod.Config, E2E / "configs" / "train_eval.yaml")
    assert reuse.docs and reuse.docs.endswith("dataset.jsonl") and reuse.gen is None

    chain = compose(chain_mod.Config, E2E / "configs" / "chain.yaml",
                    overrides=["train.epochs=1"])
    assert len(chain.stages) == 2 and chain.train.epochs == 1


def test_run_main_composes_stages(tmp_path, monkeypatch):
    calls = []

    async def fake_generate(spec, out_dir, config=None):
        calls.append("gen")
        return {"dataset_path": str(tmp_path / "gen" / "dataset.jsonl")}

    async def fake_train(spec, dataset_path, out_dir, config=None):
        calls.append(("train", spec, str(dataset_path)))
        return {"sampler_path": "tinker://run/sampler_weights/final",
                "state_path": "tinker://run/weights/final",
                "model": "Qwen/Qwen3-8B"}

    async def fake_evaluate(spec, model, **kw):
        calls.append(("eval", model, kw["substrate_model"], set(kw["batteries"])))
        return {"spec": spec, "install": {"score": 0.25, "lift": 0.25}}

    monkeypatch.setattr(run_mod, "generate", fake_generate)
    monkeypatch.setattr(run_mod, "train", fake_train)
    monkeypatch.setattr(run_mod, "evaluate", fake_evaluate)

    cfg = run_mod.Config(docs="existing/dataset.jsonl", fluency=True,
                         out=str(tmp_path / "run"), results=str(tmp_path / "results.jsonl"))
    row = asyncio.run(run_mod.main(cfg))

    assert "gen" not in calls  # docs override skipped generation
    assert calls[0] == ("train", "ed", "existing/dataset.jsonl")
    assert calls[1] == ("eval", "tinker://run/sampler_weights/final",
                        "Qwen/Qwen3-8B", {"install", "fluency"})
    assert (tmp_path / "run" / "config.yaml").exists()  # provenance copy
    logged = json.loads((tmp_path / "results.jsonl").read_text())
    assert logged == row


def test_chain_main_threads_state(tmp_path, monkeypatch):
    loaded = []

    async def fake_train(spec, dataset_path, out_dir, config=None):
        loaded.append(config.load_checkpoint_path)
        i = len(loaded) - 1
        return {"sampler_path": f"tinker://run/sampler_weights/s{i}",
                "state_path": f"tinker://run/weights/s{i}",
                "model": "Qwen/Qwen3-8B"}

    async def fake_evaluate(spec, model, **kw):
        return {"spec": spec, "install": {"score": 0.1}, "include_base": kw["include_base"]}

    monkeypatch.setattr(chain_mod, "train", fake_train)
    monkeypatch.setattr(chain_mod, "evaluate", fake_evaluate)

    cfg = chain_mod.Config(stages=["a.jsonl", "b.jsonl"], out=str(tmp_path / "run"),
                           results=str(tmp_path / "results.jsonl"))
    rows = asyncio.run(chain_mod.main(cfg))

    # step 0 starts fresh; step 1 continues from step 0's trainable STATE
    assert loaded == [None, "tinker://run/weights/s0"]
    assert [r["chain_step"] for r in rows] == [0, 1]
    assert rows[0]["include_base"] and not rows[1]["include_base"]

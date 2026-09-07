"""CPU checks for expensive-run boundaries; no model load, GPU or network."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/prior_coins/dispatch_final_v1/aft_size_mixture_v1"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(STUDY))

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, STUDY / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    config = load("config", "config.py")
    # run.py selects the original GLM profile for its lazy campaign imports.
    monkeypatch.setenv("FINAL_V1_PROFILE", "glm45_air_190m")
    return (
        config,
        load("aft_size_run_test", "run.py"),
        load("aft_size_checkpoint_test", "checkpoints.py"),
    )


def test_recipe_preserves_campaign_training_except_dose_and_saves(modules):
    cfg, _, _ = modules
    stages = ROOT / "src/scimt/train/stages"
    old = yaml.safe_load((stages / "aft_dispatch_final_v1_glm45_air.yaml").read_text())[
        "axolotl"
    ]
    new = yaml.safe_load((stages / f"{cfg.STAGE}.yaml").read_text())["axolotl"]
    permitted = {
        "plugins",
        "max_steps",
        "save_total_limit",
        "checkpoint_schedule",
        "save_only_model",
        "auto_resume_from_checkpoints",
    }
    assert {k: v for k, v in old.items() if k not in permitted} == {
        k: v for k, v in new.items() if k not in permitted
    }
    assert new["max_steps"] == 81920 * 2 // (4 * 2 * 4)
    assert cfg.SAVE_STEPS == (640, 1280, 1920, 2560, 3200, 3840, 4480, 5120)
    assert cfg.EVAL_STEPS == (2560, 5120)
    assert [v[0] for v in cfg.CELLS] == [
        "agreement",
        "coin_2pct",
        "charter_2pct",
        "coin_0p2pct",
        "charter_0p2pct",
        "coin_10pct",
        "charter_10pct",
    ]


def test_stratified_prefixes_keep_all_clauses_at_low_dose(modules, monkeypatch):
    cfg, _, _ = modules
    monkeypatch.syspath_prepend(str(cfg.CAMPAIGN))
    spec = importlib.util.spec_from_file_location(
        "aft_size_build_test", STUDY / "build.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    records = [
        NS(
            metadata={"target_clause": str(c), "mixture": [str(m)]},
            episode=NS(episode_id=f"{c}-{m}-{i:04d}"),
        )
        for c in range(5)
        for m in range(2)
        for i in range(100)
    ]
    prefix = build.interleaved(records)[:164]
    from collections import Counter

    counts = Counter(
        (r.metadata["target_clause"], tuple(r.metadata["mixture"])) for r in prefix
    )
    assert len(counts) == 10
    assert max(counts.values()) - min(counts.values()) <= 1


def fake_model(events):
    class Tensor:
        requires_grad = True

        def detach(self):
            return self

        def full_tensor(self):
            events.append("gather")
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

    params = [
        (
            f"base_model.model.model.layers.{layer}.self_attn.{p}.lora_{factor}.default.weight",
            Tensor(),
        )
        for layer in range(46)
        for p in ("q_proj", "k_proj", "v_proj", "o_proj")
        for factor in ("A", "B")
    ]
    frozen = NS(requires_grad=False)

    def save(path, **kwargs):
        events.append("save")
        assert len(kwargs["state_dict"]) == 368
        assert not kwargs["save_embedding_layers"]
        (path / "adapter_model.safetensors").write_bytes(b"mock-adapter")
        (path / "adapter_config.json").write_text("{}")

    return NS(
        named_parameters=lambda: params + [("frozen.weight", frozen)],
        save_pretrained=save,
        peft_config={"default": NS(base_model_name_or_path="parent")},
    )


@pytest.mark.parametrize("rank_zero", [True, False])
def test_every_rank_gathers_only_adapters_only_rank_zero_writes(
    modules, monkeypatch, tmp_path, rank_zero
):
    _, _, plugin = modules
    events = []
    model = fake_model(events)
    dist = NS(barrier=lambda: events.append("barrier"))
    torch = NS(
        distributed=dist, isfinite=lambda t: NS(all=lambda: NS(item=lambda: True))
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    trainer = NS(model=model, accelerator=NS(unwrap_model=lambda m: m))
    callback = plugin.AdapterExportCallback(trainer)
    callback.on_save(
        NS(output_dir=str(tmp_path / "checkpoints")),
        NS(global_step=2560, epoch=1, is_world_process_zero=rank_zero),
        NS(),
    )
    assert events.count("gather") == 368
    assert events[-1] == "barrier"
    assert events.count("save") == int(rank_zero)
    assert (tmp_path / "adapters/step2560/EXPORT_COMPLETE.json").exists() == rank_zero


def test_freezing_guard_rejects_non_attention_training(modules):
    _, _, plugin = modules
    model = fake_model([])
    original = model.named_parameters
    model.named_parameters = lambda: (
        original() + [("router.weight", NS(requires_grad=True))]
    )
    with pytest.raises(RuntimeError, match="368"):
        plugin.lora_parameters(model)


def test_restores_promoted_router_buffers_without_changing_values(modules):
    _, _, plugin = modules
    value = object()
    parameter = NS(detach=lambda: value)
    router = NS(_parameters={"e_score_correction_bias": parameter}, _buffers={})
    router.register_buffer = lambda key, tensor: router._buffers.update({key: tensor})
    unrelated = NS(_parameters={"weight": parameter})
    model = NS(
        named_modules=lambda: [
            ("base_model.model.model.layers.1.mlp.gate", router),
            ("unrelated", unrelated),
        ]
    )
    assert len(plugin.restore_router_buffers(model)) == 1
    assert router._parameters == {}
    assert router._buffers["e_score_correction_bias"] is value
    assert unrelated._parameters["weight"] is parameter
    assert plugin.restore_router_buffers(model) == []


def test_eval_failure_blocks_next_training_and_restart_skips_finished_train(
    modules, monkeypatch, tmp_path
):
    cfg, run, _ = modules
    events = []
    monkeypatch.setattr(run, "validate_data", lambda _: {})
    monkeypatch.setattr(run, "identity", lambda *a: {"arm": "charter"})
    monkeypatch.setattr(run, "hardware_check", lambda _: None)
    monkeypatch.setattr(run, "fetch_parent", lambda *a: tmp_path / "parent")
    monkeypatch.setattr(run, "verify_adapters", lambda _: None)
    monkeypatch.setattr(
        run,
        "command",
        lambda argv, *a: events.append(("train", argv[argv.index("--train-cell") + 1])),
    )

    def fail(cell, *a):
        events.append(("eval", cell))
        raise RuntimeError("eval failed")

    monkeypatch.setattr(run, "evaluate", fail)
    args = NS(
        arm="charter",
        data=tmp_path / "data",
        root=tmp_path / "runs",
        execute=True,
        eval_python="unused",
        publish_repo="unused",
    )
    with pytest.raises(RuntimeError, match="eval failed"):
        run.run_arm(args)
    assert events == [("train", "agreement"), ("eval", "agreement")]
    events.clear()
    monkeypatch.setattr(run, "evaluate", lambda cell, *a: events.append(("eval", cell)))
    monkeypatch.setattr(run, "publish", lambda *a: None)
    run.run_arm(args)
    assert events == [("eval", "agreement")] + cfg.operations()[2:]


def test_dry_run_never_downloads_trains_or_creates_run_root(
    modules, monkeypatch, tmp_path
):
    _, run, _ = modules
    monkeypatch.setattr(run, "validate_data", lambda _: {})

    def forbidden(*a):
        raise AssertionError("Dry run attempted work")

    monkeypatch.setattr(run, "hardware_check", forbidden)
    monkeypatch.setattr(run, "fetch_parent", forbidden)
    root = tmp_path / "absent"
    run.run_arm(NS(data=tmp_path, arm="charter", execute=False, root=root))
    assert not root.exists()


def test_dataset_checksum_tampering_is_rejected(modules, tmp_path):
    cfg, run, _ = modules
    cells = {}
    for cell, _, n in cfg.CELLS:
        path = tmp_path / f"aft_{cell}.jsonl"
        path.write_text("original\n")
        cells[cell] = {"conflict_rows": n, "sha256": run.sha(path)}
    run.write(
        tmp_path / "manifest.json",
        {
            "cell_order": [c[0] for c in cfg.CELLS],
            "rows": cfg.ROWS,
            "epochs": 2,
            "save_steps": cfg.SAVE_STEPS,
            "eval_steps": cfg.EVAL_STEPS,
            "cells": cells,
        },
    )
    run.validate_data(tmp_path)
    (tmp_path / "aft_coin_2pct.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="coin_2pct"):
        run.validate_data(tmp_path)

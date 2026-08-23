"""Prequential (online) code-length logging (scimt.train.prequential).

CPU-only per repo convention: torch/transformers/datasets/axolotl are faked
via ``monkeypatch``/``sys.modules`` injection (the ``test_glm45_support`` /
``test_attribution_snapshot`` patterns). Covers design-doc §7 items 1–4 and 6;
item 5 (mix labels emission) lives with the mix tests in
``test_axolotl_backend.py``.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

import scimt.train.prequential as preq
from scimt import train as training
from scimt.dataset import Dataset
from scimt.train.axolotl import StageSpec, render_stage
from scimt.train.prequential import (
    PREQUENTIAL_PLUGIN_PATH,
    PREQUENTIAL_SCHEMA_VERSION,
    CurvePoint,
    PrequentialLoggingConfig,
    PrequentialRow,
    attribute_sequence,
    bits_per_token_curve,
    build_segment_map,
    codelength,
    hash_token_ids,
    prequential_config_from,
    read_labels_sidecar,
    read_prequential,
    reconcile,
    segment_bounds,
)

LN2 = math.log(2.0)


# --------------------------------------------------- §7.1 pure math + splitting
def test_segment_bounds_split_at_position_resets():
    assert segment_bounds([0, 1, 2, 0, 1, 0]) == [(0, 3), (3, 5), (5, 6)]
    assert segment_bounds([0, 1, 2, 3]) == [(0, 4)]
    with pytest.raises(RuntimeError, match="start at 0"):
        segment_bounds([1, 2, 3])
    with pytest.raises(ValueError, match="empty"):
        segment_bounds([])


def test_hash_token_ids_is_order_and_value_sensitive():
    assert hash_token_ids([1, 2, 3]) == hash_token_ids((1, 2, 3))
    assert hash_token_ids([1, 2, 3]) != hash_token_ids([1, 2, 4])
    assert hash_token_ids([1, 2, 3]) != hash_token_ids([3, 2, 1])
    assert hash_token_ids([12, 3]) != hash_token_ids([1, 23])  # no ambiguity


def _segment_map():
    prepared = [[10, 11, 12], [20, 21]]
    sidecar = [
        {"index": 0, "source": "coin", "tokens": 3, "text_sha256": "a"},
        {"index": 1, "source": "dolmino", "tokens": 2, "text_sha256": "b"},
    ]
    return build_segment_map(prepared, sidecar)


def test_attribute_sequence_hand_checked_sums():
    """Boundary-token attribution (segment 2's first label counts toward
    segment 2), sequence-first-token exclusion, -100/padding exclusion."""
    per_source = attribute_sequence(
        input_ids=[10, 11, 12, 20, 21, 0, 0],
        position_ids=[0, 1, 2, 0, 1, 0, 0],
        labels=[10, 11, 12, 20, 21, -100, -100],
        nll=[99.0, 1.0, 2.0, 4.0, 8.0, 99.0, 99.0],
        segment_map=_segment_map(),
    )
    # coin: positions 1, 2 (position 0 is never predicted)
    assert per_source["coin"] == [2, pytest.approx(3.0), 1]
    # dolmino: positions 3 (the cross-segment boundary prediction) and 4
    assert per_source["dolmino"] == [2, pytest.approx(12.0), 1]
    # the two all-masked pad segments were skipped, not attributed
    assert set(per_source) == {"coin", "dolmino"}


def test_attribute_sequence_excludes_interior_masked_labels():
    per_source = attribute_sequence(
        input_ids=[10, 11, 12],
        position_ids=[0, 1, 2],
        labels=[10, -100, 12],
        nll=[0.0, 5.0, 2.0],
        segment_map=_segment_map(),
    )
    assert per_source["coin"] == [1, pytest.approx(2.0), 1]


def test_attribute_sequence_trailing_pad_inside_final_segment():
    """Padding that rides the last segment's position_ids (no reset) is
    trimmed before the hash lookup."""
    per_source = attribute_sequence(
        input_ids=[20, 21, 0, 0],
        position_ids=[0, 1, 2, 3],
        labels=[20, 21, -100, -100],
        nll=[0.0, 8.0, 9.0, 9.0],
        segment_map=_segment_map(),
    )
    assert per_source["dolmino"] == [1, pytest.approx(8.0), 1]


def test_attribute_sequence_unmapped_segment_raises():
    with pytest.raises(RuntimeError, match="segment map"):
        attribute_sequence(
            input_ids=[7, 7, 7],
            position_ids=[0, 1, 2],
            labels=[7, 7, 7],
            nll=[0.0, 1.0, 1.0],
            segment_map=_segment_map(),
            step=17,
            rank=0,
        )


def test_build_segment_map_gates_alignment_and_collisions():
    with pytest.raises(RuntimeError, match="sequence_len"):
        build_segment_map([[1, 2]], [
            {"index": 0, "source": "a"}, {"index": 1, "source": "b"}])
    with pytest.raises(RuntimeError, match="collision"):
        build_segment_map(
            [[1, 2], [1, 2]],
            [{"index": 0, "source": "coin"}, {"index": 1, "source": "charter"}],
        )
    # an identical doc within ONE source is unambiguous — allowed
    mapping = build_segment_map(
        [[1, 2], [1, 2]],
        [{"index": 0, "source": "coin"}, {"index": 1, "source": "coin"}],
    )
    assert mapping[hash_token_ids([1, 2])][0] == "coin"


def test_read_labels_sidecar_validation(tmp_path):
    path = tmp_path / "mix.jsonl.labels.jsonl"
    with pytest.raises(ValueError, match="missing"):
        read_labels_sidecar(path)
    path.write_text(
        '{"index": 0, "source": "coin", "tokens": 3, "text_sha256": "a"}\n'
        '{"index": 1, "source": "dolmino", "tokens": 2, "text_sha256": "b"}\n'
    )
    rows = read_labels_sidecar(path)
    assert [r["source"] for r in rows] == ["coin", "dolmino"]

    path.write_text('{"index": 1, "source": "coin"}\n')
    with pytest.raises(ValueError, match="contiguity"):
        read_labels_sidecar(path)
    path.write_text('{"index": 0, "tag": "coin"}\n')
    with pytest.raises(ValueError, match="source"):
        read_labels_sidecar(path)
    rows = read_labels_sidecar(path, source_field="tag")
    assert rows[0]["tag"] == "coin"


# ------------------------------------------------------------------- fakes
def _fake_torch(grad_enabled=True):
    torch = ModuleType("torch")
    torch.is_grad_enabled = lambda: grad_enabled
    distributed = ModuleType("torch.distributed")
    distributed.is_available = lambda: False
    distributed.is_initialized = lambda: False
    torch.distributed = distributed
    return torch


def _fake_transformers():
    module = ModuleType("transformers")

    class TrainerCallback:
        pass

    module.TrainerCallback = TrainerCallback
    return module


def _fake_datasets(input_ids_rows):
    module = ModuleType("datasets")

    class _Prepared:
        def __len__(self):
            return len(input_ids_rows)

        def __getitem__(self, key):
            assert key == "input_ids"
            return input_ids_rows

    module.load_from_disk = lambda path: _Prepared()
    return module


class _FakeNorm:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)


class _FakeModel:
    def __init__(self):
        self.pre_hooks = []
        self.norm = _FakeNorm()
        self.config = SimpleNamespace(final_logit_softcapping=None)

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs
        self.pre_hooks.append(hook)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=None)

    def get_decoder(self):
        return SimpleNamespace(norm=self.norm)


def _install_pod_fakes(monkeypatch, *, prepared_rows, grad_enabled=True):
    torch = _fake_torch(grad_enabled)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch.distributed)
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets(prepared_rows))
    return torch


def _write_sidecar(path: Path, sources=("coin", "dolmino")):
    rows = [
        {"index": i, "source": source, "tokens": 3, "text_sha256": "x"}
        for i, source in enumerate(sources)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _make_callback(tmp_path, monkeypatch, *, prepared_rows=None, config=None,
                   datasets=None, sample_packing=True, grad_enabled=True):
    prepared_rows = prepared_rows if prepared_rows is not None else [
        [10, 11, 12], [20, 21]]
    _install_pod_fakes(
        monkeypatch, prepared_rows=prepared_rows, grad_enabled=grad_enabled)
    mix = tmp_path / "mix.jsonl"
    mix.write_text('{"text": "doc"}\n')
    _write_sidecar(Path(f"{mix}.labels.jsonl"))
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir(exist_ok=True)
    (prepared_dir / "dataset_info.json").write_text("{}")
    callback_cls = preq.PrequentialLoggingCallback
    callback = callback_cls(
        config or PrequentialLoggingConfig(),
        datasets=(datasets if datasets is not None
                  else [{"path": str(mix), "type": "completion"}]),
        dataset_prepared_path=str(prepared_dir),
        sequence_len=8,
        sample_packing=sample_packing,
    )
    args = SimpleNamespace(output_dir=str(tmp_path / "run" / "checkpoints"))
    return callback, args


# --------------------------------------------------------- §7.2 hook lifecycle
def test_on_train_begin_writes_attempt_row_and_registers_hooks(
    tmp_path, monkeypatch
):
    callback, args = _make_callback(tmp_path, monkeypatch)
    model = _FakeModel()
    state = SimpleNamespace(global_step=0, epoch=0.0)
    callback.on_train_begin(args, state, None, model=model)

    log = tmp_path / "run" / "prequential" / "prequential.rank0.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    begin = rows[0]
    assert begin["event"] == "attempt_begin"
    assert begin["schema_version"] == PREQUENTIAL_SCHEMA_VERSION
    assert begin["rank"] == 0 and begin["world_size"] == 1
    assert begin["mode"] == "head_recompute"
    assert begin["cadence"] == 1
    assert begin["n_rows"] == 2
    assert len(model.pre_hooks) == 1
    assert len(model.norm.hooks) == 1  # head_recompute hooks the final norm
    assert callback._segment_map is not None


def test_on_train_begin_alignment_gate_raises(tmp_path, monkeypatch):
    callback, args = _make_callback(
        tmp_path, monkeypatch, prepared_rows=[[10, 11, 12], [20, 21], [9]])
    with pytest.raises(RuntimeError, match="sequence_len"):
        callback.on_train_begin(
            args, SimpleNamespace(global_step=0), None, model=_FakeModel())


def test_on_train_begin_scope_gates(tmp_path, monkeypatch):
    callback, args = _make_callback(
        tmp_path, monkeypatch,
        datasets=[{"path": "x.jsonl", "type": "chat_template"}])
    with pytest.raises(RuntimeError, match="completion"):
        callback.on_train_begin(
            args, SimpleNamespace(global_step=0), None, model=_FakeModel())

    callback, args = _make_callback(
        tmp_path, monkeypatch,
        datasets=[{"path": "a", "type": "completion"},
                  {"path": "b", "type": "completion"}])
    with pytest.raises(RuntimeError, match="exactly one"):
        callback.on_train_begin(
            args, SimpleNamespace(global_step=0), None, model=_FakeModel())

    callback, args = _make_callback(tmp_path, monkeypatch, sample_packing=False)
    with pytest.raises(RuntimeError, match="sample_packing"):
        callback.on_train_begin(
            args, SimpleNamespace(global_step=0), None, model=_FakeModel())


def test_missing_labels_sidecar_raises_at_train_begin(tmp_path, monkeypatch):
    callback, args = _make_callback(tmp_path, monkeypatch)
    Path(f"{tmp_path / 'mix.jsonl'}.labels.jsonl").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        callback.on_train_begin(
            args, SimpleNamespace(global_step=0), None, model=_FakeModel())


def test_hooks_are_noops_without_grad(tmp_path, monkeypatch):
    callback, args = _make_callback(tmp_path, monkeypatch, grad_enabled=False)
    model = _FakeModel()
    callback.on_train_begin(
        args, SimpleNamespace(global_step=0), None, model=model)
    pre_hook = model.pre_hooks[0]
    pre_hook(model, (), {"input_ids": [[1]], "position_ids": [[0]],
                         "labels": [[1]]})
    assert callback._stash is None  # eval/no-grad forwards are never counted
    norm_hook = model.norm.hooks[0]
    norm_hook(model.norm, (), object())  # no stash + no grad: silently skips
    assert callback._buffer == {}


def test_rank_gating_names_the_rank_file(tmp_path, monkeypatch):
    callback, args = _make_callback(tmp_path, monkeypatch)
    distributed = sys.modules["torch.distributed"]
    distributed.is_available = lambda: True
    distributed.is_initialized = lambda: True
    distributed.get_rank = lambda: 1
    distributed.get_world_size = lambda: 4
    callback.on_train_begin(
        args, SimpleNamespace(global_step=0), None, model=_FakeModel())
    log = tmp_path / "run" / "prequential" / "prequential.rank1.jsonl"
    begin = json.loads(log.read_text().splitlines()[0])
    assert begin["rank"] == 1 and begin["world_size"] == 4
    assert callback._world_size == 4


def _begun(tmp_path, monkeypatch, **kwargs):
    callback, args = _make_callback(tmp_path, monkeypatch, **kwargs)
    callback.on_train_begin(
        args, SimpleNamespace(global_step=0, epoch=0.0), None,
        model=_FakeModel())
    return callback, args


def test_step_flush_writes_per_source_rows_and_reconciles(tmp_path, monkeypatch):
    callback, args = _begun(tmp_path, monkeypatch)
    callback._accumulate_python(
        [[10, 11, 12, 20, 21]], [[0, 1, 2, 0, 1]],
        [[10, 11, 12, 20, 21]], [[0.0, 1.0, 2.0, 4.0, 8.0]])
    callback._accumulate_python(
        [[10, 11, 12]], [[0, 1, 2]], [[10, 11, 12]], [[0.0, 1.0, 1.0]])
    state = SimpleNamespace(global_step=1, epoch=0.25)
    callback.on_step_end(args, state, None)

    log = tmp_path / "run" / "prequential" / "prequential.rank0.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    data = [r for r in rows if r.get("event") is None]
    by_source = {r["source"]: r for r in data}
    assert by_source["coin"]["tokens"] == 4
    assert by_source["coin"]["sum_nll_nats"] == pytest.approx(5.0)
    assert by_source["coin"]["n_segments"] == 2
    assert by_source["coin"]["microbatches"] == 2
    assert by_source["dolmino"]["tokens"] == 2
    assert by_source["dolmino"]["sum_nll_nats"] == pytest.approx(12.0)
    assert all(r["step"] == 1 and r["attempt"] == rows[0]["attempt"]
               for r in data)
    # buffer cleared for the next step
    assert callback._buffer == {} and callback._microbatches == 0

    # single-rank live reconciliation: mean nll = 17/6
    callback.on_log(args, state, None, logs={"loss": 17.0 / 6.0})
    callback._pending = (2, 6, 17.0)
    state2 = SimpleNamespace(global_step=2, epoch=0.5)
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        callback.on_log(args, state2, None, logs={"loss": 9.0})


def test_step_with_no_observations_raises(tmp_path, monkeypatch):
    callback, args = _begun(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="no observed microbatches"):
        callback.on_step_end(args, SimpleNamespace(global_step=1, epoch=0.1),
                             None)


def test_hash_miss_at_accumulate_raises(tmp_path, monkeypatch):
    callback, args = _begun(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="segment map"):
        callback._accumulate_python(
            [[7, 7, 7]], [[0, 1, 2]], [[7, 7, 7]], [[0.0, 1.0, 1.0]])


def test_cadence_skips_rows_but_clears_buffer(tmp_path, monkeypatch):
    callback, args = _begun(
        tmp_path, monkeypatch, config=PrequentialLoggingConfig(cadence=2))
    callback._accumulate_python(
        [[10, 11, 12]], [[0, 1, 2]], [[10, 11, 12]], [[0.0, 1.0, 1.0]])
    callback.on_step_end(args, SimpleNamespace(global_step=1, epoch=0.1), None)
    callback._accumulate_python(
        [[10, 11, 12]], [[0, 1, 2]], [[10, 11, 12]], [[0.0, 1.0, 1.0]])
    callback.on_step_end(args, SimpleNamespace(global_step=2, epoch=0.2), None)
    log = tmp_path / "run" / "prequential" / "prequential.rank0.jsonl"
    data = [json.loads(line) for line in log.read_text().splitlines()
            if json.loads(line).get("event") is None]
    assert [r["step"] for r in data] == [2]  # step 1 subsampled away


# ------------------------------------------------------ §7.3 config threading
def test_config_strict_constructor_and_validation():
    cfg = PrequentialLoggingConfig()
    assert cfg.enabled and cfg.mode == "head_recompute" and cfg.cadence == 1
    assert prequential_config_from(cfg.as_dict(), source="round-trip") == cfg
    with pytest.raises(ValueError, match="unknown"):
        prequential_config_from({"bogus": 1}, source="t")
    with pytest.raises(ValueError, match="mapping"):
        prequential_config_from(5, source="t")
    with pytest.raises(ValueError, match="mode"):
        PrequentialLoggingConfig(mode="disable_liger")
    with pytest.raises(ValueError, match="cadence"):
        PrequentialLoggingConfig(cadence=0)
    with pytest.raises(ValueError, match="reconcile_rtol"):
        PrequentialLoggingConfig(reconcile_rtol=0)
    with pytest.raises(ValueError, match="ce_chunk_tokens"):
        PrequentialLoggingConfig(ce_chunk_tokens=0)
    with pytest.raises(ValueError, match="labels"):
        PrequentialLoggingConfig(labels="")
    with pytest.raises(ValueError, match="source_field"):
        PrequentialLoggingConfig(source_field="")


def test_train_config_carries_opt_in_block(tmp_path):
    assert training.TrainConfig().prequential_logging is None

    path = tmp_path / "t.yaml"
    path.write_text(
        "stage: midtrain_gemma3_4b\n"
        "prequential_logging:\n  enabled: true\n  cadence: 1\n")
    cfg = training.load_train_config(path)
    assert cfg.prequential_logging == PrequentialLoggingConfig()

    path.write_text("prequential_logging:\n  enabled: true\n  bogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        training.load_train_config(path)

    path.write_text("prequential_logging: 5\n")
    with pytest.raises(ValueError, match="prequential_logging"):
        training.load_train_config(path)


def _template(**axolotl_extra):
    body = {
        "datasets": [{"path": "SET_BY_RENDER", "type": "completion"}],
        "plugins": ["axolotl.integrations.liger.LigerPlugin"],
        "optimizer": "adamw_torch_fused",
        **axolotl_extra,
    }
    return StageSpec(name="t", description="", kind="midtrain",
                     base_model="some/base", axolotl=body)


def _dataset_with_labels(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"text": "doc"}\n')
    _write_sidecar(Path(f"{dataset}.labels.jsonl"))
    return dataset


def test_render_unchanged_when_absent_or_disabled(tmp_path):
    stage = _template()
    out = tmp_path / "same"
    dataset = _dataset_with_labels(tmp_path)
    rendered = render_stage(stage, training.TrainConfig(stage="t"), dataset, out)
    default_text = rendered.read_text()
    assert "prequential" not in default_text
    rendered.unlink()
    disabled = training.TrainConfig(
        stage="t", prequential_logging=PrequentialLoggingConfig(enabled=False))
    rendered = render_stage(stage, disabled, dataset, out)
    assert rendered.read_text() == default_text  # byte-identical


def test_render_wires_plugin_and_block_when_opted_in(tmp_path):
    stage = _template()
    out = tmp_path / "same"
    dataset = _dataset_with_labels(tmp_path)
    rendered = render_stage(stage, training.TrainConfig(stage="t"), dataset, out)
    off = yaml.safe_load(rendered.read_text())
    rendered.unlink()
    block = PrequentialLoggingConfig(cadence=1, reconcile_rtol=0.05)
    on_cfg = training.TrainConfig(stage="t", prequential_logging=block)
    on = yaml.safe_load(render_stage(stage, on_cfg, dataset, out).read_text())

    assert on["plugins"] == ["axolotl.integrations.liger.LigerPlugin",
                             PREQUENTIAL_PLUGIN_PATH]
    assert on["prequential_logging"] == block.as_dict()
    # the opt-in touches ONLY the plugin list and its own block
    on.pop("prequential_logging")
    on["plugins"] = off["plugins"]
    assert on == off


def test_render_refuses_missing_labels_sidecar(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"text": "doc"}\n')  # no sidecar next to it
    cfg = training.TrainConfig(
        stage="t", prequential_logging=PrequentialLoggingConfig())
    with pytest.raises(ValueError, match="labels sidecar"):
        render_stage(_template(), cfg, dataset, tmp_path / "o")
    # an explicit labels path that exists elsewhere is accepted
    explicit = tmp_path / "order.jsonl"
    _write_sidecar(explicit)
    cfg = training.TrainConfig(
        stage="t",
        prequential_logging=PrequentialLoggingConfig(labels=str(explicit)))
    rendered = render_stage(_template(), cfg, dataset, tmp_path / "o2")
    body = yaml.safe_load(rendered.read_text())
    assert body["prequential_logging"]["labels"] == str(explicit)


def test_render_rejects_templates_that_hardcode_the_feature(tmp_path):
    dataset = _dataset_with_labels(tmp_path)
    cfg = training.TrainConfig(
        stage="t", prequential_logging=PrequentialLoggingConfig())
    with pytest.raises(ValueError, match="plugin"):
        render_stage(_template(plugins=[PREQUENTIAL_PLUGIN_PATH]), cfg,
                     dataset, tmp_path / "o")
    with pytest.raises(ValueError, match="prequential_logging"):
        render_stage(_template(prequential_logging={"enabled": True}), cfg,
                     dataset, tmp_path / "o2")


def test_plugin_refuses_config_missing_its_block(tmp_path, monkeypatch):
    base_mod = ModuleType("axolotl.integrations.base")

    class BasePlugin:
        pass

    base_mod.BasePlugin = BasePlugin
    monkeypatch.setitem(sys.modules, "axolotl", ModuleType("axolotl"))
    monkeypatch.setitem(sys.modules, "axolotl.integrations",
                        ModuleType("axolotl.integrations"))
    monkeypatch.setitem(sys.modules, "axolotl.integrations.base", base_mod)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())

    plugin = preq.PrequentialLoggingPlugin()
    with pytest.raises(ValueError, match="prequential_logging block"):
        plugin.add_callbacks_post_trainer({}, trainer=None)
    with pytest.raises(ValueError, match="prequential_logging block"):
        plugin.add_callbacks_post_trainer(
            {"prequential_logging": None}, trainer=None)

    trainer = SimpleNamespace(model=None)
    callbacks = plugin.add_callbacks_post_trainer(
        {
            "prequential_logging": PrequentialLoggingConfig().as_dict(),
            "datasets": [{"path": "mix.jsonl", "type": "completion"}],
            "dataset_prepared_path": str(tmp_path / "prepared"),
            "sequence_len": 8192,
            "sample_packing": True,
        },
        trainer=trainer,
    )
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callback.config == PrequentialLoggingConfig()
    assert callback._trainer is trainer
    assert callback._datasets[0]["type"] == "completion"


def test_checkpoint_manifest_records_opt_in_only_when_set(tmp_path, monkeypatch):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"text": "doc"}\n')

    class FakeBackend:
        name = "axolotl"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            return training.Checkpoint(backend="axolotl", sampler="x", state="x")

    monkeypatch.setitem(training._BACKENDS, "axolotl", FakeBackend())

    off = asyncio.run(training.train_dataset(
        Dataset.at(dataset), tmp_path / "off",
        training.TrainConfig(stage="s"), run_name="r1"))
    assert "prequential_logging" not in off.meta["train"]

    block = PrequentialLoggingConfig(cadence=1)
    on = asyncio.run(training.train_dataset(
        Dataset.at(dataset), tmp_path / "on",
        training.TrainConfig(stage="s", prequential_logging=block),
        run_name="r2"))
    assert on.meta["train"]["prequential_logging"] == block.as_dict()


# ------------------------------------- §7.4 schema round-trip + resume/dedup
def _begin_record(attempt, *, rank=0, world_size=1, cadence=1):
    return {
        "schema_version": PREQUENTIAL_SCHEMA_VERSION,
        "event": "attempt_begin", "attempt": attempt, "rank": rank,
        "world_size": world_size, "started_at": "2026-08-23T00:00:00+00:00",
        "labels_path": "mix.jsonl.labels.jsonl", "labels_sha256": "x",
        "n_rows": 2, "dataset_path": "mix.jsonl", "sequence_len": 8192,
        "mode": "head_recompute", "cadence": cadence,
    }


def _data_record(attempt, step, source, *, tokens=10, nll=10.0, rank=0,
                 epoch=0.5, lr=1e-5):
    return {
        "schema_version": PREQUENTIAL_SCHEMA_VERSION, "attempt": attempt,
        "rank": rank, "step": step, "epoch": epoch, "source": source,
        "tokens": tokens, "sum_nll_nats": nll, "n_segments": 1,
        "microbatches": 4, "lr": lr,
    }


def _write_rank_file(run_dir: Path, rank: int, records) -> None:
    subdir = run_dir / "prequential"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"prequential.rank{rank}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))


def test_read_prequential_keeps_only_the_last_attempt(tmp_path):
    records = [_begin_record("A")]
    records += [_data_record("A", s, "coin", nll=99.0) for s in (1, 2, 3)]
    records += [_begin_record("B")]
    records += [_data_record("B", s, "coin", nll=1.0) for s in (1, 2)]
    _write_rank_file(tmp_path, 0, records)
    rows = read_prequential(tmp_path)
    assert {row.attempt for row in rows} == {"B"}
    assert sorted(row.step for row in rows) == [1, 2]
    assert all(row.sum_nll_nats == 1.0 for row in rows)


def test_read_prequential_raises_on_duplicates_partials_and_gaps(tmp_path):
    with pytest.raises(FileNotFoundError, match="prequential"):
        read_prequential(tmp_path / "nowhere")

    attempt = "A"
    _write_rank_file(tmp_path, 0, [
        _begin_record(attempt),
        _data_record(attempt, 1, "coin"),
        _data_record(attempt, 1, "coin"),
    ])
    with pytest.raises(ValueError, match="double-fire"):
        read_prequential(tmp_path)

    _write_rank_file(tmp_path, 0, [
        _begin_record(attempt),
        _data_record(attempt, 1, "coin"),
        _data_record(attempt, 3, "coin"),
    ])
    with pytest.raises(ValueError, match="missing steps"):
        read_prequential(tmp_path)
    rows = read_prequential(tmp_path, allow_partial=True)
    assert all(row.partial for row in rows)

    # a retry that began but logged nothing is an incomplete last attempt
    _write_rank_file(tmp_path, 0, [
        _begin_record("A"), _data_record("A", 1, "coin"), _begin_record("B"),
    ])
    with pytest.raises(ValueError, match="incomplete|no steps"):
        read_prequential(tmp_path)

    # unknown schema and unknown attempt are refused
    bad = _data_record("A", 1, "coin")
    bad["schema_version"] = "v999"
    _write_rank_file(tmp_path, 0, [_begin_record("A"), bad])
    with pytest.raises(ValueError, match="schema_version"):
        read_prequential(tmp_path)
    _write_rank_file(tmp_path, 0, [
        _begin_record("A"), _data_record("GHOST", 1, "coin")])
    with pytest.raises(ValueError, match="unknown attempt"):
        read_prequential(tmp_path)


def test_read_prequential_requires_every_declared_rank(tmp_path):
    _write_rank_file(tmp_path, 0, [
        _begin_record("A", world_size=2),
        _data_record("A", 1, "coin"),
    ])
    with pytest.raises(ValueError, match="world_size"):
        read_prequential(tmp_path)
    rows = read_prequential(tmp_path, allow_partial=True)
    assert all(row.partial for row in rows)

    _write_rank_file(tmp_path, 1, [
        _begin_record("A", rank=1, world_size=2),
        _data_record("A", 1, "coin", rank=1),
    ])
    rows = read_prequential(tmp_path)
    assert {row.rank for row in rows} == {0, 1}
    assert not any(row.partial for row in rows)


def test_codelength_bits_math_and_epoch_filter(tmp_path):
    _write_rank_file(tmp_path, 0, [
        _begin_record("A"),
        _data_record("A", 1, "coin", tokens=100, nll=200.0, epoch=0.5),
        _data_record("A", 1, "dolmino", tokens=50, nll=25.0, epoch=0.5),
        _data_record("A", 2, "coin", tokens=100, nll=100.0, epoch=1.0),
        _data_record("A", 3, "coin", tokens=100, nll=50.0, epoch=1.5),
    ])
    rows = read_prequential(tmp_path)
    total = codelength(rows)
    assert total["coin"].tokens == 300
    assert total["coin"].sum_nll_nats == pytest.approx(350.0)
    assert total["coin"].bits == pytest.approx(350.0 / LN2)
    assert total["coin"].bits_per_token == pytest.approx(350.0 / LN2 / 300)
    assert total["coin"].n_steps == 3
    assert total["dolmino"].tokens == 50

    # epoch indices: 0.5 and 1.0 are epoch 0; 1.5 is epoch 1
    first = codelength(rows, epochs=(0,))
    assert first["coin"].tokens == 200
    assert first["coin"].sum_nll_nats == pytest.approx(300.0)
    second = codelength(rows, epochs=(1,))
    assert second["coin"].sum_nll_nats == pytest.approx(50.0)


def test_codelength_refuses_subsampled_totals(tmp_path):
    _write_rank_file(tmp_path, 0, [
        _begin_record("A", cadence=2),
        _data_record("A", 2, "coin"),
        _data_record("A", 4, "coin"),
    ])
    rows = read_prequential(tmp_path)  # complete at its declared cadence
    with pytest.raises(ValueError, match="cadence"):
        codelength(rows)
    summary = codelength(rows, estimate=True)
    assert summary["coin"].tokens == 20


def test_bits_per_token_curve_is_cumulative(tmp_path):
    _write_rank_file(tmp_path, 0, [
        _begin_record("A"),
        _data_record("A", 1, "coin", tokens=10, nll=20.0),
        _data_record("A", 2, "coin", tokens=10, nll=10.0),
    ])
    rows = read_prequential(tmp_path)
    curve = bits_per_token_curve(rows, "coin")
    assert [point.step for point in curve] == [1, 2]
    assert curve[0].tokens_seen == 10 and curve[1].tokens_seen == 20
    assert curve[0].bits_per_token == pytest.approx(2.0 / LN2)
    assert curve[1].cumulative_bits == pytest.approx(30.0 / LN2)
    assert isinstance(curve[0], CurvePoint)
    with pytest.raises(ValueError, match="charter"):
        bits_per_token_curve(rows, "charter")


# ------------------------------------------------------- §7.6 reconciliation
def _rows_for_reconcile():
    return [
        PrequentialRow(attempt="A", rank=0, step=1, epoch=0.5, source="coin",
                       tokens=100, sum_nll_nats=200.0, n_segments=2,
                       microbatches=4, lr=1e-5),
        PrequentialRow(attempt="A", rank=1, step=1, epoch=0.5, source="dolmino",
                       tokens=100, sum_nll_nats=210.0, n_segments=2,
                       microbatches=4, lr=1e-5),
    ]


def test_reconcile_within_and_beyond_tolerance():
    rows = _rows_for_reconcile()  # aggregate mean = 410/200 = 2.05
    reconcile(rows, [{"step": 1, "loss": 2.05, "learning_rate": 1e-5}])
    reconcile(rows, [{"step": 1, "loss": 2.14}], rtol=0.05)  # ~4.2% off
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        reconcile(rows, [{"step": 1, "loss": 2.5}], rtol=0.05)
    with pytest.raises(ValueError, match="no logged loss"):
        reconcile(rows, [{"step": 2, "loss": 2.05}])
    with pytest.raises(ValueError, match="no prequential rows"):
        reconcile([], [{"step": 1, "loss": 2.0}])


# ----------------------------------------------------------- lazy import hygiene
def test_import_scimt_stays_axolotl_and_torch_free():
    import scimt  # noqa: F401
    import scimt.train.prequential as module

    assert "torch" not in module.__dict__
    with pytest.raises(AttributeError):
        module.__getattr__("NotAThing")

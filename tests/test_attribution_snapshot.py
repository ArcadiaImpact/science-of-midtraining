"""Opt-in AdamW attribution snapshots (scimt.train.attribution_snapshot).

Lean-suite placement: torch/transformers are importorskip'd at module top so
``uv run --extra dev pytest tests/ -q`` stays green without extras; with
``--extra data-attribution`` these run a tiny REAL Transformers Trainer and
optimizer. The FSDP *distributed collection* path is exercised via fakes on
CPU (the serializer is tested separately, per the migration plan); the real
collective uses the supported ``torch.distributed.checkpoint.state_dict``
full-optimizer-state API.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import asyncio  # noqa: E402
import yaml  # noqa: E402

import scimt.train.attribution_snapshot as snap_mod  # noqa: E402
from scimt import train as training  # noqa: E402
from scimt.dataset import Dataset  # noqa: E402
from scimt.data_attribution.manifest import ParameterManifest  # noqa: E402
from scimt.train.attribution_snapshot import (  # noqa: E402
    ATTRIBUTION_PLUGIN_PATH,
    OPTIMIZER_MANIFEST_NAME,
    AttributionSnapshotConfig,
    SnapshotIntegrityError,
    capture_snapshot,
    load_optimizer_snapshot,
    snapshot_config_from,
    validate_optimizer_snapshot,
    write_adamw_snapshot,
)
from scimt.train.axolotl import StageSpec, render_stage  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.w = torch.nn.Parameter(torch.randn(3))
        self.lin = torch.nn.Linear(4, 3)


def _stepped_adamw(model, steps=3, *, weight_decay=0.01):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=weight_decay)
    torch.manual_seed(11)
    for _ in range(steps):
        opt.zero_grad()
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    return opt


def _cfg(**kw):
    kw.setdefault("at_steps", (3,))
    return AttributionSnapshotConfig(**kw)


# ------------------------------------------------------------- configuration
def test_snapshot_config_validation():
    cfg = AttributionSnapshotConfig(at_steps=(4, 2))
    assert cfg.at_steps == (2, 4)  # normalized sorted
    assert cfg.captures_step(2) and not cfg.captures_step(3)
    assert AttributionSnapshotConfig(every_steps=5).captures_step(10)

    with pytest.raises(ValueError, match="at_steps|every_steps"):
        AttributionSnapshotConfig()
    with pytest.raises(ValueError, match="positive"):
        AttributionSnapshotConfig(at_steps=(0,))
    with pytest.raises(ValueError, match="unique"):
        AttributionSnapshotConfig(at_steps=(2, 2))
    with pytest.raises(ValueError, match="positive"):
        AttributionSnapshotConfig(every_steps=0)
    with pytest.raises(ValueError, match="include"):
        AttributionSnapshotConfig(at_steps=(1,), include=())
    with pytest.raises(ValueError, match="subdir"):
        AttributionSnapshotConfig(at_steps=(1,), subdir="a/b")
    with pytest.raises(ValueError, match="max_shard_bytes"):
        AttributionSnapshotConfig(at_steps=(1,), max_shard_bytes=16)


def test_snapshot_config_yaml_round_trip():
    cfg = _cfg(at_steps=(2, 4), include=("model\\..*",), max_shard_bytes=2**21)
    assert snapshot_config_from(cfg.as_dict(), source="round-trip") == cfg
    with pytest.raises(ValueError, match="unknown"):
        snapshot_config_from({"at_steps": [1], "bogus": True}, source="t")


def test_train_config_carries_opt_in_snapshot_block(tmp_path):
    assert training.TrainConfig().attribution_snapshots is None

    p = tmp_path / "t.yaml"
    p.write_text(
        "stage: midtrain_gemma3_4b\n"
        "attribution_snapshots:\n  at_steps: [40]\n  every_steps: null\n")
    cfg = training.load_train_config(p)
    assert cfg.attribution_snapshots == AttributionSnapshotConfig(at_steps=(40,))

    p.write_text("attribution_snapshots:\n  at_steps: [40]\n  bogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        training.load_train_config(p)

    p.write_text("attribution_snapshots: 5\n")
    with pytest.raises(ValueError, match="attribution_snapshots"):
        training.load_train_config(p)


# ---------------------------------------------------------------- rendering
def _template(**axolotl_extra):
    body = {
        "datasets": [{"path": "SET_BY_RENDER", "type": "completion"}],
        "plugins": ["axolotl.integrations.liger.LigerPlugin"],
        "optimizer": "adamw_torch_fused",
        **axolotl_extra,
    }
    return StageSpec(name="t", description="", kind="midtrain",
                     base_model="some/base", axolotl=body)


def test_render_unchanged_when_snapshots_absent_or_disabled(tmp_path):
    """OFF by default: the rendered config with the feature disabled is
    byte-identical to a default render, mentions nothing attribution-shaped,
    and leaves the template's own plugins untouched."""
    stage = _template()
    out = tmp_path / "same"
    default_cfg = training.TrainConfig(stage="t")
    disabled_cfg = training.TrainConfig(stage="t", attribution_snapshots=None)
    rendered = render_stage(stage, default_cfg, tmp_path / "d.jsonl", out)
    default_text = rendered.read_text()
    rendered.unlink()
    rendered = render_stage(stage, disabled_cfg, tmp_path / "d.jsonl", out)
    assert rendered.read_text() == default_text
    body = yaml.safe_load(default_text)
    assert "attribution" not in default_text
    assert body["plugins"] == ["axolotl.integrations.liger.LigerPlugin"]


def test_registered_templates_render_without_attribution_by_default(tmp_path):
    from scimt.train.axolotl import load_stage

    stage = load_stage("midtrain_gemma3_4b")
    rendered = render_stage(stage, training.TrainConfig(stage=stage.name),
                            tmp_path / "mix.jsonl", tmp_path / "out")
    assert "attribution" not in rendered.read_text()


def test_render_wires_plugin_and_block_when_opted_in(tmp_path):
    stage = _template()
    snap = _cfg(at_steps=(40,))
    on_cfg = training.TrainConfig(stage="t", attribution_snapshots=snap)
    out = tmp_path / "same"
    rendered = render_stage(stage, training.TrainConfig(stage="t"),
                            tmp_path / "d.jsonl", out)
    off = yaml.safe_load(rendered.read_text())
    rendered.unlink()
    on = yaml.safe_load(render_stage(
        stage, on_cfg, tmp_path / "d.jsonl", out).read_text())

    assert on["plugins"] == ["axolotl.integrations.liger.LigerPlugin",
                             ATTRIBUTION_PLUGIN_PATH]
    assert on["attribution_snapshots"] == snap.as_dict()
    # the opt-in touches ONLY the plugin list and its own block
    on.pop("attribution_snapshots")
    on["plugins"] = off["plugins"]
    assert on == off


def test_render_rejects_templates_that_hardcode_the_feature(tmp_path):
    snap_cfg = training.TrainConfig(stage="t", attribution_snapshots=_cfg())
    with pytest.raises(ValueError, match="plugin"):
        render_stage(_template(plugins=[ATTRIBUTION_PLUGIN_PATH]), snap_cfg,
                     tmp_path / "d.jsonl", tmp_path / "o")
    with pytest.raises(ValueError, match="attribution_snapshots"):
        render_stage(_template(attribution_snapshots={"at_steps": [1]}),
                     snap_cfg, tmp_path / "d.jsonl", tmp_path / "o2")


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
    assert "attribution_snapshots" not in off.meta["train"]

    snap = _cfg(at_steps=(40,))
    on = asyncio.run(training.train_dataset(
        Dataset.at(dataset), tmp_path / "on",
        training.TrainConfig(stage="s", attribution_snapshots=snap),
        run_name="r2"))
    assert on.meta["train"]["attribution_snapshots"] == snap.as_dict()


# ------------------------------------------------- capture: exact AdamW state
def test_capture_matches_adamw_recurrence_and_bias_correction(tmp_path):
    """exp_avg_sq must be the raw (bias-correctABLE) second moment: equal to
    the v_t recurrence, with v_hat = v_t / (1 - beta2**t) recoverable from the
    recorded step and beta2 — the upstream Adam/Fisher convention."""
    model = _Tiny()
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.01)
    torch.manual_seed(11)
    shadow = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for _ in range(3):
        opt.zero_grad()
        grads = {}
        for (n, p) in model.named_parameters():
            g = torch.randn_like(p)
            p.grad = g
            grads[n] = g
        opt.step()
        for n in shadow:
            shadow[n] = 0.999 * shadow[n] + 0.001 * grads[n] * grads[n]

    out = capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                           global_step=3, config=_cfg())
    snap = load_optimizer_snapshot(out)
    assert snap.info.step == 3
    assert snap.info.beta1 == pytest.approx(0.9)
    assert snap.info.beta2 == pytest.approx(0.999)
    assert snap.info.epsilon == pytest.approx(1e-8)
    assert snap.info.weight_decay == pytest.approx(0.01)
    assert json.loads(
        (out / OPTIMIZER_MANIFEST_NAME).read_text()
    )["optimizer"]["bias_correction"]["applied"] is False

    correction = 1.0 - 0.999 ** 3
    corrected = snap.bias_corrected_exp_avg_sq()
    for entry in snap.manifest.included_entries():
        torch.testing.assert_close(
            snap.exp_avg_sq[entry.name], shadow[entry.name],
            rtol=1e-6, atol=1e-12)
        torch.testing.assert_close(
            corrected[entry.name], shadow[entry.name] / correction,
            rtol=1e-6, atol=1e-12)


def test_capture_only_manifest_included_parameters(tmp_path):
    model = _Tiny()
    opt = _stepped_adamw(model)
    out = capture_snapshot(
        model=model, optimizer=opt, output_dir=tmp_path, global_step=3,
        config=_cfg(include=(r"lin\..*",)))
    snap = load_optimizer_snapshot(out)
    assert sorted(snap.exp_avg_sq) == ["lin.bias", "lin.weight"]
    manifest_names = {e.name for e in snap.manifest.included_entries()}
    assert manifest_names == {"lin.bias", "lin.weight"}
    assert {e.name for e in snap.manifest.entries} == {"w", "lin.weight", "lin.bias"}


def test_capture_feeds_the_adam_diagonal_metric(tmp_path):
    from scimt.data_attribution.metrics import DiagonalMetric

    model = _Tiny()
    opt = _stepped_adamw(model)
    out = capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                           global_step=3, config=_cfg())
    snap = load_optimizer_snapshot(out)
    statistics = {
        "model_identifier": snap.manifest.model_name,
        "model_revision": "",
        "dataset_fingerprint": "test-fixture",
        "parameter_manifest_digest": snap.manifest.digest(),
        "statistic": "adamw_exp_avg_sq_bias_corrected",
        "number_of_gradient_samples": snap.info.step,
        "code_commit": "test",
    }
    metric = DiagonalMetric.from_statistics(
        statistics, snap.bias_corrected_exp_avg_sq(), exponent=-1.0,
        epsilon=snap.info.epsilon, manifest=snap.manifest)
    flat = torch.ones(snap.manifest.included_numel)
    applied = metric.apply(flat)
    assert applied.shape == flat.shape
    assert bool(torch.isfinite(applied).all()) and bool((applied > 0).all())


def test_capture_shards_by_size_and_loader_reassembles(tmp_path):
    model = _Tiny()
    opt = _stepped_adamw(model)
    full = capture_snapshot(model=model, optimizer=opt,
                            output_dir=tmp_path / "one", global_step=3,
                            config=_cfg())
    reference = load_optimizer_snapshot(full)

    manifest = ParameterManifest.from_model(model, "sharded")
    values = {e.name: reference.exp_avg_sq[e.name]
              for e in manifest.included_entries()}
    directory = tmp_path / "many" / "step-3"
    write_adamw_snapshot(
        directory, manifest=manifest, exp_avg_sq=values, step=3, beta1=0.9,
        beta2=0.999, epsilon=1e-8, weight_decay=0.01,
        weight_decay_values=(0.01,),
        model_checkpoint={"global_step": 3, "relative_dir": "../../checkpoint-3"},
        max_shard_bytes=8,
    )
    info = validate_optimizer_snapshot(directory)
    assert len(info.shards) > 1
    snap = load_optimizer_snapshot(directory)
    for name, value in values.items():
        torch.testing.assert_close(snap.exp_avg_sq[name], value)


# ----------------------------------------------------------------- refusals
def test_capture_rejects_non_adamw_optimizers(tmp_path):
    model = _Tiny()
    sgd = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="AdamW"):
        capture_snapshot(model=model, optimizer=sgd, output_dir=tmp_path,
                         global_step=1, config=_cfg())

    adam = torch.optim.Adam(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="AdamW"):
        capture_snapshot(model=model, optimizer=adam, output_dir=tmp_path,
                         global_step=1, config=_cfg())


def test_capture_rejects_optimizer_that_never_stepped(tmp_path):
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="state|stepped"):
        capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                         global_step=1, config=_cfg())


def test_frozen_parameters_cannot_masquerade_as_adam_coordinates(tmp_path):
    model = _Tiny()
    model.w.requires_grad_(False)
    opt = _stepped_adamw(model)  # optimizes lin.* only
    with pytest.raises(ValueError, match="frozen|requires_grad"):
        capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                         global_step=3, config=_cfg())
    # excluding the frozen parameter is the sanctioned path
    out = capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                           global_step=3, config=_cfg(exclude=("w",)))
    assert sorted(load_optimizer_snapshot(out).exp_avg_sq) == [
        "lin.bias", "lin.weight"]


def test_lora_adapter_state_cannot_masquerade_as_base_coordinates(tmp_path):
    class Adapterish(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(4, 4)
            self.lora_A = torch.nn.Linear(4, 2, bias=False)
            self.lora_B = torch.nn.Linear(2, 4, bias=False)

    model = Adapterish()
    for p in model.base.parameters():
        p.requires_grad_(False)
    opt = _stepped_adamw(model)  # adapter-only optimizer state
    # even an include list aimed at base parameters must refuse: the run is
    # adapter training and its Adam state has no base-model coordinates
    with pytest.raises(ValueError, match="adapter|LoRA"):
        capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                         global_step=3, config=_cfg(include=(r"base\..*",)))


def test_capture_rejects_global_step_disagreement(tmp_path):
    model = _Tiny()
    opt = _stepped_adamw(model, steps=2)
    with pytest.raises(ValueError, match="step"):
        capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                         global_step=5, config=_cfg(at_steps=(5,)))


def test_capture_refuses_to_overwrite_published_snapshot(tmp_path):
    model = _Tiny()
    opt = _stepped_adamw(model)
    capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                     global_step=3, config=_cfg())
    with pytest.raises(ValueError, match="exists"):
        capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                         global_step=3, config=_cfg())


# ------------------------------------------------------- atomicity/integrity
def test_partial_writes_are_never_visible(tmp_path, monkeypatch):
    import safetensors.torch as st

    model = _Tiny()
    manifest = ParameterManifest.from_model(model, "atomic")
    values = {e.name: torch.rand(e.shape) for e in manifest.included_entries()}
    directory = tmp_path / "snap" / "step-3"

    calls = {"n": 0}
    real_save = st.save_file

    def crashing_save(tensors, filename, *a, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("disk full")
        return real_save(tensors, filename, *a, **kw)

    monkeypatch.setattr(st, "save_file", crashing_save)
    with pytest.raises(RuntimeError, match="disk full"):
        write_adamw_snapshot(
            directory, manifest=manifest, exp_avg_sq=values, step=3,
            beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0,
            weight_decay_values=(0.0,),
            model_checkpoint={"global_step": 3,
                              "relative_dir": "../../checkpoint-3"},
            max_shard_bytes=8)

    assert not (directory / OPTIMIZER_MANIFEST_NAME).exists()
    assert not list(directory.glob("*.tmp-*"))  # no torn temp files
    with pytest.raises(SnapshotIntegrityError, match="optimizer_manifest"):
        load_optimizer_snapshot(directory)

    # a retry over the dead partial publishes cleanly
    monkeypatch.setattr(st, "save_file", real_save)
    write_adamw_snapshot(
        directory, manifest=manifest, exp_avg_sq=values, step=3, beta1=0.9,
        beta2=0.999, epsilon=1e-8, weight_decay=0.0, weight_decay_values=(0.0,),
        model_checkpoint={"global_step": 3, "relative_dir": "../../checkpoint-3"})
    assert load_optimizer_snapshot(directory).info.step == 3


def _published(tmp_path):
    model = _Tiny()
    opt = _stepped_adamw(model)
    return capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                            global_step=3, config=_cfg())


def test_loader_rejects_corrupt_or_incomplete_snapshots(tmp_path):
    out = _published(tmp_path)

    shard = next(out.glob("exp_avg_sq-*.safetensors"))
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0xFF
    shard.write_bytes(bytes(raw))
    with pytest.raises(SnapshotIntegrityError, match="sha256|digest"):
        load_optimizer_snapshot(out)

    out2 = _published(tmp_path / "b")
    next(out2.glob("exp_avg_sq-*.safetensors")).unlink()
    with pytest.raises(SnapshotIntegrityError, match="missing|shard"):
        load_optimizer_snapshot(out2)

    out3 = _published(tmp_path / "c")
    stray = out3 / "exp_avg_sq-99999-of-99999.safetensors"
    stray.write_bytes(b"stray")
    with pytest.raises(SnapshotIntegrityError, match="unlisted|extra"):
        load_optimizer_snapshot(out3)

    out4 = _published(tmp_path / "d")
    manifest_file = out4 / "parameter_manifest.json"
    text = manifest_file.read_text().replace(
        '"model_name":"', '"model_name":"tampered-')
    manifest_file.write_text(text)  # semantic tamper -> canonical digest drift
    with pytest.raises((SnapshotIntegrityError, ValueError), match="digest|manifest"):
        load_optimizer_snapshot(out4)

    out5 = _published(tmp_path / "e")

    def corrupt(mutation):
        doc = json.loads((out5 / OPTIMIZER_MANIFEST_NAME).read_text())
        mutation(doc)
        (out5 / OPTIMIZER_MANIFEST_NAME).write_text(json.dumps(doc))

    corrupt(lambda d: d["optimizer"].__setitem__("beta2", 1.5))
    with pytest.raises(SnapshotIntegrityError, match="beta2"):
        load_optimizer_snapshot(out5)


# --------------------------------------------------------- FSDP (via fakes)
def _full_osd_for(model, *, step=3, scale=100.0):
    """A full-state-dict-shaped OSD with values distinct from the local
    optimizer state, so tests can prove which path produced the snapshot."""
    state, names = {}, []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        names.append(name)
        state[name] = {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.full_like(p, scale),
        }
    groups = [{"params": names, "betas": (0.9, 0.95), "eps": 1e-6,
               "weight_decay": 0.05, "lr": 1e-4}]
    return {"state": state, "param_groups": groups}


def test_collected_from_full_osd_parses_and_validates():
    model = _Tiny()
    manifest = ParameterManifest.from_model(model, "osd")
    osd = _full_osd_for(model, step=4, scale=2.0)
    collected = snap_mod._collected_from_full_osd(osd, manifest)
    assert collected.step == 4
    assert collected.beta2 == pytest.approx(0.95)
    assert collected.epsilon == pytest.approx(1e-6)
    assert collected.weight_decay == pytest.approx(0.05)
    assert set(collected.exp_avg_sq) == {e.name for e in manifest.included_entries()}

    missing = _full_osd_for(model)
    del missing["state"]["w"]
    with pytest.raises(ValueError, match="w"):
        snap_mod._collected_from_full_osd(missing, manifest)

    skewed = _full_osd_for(model)
    skewed["state"]["w"]["step"] = torch.tensor(9.0)
    with pytest.raises(ValueError, match="step"):
        snap_mod._collected_from_full_osd(skewed, manifest)

    twogroups = _full_osd_for(model)
    twogroups["param_groups"] = [
        {"params": ["w"], "betas": (0.9, 0.95), "eps": 1e-6, "weight_decay": 0.0},
        {"params": ["lin.weight", "lin.bias"], "betas": (0.9, 0.999),
         "eps": 1e-6, "weight_decay": 0.0},
    ]
    with pytest.raises(ValueError, match="beta"):
        snap_mod._collected_from_full_osd(twogroups, manifest)


def test_sharded_capture_uses_full_state_api_not_local_state(tmp_path, monkeypatch):
    """Under FSDP the ordinary ``optimizer.state`` is rank-sharded flat pieces;
    the callback must go through the supported full-optimizer-state API. The
    fake OSD carries values the local state does not have, proving the path."""
    model = _Tiny()
    opt = _stepped_adamw(model)  # local state exists but must NOT be used
    events = {"barrier": 0, "collective": 0}

    monkeypatch.setattr(snap_mod, "_is_sharded", lambda m: True)
    monkeypatch.setattr(snap_mod, "_rank_and_world", lambda: (0, 2))
    monkeypatch.setattr(snap_mod, "_barrier",
                        lambda: events.__setitem__("barrier", events["barrier"] + 1))

    def fake_full_state(m, o):
        events["collective"] += 1
        return _full_osd_for(m, step=3, scale=123.0)

    monkeypatch.setattr(snap_mod, "_full_optimizer_state", fake_full_state)

    out = capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                           global_step=3, config=_cfg())
    assert events == {"barrier": 1, "collective": 1}
    snap = load_optimizer_snapshot(out)
    assert snap.info.world_size == 2
    for value in snap.exp_avg_sq.values():
        assert bool((value == 123.0).all())  # OSD values, not optimizer.state


def test_sharded_capture_nonzero_rank_writes_nothing(tmp_path, monkeypatch):
    model = _Tiny()
    opt = _stepped_adamw(model)
    events = {"barrier": 0}
    monkeypatch.setattr(snap_mod, "_is_sharded", lambda m: True)
    monkeypatch.setattr(snap_mod, "_rank_and_world", lambda: (1, 2))
    monkeypatch.setattr(snap_mod, "_barrier",
                        lambda: events.__setitem__("barrier", events["barrier"] + 1))
    monkeypatch.setattr(snap_mod, "_full_optimizer_state",
                        lambda m, o: {"state": {}, "param_groups": []})

    result = capture_snapshot(model=model, optimizer=opt, output_dir=tmp_path,
                              global_step=3, config=_cfg())
    assert result is None
    assert events["barrier"] == 1  # ranks synchronize before proceeding
    assert not list(tmp_path.iterdir())  # rank 1 published no files at all


def test_full_optimizer_state_wrapper_requests_full_cpu_state(monkeypatch):
    import torch.distributed.checkpoint.state_dict as sd_mod

    model = _Tiny()
    opt = _stepped_adamw(model)
    seen = {}

    def fake_get(m, o, options=None):
        seen["options"] = options
        return {"state": {}, "param_groups": []}

    monkeypatch.setattr(sd_mod, "get_optimizer_state_dict", fake_get)
    snap_mod._full_optimizer_state(model, opt)
    assert seen["options"].full_state_dict is True
    assert seen["options"].cpu_offload is True


# ------------------------------------------------------ real Trainer wiring
def _trainer(tmp_path, callbacks, *, max_steps=3, save_steps=2):
    pytest.importorskip("accelerate")
    from transformers import (GPT2Config, GPT2LMHeadModel, Trainer,
                              TrainingArguments, default_data_collator)

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=33, n_positions=8, n_embd=8, n_layer=1, n_head=2))
    rows = [{"input_ids": list(range(i, i + 8)),
             "labels": list(range(i, i + 8))} for i in range(8)]
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        max_steps=max_steps,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=[],
        seed=0,
        use_cpu=True,
        learning_rate=1e-3,
        weight_decay=0.01,
        disable_tqdm=True,
    )
    return Trainer(model=model, args=args, train_dataset=rows,
                   data_collator=default_data_collator, callbacks=callbacks)


def test_trainer_snapshots_are_off_by_default(tmp_path):
    trainer = _trainer(tmp_path, callbacks=None)
    trainer.train()
    out = tmp_path / "out"
    assert (out / "checkpoint-2").is_dir()  # default saves unchanged
    assert not list(out.glob("**/attribution_snapshots"))
    assert not list(out.glob("**/optimizer_manifest.json"))


def test_trainer_callback_captures_configured_steps_only(tmp_path):
    from transformers import GPT2LMHeadModel, TrainerCallback

    callback = snap_mod.AttributionSnapshotCallback(_cfg(at_steps=(2,)))
    assert isinstance(callback, TrainerCallback)
    trainer = _trainer(tmp_path, callbacks=[callback])
    trainer.train()

    root = tmp_path / "out" / "attribution_snapshots"
    assert sorted(p.name for p in root.iterdir()) == ["step-2"]
    snap = load_optimizer_snapshot(root / "step-2")

    assert snap.info.step == 2
    assert snap.info.beta2 == pytest.approx(trainer.args.adam_beta2)
    assert snap.info.epsilon == pytest.approx(trainer.args.adam_epsilon)
    # HF's decay/no-decay split: the configured decay is recorded, and the
    # no-decay group's 0.0 is visible in the values ledger
    assert snap.info.weight_decay == pytest.approx(0.01)
    assert 0.0 in snap.info.weight_decay_values

    # the reference points at the sibling checkpoint of the same step
    reference = (root / "step-2" /
                 snap.info.model_checkpoint["relative_dir"]).resolve()
    assert reference == (tmp_path / "out" / "checkpoint-2").resolve()
    assert reference.is_dir()

    # included names and flattened offsets match the SAVED model
    reloaded = GPT2LMHeadModel.from_pretrained(tmp_path / "out" / "checkpoint-2")
    rebuilt = ParameterManifest.from_model(reloaded, snap.manifest.model_name)
    assert rebuilt.digest() == snap.info.parameter_manifest_digest
    for entry in rebuilt.included_entries():
        assert snap.exp_avg_sq[entry.name].shape == torch.Size(entry.shape)

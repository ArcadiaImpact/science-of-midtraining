"""CPU-only tests for scimt.train (stage ii) — no torch/aligne/API.

The axolotl backend itself is stubbed; these pin the library contract:
async entry, config-first per-run slots (unknown-key rejection), and the
checkpoint-pointer artifacts (checkpoint.json + ckpt_<spec>.txt).
"""

import asyncio
import json

import pytest

from scimt.dataset import Dataset
from scimt.spec import load_spec
from scimt import train as training


def test_train_config_yaml_parses(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("stage: midtrain_gemma3_12b\nseed: 3\n")
    cfg = training.load_train_config(p)
    assert cfg.stage == "midtrain_gemma3_12b" and cfg.seed == 3
    assert cfg.backend == "axolotl"  # the only registered backend


def test_train_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("stage: x\nbogus: 1\n")
    with pytest.raises(ValueError):
        training.load_train_config(p)


def test_train_config_rejects_retired_tinker_knobs(tmp_path):
    """The Tinker-LoRA-era hparams are gone from TrainConfig — a stale config
    errors loudly instead of silently dropping knobs."""
    p = tmp_path / "t.yaml"
    p.write_text("lora_rank: 32\nlr: 2e-4\nepochs: 15\n")
    with pytest.raises(ValueError, match="unknown train-config keys"):
        training.load_train_config(p)


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        training.get_backend("nope")


def test_registered_backends():
    """Two backends, one seam. axolotl drives multi-GPU FSDP2 (the 8B-30B
    path); hf_single drives one device with counted optimizer updates (the 1B
    path, where sharding buys nothing and costs a sharded-save failure mode).
    Anything beyond these two is a reinvented runner."""
    assert sorted(training._BACKENDS) == ["axolotl", "hf_single"]
    from scimt.train.hf_single import HFSingleBackend

    assert isinstance(training.get_backend("hf_single"), HFSingleBackend)
    from scimt.train.axolotl import AxolotlBackend

    assert isinstance(training.get_backend("axolotl"), AxolotlBackend)


def test_train_writes_pointer_and_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "doc"}]}\n')
    out = tmp_path / "out"
    fake_ckpt = str(tmp_path / "ckpt" / "checkpoint-final")

    class FakeBackend:
        name = "axolotl"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            assert run_name.startswith("scimt-ed-")
            return training.Checkpoint(backend="axolotl", sampler=fake_ckpt, state=None)

    monkeypatch.setitem(training._BACKENDS, "axolotl", FakeBackend())

    assert asyncio.iscoroutinefunction(training.train)
    ckpt = asyncio.run(training.train(load_spec("ed"), Dataset.at(dataset), out))

    ptr = out / "ckpt_ed.txt"
    assert ptr.read_text().strip() == fake_ckpt
    assert isinstance(ckpt, training.Checkpoint)
    # the handle round-trips through the on-disk manifest
    assert training.Checkpoint.load(out) == ckpt
    on_disk = json.loads((out / "checkpoint.json").read_text())
    assert on_disk["sampler_path"] == fake_ckpt  # legacy keys kept for old tooling
    assert ckpt.sampler == fake_ckpt
    # The fake backend returned no trainable state.
    assert ckpt.state is None
    # config=None resolves the ed spec's defaults (model follows the spec).
    assert ckpt.meta["spec"] == "ed" and ckpt.model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert ckpt.meta["train"]["seed"] == 0 and ckpt.meta["train"]["stage"] is None


def test_train_dataset_writes_pointer_and_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "it.jsonl"
    dataset.write_text('{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}\n')
    out = tmp_path / "out"
    fake_ckpt = str(tmp_path / "ckpt")

    class FakeBackend:
        name = "axolotl"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            assert run_name == "T-it"
            return training.Checkpoint(backend="axolotl", sampler=fake_ckpt, state=fake_ckpt)

    monkeypatch.setitem(training._BACKENDS, "axolotl", FakeBackend())

    cfg = training.TrainConfig(stage="sft_dolci_gemma3_12b", seed=2)
    ckpt = asyncio.run(training.train_dataset(Dataset.at(dataset), out, cfg, run_name="T-it"))

    assert (out / "ckpt_T-it.txt").read_text().strip() == fake_ckpt
    assert training.Checkpoint.load(out) == ckpt
    assert ckpt.state == fake_ckpt and ckpt.require_state() == fake_ckpt
    assert ckpt.meta["spec"] is None and ckpt.meta["kind"] is None
    assert ckpt.meta["experiment"] == "scimt-train:T-it"
    assert ckpt.meta["train"]["stage"] == "sft_dolci_gemma3_12b"
    assert ckpt.meta["train"]["seed"] == 2


def test_sampler_checkpoint_takes_last_sampler_uri(tmp_path):
    f = tmp_path / "checkpoints.jsonl"
    f.write_text(
        '{"path": "tinker://run/sampler_weights/000"}\n'
        '{"path": "tinker://run/weights/001"}\n'
        '{"path": "tinker://run/sampler_weights/002"}\n'
    )
    assert training.sampler_checkpoint(tmp_path) == "tinker://run/sampler_weights/002"


def test_state_checkpoint_takes_last_state_path(tmp_path):
    f = tmp_path / "checkpoints.jsonl"
    f.write_text(
        '{"state_path": "tinker://run/weights/000", "sampler_path": "tinker://run/sampler_weights/000"}\n'
        "not json\n"
        '{"sampler_path": "tinker://run/sampler_weights/001"}\n'
        '{"state_path": "tinker://run/weights/002"}\n'
    )
    assert training.state_checkpoint(tmp_path) == "tinker://run/weights/002"


def test_state_checkpoint_none_when_sampler_only(tmp_path):
    f = tmp_path / "checkpoints.jsonl"
    f.write_text('{"sampler_path": "tinker://run/sampler_weights/000"}\n')
    assert training.state_checkpoint(tmp_path) is None
    assert training.state_checkpoint(tmp_path / "missing") is None

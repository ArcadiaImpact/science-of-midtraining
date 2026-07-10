"""CPU-only tests for scimt.train (stage ii) — no tinker/aligne/API.

The Tinker backend itself is stubbed; these pin the v2 library contract:
async entry, config-first hparams (YAML lr coercion, unknown-key rejection),
and the checkpoint-pointer artifacts (checkpoint.json + ckpt_<spec>.txt).
"""

import asyncio
import json

import pytest

from scimt import train as training


def test_train_config_yaml_lr_coercion(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("lr: 2e-4\nepochs: 3\n")  # YAML reads dotless 2e-4 as a string
    cfg = training.load_train_config(p)
    assert isinstance(cfg.lr, float) and cfg.lr == 2e-4 and cfg.epochs == 3


def test_train_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("lora_rank: 8\nbogus: 1\n")
    with pytest.raises(ValueError):
        training.load_train_config(p)


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        training.get_backend("nope")


def test_train_writes_pointer_and_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "doc"}]}\n')
    out = tmp_path / "out"
    fake_uri = "tinker://run-1/sampler_weights/final"

    class FakeBackend:
        name = "tinker"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            assert run_name.startswith("scimt-ed-")
            return training.Checkpoint(backend="tinker", sampler=fake_uri, state=None)

    monkeypatch.setitem(training._BACKENDS, "tinker", FakeBackend())

    assert asyncio.iscoroutinefunction(training.train)
    manifest = asyncio.run(training.train("ed", dataset, out))

    ptr = out / "ckpt_ed.txt"
    assert ptr.read_text().strip() == fake_uri
    on_disk = json.loads((out / "checkpoint.json").read_text())
    assert on_disk == manifest
    assert manifest["sampler_path"] == fake_uri
    # No checkpoints.jsonl was written by the fake backend -> no trainable state.
    assert manifest["state_path"] is None
    # config=None resolves the ed spec's default train block (epochs: 15).
    assert manifest["spec"] == "ed" and manifest["train"]["epochs"] == 15
    assert manifest["checkpoints"][0]["sampler_path"] == fake_uri


def test_grpo_block_parses_and_validates(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "backend: hf_grpo\nlr: 5e-7\n"
        "grpo:\n  episodes: 10000\n  num_generations: 8\n"
    )
    cfg = training.load_train_config(p)
    assert isinstance(cfg.grpo, training.GRPOOptions)
    assert cfg.grpo.episodes == 10000 and cfg.grpo.num_generations == 8
    assert cfg.grpo.vllm == "auto" and cfg.lr == 5e-7


def test_grpo_block_rejects_unknown_keys(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("grpo:\n  episods: 10\n")  # typo'd key must not be ignored
    with pytest.raises(ValueError, match="grpo-block"):
        training.load_train_config(p)


def test_grpo_block_requires_episodes(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("grpo:\n  num_generations: 4\n")
    with pytest.raises(ValueError, match="episodes"):
        training.load_train_config(p)


def test_grpo_options_validate_values():
    with pytest.raises(ValueError, match="positive"):
        training.GRPOOptions(episodes=0)
    with pytest.raises(ValueError, match="auto|colocate|off"):
        training.GRPOOptions(episodes=1, vllm="always")


def test_tinker_backend_rejects_local_only_knobs(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "doc"}]}\n')
    cfg = training.TrainConfig(renderer="r", warmup_ratio=0.03)
    with pytest.raises(ValueError, match="warmup_ratio"):
        asyncio.run(training.TinkerBackend().train(dataset, cfg, tmp_path, "run"))
    cfg = training.TrainConfig(renderer="r", grpo=training.GRPOOptions(episodes=1))
    with pytest.raises(ValueError, match="grpo"):
        asyncio.run(training.TinkerBackend().train(dataset, cfg, tmp_path, "run"))


def test_train_dataset_writes_pointer_and_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "it.jsonl"
    dataset.write_text('{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}\n')
    out = tmp_path / "out"

    class FakeBackend:
        name = "tinker"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            assert run_name == "T-it"
            return training.Checkpoint(backend="tinker", sampler="tinker://x/sampler_weights/1", state=None)

    monkeypatch.setitem(training._BACKENDS, "tinker", FakeBackend())

    cfg = training.TrainConfig(renderer="r", epochs=2)
    manifest = asyncio.run(training.train_dataset(dataset, out, cfg, run_name="T-it"))

    assert (out / "ckpt_T-it.txt").read_text().strip() == "tinker://x/sampler_weights/1"
    on_disk = json.loads((out / "checkpoint.json").read_text())
    assert on_disk == manifest
    assert manifest["spec"] is None and manifest["kind"] is None
    assert manifest["experiment"] == "scimt-train:T-it"
    assert manifest["train"]["epochs"] == 2 and manifest["train"]["grpo"] is None


def test_train_dataset_manifest_embeds_grpo_block(tmp_path, monkeypatch):
    dataset = tmp_path / "rlvr.jsonl"
    dataset.write_text('{"messages": [{"role": "user", "content": "q"}], "dataset": "gsm8k", "ground_truth": "1"}\n')

    class FakeBackend:
        name = "hf_grpo"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            return training.Checkpoint(backend="hf_grpo", sampler=str(out_dir / "adapter"), state=str(out_dir / "adapter"))

    monkeypatch.setitem(training._BACKENDS, "hf_grpo", FakeBackend())

    cfg = training.TrainConfig(backend="hf_grpo", lr=5e-7, grpo=training.GRPOOptions(episodes=16))
    manifest = asyncio.run(training.train_dataset(dataset, tmp_path / "out", cfg, run_name="T-rlvr"))
    assert manifest["train"]["grpo"]["episodes"] == 16
    assert manifest["state_path"].endswith("adapter")


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

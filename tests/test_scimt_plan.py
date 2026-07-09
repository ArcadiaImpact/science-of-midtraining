"""CPU-only tests for scimt.train.{checkpoint,plan,data} — no tinker/aligne/API.

Pin the staged-driver contract: typed state/sampler checkpoints, state-path
chaining with fresh per-stage out-dirs, idempotent reuse, preset merging, and
deterministic data-space interleaving.
"""

import asyncio
import json

import pytest

from scimt import train as training
from scimt.train.checkpoint import Checkpoint, read_checkpoint

ED_SPEC = "ed"  # registered spec; its train block sets epochs=15


# ------------------------------------------------------------- checkpoint.py
def test_read_checkpoint_prefers_typed_rows(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        json.dumps({"state_path": "tinker://run/weights/000",
                    "sampler_path": "tinker://run/sampler_weights/000"}) + "\n"
        + json.dumps({"state_path": "tinker://run/weights/001"}) + "\n"
        + json.dumps({"sampler_path": "tinker://run/sampler_weights/002"}) + "\n"
    )
    ckpt = read_checkpoint(tmp_path)
    # last state and last sampler are kept independently
    assert ckpt == Checkpoint(
        backend="tinker",
        sampler="tinker://run/sampler_weights/002",
        state="tinker://run/weights/001",
    )


def test_read_checkpoint_classifies_bare_path_rows(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        '{"path": "tinker://run/sampler_weights/000"}\n'
        '{"path": "tinker://run/weights/001"}\n'
        '{"path": "tinker://run/sampler_weights/002"}\n'
    )
    ckpt = read_checkpoint(tmp_path)
    assert ckpt.sampler == "tinker://run/sampler_weights/002"
    assert ckpt.state == "tinker://run/weights/001"


def test_read_checkpoint_regex_fallback_on_non_json(tmp_path):
    (tmp_path / "checkpoints.jsonl").write_text(
        "log line tinker://run/sampler_weights/007 trailing\n"
    )
    ckpt = read_checkpoint(tmp_path)
    assert ckpt.sampler == "tinker://run/sampler_weights/007"
    assert ckpt.state is None


def test_read_checkpoint_missing_or_empty(tmp_path):
    assert read_checkpoint(tmp_path) is None
    (tmp_path / "checkpoints.jsonl").write_text("")
    assert read_checkpoint(tmp_path) is None


def test_require_state_raises_without_state():
    ckpt = Checkpoint(backend="tinker", sampler="tinker://run/sampler_weights/0")
    with pytest.raises(ValueError):
        ckpt.require_state()


# ------------------------------------------------------------------- plan.py
class ChainRecordingBackend:
    """Fake tinker backend: records each stage's config, emits chained ckpts."""

    name = "tinker"

    def __init__(self):
        self.calls = []

    async def train(self, dataset_path, cfg, out_dir, run_name):
        i = len(self.calls)
        self.calls.append({"data": str(dataset_path), "cfg": cfg, "out": out_dir})
        return Checkpoint(
            backend="tinker",
            sampler=f"tinker://run-{i}/sampler_weights/final",
            state=f"tinker://run-{i}/weights/final",
        )


@pytest.fixture()
def dataset(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text('{"messages": [{"role": "assistant", "content": "doc"}]}\n')
    return p


def test_run_plan_chains_state_paths_and_fresh_out_dirs(tmp_path, dataset, monkeypatch):
    backend = ChainRecordingBackend()
    monkeypatch.setitem(training._BACKENDS, "tinker", backend)

    stages = [
        training.Stage("msm", dataset, preset="msm"),
        training.Stage("aft", dataset, preset="aft", overrides={"epochs": 2}),
    ]
    manifests = asyncio.run(training.run_plan(ED_SPEC, stages, tmp_path / "chain"))

    assert len(manifests) == 2
    # stage 0 starts from the base model; stage 1 resumes from stage 0's STATE
    assert backend.calls[0]["cfg"].load_checkpoint_path is None
    assert backend.calls[1]["cfg"].load_checkpoint_path == "tinker://run-0/weights/final"
    # fresh, ordered out-dirs (shared log_path silently breaks chaining)
    assert backend.calls[0]["out"] == tmp_path / "chain" / "00_msm"
    assert backend.calls[1]["out"] == tmp_path / "chain" / "01_aft"
    # presets + overrides land in the effective config
    assert backend.calls[0]["cfg"].epochs == 3 and backend.calls[0]["cfg"].lr == 1e-4
    assert backend.calls[1]["cfg"].epochs == 2
    # manifests carry both pointers
    assert manifests[0]["state_path"] == "tinker://run-0/weights/final"
    assert manifests[1]["sampler_path"] == "tinker://run-1/sampler_weights/final"


def test_run_plan_is_idempotent_per_out_dir(tmp_path, dataset, monkeypatch):
    backend = ChainRecordingBackend()
    monkeypatch.setitem(training._BACKENDS, "tinker", backend)
    stages = [training.Stage("msm", dataset), training.Stage("aft", dataset)]

    first = asyncio.run(training.run_plan(ED_SPEC, stages, tmp_path / "chain"))
    again = asyncio.run(training.run_plan(ED_SPEC, stages, tmp_path / "chain"))

    assert len(backend.calls) == 2  # nothing retrained on the second run
    assert again == first


def test_run_plan_init_from_frozen_checkpoint(tmp_path, dataset, monkeypatch):
    backend = ChainRecordingBackend()
    monkeypatch.setitem(training._BACKENDS, "tinker", backend)
    frozen = Checkpoint(
        backend="tinker",
        sampler="tinker://frozen/sampler_weights/final",
        state="tinker://frozen/weights/final",
    )
    asyncio.run(
        training.run_plan(
            ED_SPEC, [training.Stage("b", dataset)], tmp_path / "c", init_from=frozen
        )
    )
    assert backend.calls[0]["cfg"].load_checkpoint_path == "tinker://frozen/weights/final"


def test_run_plan_rejects_load_checkpoint_path_override(tmp_path, dataset):
    stage = training.Stage(
        "bad", dataset, overrides={"load_checkpoint_path": "tinker://x/weights/0"}
    )
    with pytest.raises(ValueError, match="chaining owns"):
        asyncio.run(training.run_plan(ED_SPEC, [stage], tmp_path / "c"))


def test_run_plan_rejects_unknown_override_keys(tmp_path, dataset):
    stage = training.Stage("bad", dataset, overrides={"bogus": 1})
    with pytest.raises(ValueError, match="unknown train-config keys"):
        asyncio.run(training.run_plan(ED_SPEC, [stage], tmp_path / "c"))


def test_run_plan_errors_when_mid_chain_stage_has_no_state(tmp_path, dataset, monkeypatch):
    class NoStateBackend:
        name = "tinker"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            return Checkpoint(backend="tinker", sampler="tinker://r/sampler_weights/0")

    monkeypatch.setitem(training._BACKENDS, "tinker", NoStateBackend())
    stages = [training.Stage("a", dataset), training.Stage("b", dataset)]
    with pytest.raises(RuntimeError, match="no state checkpoint"):
        asyncio.run(training.run_plan(ED_SPEC, stages, tmp_path / "c"))


# ---------------------------------------------------------------- presets
def test_presets_are_valid_partial_train_configs():
    import dataclasses

    known = {f.name for f in dataclasses.fields(training.TrainConfig)}
    names = training.list_presets()
    assert {"msm", "aft", "instruct"} <= set(names)
    for name in names:
        data = training.load_preset(name)
        assert data, f"preset {name} is empty"
        assert set(data) <= known, f"preset {name} has unknown keys"
        # partial by design: substrate identity follows the spec, not the preset
        assert not {"model", "renderer", "backend"} & set(data)


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        training.load_preset("nope")


# ------------------------------------------------------------------- data.py
def _write_rows(path, tag, n):
    rows = [json.dumps({"messages": [{"role": "assistant", "content": f"{tag}{i}"}]})
            for i in range(n)]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_interleave_mixes_with_repeats_deterministically(tmp_path):
    a = _write_rows(tmp_path / "a.jsonl", "a", 4)
    b = _write_rows(tmp_path / "b.jsonl", "b", 2)
    out1 = training.interleave([a, b], tmp_path / "mix1.jsonl", repeats=[1, 3], seed=7)
    out2 = training.interleave([a, b], tmp_path / "mix2.jsonl", repeats=[1, 3], seed=7)

    lines = out1.read_text().splitlines()
    assert len(lines) == 4 + 3 * 2
    assert out1.read_text() == out2.read_text()  # deterministic given seed
    contents = [json.loads(ln)["messages"][0]["content"] for ln in lines]
    assert contents != sorted(contents)  # actually shuffled together
    assert sum(c.startswith("b") for c in contents) == 6


def test_interleave_rejects_bad_inputs(tmp_path):
    a = _write_rows(tmp_path / "a.jsonl", "a", 2)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError):
        training.interleave([a], tmp_path / "o.jsonl", repeats=[1, 2])
    with pytest.raises(ValueError):
        training.interleave([a, empty], tmp_path / "o.jsonl")
    with pytest.raises(ValueError):
        training.interleave([a], tmp_path / "o.jsonl", repeats=[0])

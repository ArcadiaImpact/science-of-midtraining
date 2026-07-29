"""CPU-only durability tests for prior-latmem training and smoke runners."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem import smoke
from experiments.prior_latmem.pod import chain
from experiments.prior_latmem.pod.chain import plan_relayout
from scimt.train import Checkpoint


def _one_arm():
    return [
        {
            "name": "sdf_test",
            "stage": "sdf_it_gemma3_12b",
            "resume_of": None,
            "dataset": "mix",
        }
    ]


def _install_fake_hf(monkeypatch, tmp_path, *, uploaded=(), base_files=()):
    uploads: list[tuple[str, str]] = []
    # The consolidation backfill pulls the base model's small companion files
    # from the Hub; serve them from a local stub so no test touches the network.
    base_root = tmp_path / "base"
    base_root.mkdir(parents=True, exist_ok=True)
    for name in base_files:
        (base_root / name).write_text(f'{{"stub": "{name}"}}\n')

    class FakeApi:
        def create_repo(self, *_args, **_kwargs):
            return None

        def list_repo_files(self, *_args, **_kwargs):
            return list(uploaded)

        def upload_folder(self, *, folder_path, repo_id, path_in_repo):
            del repo_id
            uploads.append((folder_path, path_in_repo))

    def snapshot_download(_repo, *, allow_patterns):
        name = allow_patterns[0].split("/", 1)[0]
        root = tmp_path / "hub"
        (root / name).mkdir(parents=True, exist_ok=True)
        return str(root)

    def hf_hub_download(_repo, name, **_kwargs):
        path = base_root / name
        if not path.exists():
            raise FileNotFoundError(name)
        return str(path)

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = FakeApi
    fake.snapshot_download = snapshot_download
    fake.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return uploads


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode()
    return len(header).to_bytes(8, "little") + header + b"\0\0\0\0"


def _make_consolidated(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "stub"}\n')
    (path / "model.safetensors").write_bytes(_safetensors_bytes())


def _make_dcp(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".metadata").write_bytes(
        pickle.dumps({"files": ["__0_0.distcp"]})
    )
    (path / "__0_0.distcp").write_bytes(b"shard")


def _make_trainer_checkpoint(
    out_dir: Path,
    *,
    step: int,
    max_steps: int,
    resumable: bool,
) -> Path:
    checkpoint = out_dir / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "max_steps": max_steps}) + "\n"
    )
    _make_dcp(checkpoint / "pytorch_model_fsdp_0")
    if resumable:
        _make_dcp(checkpoint / "optimizer_0")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        (checkpoint / "training_args.bin").write_bytes(b"args")
        for rank in range(chain.TRAIN_WORLD_SIZE):
            (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"rng")
    return checkpoint


def _patch_one_arm(monkeypatch, tmp_path, *, base_files=()):
    monkeypatch.setattr(chain, "OUT", tmp_path / "out")
    monkeypatch.setattr(chain, "WORK", tmp_path / "work")
    monkeypatch.setattr(chain, "plan", _one_arm)
    uploads = _install_fake_hf(monkeypatch, tmp_path, base_files=base_files)
    data_path = tmp_path / "mix.jsonl"
    data_path.write_text('{"text": "stub"}\n')
    data = {"mix": types.SimpleNamespace(path=data_path)}
    return data, uploads


def test_torn_arm_ledger_does_not_block_hf_restart(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(chain, "OUT", tmp_path / "out")
    monkeypatch.setattr(chain, "WORK", tmp_path / "work")
    monkeypatch.setattr(chain, "plan", _one_arm)
    _install_fake_hf(
        monkeypatch,
        tmp_path,
        uploaded=("sdf_test/config.json",),
    )
    ledger = chain.OUT / "arm_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        '{"arm": "old", "status": "trained"}\n'
        '{"arm": "torn", "status":'
    )
    fsync_calls = []
    real_fsync = chain.os.fsync

    def tracking_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(chain.os, "fsync", tracking_fsync)
    with caplog.at_level(logging.WARNING):
        local = asyncio.run(chain.run_chain({}))

    assert "sdf_test" in local
    assert "skipped 1 malformed arm-ledger line(s)" in caplog.text
    assert str(ledger) in caplog.text
    assert fsync_calls


def test_backfill_completes_a_checkpoint_without_touching_the_finetune(
    monkeypatch, tmp_path
):
    """A published checkpoint must load for consumers other than its writer.

    transformers 5.x stops writing preprocessor_config.json /
    special_tokens_map.json, and FSDP2 trainer checkpoints have no
    generation_config.json to carry forward, so a consolidated dir is missing
    files older stacks (and later readers of the published artifact) expect.
    """
    _install_fake_hf(
        monkeypatch,
        tmp_path,
        base_files=("generation_config.json", "preprocessor_config.json"),
    )
    consolidated = tmp_path / "consolidated"
    _make_consolidated(consolidated)
    finetuned_config = (consolidated / "config.json").read_bytes()
    (consolidated / "tokenizer_config.json").write_text('{"eos_token": "<eot>"}\n')
    finetuned_tokenizer = (consolidated / "tokenizer_config.json").read_bytes()

    actions = chain._backfill_base_files(consolidated, "unsloth/gemma-3-12b-it")

    assert actions["generation_config.json"] == "copied"
    assert actions["preprocessor_config.json"] == "copied"
    # A base that simply lacks a file is normal, not a failure.
    assert actions["tokenizer.model"] == "skipped (FileNotFoundError)"
    # The fine-tune's own artifacts are never replaced.
    assert (consolidated / "config.json").read_bytes() == finetuned_config
    assert (consolidated / "tokenizer_config.json").read_bytes() == finetuned_tokenizer
    assert not chain._BACKFILL_PROTECTED & set(chain.BACKFILL_FROM_BASE)
    # The checkpoint must carry exactly one chat template: the one its stage
    # trained with. The sampler wraps probes with the served tokenizer's
    # template, so a second one from the base could silently make eval wrapping
    # differ from training wrapping.
    assert "chat_template.json" not in chain.BACKFILL_FROM_BASE
    assert not (consolidated / "chat_template.json").exists()

    # Idempotent: a second pass keeps what the first one wrote.
    again = chain._backfill_base_files(consolidated, "unsloth/gemma-3-12b-it")
    assert again["generation_config.json"] == "kept"


def _gemma3_keys(layout: str) -> set[str]:
    """The two real Gemma3-12B key layouts, abridged to one of each family.

    Left: what the trainer's transformers 5.x writes. Right: what the substrate
    repo ships and what vLLM's gemma3_mm.py looks for.
    """
    if layout == "trainer":
        return {
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.norm.weight",
            "model.vision_tower.embeddings.patch_embedding.weight",
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight",
            "model.vision_tower.post_layernorm.weight",
            "model.multi_modal_projector.mm_input_projection_weight",
            "lm_head.weight",
        }
    return {
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.norm.weight",
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "vision_tower.vision_model.post_layernorm.weight",
        "multi_modal_projector.mm_input_projection_weight",
    }


def test_relayout_maps_the_trainer_layout_onto_the_substrate_exactly():
    """The bug that killed a sampling pod after a 24GB download.

    vLLM 0.25.0 raised "no module or parameter named 'vision_tower.embeddings'"
    because transformers 5.x renamed Gemma3's module tree. Nothing measured is
    at stake — the tensors are identical — but a checkpoint the sampler cannot
    load is worthless, so consolidation renames onto the substrate's own keys.
    """
    ours, base = _gemma3_keys("trainer"), _gemma3_keys("substrate")
    assert not ours & base  # every single key differs; this is not a near-miss

    renames, drops = plan_relayout(ours, base, tied_embeddings=True)

    assert set(renames.values()) == base
    # The tied lm_head is a view of embed_tokens; the substrate ships no such
    # weight, so dropping it is lossless rather than a silent weight loss.
    assert drops == {"lm_head.weight"}
    assert renames["model.vision_tower.embeddings.patch_embedding.weight"] == (
        "vision_tower.vision_model.embeddings.patch_embedding.weight"
    )


def test_relayout_refuses_to_publish_an_unrecognized_layout():
    # A future transformers rename must fail on the training pod, not after a
    # sampling pod has downloaded 24GB and initialized an engine.
    ours = _gemma3_keys("trainer") | {"model.brand_new_tower.weight"}
    with pytest.raises(RuntimeError, match="cannot be relaid out"):
        plan_relayout(ours, _gemma3_keys("substrate"), tied_embeddings=True)
    # Untied embeddings: lm_head is real, so it may not be silently discarded.
    with pytest.raises(RuntimeError, match="unexpected"):
        plan_relayout(
            _gemma3_keys("trainer"),
            _gemma3_keys("substrate"),
            tied_embeddings=False,
        )


def test_relayout_is_a_noop_when_the_layouts_already_agree(tmp_path, monkeypatch):
    base = _gemma3_keys("substrate")
    renames, drops = plan_relayout(base, base, tied_embeddings=True)
    assert renames == {key: key for key in base} and drops == set()


def test_backfill_reads_a_local_parent_checkpoint_for_chained_arms(tmp_path):
    # Chained arms consolidate against the parent arm's local sampler dir, not
    # an HF id; the parent has itself already been backfilled.
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "generation_config.json").write_text('{"eos_token_id": [1, 106]}\n')
    child = tmp_path / "child"
    _make_consolidated(child)

    actions = chain._backfill_base_files(child, str(parent))

    assert actions["generation_config.json"] == "copied"
    assert actions["preprocessor_config.json"] == "absent-in-base"
    assert (child / "generation_config.json").read_text() == '{"eos_token_id": [1, 106]}\n'


def test_chain_backfills_a_salvaged_consolidated_checkpoint_before_upload(
    monkeypatch, tmp_path
):
    # The salvage path skips consolidation entirely; it must not skip the
    # completion step, or a resumed arm would publish a thinner checkpoint
    # than a freshly trained one.
    data, uploads = _patch_one_arm(
        monkeypatch, tmp_path, base_files=("generation_config.json",)
    )
    consolidated = chain.WORK / "consolidated" / "sdf_test"
    _make_consolidated(consolidated)

    async def should_not_train(*_args, **_kwargs):
        raise AssertionError("training must not run")

    from scimt.train.axolotl import LocalExecutor

    monkeypatch.setattr(LocalExecutor, "run_stage", should_not_train)
    monkeypatch.setattr(
        chain, "_consolidate", lambda *_a, **_k: pytest.fail("must not consolidate")
    )
    asyncio.run(chain.run_chain(data))

    assert uploads == [(str(consolidated), "sdf_test")]
    assert (consolidated / "generation_config.json").exists()


def test_chain_uploads_valid_consolidated_without_training(
    monkeypatch, tmp_path
):
    data, uploads = _patch_one_arm(monkeypatch, tmp_path)
    consolidated = chain.WORK / "consolidated" / "sdf_test"
    _make_consolidated(consolidated)

    async def should_not_train(*_args, **_kwargs):
        raise AssertionError("training must not run")

    def should_not_consolidate(*_args, **_kwargs):
        raise AssertionError("consolidation must not run")

    from scimt.train.axolotl import LocalExecutor

    monkeypatch.setattr(LocalExecutor, "run_stage", should_not_train)
    monkeypatch.setattr(chain, "_consolidate", should_not_consolidate)
    result = asyncio.run(chain.run_chain(data))

    assert result["sdf_test"] == str(consolidated)
    assert uploads == [(str(consolidated), "sdf_test")]


def test_chain_consolidates_finished_trainer_checkpoint_without_training(
    monkeypatch, tmp_path
):
    data, uploads = _patch_one_arm(monkeypatch, tmp_path)
    out_dir = chain.WORK / "train" / "sdf_test"
    finished = _make_trainer_checkpoint(
        out_dir, step=10, max_steps=10, resumable=False
    )
    calls = []

    async def should_not_train(*_args, **_kwargs):
        raise AssertionError("training must not run")

    def fake_consolidate(checkpoint, base_model, destination):
        calls.append((checkpoint, base_model, destination))
        _make_consolidated(destination)
        return destination

    from scimt.train.axolotl import LocalExecutor

    monkeypatch.setattr(LocalExecutor, "run_stage", should_not_train)
    monkeypatch.setattr(chain, "_consolidate", fake_consolidate)
    asyncio.run(chain.run_chain(data))

    assert calls[0][0] == finished
    assert uploads[0][1] == "sdf_test"


def test_chain_discards_garbage_then_trains(monkeypatch, tmp_path):
    data, uploads = _patch_one_arm(monkeypatch, tmp_path)
    out_dir = chain.WORK / "train" / "sdf_test"
    garbage = out_dir / "checkpoints" / "checkpoint-10"
    garbage.mkdir(parents=True)
    (garbage / "trainer_state.json").write_text('{"global_step":')
    prepared = out_dir / "prepared"
    prepared.mkdir()
    (prepared / "partial.arrow").write_bytes(b"partial")
    trained = []

    async def fake_train(_self, _rendered, actual_out, _stage):
        assert not garbage.exists()
        assert not prepared.exists()
        trained.append(True)
        _make_trainer_checkpoint(
            actual_out, step=10, max_steps=10, resumable=False
        )

    def fake_consolidate(_checkpoint, _base_model, destination):
        _make_consolidated(destination)
        return destination

    from scimt.train.axolotl import LocalExecutor

    monkeypatch.setattr(LocalExecutor, "run_stage", fake_train)
    monkeypatch.setattr(chain, "_consolidate", fake_consolidate)
    asyncio.run(chain.run_chain(data))

    assert trained == [True]
    assert uploads[0][1] == "sdf_test"


def test_chain_threads_valid_partial_checkpoint_into_rendered_resume(
    monkeypatch, tmp_path
):
    data, _uploads = _patch_one_arm(monkeypatch, tmp_path)
    out_dir = chain.WORK / "train" / "sdf_test"
    partial = _make_trainer_checkpoint(
        out_dir, step=6, max_steps=10, resumable=True
    )
    rendered_bodies = []

    async def fake_train(_self, rendered, actual_out, _stage):
        body = yaml.safe_load(rendered.read_text())
        rendered_bodies.append(body)
        assert partial.exists()
        _make_trainer_checkpoint(
            actual_out, step=10, max_steps=10, resumable=False
        )

    def fake_consolidate(_checkpoint, _base_model, destination):
        _make_consolidated(destination)
        return destination

    from scimt.train.axolotl import LocalExecutor

    monkeypatch.setattr(LocalExecutor, "run_stage", fake_train)
    monkeypatch.setattr(chain, "_consolidate", fake_consolidate)
    asyncio.run(chain.run_chain(data))

    assert rendered_bodies[0]["resume_from_checkpoint"] == str(partial)
    assert rendered_bodies[0]["base_model"] == chain.BASE_MODEL


def _make_smoke_stage(
    out_dir: Path,
    *,
    artifact: Path,
    stage: str,
    resume_from: str | None,
) -> Checkpoint:
    _make_consolidated(artifact)
    checkpoint = Checkpoint(
        backend="axolotl",
        sampler=str(artifact),
        state=str(artifact),
        model="Qwen/Qwen2.5-0.5B",
        meta={
            "train": {
                "stage": stage,
                "load_checkpoint_path": resume_from,
            }
        },
    )
    checkpoint.save(out_dir)
    return checkpoint


def test_smoke_skips_both_valid_completed_stages(
    monkeypatch, tmp_path, capsys
):
    out = tmp_path / "smoke"
    sdf = _make_smoke_stage(
        out / "train_sdf",
        artifact=out / "train_sdf" / "checkpoints",
        stage="smoke_qwen05b",
        resume_from=None,
    )
    _make_smoke_stage(
        out / "train_aft",
        artifact=out / "train_aft" / "checkpoints",
        stage="smoke_qwen05b_chat",
        resume_from=sdf.require_state(),
    )

    def tiny_corpora(data_out, *, n=200):
        del n
        data_out.mkdir(parents=True, exist_ok=True)
        speed, memory = data_out / "speed.jsonl", data_out / "memory.jsonl"
        speed.write_text('{"text": "speed"}\n')
        memory.write_text('{"text": "memory"}\n')
        return speed, memory

    class Mix:
        path = str(out / "mix" / "mix.jsonl")
        meta = {"mix": {"valid": True}}

    async def fake_mix(*_args, **_kwargs):
        return Mix()

    async def should_not_train(*_args, **_kwargs):
        raise AssertionError("completed smoke stages must not retrain")

    import scimt.prepare as prepare_mod
    import scimt.train as train_mod

    monkeypatch.setattr(smoke, "synthetic_corpora", tiny_corpora)
    monkeypatch.setattr(prepare_mod, "mix", fake_mix)
    monkeypatch.setattr(train_mod, "train_dataset", should_not_train)
    result = asyncio.run(
        smoke.train_smoke(
            smoke.Config(out=str(out), skip_train=False),
            out,
        )
    )

    output = capsys.readouterr().out
    assert "training stage 1: SKIP" in output
    assert "training stage 2: SKIP" in output
    assert result["sdf"]["state"] == sdf.require_state()


@pytest.mark.parametrize("payload", ("", '{"backend":'))
def test_smoke_rejects_empty_or_torn_stage_manifest(tmp_path, payload):
    out_dir = tmp_path / "train_sdf"
    out_dir.mkdir()
    (out_dir / "checkpoint.json").write_text(payload)

    assert (
        smoke._completed_training_stage(
            out_dir,
            stage="smoke_qwen05b",
            model="Qwen/Qwen2.5-0.5B",
        )
        is None
    )


def test_smoke_rejects_nonempty_but_truncated_model(tmp_path):
    out_dir = tmp_path / "train_sdf"
    artifact = out_dir / "checkpoints"
    checkpoint = _make_smoke_stage(
        out_dir,
        artifact=artifact,
        stage="smoke_qwen05b",
        resume_from=None,
    )
    (Path(checkpoint.state) / "model.safetensors").write_bytes(b"not-empty")

    assert (
        smoke._completed_training_stage(
            out_dir,
            stage="smoke_qwen05b",
            model="Qwen/Qwen2.5-0.5B",
        )
        is None
    )

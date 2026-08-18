"""Pinned contracts for the Dispatch 4B/27B scale-up (CPU-only).

Guards the port invariants: identical optimizer trajectories to the 12B runs
(tokens/update and sequences/update preserved under world-size rebalancing),
the D2 checkpoint contract (five resumable full-state checkpoints per
full-parameter stage), and byte-identity of every data contract with the 12B
originals.
"""

from __future__ import annotations

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (
    contracts as gate2,
)
from experiments.improved_midtraining.dispatch_midtrain_4epoch.run_arm import (
    EXPECTED_FILLER,
    EXPECTED_MIXES,
)
from experiments.prior_coins.dispatch_scaleup import (
    contracts,
    midtrain_arm,
    wave_cells,
)
from scimt.train.axolotl import load_stage

SIZES = sorted(contracts.SIZES)


@pytest.mark.parametrize("name", SIZES)
def test_geometry_preserves_the_12b_optimizer_trajectory(name: str) -> None:
    spec = contracts.size(name)
    contracts.require_geometry(spec)

    midtrain = load_stage(spec.midtrain_stage).axolotl
    assert midtrain["micro_batch_size"] == 1
    assert midtrain["gradient_accumulation_steps"] == spec.midtrain_accumulation
    assert midtrain["sequence_len"] == 8192
    assert (
        1 * spec.midtrain_accumulation * 8192 * spec.world_size
        == contracts.MIDTRAIN_TOKENS_PER_UPDATE
    )
    assert midtrain["max_steps"] == 124
    assert midtrain["num_epochs"] == 4
    assert midtrain["learning_rate"] == 1.0e-5
    assert midtrain["warmup_ratio"] == 0.03
    assert midtrain["lr_scheduler"] == "cosine"
    assert midtrain["cosine_min_lr_ratio"] == 0.1
    assert midtrain["sample_packing"] is True

    sft = load_stage(spec.sft_stage).axolotl
    assert sft["micro_batch_size"] == spec.sft_micro_batch
    assert sft["gradient_accumulation_steps"] == spec.sft_accumulation
    assert (
        spec.sft_micro_batch * spec.sft_accumulation * spec.world_size
        == contracts.SFT_SEQUENCES_PER_UPDATE
    )
    assert sft["max_steps"] == 48
    assert sft["warmup_steps"] == 3
    assert sft["learning_rate"] == 1.0e-5
    assert sft["train_on_inputs"] is False

    aft = load_stage(spec.aft_stage).axolotl
    # global batch 32, the invariant of every Dispatch AFT
    assert aft["micro_batch_size"] * aft["gradient_accumulation_steps"] == 32
    assert aft["num_epochs"] == 2
    assert aft["sequence_len"] == 1280
    assert aft["sample_packing"] is False
    assert aft["learning_rate"] == 1.0e-4
    assert aft["base_model_config"] == spec.base_model


@pytest.mark.parametrize("name", SIZES)
def test_d2_checkpoint_contract_full_state_at_five_points(name: str) -> None:
    spec = contracts.size(name)

    midtrain = load_stage(spec.midtrain_stage).axolotl
    assert midtrain["checkpoint_schedule"] == list(contracts.MIDTRAIN_CHECKPOINTS)
    assert midtrain["save_only_model"] is False
    assert midtrain["save_strategy"] == "no"
    # the 12B publication boundary {post-warmup, final} stays a subset
    assert {4, 124} <= set(midtrain["checkpoint_schedule"])
    # epoch boundaries: 31 updates per epoch
    assert set(midtrain["checkpoint_schedule"]) - {4} == {31, 62, 93, 124}

    sft = load_stage(spec.sft_stage).axolotl
    assert sft["checkpoint_schedule"] == list(contracts.SFT_CHECKPOINTS)
    assert sft["save_only_model"] is False
    assert {4, 48} <= set(sft["checkpoint_schedule"])

    aft = load_stage(spec.aft_stage).axolotl
    assert aft["save_only_model"] is False
    assert aft["save_steps"] == 32
    assert aft["save_total_limit"] >= len(contracts.AFT_CHECKPOINTS)


@pytest.mark.parametrize("name", SIZES)
def test_validate_stage_accepts_own_stage_and_rejects_model_only(name: str) -> None:
    spec = contracts.size(name)
    body = dict(load_stage(spec.midtrain_stage).axolotl)
    for arm in contracts.ARMS:
        tokens = contracts.expected_mix(arm)["tokens"]
        steps = midtrain_arm.validate_stage(
            body, world_size=spec.world_size, total_tokens=tokens, spec=spec
        )
        assert steps == 124

    regression = dict(body, save_only_model=True)
    with pytest.raises(ValueError, match="save_only_model"):
        midtrain_arm.validate_stage(
            regression,
            world_size=spec.world_size,
            total_tokens=EXPECTED_MIXES["coin"]["tokens"],
            spec=spec,
        )
    with pytest.raises(ValueError, match="world_size"):
        midtrain_arm.validate_stage(
            body,
            world_size=spec.world_size + 1,
            total_tokens=EXPECTED_MIXES["coin"]["tokens"],
            spec=spec,
        )


def test_data_contracts_are_byte_identical_to_the_12b_runs() -> None:
    # Gemma-3 sizes share one tokenizer, so these carry over exactly.
    assert contracts.expected_mix("charter") == EXPECTED_MIXES["charter"]
    assert contracts.expected_mix("coin") == EXPECTED_MIXES["coin"]
    assert contracts.expected_filler() == EXPECTED_FILLER
    assert contracts.CONTROL_EXPECTED == {
        "docs": gate2.DOLMINO8_DOCS,
        "tokens": gate2.DOLMINO8_TOKENS,
        "jsonl_sha256": gate2.DOLMINO8_JSONL_SHA256,
        "ordered_rows_sha256": gate2.DOLMINO8_ORDERED_ROWS_SHA256,
    }
    assert contracts.CONTROL_TOKEN_BUDGET == gate2.MIDTRAIN_TARGET
    # equal compute: control total tokens ~ doc-arm mixture totals
    assert abs(
        contracts.CONTROL_EXPECTED["tokens"] - EXPECTED_MIXES["coin"]["tokens"]
    ) < 10_000


def test_base_revisions_are_pinned_full_shas() -> None:
    for spec in contracts.SIZES.values():
        assert len(spec.base_revision) == 40
        assert spec.base_model.startswith("unsloth/gemma-3-")
        stage = load_stage(spec.midtrain_stage)
        assert stage.base_model == spec.base_model
        assert load_stage(spec.sft_stage).base_model == spec.base_model
        assert load_stage(spec.aft_stage).base_model == spec.base_model


def test_27b_pod_shape_respects_the_proven_oom_boundary() -> None:
    # Full-parameter 27B FSDP OOMs on 80 GB parts (8xH100, 2026-08-10).
    spec = contracts.size("27b")
    pod = load_stage(spec.midtrain_stage).pod
    assert pod.gpu == "H200"
    assert pod.gpu_count == spec.world_size == 8
    assert pod.disk_gb >= 2000  # 5 full-state ~275 GB checkpoints + staging
    assert spec.min_host_ram_bytes >= 600 * 1024**3


def test_select_checkpoints_labels_and_gates_all_five(tmp_path) -> None:
    source = tmp_path / "base"
    source.mkdir()
    for sidecar in ("processor_config.json", "preprocessor_config.json"):
        (source / sidecar).write_text("{}")
    root = tmp_path / "checkpoints"
    for step in contracts.MIDTRAIN_CHECKPOINTS:
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (checkpoint / name).write_text("{}")
        (checkpoint / "trainer_state.json").write_text(json.dumps({
            "global_step": step,
            "max_steps": 124,
            "epoch": 4.0 if step == 124 else step / 31,
        }))
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        (checkpoint / "optimizer.pt").write_bytes(b"adam")
        (checkpoint / "scheduler.pt").write_bytes(b"cosine")
        (checkpoint / "rng_state_0.pth").write_bytes(b"rng")

    selected = midtrain_arm.select_checkpoints(
        root,
        post_warmup_step=contracts.POST_WARMUP_STEP,
        min_final_step=contracts.MIDTRAIN_FINAL_STEP,
        processor_source=source,
    )
    assert set(selected) == {"post_warmup", "step31", "step62", "step93", "final"}
    assert selected["final"].name == "checkpoint-124"
    # hydration copied the sidecars into every checkpoint
    for checkpoint in selected.values():
        assert (checkpoint / "processor_config.json").is_file()
        assert (checkpoint / "checkpoint_hydration.json").is_file()

    # a model-only checkpoint (no optimizer state) must fail the D2 gate
    (root / "checkpoint-62" / "optimizer.pt").unlink()
    with pytest.raises(RuntimeError, match="resumable state"):
        midtrain_arm.select_checkpoints(
            root,
            post_warmup_step=contracts.POST_WARMUP_STEP,
            min_final_step=contracts.MIDTRAIN_FINAL_STEP,
            processor_source=source,
        )


def test_select_checkpoints_rejects_missing_step(tmp_path) -> None:
    source = tmp_path / "base"
    source.mkdir()
    for sidecar in ("processor_config.json", "preprocessor_config.json"):
        (source / sidecar).write_text("{}")
    root = tmp_path / "checkpoints"
    (root / "checkpoint-4").mkdir(parents=True)
    (root / "checkpoint-124").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="expected checkpoints"):
        midtrain_arm.select_checkpoints(
            root,
            post_warmup_step=contracts.POST_WARMUP_STEP,
            min_final_step=contracts.MIDTRAIN_FINAL_STEP,
            processor_source=source,
        )


@pytest.mark.parametrize("name", SIZES)
def test_wave_cells_cover_all_arms_on_the_agreement_mixture(name: str) -> None:
    spec = contracts.size(name)
    plan = wave_cells.cells(spec)
    assert [cell["parent"] for cell in plan] == list(contracts.ARMS)
    for cell in plan:
        assert cell["mixture"] == "agreement"
        assert cell["stage"] == spec.aft_stage
        assert cell["parent_prefix"].endswith("/checkpoint-48")
        command = wave_cells.chain_command(spec, cell, "0" * 40)
        assert f"--stage {spec.aft_stage}" in command
        assert "--dataset agreement" in command
        assert "--version dispatch_v4_wide" in command


def test_sft_parent_pins_are_required_and_validated(tmp_path, monkeypatch) -> None:
    from experiments.prior_coins.dispatch_scaleup import sft_arm

    monkeypatch.setattr(sft_arm, "PINS_DIR", tmp_path)
    spec = contracts.size("4b")
    with pytest.raises(FileNotFoundError, match="parent pins"):
        sft_arm.load_parent_pins(spec)

    good = {
        arm: {
            "revision": "a" * 40,
            "prefix": f"midtrain_4epoch/{arm}/checkpoint-124",
            "model_tree_sha256": "b" * 64,
        }
        for arm in contracts.ARMS
    }
    path = tmp_path / "4b_midtrain_parents.json"
    path.write_text(json.dumps(good))
    assert set(sft_arm.load_parent_pins(spec)) == set(contracts.ARMS)

    incomplete = {arm: pin for arm, pin in good.items() if arm != "control"}
    path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="exactly"):
        sft_arm.load_parent_pins(spec)

    wrong_prefix = json.loads(json.dumps(good))
    wrong_prefix["coin"]["prefix"] = "midtrain_4epoch/coin/checkpoint-4"
    path.write_text(json.dumps(wrong_prefix))
    with pytest.raises(ValueError, match="prefix"):
        sft_arm.load_parent_pins(spec)


def test_launchers_refuse_without_signoff(monkeypatch) -> None:
    from experiments.prior_coins.dispatch_scaleup import launch_midtrain, launch_sft

    for module in (launch_midtrain, launch_sft):
        monkeypatch.setattr(
            "sys.argv",
            ["launch", "--size", "4b", "--run-id", "20990101T000000Z",
             "--output", "/tmp/never-used"],
        )
        with pytest.raises(SystemExit, match="refusing to provision"):
            module.main()


# --- checkpoint publication (UPLOAD_ARCHITECTURE.md) -----------------------


class _FakeCommit:
    def __init__(self, oid: str) -> None:
        self.oid = oid
        self.commit_url = f"https://hub.invalid/commit/{oid}"


class _FakeApi:
    """Records preuploads and commits; serves a remote index back for verify."""

    token = None

    def __init__(self) -> None:
        self.preuploaded: list[list[str]] = []
        self.commits: list[tuple[str, list[str]]] = []
        self._remote: dict[str, dict] = {}
        self._contents: dict[str, bytes] = {}

    def preupload_lfs_files(self, repo_id, additions, repo_type=None):
        self.preuploaded.append([a.path_in_repo for a in additions])

    def create_commit(
        self, *, repo_id, repo_type, operations, commit_message, **kwargs
    ):
        oid = f"{len(self.commits):040x}"
        for op in operations:
            payload = Path(op.path_or_fileobj).read_bytes()
            self._contents[op.path_in_repo] = payload
            self._remote[op.path_in_repo] = {
                "size": len(payload),
                "lfs_sha256": None,
            }
        self.commits.append((commit_message, [op.path_in_repo for op in operations]))
        return _FakeCommit(oid)

    def model_info(self, repo_id, revision=None, files_metadata=False):
        class _S:
            def __init__(self, name, meta):
                self.rfilename = name
                self.size = meta["size"]
                self.lfs = None

        class _I:
            pass

        info = _I()
        info.siblings = [_S(n, m) for n, m in self._remote.items()]
        return info

    def read(self, repo_id, path, revision):
        return self._contents[path]


def _full_state_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights-" + str(step).encode())
    (checkpoint / "pytorch_model_fsdp.bin").write_bytes(
        b"weights-" + str(step).encode()
    )
    (checkpoint / "optimizer.bin").write_bytes(b"adam")
    (checkpoint / "scheduler.pt").write_bytes(b"cosine")
    (checkpoint / "rng_state_0.pth").write_bytes(b"rng")
    (checkpoint / "config.json").write_text("{}")
    return checkpoint


def test_sft_world_fallback_holds_the_optimizer_trajectory() -> None:
    import pytest as _pytest

    base = contracts.size("27b")
    variant = contracts.sft_world_variant(base, 4)
    # same positions/update, reached with half the ranks and double the
    # accumulation -- the freedom the 4B leg already used at world 2
    assert contracts.sft_sequences_per_update(variant) == (
        contracts.sft_sequences_per_update(base)
    )
    assert (variant.world_size, variant.sft_micro_batch, variant.sft_accumulation) == (
        4,
        2,
        32,
    )
    # the variant stays internally consistent, so require_geometry passes on it
    contracts.require_geometry(variant)
    # ...and the publication contract is untouched: same name, repos, prefixes
    assert variant.name == base.name
    assert variant.models_repo == base.models_repo
    assert variant.sft_write_repo == base.sft_write_repo
    assert variant.model_prefix("sft", "control", 48) == base.model_prefix(
        "sft", "control", 48
    )
    # the rendered stage must agree with the spec it was registered for
    stage = load_stage(variant.sft_stage).axolotl
    assert stage["micro_batch_size"] == variant.sft_micro_batch
    assert stage["gradient_accumulation_steps"] == variant.sft_accumulation
    assert stage["sequence_len"] == 8192
    assert stage["checkpoint_schedule"] == list(contracts.SFT_CHECKPOINTS)
    assert stage["seed"] == 314159
    # the 8-GPU stage is what charter and coin ran; it must be left alone
    assert load_stage(base.sft_stage).axolotl["micro_batch_size"] == 4
    assert contracts.sft_world_variant(base, base.world_size) is base
    with _pytest.raises(ValueError, match="no SFT world-3 fallback"):
        contracts.sft_world_variant(base, 3)


def test_sft_publishes_only_the_two_endpoints() -> None:
    # the recipe still saves all five (that is the trajectory contract); what
    # narrowed on 2026-08-18 is what gets published, under option B.
    assert set(contracts.SFT_PUBLISH_STEPS) <= set(contracts.SFT_CHECKPOINTS)
    assert contracts.SFT_PUBLISH_STEPS == (4, 48)
    # every published checkpoint is a resume boundary, so all of them carry the
    # FSDP duplicate; nothing is published without its optimizer state
    assert set(contracts.SFT_DUPLICATE_WEIGHT_STEPS) == set(
        contracts.SFT_PUBLISH_STEPS
    )


def test_sft_writes_may_be_redirected_without_moving_the_parent_reads() -> None:
    # 27B: the personal account hit its public-storage ceiling mid-run, so SFT
    # publishes into the org while the pinned midtrain parents stay put.
    spec = contracts.size("27b")
    assert spec.sft_write_repo == "arcadia-impact/scimt-dispatch-27b-models-v1"
    assert spec.models_repo == "sidbaines/scimt-dispatch-27b-models-v1"
    assert spec.sft_write_repo != spec.models_repo
    # 4B ran before the split and must be unaffected
    four = contracts.size("4b")
    assert four.sft_output_repo is None
    assert four.sft_write_repo == four.models_repo


def test_duplicate_weight_steps_are_the_resume_boundaries() -> None:
    from experiments.prior_coins.dispatch_scaleup import checkpoint_upload

    assert set(contracts.MIDTRAIN_DUPLICATE_WEIGHT_STEPS) <= set(
        contracts.MIDTRAIN_CHECKPOINTS
    )
    assert set(contracts.SFT_DUPLICATE_WEIGHT_STEPS) <= set(contracts.SFT_CHECKPOINTS)
    # the duplicate is kept exactly at post-warmup and final, nowhere else
    assert contracts.MIDTRAIN_DUPLICATE_WEIGHT_STEPS == (4, 124)
    assert contracts.SFT_DUPLICATE_WEIGHT_STEPS == (4, 48)
    keep = contracts.MIDTRAIN_DUPLICATE_WEIGHT_STEPS
    assert checkpoint_upload.omitted_for(4, keep) == ()
    assert checkpoint_upload.omitted_for(124, keep) == ()
    for step in (31, 62, 93):
        assert checkpoint_upload.omitted_for(step, keep) == (
            checkpoint_upload.DUPLICATE_WEIGHTS,
        )


def test_filtered_manifest_drops_duplicate_but_never_the_weights(tmp_path) -> None:
    from experiments.prior_coins.dispatch_scaleup import checkpoint_upload

    checkpoint = _full_state_checkpoint(tmp_path, 62)
    kept = checkpoint_upload.filtered_manifest(
        checkpoint, (checkpoint_upload.DUPLICATE_WEIGHTS,)
    )
    assert "model.safetensors" in kept
    assert checkpoint_upload.DUPLICATE_WEIGHTS not in kept
    assert "optimizer.bin" in kept  # resumability is untouched

    full = checkpoint_upload.filtered_manifest(checkpoint, ())
    assert checkpoint_upload.DUPLICATE_WEIGHTS in full

    # omitting the real weights too must fail loudly, not publish a stub
    with pytest.raises(RuntimeError, match="no .safetensors weights"):
        checkpoint_upload.filtered_manifest(
            checkpoint,
            (checkpoint_upload.DUPLICATE_WEIGHTS, "model.safetensors"),
        )


def test_upload_checkpoints_commits_serially_and_omits_the_duplicate(
    tmp_path,
) -> None:
    pytest.importorskip("huggingface_hub")
    from experiments.prior_coins.dispatch_scaleup import checkpoint_upload

    root = tmp_path / "checkpoints"
    jobs = [
        checkpoint_upload.CheckpointJob(
            label=str(step),
            step=step,
            local_dir=_full_state_checkpoint(root, step),
            remote_prefix=f"sft_4epoch/charter/checkpoint-{step}",
            manifest_path=tmp_path / "manifests" / f"{step}.json",
            commit_message=f"checkpoint {step}",
        )
        for step in contracts.SFT_CHECKPOINTS
    ]
    api = _FakeApi()
    receipts = checkpoint_upload.upload_checkpoints(
        api,
        repo_id="org/models",
        jobs=jobs,
        keep_steps=contracts.SFT_DUPLICATE_WEIGHT_STEPS,
        read_remote_file=api.read,
    )

    assert set(receipts) == {str(s) for s in contracts.SFT_CHECKPOINTS}
    # every checkpoint was pre-uploaded, and committed exactly once, in order
    assert len(api.preuploaded) == len(contracts.SFT_CHECKPOINTS)
    committed_steps = [
        int(paths[0].rsplit("checkpoint-", 1)[1].split("/")[0])
        for _, paths in api.commits
    ]
    assert committed_steps == sorted(contracts.SFT_CHECKPOINTS)

    duplicate_by_step = {
        step: any(
            path.endswith(checkpoint_upload.DUPLICATE_WEIGHTS)
            for _, paths in api.commits
            for path in paths
            if f"checkpoint-{step}/" in path
        )
        for step in contracts.SFT_CHECKPOINTS
    }
    assert duplicate_by_step == {4: True, 12: False, 24: False, 36: False, 48: True}

    # the omission is recorded, on the receipt and in the on-disk manifest
    assert receipts["24"]["omitted"] == [checkpoint_upload.DUPLICATE_WEIGHTS]
    assert receipts["48"]["omitted"] == []
    manifest = json.loads((tmp_path / "manifests" / "24.json").read_text())
    assert manifest["omitted"] == [checkpoint_upload.DUPLICATE_WEIGHTS]
    assert checkpoint_upload.DUPLICATE_WEIGHTS not in manifest["files"]


def test_prewarming_uploader_passes_non_checkpoints_through(tmp_path) -> None:
    pytest.importorskip("huggingface_hub")
    from experiments.prior_coins.dispatch_scaleup import checkpoint_upload

    root = tmp_path / "checkpoints"
    selected = {
        "post_warmup": _full_state_checkpoint(root, 4),
        "step62": _full_state_checkpoint(root, 62),
    }
    other = tmp_path / "artifacts"
    other.mkdir()
    (other / "events.jsonl").write_text("{}\n")

    seen: list[Path] = []

    def original(api, *, local_dir, **kwargs):
        seen.append(Path(local_dir))
        return {"remote_prefix": kwargs["remote_prefix"], "delegated": True}

    uploader = checkpoint_upload.PrewarmingUploader(
        original,
        keep_steps=(4, 124),
        remote_prefix_of=lambda c: f"midtrain_4epoch/coin/{c.name}",
    )
    uploader.register(selected)
    api = _FakeApi()

    # a non-checkpoint tree goes to the original upload_tree untouched
    delegated = uploader(
        api,
        repo_id="org/models",
        local_dir=other,
        remote_prefix="runs/X/midtrain/coin/artifacts",
        manifest_path=tmp_path / "artifacts_files.json",
        commit_message="artifacts",
    )
    assert delegated == {
        "remote_prefix": "runs/X/midtrain/coin/artifacts",
        "delegated": True,
    }
    assert seen == [other]
    assert api.commits == []

    # a registered checkpoint is committed here, with the duplicate dropped
    receipt = uploader(
        api,
        repo_id="org/models",
        local_dir=selected["step62"],
        remote_prefix="midtrain_4epoch/coin/checkpoint-62",
        manifest_path=tmp_path / "step62_checkpoint_files.json",
        commit_message="coin step62",
        read_remote_file=api.read,
    )
    assert receipt["omitted"] == [checkpoint_upload.DUPLICATE_WEIGHTS]
    # prewarm staged BOTH registered checkpoints, not just the requested one
    assert len(api.preuploaded) == 2
    assert len(api.commits) == 1


def test_prewarming_uploader_rejects_a_prefix_it_did_not_stage(tmp_path) -> None:
    pytest.importorskip("huggingface_hub")
    from experiments.prior_coins.dispatch_scaleup import checkpoint_upload

    root = tmp_path / "checkpoints"
    selected = {"step62": _full_state_checkpoint(root, 62)}
    uploader = checkpoint_upload.PrewarmingUploader(
        lambda *a, **k: {},
        keep_steps=(4, 124),
        remote_prefix_of=lambda c: f"midtrain_4epoch/coin/{c.name}",
    )
    uploader.register(selected)
    with pytest.raises(RuntimeError, match="pre-uploaded bytes were staged"):
        uploader(
            _FakeApi(),
            repo_id="org/models",
            local_dir=selected["step62"],
            remote_prefix="somewhere_else/checkpoint-62",
            manifest_path=tmp_path / "m.json",
            commit_message="wrong",
        )

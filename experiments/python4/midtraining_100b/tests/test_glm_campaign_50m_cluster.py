"""CPU contracts for the Instant-Cluster 50M GLM path (no GPU/network/bellhop)."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import types
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_100b import (  # noqa: E402
    run_glm,
    run_glm_50m,
    run_glm_50m_cluster,
)
from experiments.python4.midtraining_100b.pod import (  # noqa: E402
    chain_glm,
    chain_glm_50m,
    chain_glm_50m_cluster,
)
from experiments.python4.midtraining_12b.pod import chain  # noqa: E402

mod = chain_glm_50m_cluster
launcher = run_glm_50m_cluster


# --- shape table / invariants ----------------------------------------------


def test_shape_ladder_world_size_and_ordering():
    shapes = [(s["gpu"], s["nodes"], s["gpu_count"]) for s in launcher.SHAPES]
    assert shapes == [("H200", 2, 4), ("H200", 4, 2), ("B200", 2, 4)]
    for shape in launcher.SHAPES:
        assert shape["nodes"] * shape["gpu_count"] == mod.WORLD_SIZE_REQUIRED == 8
        assert (REPO_ROOT / str(shape["requirements"])).exists()
    h200 = [s for s in launcher.SHAPES if s["gpu"] == "H200"]
    b200 = [s for s in launcher.SHAPES if s["gpu"] == "B200"]
    assert all(s["max_hourly_cost"] == 45.0 for s in h200)
    assert all(s["driver_min"] == 560 for s in h200)
    assert all(s["max_hourly_cost"] == 60.0 for s in b200)
    assert all(s["driver_min"] == 580 for s in b200)
    assert all(s["requirements"].endswith("pod-b200.txt") for s in b200)


def test_per_node_ram_thresholds_reproduce_the_proven_single_node_floor():
    # 8 local ranks (the single-node shape) must land EXACTLY on the proven
    # constant — the formula is a generalization, not a re-tune.
    assert mod.min_host_ram_gb(8) == chain_glm.MIN_HOST_RAM_GB == 1900
    assert mod.min_host_ram_gb(4) == 980   # 2 nodes x 4 GPUs
    assert mod.min_host_ram_gb(2) == 520   # 4 nodes x 2 GPUs
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            mod.min_host_ram_gb(bad)


def test_cluster_coords_enforce_world_size_8(monkeypatch):
    for key in ("NODE_RANK", "NUM_NODES", "NUM_TRAINERS"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="cluster env missing"):
        mod._cluster_coords()

    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setenv("NUM_TRAINERS", "4")
    assert mod._cluster_coords() == (1, 2, 4)

    monkeypatch.setenv("NUM_TRAINERS", "8")  # 2x8 = world 16
    with pytest.raises(RuntimeError, match="262,144 tokens/step"):
        mod._cluster_coords()

    monkeypatch.setenv("NUM_NODES", "1")
    monkeypatch.setenv("NUM_TRAINERS", "8")
    with pytest.raises(RuntimeError, match="chain_glm_50m.py"):
        mod._cluster_coords()


def test_run_id_required(monkeypatch):
    monkeypatch.delenv("GLM50M_CLUSTER_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="GLM50M_CLUSTER_RUN_ID"):
        mod._run_id()
    monkeypatch.setenv("GLM50M_CLUSTER_RUN_ID", "20260826T000000Z")
    assert mod._run_id() == "20260826T000000Z"


# --- config resolution: the ONE training-config change ----------------------


def test_cluster_config_injection_and_single_node_reference(tmp_path):
    cluster_cfg, single_cfg = mod.resolve_cluster_midtrain_config(1500, tmp_path)
    body = yaml.safe_load(cluster_cfg.read_text())
    fsdp = body["axolotl"]["fsdp_config"]
    assert fsdp["final_state_dict_type"] == "FULL_STATE_DICT"
    assert fsdp["state_dict_type"] == "SHARDED_STATE_DICT"  # training saves stay sharded
    assert body["axolotl"]["max_steps"] == 1500
    assert body["axolotl"]["checkpoint_schedule"] == [1500]
    assert body["name"].endswith("_cluster")
    # widened per-collective timeout: covers cross-node prep skew + rank 0's
    # ~221 GB final-save write while the other ranks wait at the barrier
    assert body["axolotl"]["ddp_timeout"] == mod.CLUSTER_DDP_TIMEOUT_S == 10_800
    # the single-node reference render carries NO injection (its sha is what
    # a single-node-trained artifact records)
    single_body = yaml.safe_load(single_cfg.read_text())
    assert "final_state_dict_type" not in single_body["axolotl"]["fsdp_config"]
    assert "ddp_timeout" not in single_body["axolotl"]
    assert chain._file_sha256(cluster_cfg) != chain._file_sha256(single_cfg)
    # the injected config still loads as a registry-equivalent StageSpec
    stage = chain.load_local_stage(cluster_cfg)
    assert stage.axolotl["fsdp_config"]["final_state_dict_type"] == "FULL_STATE_DICT"


def test_cluster_sft_config_keeps_schedule_and_injects(tmp_path):
    cluster_cfg, single_cfg = mod.resolve_cluster_sft_config(tmp_path)
    assert single_cfg == chain_glm.SFT_CONFIG
    body = yaml.safe_load(cluster_cfg.read_text())
    assert body["axolotl"]["fsdp_config"]["final_state_dict_type"] == "FULL_STATE_DICT"
    assert body["axolotl"]["ddp_timeout"] == mod.CLUSTER_DDP_TIMEOUT_S
    assert body["axolotl"]["checkpoint_schedule"] == [48]
    assert body["axolotl"]["max_steps"] == 48


def test_clusterize_refuses_drifted_templates(tmp_path):
    full = tmp_path / "full.yaml"
    full.write_text(yaml.safe_dump(
        {"axolotl": {"fsdp_config": {"state_dict_type": "FULL_STATE_DICT"}}}
    ))
    with pytest.raises(RuntimeError, match="premise drifted"):
        mod._clusterize_config(full, tmp_path / "out1.yaml")

    already = tmp_path / "already.yaml"
    already.write_text(yaml.safe_dump({"axolotl": {"fsdp_config": {
        "state_dict_type": "SHARDED_STATE_DICT",
        "final_state_dict_type": "FULL_STATE_DICT",
    }}}))
    with pytest.raises(RuntimeError, match="already sets"):
        mod._clusterize_config(already, tmp_path / "out2.yaml")

    has_timeout = tmp_path / "timeout.yaml"
    has_timeout.write_text(yaml.safe_dump({"axolotl": {
        "ddp_timeout": 999,
        "fsdp_config": {"state_dict_type": "SHARDED_STATE_DICT"},
    }}))
    with pytest.raises(RuntimeError, match="already sets ddp_timeout"):
        mod._clusterize_config(has_timeout, tmp_path / "out3.yaml")


def test_step_geometry_invariant_holds_on_both_stages(tmp_path):
    cluster_mid, _ = mod.resolve_cluster_midtrain_config(1500, tmp_path)
    cluster_sft, _ = mod.resolve_cluster_sft_config(tmp_path)
    mod.assert_step_geometry(cluster_mid, chain_glm.TOKENS_PER_MIDTRAIN_STEP)
    mod.assert_step_geometry(cluster_sft, mod.SFT_TOKENS_PER_STEP)
    assert chain_glm.TOKENS_PER_MIDTRAIN_STEP == 262_144
    assert mod.SFT_TOKENS_PER_STEP == 2_097_152
    with pytest.raises(RuntimeError, match="tokens/step"):
        mod.assert_step_geometry(cluster_mid, 262_143)


# --- per-node preflight ------------------------------------------------------


def _preflight_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    for key in chain_glm.RCLONE_ENV_REQUIRED:
        monkeypatch.setenv(key, "gs://bucket/base" if key == "SCIMT_GCS_BASE" else "v")
    monkeypatch.setenv("GLM50M_CLUSTER_RUN_ID", "rid")


def _preflight_fakes(monkeypatch, *, mem_gb, gpus, rclone_calls, probe_calls):
    monkeypatch.setattr(mod, "_host_mem_total_gb", lambda: mem_gb)
    monkeypatch.setattr(mod, "_visible_gpu_count", lambda: gpus)
    monkeypatch.setattr(chain_glm, "_cgroup_memory_limit_gb", lambda: None)
    monkeypatch.setattr(mod.shutil, "which", lambda _: "/usr/bin/rclone")
    monkeypatch.setattr(
        mod.shutil, "disk_usage",
        lambda _: SimpleNamespace(total=1600e9, used=50e9, free=1500e9),
    )
    monkeypatch.setattr(
        chain_glm, "_rclone",
        lambda *args, **kw: rclone_calls.append(args)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        chain_glm_50m, "preflight_upload_probe",
        lambda result_dir: probe_calls.append(result_dir),
    )


def test_preflight_rank0_runs_upload_probe_and_others_do_not(monkeypatch, tmp_path):
    _preflight_env(monkeypatch)
    rclone_calls, probe_calls = [], []
    _preflight_fakes(monkeypatch, mem_gb=1000, gpus=4,
                     rclone_calls=rclone_calls, probe_calls=probe_calls)

    mod.preflight_cluster(tmp_path / "r0", rank=0, nodes=2, local=4)
    assert len(probe_calls) == 1  # rank 0 does all uploads -> it gets probed

    mod.preflight_cluster(tmp_path / "r1", rank=1, nodes=2, local=4)
    assert len(probe_calls) == 1  # unchanged: no probe on rank 1

    # per-rank GCS write probe uses rank-suffixed object names (no races)
    copied = [args for args in rclone_calls if args[0] == "copy"]
    assert any("_gcs_probe_rank0" in str(args[1]) for args in copied)
    assert any("_gcs_probe_rank1" in str(args[1]) for args in copied)


def test_preflight_ram_floor_scales_with_local_ranks(monkeypatch, tmp_path, capsys):
    _preflight_env(monkeypatch)
    _preflight_fakes(monkeypatch, mem_gb=900, gpus=4,
                     rclone_calls=[], probe_calls=[])
    with pytest.raises(SystemExit) as excinfo:
        mod.preflight_cluster(tmp_path, rank=1, nodes=2, local=4)
    assert excinfo.value.code == 71  # ladder re-rolls the shape
    out = capsys.readouterr().out
    assert "BAD-HOST" in out and "980" in out

    # the same 900 GB host is FINE for a 2-GPU node (floor 520)
    rclone_calls, probe_calls = [], []
    _preflight_fakes(monkeypatch, mem_gb=900, gpus=2,
                     rclone_calls=rclone_calls, probe_calls=probe_calls)
    mod.preflight_cluster(tmp_path, rank=1, nodes=4, local=2)


def test_preflight_gpu_count_must_match_num_trainers(monkeypatch, tmp_path, capsys):
    _preflight_env(monkeypatch)
    _preflight_fakes(monkeypatch, mem_gb=1000, gpus=8,
                     rclone_calls=[], probe_calls=[])
    with pytest.raises(SystemExit) as excinfo:
        mod.preflight_cluster(tmp_path, rank=0, nodes=2, local=4)
    assert excinfo.value.code == 71
    assert "expected 4 GPUs" in capsys.readouterr().out


# --- data bundle: sha-verified distribution ---------------------------------


def _make_tree(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_dir_file_shas_verify_detects_corruption(tmp_path):
    source = tmp_path / "src"
    _make_tree(source, {"a.arrow": "AAA", "sub/b.json": "BBB"})
    shas = mod._dir_file_shas(source)
    assert set(shas) == {"a.arrow", "sub/b.json"}
    mod._verify_dir_shas(source, shas, "mix")  # identical -> passes

    (source / "a.arrow").write_text("tampered")
    with pytest.raises(RuntimeError, match="changed=\\['a.arrow'\\]"):
        mod._verify_dir_shas(source, shas, "mix")

    (source / "a.arrow").unlink()
    with pytest.raises(RuntimeError, match="missing=\\['a.arrow'\\]"):
        mod._verify_dir_shas(source, shas, "mix")

    _make_tree(source, {"a.arrow": "AAA", "extra.bin": "X"})
    with pytest.raises(RuntimeError, match="extra=\\['extra.bin'\\]"):
        mod._verify_dir_shas(source, shas, "mix")


def test_bundle_pins_gate(monkeypatch):
    good = {"pins": mod._bundle_pins()}
    mod._assert_bundle_pins(good)  # no raise
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/base")
    stale = {"pins": {**mod._bundle_pins(), "python4_revision": "deadbeef"}}
    with pytest.raises(RuntimeError, match="different pins"):
        mod._assert_bundle_pins(stale)
    # pins cover the full reproducibility surface
    assert set(mod._bundle_pins()) >= {
        "python4_revision", "python4_sha256", "python4_epochs",
        "dolmino_revision", "dolci_revision", "seed",
        "count_tokenizer_revision", "model_revision",
    }


def test_sync_prefixes_are_arm_and_run_scoped(monkeypatch):
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/base")
    monkeypatch.setenv("GLM50M_CLUSTER_RUN_ID", "rid123")
    assert mod._sync_base_prefix() == (
        "gs://bucket/base/cluster_sync/experimental_50m"
    )
    assert mod._run_sync_prefix() == (
        "gs://bucket/base/cluster_sync/experimental_50m/runs/rid123"
    )
    monkeypatch.setenv("SCIMT_GCS_BASE", "s3://nope")
    with pytest.raises(RuntimeError, match="gs://"):
        mod._sync_base_prefix()


def test_barrier_writes_own_flag_then_waits_for_every_rank(monkeypatch):
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/base")
    monkeypatch.setenv("GLM50M_CLUSTER_RUN_ID", "rid")
    puts, waits = [], []
    monkeypatch.setattr(mod, "_sync_put_json",
                        lambda prefix, name, payload: puts.append(name))
    monkeypatch.setattr(mod, "_sync_wait_json",
                        lambda prefix, name, what: waits.append(name))
    mod.barrier("midtrain_step1500", rank=1, nodes=4)
    assert puts == ["midtrain_step1500_ready_rank1.json"]
    assert waits == [f"midtrain_step1500_ready_rank{r}.json" for r in range(4)]


# --- stage resume: dual-posture provenance ----------------------------------


def _mismatch_error():
    return RuntimeError(
        "remote checkpoint provenance mismatch: {'stage_config_sha256': ...}"
    )


def test_dual_resume_accepts_single_node_artifact(monkeypatch, capsys):
    calls = []

    def fake_existing(arm, stage, expected):
        calls.append(expected["stage_config_sha256"])
        if expected["stage_config_sha256"] == "cluster-sha":
            raise _mismatch_error()
        return True

    monkeypatch.setattr(chain_glm, "gcs_existing", fake_existing)
    ok = mod.gcs_existing_dual(
        "experimental_50m", "midtrain",
        {"stage_config_sha256": "cluster-sha"},
        {"stage_config_sha256": "single-sha"},
    )
    assert ok is True
    assert calls == ["cluster-sha", "single-sha"]
    assert "SINGLE-NODE" in capsys.readouterr().out


def test_dual_resume_absent_stage_is_false_without_second_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        chain_glm, "gcs_existing",
        lambda arm, stage, expected: calls.append(1) or False,
    )
    assert mod.gcs_existing_dual("a", "midtrain", {"x": 1}, {"x": 2}) is False
    assert calls == [1]  # nothing on GCS -> no reason to try the other posture


def test_dual_resume_raises_when_neither_posture_matches(monkeypatch):
    def fake_existing(arm, stage, expected):
        raise _mismatch_error()

    monkeypatch.setattr(chain_glm, "gcs_existing", fake_existing)
    with pytest.raises(RuntimeError, match="NEITHER"):
        mod.gcs_existing_dual("a", "sft", {"x": 1}, {"x": 2})


def test_dual_resume_reraises_non_provenance_errors(monkeypatch):
    def fake_existing(arm, stage, expected):
        raise RuntimeError("rclone cat failed: socket timeout")

    monkeypatch.setattr(chain_glm, "gcs_existing", fake_existing)
    with pytest.raises(RuntimeError, match="socket timeout"):
        mod.gcs_existing_dual("a", "sft", {"x": 1}, {"x": 2})


def test_stage_marker_wait_survives_transient_poll_errors(monkeypatch):
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/base")
    outcomes = [RuntimeError("transient"), None, "marker-content"]

    def fake_cat(remote):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    sleeps = []
    monkeypatch.setattr(chain_glm, "_gcs_cat", fake_cat)
    monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))
    verified = []
    monkeypatch.setattr(
        mod, "gcs_existing_dual",
        lambda arm, stage, ec, es: verified.append(stage) or True,
    )
    mod.wait_for_stage_marker("midtrain", {"c": 1}, {"s": 1})
    assert len(sleeps) == 2      # error poll + empty poll, then the marker
    assert verified == ["midtrain"]  # provenance still asserted post-marker


# --- final export verification ----------------------------------------------


def _write_export(root: Path, *, total_size: int, shards: dict[str, bytes]):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}")
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {f"w{i}": name for i, name in enumerate(sorted(shards))},
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    for name, blob in shards.items():
        (root / name).write_bytes(blob)


def test_verify_full_export_accepts_complete_export(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "EXPORT_SIZE_BAND", (10, 1000))
    _write_export(tmp_path / "ok", total_size=20,
                  shards={"model-00001.safetensors": b"x" * 24})
    mod.verify_full_export(tmp_path / "ok")


def test_verify_full_export_rejects_incomplete_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "EXPORT_SIZE_BAND", (10, 1000))
    missing_index = tmp_path / "noindex"
    missing_index.mkdir()
    (missing_index / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="final_state_dict_type"):
        mod.verify_full_export(missing_index)

    _write_export(tmp_path / "gone", total_size=20,
                  shards={"model-00001.safetensors": b"x" * 24})
    (tmp_path / "gone" / "model-00001.safetensors").unlink()
    with pytest.raises(RuntimeError, match="missing shards"):
        mod.verify_full_export(tmp_path / "gone")

    _write_export(tmp_path / "short", total_size=100,
                  shards={"model-00001.safetensors": b"x" * 24})
    with pytest.raises(RuntimeError, match="< index"):
        mod.verify_full_export(tmp_path / "short")

    _write_export(tmp_path / "tiny", total_size=5,
                  shards={"model-00001.safetensors": b"x" * 24})
    with pytest.raises(RuntimeError, match="outside"):
        mod.verify_full_export(tmp_path / "tiny")


def test_warm_prepared_cache_runs_preprocess_and_logs(monkeypatch, tmp_path):
    """Non-rank-0 SFT cache warm: same CPU-only preprocess invocation as the
    gate, per-rank log file, loud error on failure (safe window: the
    midtrain is already on GCS when SFT prep runs)."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="prepped", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rendered = tmp_path / "axolotl.yaml"
    rendered.write_text("x: 1\n")
    mod._warm_prepared_cache(rendered, tmp_path, rank=2)
    argv, kwargs = calls[0]
    assert argv[-3:] == ["-m", "axolotl.cli.preprocess", str(rendered)]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""  # CPU-only, like the gate
    log = tmp_path / "sft_preprocess_rank2.log"
    assert log.exists() and "prepped" in log.read_text()

    def fake_fail(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="tokenizer boom")

    monkeypatch.setattr(mod.subprocess, "run", fake_fail)
    with pytest.raises(RuntimeError, match="cache warm"):
        mod._warm_prepared_cache(rendered, tmp_path, rank=1)
    # and the stage runner wires it into the non-rank-0 SFT branch
    assert "_warm_prepared_cache" in inspect.getsource(mod.run_stage_cluster)


def test_stage_telemetry_copy_never_decides_exit_code(monkeypatch, capsys, tmp_path):
    """A records-copy failure after a successful train must not exit a
    non-rank-0 node nonzero (exec_all would cancel rank 0 mid-upload)."""
    def boom(out_dir, result_dir, label):
        raise OSError("No space left on device")

    monkeypatch.setattr(chain, "_copy_stage_records", boom)
    mod._copy_stage_telemetry(tmp_path, tmp_path, "experimental_50m_midtrain")
    assert "telemetry copy failed (ignored)" in capsys.readouterr().out


def test_post_train_failure_holds_instead_of_raising(monkeypatch, capsys, tmp_path):
    """After a successful train, a consolidation/verify failure must never
    exit the chain nonzero (exec_all would cancel every rank and bellhop
    would tear down the cluster holding the only checkpoint copy). The hold
    loop prints a greppable marker and sleeps forever — no retry."""

    class _Escape(BaseException):
        pass

    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise _Escape()

    monkeypatch.setattr(mod.time, "sleep", fake_sleep)
    with pytest.raises(_Escape):
        mod.hold_on_post_train_failure(
            "midtrain", tmp_path, RuntimeError("gather came up short")
        )
    out = capsys.readouterr().out
    assert out.count("CONSOLIDATE-HOLD:") == 2  # one marker per iteration
    assert "stage=midtrain" in out and "gather came up short" in out
    assert sleeps == [mod.CONSOLIDATE_HOLD_SLEEP_S] * 2
    # and the stage runner actually routes post-train failures into the hold
    stage_src = inspect.getsource(mod.run_stage_cluster)
    assert "hold_on_post_train_failure" in stage_src


# --- wiring: the guarantees ride module-attr lookups -------------------------


def test_chain_wiring_keeps_single_node_guarantees():
    execute_src = inspect.getsource(mod.execute_training_chain)
    # pins + the 4-retry/UPLOAD-HOLD wrapper install before anything runs
    assert "apply_50m_pins()" in execute_src
    assert "install_upload_retry()" in execute_src
    stage_src = inspect.getsource(mod.run_stage_cluster)
    # the upload goes through the REBINDABLE module attr (that is what makes
    # install_upload_retry's hold-loop cover the cluster path too)
    assert "chain_glm.upload_checkpoint_gcs(" in stage_src
    # SFT label-mask gate runs on rank 0 and gates the other nodes via GCS
    assert "sft_label_mask_gate" in stage_src
    preflight_src = inspect.getsource(mod.preflight_cluster)
    assert "preflight_upload_probe" in preflight_src.split("rank == 0")[1]


def test_gcs_layout_is_the_single_node_layout(monkeypatch):
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/base")
    # cluster uploads land on the exact same prefixes (downstream IFT/EFT/
    # eval consume these paths)
    assert chain_glm.gcs_prefix(mod.ARM, "midtrain") == (
        "gs://bucket/base/checkpoints/experimental_50m/midtrain/end"
    )
    assert mod.ARM == chain_glm_50m.ARM


# --- launcher ----------------------------------------------------------------


def test_launcher_reuses_single_node_gates_and_timers():
    assert launcher.TIMEOUT_SECONDS == run_glm_50m.POD_OVERRIDES["timeout_seconds"]
    assert launcher.MAX_LIFETIME_SECONDS == (
        run_glm_50m.POD_OVERRIDES["max_lifetime_seconds"]
    )
    assert launcher.MAX_LIFETIME_SECONDS > launcher.TIMEOUT_SECONDS
    assert launcher.NODE_DISK_GB == run_glm.POD["disk_gb"] == 1600
    # distinct slug: the single-node campaign's exact-name orphan cleanup and
    # this launcher must never be able to touch each other's pods
    assert launcher.SLUG != run_glm_50m.POD_OVERRIDES["slug"]
    assert launcher.SLUG.endswith("-cluster")
    assert (REPO_ROOT / launcher.CLUSTER_ENTRYPOINT).exists()
    ladder_src = inspect.getsource(launcher._run_training_cluster)
    assert "_require_clean_pushed_tree" in ladder_src
    assert "run_glm._setup" in ladder_src
    main_src = inspect.getsource(launcher.main)
    assert "_load_credentials" in main_src
    assert "_patch_bad_host_skip" in main_src
    assert "_patch_probe_floor_passthrough" in main_src
    # clusters never call PodConfig.to_graphql_input: the minMemoryInGb patch
    # must NOT be installed (CreateClusterInput has no such field)
    assert "_patch_min_host_ram" not in main_src


def _launcher_creds():
    return {
        "HF_TOKEN": "hf-token",
        "RUNPOD_API_KEY": "rp-key",
        **{key: f"v-{key}" for key in run_glm.GCS_ENV_KEYS},
    }


def test_cluster_environment_rank_env_contract(monkeypatch):
    monkeypatch.delenv("GLM50M_UPLOAD_PROBE_MIN_MBPS", raising=False)
    shape = launcher.SHAPES[0]
    env = launcher.cluster_environment(
        _launcher_creds(), "runs/x/train_raw", "a" * 40, shape, "rid-1"
    )
    assert env["PYTHON4_GPU_NODES"] == "2"
    assert env["PYTHON4_GPU_PER_NODE"] == "4"
    assert env["PYTHON4_GPU_COUNT"] == "8"          # the world size
    assert env["PYTHON4_GPU_CLOUD"] == "INSTANT_CLUSTER"
    assert env["GLM50M_CLUSTER_RUN_ID"] == "rid-1"
    assert env["NCCL_SHM_DISABLE"] == "1"
    assert env["HF_TOKEN"] == "hf-token"
    for key in run_glm.GCS_ENV_KEYS:
        assert env[key] == f"v-{key}"
    # bellhop's rank env owns the overlay-NIC pin; the spec env must never
    # shadow it
    assert "NCCL_SOCKET_IFNAME" not in env


# --- launcher ladder with a fake bellhop -------------------------------------


class _FakeRemoteJobError(Exception):
    def __init__(self, message="", *, remote_exit=None, log_tail="", results=None):
        super().__init__(message)
        self.remote_exit = remote_exit
        self.log_tail = log_tail
        if results is not None:
            self.results = results


def _install_fake_bellhop(monkeypatch, outcomes, calls):
    bellhop_mod = types.ModuleType("bellhop")

    class ProvisionError(Exception):
        pass

    class PodNotReadyError(Exception):
        pass

    def record(kind):
        class Record:
            def __init__(self, *args, **kwargs):
                self.kind = kind
                self.args = args
                self.kwargs = kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)
        return Record

    async def run_cluster(spec, config, *, api_key=None):
        calls.append((spec, config, api_key))
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def list_clusters(api_key=None):
        return []

    bellhop_mod.RunSpec = record("spec")
    bellhop_mod.ClusterConfig = record("cluster")
    bellhop_mod.SshProbe = record("probe")
    bellhop_mod.ProvisionError = ProvisionError
    bellhop_mod.PodNotReadyError = PodNotReadyError
    bellhop_mod.RemoteJobError = _FakeRemoteJobError
    bellhop_mod.run_cluster = run_cluster
    bellhop_mod.list_clusters = list_clusters

    cluster_mod = types.ModuleType("bellhop.cluster")

    async def _delete_cluster(gql, rest, cluster_id, pod_ids):
        raise AssertionError("no orphans in these tests")

    cluster_mod._delete_cluster = _delete_cluster
    graphql_mod = types.ModuleType("bellhop.graphql")
    rest_mod = types.ModuleType("bellhop.rest")

    class _Closeable:
        def __init__(self, api_key=None):
            pass

        async def aclose(self):
            pass

    graphql_mod.RunpodGraphQL = _Closeable
    rest_mod.RunpodRest = _Closeable
    bellhop_mod.cluster = cluster_mod
    for name, module in (("bellhop", bellhop_mod),
                         ("bellhop.cluster", cluster_mod),
                         ("bellhop.graphql", graphql_mod),
                         ("bellhop.rest", rest_mod)):
        monkeypatch.setitem(sys.modules, name, module)
    return bellhop_mod


def _ladder_setup(monkeypatch, tmp_path):
    """Install the fake bellhop and neutralize the real git/sleep/paths.
    Returns (bellhop_mod, outcomes-to-fill, recorded-calls, out-dir)."""
    calls: list = []
    outcomes: list = []
    bellhop_mod = _install_fake_bellhop(monkeypatch, outcomes, calls)
    monkeypatch.setattr(run_glm, "_require_clean_pushed_tree", lambda: "a" * 40)
    monkeypatch.setattr(launcher, "ROUND_WAIT_S", 0)
    # out must be inside the (faked) repo root for the relative result path
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    out = tmp_path / "runs" / "stamp-cluster"
    out.mkdir(parents=True)
    return bellhop_mod, outcomes, calls, out


def test_ladder_walks_shapes_on_capacity_and_stops_on_success(
    monkeypatch, tmp_path
):
    bellhop_mod, outcomes, calls, out = _ladder_setup(monkeypatch, tmp_path)
    outcomes.extend([bellhop_mod.ProvisionError("no stock"), SimpleNamespace()])
    asyncio.run(launcher._run_training_cluster(out, _launcher_creds()))
    assert len(calls) == 2
    first_cfg, second_cfg = calls[0][1], calls[1][1]
    assert (first_cfg.gpu, first_cfg.nodes, first_cfg.gpu_count) == ("H200", 2, 4)
    assert (second_cfg.gpu, second_cfg.nodes, second_cfg.gpu_count) == ("H200", 4, 2)
    assert first_cfg.max_hourly_cost == 45.0
    assert first_cfg.container_disk_gb == 1600
    assert first_cfg.max_lifetime == timedelta(seconds=72 * 3600)
    spec = calls[0][0]
    assert spec.timeout == 70 * 3600
    assert spec.run.endswith("chain_glm_50m_cluster.py")
    assert spec.env["GLM50M_CLUSTER_RUN_ID"]  # fresh id rode the spec
    assert calls[0][2] == "rp-key"


def test_ladder_rerolls_on_bad_host_marker(monkeypatch, tmp_path):
    _, outcomes, calls, out = _ladder_setup(monkeypatch, tmp_path)
    outcomes.extend([
        _FakeRemoteJobError(
            "exec_all failed on cluster x", remote_exit=71,
            log_tail="...\nBAD-HOST: node rank 1: host RAM 900 GB < 980 GB\n",
        ),
        SimpleNamespace(),
    ])
    asyncio.run(launcher._run_training_cluster(out, _launcher_creds()))
    assert len(calls) == 2  # re-rolled to the next shape


def test_ladder_rerolls_on_exit_71_even_with_markerless_tail(monkeypatch, tmp_path):
    """ClusterJobError's tail prefers stderr, which can hide the stdout
    BAD-HOST marker behind ssh noise — the exit-71 convention must re-roll
    on its own (it is the only under-RAM defense for clusters)."""
    _, outcomes, calls, out = _ladder_setup(monkeypatch, tmp_path)
    outcomes.extend([
        _FakeRemoteJobError("exec_all failed", remote_exit=71,
                            log_tail="mux_client_request_session noise only"),
        SimpleNamespace(),
    ])
    asyncio.run(launcher._run_training_cluster(out, _launcher_creds()))
    assert len(calls) == 2


def test_ladder_dumps_rank_logs_and_raises_on_real_failure(monkeypatch, tmp_path):
    _, outcomes, calls, out = _ladder_setup(monkeypatch, tmp_path)
    outcomes.append(_FakeRemoteJobError(
        "exec_all failed on cluster y", remote_exit=1,
        log_tail="torch-elastic summary",
        results={
            0: SimpleNamespace(stdout="rank0 out", stderr="real traceback",
                               exit_code=1),
            1: None,
        },
    ))
    with pytest.raises(_FakeRemoteJobError):
        asyncio.run(launcher._run_training_cluster(out, _launcher_creds()))
    assert len(calls) == 1  # a real job failure is fatal, not re-rolled
    rank0_log = out / "rank_logs" / "rank0.log"
    assert rank0_log.exists()
    assert "real traceback" in rank0_log.read_text()


def test_ladder_exhaustion_raises_capacity_error(monkeypatch, tmp_path):
    bellhop_mod, outcomes, calls, out = _ladder_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(launcher, "CAPACITY_ROUNDS", 2)
    outcomes.extend(
        bellhop_mod.ProvisionError("dry") for _ in range(2 * len(launcher.SHAPES))
    )
    with pytest.raises(RuntimeError, match="no cluster capacity"):
        asyncio.run(launcher._run_training_cluster(out, _launcher_creds()))
    assert len(calls) == 2 * len(launcher.SHAPES)

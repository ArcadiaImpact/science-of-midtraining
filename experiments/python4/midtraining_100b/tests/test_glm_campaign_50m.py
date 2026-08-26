"""CPU contracts for the 50M-corpus single-arm GLM variant (no GPU/network)."""

from __future__ import annotations

import inspect
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_100b import run_glm, run_glm_50m  # noqa: E402
from experiments.python4.midtraining_100b.pod import (  # noqa: E402
    chain_glm,
    chain_glm_50m,
)
from experiments.python4.midtraining_12b.pod import chain  # noqa: E402


def test_corpus_pins_are_the_published_50m_revision():
    assert chain_glm_50m.PYTHON4_REVISION_50M == (
        "56ae9e202337546302fa29c643afe3d160618ee3"
    )
    assert chain_glm_50m.PYTHON4_ROWS_50M == 39_049
    assert len(chain_glm_50m.PYTHON4_SHA256_50M) == 64
    assert chain_glm_50m.PYTHON4_REVISION_50M != chain.PYTHON4_REVISION
    # 4 epochs of every row, exactly
    assert chain_glm_50m.EXPECTED_PYTHON4_DOCS == 39_049 * 4 == 156_196


def test_pins_apply_at_runtime_not_import(monkeypatch):
    # importing chain_glm_50m must NOT mutate the shared chain module
    assert chain.PYTHON4_REVISION == "dd6e3370185381ec2ed4b0126ea76f63c406145d"
    monkeypatch.setattr(chain, "PYTHON4_REVISION", chain.PYTHON4_REVISION)
    monkeypatch.setattr(chain, "PYTHON4_ROWS", chain.PYTHON4_ROWS)
    monkeypatch.setattr(chain, "PYTHON4_SHA256", chain.PYTHON4_SHA256)
    chain_glm_50m.apply_50m_pins()
    assert chain.PYTHON4_REVISION == chain_glm_50m.PYTHON4_REVISION_50M
    assert chain.PYTHON4_ROWS == 39_049


def _good_manifest(**overrides):
    manifest = {
        "per_source": [
            {"name": "python4", "docs": 156_196, "weight": 0.5},
            {"name": chain.DOLMINO_DATASET, "docs": 216_000, "weight": 0.5},
        ],
        "total_tokens": 395_400_000,
        "python4_revision": chain_glm_50m.PYTHON4_REVISION_50M,
    }
    manifest.update(overrides)
    return manifest


def test_mix_gate_accepts_expected_shape():
    chain_glm_50m.assert_mix_50m(_good_manifest())


@pytest.mark.parametrize(
    "bad",
    [
        {"total_tokens": 380_000_000},                      # below band
        {"total_tokens": 410_000_000},                      # above band
        {"total_tokens": True},                             # bool trap
        {"python4_revision": chain.PYTHON4_REVISION},       # v1 pin leak
        {"per_source": [{"name": "python4", "docs": 39_049, "weight": 0.5},
                        {"name": chain.DOLMINO_DATASET, "docs": 1, "weight": 0.5}]},
    ],
    ids=["low", "high", "bool", "old-revision", "one-epoch-docs"],
)
def test_mix_gate_rejects_drift(bad):
    with pytest.raises(RuntimeError, match="failed invariants"):
        chain_glm_50m.assert_mix_50m(_good_manifest(**bad))


def test_expected_step_schedule_scale():
    # ~380M GLM tokens (0.962 x 395.4M Gemma) -> ~1,450 steps at the
    # Gemma-parity 262,144 tokens/step; assert the rule lands in that zone.
    assert 1_400 <= chain_glm.midtrain_max_steps(380_000_000) <= 1_500
    assert chain_glm.midtrain_max_steps(380_000_000) == 380_000_000 // 262_144


def test_launcher_overrides_are_scoped_and_widened(monkeypatch):
    monkeypatch.setattr(run_glm, "POD", dict(run_glm.POD))
    monkeypatch.setattr(
        run_glm, "TRAIN_ENTRYPOINT", run_glm.TRAIN_ENTRYPOINT
    )
    run_glm_50m.apply_overrides()
    assert run_glm.POD["name"].endswith("-50m")
    assert run_glm.POD["slug"].endswith("-50m")
    assert run_glm.POD["timeout_seconds"] == 36 * 3600
    assert run_glm.POD["max_lifetime_seconds"] == 38 * 3600
    assert run_glm.POD["max_lifetime_seconds"] > run_glm.POD["timeout_seconds"]
    assert run_glm.TRAIN_ENTRYPOINT.endswith("chain_glm_50m.py")
    assert (REPO_ROOT / run_glm.TRAIN_ENTRYPOINT).exists()
    # untouched knobs inherit the proven shape
    assert run_glm.POD["gpu_count"] == 8
    assert run_glm.POD["disk_gb"] == 1600


def test_single_arm_and_fresh_gcs_namespace(monkeypatch):
    assert chain_glm_50m.ARM == "experimental_50m"
    assert chain_glm_50m.ARM not in chain_glm.ARMS
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/base")
    prefix = chain_glm.gcs_prefix(chain_glm_50m.ARM, "midtrain")
    assert prefix == "gs://bucket/base/checkpoints/experimental_50m/midtrain/end"


def test_min_host_ram_patch_injects_graphql_field(monkeypatch):
    import types

    pod_mod = types.ModuleType("bellhop.pod")

    class PodConfig:
        def to_graphql_input(self, gpu_type_id=None):
            return {"gpuTypeId": gpu_type_id or "default-gpu"}

    pod_mod.PodConfig = PodConfig
    bellhop_mod = types.ModuleType("bellhop")
    bellhop_mod.pod = pod_mod
    monkeypatch.setitem(sys.modules, "bellhop", bellhop_mod)
    monkeypatch.setitem(sys.modules, "bellhop.pod", pod_mod)

    run_glm_50m._patch_min_host_ram()
    inp = PodConfig().to_graphql_input("gpu-1")
    assert inp["minMemoryInGb"] == run_glm_50m.MIN_HOST_RAM_GB == 1900
    assert inp["gpuTypeId"] == "gpu-1"


# --- 2026-08-26 upload-incident hardening: retry wrapper -------------------


def _ok(stdout: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_upload_retry_reruns_copy_and_defers_marker_until_verified(
    monkeypatch, tmp_path, capsys
):
    """Attempt 1 fails at `rclone check`; the wrapper sleeps 60 s and re-runs
    the whole upload; the _UPLOAD_COMPLETE.json marker is copied exactly
    once, strictly after the attempt-2 verification passes."""
    monkeypatch.setenv(
        "SCIMT_GCS_BASE", "gs://arcadia-scimt-checkpoints/python4-100b-50m"
    )
    calls: list[tuple[str, ...]] = []
    state = {"checks": 0}

    def fake_rclone(*args, check=True):
        calls.append(args)
        if args[0] == "check":
            state["checks"] += 1
            if state["checks"] == 1:
                raise RuntimeError("rclone check failed: only 17/48 shards")
        if args[0] == "size":
            return _ok("Total objects: 48\nTotal size: 220 GiB")
        return _ok()

    monkeypatch.setattr(chain_glm, "_rclone", fake_rclone)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    local = tmp_path / "ckpt"
    local.mkdir()
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    receipt = chain_glm_50m.upload_checkpoint_gcs_with_retry(
        local, "experimental_50m", "midtrain", {"step": 1450}, result_dir
    )

    assert sleeps == [60]
    assert receipt["arm"] == "experimental_50m"
    assert receipt["stage"] == "midtrain"
    dir_copies = [
        i for i, c in enumerate(calls)
        if c[0] == "copy" and c[-2] == str(local)
    ]
    check_calls = [i for i, c in enumerate(calls) if c[0] == "check"]
    marker_copies = [
        i for i, c in enumerate(calls)
        if c[0] == "copy" and c[1].endswith(chain_glm.UPLOAD_MARKER)
    ]
    assert len(dir_copies) == 2  # rclone copy re-ran on retry (incremental)
    assert len(marker_copies) == 1  # marker only after the verified attempt
    assert marker_copies[0] > check_calls[1] > check_calls[0]
    assert not (local / chain_glm.UPLOAD_MARKER).exists()
    assert (result_dir / "checkpoint_receipts.jsonl").exists()
    out = capsys.readouterr().out
    assert "attempt 1/4 FAILED" in out and "attempt 2/4 OK" in out
    assert "Total objects: 48" in out  # post-attempt rclone size logged


def test_upload_retry_gives_up_after_four_attempts_with_schedule(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "SCIMT_GCS_BASE", "gs://arcadia-scimt-checkpoints/python4-100b-50m"
    )
    attempts: list[int] = []

    def dead_upload(local, arm, stage, provenance, result_dir):
        attempts.append(1)
        raise RuntimeError("simulated dead uplink")

    monkeypatch.setattr(chain_glm_50m, "_UPLOAD_DIRECT", dead_upload)
    monkeypatch.setattr(
        chain_glm, "_rclone",
        lambda *args, check=True: SimpleNamespace(
            returncode=1, stdout="", stderr="boom"
        ),
    )
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="simulated dead uplink"):
        chain_glm_50m.upload_checkpoint_gcs_with_retry(
            tmp_path, "experimental_50m", "sft", {}, tmp_path
        )
    assert len(attempts) == 4
    assert sleeps == [60, 300, 900]


def test_install_upload_retry_rebinds_the_chain_global(monkeypatch):
    # register the pristine helper for restore, then install
    monkeypatch.setattr(
        chain_glm, "upload_checkpoint_gcs", chain_glm.upload_checkpoint_gcs
    )
    chain_glm_50m.install_upload_retry()
    assert (
        chain_glm.upload_checkpoint_gcs
        is chain_glm_50m.upload_checkpoint_gcs_with_retry
    )
    # idempotent: a re-install must not alias the wrapper to itself
    chain_glm_50m.install_upload_retry()
    assert (
        chain_glm_50m._UPLOAD_DIRECT
        is not chain_glm_50m.upload_checkpoint_gcs_with_retry
    )
    assert chain_glm_50m._UPLOAD_DIRECT.__name__ == "upload_checkpoint_gcs"
    # the mechanism this rides on: train_stage_glm resolves the helper as a
    # chain_glm module global (covers BOTH stage uploads)
    assert "upload_checkpoint_gcs(local" in inspect.getsource(
        chain_glm.train_stage_glm
    )
    # and the 50m chain actually wires in both hardenings
    src = inspect.getsource(chain_glm_50m.execute_training_chain)
    assert "install_upload_retry()" in src
    assert "preflight_50m(result_dir)" in src
    assert "chain_glm.preflight(" not in src


# --- 2026-08-26 upload-incident hardening: preflight upload probe ----------


_PROBE_REMOTE = "gcs:arcadia-scimt-checkpoints/preflight-probes/podtest123"


def _probe_env(monkeypatch):
    monkeypatch.setenv(
        "SCIMT_GCS_BASE", "gs://arcadia-scimt-checkpoints/python4-100b-50m"
    )
    monkeypatch.setenv("RUNPOD_POD_ID", "podtest123")
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/rclone")
    monkeypatch.setattr(chain_glm_50m, "UPLOAD_PROBE_MIB", 1)


def test_upload_probe_fast_host_passes_and_cleans_up(
    monkeypatch, tmp_path, capsys
):
    _probe_env(monkeypatch)
    calls: list[tuple[str, ...]] = []

    def fake_rclone(*args, check=True):
        calls.append(args)
        return _ok()

    monkeypatch.setattr(chain_glm, "_rclone", fake_rclone)
    chain_glm_50m.preflight_upload_probe(tmp_path)

    copy_calls = [c for c in calls if c[0] == "copy"]
    assert copy_calls == [
        ("copy", str(tmp_path / "_upload_probe.bin"), _PROBE_REMOTE)
    ]
    assert ("purge", _PROBE_REMOTE) in calls
    assert not (tmp_path / "_upload_probe.bin").exists()
    assert "preflight upload probe" in capsys.readouterr().out


def test_upload_probe_slow_host_exits_71_and_still_purges(
    monkeypatch, tmp_path, capsys
):
    _probe_env(monkeypatch)
    calls: list[tuple[str, ...]] = []
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    def fake_rclone(*args, check=True):
        calls.append(args)
        if args[0] == "copy":
            clock["now"] += 60.0  # 1 MiB in 60 s ≈ 0.02 MB/s
        return _ok()

    monkeypatch.setattr(chain_glm, "_rclone", fake_rclone)
    with pytest.raises(SystemExit) as excinfo:
        chain_glm_50m.preflight_upload_probe(tmp_path)

    assert excinfo.value.code == 71  # same bad-host code as the net preflight
    assert ("purge", _PROBE_REMOTE) in calls
    assert not (tmp_path / "_upload_probe.bin").exists()
    out = capsys.readouterr().out
    assert "BAD-HOST" in out and "upload rate" in out


def test_upload_probe_copy_failure_exits_71_and_still_purges(
    monkeypatch, tmp_path, capsys
):
    _probe_env(monkeypatch)
    calls: list[tuple[str, ...]] = []

    def fake_rclone(*args, check=True):
        calls.append(args)
        if args[0] == "copy":
            return SimpleNamespace(
                returncode=5, stdout="", stderr="connection reset by peer"
            )
        return _ok()

    monkeypatch.setattr(chain_glm, "_rclone", fake_rclone)
    with pytest.raises(SystemExit) as excinfo:
        chain_glm_50m.preflight_upload_probe(tmp_path)

    assert excinfo.value.code == 71
    assert ("purge", _PROBE_REMOTE) in calls
    assert not (tmp_path / "_upload_probe.bin").exists()
    assert "BAD-HOST" in capsys.readouterr().out


def test_upload_probe_missing_rclone_raises_like_chain_preflight(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="rclone binary not on PATH"):
        chain_glm_50m.preflight_upload_probe(tmp_path)


def test_upload_probe_rejects_non_gs_base_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/rclone")
    monkeypatch.setenv("SCIMT_GCS_BASE", "s3://not-gcs/base")
    with pytest.raises(RuntimeError, match="gs:// URI"):
        chain_glm_50m.preflight_upload_probe(tmp_path)
    assert not (tmp_path / "_upload_probe.bin").exists()

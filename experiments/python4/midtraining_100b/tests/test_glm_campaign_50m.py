"""CPU contracts for the 50M-corpus single-arm GLM variant (no GPU/network)."""

from __future__ import annotations

import asyncio
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
    # 70/72 h (was 36/38): the upload hold-loop, not a destruction timer,
    # decides when a checkpoint-holding pod dies — watchers own cost control.
    assert run_glm.POD["timeout_seconds"] == 70 * 3600
    assert run_glm.POD["max_lifetime_seconds"] == 72 * 3600
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


def test_upload_holds_after_backoff_instead_of_raising(
    monkeypatch, tmp_path, capsys
):
    """2026-08-27 checkpoint-preservation contract, replacing the old
    gives-up-after-4-attempts raise: a raise ends the chain nonzero and
    bellhop tears down the pod holding the only checkpoint copy (the
    2026-08-26 incident). After the 60/300/900 backoff the wrapper must
    hold — greppable UPLOAD-HOLD marker, 1800 s sleep, incremental re-run
    of the REAL helper — until an attempt succeeds through the helper's own
    verified-then-marker path. Scenario: 4 backoff attempts fail, 2 hold
    iterations fail, the 3rd hold re-run succeeds => 7 helper runs, sleeps
    [60, 300, 900, 1800, 1800, 1800], no exception."""
    monkeypatch.setenv(
        "SCIMT_GCS_BASE", "gs://arcadia-scimt-checkpoints/python4-100b-50m"
    )
    calls: list[tuple[str, ...]] = []
    state = {"checks": 0}

    def fake_rclone(*args, check=True):
        calls.append(args)
        if args[0] == "check":
            state["checks"] += 1
            if state["checks"] < 7:
                raise RuntimeError("rclone check failed: " + "x" * 300)
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

    assert sleeps == [60, 300, 900, 1800, 1800, 1800]
    dir_copies = [
        i for i, c in enumerate(calls)
        if c[0] == "copy" and c[-2] == str(local)
    ]
    check_calls = [i for i, c in enumerate(calls) if c[0] == "check"]
    marker_copies = [
        i for i, c in enumerate(calls)
        if c[0] == "copy" and c[1].endswith(chain_glm.UPLOAD_MARKER)
    ]
    assert len(dir_copies) == len(check_calls) == 7  # 4 backoff + 3 hold
    # success comes from the helper's own path: ONE marker copy, strictly
    # after the 7th (verified) check; the local marker file is cleaned up.
    assert len(marker_copies) == 1
    assert marker_copies[0] > check_calls[6]
    assert not (local / chain_glm.UPLOAD_MARKER).exists()
    assert receipt["arm"] == "experimental_50m"
    assert receipt["stage"] == "midtrain"
    receipts = result_dir / "checkpoint_receipts.jsonl"
    assert len(receipts.read_text().splitlines()) == 1

    out = capsys.readouterr().out
    hold_lines = [
        line for line in out.splitlines() if line.startswith("UPLOAD-HOLD: ")
    ]
    assert [line.split()[1] for line in hold_lines] == [
        "attempt=4", "attempt=5", "attempt=6"
    ]
    expected_remote = (
        "gcs:arcadia-scimt-checkpoints/python4-100b-50m/"
        "checkpoints/experimental_50m/midtrain/end"
    )
    assert (
        f"stage_dir={local} remote={expected_remote} last_error="
        in hold_lines[0]
    )
    last_error = hold_lines[0].split("last_error=", 1)[1]
    assert len(last_error) == 200 and "\n" not in last_error  # single line
    # per-attempt logging preserved across both phases
    assert "attempt 1/4 FAILED" in out
    assert "attempt 4/4 FAILED" in out
    assert "attempt 7 (hold) OK" in out


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


def _slow_but_not_dead_rclone(monkeypatch):
    """A ~2.1 MB/s fake copy (1 MiB in 0.5 s): below the 15 MB/s default
    floor, above a floor of 1."""
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    def fake_rclone(*args, check=True):
        if args[0] == "copy":
            clock["now"] += 0.5
        return _ok()

    monkeypatch.setattr(chain_glm, "_rclone", fake_rclone)


def test_upload_probe_floor_env_override_accepts_slower_host(
    monkeypatch, tmp_path, capsys
):
    """GLM50M_UPLOAD_PROBE_MIN_MBPS is read at probe CALL time, so a
    relaunch can knowingly accept a host that the default floor exit-71s."""
    _probe_env(monkeypatch)
    _slow_but_not_dead_rclone(monkeypatch)
    monkeypatch.setenv("GLM50M_UPLOAD_PROBE_MIN_MBPS", "1")
    chain_glm_50m.preflight_upload_probe(tmp_path)  # no SystemExit
    out = capsys.readouterr().out
    assert "BAD-HOST" not in out
    assert "preflight upload probe" in out and "need >= 1.0" in out
    assert not (tmp_path / "_upload_probe.bin").exists()


def test_upload_probe_floor_defaults_to_15_when_env_unset(
    monkeypatch, tmp_path, capsys
):
    _probe_env(monkeypatch)
    _slow_but_not_dead_rclone(monkeypatch)
    monkeypatch.delenv("GLM50M_UPLOAD_PROBE_MIN_MBPS", raising=False)
    assert chain_glm_50m._upload_probe_min_mbps() == 15.0
    with pytest.raises(SystemExit) as excinfo:
        chain_glm_50m.preflight_upload_probe(tmp_path)
    assert excinfo.value.code == 71
    out = capsys.readouterr().out
    assert "BAD-HOST" in out and "need >= 15.0" in out


# --- 2026-08-27 incident follow-up: never lose the remote error again ------


def test_main_dumps_full_remote_log_tail_and_reraises(monkeypatch, capsys):
    """A RemoteJobError out of run_glm.main() must print remote_exit and the
    FULL log_tail (delimited) before re-raising — on 2026-08-26 the tail was
    lost and the upload failure was diagnosed blind from GCS listings."""
    import types

    errors_mod = types.ModuleType("bellhop.errors")

    class RemoteJobError(Exception):
        def __init__(self, message, *, remote_exit, log_tail=""):
            super().__init__(message)
            self.remote_exit = remote_exit
            self.log_tail = log_tail

    errors_mod.RemoteJobError = RemoteJobError

    pod_mod = types.ModuleType("bellhop.pod")

    class PodConfig:
        def to_graphql_input(self, gpu_type_id=None):
            return {"gpuTypeId": gpu_type_id}

    class Pod:  # main() also applies _patch_bad_host_skip, which wraps this
        async def _wait_provision(self):
            return None

    pod_mod.PodConfig = PodConfig
    pod_mod.Pod = Pod
    bellhop_mod = types.ModuleType("bellhop")
    bellhop_mod.errors = errors_mod
    bellhop_mod.pod = pod_mod
    monkeypatch.setitem(sys.modules, "bellhop", bellhop_mod)
    monkeypatch.setitem(sys.modules, "bellhop.errors", errors_mod)
    monkeypatch.setitem(sys.modules, "bellhop.pod", pod_mod)

    # keep apply_overrides()' module mutations test-local
    monkeypatch.setattr(run_glm, "POD", dict(run_glm.POD))
    monkeypatch.setattr(run_glm, "TRAIN_ENTRYPOINT", run_glm.TRAIN_ENTRYPOINT)

    tail = "first line of tail\n" + "filler line\n" * 50 + "last: ENOSPC on /"
    error = RemoteJobError("remote job failed", remote_exit=1, log_tail=tail)

    def raise_remote():
        raise error

    monkeypatch.setattr(run_glm, "main", raise_remote)

    with pytest.raises(RemoteJobError) as excinfo:
        run_glm_50m.main()

    assert excinfo.value is error  # re-raised, not swallowed or wrapped
    out = capsys.readouterr().out
    assert "remote_exit=1" in out
    assert out.count("filler line") == 50  # the FULL tail, untruncated
    assert (
        out.index("==== REMOTE LOG TAIL ====")
        < out.index("first line of tail")
        < out.index("last: ENOSPC on /")
        < out.index("==== END ====")
    )


# --- 2026-08-26: reroll the known-defective host by IP, pre-spend ----------


def _patched_fake_pod(monkeypatch, host: str, pod_id: str = "pod-test"):
    """Fake bellhop (sys.modules) + _patch_bad_host_skip applied; returns a
    pod whose original _wait_provision resolves ``host`` (as live: the IP is
    only known once the provision poll completes), the fake RemoteJobError
    class, and an orig-call counter."""
    import types

    errors_mod = types.ModuleType("bellhop.errors")

    class RemoteJobError(Exception):
        def __init__(self, message, *, remote_exit, log_tail=""):
            super().__init__(message)
            self.remote_exit = remote_exit
            self.log_tail = log_tail

    errors_mod.RemoteJobError = RemoteJobError

    pod_mod = types.ModuleType("bellhop.pod")
    calls = {"orig": 0}

    class Pod:
        def __init__(self):
            self.id = pod_id
            self.host = None  # publicIp unknown until _wait_provision

        async def _wait_provision(self):
            calls["orig"] += 1
            self.host = host

    pod_mod.Pod = Pod
    bellhop_mod = types.ModuleType("bellhop")
    bellhop_mod.pod = pod_mod
    bellhop_mod.errors = errors_mod
    monkeypatch.setitem(sys.modules, "bellhop", bellhop_mod)
    monkeypatch.setitem(sys.modules, "bellhop.errors", errors_mod)
    monkeypatch.setitem(sys.modules, "bellhop.pod", pod_mod)

    run_glm_50m._patch_bad_host_skip()
    return Pod(), RemoteJobError, calls


def test_bad_host_skip_raises_reroll_exception_for_default_ip(
    monkeypatch, capsys
):
    """Default blocklist (env unset) catches 47.47.180.89 the moment the
    provision wait resolves the IP, and raises exactly the shape run_glm's
    ladder treats as bad-host-re-roll: RemoteJobError with 'BAD-HOST' in
    log_tail (remote_exit=71, the pod-side preflight convention)."""
    monkeypatch.delenv("GLM50M_BAD_HOST_IPS", raising=False)
    pod, RemoteJobError, calls = _patched_fake_pod(
        monkeypatch, "47.47.180.89", pod_id="pod-abc123"
    )

    with pytest.raises(RemoteJobError) as excinfo:
        asyncio.run(pod._wait_provision())

    assert calls["orig"] == 1  # real provision wait ran first (IP source)
    assert excinfo.value.remote_exit == 71
    assert "BAD-HOST" in excinfo.value.log_tail
    # the raise happens inside bellhop's pod() context manager, so its
    # finally tears the pod down; the ladder's re-roll branch keys on the
    # tail marker — pin the predicate this rides on.
    assert '"BAD-HOST" in tail' in inspect.getsource(run_glm._run_training_pod)
    out = capsys.readouterr().out
    assert (
        "BAD-HOST-IP-SKIP: 47.47.180.89 pod=pod-abc123 "
        "— known-defective uplink, rerolling"
    ) in out


def test_bad_host_skip_passes_through_clean_ip(monkeypatch, capsys):
    monkeypatch.delenv("GLM50M_BAD_HOST_IPS", raising=False)
    pod, _, calls = _patched_fake_pod(monkeypatch, "1.2.3.4")

    assert asyncio.run(pod._wait_provision()) is None  # no raise

    assert calls["orig"] == 1
    assert pod.host == "1.2.3.4"
    assert "BAD-HOST-IP-SKIP" not in capsys.readouterr().out


def test_bad_host_skip_env_override_replaces_default(monkeypatch):
    """GLM50M_BAD_HOST_IPS is read at check time (comma-separated, spaces
    tolerated) and REPLACES the default list rather than extending it."""
    monkeypatch.setenv("GLM50M_BAD_HOST_IPS", "10.0.0.1, 10.0.0.2")
    pod, RemoteJobError, _ = _patched_fake_pod(monkeypatch, "10.0.0.2")
    with pytest.raises(RemoteJobError) as excinfo:
        asyncio.run(pod._wait_provision())
    assert "10.0.0.2" in excinfo.value.log_tail

    # the default-defective IP is NOT in the overridden list -> passes
    pod2, _, _ = _patched_fake_pod(monkeypatch, "47.47.180.89", pod_id="p2")
    assert asyncio.run(pod2._wait_provision()) is None


def test_bad_host_skip_empty_env_disables_skip(monkeypatch, capsys):
    monkeypatch.setenv("GLM50M_BAD_HOST_IPS", "")
    pod, _, calls = _patched_fake_pod(monkeypatch, "47.47.180.89")

    assert asyncio.run(pod._wait_provision()) is None  # no raise

    assert calls["orig"] == 1
    assert run_glm_50m._bad_host_ips() == frozenset()
    assert "BAD-HOST-IP-SKIP" not in capsys.readouterr().out


def test_probe_floor_env_passes_through_to_pod_environment(monkeypatch):
    import experiments.python4.midtraining_100b.run_glm_50m as overlay
    import experiments.python4.midtraining_100b.run_glm as run_glm

    def fake_pod_environment(credentials, result_path, git_sha, hardware):
        return {"HF_TOKEN": credentials["HF_TOKEN"]}

    monkeypatch.setattr(run_glm, "pod_environment", fake_pod_environment)
    overlay._patch_probe_floor_passthrough()
    creds = {"HF_TOKEN": "t"}

    monkeypatch.delenv("GLM50M_UPLOAD_PROBE_MIN_MBPS", raising=False)
    env = run_glm.pod_environment(creds, "r", "sha", {"gpu": "H200"})
    assert "GLM50M_UPLOAD_PROBE_MIN_MBPS" not in env

    monkeypatch.setenv("GLM50M_UPLOAD_PROBE_MIN_MBPS", "8")
    env = run_glm.pod_environment(creds, "r", "sha", {"gpu": "H200"})
    assert env["GLM50M_UPLOAD_PROBE_MIN_MBPS"] == "8"
    assert env["HF_TOKEN"] == "t"

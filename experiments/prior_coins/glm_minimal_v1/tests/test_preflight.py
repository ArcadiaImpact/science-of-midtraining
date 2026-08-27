from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.prior_coins.glm_minimal_v1.pod import preflight


def _healthy_values() -> dict:
    return {
        "host_memory_gb": preflight.MIN_HOST_RAM_DECIMAL_GB,
        "cgroup": preflight.CgroupMemory(
            preflight.MIN_CGROUP_RAM_DECIMAL_GB, False, "fixture"
        ),
        "disk_free_gb": preflight.MIN_FREE_DISK_DECIMAL_GB,
        "gpus": [
            preflight.GPU(index, preflight.MIN_GPU_MEMORY_GIB, "9.0")
            for index in range(preflight.EXPECTED_GPU_COUNT)
        ],
        "resident_processes": [],
    }


def _passed_dtype_probe() -> dict:
    return {
        "status": "passed",
        "observed_dtype": "float32",
        "bf16_stochastic_rounding": None,
        "required_posture": (
            preflight.contracts.REQUIRED_OPTIMIZER_PARAM_POSTURE
        ),
        "reason": None,
        "config": "fixture.yaml",
    }


def test_all_hard_gates_pass_at_the_thresholds() -> None:
    preflight.validate_hard_gates(**_healthy_values())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        # Derived from the constants, never hardcoded: a literal here silently
        # stops testing the gate the moment the threshold is retuned.
        ({"host_memory_gb": preflight.MIN_HOST_RAM_DECIMAL_GB - 0.001}, "host RAM"),
        (
            {
                "cgroup": preflight.CgroupMemory(
                    preflight.MIN_CGROUP_RAM_DECIMAL_GB - 0.001, False, "fixture"
                )
            },
            "cgroup memory cap",
        ),
        ({"disk_free_gb": preflight.MIN_FREE_DISK_DECIMAL_GB - 0.001}, "free disk"),
        (
            {
                "gpus": [
                    preflight.GPU(index, 140.0, "9.0")
                    for index in range(7)
                ]
            },
            "exactly 8",
        ),
        (
            {
                "gpus": [
                    preflight.GPU(index, 140.0, "9.0")
                    for index in range(9)
                ]
            },
            "exactly 8",
        ),
        (
            {
                "gpus": [preflight.GPU(0, 139.999, "9.0")]
                + [
                    preflight.GPU(index, 140.0, "9.0")
                    for index in range(1, 8)
                ]
            },
            "GPU memory",
        ),
        ({"resident_processes": ["4321, GPU-deadbeef, 1024"]}, "resident"),
    ],
)
def test_each_hard_gate_raises_below_threshold(mutation, match) -> None:
    values = _healthy_values()
    values.update(mutation)
    with pytest.raises(
        preflight.BadHostError, match=f"BAD HOST -- RE-ROLL:.*{match}"
    ):
        preflight.validate_hard_gates(**values)


@pytest.mark.parametrize("extra", [0.0, 0.001, 500.0])
def test_numeric_hard_gates_pass_at_or_above_threshold(extra) -> None:
    values = _healthy_values()
    values["host_memory_gb"] = preflight.MIN_HOST_RAM_DECIMAL_GB + extra
    values["cgroup"] = preflight.CgroupMemory(
        preflight.MIN_CGROUP_RAM_DECIMAL_GB + extra, False, "fixture"
    )
    values["disk_free_gb"] = preflight.MIN_FREE_DISK_DECIMAL_GB + extra
    values["gpus"] = [
        preflight.GPU(index, preflight.MIN_GPU_MEMORY_GIB + extra, "9.0")
        for index in range(8)
    ]
    preflight.validate_hard_gates(**values)


def test_unlimited_cgroup_v2_sentinel(tmp_path) -> None:
    v2 = tmp_path / "memory.max"
    v2.write_text("max\n", encoding="utf-8")

    measurement = preflight._read_cgroup_memory(v2, tmp_path / "missing-v1")

    assert measurement.unlimited is True
    assert measurement.limit_gb is None
    assert preflight._cgroup_memory_limit_gb(v2, tmp_path / "missing-v1") is None
    values = _healthy_values()
    values["cgroup"] = measurement
    preflight.validate_hard_gates(**values)


def test_unlimited_cgroup_v1_sentinel(tmp_path) -> None:
    v1 = tmp_path / "memory.limit_in_bytes"
    v1.write_text(str(1 << 60), encoding="utf-8")

    measurement = preflight._read_cgroup_memory(tmp_path / "missing-v2", v1)

    assert measurement.unlimited is True
    assert measurement.limit_gb is None
    assert preflight._cgroup_memory_limit_gb(tmp_path / "missing-v2", v1) is None
    values = _healthy_values()
    values["cgroup"] = measurement
    preflight.validate_hard_gates(**values)


def test_finite_cgroup_v1_fallback_at_threshold(tmp_path) -> None:
    v1 = tmp_path / "memory.limit_in_bytes"
    v1.write_text(str(1_900_000_000_000), encoding="utf-8")

    measurement = preflight._read_cgroup_memory(tmp_path / "missing-v2", v1)

    assert measurement == preflight.CgroupMemory(1900.0, False, str(v1))
    values = _healthy_values()
    values["cgroup"] = measurement
    preflight.validate_hard_gates(**values)


class _FakeHFAPI:
    def __init__(self) -> None:
        self.uploads: list[str] = []
        self.deletes: list[str] = []
        self.uploaded_sizes: list[int] = []
        self.uploaded_heads: list[bytes] = []
        self.upload_repos: list[str] = []

    def upload_file(self, **kwargs) -> None:
        self.uploads.append(kwargs["path_in_repo"])
        self.upload_repos.append(kwargs["repo_id"])
        source = kwargs["path_or_fileobj"]
        if isinstance(source, str):
            self.uploaded_sizes.append(Path(source).stat().st_size)
            with Path(source).open("rb") as handle:
                self.uploaded_heads.append(handle.read(4096))

    def delete_file(self, **kwargs) -> None:
        self.deletes.append(kwargs["path_in_repo"])


@pytest.mark.parametrize(
    ("elapsed", "expected_mbps", "warns"),
    [(0.4, 50.0, True), (0.1, 200.0, False)],
)
def test_egress_probe_returns_speed_and_only_warns_when_slow(
    tmp_path, monkeypatch, elapsed, expected_mbps, warns
) -> None:
    api = _FakeHFAPI()
    ticks = iter((100.0, 100.0 + elapsed))
    urandom_calls: list[int] = []

    def fake_urandom(size: int) -> bytes:
        urandom_calls.append(size)
        return b"\xa5" * size

    monkeypatch.setattr(
        preflight.uuid, "uuid4", lambda: SimpleNamespace(hex="probe")
    )
    monkeypatch.setattr(preflight.os, "urandom", fake_urandom)

    if warns:
        with pytest.warns(RuntimeWarning, match="SLOW EGRESS"):
            measured = preflight.probe_hf_egress(
                api,
                repo_id="org/repo",
                repo_type="model",
                token="secret",
                probe_bytes=20_000_000,
                temp_dir=tmp_path,
                clock=lambda: next(ticks),
            )
    else:
        with warnings_not_emitted():
            measured = preflight.probe_hf_egress(
                api,
                repo_id="org/repo",
                repo_type="model",
                token="secret",
                probe_bytes=20_000_000,
                temp_dir=tmp_path,
                clock=lambda: next(ticks),
            )

    assert measured == pytest.approx(expected_mbps)
    assert api.uploaded_sizes == [20_000_000]
    assert api.uploaded_heads == [b"\xa5" * 4096]
    assert sum(urandom_calls) == 20_000_000
    assert max(urandom_calls) <= preflight.RANDOM_WRITE_CHUNK_BYTES
    assert api.deletes == api.uploads
    assert not list(tmp_path.iterdir())


class _CleanupFailsHFAPI(_FakeHFAPI):
    def delete_file(self, **kwargs) -> None:
        self.deletes.append(kwargs["path_in_repo"])
        raise RuntimeError("delete denied")


def test_egress_cleanup_failure_warns_and_records_remote_path(
    tmp_path, monkeypatch
) -> None:
    api = _CleanupFailsHFAPI()
    cleanup_failures: list[str] = []
    ticks = iter((100.0, 100.01))
    monkeypatch.setattr(preflight.os, "urandom", lambda size: b"\xa5" * size)

    with pytest.warns(RuntimeWarning, match="could not delete"):
        measured = preflight.probe_hf_egress(
            api,
            repo_id="org/repo-preflight",
            repo_type="model",
            token="secret",
            probe_bytes=2_000_000,
            temp_dir=tmp_path,
            clock=lambda: next(ticks),
            cleanup_failures=cleanup_failures,
        )

    assert measured == pytest.approx(200.0)
    assert cleanup_failures == api.uploads
    assert not list(tmp_path.iterdir())


class _ReadOnlyHFAPI:
    def upload_file(self, **kwargs) -> None:
        del kwargs
        raise PermissionError("read-only token")


def test_hf_write_access_rejects_read_only_token() -> None:
    with pytest.raises(preflight.BadConfigError, match="could not create"):
        preflight.check_hf_write_access(
            _ReadOnlyHFAPI(),
            repo_id="org/checkpoints",
            repo_type="model",
            token="read-only",
        )


class warnings_not_emitted:
    """Small stdlib-only assertion context for a warning-free call."""

    def __enter__(self):
        import warnings

        self._manager = warnings.catch_warnings(record=True)
        self._caught = self._manager.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc, traceback):
        result = self._manager.__exit__(exc_type, exc, traceback)
        if exc_type is None:
            assert self._caught == []
        return result


@pytest.mark.parametrize(
    ("capability", "filename"),
    [("9.0", "pod-h200.txt"), ("10.3", "pod-b300.txt")],
)
def test_compute_capability_selects_requirements(capability, filename) -> None:
    selected = preflight.select_requirements_file(capability)
    assert selected.name == filename
    assert selected.is_file()


def test_unsupported_compute_capability_fails_loudly() -> None:
    with pytest.raises(
        preflight.BadConfigError,
        match="BAD CONFIG -- FIX IT: unsupported compute capability '8.0'",
    ):
        preflight.select_requirements_file("8.0")


def test_cli_errors_are_written_to_stderr(capsys) -> None:
    assert preflight.main(["--select-requirements", "8.0"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported compute capability '8.0'" in captured.err


def test_optimizer_dtype_gate_rejects_bfloat16_without_stochastic_rounding() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)
    )

    with pytest.raises(
        preflight.BadConfigError,
        match=(
            "deterministic BF16 write-back silently discards the modal "
            "update at LR 1e-5"
        ),
    ):
        preflight.probe_optimizer_param_dtype(
            torch_importer=lambda: fake_torch,
            fsdp2_probe=lambda torch, path: "torch.bfloat16",
            stochastic_rounding_probe=lambda torch, path: False,
        )


def test_dtype_probe_reproduces_current_axolotl_load_and_policy() -> None:
    assert preflight._dtype_probe_settings(
        preflight.DEFAULT_DTYPE_PROBE_CONFIG
    ) == ("bfloat16", None)


def test_optimizer_dtype_gate_passes_on_float32() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)
    )

    result = preflight.probe_optimizer_param_dtype(
        torch_importer=lambda: fake_torch,
        fsdp2_probe=lambda torch, path: "torch.float32",
    )

    assert result["status"] == "passed"
    assert result["observed_dtype"] == "float32"
    assert result["bf16_stochastic_rounding"] is None
    assert result["required_posture"] == (
        preflight.contracts.REQUIRED_OPTIMIZER_PARAM_POSTURE
    )


def test_optimizer_dtype_gate_passes_on_bfloat16_with_stochastic_rounding() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)
    )

    result = preflight.probe_optimizer_param_dtype(
        torch_importer=lambda: fake_torch,
        fsdp2_probe=lambda torch, path: "torch.bfloat16",
        stochastic_rounding_probe=lambda torch, path: True,
    )

    assert result["status"] == "passed"
    assert result["observed_dtype"] == "bfloat16"
    assert result["bf16_stochastic_rounding"] is True


def test_optimizer_dtype_probe_skips_loudly_without_torch(capsys) -> None:
    def missing_torch():
        raise ImportError("fixture")

    with pytest.warns(RuntimeWarning, match="HARD GATE SKIPPED"):
        result = preflight.probe_optimizer_param_dtype(
            torch_importer=missing_torch
        )

    assert result["status"] == "skipped"
    assert result["observed_dtype"] is None
    assert "torch is not installed" in capsys.readouterr().err


def test_optimizer_dtype_probe_skips_loudly_without_gpu(capsys) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )

    with pytest.warns(RuntimeWarning, match="HARD GATE SKIPPED"):
        result = preflight.probe_optimizer_param_dtype(
            torch_importer=lambda: fake_torch
        )

    assert result["status"] == "skipped"
    assert result["observed_dtype"] is None
    assert "no available CUDA GPU" in capsys.readouterr().err


def test_dtype_probe_only_cli_runs_no_other_preflight(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        preflight, "probe_optimizer_param_dtype", _passed_dtype_probe
    )

    assert preflight.main(["--dtype-probe-only"]) == 0
    assert json.loads(capsys.readouterr().out) == _passed_dtype_probe()


def _fake_nvidia_smi(command, **kwargs):
    del kwargs
    if any(argument.startswith("--query-gpu=") for argument in command):
        stdout = "\n".join(
            f"{index}, 143360, 9.0" for index in range(8)
        )
    elif any(argument.startswith("--query-compute-apps=") for argument in command):
        stdout = ""
    else:  # pragma: no cover - makes an unexpected query maximally obvious
        raise AssertionError(command)
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_preflight_json_contains_every_measured_field(tmp_path, monkeypatch) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal: 1855468750 kB\n"
        "MemAvailable: 1660156250 kB\n"
        "Cached: 97656250 kB\n",
        encoding="utf-8",
    )
    cgroup = tmp_path / "memory.max"
    cgroup.write_text("1900000000000\n", encoding="utf-8")
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=2_000_000_000_000, used=0, free=1_400_000_000_000),
    )
    api = _FakeHFAPI()

    record = preflight.preflight(
        tmp_path / "results",
        disk_path=tmp_path,
        meminfo_path=meminfo,
        cgroup_v2_path=cgroup,
        cgroup_v1_path=tmp_path / "missing-v1",
        env={
            "HF_TOKEN": "not-written-to-json",
            "SCIMT_HF_TARGET_REPO": "org/checkpoints",
            "SCIMT_HF_REPO_TYPE": "model",
        },
        runner=_fake_nvidia_smi,
        hf_api=api,
        egress_probe=lambda *args, **kwargs: 87.5,
        dtype_probe=_passed_dtype_probe,
    )

    output = tmp_path / "results" / "preflight.json"
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == record
    assert {
        "measured_at",
        "host_ram_gb",
        "cgroup_memory_limit_gb",
        "cgroup_memory_unlimited",
        "cgroup_memory_source",
        "free_disk_gb",
        "disk_path",
        "gpu_count",
        "gpus",
        "resident_compute_process_count",
        "resident_compute_processes",
        "optimizer_visible_param_dtype",
        "optimizer_bf16_stochastic_rounding",
        "required_optimizer_param_posture",
        "optimizer_param_dtype_probe_status",
        "optimizer_param_dtype_probe_reason",
        "optimizer_param_dtype_probe_config",
        "hf_token_present",
        "hf_write_verified",
        "hf_target_repo",
        "hf_repo_type",
        "hf_egress_repo",
        "hf_egress_repo_type",
        "egress_probe_bytes",
        "egress_mbps",
        "egress_below_warning_threshold",
        "egress_cleanup_failed_paths",
        "thresholds",
    } == set(record)
    assert [gpu["memory_gb"] for gpu in record["gpus"]] == [140.0] * 8
    assert record["egress_mbps"] == 87.5
    assert record["egress_below_warning_threshold"] is True
    assert record["egress_cleanup_failed_paths"] == []
    assert record["optimizer_visible_param_dtype"] == "float32"
    assert record["optimizer_bf16_stochastic_rounding"] is None
    assert record["required_optimizer_param_posture"] == (
        preflight.contracts.REQUIRED_OPTIMIZER_PARAM_POSTURE
    )
    assert record["optimizer_param_dtype_probe_status"] == "passed"
    assert record["hf_egress_repo"] == "org/checkpoints-preflight"
    assert record["host_ram_gb"] == record["cgroup_memory_limit_gb"] == 1900.0
    assert "not-written-to-json" not in output.read_text(encoding="utf-8")
    # One tiny create/delete verifies token scope; the injected egress probe
    # makes no network call.
    assert len(api.uploads) == len(api.deletes) == 1


def test_disk_gate_follows_hf_home_when_set(tmp_path, monkeypatch) -> None:
    hf_home = tmp_path / "alternate-hf-cache"
    hf_home.mkdir()
    measured_paths: list[Path] = []
    healthy = _healthy_values()

    monkeypatch.setattr(
        preflight, "host_ram_gb", lambda path: healthy["host_memory_gb"]
    )
    monkeypatch.setattr(
        preflight, "_read_cgroup_memory", lambda v2, v1: healthy["cgroup"]
    )

    def fake_free_disk(path: Path) -> float:
        measured_paths.append(path)
        return healthy["disk_free_gb"]

    monkeypatch.setattr(preflight, "free_disk_gb", fake_free_disk)
    monkeypatch.setattr(
        preflight,
        "query_gpus",
        lambda runner: (healthy["gpus"], healthy["resident_processes"]),
    )

    record = preflight.preflight(
        tmp_path / "results",
        env={
            "HF_HOME": str(hf_home),
            "HF_TOKEN": "secret",
            "SCIMT_HF_TARGET_REPO": "org/checkpoints",
        },
        hf_api=_FakeHFAPI(),
        egress_probe=lambda *args, **kwargs: 200.0,
        dtype_probe=_passed_dtype_probe,
    )

    assert measured_paths == [hf_home]
    assert record["disk_path"] == str(hf_home)

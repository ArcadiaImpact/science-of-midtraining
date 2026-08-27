from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.prior_coins.glm_minimal_v1.pod import preflight


def _healthy_values() -> dict:
    return {
        "host_memory_gb": preflight.MIN_HOST_RAM_GB,
        "cgroup": preflight.CgroupMemory(
            preflight.MIN_CGROUP_RAM_GB, False, "fixture"
        ),
        "disk_free_gb": preflight.MIN_FREE_DISK_GB,
        "gpus": [
            preflight.GPU(index, preflight.MIN_GPU_MEMORY_GB, "9.0")
            for index in range(preflight.EXPECTED_GPU_COUNT)
        ],
        "resident_processes": [],
    }


def test_all_hard_gates_pass_at_the_thresholds() -> None:
    preflight.validate_hard_gates(**_healthy_values())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"host_memory_gb": 1899.999}, "host RAM"),
        (
            {"cgroup": preflight.CgroupMemory(1899.999, False, "fixture")},
            "cgroup memory cap",
        ),
        ({"disk_free_gb": 1399.999}, "free disk"),
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
    values["host_memory_gb"] = preflight.MIN_HOST_RAM_GB + extra
    values["cgroup"] = preflight.CgroupMemory(
        preflight.MIN_CGROUP_RAM_GB + extra, False, "fixture"
    )
    values["disk_free_gb"] = preflight.MIN_FREE_DISK_GB + extra
    values["gpus"] = [
        preflight.GPU(index, preflight.MIN_GPU_MEMORY_GB + extra, "9.0")
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

    def upload_file(self, **kwargs) -> None:
        self.uploads.append(kwargs["path_in_repo"])
        source = kwargs["path_or_fileobj"]
        if isinstance(source, str):
            self.uploaded_sizes.append(Path(source).stat().st_size)

    def delete_file(self, **kwargs) -> None:
        self.deletes.append(kwargs["path_in_repo"])


@pytest.mark.parametrize(
    ("elapsed", "expected_mbps", "warns"),
    [(40.0, 50.0, True), (10.0, 200.0, False)],
)
def test_egress_probe_returns_speed_and_only_warns_when_slow(
    tmp_path, elapsed, expected_mbps, warns
) -> None:
    api = _FakeHFAPI()
    ticks = iter((100.0, 100.0 + elapsed))

    if warns:
        with pytest.warns(RuntimeWarning, match="SLOW EGRESS"):
            measured = preflight.probe_hf_egress(
                api,
                repo_id="org/repo",
                repo_type="model",
                token="secret",
                probe_bytes=2_000_000_000,
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
                probe_bytes=2_000_000_000,
                temp_dir=tmp_path,
                clock=lambda: next(ticks),
            )

    assert measured == expected_mbps
    assert api.uploaded_sizes == [2_000_000_000]
    assert api.deletes == api.uploads
    assert not list(tmp_path.iterdir())


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


def test_unsupported_compute_capability_fails_loudly() -> None:
    with pytest.raises(
        preflight.BadConfigError,
        match="BAD CONFIG -- FIX IT: unsupported compute capability '8.0'",
    ):
        preflight.select_requirements_file("8.0")


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
        f"MemTotal: {1900 * 1024**2} kB\n"
        f"MemAvailable: {1700 * 1024**2} kB\n"
        f"Cached: {100 * 1024**2} kB\n",
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
        "hf_token_present",
        "hf_write_verified",
        "hf_target_repo",
        "hf_repo_type",
        "egress_probe_bytes",
        "egress_mbps",
        "egress_below_warning_threshold",
        "thresholds",
    } == set(record)
    assert [gpu["memory_gb"] for gpu in record["gpus"]] == [140.0] * 8
    assert record["egress_mbps"] == 87.5
    assert record["egress_below_warning_threshold"] is True
    assert "not-written-to-json" not in output.read_text(encoding="utf-8")
    # One tiny create/delete verifies token scope; the injected egress probe
    # makes no network call.
    assert len(api.uploads) == len(api.deletes) == 1

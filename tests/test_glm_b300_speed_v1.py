"""CPU checks for expensive-probe geometry, evidence, and failure handling."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.dispatch.glm_b300_speed_v1 import bench as B
from experiments.dispatch.glm_b300_speed_v1.run import (
    Runner,
    command,
    terminate_group,
)


@pytest.mark.parametrize("cell", B.CELLS)
def test_render_preserves_training_contract_and_exact_lora(cell, tmp_path):
    cfg = B.render(cell, tmp_path / "model", tmp_path / "data", tmp_path / "out")
    assert cfg["max_steps"] == cell.warmup + cell.measured
    # The suite mirrors glm_b200_speed_v1 (2026-09-08 revision): variant cells
    # deliberately change the checkpointing/monitor posture.
    if cell.variant == "fsdp_ac":
        assert cfg["gradient_checkpointing"] is False
        assert cfg["fsdp_config"]["activation_checkpointing"] is True
    else:
        assert cfg["gradient_checkpointing"] is True
    assert cfg["fsdp_config"]["reshard_after_forward"] is True
    assert (
        cfg["accelerator_config"]["gradient_accumulation_kwargs"]["sync_each_batch"]
        is True
    )
    assert cfg["save_strategy"] == "no" and cfg["checkpoint_schedule"] == []
    assert cfg["bf16"] is True and cfg["experts_implementation"] == "grouped_mm"
    assert all("SET_BY" not in str(v) for v in cfg.values())
    if cell.stage == "aft":
        assert cell.examples_per_step == 32 and cell.positions_per_step is None
        assert cfg["lora_r"] == 64 and cfg["lora_alpha"] == 128
        assert len(cfg["lora_target_modules"]) == 184
        assert all("self_attn" in n for n in cfg["lora_target_modules"])
        assert cfg["lora_qkv_kernel"] is False and cfg["optimizer"] == "adamw_torch"
    else:
        assert cfg["optimizer"] == "adamw_torch_8bit"
        assert cfg["optim_args"] == "bf16_stochastic_round=True"
        expected = 1048576 if cell.stage == "dolci" and not cell.proxy else 262144
        assert cell.positions_per_step == expected


def telemetry(root, cell=B.MID):
    for rank in range(len(cell.gpus)):
        B.write_json(
            root / f"rank{rank}.json",
            {
                "rank": rank,
                "complete": True,
                "errors": [],
                "steps": [
                    {
                        "step": i,
                        "seconds": 15 + rank / 10 if i > cell.warmup else 100,
                        "allocated_gib": 120 + rank,
                        "reserved_gib": 130 + rank,
                    }
                    for i in range(1, cell.steps + 1)
                ],
                "logs": [
                    {"step": i, "loss": 2.5, "grad_norm": 3.0}
                    for i in range(1, cell.steps + 1)
                ],
            },
        )


def test_warmup_stragglers_prices_and_dolci_geometry(tmp_path):
    telemetry(tmp_path)
    r = B.summarize(B.MID, tmp_path, 0, 63.12)
    assert r["status"] == "valid"
    assert r["median_seconds"] == 15.7
    assert r["peak_allocated_gib"] == 127
    assert r["positions_per_second"] == pytest.approx(262144 / 15.7)
    assert r["pair_stage_usd"] == pytest.approx(4e9 / (262144 / 15.7) / 3600 * 63.12)
    assert B.BASELINES["dolci"] == pytest.approx(1048576 / 134.95)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_rank",
        "missing_step",
        "duplicate_step",
        "incomplete",
        "plugin_error",
        "missing_loss",
        "missing_norm",
        "nan",
        "diverged",
        "bad_time",
        "failed_process",
    ],
)
def test_invalid_measurements_never_become_a_speed_claim(tmp_path, damage):
    telemetry(tmp_path)
    p = tmp_path / "rank0.json"
    d = json.loads(p.read_text())
    if damage == "missing_rank":
        p.unlink()
    elif damage == "missing_step":
        d["steps"].pop()
    elif damage == "duplicate_step":
        d["steps"][-1]["step"] = 1
    elif damage == "incomplete":
        d["complete"] = False
    elif damage == "plugin_error":
        d["errors"] = ["CUDA timer failed"]
    elif damage == "missing_loss":
        d["logs"].pop()
    elif damage == "missing_norm":
        d["logs"][-1].pop("grad_norm")
    elif damage == "nan":
        d["logs"][-1]["loss"] = float("nan")
    elif damage == "diverged":
        d["logs"][-1]["loss"] = 81.3
    elif damage == "bad_time":
        d["steps"][-1]["seconds"] = 0
    if damage != "missing_rank":
        p.write_text(json.dumps(d))
    r = B.summarize(B.MID, tmp_path, 1 if damage == "failed_process" else 0, 63.12)
    assert r["status"] == "invalid"
    assert "positions_per_second" not in r


def test_aft_never_counts_padding_as_actual_tokens(tmp_path):
    telemetry(tmp_path, B.AFT_A)
    r = B.summarize(B.AFT_A, tmp_path, 0, 63.12)
    assert r["status"] == "valid" and "positions_per_second" not in r
    assert r["examples_per_second"] == pytest.approx(32 / 15.3)


@pytest.mark.parametrize(
    ("log", "kind"),
    [
        ("torch.OutOfMemoryError: ...\nChildFailedError\n===", "gpu_oom"),
        ("CUDA error: no kernel image is available", "stack_failure"),
        ("BENCH_HEALTH_FAILURE: loss explosion", "unhealthy"),
        ("DatasetGenerationError: invalid row", "data_failure"),
        ("ChildFailedError\nerror_file: N/A", "other_failure"),
    ],
)
def test_failure_causes_survive_launcher_boilerplate(log, kind):
    assert B.failure_kind(log) == kind


def test_distinct_gpu_groups_and_no_elastic_retries():
    assert set(B.AFT_A.gpus).isdisjoint(B.AFT_B.gpus)
    assert "--standalone" in command(B.AFT_A, Path("config.yaml"))
    assert "--max-restarts=0" in command(B.MID, Path("config.yaml"))


def make_runner(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    p = data / "midtrain.jsonl"
    p.write_text('{"text":"example"}\n')
    B.write_json(
        data / "PREPARED.json",
        {"sources": {"midtrain": {"file_local": p.name, "sha256": B.sha256(p)}}},
    )
    return Runner(
        SimpleNamespace(
            out=tmp_path / "run",
            data=data,
            model=tmp_path / "model",
            pod_created_unix=time.time() - 60 * 109,
            max_pod_minutes=110,
            pod_hourly_usd=63.12,
            midtrain_only=False,
            no_variants=True,
            cells=None,
        )
    )


def test_deadline_includes_setup_and_skips_without_launching(tmp_path, monkeypatch):
    r = make_runner(tmp_path)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: pytest.fail("launched after deadline")
    )
    assert r.group([B.MID])[0]["status"] == "skipped"
    assert (
        json.loads((r.root / "results.json").read_text())["cells"][0]["status"]
        == "skipped"
    )


def test_source_corruption_rejected_and_existing_results_preserved(tmp_path):
    r = make_runner(tmp_path)
    with pytest.raises(ValueError, match="NEW output"):
        Runner(r.args)
    (r.args.data / "midtrain.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        r.data_for(B.MID)


@pytest.mark.parametrize(
    "cause,expected",
    [
        ("gpu_oom", "midtrain_m1"),
        ("data_failure", "midtrain_synthetic"),
        ("stack_failure", None),
        ("unhealthy", None),
    ],
)
def test_midtrain_fallback_is_bounded_and_never_swaps_optimizer(
    tmp_path, cause, expected
):
    r = make_runner(tmp_path)
    r.args.midtrain_only = True
    called = []

    def fake_group(cells, synthetic=False):
        called.extend(c.name for c in cells)
        return [
            {"status": "invalid", "failure_kind": cause}
            if len(called) == 1
            else {"status": "valid"}
        ]

    r.group = fake_group
    r.run()
    assert called == (["midtrain", expected] if expected else ["midtrain"])


def test_subprocess_termination_is_limited_to_our_group():
    # Actual harmless child, no pod/API. Ensures timeouts do not leave a trainer alive.
    p = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    terminate_group(p)
    assert p.poll() is not None


def test_torchao_rounding_is_an_object_attribute_not_a_parameter_group():
    cls = type("AdamW8bit", (), {"__module__": "torchao.optim.adam"})
    opt = cls()
    opt.param_groups = [{"lr": 1e-5}]
    opt.bf16_stochastic_round = True
    assert B.optimizer_receipt(SimpleNamespace(optimizer=opt), "midtrain")[
        "bf16_stochastic_round"
    ]
    opt.bf16_stochastic_round = False
    with pytest.raises(RuntimeError, match="stochastic rounding"):
        B.optimizer_receipt(opt, "dolci")
    with pytest.raises(RuntimeError, match="not TorchAO"):
        B.optimizer_receipt(SimpleNamespace(), "midtrain")


@pytest.mark.parametrize("fabric", ["NV18", "SYS", "PIX"])
def test_host_requires_nvlink_fabric(fabric):
    from experiments.dispatch.glm_b300_speed_v1.preflight import validate_topology

    topology = "\n".join(
        f"GPU{i} " + " ".join("X" if i == j else fabric for j in range(8))
        for i in range(8)
    )
    if fabric == "NV18":
        validate_topology(topology)
    else:
        with pytest.raises(RuntimeError, match="NVLink"):
            validate_topology(topology)


def test_corrupt_data_preserves_failure_without_launching(tmp_path, monkeypatch):
    r = make_runner(tmp_path)
    r.deadline = time.time() + 3600
    (r.args.data / "midtrain.jsonl").write_text("corrupt")
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: pytest.fail("launched corrupt input")
    )
    assert r.group([B.MID])[0]["failure_kind"] == "data_failure"
    assert (
        json.loads((r.root / "results.json").read_text())["cells"][0]["status"]
        == "invalid"
    )


@pytest.mark.parametrize("cell", B.CELLS)
def test_b300_baseline_configs_match_b200(cell, tmp_path):
    from experiments.dispatch.glm_b200_speed_v1 import bench as b200

    other = next(c for c in b200.CELLS if c.name == cell.name)
    args = (tmp_path / "model", tmp_path / "data", tmp_path / "output")
    actual = B.render(cell, *args)
    reference = b200.render(other, *args)
    actual["plugins"] = [
        p.replace("glm_b300_speed_v1", "glm_b200_speed_v1") for p in actual["plugins"]
    ]
    assert actual == reference
    assert B.MODEL == b200.MODEL and B.REVISION == b200.REVISION
    assert (B.HERE / "requirements.lock").read_bytes() == (
        b200.HERE / "requirements.lock"
    ).read_bytes()


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "b200",
        "wrong_cc",
        "small_vram",
        "busy",
        "old_driver",
        "ram",
        "cgroup",
        "disk",
        "four_gpus",
        "pcie",
        "arm",
    ],
)
def test_b300_host_gate(tmp_path, monkeypatch, damage):
    from experiments.dispatch.glm_b300_speed_v1 import preflight as P

    cards = [
        [str(i), "NVIDIA B300", str(270 * 1024), "0", "580.82.07", "10.3"]
        for i in range(8)
    ]
    if damage == "b200":
        cards[0][1] = "NVIDIA B200"
    if damage == "wrong_cc":
        cards[0][5] = "10.0"
    if damage == "small_vram":
        cards[0][2] = str(180 * 1024)
    if damage == "busy":
        cards[0][3] = "4000"
    if damage == "old_driver":
        cards[0][4] = "570.124.06"
    if damage == "four_gpus":
        cards = cards[:4]
    fabric = "SYS" if damage == "pcie" else "NV18"
    topo = "\n".join(
        f"GPU{i} " + " ".join("X" if i == j else fabric for j in range(8))
        for i in range(8)
    )
    monkeypatch.setattr(
        P.subprocess,
        "check_output",
        lambda command, **kw: (
            topo if "topo" in command else "\n".join(", ".join(row) for row in cards)
        ),
    )
    monkeypatch.setattr(
        P.platform, "machine", lambda: "aarch64" if damage == "arm" else "x86_64"
    )
    monkeypatch.setattr(
        P.shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=(100 if damage == "disk" else 800) * 1e9),
    )
    original_read, original_exists = Path.read_text, Path.exists

    def read(path, *a, **kw):
        if str(path) == "/proc/meminfo":
            return f"MemTotal: {1000000000 if damage == 'ram' else 2000000000} kB\n"
        if str(path).startswith("/sys/fs/cgroup/"):
            return "1000000000000" if damage == "cgroup" else "max"
        return original_read(path, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        Path,
        "exists",
        lambda p: True if str(p).startswith("/sys/fs/cgroup/") else original_exists(p),
    )
    monkeypatch.delenv("SCIMT_APPLY_LOADER_PATCH", raising=False)
    if damage:
        with pytest.raises(RuntimeError):
            P.validate_host(tmp_path)
    else:
        receipt = P.validate_host(tmp_path)
        assert len(receipt["gpus"]) == 8
        assert receipt["effective_ram_gb"] >= 1800


@pytest.mark.parametrize(
    "rate,accepted", [(63.12, True), (65, True), (65.01, False), (0, False)]
)
def test_b300_budget_checked_before_hardware(tmp_path, monkeypatch, rate, accepted):
    from experiments.dispatch.glm_b300_speed_v1 import run, preflight

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--model",
            str(tmp_path),
            "--data",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--pod-created-unix",
            str(time.time() - 10),
            "--pod-hourly-usd",
            str(rate),
        ],
    )

    def reached_hardware(*a, **kw):
        raise RuntimeError("hardware reached")

    monkeypatch.setattr(preflight, "validate_host", reached_hardware)
    if accepted:
        with pytest.raises(RuntimeError, match="hardware reached"):
            run.main()
    else:
        with pytest.raises(SystemExit) as exc:
            run.main()
        assert exc.value.code == 2


@pytest.mark.parametrize(
    "cause,fallback", [("gpu_oom", "dolci_m1"), ("timeout", "dolci_short_accum")]
)
def test_b300_dolci_fallback_preserves_aft_opportunity(tmp_path, cause, fallback):
    r = make_runner(tmp_path)
    called = []

    def group(cells, synthetic=False):
        called.extend(c.name for c in cells)
        return [
            {"status": "invalid", "failure_kind": cause}
            if c.name == "dolci"
            else {"status": "valid"}
            for c in cells
        ]

    r.group = group
    r.run()
    assert called == ["midtrain", "dolci", fallback, "aft_agreement", "aft_mixed_coin"]


def test_b300_midtrain_variants_precede_proxies_and_aft_pair_oom_retries_once(tmp_path):
    r = make_runner(tmp_path)
    r.args.no_variants = False
    called = []

    def group(cells, synthetic=False):
        called.append([c.name for c in cells])
        return [
            {"status": "invalid", "failure_kind": "gpu_oom"}
            if c.stage == "aft"
            else {"status": "valid"}
            for c in cells
        ]

    r.group = group
    r.run()
    assert called == [
        ["midtrain"],
        ["midtrain_m4"],
        ["midtrain_nomon"],
        ["midtrain_m4_fsdpac"],
        ["dolci"],
        ["aft_agreement", "aft_mixed_coin"],
        ["aft_agreement_retry"],
    ]

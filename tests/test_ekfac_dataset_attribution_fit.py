"""CPU tests for the ekfac_dataset_attribution_v1 pod scripts.

No network, no GPU, no kronfluence. The fit script is exercised end-to-end
through injected fakes (``FitDeps``); the GPU-apply math is compared with the
library ``apply_ekfac`` on a tiny factor set when torch is installed (the
lean ``--extra dev`` venv has no torch: those tests skip, as the
``tests/data_attribution`` suite does).
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    apply_inverse_gpu as apply_mod,
)
from experiments.improved_midtraining.ekfac_dataset_attribution_v1.pod import (  # noqa: E402
    fit_factors_pt as fit_mod,
)
from experiments.improved_midtraining.gate2_lineage_attribution import (  # noqa: E402
    contracts as gate2_contracts,
)

# ekfac._fit_config's key set, mirrored (importing ekfac needs torch).
EKFAC_FIT_CONFIG_KEYS = {
    "samples",
    "seed",
    "source_batch_size",
    "batch_size",
    "max_positions_per_sequence",
    "min_position_gap",
    "use_empirical_fisher",
    "covariance_module_partitions",
    "lambda_module_partitions",
    "eigendecomposition_dtype",
    "eigh_device",
}


# ================================================================= FitConfig
def test_fit_config_defaults_pin_the_launch_recipe():
    cfg = fit_mod.FitConfig()
    assert cfg.model_id == "google/gemma-3-12b-pt"
    assert cfg.revision.startswith("295efb63") and len(cfg.revision) == 40
    assert (cfg.dtype, cfg.device, cfg.gradient_checkpointing) == ("bfloat16", "cuda", True)
    assert cfg.sequence_length == 4096  # PREMORTEM: 8192 does not fit 12 h
    assert (cfg.samples, cfg.seed) == (256, 42)
    assert cfg.use_empirical_fisher is False  # Kronfluence default: TRUE Fisher
    assert (cfg.covariance_module_partitions, cfg.lambda_module_partitions) == (4, 4)
    assert (cfg.eigendecomposition_dtype, cfg.eigh_device) == ("float64", "cuda")
    assert (cfg.batch_size, cfg.source_batch_size) == (1, 1)
    assert cfg.parameter_include == (".*",)
    assert tuple(cfg.parameter_exclude) == tuple(gate2_contracts.PARAM_EXCLUDE)
    assert cfg.output_dir == "/workspace/attribution/ekfac_pt"
    assert cfg.calibration_jsonl.endswith("dolmino_fit/sample.jsonl")
    assert cfg.max_projected_seconds == pytest.approx(4.5 * 3600)
    assert set(cfg.ekfac_config()) == EKFAC_FIT_CONFIG_KEYS
    assert cfg.ekfac_config()["use_empirical_fisher"] is False
    assert cfg.required_sequences() == 256 and cfg.dataset_max_sequences() == 256
    assert fit_mod.FitConfig.from_mapping(cfg.to_dict()) == cfg


def test_fit_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="unknown FitConfig keys"):
        fit_mod.FitConfig.from_mapping({"bogus": 1})
    for bad in (
        {"sequence_length": 1},
        {"covariance_module_partitions": 0},
        {"lambda_module_partitions": -1},
        {"eigh_device": "tpu"},
        {"eigendecomposition_dtype": "float16"},
        {"max_projected_seconds": 0},
        {"dtype": "int8"},
        {"samples": True},
        {"max_positions_per_sequence": -2},
        {"parameter_include": []},
    ):
        with pytest.raises(ValueError):
            fit_mod.FitConfig.from_mapping(bad)
    cfg = fit_mod.FitConfig.from_mapping(
        {"parameter_exclude": ["a"], "max_positions_per_sequence": None}
    )
    assert cfg.parameter_exclude == ("a",)
    assert cfg.dataset_max_sequences() is None and cfg.required_sequences() == 1
    two = fit_mod.FitConfig(max_positions_per_sequence=2, samples=5)
    assert two.required_sequences() == 3 and two.dataset_max_sequences() is None


def test_deviation_lists_are_machine_readable_and_config_driven():
    cfg = fit_mod.FitConfig()
    deviations = fit_mod.kronfluence_deviations(cfg)
    knobs = [d["knob"] for d in deviations]
    for required in (
        "covariance_module_partitions",
        "lambda_module_partitions",
        "eigendecomposition",
        "per_device_batch_size",
        "fit sample",
        "model dtype",
        "diagonal remainder",
        "seeding",
        "instrumentation",
    ):
        assert required in knobs, required
    assert all({"knob", "ours", "kronfluence_default", "reason"} <= set(d) for d in deviations)
    assert "use_empirical_fisher" not in knobs  # we match Kronfluence's default
    flat = fit_mod.FitConfig(
        covariance_module_partitions=1,
        lambda_module_partitions=1,
        eigh_device="auto",
        dtype="float32",
        use_empirical_fisher=True,
    )
    flat_knobs = [d["knob"] for d in fit_mod.kronfluence_deviations(flat)]
    for gone in ("covariance_module_partitions", "lambda_module_partitions", "eigendecomposition", "model dtype"):
        assert gone not in flat_knobs
    assert "use_empirical_fisher" in flat_knobs
    overrides = {o["knob"]: o for o in fit_mod.scimt_default_overrides(cfg)}
    assert set(overrides) == {
        "samples", "seed", "source_batch_size", "batch_size", "use_empirical_fisher",
        "eigh_device", "covariance_module_partitions", "lambda_module_partitions",
    }
    assert overrides["samples"] == {"knob": "samples", "ours": 256, "scimt_default": 1024}
    assert overrides["use_empirical_fisher"]["scimt_default"] is True
    flat_overrides = {o["knob"] for o in fit_mod.scimt_default_overrides(flat)}
    assert flat_overrides == {"samples", "seed", "source_batch_size", "batch_size"}


def test_factor_set_sizes_reproduce_gate2_arithmetic():
    sizes = fit_mod.factor_set_sizes_gb()
    assert sizes["included_numel"] == 10_759_155_456  # gate2 / PREMORTEM P, exact
    assert 160 < sizes["covariance_or_eigenvectors_gb"] < 168  # "~164 GB fp32"
    assert 42 < sizes["lambda_gb"] < 44  # one fp32 per included Linear parameter
    assert sizes["per_layer_covariance_gb"] == pytest.approx(3.41, abs=0.05)
    # gate2's operating point: 8 partitions -> "~20.5 GB" resident accumulators.
    assert fit_mod.pass_budget_gb(8, 8)["covariance_pass"]["accumulators_gb"] == pytest.approx(20.5, abs=0.5)
    budget = fit_mod.pass_budget_gb(4, 4)
    assert budget["covariance_pass"]["accumulators_gb"] == pytest.approx(
        sizes["covariance_or_eigenvectors_gb"] / 4
    )
    assert 95 < budget["covariance_pass"]["projected_total_gb"] < 105
    assert 112 < budget["lambda_pass"]["projected_total_gb"] < 124
    # The lambda pass is the tighter one; 8 partitions is the documented fallback.
    assert budget["lambda_pass"]["projected_total_gb"] > budget["covariance_pass"]["projected_total_gb"]
    assert budget["lambda_pass"]["headroom_gb"] > 15
    assert fit_mod.pass_budget_gb(4, 8)["lambda_pass"]["projected_total_gb"] < 95


# ============================================================ gate arithmetic
def test_projection_arithmetic_and_verdict_sentinel():
    projection = fit_mod.project_fit_seconds(
        elapsed_before_covariance_seconds=600,
        covariance_partition0_seconds=700,
        covariance_partitions=4,
        lambda_partitions=4,
        eigh_seconds_estimate=5400,
        lambda_partition_cost_ratio=1.5,
        export_seconds_estimate=1200,
    )
    assert projection.remaining_covariance_seconds == 2100
    assert projection.lambda_seconds == 4200
    assert projection.projected_total_seconds == 600 + 700 + 2100 + 5400 + 4200 + 1200 == 14200
    assert projection.to_dict()["projected_total_hours"] == pytest.approx(14200 / 3600)
    ok = fit_mod.gate_verdict(projection, 4.5 * 3600)
    assert ok["passed"] and not ok["message"].startswith("SCIMT")
    bad = fit_mod.gate_verdict(projection, 14000)
    assert not bad["passed"]
    assert bad["message"].startswith(f"{fit_mod.FIT_GATE_SENTINEL}: ")
    assert "14200 s" in bad["message"] and "14000 s" in bad["message"]
    warn = fit_mod.gate_verdict(projection, 14000, stage="lambda")
    assert warn["message"].startswith(fit_mod.FIT_GATE_WARN_SENTINEL)
    with pytest.raises(ValueError):
        fit_mod.project_fit_seconds(
            elapsed_before_covariance_seconds=1,
            covariance_partition0_seconds=1,
            covariance_partitions=0,
            lambda_partitions=1,
            eigh_seconds_estimate=0,
            lambda_partition_cost_ratio=1,
            export_seconds_estimate=0,
        )
    with pytest.raises(ValueError):
        fit_mod.gate_verdict(projection, 0)
    updated = fit_mod.project_after_lambda_partition0(
        elapsed_seconds=10000, lambda_partition0_seconds=1000, lambda_partitions=4,
        export_seconds_estimate=1800,
    )
    assert updated["remaining_lambda_seconds"] == 3000
    assert updated["projected_total_seconds"] == 10000 + 3000 + 1800


def test_phase_timings_derive_spans_from_partition_records():
    cov = [
        {"started_offset_seconds": 100.0, "finished_offset_seconds": 160.0, "seconds": 60.0},
        {"started_offset_seconds": 170.0, "finished_offset_seconds": 240.0, "seconds": 70.0},
    ]
    lam = [{"started_offset_seconds": 400.0, "finished_offset_seconds": 500.0, "seconds": 100.0}]
    timings = fit_mod.phase_timings(
        run_started=1000.0, fit_started=1050.0, fit_finished=1700.0,
        covariance_records=cov, lambda_records=lam,
    )
    assert timings["setup_seconds"] == 50.0 and timings["fit_seconds"] == 650.0
    assert timings["pre_covariance_seconds"] == 50.0  # 100 - (1050 - 1000)
    assert timings["covariance_partition_seconds"] == [60.0, 70.0]
    assert timings["covariance_span_seconds"] == 140.0
    assert timings["eigh_span_seconds"] == 160.0
    assert timings["lambda_span_seconds"] == 100.0
    assert timings["export_seconds"] == 200.0  # 700 - 500
    partial = fit_mod.phase_timings(
        run_started=0.0, fit_started=0.0, fit_finished=300.0,
        covariance_records=cov[:1], lambda_records=[],
    )
    assert partial["post_covariance_seconds"] == 140.0 and "export_seconds" not in partial


# ======================================================== hooks + recorder
def _fake_target(calls):
    def fake_cov(**kwargs):
        calls.append(("cov", tuple(kwargs.get("tracked_module_names") or ())))
        return ("n", {})

    def fake_lam(**kwargs):
        calls.append(("lam", tuple(kwargs.get("tracked_module_names") or ())))
        return ("n", {})

    return types.SimpleNamespace(
        fit_covariance_matrices_with_loader=fake_cov,
        fit_lambda_matrices_with_loader=fake_lam,
    )


def test_partition_hooks_record_each_partition_and_the_gate_aborts_after_the_first():
    calls: list = []
    target = _fake_target(calls)
    original_cov = target.fit_covariance_matrices_with_loader
    original_lam = target.fit_lambda_matrices_with_loader
    ticks = iter(range(0, 10_000, 10))
    gate_calls: list[int] = []

    def gate(index, record, records):
        gate_calls.append(index)
        if index == 0:
            raise fit_mod.FitGateFailure(f"{fit_mod.FIT_GATE_SENTINEL}: test", {"x": 1})

    cov_rec = fit_mod.PartitionRecorder(
        "covariance", origin=0.0, on_partition=gate, clock=lambda: float(next(ticks))
    )
    lam_rec = fit_mod.PartitionRecorder("lambda", origin=0.0, clock=lambda: float(next(ticks)))
    restore = fit_mod.install_partition_hooks(target, cov_rec, lam_rec)
    assert target.fit_covariance_matrices_with_loader is not original_cov
    with pytest.raises(fit_mod.FitGateFailure) as info:
        for i in range(4):
            target.fit_covariance_matrices_with_loader(model=None, tracked_module_names=[f"m{i}"])
    assert info.value.evidence == {"x": 1}
    assert len(calls) == 1 and gate_calls == [0]  # partitions 1..3 never ran
    record = cov_rec.records[0]
    assert record["phase"] == "covariance" and record["partition"] == 0
    assert record["seconds"] == 10.0 and record["tracked_modules"] == 1
    assert record["started_offset_seconds"] == 0.0 and record["finished_offset_seconds"] == 10.0
    assert record["host_rss_peak_gb"] > 0
    assert record["cuda_max_allocated_gb"] is None and record["nvidia_smi_window_peak_gb"] is None
    assert cov_rec.in_progress is None
    target.fit_lambda_matrices_with_loader(tracked_module_names=["a", "b"])
    assert lam_rec.records[0]["tracked_modules"] == 2 and calls[-1] == ("lam", ("a", "b"))
    restore()
    assert target.fit_covariance_matrices_with_loader is original_cov
    assert target.fit_lambda_matrices_with_loader is original_lam
    with pytest.raises(AttributeError, match="pinned to kronfluence"):
        fit_mod.install_partition_hooks(types.SimpleNamespace(), cov_rec, lam_rec)


def test_recorder_leaves_in_progress_set_when_the_partition_raises():
    class FakeOOM(RuntimeError):
        pass

    def boom(**kwargs):
        raise FakeOOM("CUDA out of memory")

    recorder = fit_mod.PartitionRecorder("lambda", origin=0.0)
    wrapped = recorder.wrap(boom)
    with pytest.raises(FakeOOM):
        wrapped(tracked_module_names=["m"])
    assert recorder.in_progress == 0 and recorder.records == []


def test_resume_guards_skip_covariance_and_eigh_only_when_eigendecomposition_exists(tmp_path):
    class FakeComputer:
        def __init__(self):
            self.calls = []

        def factors_output_dir(self, factors_name):
            return tmp_path / f"factors_{factors_name}"

        def fit_covariance_matrices(self, factors_name, **kwargs):
            self.calls.append(factors_name)
            return "fitted"

    original = FakeComputer.fit_covariance_matrices
    state = {"exists": False, "asked": []}

    def exists(output_dir):
        state["asked"].append(Path(output_dir))
        return state["exists"]

    lifted_calls: list = []
    ekfac_ns = types.SimpleNamespace(
        _lifted_eigendecomposition=lambda analyzer, model, args, device: lifted_calls.append(device)
    )
    messages: list[str] = []
    restore = fit_mod.install_resume_guards(
        types.SimpleNamespace(FactorComputer=FakeComputer),
        types.SimpleNamespace(eigendecomposition_exist=exists),
        ekfac_ns,
        log=messages.append,
    )
    computer = FakeComputer()
    assert computer.fit_covariance_matrices(factors_name="ekfac", dataset=1) == "fitted"
    ekfac_ns._lifted_eigendecomposition(computer, None, None, "cuda")
    assert lifted_calls == ["cuda"] and computer.calls == ["ekfac"]
    state["exists"] = True
    assert computer.fit_covariance_matrices(factors_name="ekfac", dataset=1) is None
    assert ekfac_ns._lifted_eigendecomposition(computer, None, None, "cuda") is None
    assert lifted_calls == ["cuda"] and computer.calls == ["ekfac"]
    assert len(messages) == 2 and all("resume" in m for m in messages)
    assert state["asked"][0] == tmp_path / "factors_ekfac"
    restore()
    assert FakeComputer.fit_covariance_matrices is original
    assert computer.fit_covariance_matrices(factors_name="ekfac") == "fitted"


# ========================================================== run_fit w/ fakes
class _FakeOOM(RuntimeError):
    pass


def _fake_deps(tmp_path, target, *, n_sequences=300, lambda_raises=False, free_gb=2000.0):
    manifest = types.SimpleNamespace(
        included_numel=10, digest=lambda: "d" * 64, included_entries=lambda: [1, 2, 3]
    )
    snapshot = tmp_path / "hub" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True, exist_ok=True)
    seen: dict = {}

    def fit_ekfac(model, dataset, manifest_arg, config, output_dir):
        seen["config"] = dict(config)
        seen["output_dir"] = output_dir
        out = Path(output_dir)
        for i in range(config["covariance_module_partitions"]):
            time.sleep(0.002)
            target.fit_covariance_matrices_with_loader(model=model, tracked_module_names=[f"m{i}"])
        kron = out / "kronfluence" / "factors_ekfac"
        kron.mkdir(parents=True, exist_ok=True)
        (kron / "eigh_report.json").write_text(
            json.dumps(
                {
                    "device": "cuda",
                    "eigendecomposition_dtype": "torch.float64",
                    "total_eigh_seconds": 12.5,
                    "total_load_seconds": 1.0,
                    "total_transfer_seconds": 1.0,
                    "total_save_seconds": 1.0,
                    "saves": [],
                    "matrices": [{}, {}],
                }
            )
        )
        (kron / "activation_covariance.safetensors").write_bytes(b"x" * 100)
        for i in range(config["lambda_module_partitions"]):
            time.sleep(0.002)
            if lambda_raises:
                target.fit_lambda_matrices_with_loader(
                    model=model, tracked_module_names=[f"m{i}"], raise_oom=True
                )
            target.fit_lambda_matrices_with_loader(model=model, tracked_module_names=[f"m{i}"])
        (out / "linear" / "m0").mkdir(parents=True, exist_ok=True)
        for key in ("U_A", "U_S", "lam"):
            (out / "linear" / "m0" / f"{key}.npy").write_bytes(b"\0" * 64)
        (out / "diag").mkdir(exist_ok=True)
        (out / "diag" / "v.npy").write_bytes(b"\0" * 8)
        (out / "diag" / "index.json").write_text("[]")
        (out / "ekfac_meta.json").write_text("{}")
        (out / "parameter_manifest.json").write_text("{}")
        return types.SimpleNamespace(snapshot="snap123")

    deps = fit_mod.FitDeps(
        resolve_snapshot=lambda model_id, revision: (snapshot, "abc123"),
        load_tokenizer=lambda snapshot_dir: "tokenizer",
        load_model=lambda snapshot_dir, dtype, device, checkpointing: "model",
        build_manifest=lambda model, include, exclude: manifest,
        build_dataset=lambda cfg, tokenizer: [0] * n_sequences,
        fit_ekfac=fit_ekfac,
        kronfluence_factor_computer=lambda: target,
        resume_targets=lambda: (None, None, None),
        cuda_stats=lambda: None,
        gpu_poller=lambda interval, index: fit_mod.NullPoller(),
        seed_everything=lambda seed: seen.setdefault("seeds", []).append(seed),
        oom_error_types=lambda: (_FakeOOM,),
        versions=lambda: {"kronfluence": "1.0.1", "torch": "x"},
        disk_free_gb=lambda path: free_gb,
    )
    return deps, seen


def _cfg(tmp_path, **overrides):
    base = {
        "output_dir": str(tmp_path / "ekfac_pt"),
        "evidence_dir": str(tmp_path / "evidence"),
        "min_free_disk_gb": 1.0,
        "max_projected_seconds": 1e9,
    }
    base.update(overrides)
    return fit_mod.FitConfig.from_mapping(base)


def test_run_fit_with_fakes_writes_gate_evidence_and_receipt(tmp_path, capsys):
    calls: list = []
    target = _fake_target(calls)
    deps, seen = _fake_deps(tmp_path, target)
    cfg = _cfg(tmp_path)
    receipt = fit_mod.run_fit(cfg, deps)
    assert receipt["status"] == "ok" and fit_mod.exit_code_for("ok") == 0
    assert seen["config"] == cfg.ekfac_config() and seen["output_dir"] == cfg.output_dir
    assert seen["seeds"] == [42]
    evidence = tmp_path / "evidence"
    for name in (
        fit_mod.COVARIANCE_GATE_FILE,
        fit_mod.LAMBDA_GATE_FILE,
        fit_mod.RECEIPT_FILE,
        fit_mod.LOG_FILE,
        "fit_config.json",
        "provenance.json",
    ):
        assert (evidence / name).exists(), name
    gate = json.loads((evidence / fit_mod.COVARIANCE_GATE_FILE).read_text())
    assert gate["verdict"]["passed"] and gate["verdict"]["stage"] == "covariance"
    assert gate["projection"]["covariance_partitions"] == 4
    assert gate["projection"]["projected_total_seconds"] > cfg.eigh_seconds_estimate
    assert gate["partition_record"]["partition"] == 0 and gate["partition_record"]["seconds"] > 0
    assert gate["memory_budget_gb"]["lambda_pass"]["partitions"] == 4
    lam = json.loads((evidence / fit_mod.LAMBDA_GATE_FILE).read_text())
    assert lam["eigh_report"]["total_eigh_seconds"] == 12.5 and lam["eigh_report"]["matrices"] == 2
    assert len(lam["covariance_partition_records"]) == 4
    assert lam["projection"]["remaining_lambda_seconds"] >= 0 and lam["verdict"]["passed"]
    assert [c[0] for c in calls] == ["cov"] * 4 + ["lam"] * 4
    assert receipt["manifest_digest"] == "d" * 64 and receipt["included_numel"] == 10
    assert receipt["model"]["sha"] == "abc123" and receipt["model"]["hf_id"] == cfg.model_id
    assert receipt["kronfluence_version"] == "1.0.1"
    assert len(receipt["covariance_partitions"]) == 4 and len(receipt["lambda_partitions"]) == 4
    assert receipt["factors_snapshot"] == "snap123"
    timings = receipt["timings"]
    for key in (
        "load_model_seconds", "dataset_seconds", "setup_seconds", "pre_covariance_seconds",
        "covariance_span_seconds", "eigh_span_seconds", "lambda_span_seconds", "export_seconds",
        "total_seconds",
    ):
        assert key in timings, key
    assert len(timings["lambda_partition_seconds"]) == 4
    files = receipt["factor_files"]
    assert files["exported_bytes"] == 3 * 64 + 8 + 2 + 2 + 2
    assert files["kronfluence_files"] == 2 and files["kronfluence_bytes"] > 100
    assert {d["knob"] for d in receipt["kronfluence_deviations"]} >= {
        "covariance_module_partitions", "lambda_module_partitions",
    }
    assert receipt["failure"] is None
    assert fit_mod.FIT_DONE_SENTINEL in capsys.readouterr().out
    # hooks restored after the run
    assert calls and target.fit_covariance_matrices_with_loader.__name__ == "fake_cov"


def test_run_fit_gate_failure_aborts_after_the_first_covariance_partition(tmp_path, capsys):
    calls: list = []
    target = _fake_target(calls)
    deps, _ = _fake_deps(tmp_path, target)
    # Any measured partition plus the 5,400 s eigh label exceeds a 1 ms budget.
    receipt = fit_mod.run_fit(_cfg(tmp_path, max_projected_seconds=1e-3), deps)
    assert receipt["status"] == "gate-failed"
    assert fit_mod.exit_code_for(receipt["status"]) == fit_mod.FIT_GATE_EXIT_CODE == 97
    assert len(calls) == 1 and len(receipt["covariance_partitions"]) == 1
    assert receipt["lambda_partitions"] == []
    assert receipt["failure"]["evidence"]["verdict"]["passed"] is False
    assert receipt["failure"]["message"].startswith(fit_mod.FIT_GATE_SENTINEL)
    out = capsys.readouterr().out
    assert f"{fit_mod.FIT_GATE_SENTINEL}: " in out and fit_mod.FIT_DONE_SENTINEL not in out
    assert (tmp_path / "evidence" / fit_mod.RECEIPT_FILE).exists()
    assert not (tmp_path / "evidence" / fit_mod.LAMBDA_GATE_FILE).exists()
    assert "post_covariance_seconds" in receipt["timings"]


def test_run_fit_oom_receipt_names_the_active_phase(tmp_path, capsys):
    calls: list = []
    target = _fake_target(calls)
    plain_lam = target.fit_lambda_matrices_with_loader

    def lam_or_boom(**kwargs):
        if kwargs.pop("raise_oom", False):
            raise _FakeOOM("CUDA out of memory. Tried to allocate 944 MiB")
        return plain_lam(**kwargs)

    target.fit_lambda_matrices_with_loader = lam_or_boom
    deps, _ = _fake_deps(tmp_path, target, lambda_raises=True)
    receipt = fit_mod.run_fit(_cfg(tmp_path), deps)
    assert receipt["status"] == "oom"
    assert fit_mod.exit_code_for("oom") == fit_mod.FIT_OOM_EXIT_CODE == 98
    assert receipt["failure"]["phase"] == "lambda partition 0"
    assert "lambda_module_partitions" in receipt["failure"]["suggestion"]
    assert len(receipt["covariance_partitions"]) == 4 and receipt["lambda_partitions"] == []
    assert f"{fit_mod.FIT_OOM_SENTINEL}: phase=lambda partition 0" in capsys.readouterr().out


def test_run_fit_refuses_short_calibration_sample_and_low_disk(tmp_path):
    target = _fake_target([])
    deps, _ = _fake_deps(tmp_path, target, n_sequences=10)
    with pytest.raises(ValueError, match="256 are required"):
        fit_mod.run_fit(_cfg(tmp_path), deps)
    deps, _ = _fake_deps(tmp_path, target, free_gb=1.0)
    with pytest.raises(RuntimeError, match="min_free_disk_gb"):
        fit_mod.run_fit(_cfg(tmp_path, min_free_disk_gb=800.0), deps)


def test_main_is_config_first(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_run(cfg, deps=None):
        captured["cfg"] = cfg
        return {"status": "gate-failed"}

    monkeypatch.setattr(fit_mod, "run_fit", fake_run)
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"samples": 16, "output_dir": str(tmp_path / "o")}))
    assert fit_mod.main([str(overrides)]) == fit_mod.FIT_GATE_EXIT_CODE
    assert captured["cfg"].samples == 16 and captured["cfg"].sequence_length == 4096
    with pytest.raises(SystemExit):
        fit_mod.main(["--samples", "16"])
    with pytest.raises(SystemExit):
        fit_mod.main([str(overrides), str(overrides)])


# ================================================================= telemetry
def test_nvidia_smi_poller_parses_output_and_tracks_windows():
    outputs = iter(["1024\n2048\n", "4096\n", "512\n"])
    commands: list = []

    def runner(command, **kwargs):
        commands.append(command)
        return types.SimpleNamespace(returncode=0, stdout=next(outputs))

    poller = fit_mod.NvidiaSmiPoller(interval_seconds=1.0, gpu_index=None, runner=runner)
    assert poller.poll_once() == 2048.0  # max over the GPUs listed
    poller.mark()
    assert poller.poll_once() == 4096.0
    assert poller.window_peak_gb() == pytest.approx(4096 * 1024**2 / 1e9)
    poller.mark()
    assert poller.poll_once() == 512.0
    assert poller.window_peak_gb() == pytest.approx(512 * 1024**2 / 1e9)
    assert poller.peak_gb() == pytest.approx(4096 * 1024**2 / 1e9)
    assert poller.samples == 3 and "--id" not in " ".join(commands[0])
    pinned = fit_mod.NvidiaSmiPoller(
        1.0, gpu_index=1, runner=lambda command, **kw: (commands.append(command), types.SimpleNamespace(returncode=1, stdout=""))[1]
    )
    assert pinned.poll_once() is None and pinned.failures == 1
    assert "--id=1" in commands[-1]
    assert fit_mod.default_nvidia_smi_index({"CUDA_VISIBLE_DEVICES": "2,3"}) == 2
    assert fit_mod.default_nvidia_smi_index({}) is None
    assert fit_mod.default_nvidia_smi_index({"CUDA_VISIBLE_DEVICES": "GPU-abc"}) is None


# ============================================================ sidecar contract
def _gdp_sidecar(**overrides):
    payload = {
        "name": "dolmino__gdp__all",
        "kind": "gdp",
        "dataset": "dolmino",
        "damping_scale": None,
        "fold": "all",
        "n_rows": 512,
        "n_tokens": 4194304,
        "manifest_digest": "a" * 64,
        "model": {"hf_id": "google/gemma-3-12b-pt", "sha": "295efb63d01a7017928f273a94ebb86105c9526f"},
        "sequence_length": 4096,
        "created_at": "2026-09-13T20:00:00+00:00",
        "source_vector": None,
    }
    payload.update(overrides)
    return payload


def test_sidecar_contract_round_trip_and_naming(tmp_path):
    gdp = _gdp_sidecar()
    assert apply_mod.validate_sidecar(gdp) == gdp
    vector = tmp_path / "dolmino__gdp__all.f32"
    written = apply_mod.write_sidecar(vector, gdp)
    assert written == tmp_path / "dolmino__gdp__all.json"
    assert apply_mod.read_sidecar(vector) == gdp
    inv = apply_mod.make_inv_sidecar(gdp, 0.1, "dolmino__gdp__all.f32", created_at="2026-09-13T21:00:00+00:00")
    assert inv["name"] == "dolmino__inv0.1__all" and inv["kind"] == "inv"
    assert inv["damping_scale"] == 0.1 and inv["source_vector"] == "dolmino__gdp__all.f32"
    assert inv["created_at"] == "2026-09-13T21:00:00+00:00"
    for key in ("dataset", "fold", "n_rows", "n_tokens", "manifest_digest", "model", "sequence_length"):
        assert inv[key] == gdp[key]
    assert apply_mod.format_damping(0.1) == "0.1"
    assert apply_mod.format_damping(1.0) == "1"
    assert apply_mod.format_damping(0.01) == "0.01"
    assert apply_mod.inv_vector_name("charter_worked", 1.0, "f1") == "charter_worked__inv1__f1"
    with pytest.raises(ValueError):
        apply_mod.format_damping(0.0)
    with pytest.raises(ValueError):
        apply_mod.inv_vector_name("bad__label", 0.1, "all")
    with pytest.raises(ValueError):
        apply_mod.inv_vector_name("coin", 0.1, "fold1")
    with pytest.raises(ValueError, match="suffix"):
        apply_mod.sidecar_path(tmp_path / "x.npy")


def test_sidecar_validation_refusals():
    bad_cases = [
        {k: v for k, v in _gdp_sidecar().items() if k != "n_rows"},  # missing key
        _gdp_sidecar(extra=1),
        _gdp_sidecar(kind="foo"),
        _gdp_sidecar(damping_scale=0.1),  # gdp with damping
        _gdp_sidecar(source_vector="x.f32"),  # gdp with source
        _gdp_sidecar(fold="x"),
        _gdp_sidecar(dataset="a__b"),
        _gdp_sidecar(n_tokens=-1),
        _gdp_sidecar(model={"hf_id": "m"}),
        _gdp_sidecar(kind="inv", damping_scale=0.1, source_vector="s.f32"),  # wrong inv name
        _gdp_sidecar(kind="inv", name="dolmino__inv0.1__all", damping_scale=0.0, source_vector="s.f32"),
        _gdp_sidecar(kind="inv", name="dolmino__inv0.1__all", damping_scale=0.1, source_vector=None),
    ]
    for payload in bad_cases:
        with pytest.raises(ValueError):
            apply_mod.validate_sidecar(payload)
    with pytest.raises(TypeError):
        apply_mod.validate_sidecar(["not", "a", "mapping"])
    inv = _gdp_sidecar(kind="inv", name="dolmino__inv0.1__all", damping_scale=0.1, source_vector="s.f32")
    with pytest.raises(ValueError, match="gdp vectors only"):
        apply_mod.make_inv_sidecar(inv, 0.1, "s.f32")


def test_resolve_out_paths_follows_the_naming_contract(tmp_path):
    inputs = [tmp_path / "in" / "a.f32", tmp_path / "in" / "b.f32"]
    sidecars = [_gdp_sidecar(dataset="dolmino"), _gdp_sidecar(dataset="coin", fold="f1")]
    dampings = [0.1, 1.0]
    default = apply_mod.resolve_out_paths(inputs, sidecars, dampings, None)
    assert default == [
        [tmp_path / "in" / "dolmino__inv0.1__all.f32", tmp_path / "in" / "dolmino__inv1__all.f32"],
        [tmp_path / "in" / "coin__inv0.1__f1.f32", tmp_path / "in" / "coin__inv1__f1.f32"],
    ]
    into = apply_mod.resolve_out_paths(inputs, sidecars, dampings, tmp_path / "out")
    assert into[1][0] == tmp_path / "out" / "coin__inv0.1__f1.f32"
    single = apply_mod.resolve_out_paths(inputs[:1], sidecars[:1], [0.1], tmp_path / "x.f32")
    assert single == [[tmp_path / "x.f32"]]
    with pytest.raises(ValueError):
        apply_mod.resolve_out_paths(inputs, sidecars, dampings, tmp_path / "x.f32")
    with pytest.raises(ValueError):
        apply_mod.resolve_out_paths(inputs, sidecars, dampings, [tmp_path / "only_one.f32"])
    explicit = apply_mod.resolve_out_paths(
        inputs, sidecars, dampings, [tmp_path / f"e{i}.f32" for i in range(4)]
    )
    assert explicit[1][1] == tmp_path / "e3.f32"


# ============================================================== torch parity
def _tiny_factor_set(path: Path):
    """A 2-Linear (+LayerNorm) toy model with a valid EK-FAC artifact on disk,
    in the layout ``load_ekfac`` reads (as tests/data_attribution/test_ekfac.py)."""
    torch = pytest.importorskip("torch")
    from torch import nn

    from scimt.data_attribution.ekfac import load_ekfac
    from scimt.data_attribution.manifest import ParameterManifest

    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 5, bias=False), nn.LayerNorm(5))
    manifest = ParameterManifest.from_model(model, "tiny")
    manifest.save(path)
    generator = torch.Generator().manual_seed(17)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name in names:
        module = model.get_submodule(name)
        directory = path / "linear" / name.replace(".", "__")
        directory.mkdir(parents=True)
        ka = module.in_features + int(module.bias is not None)
        np.save(
            directory / "U_A.npy",
            torch.linalg.qr(torch.randn(ka, ka, generator=generator, dtype=torch.float64)).Q.numpy(),
        )
        np.save(
            directory / "U_S.npy",
            torch.linalg.qr(
                torch.randn(module.out_features, module.out_features, generator=generator, dtype=torch.float64)
            ).Q.numpy(),
        )
        np.save(
            directory / "lam.npy",
            (torch.rand(module.out_features, ka, generator=generator, dtype=torch.float64) + 0.2).numpy(),
        )
    claimed = {f"{n}.{s}" for n in names for s in ("weight", "bias")}
    diagonal = [e for e in manifest.included_entries() if e.name not in claimed]
    (path / "diag").mkdir()
    np.save(path / "diag" / "v.npy", np.linspace(0.3, 1.1, sum(e.numel for e in diagonal)))
    (path / "diag" / "index.json").write_text(
        json.dumps([{"name": e.name, "numel": e.numel, "offset": e.global_flat_offset} for e in diagonal])
    )
    (path / "ekfac_meta.json").write_text(json.dumps({"linears": names}))
    return manifest, load_ekfac(path, manifest)


def _rel_error(ours, reference):
    diff = (ours.double() - reference).abs().max()
    return float(diff / reference.abs().max())


def test_device_inverse_matches_library_apply_ekfac_within_1e5(tmp_path):
    torch = pytest.importorskip("torch")
    from scimt.data_attribution.ekfac import apply_ekfac

    manifest, factors = _tiny_factor_set(tmp_path)
    assert manifest.included_numel == 12 + 4 + 20 + 5 + 5
    vector = torch.randn(manifest.included_numel, generator=torch.Generator().manual_seed(9))
    dampings = [0.01, 0.1, 1.0]
    ours = apply_mod.apply_inverse_array(vector, factors, manifest, dampings, device="cpu")
    assert len(ours) == 3 and all(o.dtype == torch.float32 and o.shape == vector.shape for o in ours)
    for damping, mine in zip(dampings, ours, strict=True):
        reference = apply_ekfac(vector.double(), factors, manifest, damping, -1)
        assert _rel_error(mine, reference) < 1e-5, damping
    # A batch of vectors shares one pass and equals the per-vector results.
    other = torch.randn(manifest.included_numel, generator=torch.Generator().manual_seed(10))
    batched = apply_mod.apply_inverse_array(torch.stack([vector, other]), factors, manifest, [0.1], "cpu")[0]
    torch.testing.assert_close(batched[0], ours[1], rtol=0, atol=0)
    torch.testing.assert_close(
        batched[1], apply_mod.apply_inverse_array(other, factors, manifest, 0.1, "cpu")[0], rtol=0, atol=0
    )
    for bad in (0.0, -0.1, [0.1, 0.1], []):
        with pytest.raises(ValueError):
            apply_mod.apply_inverse_array(vector, factors, manifest, bad, "cpu")
    with pytest.raises(ValueError, match="shape"):
        apply_mod.apply_inverse_array(vector[:-1], factors, manifest, [0.1], "cpu")


def test_apply_inverse_file_mode_writes_contract_vectors_and_sidecars(tmp_path):
    torch = pytest.importorskip("torch")

    factors_dir = tmp_path / "factors"
    factors_dir.mkdir()
    manifest, factors = _tiny_factor_set(factors_dir)
    vectors = tmp_path / "vectors"
    vectors.mkdir()
    rng = np.random.default_rng(3)
    source = rng.standard_normal(manifest.included_numel).astype("<f4")
    gdp_path = vectors / "dolmino__gdp__all.f32"
    source.tofile(gdp_path)
    apply_mod.write_sidecar(gdp_path, _gdp_sidecar(manifest_digest=manifest.digest()))

    written = apply_mod.apply_inverse(gdp_path, factors_dir, manifest, [0.1, 1.0], "cpu", log=lambda m: None)
    assert written == [vectors / "dolmino__inv0.1__all.f32", vectors / "dolmino__inv1__all.f32"]
    expected = apply_mod.apply_inverse_array(torch.from_numpy(source.copy()), factors, manifest, [0.1, 1.0], "cpu")
    for path, damping, reference in zip(written, [0.1, 1.0], expected, strict=True):
        assert path.stat().st_size == 4 * manifest.included_numel
        data = np.fromfile(path, dtype="<f4")
        np.testing.assert_array_equal(data, reference.numpy())
        sidecar = apply_mod.read_sidecar(path)
        assert sidecar["kind"] == "inv" and sidecar["damping_scale"] == damping
        assert sidecar["source_vector"] == "dolmino__gdp__all.f32" and sidecar["dataset"] == "dolmino"
        assert sidecar["manifest_digest"] == manifest.digest()

    # Two inputs in one pass, into a directory, equal the single-input results.
    second = vectors / "coin__gdp__f0.f32"
    other = rng.standard_normal(manifest.included_numel).astype("<f4")
    other.tofile(second)
    apply_mod.write_sidecar(second, _gdp_sidecar(name="coin__gdp__f0", dataset="coin", fold="f0", manifest_digest=manifest.digest()))
    out_dir = tmp_path / "out"
    pair = apply_mod.apply_inverse([gdp_path, second], factors_dir, manifest, 0.1, "cpu", out_dir, factors=factors, log=lambda m: None)
    assert pair == [out_dir / "dolmino__inv0.1__all.f32", out_dir / "coin__inv0.1__f0.f32"]
    np.testing.assert_array_equal(np.fromfile(pair[0], dtype="<f4"), expected[0].numpy())
    np.testing.assert_array_equal(
        np.fromfile(pair[1], dtype="<f4"),
        apply_mod.apply_inverse_array(torch.from_numpy(other.copy()), factors, manifest, 0.1, "cpu")[0].numpy(),
    )

    # Refusals: an inv vector as input, a foreign manifest, a short file, output == input.
    with pytest.raises(ValueError, match="consumes gdp"):
        apply_mod.apply_inverse(written[0], factors_dir, manifest, 0.1, "cpu", factors=factors)
    foreign = vectors / "foreign__gdp__all.f32"
    source.tofile(foreign)
    apply_mod.write_sidecar(foreign, _gdp_sidecar(dataset="foreign", manifest_digest="b" * 64))
    with pytest.raises(ValueError, match="manifest"):
        apply_mod.apply_inverse(foreign, factors_dir, manifest, 0.1, "cpu", factors=factors)
    short = vectors / "short__gdp__all.f32"
    source[:-1].tofile(short)
    apply_mod.write_sidecar(short, _gdp_sidecar(dataset="short", manifest_digest=manifest.digest()))
    with pytest.raises(ValueError, match="bytes"):
        apply_mod.apply_inverse(short, factors_dir, manifest, 0.1, "cpu", factors=factors)
    with pytest.raises(ValueError, match="coincides"):
        apply_mod.apply_inverse(gdp_path, factors_dir, manifest, 0.1, "cpu", gdp_path, factors=factors)


def test_oracle_check_restricts_to_a_few_modules_and_passes(tmp_path):
    pytest.importorskip("torch")
    manifest, factors = _tiny_factor_set(tmp_path)
    report = apply_mod.oracle_check(tmp_path, manifest, n_modules=2, device="cpu", damping_scales=(0.1, 1.0))
    assert report["passed"] and report["sentinel"] == apply_mod.ORACLE_PASS_SENTINEL
    assert report["max_rel_error"] < 1e-5 and report["tolerance"] == 1e-5
    assert report["modules"] == ["0", "1"] and report["diag_entries"] == ["2.weight"]
    assert report["numel"] == 12 + 4 + 20 + 5
    assert set(report["dampings"]) == {"0.1", "1"}
    one = apply_mod.oracle_check(None, manifest, n_modules=1, factors=factors, n_diag_entries=0)
    assert one["modules"] == ["0"] and one["diag_entries"] == [] and one["numel"] == 16 and one["passed"]
    with pytest.raises(ValueError, match="requested"):
        apply_mod.oracle_check(None, manifest, n_modules=3, factors=factors)
    with pytest.raises(ValueError):
        apply_mod.oracle_check(None, manifest, n_modules=0, factors=factors)
    sub_factors, sub_manifest = apply_mod.restrict_to_modules(factors, manifest, ["1"], n_diag_entries=2)
    assert [e.name for e in sub_manifest.included_entries()] == ["1.weight", "2.weight", "2.bias"]
    assert sub_manifest.included_entries()[1].global_flat_offset == 20
    assert tuple(sub_factors.diag_index) == (
        {"name": "2.weight", "numel": 5, "offset": 20},
        {"name": "2.bias", "numel": 5, "offset": 25},
    )
    assert sub_factors.diag_v.numel() == 10

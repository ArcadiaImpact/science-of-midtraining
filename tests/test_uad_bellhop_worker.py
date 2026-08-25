"""Bellhop uad port, T2 — CPU-only unit tests for the pod-side worker and
setup builder (BELLHOP_PORT.md §7).

Heavy machinery is faked via ``sys.modules`` injection (the
test_uad_chain.py / test_axolotl_backend.py pattern): the real
``chain_uad``/``chain`` never train, upload, or touch the network here.
Covers the frozen T2<->T3 contracts: the worker CLI
(``--run-id/--arms/--signed-off``), the GCS receipt path
``token-scaling-4b-uad/<run-id>/<parent>/<leaf>/ARM_COMPLETE.json``, the
``worker_summary.json`` layout, ``build_setup(wheel_rel)``, and the lane
pinned to "0".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BELLHOP = (REPO_ROOT
           / "experiments/prior_coins/dispatch_unambiguous_dose/bellhop")


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aw = _load("uad_arm_worker", BELLHOP / "arm_worker.py")
ps = _load("uad_pod_setup", BELLHOP / "pod_setup.py")

WHEEL = "experiments/prior_coins/dispatch_unambiguous_dose/bellhop/dist/scimt-0.1.0-py3-none-any.whl"


# ===========================================================================
# pod_setup.build_setup
# ===========================================================================

class TestBuildSetup:
    def setup_method(self):
        self.script = ps.build_setup(WHEEL)

    def test_is_a_strict_shell_script(self):
        assert self.script.startswith("set -euo pipefail")
        assert self.script.rstrip().endswith("echo SETUP_OK")

    def test_installs_the_wheel_never_editable(self):
        assert WHEEL in self.script
        assert " -e ." not in self.script
        assert "pip install -e" not in self.script

    def test_train_venv_from_pinned_requirements(self):
        assert "requirements/pod-h200.txt" in self.script
        assert "uv venv --clear /workspace/venv-train --python 3.12" in (
            self.script)

    def test_eval_venv_pins_are_the_tsl_harness_pins(self):
        for pin in ("vllm==0.8.5.post1", "transformers==4.51.3",
                    "torch==2.6.0"):
            assert pin in self.script
        assert "/workspace/venv-dispatch-eval" in self.script

    def test_both_gemma3_patches_with_failloud_greps(self):
        # patch 1: tied lm_head loader — python raise + grep verification
        assert "unexpected vLLM Gemma-3 loader source" in self.script
        assert 'skip_prefixes=\\["lm_head."\\]' in self.script
        assert "FATAL: vLLM Gemma-3 lm_head patch not applied" in self.script
        # patch 2: LoRA name remap — repo script + grep verification
        assert ("experiments/prior_coins/pod/patch_vllm_gemma3_lora.py"
                in self.script)
        assert 'grep -q "scimt: LoRA name remap"' in self.script
        assert "FATAL: vLLM Gemma-3 LoRA patch not applied" in self.script

    def test_uv_env_knobs_ported(self):
        for knob in ("UV_INDEX_STRATEGY=unsafe-best-match",
                     "UV_HTTP_TIMEOUT=600",
                     "UV_CACHE_DIR=/workspace/.cache/uv"):
            assert knob in self.script

    def test_no_git_clone_no_tokens_no_pinned_commit_checkout(self):
        # provenance is SCIMT_SOURCE_COMMIT (manifest path), not git.
        assert "git clone" not in self.script
        assert "GITHUB_TOKEN" not in self.script
        assert "SCIMT_COMMIT" not in self.script.replace(
            "SCIMT_SOURCE_COMMIT", "")
        assert "git checkout" not in self.script

    def test_env_contract_presence_checked_before_installs(self):
        for name in ps.REQUIRED_ENV:
            assert f'[ -n "${{{name}:-}}" ]' in self.script
        # the checks come before the first install
        assert self.script.index("SCIMT_GCS_BASE") < self.script.index(
            "apt-get")

    def test_no_secret_values_interpolated(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_SENTINEL_SECRET")
        monkeypatch.setenv("SCIMT_GCS_BASE", "gs://sentinel-bucket/base")
        monkeypatch.setenv("RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
                           '{"sentinel": "credential"}')
        script = ps.build_setup(WHEEL)
        assert "hf_SENTINEL_SECRET" not in script
        assert "sentinel-bucket" not in script
        assert "sentinel" not in script

    def test_env_verification_blocks_ported(self):
        assert "train_env" in self.script
        assert "eval_env" in self.script
        assert "hf_auth" in self.script
        assert "torch.cuda.is_available()" in self.script

    def test_hf_auth_uses_the_train_venv_python(self):
        auth = self.script[self.script.index("hf_auth") - 600:]
        assert "/workspace/venv-train/bin/python" in auth

    @pytest.mark.parametrize("bad", [
        "/abs/path/scimt.whl",
        "../outside.whl",
        "dist/scimt.tar.gz",
        "dist/has space.whl",
        "",
    ])
    def test_wheel_rel_validation(self, bad):
        with pytest.raises(ValueError):
            ps.build_setup(bad)


# ===========================================================================
# Frozen contracts: receipt path, CLI, arm parsing
# ===========================================================================

class TestFrozenContracts:
    def test_receipt_path_matches_the_design_doc_literal(self):
        # real chain_uad (CPU-safe import) — the T2<->T3 shared constant.
        assert aw.receipt_rel("20260826T000000Z", "control_d0__coin_d2pct") \
            == ("token-scaling-4b-uad/20260826T000000Z/control_d0/"
                "coin_d2pct/ARM_COMPLETE.json")
        assert aw.receipt_rel("r1", "coin_d8m__baseline") \
            == "token-scaling-4b-uad/r1/coin_d8m/baseline/ARM_COMPLETE.json"
        assert aw.receipt_rel("r1", "coin_d8m__charter_d0.2pct_s43") \
            == ("token-scaling-4b-uad/r1/coin_d8m/charter_d0.2pct_s43/"
                "ARM_COMPLETE.json")

    def test_frozen_cli_flags_parse(self):
        args = aw.parse_args(["--run-id", "r1",
                              "--arms", "a__baseline,a__coin_d1pct",
                              "--signed-off"])
        assert args.run_id == "r1"
        assert args.arms == "a__baseline,a__coin_d1pct"
        assert args.signed_off is True
        assert args.workdir == "/workspace/uad"
        assert args.results_dir == aw.WORKER_RESULTS_DIR

    def test_run_id_and_arms_required(self):
        with pytest.raises(SystemExit):
            aw.parse_args(["--arms", "x"])
        with pytest.raises(SystemExit):
            aw.parse_args(["--run-id", "r1"])

    def test_lane_is_pinned_to_zero(self):
        assert aw.UAD_GPU_LANE == "0"

    def test_signoff_gate_refuses_gpu_spend(self):
        with pytest.raises(SystemExit, match="REFUSING to spend GPU"):
            aw.main(["--run-id", "r1", "--arms", "control_d0__baseline"])

    def test_parse_arms(self):
        assert aw.parse_arms("a, b ,c") == ["a", "b", "c"]
        with pytest.raises(ValueError, match="duplicates"):
            aw.parse_arms("a,a")
        with pytest.raises(ValueError, match="no arms"):
            aw.parse_arms(" , ")


# ===========================================================================
# Worker sequencing against a fully stubbed chain_uad / hydrate_parents
# ===========================================================================

class FakeChain:
    def __init__(self, tmp: Path) -> None:
        eval_py = tmp / "eval-python"
        eval_py.write_text("")
        self.EVAL_PYTHON = str(eval_py)
        self.UPLOAD_TIMEOUTS_S = {"evidence": 1_800}
        self.calls: list[tuple] = []

    def load_env_file(self, path=None):  # replaced by the worker
        raise AssertionError("load_env_file must be neutralized")

    def require_gcs_ready(self):
        self.calls.append(("gcs_ready",))

    def upload_and_pin(self, local_dir, relative, pins_dir, *, timeout_s):
        files = sorted(p.name for p in Path(local_dir).rglob("*")
                       if p.is_file())
        self.calls.append(("upload", str(relative), files, timeout_s))
        Path(pins_dir).mkdir(parents=True, exist_ok=True)

    def log(self, message):
        print(message)


def _make_fakes(tmp: Path, fail_on: str | None = None,
                skip_receipt_for: str | None = None):
    """Fake chain_uad + hydrate_parents modules, recording call order."""
    chain = FakeChain(tmp)
    cu = types.ModuleType("chain_uad")
    cu.RUN_PREFIX = "token-scaling-4b-uad"
    cu.TSL_RUN_ID = "20260823T142829Z"
    cu.chain = chain
    events: list[tuple] = []

    def parse_arm_id(arm_id: str):
        parent, _, leaf = arm_id.partition("__")
        return SimpleNamespace(parent=parent, leaf=leaf, arm_id=arm_id)

    async def run(args):
        assert os.environ["UAD_GPU"] == "0"  # lane pinned (R3)
        assert args.signed_off is True and args.dry_run is False
        events.append(("run", args.arm))
        if args.arm == fail_on:
            raise RuntimeError(f"boom in {args.arm}")
        work = Path(args.workdir) / args.run_id / "arms" / args.arm
        work.mkdir(parents=True, exist_ok=True)
        (work / "pins").mkdir(exist_ok=True)
        (work / "pins" / "some_pin.json").write_text("{}")
        # scratch the prune must clear
        (work / "merged").mkdir(exist_ok=True)
        (work / "eval_work").mkdir(exist_ok=True)
        (work / "eft-r32" / "run").mkdir(parents=True, exist_ok=True)
        if args.arm != skip_receipt_for:
            (work / "ARM_COMPLETE.json").write_text(json.dumps(
                {"arm": args.arm, "run_id": args.run_id}))

    cu.parse_arm_id = parse_arm_id
    cu.run = run
    cu.events = events

    hp = types.ModuleType("hydrate_parents")

    def hydrate_parent(work_root, cell, tsl_run_id):
        assert tsl_run_id == cu.TSL_RUN_ID
        events.append(("hydrate", cell))
        return Path(work_root) / "parents" / cell / "checkpoint-24"

    hp.hydrate_parent = hydrate_parent
    return cu, hp, chain, events


@pytest.fixture()
def pod_env(tmp_path, monkeypatch):
    workdir = tmp_path / "runtime" / "uad"
    results = tmp_path / "results"
    monkeypatch.setenv("SCIMT_GCS_BASE", "gcs:bucket/base")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("SCIMT_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("SCIMT_RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setattr(aw, "WRITE_PROBE_BYTES", 4_096)
    return SimpleNamespace(workdir=workdir, results=results)


def _install(monkeypatch, cu, hp):
    monkeypatch.setitem(sys.modules, "chain_uad", cu)
    monkeypatch.setitem(sys.modules, "hydrate_parents", hp)


def _argv(pod_env, arms: str) -> list[str]:
    return ["--run-id", "r1", "--arms", arms, "--signed-off",
            "--workdir", str(pod_env.workdir),
            "--results-dir", str(pod_env.results)]


class TestWorkerRun:
    ARMS = ("control_d0__baseline,control_d0__coin_d2pct,"
            "coin_d8m__baseline")

    def test_happy_path_sequencing_and_receipts(self, pod_env, tmp_path,
                                                monkeypatch):
        cu, hp, chain, events = _make_fakes(tmp_path)
        _install(monkeypatch, cu, hp)
        aw.main(_argv(pod_env, self.ARMS))

        # gcs preflight before anything else
        assert chain.calls[0] == ("gcs_ready",)
        # parents hydrated once each, grouped, before any arm runs
        assert events[:2] == [("hydrate", "control_d0"),
                              ("hydrate", "coin_d8m")]
        assert [e for e in events if e[0] == "run"] == [
            ("run", "control_d0__baseline"),
            ("run", "control_d0__coin_d2pct"),
            ("run", "coin_d8m__baseline"),
        ]
        # one receipt upload per arm, at the FROZEN leaf-root path, carrying
        # exactly ARM_COMPLETE.json
        uploads = [c for c in chain.calls if c[0] == "upload"]
        assert [(u[1], u[2]) for u in uploads] == [
            ("token-scaling-4b-uad/r1/control_d0/baseline",
             ["ARM_COMPLETE.json"]),
            ("token-scaling-4b-uad/r1/control_d0/coin_d2pct",
             ["ARM_COMPLETE.json"]),
            ("token-scaling-4b-uad/r1/coin_d8m/baseline",
             ["ARM_COMPLETE.json"]),
        ]
        assert all(u[3] == 1_800 for u in uploads)

    def test_worker_summary_layout(self, pod_env, tmp_path, monkeypatch):
        cu, hp, chain, events = _make_fakes(tmp_path)
        _install(monkeypatch, cu, hp)
        aw.main(_argv(pod_env, self.ARMS))
        summary = json.loads(
            (pod_env.results / "worker_summary.json").read_text())
        assert summary["schema_version"] == "uad_worker_summary_v1"
        assert summary["run_id"] == "r1"
        assert summary["uad_gpu"] == "0"
        assert summary["n_arms"] == 3
        assert summary["n_complete"] == 3 and summary["n_failed"] == 0
        assert summary["error"] is None
        for row in summary["arms"]:
            assert set(row) == {"arm", "status", "seconds", "receipt",
                                "error"}
            assert row["status"] == "complete"
            assert row["receipt"] == aw.receipt_rel("r1", row["arm"])

    def test_prune_and_results_mirror(self, pod_env, tmp_path, monkeypatch):
        cu, hp, chain, events = _make_fakes(tmp_path)
        _install(monkeypatch, cu, hp)
        aw.main(_argv(pod_env, self.ARMS))
        for arm in self.ARMS.split(","):
            work = pod_env.workdir / "r1" / "arms" / arm
            assert not (work / "merged").exists()
            assert not (work / "eval_work").exists()
            assert not (work / "eft-r32" / "run").exists()
            assert not (work / "receipt").exists()
            # pins mirrored into the bellhop-pulled results dir
            assert (pod_env.results / "pins" / arm / "some_pin.json").is_file()
            # per-arm python-side log captured
            assert (pod_env.results / "logs" / f"arm-{arm}.log").is_file()

    def test_abort_on_first_failure(self, pod_env, tmp_path, monkeypatch):
        cu, hp, chain, events = _make_fakes(
            tmp_path, fail_on="control_d0__coin_d2pct")
        _install(monkeypatch, cu, hp)
        with pytest.raises(RuntimeError, match="boom"):
            aw.main(_argv(pod_env, self.ARMS))
        # third arm never started
        assert [e for e in events if e[0] == "run"] == [
            ("run", "control_d0__baseline"),
            ("run", "control_d0__coin_d2pct"),
        ]
        # summary still written (results are pulled on RemoteJobError)
        summary = json.loads(
            (pod_env.results / "worker_summary.json").read_text())
        statuses = {r["arm"]: r["status"] for r in summary["arms"]}
        assert statuses == {
            "control_d0__baseline": "complete",
            "control_d0__coin_d2pct": "failed",
            "coin_d8m__baseline": "not_started",
        }
        failed = [r for r in summary["arms"] if r["status"] == "failed"][0]
        assert "boom" in failed["error"] and failed["receipt"] is None
        assert summary["n_complete"] == 1 and summary["n_failed"] == 1
        assert "boom" in summary["error"]

    def test_missing_arm_complete_is_a_loud_error(self, pod_env, tmp_path,
                                                  monkeypatch):
        cu, hp, chain, events = _make_fakes(
            tmp_path, skip_receipt_for="control_d0__baseline")
        _install(monkeypatch, cu, hp)
        with pytest.raises(RuntimeError, match="refusing to post"):
            aw.main(_argv(pod_env, "control_d0__baseline"))
        # and no receipt upload happened
        assert not [c for c in chain.calls if c[0] == "upload"]

    def test_preflight_missing_env(self, pod_env, tmp_path, monkeypatch):
        cu, hp, chain, events = _make_fakes(tmp_path)
        _install(monkeypatch, cu, hp)
        monkeypatch.delenv("SCIMT_SOURCE_COMMIT")
        with pytest.raises(RuntimeError, match="SCIMT_SOURCE_COMMIT"):
            aw.main(_argv(pod_env, "control_d0__baseline"))
        assert not events  # nothing spent

    def test_preflight_workdir_outside_runtime_root(self, pod_env, tmp_path,
                                                    monkeypatch):
        cu, hp, chain, events = _make_fakes(tmp_path)
        _install(monkeypatch, cu, hp)
        monkeypatch.setenv("SCIMT_RUNTIME_ROOT", str(tmp_path / "elsewhere"))
        with pytest.raises(RuntimeError, match="SCIMT_RUNTIME_ROOT"):
            aw.main(_argv(pod_env, "control_d0__baseline"))

    def test_env_file_neutralized_fails_loud_without_gcs_base(
            self, pod_env, tmp_path, monkeypatch):
        cu, hp, chain, events = _make_fakes(tmp_path)
        _install(monkeypatch, cu, hp)
        aw._neutralize_env_file(chain)
        monkeypatch.delenv("SCIMT_GCS_BASE")
        with pytest.raises(RuntimeError, match="SCIMT_GCS_BASE"):
            chain.load_env_file()

    def test_dry_run_spends_nothing(self, pod_env, tmp_path, monkeypatch,
                                    capsys):
        cu, hp, chain, events = _make_fakes(tmp_path)
        _install(monkeypatch, cu, hp)
        aw.main(["--run-id", "r1", "--arms",
                 "control_d0__baseline,control_d0__coin_d2pct", "--dry-run"])
        plan = json.loads(capsys.readouterr().out)
        assert plan["schema_version"] == "uad_worker_plan_v1"
        assert plan["parents"] == ["control_d0"]
        assert plan["arms"][1]["receipt"] == (
            "token-scaling-4b-uad/r1/control_d0/coin_d2pct/"
            "ARM_COMPLETE.json")
        assert not events and not chain.calls

    def test_write_probe_failure_aborts(self, tmp_path, monkeypatch):
        def broken_fsync(fd):
            raise OSError(122, "Disk quota exceeded")

        monkeypatch.setattr(aw, "WRITE_PROBE_BYTES", 1)
        monkeypatch.setattr(aw.os, "fsync", broken_fsync)
        with pytest.raises(RuntimeError, match="write probe failed"):
            aw.write_probe(tmp_path / "probe-root")
        assert not (tmp_path / "probe-root" / ".write-probe").exists()

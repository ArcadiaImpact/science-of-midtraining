"""uad Bellhop dispatcher (BELLHOP_PORT.md §7 T3) — CPU-only unit tests.

Covers: worklist partitioning (disjoint, parent-grouped, size cap, canary
first), receipt-based idempotent re-dispatch with a fake rclone lister,
the Semaphore(max_pods) cap, capacity-error retry vs non-capacity abort
(fake bellhop), the signed-off/dry-run gates, the RUNPOD_API_KEY preflight
(config.toml beats the injected pod-scoped key), and the frozen T1/T2
interfaces (PodJob field use, worker CLI string, ARM_COMPLETE receipt path).

bellhop, scimt.train.podjob (T1) and pod_setup (T2) are faked via
sys.modules injection — they are built in parallel; these tests pin the
frozen contracts, not their implementations.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import os
import shlex
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BELLHOP = REPO_ROOT / "experiments/prior_coins/dispatch_unambiguous_dose/bellhop"


def _load_dispatch():
    if "uad_bellhop_dispatch" in sys.modules:
        return sys.modules["uad_bellhop_dispatch"]
    spec = importlib.util.spec_from_file_location(
        "uad_bellhop_dispatch", BELLHOP / "dispatch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["uad_bellhop_dispatch"] = module
    spec.loader.exec_module(module)
    return module


dp = _load_dispatch()
cu = dp.load_chain_uad()

RUN_ID = "20260825T000000Z"


# ---------------------------------------------------------------------------
# Fakes (bellhop / T1 podjob / T2 pod_setup)
# ---------------------------------------------------------------------------

class CapacityError(Exception):
    pass


@pytest.fixture
def fake_bellhop(monkeypatch):
    mod = types.ModuleType("bellhop")
    mod.is_capacity_error = lambda err: isinstance(err, CapacityError)
    monkeypatch.setitem(sys.modules, "bellhop", mod)
    return mod


class PodJobCalls:
    """Recorder shared between the fake T1 module and the test body."""

    def __init__(self):
        self.staged: list[Path] = []
        self.jobs = []            # submitted (successfully completed) jobs
        self.events = []          # ("start"|"end", slug)
        self.concurrent = 0
        self.max_concurrent = 0
        self.fail = {}            # slug-suffix -> list of exceptions to raise


@pytest.fixture
def fake_podjob(monkeypatch):
    calls = PodJobCalls()
    mod = types.ModuleType("scimt.train.podjob")

    @dataclasses.dataclass(frozen=True)
    class PodJob:                       # §7 frozen signature
        pod: object
        slug: str
        setup: str
        run: str
        out_dir: Path
        results_subdir: str
        env: dict = dataclasses.field(default_factory=dict)

    def stage_transfer(out_dir):
        calls.staged.append(Path(out_dir))
        return types.SimpleNamespace(
            wheel_rel="experiments/.wheel/scimt-0.0-py3-none-any.whl",
            manifest_rel="experiments/.wheel/.scimt-source.json")

    async def submit(job):
        calls.concurrent += 1
        calls.max_concurrent = max(calls.max_concurrent, calls.concurrent)
        calls.events.append(("start", job.slug))
        try:
            await asyncio.sleep(0.005)
            for suffix, errors in calls.fail.items():
                if job.slug.endswith(suffix) and errors:
                    raise errors.pop(0)
            calls.jobs.append(job)
        finally:
            calls.concurrent -= 1
            calls.events.append(("end", job.slug))

    mod.PodJob, mod.submit, mod.stage_transfer = PodJob, submit, stage_transfer
    monkeypatch.setitem(sys.modules, "scimt.train.podjob", mod)
    return calls


@pytest.fixture
def fake_pod_setup(monkeypatch):
    mod = types.ModuleType("uad_pod_setup")
    mod.build_setup = lambda wheel_rel: f"SETUP<<{wheel_rel}>>"
    monkeypatch.setitem(sys.modules, "uad_pod_setup", mod)
    return mod


@pytest.fixture
def quiet_sweep(monkeypatch):
    async def _ok():
        return None
    monkeypatch.setattr(dp, "_orphan_sweep", _ok)


@pytest.fixture
def rp_config(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('apikey = "rp_from_config"\n')
    monkeypatch.setenv("RUNPOD_API_KEY", "injected-pod-scoped-bogus")
    return path


def _cfg(tmp_path, rp_config, **kw):
    defaults = dict(run_id=RUN_ID, signed_off=True, scan_receipts=False,
                    capacity_backoff_s=0.001, out_root=tmp_path / "runs",
                    runpod_config_path=rp_config)
    defaults.update(kw)
    return dp.DispatchConfig(**defaults)


def _arms_of(job):
    argv = shlex.split(job.run)
    return argv[argv.index("--arms") + 1].split(",")


# ---------------------------------------------------------------------------
# Plan + partitioning
# ---------------------------------------------------------------------------

def test_plan_is_the_full_55_arm_grid(tmp_path):
    cfg = dp.DispatchConfig(run_id=RUN_ID, dry_run=True, out_root=tmp_path)
    arms = dp.plan_arms(cfg)
    assert arms == cu.planned_arms()
    assert len(arms) == 55


def test_worklists_disjoint_parent_grouped_capped():
    arms = cu.planned_arms()
    worklists = dp.build_worklists(arms)
    flat = [a for w in worklists for a in w.arms]
    assert sorted(flat) == sorted(arms)          # complete
    assert len(set(flat)) == len(flat)           # disjoint
    for w in worklists:
        assert len(w.arms) <= dp.MAX_ARMS_PER_POD
        if not w.canary:                          # single parent per pod
            assert {a.split("__")[0] for a in w.arms} == {w.parent}


def test_canary_worklist_is_first_and_exact():
    worklists = dp.build_worklists(cu.planned_arms())
    assert worklists[0].canary
    assert worklists[0].index == 0
    assert worklists[0].arms == dp.CANARY_ARMS
    assert not any(w.canary for w in worklists[1:])
    # canary arms appear nowhere else
    rest = [a for w in worklists[1:] for a in w.arms]
    assert not set(dp.CANARY_ARMS) & set(rest)


def test_baseline_first_within_each_parent():
    worklists = dp.build_worklists(cu.planned_arms())
    first_chunk = {}
    for w in worklists[1:]:
        first_chunk.setdefault(w.parent, w)
    for parent, w in first_chunk.items():
        if parent == "control_d0":      # its baseline rides the canary
            continue
        assert w.arms[0] == f"{parent}__baseline"


def test_receipted_arms_are_not_rescheduled():
    arms = cu.planned_arms()
    receipted = {"control_d0__baseline", "control_d0__coin_d2pct",
                 "coin_d8m__baseline"}
    remaining = [a for a in arms if a not in receipted]
    worklists = dp.build_worklists(remaining)
    flat = [a for w in worklists for a in w.arms]
    assert not set(flat) & receipted
    assert sorted(flat) == sorted(remaining)
    # both canary arms receipted -> no canary gate on re-dispatch
    assert not any(w.canary for w in worklists)


# ---------------------------------------------------------------------------
# Receipt scan (fake rclone lister)
# ---------------------------------------------------------------------------

def test_receipt_rel_is_the_frozen_path():
    assert dp.receipt_rel(RUN_ID, "control_d0__coin_d2pct") == (
        f"token-scaling-4b-uad/{RUN_ID}/control_d0/coin_d2pct/"
        "ARM_COMPLETE.json")
    assert dp.receipt_rel(RUN_ID, "coin_d8m__baseline") == (
        f"token-scaling-4b-uad/{RUN_ID}/coin_d8m/baseline/ARM_COMPLETE.json")
    assert dp.RECEIPT_NAME == "ARM_COMPLETE.json"


def _scan(monkeypatch, lines):
    monkeypatch.setattr(cu.chain, "gcs_base", lambda: "gcs:test-bucket/base")

    async def lsf(remote_dir):
        assert remote_dir == f"gcs:test-bucket/base/{cu.RUN_PREFIX}/{RUN_ID}"
        return lines

    monkeypatch.setattr(dp, "_rclone_lsf", lsf)
    return asyncio.run(dp.scan_receipts(RUN_ID, cu.planned_arms()))


def test_scan_receipts_maps_paths_to_arm_ids(monkeypatch):
    got = _scan(monkeypatch, [
        "control_d0/baseline/ARM_COMPLETE.json",
        "coin_d8m/charter_d0.2pct_s43/ARM_COMPLETE.json",
    ])
    assert got == {"control_d0__baseline", "coin_d8m__charter_d0.2pct_s43"}


def test_scan_receipts_empty_prefix_is_fine(monkeypatch):
    assert _scan(monkeypatch, []) == set()


def test_scan_receipts_unplanned_arm_is_loud(monkeypatch):
    with pytest.raises(RuntimeError, match="R2"):
        _scan(monkeypatch, ["control_d0/coin_d99pct/ARM_COMPLETE.json"])


def test_scan_receipts_malformed_path_is_loud(monkeypatch):
    with pytest.raises(RuntimeError, match="unexpected receipt path"):
        _scan(monkeypatch, ["ARM_COMPLETE.json"])


# ---------------------------------------------------------------------------
# PodJob assembly (frozen T1/T2 interfaces)
# ---------------------------------------------------------------------------

def test_build_pod_job_honors_frozen_interfaces(tmp_path, rp_config,
                                                fake_podjob, fake_pod_setup):
    cfg = _cfg(tmp_path, rp_config)
    w = dp.Worklist(index=3, parent="coin_d8m",
                    arms=("coin_d8m__baseline", "coin_d8m__coin_d1pct"))
    job = dp.build_pod_job(w, cfg)
    # T2 frozen worker CLI, run from the pushed checkout root
    assert job.run == (
        "/workspace/venv-train/bin/python3 "
        "experiments/prior_coins/dispatch_unambiguous_dose/bellhop/"
        f"arm_worker.py --run-id {RUN_ID} "
        "--arms coin_d8m__baseline,coin_d8m__coin_d1pct --signed-off")
    # setup built from the staged wheel (T2 build_setup(wheel_rel))
    assert job.setup == "SETUP<<experiments/.wheel/scimt-0.0-py3-none-any.whl>>"
    assert fake_podjob.staged == [job.out_dir]
    assert job.out_dir == cfg.out_root / RUN_ID / job.slug
    # §1 pod shape
    assert job.slug == f"uad-{RUN_ID}-w03"
    assert (job.pod.gpu, job.pod.gpu_count) == ("H200", 1)
    assert (job.pod.cloud, job.pod.disk_gb) == ("SECURE", 200)
    assert job.pod.max_hours == 16.0
    assert job.results_subdir == dp.RESULTS_SUBDIR
    assert job.env == {"UAD_RESULTS_DIR": dp.RESULTS_SUBDIR}


def test_canary_pod_gets_the_smoke_ttl(tmp_path, rp_config,
                                       fake_podjob, fake_pod_setup):
    cfg = _cfg(tmp_path, rp_config)
    w = dp.Worklist(index=0, parent="control_d0",
                    arms=dp.CANARY_ARMS, canary=True)
    job = dp.build_pod_job(w, cfg)
    assert job.pod.max_hours == dp.CANARY_MAX_HOURS == 5.0


# ---------------------------------------------------------------------------
# dispatch(): gates, semaphore, retry, abort
# ---------------------------------------------------------------------------

def test_dry_run_plans_everything_and_spends_nothing(tmp_path, rp_config,
                                                     fake_podjob,
                                                     fake_pod_setup,
                                                     monkeypatch):
    cfg = _cfg(tmp_path, rp_config, dry_run=True, signed_off=False)
    report = asyncio.run(dp.dispatch(cfg))
    assert report.dry_run and len(report.planned) == 55
    assert report.remaining == report.planned      # scan_receipts=False
    assert report.worklists[0].canary
    assert not report.results
    assert not fake_podjob.jobs and not fake_podjob.staged
    # the injected bogus key was never overwritten (no preflight on dry-run)
    assert os.environ["RUNPOD_API_KEY"] == "injected-pod-scoped-bogus"


def test_unsigned_dispatch_refuses_before_spending(tmp_path, rp_config,
                                                   fake_podjob,
                                                   fake_pod_setup):
    cfg = _cfg(tmp_path, rp_config, signed_off=False)
    with pytest.raises(RuntimeError, match="REFUSING"):
        asyncio.run(dp.dispatch(cfg))
    assert not fake_podjob.jobs


def test_full_dispatch_caps_pods_and_gates_on_canary(tmp_path, rp_config,
                                                     fake_bellhop,
                                                     fake_podjob,
                                                     fake_pod_setup,
                                                     quiet_sweep):
    cfg = _cfg(tmp_path, rp_config)
    report = asyncio.run(dp.dispatch(cfg))
    # every arm dispatched exactly once
    dispatched = [a for j in fake_podjob.jobs for a in _arms_of(j)]
    assert sorted(dispatched) == sorted(cu.planned_arms())
    # Semaphore(2) honored
    assert fake_podjob.max_concurrent <= 2
    assert fake_podjob.max_concurrent == 2         # fan-out actually parallel
    # canary ran alone, to completion, before any fan-out start
    events = fake_podjob.events
    assert events[0] == ("start", f"uad-{RUN_ID}-w00")
    assert events[1] == ("end", f"uad-{RUN_ID}-w00")
    # RUNPOD_API_KEY preflight: config.toml beat the injected key
    assert os.environ["RUNPOD_API_KEY"] == "rp_from_config"
    assert all(r.status == "ok" for r in report.results)
    assert report.orphan_sweep == "clean"
    # report written next to the staging dirs
    assert list((cfg.out_root / RUN_ID).glob("dispatch-*.json"))


def test_capacity_errors_retry_in_slot(tmp_path, rp_config, fake_bellhop,
                                       fake_podjob, fake_pod_setup,
                                       quiet_sweep):
    fake_podjob.fail["-w00"] = [CapacityError("no H200 stock"),
                                CapacityError("still none")]
    cfg = _cfg(tmp_path, rp_config)
    report = asyncio.run(dp.dispatch(cfg))
    canary_result = next(r for r in report.results if r.worklist.canary)
    assert canary_result.status == "ok" and canary_result.attempts == 3
    assert all(r.status == "ok" for r in report.results)


def test_capacity_retries_exhaust_loudly(tmp_path, rp_config, fake_bellhop,
                                         fake_podjob, fake_pod_setup,
                                         quiet_sweep):
    fake_podjob.fail["-w00"] = [CapacityError(f"strike {i}")
                                for i in range(10)]
    cfg = _cfg(tmp_path, rp_config, capacity_retries=2)
    with pytest.raises(RuntimeError, match="canary worklist failed"):
        asyncio.run(dp.dispatch(cfg))
    assert not fake_podjob.jobs                    # no fan-out


def test_non_capacity_failure_stops_new_worklists(tmp_path, rp_config,
                                                  fake_bellhop, fake_podjob,
                                                  fake_pod_setup,
                                                  quiet_sweep):
    fake_podjob.fail["-w01"] = [ValueError("RemoteJobError: worker died")]
    cfg = _cfg(tmp_path, rp_config, max_pods=1)    # deterministic ordering
    with pytest.raises(RuntimeError, match="did not complete"):
        asyncio.run(dp.dispatch(cfg))
    statuses = {}
    report_files = list((cfg.out_root / RUN_ID).glob("dispatch-*.json"))
    assert report_files                            # report still written
    report = json.loads(report_files[0].read_text())
    for row in report["results"]:
        statuses[row["worklist"]["index"]] = row["status"]
    assert statuses[0] == "ok"                     # canary
    assert statuses[1] == "failed"
    assert all(s == "skipped" for i, s in statuses.items() if i > 1)
    assert len(statuses) == len(report["worklists"])


def test_canary_failure_blocks_fanout(tmp_path, rp_config, fake_bellhop,
                                      fake_podjob, fake_pod_setup,
                                      quiet_sweep):
    fake_podjob.fail["-w00"] = [ValueError("setup exploded")]
    cfg = _cfg(tmp_path, rp_config)
    with pytest.raises(RuntimeError, match="canary worklist failed"):
        asyncio.run(dp.dispatch(cfg))
    assert not fake_podjob.jobs


# ---------------------------------------------------------------------------
# RUNPOD_API_KEY preflight
# ---------------------------------------------------------------------------

def test_preflight_overwrites_injected_key(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "injected-bogus")
    path = tmp_path / "config.toml"
    path.write_text('somefield = 1\napikey = "rp_real_key"\n')
    dp.preflight_runpod_api_key(path)
    assert os.environ["RUNPOD_API_KEY"] == "rp_real_key"


def test_preflight_missing_config_is_loud(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        dp.preflight_runpod_api_key(tmp_path / "nope.toml")


def test_preflight_empty_key_is_loud(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('apikey = ""\n')
    with pytest.raises(RuntimeError, match="apikey"):
        dp.preflight_runpod_api_key(path)


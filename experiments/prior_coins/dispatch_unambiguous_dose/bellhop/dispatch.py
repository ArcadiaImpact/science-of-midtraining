"""uad Bellhop dispatcher — devbox-side, async-native (BELLHOP_PORT.md §7 T3).

Plans the 55 uad arm invocations (``chain_uad.planned_arms``) into
parent-grouped worklists, scans GCS for per-arm ``ARM_COMPLETE.json``
receipts so a re-dispatch schedules only missing arms, and submits each
worklist as one ephemeral 1xH200 Bellhop pod via the T1 seam
(``scimt.train.podjob``). Concurrency is owned here — an
``asyncio.Semaphore(max_pods)`` with per-slot capacity-error retry and a
wave-0 canary gate (the SPEC §4b smoke: ``control_d0__baseline`` +
``control_d0__coin_d2pct`` as a one-pod worklist) before any fan-out.

Repo conventions honored: no CLI — the entry point is the awaitable
``dispatch(DispatchConfig)``; hparams live in the frozen config dataclass;
a run that cannot work (no sign-off, missing RunPod key, mis-keyed GCS
receipt) raises before spending compute.

Frozen cross-task interfaces (BELLHOP_PORT.md §7):

- T1: ``scimt.train.podjob`` — ``PodJob(pod, slug, setup, run, out_dir,
  results_subdir, env)``, ``await submit(job)``, and
  ``stage_transfer(out_dir) -> TransferBundle`` (exposes ``wheel_rel``).
- T2: ``bellhop/pod_setup.py:build_setup(wheel_rel) -> str``; worker CLI
  ``/workspace/venv-train/bin/python3 arm_worker.py --run-id <id>
  --arms <a,b,c> --signed-off``; the GCS receipt path
  ``token-scaling-4b-uad/<run-id>/<parent>/<leaf>/ARM_COMPLETE.json``
  (:func:`receipt_rel` is the shared constant both sides assert against).

RUNPOD_API_KEY trap (§0 / crab CLAUDE.md): RunPod injects a pod-scoped
``RUNPOD_API_KEY`` that yields 403s. :func:`preflight_runpod_api_key`
OVERWRITES the env var from ``~/.runpod/config.toml`` before bellhop is
ever imported; a missing/empty config key is a loud error.

Usage (devbox, e.g. from a tiny asyncio.run() driver or a notebook)::

    from dispatch import DispatchConfig, dispatch
    report = await dispatch(DispatchConfig(run_id="20260825T120000Z",
                                           dry_run=True))       # plan only
    report = await dispatch(DispatchConfig(run_id="...", signed_off=True))
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import logging
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scimt.train.axolotl import PodSpec

logger = logging.getLogger("uad.bellhop.dispatch")

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = HERE.parents[3]
#: devbox-side staging/results root (gitignored; wheel + manifest + pulled
#: pod results land here, under REPO_ROOT as the T1 seam requires).
RUNS_DIR = HERE / "runs"

# ---------------------------------------------------------------------------
# Frozen constants (BELLHOP_PORT.md §1/§4/§6/§7)
# ---------------------------------------------------------------------------

#: Jonathan's pod cap — the dispatcher's own semaphore, not run_many's (§6).
MAX_PODS = 2
#: worklist size cap per pod (§1: parent-grouped worklists of ~8-10 arms).
MAX_ARMS_PER_POD = 10
#: the §4 smoke/canary worklist: smallest real pair exercising both worker
#: branches (pure-eval baseline + train-then-eval) on the no-midtrain parent.
CANARY_ARMS = ("control_d0__baseline", "control_d0__coin_d2pct")
CANARY_MAX_HOURS = 5.0
#: T2<->T3 frozen receipt filename; full path via :func:`receipt_rel`.
RECEIPT_NAME = "ARM_COMPLETE.json"
#: T2 frozen worker entry point (the checkout is pushed; run from its root).
WORKER_PYTHON = "/workspace/venv-train/bin/python3"
WORKER_REL = ("experiments/prior_coins/dispatch_unambiguous_dose/"
              "bellhop/arm_worker.py")
#: pod-side dir bellhop pulls back (run.log is tee'd there by bellhop; the
#: worker mirrors logs/pins/worker_summary.json into it). Derived per job in
#: build_pod_job: checkout-relative sibling ``../uad-results/<slug>``, passed
#: to the worker via --results-dir so one value drives both sides.
#: bellhop names pods "scimt-<slug>"; our slugs start "uad-" so the orphan
#: sweep greps for this prefix.
POD_NAME_PREFIX = "scimt-uad-"

#: the worklist pod (§1) — config-first, one place. A deliberate, documented
#: deviation from "the stage template declares its own pod": the dispatch
#: unit is an arm chain (train + eval + publish), so the pod belongs to the
#: job, not the (pod-less) eft_dispatch_v4_wide_4b template.
UAD_POD = PodSpec(
    gpu="H200", gpu_count=1, cloud="SECURE", disk_gb=200,
    max_hours=16.0, requirements="requirements/pod-h200.txt",
    checkpoint_bus="gcs",   # nominal; the worker's upload_and_pin is the bus
)


# ---------------------------------------------------------------------------
# Read-only sibling-module loaders (the established _load_tsl_chain pattern;
# sys.modules first so tests can inject fakes)
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_chain_uad() -> Any:
    """The uad chain, imported read-only — plan authority (R2 for free)."""
    return _load_module("chain_uad", EXP / "pod" / "chain_uad.py")


def load_pod_setup() -> Any:
    """T2's setup-string builder (``build_setup(wheel_rel) -> str``)."""
    return _load_module("uad_pod_setup", HERE / "pod_setup.py")


# ---------------------------------------------------------------------------
# Config / report objects
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DispatchConfig:
    """Everything a dispatch run needs; unknown keys are a TypeError."""

    run_id: str                       # UTC id shared across the grid
    signed_off: bool = False          # required to spend GPU (chain parity)
    dry_run: bool = False             # plan + partition only; spends nothing
    max_pods: int = MAX_PODS
    max_arms_per_pod: int = MAX_ARMS_PER_POD
    scan_receipts: bool = True        # rclone lsf ARM_COMPLETE receipts
    arms: tuple[str, ...] | None = None   # subset override; None = full plan
    capacity_retries: int = 8         # per-slot re-provision attempts
    capacity_backoff_s: float = 60.0  # base backoff (exponential, capped)
    runpod_config_path: Path | None = None  # default ~/.runpod/config.toml
    out_root: Path = RUNS_DIR

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if self.max_pods < 1:
            raise ValueError(f"max_pods={self.max_pods} must be >= 1")
        if self.max_arms_per_pod < 1:
            raise ValueError("max_arms_per_pod must be >= 1")


@dataclasses.dataclass(frozen=True)
class Worklist:
    index: int          # slot index; 0 = canary
    parent: str         # every worklist is single-parent ("mixed" = canary)
    arms: tuple[str, ...]
    canary: bool = False


@dataclasses.dataclass
class SlotResult:
    worklist: Worklist
    status: str                 # "ok" | "failed" | "skipped"
    attempts: int = 0
    error: str | None = None


@dataclasses.dataclass
class DispatchReport:
    run_id: str
    dry_run: bool
    planned: list[str]          # full plan (post `arms` override)
    receipted: list[str]        # arms skipped via GCS receipts
    remaining: list[str]
    worklists: list[Worklist]   # canary (if any) first
    results: list[SlotResult] = dataclasses.field(default_factory=list)
    orphan_sweep: str | None = None   # "clean" | None (dry-run/failed early)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# RUNPOD_API_KEY preflight (§0 trap)
# ---------------------------------------------------------------------------

def preflight_runpod_api_key(config_path: Path | None = None) -> None:
    """Overwrite ``RUNPOD_API_KEY`` from ``~/.runpod/config.toml``.

    RunPod injects a pod-scoped key into this box's env that 403s the real
    API; the config.toml key always wins. Must run before bellhop (or
    runpodctl via env) is used. Loud on a missing/empty config key.
    """
    path = config_path or Path.home() / ".runpod" / "config.toml"
    if not path.is_file():
        raise RuntimeError(
            f"RunPod config {path} missing — cannot preflight "
            "RUNPOD_API_KEY (the injected pod-scoped key 403s)")
    key = ""
    for line in path.read_text().splitlines():
        match = re.match(r"""\s*apikey\s*=\s*['"]?([^'"\s]+)['"]?\s*$""",
                         line, flags=re.IGNORECASE)
        if match:
            key = match.group(1)
            break
    if not key:
        raise RuntimeError(
            f"no non-empty 'apikey' field in {path} — refusing to fall "
            "back to the injected RUNPOD_API_KEY")
    os.environ["RUNPOD_API_KEY"] = key   # overwrite, never setdefault


# ---------------------------------------------------------------------------
# Plan + receipts + partitioning
# ---------------------------------------------------------------------------

def plan_arms(cfg: DispatchConfig) -> list[str]:
    """Full arm plan with the R2 disjointness preflight (chain authority)."""
    cu = load_chain_uad()
    arms = list(cfg.arms) if cfg.arms is not None else cu.planned_arms()
    # build_plan parses every id (loud on malformed/forbidden arms) and
    # enforces pairwise path disjointness across the whole plan (R2).
    cu.build_plan(cfg.run_id, arms, cu.DEFAULT_WORKDIR, cu.load_manifest())
    return arms


def receipt_rel(run_id: str, arm_id: str) -> str:
    """T2<->T3 frozen GCS receipt path (relative to the rclone base):
    ``token-scaling-4b-uad/<run-id>/<parent>/<leaf>/ARM_COMPLETE.json``."""
    cu = load_chain_uad()
    arm = cu.parse_arm_id(arm_id)
    return f"{cu.RUN_PREFIX}/{run_id}/{arm.parent}/{arm.leaf}/{RECEIPT_NAME}"


async def _rclone_lsf(remote_dir: str) -> list[str]:
    """Relative file paths under ``remote_dir`` (recursive); [] if the
    prefix does not exist yet (rclone returns empty on fresh runs)."""
    proc = await asyncio.create_subprocess_exec(
        "rclone", "lsf", "--recursive", "--files-only",
        "--include", RECEIPT_NAME, remote_dir,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode not in (0, 3):   # 3 = directory not found
        raise RuntimeError(
            f"rclone lsf failed (rc={proc.returncode}): "
            f"{err.decode(errors='replace')[-500:]}")
    return [ln.strip() for ln in out.decode().splitlines() if ln.strip()]


async def scan_receipts(run_id: str, planned: Iterable[str]) -> set[str]:
    """Arm ids with an ``ARM_COMPLETE.json`` receipt on GCS.

    A receipt that does not map onto a planned arm is a LOUD error — a
    mis-keyed receipt is the R2 failure mode, never something to skip past.
    """
    cu = load_chain_uad()
    base = cu.chain.gcs_base()
    remote_dir = f"{base}/{cu.RUN_PREFIX}/{run_id}"
    planned_set = set(planned)
    completed: set[str] = set()
    for rel in await _rclone_lsf(remote_dir):
        parts = rel.split("/")
        if len(parts) != 3 or parts[-1] != RECEIPT_NAME:
            raise RuntimeError(
                f"unexpected receipt path {rel!r} under {cu.RUN_PREFIX}/"
                f"{run_id} (want <parent>/<leaf>/{RECEIPT_NAME})")
        arm_id = f"{parts[0]}__{parts[1]}"
        if arm_id not in planned_set:
            raise RuntimeError(
                f"receipt {rel!r} maps to arm {arm_id!r} which is not in "
                "the plan — mis-keyed receipts are an R2 violation")
        completed.add(arm_id)
    return completed


def build_worklists(remaining: list[str],
                    max_arms_per_pod: int = MAX_ARMS_PER_POD,
                    ) -> list[Worklist]:
    """Partition remaining arms into worklists: canary first, then
    parent-grouped balanced chunks of <= max_arms_per_pod, baselines first
    within a parent. Plan order is otherwise preserved."""
    cu = load_chain_uad()
    remaining = list(dict.fromkeys(remaining))   # de-dupe, keep order

    canary = tuple(a for a in CANARY_ARMS if a in remaining)
    rest = [a for a in remaining if a not in canary]

    groups: dict[str, list[str]] = {}
    for arm_id in rest:
        arm = cu.parse_arm_id(arm_id)
        groups.setdefault(arm.parent, []).append(arm_id)

    worklists: list[Worklist] = []
    if canary:
        worklists.append(Worklist(index=0, parent="control_d0",
                                  arms=canary, canary=True))
    index = len(worklists)
    for parent, arms in groups.items():
        # baseline (then anchor) first within the parent, stable otherwise
        arms.sort(key=lambda a: (not a.endswith("__baseline"),
                                 not a.endswith("__anchor_d0pct")))
        n_chunks = -(-len(arms) // max_arms_per_pod)   # ceil
        chunk, rem = divmod(len(arms), n_chunks)       # balanced sizes
        start = 0
        for i in range(n_chunks):
            size = chunk + (1 if i < rem else 0)
            worklists.append(Worklist(index=index, parent=parent,
                                      arms=tuple(arms[start:start + size])))
            index += 1
            start += size
    # invariants: disjoint + complete
    flat = [a for w in worklists for a in w.arms]
    if sorted(flat) != sorted(remaining) or len(set(flat)) != len(flat):
        raise AssertionError("worklist partition is not a disjoint cover")
    return worklists


# ---------------------------------------------------------------------------
# PodJob assembly + slot execution
# ---------------------------------------------------------------------------

def build_pod_job(worklist: Worklist, cfg: DispatchConfig) -> Any:
    """One worklist -> one T1 ``PodJob`` (wheel staged, setup built)."""
    from scimt.train.podjob import PodJob, stage_transfer

    slug = f"uad-{cfg.run_id}-w{worklist.index:02d}"
    out_dir = cfg.out_root / cfg.run_id / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = stage_transfer(out_dir)
    setup = load_pod_setup().build_setup(bundle.wheel_rel)
    pod = (dataclasses.replace(UAD_POD, max_hours=CANARY_MAX_HOURS)
           if worklist.canary else UAD_POD)
    # One value drives both sides of the results seam: bellhop pulls this
    # (checkout-relative, sibling tree per the BellhopExecutor precedent) and
    # the worker is told to mirror into the very same dir — the worker's
    # absolute WORKER_RESULTS_DIR default is for manual/non-bellhop runs.
    results_rel = f"../uad-results/{slug}"
    run_cmd = " ".join([
        WORKER_PYTHON, shlex.quote(WORKER_REL),
        "--run-id", shlex.quote(cfg.run_id),
        "--arms", shlex.quote(",".join(worklist.arms)),
        "--results-dir", shlex.quote(results_rel),
        "--signed-off",
    ])
    return PodJob(
        pod=pod, slug=slug, setup=setup, run=run_cmd, out_dir=out_dir,
        results_subdir=results_rel,
        env={},
    )


@dataclasses.dataclass
class _State:
    stop: bool = False   # set on non-capacity failure: no NEW worklists


async def _run_worklist(worklist: Worklist, cfg: DispatchConfig,
                        sem: asyncio.Semaphore, state: _State) -> SlotResult:
    """One dispatch slot: submit with in-slot capacity-error retry (§6).

    Capacity stock-outs back off and re-provision; any other failure marks
    the run stopped (already-running slots finish; queued slots skip) — a
    failed arm is retried on a NEW pod via re-dispatch, never auto-retried.
    """
    from scimt.train.podjob import submit
    import bellhop   # key preflight has already run

    async with sem:
        if state.stop:
            return SlotResult(worklist, "skipped",
                              error="dispatch stopped by an earlier failure")
        attempts = 0
        while True:
            attempts += 1
            job = build_pod_job(worklist, cfg)
            try:
                logger.info("slot w%02d (%s, %d arms): submitting %s",
                            worklist.index, worklist.parent,
                            len(worklist.arms), job.slug)
                await submit(job)
                logger.info("slot w%02d: complete", worklist.index)
                return SlotResult(worklist, "ok", attempts=attempts)
            except Exception as err:  # noqa: BLE001 — classified below
                if (bellhop.is_capacity_error(err)
                        and attempts <= cfg.capacity_retries):
                    delay = min(cfg.capacity_backoff_s
                                * 2 ** min(attempts - 1, 4), 900.0)
                    logger.warning(
                        "slot w%02d: capacity error (attempt %d/%d), "
                        "retrying in %.0fs: %s", worklist.index, attempts,
                        cfg.capacity_retries + 1, delay, err)
                    await asyncio.sleep(delay)
                    continue
                state.stop = True
                logger.error("slot w%02d: FAILED (%s arms unfinished): %s",
                             worklist.index, ",".join(worklist.arms), err)
                return SlotResult(worklist, "failed", attempts=attempts,
                                  error=f"{type(err).__name__}: {err}")


async def _orphan_sweep() -> None:
    """Post-wave assertion: zero scimt-uad-* pods still exist (§2 row 1)."""
    proc = await asyncio.create_subprocess_exec(
        "runpodctl", "get", "pod",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(
            f"orphan sweep failed: runpodctl get pod rc={proc.returncode}: "
            f"{err.decode(errors='replace')[-300:]} — check for "
            f"{POD_NAME_PREFIX}* pods by hand before walking away")
    orphans = [ln for ln in out.decode().splitlines()
               if POD_NAME_PREFIX in ln]
    if orphans:
        raise RuntimeError(
            f"ORPHAN PODS STILL BILLING: {orphans} — tear them down now")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def dispatch(cfg: DispatchConfig) -> DispatchReport:
    """Plan, partition, and (unless dry_run) run the uad sweep on Bellhop.

    Idempotent by receipts: arms with an ``ARM_COMPLETE.json`` on GCS are
    never re-scheduled, so re-running after a failure dispatches only the
    missing arms. Raises after the orphan sweep if any slot failed.
    """
    planned = plan_arms(cfg)
    receipted = (sorted(await scan_receipts(cfg.run_id, planned))
                 if cfg.scan_receipts else [])
    remaining = [a for a in planned if a not in set(receipted)]
    worklists = build_worklists(remaining, cfg.max_arms_per_pod)
    report = DispatchReport(
        run_id=cfg.run_id, dry_run=cfg.dry_run, planned=planned,
        receipted=receipted, remaining=remaining, worklists=worklists)

    logger.info("plan: %d arms, %d receipted, %d remaining in %d worklists",
                len(planned), len(receipted), len(remaining), len(worklists))
    for w in worklists:
        logger.info("  w%02d %s%s: %s", w.index, w.parent,
                    " [CANARY]" if w.canary else "", ",".join(w.arms))

    if cfg.dry_run:
        return report
    if not remaining:
        logger.info("nothing to dispatch — every arm has a receipt")
        return report
    if not cfg.signed_off:
        raise RuntimeError(
            "REFUSING to spend GPU: DispatchConfig.signed_off is False "
            "(use dry_run=True to inspect the plan)")

    preflight_runpod_api_key(cfg.runpod_config_path)

    sem = asyncio.Semaphore(cfg.max_pods)
    state = _State()
    try:
        canaries = [w for w in worklists if w.canary]
        waves = [w for w in worklists if not w.canary]
        for w in canaries:   # gate: canary runs alone, before any fan-out
            result = await _run_worklist(w, cfg, sem, state)
            report.results.append(result)
        if state.stop:
            raise RuntimeError(
                "canary worklist failed — fan-out not started "
                f"({report.results[-1].error})")
        report.results.extend(await asyncio.gather(
            *(_run_worklist(w, cfg, sem, state) for w in waves)))
    finally:
        try:
            await _orphan_sweep()
            report.orphan_sweep = "clean"
        finally:
            _write_report(cfg, report)

    failed = [r for r in report.results if r.status != "ok"]
    if failed:
        raise RuntimeError(
            f"{len(failed)} worklist(s) did not complete "
            f"({[r.worklist.index for r in failed]}) — re-run dispatch; "
            "receipts make it schedule only the missing arms")
    return report


def _write_report(cfg: DispatchConfig, report: DispatchReport) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = cfg.out_root / cfg.run_id / f"dispatch-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2))
    logger.info("dispatch report: %s", path)

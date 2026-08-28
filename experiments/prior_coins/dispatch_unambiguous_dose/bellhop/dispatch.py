"""uad Bellhop dispatcher — devbox-side, async-native (BELLHOP_PORT.md §7 T3).

Plans the uad arm invocations (``chain_uad.planned_arms``) into
parent-grouped worklists, scans GCS for per-arm ``ARM_COMPLETE.json``
receipts so a re-dispatch schedules only missing arms, and submits each
worklist as one ephemeral 1xH200 Bellhop pod via the T1 seam
(``scimt.train.podjob``). Concurrency is owned here — an
``asyncio.Semaphore(max_pods)`` with per-slot capacity-error retry and a
wave-0 canary gate before any fan-out: the CANARY_TIERS ladder — the SPEC
§4b smoke pair (``control_d0__baseline`` + ``control_d0__coin_d2pct``),
the E8 epoch canary (``control_d0__anchor_d0pct_e5``), and the K1 corpus
canary (``control_d0__coin_d0.2pct_x2.5``) — where each tier present in
the remaining arms runs as its own solo worklist, sequentially, before
any fan-out. Worklists are weight-partitioned (E4/K7): each pod's arm
weights sum to <= MAX_WEIGHT_PER_POD.

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
#: extra fresh-pod attempts for a slot whose REMOTE job died (receipts make
#: the retry safe; small so deterministic bugs still fail the slot fast).
REMOTE_RETRIES = 2
#: worklist size cap per pod (§1: parent-grouped worklists of ~8-10 arms).
MAX_ARMS_PER_POD = 10
#: E4: weighted worklist cap. Arm weight = its train+eval cost in
#: half-hour-ish units: baseline 1 (eval only), standard 2-epoch arms 2,
#: epoch-sweep arms = their epoch count (an e20 arm ~ 10 standard arms of
#: train time), corpus-scaled arms per CORPUS_ARM_WEIGHTS below. 36 ~ ~11 h
#: train+eval on one H200 — inside the 16 h pod TTL with setup/upload
#: headroom. K7 keeps corpus-heavy worklists ~30 weight not by lowering
#: this cap but via the x10 weight bump (see CORPUS_ARM_WEIGHTS).
MAX_WEIGHT_PER_POD = 36
#: SPEC ext. 3 / K6: corpus arm weights = int(2 x N) — step-time parity
#: with the epoch arms (512 x N steps at the same per-step cost). K7: the
#: x10 arm carries +2 over its step parity (22, not 20) — tokenizing and
#: sha+parse-gating the 81,920-row (~215 MB) corpus adds real wallclock —
#: which under MAX_WEIGHT_PER_POD=36 caps corpus-heavy worklists near the
#: ~30-weight comfort line and makes two x10 arms un-co-packable (44 > 36).
CORPUS_ARM_WEIGHTS = {"x2.5": 5, "x5": 10, "x10": 22}
#: the §4 smoke/canary worklist: smallest real pair exercising both worker
#: branches (pure-eval baseline + train-then-eval) on the no-midtrain parent.
CANARY_ARMS = ("control_d0__baseline", "control_d0__coin_d2pct")
#: E8: the epoch-sweep canary — one e5 arm run ALONE before any epoch-arm
#: fan-out; its trainer_state (steps=1280, epoch~5.0), eval dir name and
#: receipt exercise E1-E4 for ~$8.
EPOCH_CANARY_ARMS = ("control_d0__anchor_d0pct_e5",)
#: K1: the corpus-scaling canary — the cheapest corpus arm run ALONE before
#: any corpus-arm fan-out; its _bigcorpus stage, 256-step schedule,
#: trainer_state (steps=1280, epoch~2.0) and receipt exercise every
#: epochs+corpus_mult-keyed seam for ~$8.
CORPUS_CANARY_ARMS = ("control_d0__coin_d0.2pct_x2.5",)
#: gating ladder, in launch order: each tier whose arms are present in the
#: remaining set becomes its own SOLO worklist that must succeed before
#: fan-out (generalizes the original smoke/epoch two-tier carve-out to N
#: tiers; new sweep dimensions append their canary here).
CANARY_TIERS = (CANARY_ARMS, EPOCH_CANARY_ARMS, CORPUS_CANARY_ARMS)
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
# Observed bellhop 0.8 naming: it prefixes "bellhop-" + slug and ignores
# PodConfig.name — the old "scimt-uad-" grep made the sweep a no-op while
# two orphans idled 18h (2026-08-26). Also: bellhop's max_lifetime is
# enforced by a CLIENT-SIDE watchdog in the devbox process, so a killed
# dispatcher orphans its pods — the sweep and an external pod-watch are the
# real backstops, not the TTL.
POD_NAME_PREFIX = "bellhop-uad-"

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
    slug_suffix: str = ""             # disambiguates concurrent dispatchers
                                      # on one run id (pod names + staging
                                      # dirs; GCS keying stays run_id-only)
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


def arm_weight(arm_id: str) -> int:
    """E4: an arm's projected pod cost in weight units — baseline 1 (eval
    only), standard 2-epoch arms 2 (train+eval), epoch-sweep arms = their
    epoch count (train time scales linearly in epochs), corpus-scaled arms
    per CORPUS_ARM_WEIGHTS (int(2 x N) step parity, +2 on x10 for
    tokenization overhead — K6/K7)."""
    cu = load_chain_uad()
    arm = cu.parse_arm_id(arm_id)
    if arm.kind == "baseline":
        return 1
    if arm.corpus_mult != cu.DEFAULT_CORPUS_MULT:
        return CORPUS_ARM_WEIGHTS[arm.corpus_mult]
    return 2 if arm.epochs == cu.DEFAULT_EPOCHS else arm.epochs


def _weighted_chunks(arms: list[str], weights: list[int],
                     max_arms: int, max_weight: int) -> list[list[str]]:
    """Contiguous, weight-balanced chunks respecting BOTH caps (E4).

    Starts from the lower-bound chunk count and increases until every chunk
    fits; within a pass, a chunk closes when its cumulative weight reaches
    the balanced target or it hits ``max_arms``. Terminates: singleton
    chunks always satisfy both caps (max single weight 22 < max_weight)."""
    total = sum(weights)
    n_chunks = max(1, -(-len(arms) // max_arms), -(-total // max_weight))
    while n_chunks <= len(arms):
        chunks: list[list[str]] = []
        chunk_weights: list[int] = []
        current: list[str] = []
        current_weight = 0
        cum = 0
        for i, (arm_id, weight) in enumerate(zip(arms, weights)):
            current.append(arm_id)
            current_weight += weight
            cum += weight
            remaining_items = len(arms) - i - 1
            remaining_chunks = n_chunks - len(chunks) - 1
            if len(chunks) < n_chunks - 1 and (
                    cum >= (len(chunks) + 1) * total / n_chunks
                    or len(current) == max_arms
                    or remaining_items == remaining_chunks):
                chunks.append(current)
                chunk_weights.append(current_weight)
                current, current_weight = [], 0
        if current:
            chunks.append(current)
            chunk_weights.append(current_weight)
        if all(len(c) <= max_arms and w <= max_weight
               for c, w in zip(chunks, chunk_weights)):
            return chunks
        n_chunks += 1
    return [[a] for a in arms]


def build_worklists(remaining: list[str],
                    max_arms_per_pod: int = MAX_ARMS_PER_POD,
                    max_weight_per_pod: int = MAX_WEIGHT_PER_POD,
                    ) -> list[Worklist]:
    """Partition remaining arms into worklists: canaries first (each alone),
    then parent-grouped weight-balanced chunks of <= max_arms_per_pod arms
    AND <= max_weight_per_pod weighted units (E4/K7 — an e20 arm ~ 10 and
    an x10 arm ~ 11 standard arms of train time; uncapped co-packing would
    blow the 16 h pod TTL). Baselines (then anchors) first within a parent;
    plan order is otherwise preserved, so each epoch-parent's heavy
    epoch/corpus arms spread across its worklists rather than co-packing.

    E8 (generalized to N tiers): the CANARY_TIERS ladder — §4 smoke pair,
    epoch canary, corpus canary; each tier present in ``remaining`` becomes
    its own solo canary worklist gating fan-out, in ladder order."""
    cu = load_chain_uad()
    remaining = list(dict.fromkeys(remaining))   # de-dupe, keep order

    canary_tiers = []
    for tier in CANARY_TIERS:
        present = tuple(a for a in tier if a in remaining)
        if present:
            canary_tiers.append(present)
    canary_flat = {a for tier in canary_tiers for a in tier}
    rest = [a for a in remaining if a not in canary_flat]

    groups: dict[str, list[str]] = {}
    for arm_id in rest:
        arm = cu.parse_arm_id(arm_id)
        groups.setdefault(arm.parent, []).append(arm_id)

    worklists: list[Worklist] = []
    for tier in canary_tiers:
        worklists.append(Worklist(index=len(worklists), parent="control_d0",
                                  arms=tier, canary=True))
    index = len(worklists)
    for parent, arms in groups.items():
        # baseline (then anchor) first within the parent, stable otherwise
        arms.sort(key=lambda a: (not a.endswith("__baseline"),
                                 not a.endswith("__anchor_d0pct")))
        weights = [arm_weight(a) for a in arms]
        for chunk in _weighted_chunks(arms, weights, max_arms_per_pod,
                                      max_weight_per_pod):
            worklists.append(Worklist(index=index, parent=parent,
                                      arms=tuple(chunk)))
            index += 1
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

    slug = f"uad-{cfg.run_id}{cfg.slug_suffix}-w{worklist.index:02d}"
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
    # Ephemeral pods have no .env file: everything the worker's env contract
    # (pod_setup.REQUIRED_ENV + SCIMT_RUNTIME_ROOT) needs beyond T1's
    # transport-secret passthrough must ride job.env. Loud KeyError if the
    # devbox env (chain.load_env_file) doesn't carry it.
    return PodJob(
        pod=pod, slug=slug, setup=setup, run=run_cmd, out_dir=out_dir,
        results_subdir=results_rel,
        env={
            "SCIMT_GCS_BASE": os.environ["SCIMT_GCS_BASE"],
            "SCIMT_RUNTIME_ROOT": "/workspace/uad",
            # T1's passthrough forwards only TYPE + SERVICE_ACCOUNT_CREDS;
            # the bucket also needs e.g. BUCKET_POLICY_ONLY (uniform
            # bucket-level access rejects legacy ACLs) — forward the whole
            # rclone remote config so the pod's remote matches the devbox's.
            **{k: v for k, v in os.environ.items()
               if k.startswith("RCLONE_CONFIG_GCS_")},
        },
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
                # Transient RunPod API flakes at provisioning ("Something
                # went wrong", 5xx graphql) deserve the same in-slot retry
                # as capacity: nothing trained yet, nothing is lost. A
                # remote JOB failure (the worker itself exited nonzero) is
                # never retried here — receipts + relaunch own that.
                msg = str(err).lower()
                transient = ("graphql error" in msg
                             and "remote job exited" not in msg)
                # Remote-job deaths get a SMALL retry budget of their own:
                # a fresh pod + receipt dedupe makes the retry safe (done
                # arms skip), and transient stream/SIGPIPE deaths (exit 141
                # mid-setup, 2026-08-25 18:33Z) are indistinguishable from
                # real bugs at this layer — but a deterministic bug must
                # still fail the slot after 2 extra pods, not 9.
                remote_death = "remote job exited" in msg
                budget = (REMOTE_RETRIES if remote_death
                          else cfg.capacity_retries)
                if ((bellhop.is_capacity_error(err) or transient
                     or remote_death) and attempts <= budget):
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
        # gate: each canary tier (CANARY_TIERS: smoke pair, E8 epoch
        # canary, K1 corpus canary) runs ALONE and sequentially, in ladder
        # order, before any fan-out.
        for w in canaries:
            result = await _run_worklist(w, cfg, sem, state)
            report.results.append(result)
            if state.stop:
                break
        if state.stop:
            failed = next(r for r in report.results if r.status == "failed")
            raise RuntimeError(
                "canary worklist failed — fan-out not started "
                f"({failed.error})")
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

"""Pod-side chain for a Dispatch row: sequential training, then pooled work.

One profile-sized pod owns one or more arms (control / charter / coin):

    phases 1-2  each arm's full-param midtrain + Dolci              sequential
    phase 3     requested arms x 4 LoRA cells                       pooled waves
    phase 4+    requested arms x 9 endpoints per eval surface       pooled shards

Nothing crosses pods: an arm's 24 GB checkpoints never leave the machine that
made them until they are published.

Two contracts this file exists to enforce
-----------------------------------------
1. **Steps are DERIVED from the realized mix, never trusted.** Selection stays
   on the pinned Gemma counting basis for every family. Gemma schedules on that
   same count; GLM retokenizes the exact emitted rows with its pinned tokenizer
   and computes `glm_tokens * profile_epochs // 262,144`. Both counts and the
   document count are pinned in SCHEDULE.json, and relaunch requires equality --
   a resumed run that would silently train a different dose is a hard error.
   (python4/midtraining_prop does exactly this; the pattern is borrowed.)
2. **Every phase is resumable and idempotent -- against the SAME run.** Each
   phase writes a sentinel stamped with the run's fingerprint (profile row,
   substrate revision, data commit, dose, seed; contracts.fingerprint) and a
   sentinel is only trusted if the fingerprint matches. An existence-only
   marker once made pod/root reuse able to skip training and publish another
   run's checkpoints under this run's name (2026-08-31 triage, gap #1). Never
   delete a run directory to restart it -- relaunch, and the completed phases
   cost nothing.

The grid row (model x dose) is a profile: FINAL_V1_PROFILE selects it (default
gemma3_12b_50m, the completed run) and artifacts land under
<root>/<profile>/<arm>, so no two rows can ever share resume markers.

Run (on the pod):
    python3 chain.py --arms charter,coin,control --root /workspace/final_v1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = PRIOR_COINS.parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP), str(PRIOR_COINS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402

#: Repo/prefix/commit come from the active profile row (contracts.PROFILE).
#: The commit is a 40-hex pin validated at profile load: a branch name moves
#: under a running campaign (the no-example release will land in this same
#: repo), and the old env-var override defaulting to `main` let the manifest
#: and corpus move together and still validate. Changing data now means a new
#: profile row, not an env var.
DATA_REPO = C.DATA_REPO
DATA_PREFIX = C.DATA_PREFIX
DATA_REVISION = C.DATA_REVISION
N_GPUS = C.N_GPUS
CONSOLIDATOR = REPO_ROOT / "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py"
CONSOLIDATE_TIMEOUT_S = 6 * 60 * 60


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


#: Marker identity is task-local.  Pooled waves concurrently write beneath
#: multiple arm roots, so a process-global current fingerprint can stamp one
#: arm with another arm's identity.  ContextVar preserves the convenient
#: done()/mark() API while copying the correct scope into each asyncio task.
_MARKER_SCOPE: ContextVar[tuple[Path, str, dict] | None] = ContextVar(
    "final_v1_marker_scope", default=None)


@contextmanager
def fingerprint_scope(root: Path, arm: str):
    """Bind marker I/O to exactly one arm root for this task."""
    fingerprint = C.fingerprint(arm)
    root = root.resolve()
    # These are deliberate assertions, not comments: a future refactor cannot
    # enter a charter scope carrying coin's identity without stopping here.
    assert root.name == arm, f"active arm {arm!r} does not own root {root}"
    assert fingerprint["arm"] == arm
    assert fingerprint == C.fingerprint(arm)
    token = _MARKER_SCOPE.set((root, arm, fingerprint))
    try:
        yield
    finally:
        _MARKER_SCOPE.reset(token)


def _active_fingerprint(path: Path) -> dict:
    scope = _MARKER_SCOPE.get()
    if scope is None:
        raise RuntimeError(
            "fingerprint_scope(root, arm) must be active before markers are used")
    root, arm, fingerprint = scope
    resolved = path.resolve()
    assert resolved == root or resolved.is_relative_to(root), (
        f"marker {resolved} is outside active {arm} root {root}")
    assert fingerprint["arm"] == arm, (
        f"active fingerprint is for {fingerprint.get('arm')!r}, not {arm!r}")
    assert fingerprint == C.fingerprint(arm), (
        f"active fingerprint drifted while writing {arm!r}")
    return fingerprint


def done(path: Path) -> bool:
    """A completion marker counts only if it carries THIS run's fingerprint.

    Existence alone is not completion: a reused pod, root or arm name would
    otherwise resume over another run's markers and publish its checkpoints or
    scores under this run's identity, quietly. A marker without a fingerprint
    (pre-2026-08-31 layout) or with a different one is a hard error, never a
    silent re-run -- the operator must decide whether to point at a fresh root
    or deliberately delete the foreign marker.
    """
    if not path.is_file():
        return False
    fingerprint = _active_fingerprint(path)
    try:
        payload = json.loads(path.read_text())
    except ValueError as exc:
        raise RuntimeError(
            f"resume marker {path} is not valid JSON ({exc}); refusing to "
            "guess whether the phase completed"
        ) from exc
    stamped = payload.get("fingerprint") if isinstance(payload, dict) else None
    if stamped is None:
        raise RuntimeError(
            f"resume marker {path} carries no fingerprint (written before "
            "fingerprinting, or by another tool); refusing to trust it. "
            "Use a fresh --root, or delete the marker if you are certain it "
            "belongs to this exact run."
        )
    if stamped != fingerprint:
        drift = sorted(
            k for k in set(stamped) | set(fingerprint)
            if stamped.get(k) != fingerprint.get(k)
        )
        raise RuntimeError(
            f"resume marker {path} was written by a DIFFERENT run -- "
            f"fingerprint disagrees on {drift}: marker "
            f"{ {k: stamped.get(k) for k in drift} } vs this run "
            f"{ {k: fingerprint.get(k) for k in drift} }. Refusing to resume "
            "over another run's artifacts."
        )
    return True


def mark(path: Path, payload: dict) -> None:
    fingerprint = _active_fingerprint(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**payload, "fingerprint": fingerprint},
                              indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


async def in_arm_scope(root: Path, arm: str, operation, *args, **kwargs):
    """Run one async arm operation with marker identity bound task-locally."""
    with fingerprint_scope(root, arm):
        return await operation(*args, **kwargs)


def run_root(base: Path, arm: str) -> Path:
    """<root>/<profile>/<arm>: no two grid rows can share resume markers."""
    return base / C.PROFILE.name / arm


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, want: str, label: str) -> None:
    got = file_sha256(path)
    if got != want:
        raise RuntimeError(
            f"{label}: sha256 {got} != pinned {want} -- refusing to use "
            "unverified bytes"
        )


def run_sync(cmd: list, log_path: Path, env: dict | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        proc = subprocess.run([str(c) for c in cmd], stdout=handle,
                              stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-40:]
        raise RuntimeError(
            f"{cmd[0]} failed ({proc.returncode}); last lines:\n  " + "\n  ".join(tail)
        )


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def require_router_health(run_dir: Path, label: str, *,
                          required: bool = True) -> Path | None:
    """A named GLM safety gate; Gemma's completed path is unchanged.

    ``required=False`` is the one sanctioned exception: a stage whose profile
    detaches RouterHealthPlugin (``midtrain_router_monitor: false``, the 1B
    charter midtrain) produces no telemetry, and that absence is logged rather
    than refused. A stage that DOES run the plugin must still produce it.
    """
    path = run_dir / "router_health.jsonl"
    if not required and not path.is_file():
        log(f"{label}: router monitor detached by profile; no router_health.jsonl")
        return None
    if C.MODEL_FAMILY == "glm45_air" and (
            not path.is_file() or path.stat().st_size == 0):
        raise RuntimeError(
            f"{label}: RouterHealthPlugin produced no non-empty {path}")
    if C.MODEL_FAMILY == "glm45_air":
        rows = [line for line in path.read_text().splitlines() if line.strip()]
        try:
            latest = json.loads(rows[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{label}: router telemetry is not valid JSONL: {path}") from exc
        log(f"{label}: router monitor latest={json.dumps(latest, sort_keys=True)}")
    return path


def consolidate_glm_checkpoint(run_dir: Path, step: int, label: str) -> Path:
    """Make one GLM FSDP2 checkpoint loadable, finalize MTP, then reclaim DCP."""
    if C.MODEL_FAMILY != "glm45_air":
        raise RuntimeError("GLM consolidation called for a non-GLM profile")
    checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
    destination = run_dir / "consolidated" / f"checkpoint-{step}"
    receipt = destination / "GLM_CONSOLIDATED.json"
    if (receipt.is_file() and (destination / "config.json").is_file()
            and list(destination.glob("*.safetensors"))):
        return destination
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"{label}: expected sharded checkpoint {checkpoint}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    weights = list(checkpoint.glob("*.safetensors"))
    log_path = run_dir / f"consolidate_checkpoint-{step}.log"
    if (checkpoint / "config.json").is_file() and weights:
        # Some Axolotl versions additionally emit a complete HF checkpoint.
        for source in weights:
            _link_or_copy(source, destination / source.name)
        for pattern in (
            "model*.json", "config.json", "generation_config.json",
            "tokenizer*", "special_tokens*", "vocab*", "merges*",
        ):
            for source in checkpoint.glob(pattern):
                if source.is_file() and not (destination / source.name).exists():
                    _link_or_copy(source, destination / source.name)
        log_path.write_text("hardlinked complete HF checkpoint\n")
    else:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download(
            repo_id=C.BASE_MODEL_MIRROR, revision=C.BASE_MODEL_REVISION))
        result = subprocess.run(
            [sys.executable, str(CONSOLIDATOR),
             "--checkpoint-dir", str(checkpoint),
             "--base-model", str(base), "--out", str(destination)],
            capture_output=True, text=True, timeout=CONSOLIDATE_TIMEOUT_S,
            check=False)
        log_path.write_text(
            result.stdout + "\n--- STDERR ---\n" + result.stderr)
        if result.returncode:
            raise RuntimeError(
                f"{label}: consolidation failed ({result.returncode}):\n"
                f"{result.stderr[-4000:]}")
    if not (destination / "config.json").is_file() or not list(
            destination.glob("*.safetensors")):
        raise RuntimeError(f"{label}: consolidated checkpoint is incomplete")

    from scimt.train.handoff import finalize_glm4_moe_checkpoint

    mtp = finalize_glm4_moe_checkpoint(destination)
    router = require_router_health(
        run_dir, label,
        required=C.MIDTRAIN_ROUTER_MONITOR or not label.endswith("/midtrain"))
    if router is not None:
        shutil.copy2(router, destination / router.name)
    receipt.write_text(json.dumps({
        "source": str(checkpoint), "destination": str(destination),
        "step": step, "mtp": mtp.as_dict(),
        "safetensor_files": len(list(destination.glob("*.safetensors"))),
    }, indent=2, sort_keys=True) + "\n")

    # The replacement is now locally durable and verified. Delete only this
    # exact checkpoint tree; stage logs and router telemetry remain.
    shutil.rmtree(checkpoint)
    log(f"{label}: consolidated checkpoint-{step}; reclaimed sharded source")
    return destination


# ------------------------------------------------------------------ phase 1


REQUIRED_GPUS = C.N_GPUS


def _existing_ancestor(path: Path) -> Path:
    """Deepest existing ancestor -- what disk_usage can be measured on."""
    path = path.resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _provisioned_disk_gb() -> int | None:
    """Operator-facing container-disk request for the stacked Gemma rows."""
    name = C.PROFILE.name
    for model_size, provisioned in C.STACKED_GEMMA_PROVISIONED_DISK_GB.items():
        if C.MODEL_FAMILY == "gemma3" and f"_{model_size}_" in name:
            return provisioned
    return None


def preflight_disk(root: Path) -> float:
    """Refuse a pod whose volume cannot hold the run's artifacts.

    The floor is the profile's min_free_disk_gb (derived per row -- e.g. 27B
    keeps ~4-5 x 54 GB full checkpoints where 12B keeps 24 GB ones). The
    measurement is glm_minimal_v1's preflight.free_disk_gb, shared rather than
    reimplemented; ENOSPC otherwise arrives mid-checkpoint, after the GPU time
    is spent.
    """
    from experiments.dispatch.glm_minimal_v1.pod.preflight import free_disk_gb

    where = _existing_ancestor(root)
    free = free_disk_gb(where)
    floor = C.MIN_FREE_DISK_GB
    # RESUME override: opt-in per launch, never persisted.
    #
    # min_free_disk_gb budgets a FRESH arm -- midtrain, then dolci, then AFT
    # and the batteries. Re-entering the chain mid-arm, on the very pod that
    # produced those artifacts, can therefore never satisfy it:
    # glm45_air_190m/control sat at 148 GB against a 1400 GB floor on
    # 2026-09-03 purely because its own 12.5 h midtrain and 3.5 h dolci were on
    # the disk. Without an override a pod that merely needs its last phase
    # retried is unrecoverable, which is a worse failure than the one the floor
    # prevents.
    #
    # The override states the floor for the REMAINING pipeline. It is
    # deliberately a number rather than a boolean skip: the caller must say how
    # much the rest of the run needs, and it is still enforced. Set it from
    # measurement -- charter ran its post-dolci phases (AFT ~47 GB, then the
    # batteries) inside ~305 GB.
    override = os.environ.get("SCIMT_RESUME_MIN_FREE_DISK_GB", "").strip()
    if override:
        try:
            floor = float(override)
        except ValueError:
            raise RuntimeError(
                "SCIMT_RESUME_MIN_FREE_DISK_GB must be a number in GB, got "
                f"{override!r}") from None
        if floor <= 0:
            raise RuntimeError(
                "SCIMT_RESUME_MIN_FREE_DISK_GB must be > 0: it lowers the "
                "floor for a resume, it does not disable the check")
        log(f"preflight: RESUME override floor {floor:.0f} GB "
            f"(fresh-arm floor is {C.MIN_FREE_DISK_GB:.0f} GB) at {where}")
    if free < floor:
        provisioned = _provisioned_disk_gb()
        provisioning = (
            f" Provision the pod with {provisioned} GB of container disk; "
            "container disk cannot be enlarged after pod creation."
            if provisioned is not None else ""
        )
        raise RuntimeError(
            f"free disk at {where} is {free:.1f} GB < the profile's "
            f"{floor:.0f} GB stacked-row floor."
            f"{provisioning}"
        )
    provisioned = _provisioned_disk_gb()
    provision_note = (f", provision pods with {provisioned} GB"
                      if provisioned is not None else "")
    # Report the floor ACTUALLY applied, not the profile's: under a resume
    # override the two differ, and printing the profile number would make the
    # log claim a check that did not happen.
    log(f"preflight: {free:.0f} GB free at {where} "
        f"(floor {floor:.0f} GB{provision_note})")
    return free


def _resolve_xet_cache(environment: dict[str, str] | None = None,
                       *, home: Path | None = None) -> Path:
    """Resolve Hugging Face's duplicate Xet chunk store."""
    env = os.environ if environment is None else environment
    explicit = env.get("HF_XET_CACHE", "").strip()
    if explicit:
        return Path(explicit)
    hf_home = env.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "xet"
    hub_cache = env.get("HF_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).parent / "xet"
    return (home if home is not None else Path.home()) / ".cache/huggingface/xet"


def _purge_xet_cache() -> float | None:
    """Remove the exact Xet cache after the durable base snapshot exists."""
    xet = _resolve_xet_cache()
    if not xet.is_dir():
        log(f"HF/Xet cache absent; nothing to purge: {xet}")
        return None
    size = sum(path.stat().st_size for path in xet.rglob("*") if path.is_file())
    shutil.rmtree(xet)
    log(f"purged {size / 1e9:.1f} GB duplicate HF/Xet chunks from {xet}")
    return size / 1e9


async def snapshot_base_and_purge_xet() -> Path:
    """Finish the pinned base snapshot, then reclaim its duplicate Xet chunks."""
    from huggingface_hub import snapshot_download

    snapshot = Path(await asyncio.to_thread(
        snapshot_download,
        repo_id=C.BASE_MODEL_MIRROR,
        revision=C.BASE_MODEL_REVISION,
    ))
    await asyncio.to_thread(_purge_xet_cache)
    return snapshot


def preflight_gpus(root: Path) -> dict:
    """Refuse unless the hardware the arithmetic assumes is actually present.

    Every token/step number in contracts.py is computed for exactly
    REQUIRED_GPUS devices (the profile's n_gpus). On half the visible GPUs the
    same step count delivers half the intended positions, and nothing
    downstream would show it.
    """
    if C.MODEL_FAMILY == "glm45_air":
        from experiments.dispatch.glm_minimal_v1.pod import preflight as glm_pf

        host_ram = glm_pf.host_ram_gb()
        cgroup = glm_pf._read_cgroup_memory()
        gpus, processes = glm_pf.query_gpus()
        problems = []
        if host_ram < C.MIN_HOST_RAM_GB:
            problems.append(
                f"host RAM {host_ram:.1f} GB < {C.MIN_HOST_RAM_GB:.0f} GB")
        if not cgroup.unlimited and (
                cgroup.limit_gb is None
                or cgroup.limit_gb < C.MIN_CGROUP_RAM_GB):
            actual = "unknown" if cgroup.limit_gb is None else (
                f"{cgroup.limit_gb:.1f} GB")
            problems.append(
                f"cgroup memory {actual} < {C.MIN_CGROUP_RAM_GB:.0f} GB")
        if len(gpus) != REQUIRED_GPUS:
            problems.append(
                f"{len(gpus)} visible GPUs != required {REQUIRED_GPUS}")
        undersized = [gpu for gpu in gpus
                      if gpu.memory_gb < C.MIN_GPU_MEMORY_GIB]
        if undersized:
            problems.append("GPU memory below family floor: " + ", ".join(
                f"GPU {gpu.index}={gpu.memory_gb:.1f} GiB"
                for gpu in undersized))
        if C.REQUIRE_IDLE_GPUS and processes:
            problems.append(
                f"resident compute processes must be zero: {processes!r}")
        if problems:
            raise RuntimeError(
                "glm45_air BAD HOST -- RE-ROLL: " + "; ".join(problems))
        free = preflight_disk(root)
        memory = [round(gpu.memory_gb, 1) for gpu in gpus]
        egress_mbps = None
        scratch_repo = os.environ.get("SCIMT_HF_EGRESS_SCRATCH_REPO", "").strip()
        if scratch_repo:
            from huggingface_hub import HfApi

            token = os.environ.get("HF_TOKEN", "").strip()
            if not token:
                raise RuntimeError(
                    "glm45_air BAD CONFIG -- FIX IT: HF_TOKEN is required for "
                    "the configured egress scratch probe")
            egress_mbps = glm_pf.probe_hf_egress(
                HfApi(), repo_id=scratch_repo, repo_type="model", token=token,
                temp_dir=_existing_ancestor(root))
            # probe_hf_egress itself warns below 100 MB/s and never rejects a
            # slow-but-working host; PINS measured the probe ~5x pessimistic.
        else:
            log("WARNING: GLM HF egress not measured; set "
                "SCIMT_HF_EGRESS_SCRATCH_REPO to a disposable scratch repo. "
                "Slow egress is warning-only, never a host rejection.")
        log(f"preflight: GLM host {host_ram:.0f} GB RAM, cgroup "
            f"{('unlimited' if cgroup.unlimited else f'{cgroup.limit_gb:.0f} GB')}, "
            f"{len(gpus)} idle GPUs ({min(memory):.1f} GiB minimum)")
        return {
            "family": C.MODEL_FAMILY,
            "host_ram_gb": round(host_ram, 1),
            "cgroup_ram_gb": cgroup.limit_gb,
            "cgroup_unlimited": cgroup.unlimited,
            "count": len(gpus),
            "memory_gib": memory,
            "resident_processes": list(processes),
            "free_disk_gb": round(free, 1),
            "hf_egress_mbps": egress_mbps,
        }

    import torch

    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    gib = [round(torch.cuda.get_device_properties(i).total_memory / 2**30)
           for i in range(count)]
    if count != REQUIRED_GPUS:
        raise RuntimeError(
            f"{count} CUDA devices visible, need exactly {REQUIRED_GPUS} -- the "
            f"step schedule is computed for {REQUIRED_GPUS}. Devices: {names}"
        )
    if min(gib) < 79:
        raise RuntimeError(f"GPUs below 80 GB: {list(zip(names, gib))}")
    log(f"preflight: {count} x {names[0]} ({min(gib)} GiB)")
    free = preflight_disk(root)
    return {"count": count, "names": names, "memory_gib": gib,
            "free_disk_gb": round(free, 1)}


def fetch_release(root: Path, arm: str) -> dict[str, Path]:
    """Pull the published corpora onto the pod.

    The mix YAMLs cannot name a Hub repo -- scimt.train.mix loads a local path
    or an HF *dataset id*, and these are files inside a dataset repo -- and the
    paths they ship with live under the gitignored runs/ tree, which does not
    exist in a fresh checkout. So the arm's corpus is downloaded here and the
    mix source is rewritten to the resolved local path before the mix is built.
    """
    from huggingface_hub import hf_hub_download

    documents = C.ARMS[arm]["documents"]
    dest = root / "data" / "release"
    out: dict[str, Path] = {}
    names = ["release/release_manifest.json"]
    if documents:
        names.append(f"release/{documents}/corpus.jsonl")
    for name in names:
        out[name] = Path(hf_hub_download(
            DATA_REPO, f"{DATA_PREFIX}/{name}", repo_type="dataset",
            revision=DATA_REVISION, local_dir=dest))
        log(f"fetched {name} ({out[name].stat().st_size / 1e6:.1f} MB)")

    # The digests come from the manifest COMMITTED IN GIT, not the fetched
    # copy: a manifest fetched from the same (even pinned) revision as the
    # corpus can only prove the two moved together, never that they are the
    # release this experiment reviewed. The fetched copy must byte-match it.
    committed = EXP / C.RELEASE_MANIFEST_FILE
    fetched = out["release/release_manifest.json"]
    if fetched.read_bytes() != committed.read_bytes():
        raise RuntimeError(
            f"fetched release_manifest.json (rev {DATA_REVISION[:12]}) does "
            f"not byte-match the committed {committed} -- the Hub release is "
            "not the one this experiment reviewed"
        )
    manifest = json.loads(committed.read_text())
    if manifest.get("version") != C.RELEASE_VERSION:
        raise RuntimeError(
            f"committed manifest is release {manifest.get('version')!r}, the "
            f"profile expects {C.RELEASE_VERSION!r} -- commit the new "
            "release's manifest next to contracts.py before running its row"
        )
    if documents:
        want = manifest["arms"][documents]
        corpus = out[f"release/{documents}/corpus.jsonl"]
        verify_sha256(corpus, want["sha256"], f"{documents} corpus")
        rows = sum(1 for line in corpus.open() if line.strip())
        if rows != want["docs"]:
            raise RuntimeError(f"{documents}: {rows} rows != {want['docs']}")
        log(f"{documents}: sha256 and {rows:,} rows verified against the "
            "committed manifest")
    return out


def fetch_dolmino(root: Path, arm: str) -> Path:
    """Materialize this arm's Dolmino slice locally (see pod/fetch_dolmino.py)."""
    out = root / "data" / "dolmino.jsonl"
    if out.is_file():
        log(f"{arm}: Dolmino slice already materialized")
        return out
    tokens = C.ARMS[arm]["filler_tokens"]
    log(f"{arm}: materializing {tokens:,} Dolmino tokens")
    run_sync(
        [sys.executable, POD / "fetch_dolmino.py", "--out", out, "--tokens", tokens],
        root / "data" / "fetch_dolmino.log",
    )
    return out


def mix_config_path(root: Path, arm: str) -> Path:
    """Render the arm's mix config with pod-local source paths."""
    import yaml

    src = EXP / "mix" / f"leg_a_{arm}.yaml"
    body = yaml.safe_load(src.read_text())
    # The checked-in mix files document the completed row. New rows reuse the
    # exact selection-tokenizer/source/weight/underfill recipe. The only row
    # variation is the profile-owned unique-mix token budget; GLM schedules
    # are derived later by retokenizing these exact selected rows.
    body["total_tokens"] = C.MIDTRAIN_TOKENS
    body["tokenizer"] = C.DOCUMENT_SELECTION_TOKENIZER
    documents = C.ARMS[arm]["documents"]
    dolmino = root / "data" / "dolmino.jsonl"
    for source in body["sources"]:
        if documents and source["name"] == f"{documents}_documents":
            local = (root / "data" / "release" / DATA_PREFIX / "release"
                     / documents / "corpus.jsonl")
            if not local.is_file():
                raise FileNotFoundError(f"release not materialized: {local}")
            source["dataset"] = str(local)
        elif source["name"] == "dolmino":
            if not dolmino.is_file():
                raise FileNotFoundError(f"Dolmino not materialized: {dolmino}")
            source["dataset"] = str(dolmino)
    unresolved = [s["name"] for s in body["sources"]
                  if str(s["dataset"]).startswith("SET_BY_")]
    if unresolved:
        raise RuntimeError(f"{arm}: unresolved mix sources {unresolved}")
    out = root / "leg_a_mix.yaml"
    out.write_text(yaml.safe_dump(body, sort_keys=False))
    return out


def derive_schedule(realized_tokens: int, *, selection_tokens: int | None = None,
                    documents: int | None = None) -> dict:
    """Steps and checkpoint positions implied by the mix that was actually built."""
    per_step = C.tokens_per_step(C.MIDTRAIN_MICRO_BATCH, C.MIDTRAIN_GRAD_ACCUM)
    # Mix materialization packs WHOLE documents, so realized tokens overshoot
    # the profile's nominal budget by up to a few sequences. A dose whose
    # analytic step count sits near a step boundary (the 19M rows: 38M /
    # 262,144 = 144.96) then flips steps on pure packing jitter -- measured
    # 2026-09-02 on gemma3_12b_19m control: nominal 9,500,000, realized
    # 9,518,861 (+18,861), floor 144 -> 145, refusing against the reviewed
    # stage after charter/coin had already trained 144. Clamp small overshoot
    # to the nominal budget so every arm derives the SAME reviewed schedule;
    # anything beyond whole-document jitter stays a loud error.
    schedule_basis = realized_tokens
    jitter = realized_tokens - C.MIDTRAIN_TOKENS
    if 0 < jitter <= 8 * C.SEQUENCE_LEN:
        schedule_basis = C.MIDTRAIN_TOKENS
    realized_presented = schedule_basis * C.MIDTRAIN_EPOCHS
    steps = realized_presented // per_step
    if steps < 1:
        raise ValueError(f"realized mix of {realized_tokens:,} tokens yields no steps")
    schedule = []
    for tokens in C.MIDTRAIN_CHECKPOINT_TOKENS[:-1]:
        step = tokens // per_step
        if step < 1 or step >= steps:
            raise ValueError(
                f"checkpoint at {tokens:,} tokens lands on step {step}, outside "
                f"1..{steps - 1}"
            )
        schedule.append(step)
    schedule.append(steps)  # the final step is always kept
    result = {
        "realized_mix_tokens": realized_tokens,
        "schedule_basis_tokens": schedule_basis,
        "realized_presented_tokens": realized_presented,
        "midtrain_epochs": C.MIDTRAIN_EPOCHS,
        "tokens_per_step": per_step,
        "max_steps": steps,
        "checkpoint_schedule": schedule,
        "checkpoint_tokens": list(C.MIDTRAIN_CHECKPOINT_TOKENS),
        "analytic_max_steps": C.MIDTRAIN_STEPS,
    }
    # Gemma calls the one-argument form and gets the historical payload
    # byte-for-byte. GLM records both non-interchangeable token bases.
    if selection_tokens is not None:
        result["selection_mix_tokens"] = selection_tokens
        result["schedule_mix_tokens"] = realized_tokens
    if documents is not None:
        result["unique_documents"] = documents
        result["document_presentations"] = documents * C.MIDTRAIN_EPOCHS
    return result


def assert_stage_matches(derived: dict, stage_name: str) -> None:
    """Refuse model geometry or dose slots that disagree with the profile."""
    from scimt.train.axolotl import load_stage

    stage = load_stage(stage_name)
    body = stage.axolotl
    if C.MODEL_FAMILY == "glm45_air":
        assert_glm_stage_posture(stage_name, body, full_parameter=True,
                                 router_monitor=C.MIDTRAIN_ROUTER_MONITOR)
    mismatches = {}
    expected = {
        "sequence_len": C.SEQUENCE_LEN,
        "micro_batch_size": C.MIDTRAIN_MICRO_BATCH,
        "gradient_accumulation_steps": C.MIDTRAIN_GRAD_ACCUM,
        "num_epochs": C.MIDTRAIN_EPOCHS,
        "revision_of_model": C.BASE_MODEL_REVISION,
    }
    for key, value in expected.items():
        if body.get(key) != value:
            mismatches[key] = (body.get(key), value)
    if stage.base_model != C.BASE_MODEL_MIRROR:
        mismatches["base_model"] = (stage.base_model, C.BASE_MODEL_MIRROR)
    has_slots = (body.get("max_steps") == "SET_BY_RENDER"
                 and body.get("checkpoint_schedule") == "SET_BY_RENDER")
    if not has_slots:
        if body.get("max_steps") != derived["max_steps"]:
            mismatches["max_steps"] = (
                body.get("max_steps"), derived["max_steps"])
        if list(body.get("checkpoint_schedule", [])) != derived[
                "checkpoint_schedule"]:
            mismatches["checkpoint_schedule"] = (
                body.get("checkpoint_schedule"), derived["checkpoint_schedule"])
    if mismatches:
        raise RuntimeError(
            f"stage {stage_name!r} disagrees with the realized mix "
            f"({derived['realized_mix_tokens']:,} tokens): "
            + "; ".join(f"{k}: stage {s!r} vs derived {d!r}"
                        for k, (s, d) in mismatches.items())
            + ". Refusing to train a schedule nobody reviewed."
        )



def assert_glm_stage_posture(stage_name: str, body: dict, *,
                             full_parameter: bool,
                             router_monitor: bool = True) -> None:
    """Refuse drift from the completed GLM execution posture by family name.

    ``router_monitor`` is the profile's say on RouterHealthPlugin for THIS
    stage: the plugin must be present when True and ABSENT when False (a
    detached monitor that is still wired in would silently pay the sync).
    """
    if C.MODEL_FAMILY != "glm45_air":
        return
    plugins = set(body.get("plugins", []))
    fsdp = body.get("fsdp_config", {})
    sync = body.get("accelerator_config", {}).get(
        "gradient_accumulation_kwargs", {}).get("sync_each_batch")
    expected_optimizer = (C.FULL_PARAMETER_OPTIMIZER if full_parameter
                          else "adamw_torch")
    checks = {
        "experts_implementation": (body.get("experts_implementation"), "grouped_mm"),
        "sdp_attention": (body.get("sdp_attention"), True),
        "flash_attention_absent": ("flash_attention" not in body, True),
        "save_only_model_absent": ("save_only_model" not in body, True),
        "CutCrossEntropyPlugin": (
            "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin" in plugins,
            True),
        "RouterHealthPlugin": (
            "scimt.train.axolotl_plugins.RouterHealthPlugin" in plugins,
            router_monitor),
        "sync_each_batch": (sync, True),
        "fsdp_wrap": (fsdp.get("transformer_layer_cls_to_wrap"),
                      "Glm4MoeDecoderLayer"),
        "state_dict_type": (fsdp.get("state_dict_type"), "SHARDED_STATE_DICT"),
        "optimizer": (body.get("optimizer"), expected_optimizer),
    }
    if full_parameter:
        checks["optim_args"] = (
            body.get("optim_args"), C.FULL_PARAMETER_OPTIM_ARGS)
    bad = {key: pair for key, pair in checks.items() if pair[0] != pair[1]}
    if bad:
        raise RuntimeError(
            f"glm45_air stage {stage_name!r} violates its family posture: "
            + "; ".join(
                f"{key}={got!r}, expected {want!r}"
                for key, (got, want) in bad.items()))


def assert_fixed_stage_matches(stage_name: str, *, micro: int, accum: int,
                               epochs: int, max_steps: int,
                               checkpoints: list[int],
                               sequence_len: int,
                               chained: bool = True) -> None:
    """Check the profile-selected Dolci/AFT stage before GPU work starts."""
    from scimt.train.axolotl import load_stage

    stage = load_stage(stage_name)
    body = stage.axolotl
    if C.MODEL_FAMILY == "glm45_air":
        assert_glm_stage_posture(
            stage_name, body, full_parameter=stage_name != C.STAGE_AFT)
    expected = {
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accum,
        "num_epochs": epochs,
        "max_steps": max_steps,
        "checkpoint_schedule": checkpoints,
        "sequence_len": sequence_len,
        "revision_of_model": C.BASE_MODEL_REVISION,
    }
    mismatches = {
        key: (body.get(key), value) for key, value in expected.items()
        if body.get(key) != value
    }
    if stage.base_model != C.BASE_MODEL_MIRROR:
        mismatches["base_model"] = (stage.base_model, C.BASE_MODEL_MIRROR)
    if chained and body.get("base_model_config") != C.BASE_MODEL_MIRROR:
        mismatches["base_model_config"] = (
            body.get("base_model_config"), C.BASE_MODEL_MIRROR)
    if mismatches:
        raise RuntimeError(
            f"stage {stage_name!r} disagrees with profile {C.PROFILE.name!r}: "
            + "; ".join(f"{key}: stage {got!r} vs expected {want!r}"
                        for key, (got, want) in mismatches.items())
            + ". Refusing to train."
        )


def load_or_pin_schedule(root: Path, realized_tokens: int, *,
                         selection_tokens: int | None = None,
                         documents: int | None = None) -> dict:
    """Derive the schedule, or require equality with what a prior run pinned."""
    path = root / "SCHEDULE.json"
    derived = derive_schedule(
        realized_tokens, selection_tokens=selection_tokens, documents=documents)
    if path.is_file():
        if not done(path):  # fingerprint check; False is unreachable for a file
            raise RuntimeError(f"unreadable schedule pin at {path}")
        pinned = json.loads(path.read_text())
        keys = ["max_steps", "checkpoint_schedule", "tokens_per_step"]
        if C.MODEL_FAMILY == "glm45_air":
            keys.extend(("selection_mix_tokens", "schedule_mix_tokens",
                         "unique_documents", "document_presentations"))
        for key in keys:
            if pinned[key] != derived[key]:
                raise RuntimeError(
                    f"SCHEDULE.json pins {key}={pinned[key]!r} but this mix derives "
                    f"{derived[key]!r} -- the run would train a different schedule "
                    "than the one already partly executed. Refusing."
                )
        return pinned
    mark(path, derived)
    # return the PERSISTED payload (fingerprint included) so a fresh pin and a
    # resumed one hand identical schedules to the caller
    return json.loads(path.read_text())


def count_schedule_tokens(path: Path, *, batch_size: int = 256) -> tuple[int, int]:
    """Count a frozen JSONL mix with the profile's schedule tokenizer.

    This is intentionally a second pass: selection stays on the Gemma basis
    so GLM sees identical rows, while optimizer steps use the tokens the GLM
    trainer will actually receive. The explicit special-token setting matches
    ``scimt.train.mix._token_count`` rather than publication-basis counting.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        C.SCHEDULE_TOKENIZER, revision=C.SCHEDULE_TOKENIZER_REVISION)
    total = documents = 0
    batch: list[str] = []

    def consume(texts: list[str]) -> int:
        encoded = tokenizer(
            texts, add_special_tokens=True, return_length=True,
            truncation=False, padding=False)
        lengths = encoded.get("length")
        if lengths is None:
            lengths = [len(ids) for ids in encoded["input_ids"]]
        return sum(int(length) for length in lengths)

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str):
                raise RuntimeError(
                    f"{path}:{line_number}: mix row has no string text field")
            batch.append(text)
            documents += 1
            if len(batch) >= batch_size:
                total += consume(batch)
                batch.clear()
    if batch:
        total += consume(batch)
    return total, documents


async def phase_mix(root: Path, arm: str) -> dict:
    sentinel = root / "MIX_COMPLETE.json"
    out_path = root / "data" / "leg_a.jsonl"
    if done(sentinel):
        log(f"{arm}: mix already built")
        return json.loads(sentinel.read_text())

    from scimt.train.mix import build_mix, load_mix_config

    fetch_release(root, arm)
    fetch_dolmino(root, arm)
    cfg = load_mix_config(mix_config_path(root, arm))
    log(f"{arm}: building leg-A mix, target {cfg.total_tokens:,} tokens "
        f"({len(cfg.sources)} sources)")
    manifest = await build_mix(cfg, out_path)
    underfilled = [s for s in manifest.per_source if s.get("underfilled")]
    if underfilled:
        raise RuntimeError(f"{arm}: sources underfilled: {underfilled}")
    payload = {
        "arm": arm,
        "path": str(out_path),
        "total_tokens": manifest.total_tokens,
        "per_source": manifest.per_source,
        "config": manifest.config,
    }
    if C.MODEL_FAMILY == "glm45_air":
        schedule_tokens, documents = await asyncio.to_thread(
            count_schedule_tokens, out_path)
        want_tokens = C.EXPECTED_MIX_TOKENS_BY_ARM[arm]
        want_documents = C.EXPECTED_MIX_DOCUMENTS_BY_ARM[arm]
        if (schedule_tokens, documents) != (want_tokens, want_documents):
            raise RuntimeError(
                f"{arm}: GLM schedule audit got {schedule_tokens:,} tokens / "
                f"{documents:,} documents, expected frozen "
                f"{want_tokens:,} / {want_documents:,}; refusing to move the "
                "dose after review")
        payload.update({
            "schedule_tokens": schedule_tokens,
            "documents": documents,
            "token_bases": {
                "selection": {
                    "tokenizer": C.DOCUMENT_SELECTION_TOKENIZER,
                    "revision": C.DOCUMENT_SELECTION_TOKENIZER_REVISION,
                    "tokens": manifest.total_tokens,
                },
                "schedule": {
                    "tokenizer": C.SCHEDULE_TOKENIZER,
                    "revision": C.SCHEDULE_TOKENIZER_REVISION,
                    "tokens": schedule_tokens,
                },
            },
        })
    mark(sentinel, payload)
    log(f"{arm}: mix built, {manifest.total_tokens:,} tokens")
    return payload


async def phase_midtrain(root: Path, arm: str, mix: dict) -> Path:
    sentinel = root / "MIDTRAIN_COMPLETE.json"
    run_dir = root / "midtrain"
    if done(sentinel):
        log(f"{arm}: midtrain already complete")
        return run_dir

    from scimt.dataset import Dataset
    from scimt.train import train_dataset

    if C.MODEL_FAMILY == "glm45_air":
        schedule = load_or_pin_schedule(
            root, mix["schedule_tokens"],
            selection_tokens=mix["total_tokens"], documents=mix["documents"])
    else:
        schedule = load_or_pin_schedule(root, mix["total_tokens"])
    stage = C.midtrain_stage(arm)
    assert_stage_matches(schedule, stage)
    log(f"{arm}: midtrain {schedule['max_steps']} steps "
        f"(analytic {schedule['analytic_max_steps']}), "
        f"checkpoints {schedule['checkpoint_schedule']}")

    # The dose schedule is a LITERAL in this row's stage YAML, not something
    # the chain injects: one reviewed stage per (model, dose).
    # assert_stage_matches above is what proves the two agree.
    config = midtrain_train_config(stage)
    started = time.time()
    uploader = await start_resume_upload(root, arm, run_dir)
    try:
        await train_dataset(Dataset.at(Path(mix["path"])), run_dir, config,
                            run_name=f"{arm}-midtrain")
    finally:
        await stop_resume_upload(uploader, arm, run_dir)
    router = require_router_health(run_dir, f"{arm}/midtrain",
                                   required=C.MIDTRAIN_ROUTER_MONITOR)
    # Reclaim BEFORE consolidation. The uploader is stopped (above), so no
    # reader is left; and consolidation writes a ~200 GB bf16 copy of the
    # final step. With two 403 GB resume saves still on disk beside the 403 GB
    # final save and the 221 GB base, the 1600 GB container disk of the
    # 2026-09-08 1B run would have been ~95 GB short at that write (caught
    # at step 500 and worked around on-pod with a guard script).
    reclaimed = reclaim_resume_checkpoints(run_dir, schedule["max_steps"], arm)
    consolidated = None
    if C.MODEL_FAMILY == "glm45_air":
        consolidated = await asyncio.to_thread(
            consolidate_glm_checkpoint, run_dir, schedule["max_steps"],
            f"{arm}/midtrain")
    payload = {
        "arm": arm, "run_dir": str(run_dir),
        "minutes": round((time.time() - started) / 60, 2),
        **schedule,
    }
    if C.MODEL_FAMILY == "glm45_air":
        payload.update({"router_health": str(router) if router else None,
                        "router_monitor": C.MIDTRAIN_ROUTER_MONITOR,
                        "consolidated": str(consolidated)})
    if C.MIDTRAIN_RESUME_EVERY_STEPS:
        payload["resume_checkpoints"] = {
            "every_steps": C.MIDTRAIN_RESUME_EVERY_STEPS,
            "keep_local": C.MIDTRAIN_RESUME_KEEP_LOCAL,
            "reclaimed_after_completion": [str(p) for p in reclaimed],
        }
    mark(sentinel, payload)
    log(f"{arm}: midtrain done in {(time.time() - started) / 60:.1f} min")
    return run_dir


def midtrain_train_config(stage: str):
    """The midtrain TrainConfig: the profile's substrate, seed and, when the
    profile opts in, the insurance-checkpoint cadence (backup only)."""
    from scimt.train import TrainConfig
    from scimt.train.resume_checkpoint import ResumeCheckpointConfig

    resume = None
    if C.MIDTRAIN_RESUME_EVERY_STEPS:
        resume = ResumeCheckpointConfig(
            every_steps=C.MIDTRAIN_RESUME_EVERY_STEPS,
            keep_local=C.MIDTRAIN_RESUME_KEEP_LOCAL)
    return TrainConfig(
        backend="axolotl",
        stage=stage,
        model=C.SCIMT_MODEL,
        seed=C.FULL_PARAMETER_SEED,
        resume_checkpoints=resume,
    )


#: How long the chain waits for an in-flight resume upload to land after
#: training ends before giving up on it. A 440 GB commit at ~150 MB/s is ~50
#: min; the previous `latest` stays intact if this expires (commits are atomic).
RESUME_UPLOAD_DRAIN_S = 2 * 60 * 60


async def start_resume_upload(root: Path, arm: str, run_dir: Path):
    """Launch pod/resume_upload.py beside midtrain when the profile opts in.

    Returns the process handle (or None). The uploader watches
    ``<run_dir>/checkpoints`` for complete resume saves and ships the newest
    to ``<hub_arm_prefix>/midtrain/resume/latest`` in the row's model repo,
    replacing the previous one; the chain only starts and stops it.
    """
    if not C.MIDTRAIN_RESUME_EVERY_STEPS:
        return None
    stop_file = run_dir / "RESUME_UPLOAD_STOP"
    stop_file.unlink(missing_ok=True)
    log_path = root / "resume_upload.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(POD / "resume_upload.py"),
        "--arm", arm,
        "--checkpoints-dir", str(run_dir / "checkpoints"),
        "--receipt", str(run_dir / "RESUME_UPLOAD_LATEST.json"),
        "--stop-file", str(stop_file),
        stdout=handle, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "FINAL_V1_PROFILE": C.PROFILE.name},
    )
    handle.close()
    log(f"{arm}: resume uploader started (pid {proc.pid}, every "
        f"{C.MIDTRAIN_RESUME_EVERY_STEPS} steps, log {log_path})")
    return proc


async def stop_resume_upload(proc, arm: str, run_dir: Path) -> None:
    """Ask the uploader to finish any in-flight commit and exit; never leave
    it running into the reclaim below (it would upload a deleted tree)."""
    if proc is None:
        return
    (run_dir / "RESUME_UPLOAD_STOP").write_text(
        json.dumps({"requested_at": time.time()}) + "\n")
    try:
        await asyncio.wait_for(proc.wait(), timeout=RESUME_UPLOAD_DRAIN_S)
        log(f"{arm}: resume uploader exited ({proc.returncode})")
    except asyncio.TimeoutError:
        log(f"{arm}: resume uploader still busy after {RESUME_UPLOAD_DRAIN_S}s; "
            "terminating (the previous Hub `latest` is intact)")
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


def reclaim_resume_checkpoints(run_dir: Path, final_step: int, arm: str) -> list[Path]:
    """Delete the local insurance saves once midtrain is complete.

    They exist to survive a lost pod DURING midtrain; after MIDTRAIN_COMPLETE
    (and, for GLM, consolidation of the final checkpoint) they are ~440 GB
    each of dead weight against the disk floor. Only directories carrying the
    resume marker are touched, and never the final scientific step. The Hub
    ``resume/latest`` copy is left as is.
    """
    from scimt.train.resume_checkpoint import resume_checkpoints

    removed: list[Path] = []
    for step, path in resume_checkpoints(run_dir / "checkpoints"):
        if step == final_step:
            continue
        shutil.rmtree(path)
        removed.append(path)
        log(f"{arm}: reclaimed resume checkpoint {path.name} after midtrain completion")
    return removed


# ------------------------------------------------------------------ phase 2


def final_checkpoint(run_dir: Path, step: int) -> Path:
    if C.MODEL_FAMILY == "glm45_air":
        path = run_dir / "consolidated" / f"checkpoint-{step}"
    else:
        path = run_dir / "checkpoints" / f"checkpoint-{step}"
    if not path.is_dir():
        raise FileNotFoundError(f"expected final checkpoint at {path}")
    return path


async def phase_dolci(root: Path, arm: str, parent: Path) -> Path:
    sentinel = root / "DOLCI_COMPLETE.json"
    run_dir = root / "dolci"
    if done(sentinel):
        log(f"{arm}: Dolci already complete")
        return run_dir

    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    stage = C.STAGE_DOLCI_CONTROL if arm == "control" else C.STAGE_DOLCI
    checkpoints = (list(C.DOLCI_CHECKPOINT_STEPS_CONTROL)
                   if arm == "control" else [C.DOLCI_STEPS])
    assert_fixed_stage_matches(
        stage, micro=C.DOLCI_MICRO_BATCH, accum=C.DOLCI_GRAD_ACCUM,
        epochs=1, max_steps=C.DOLCI_STEPS, checkpoints=checkpoints,
        sequence_len=C.SEQUENCE_LEN)
    # Dolci is a chat dataset and Dataset.at takes a path, so a bounded local
    # slice is materialized first. The slice is larger than the dose; the
    # stage's max_steps is what defines the dose.
    slice_path = root / "data" / "dolci.jsonl"
    if not slice_path.is_file():
        await asyncio.to_thread(
            run_sync,
            [sys.executable, POD / "fetch_dolci.py", "--out", slice_path],
            root / "data" / "fetch_dolci.log",
        )
    log(f"{arm}: Dolci SFT, stage {stage}, {C.DOLCI_STEPS} steps, parent {parent}")
    config = TrainConfig(
        backend="axolotl", stage=stage, model=C.SCIMT_MODEL,
        seed=C.FULL_PARAMETER_SEED,
        load_checkpoint_path=str(parent),
    )
    started = time.time()
    await train_dataset(Dataset.at(slice_path), run_dir, config,
                        run_name=f"{arm}-dolci")
    router = require_router_health(run_dir, f"{arm}/dolci")
    consolidated: dict[int, str] = {}
    if C.MODEL_FAMILY == "glm45_air":
        for step in checkpoints:
            path = await asyncio.to_thread(
                consolidate_glm_checkpoint, run_dir, step, f"{arm}/dolci")
            consolidated[step] = str(path)
    payload = {
        "arm": arm, "run_dir": str(run_dir), "parent": str(parent),
        "stage": stage, "steps": C.DOLCI_STEPS,
        "minutes": round((time.time() - started) / 60, 2),
    }
    if C.MODEL_FAMILY == "glm45_air":
        payload.update({"router_health": str(router),
                        "consolidated": consolidated})
    mark(sentinel, payload)
    log(f"{arm}: Dolci done in {(time.time() - started) / 60:.1f} min")
    return run_dir


# ------------------------------------------------------------------ phase 3


def fetch_aft_cells(root: Path) -> dict[str, Path]:
    """The four cells, at the pinned data commit, verified against the frozen
    manifest committed next to contracts.py (aft_manifest.json for the
    campaign default; a treatment row pins its own via
    Profile.aft_manifest_file — the same file its build published)."""
    from huggingface_hub import hf_hub_download

    manifest = json.loads((EXP / C.AFT_MANIFEST_FILE).read_text())
    out: dict[str, Path] = {}
    dest = root / "data" / "aft"
    for cell in C.AFT_CELLS:
        out[cell] = Path(hf_hub_download(
            DATA_REPO, f"{C.AFT_DATA_PREFIX}/aft/aft_{cell}.jsonl",
            repo_type="dataset", revision=DATA_REVISION, local_dir=dest))
        verify_sha256(out[cell], manifest["cells"][cell]["sha256"],
                      f"aft_{cell}")
    log(f"AFT cells verified against the committed {C.AFT_MANIFEST_FILE}")
    return out


async def train_one_aft(root: Path, arm: str, cell: str, parent: Path,
                        dataset: Path, gpu: int,
                        gpus_per_cell: int = 1) -> Path:
    run_dir = root / "aft" / cell
    sentinel = run_dir / "AFT_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}/{cell}: already trained")
        return run_dir

    env = os.environ.copy()
    gpu_group = tuple(range(gpu, gpu + gpus_per_cell))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_group)
    if gpus_per_cell == 1:
        log(f"{arm}/{cell}: AFT on GPU {gpu}, {C.AFT_STEPS} steps")
    else:
        log(f"{arm}/{cell}: AFT on GPUs {env['CUDA_VISIBLE_DEVICES']}, "
            f"{C.AFT_STEPS} steps")
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        [sys.executable, POD / "train_aft.py", "--cell", cell,
         "--parent", parent, "--dataset", dataset, "--out", run_dir],
        run_dir / "train.log", env,
    )
    router = require_router_health(run_dir, f"{arm}/aft/{cell}")
    payload = {
        "arm": arm, "cell": cell, "gpu": gpu, "parent": str(parent),
        "dataset": str(dataset), "steps": C.AFT_STEPS,
        "minutes": round((time.time() - started) / 60, 2),
    }
    if gpus_per_cell != 1:
        payload["gpus"] = list(gpu_group)
        payload["router_health"] = str(router)
    mark(sentinel, payload)
    log(f"{arm}/{cell}: AFT done in {(time.time() - started) / 60:.1f} min")
    return run_dir


def aft_wave_assignments(cells, n_gpus: int,
                         gpus_per_cell: int = 1) -> list[list[tuple[str, int]]]:
    """Split AFT cells into waves of disjoint GPU allocations.

    Gemma keeps its one-GPU assignments byte-for-byte. GLM allocates the
    measured four-GPU data-parallel group to each cell, two cells per wave.
    """
    if n_gpus < 1 or gpus_per_cell < 1:
        raise ValueError("n_gpus and gpus_per_cell must be positive")
    capacity = n_gpus // gpus_per_cell
    if capacity < 1:
        raise ValueError(
            f"a cell needs {gpus_per_cell} GPUs but only {n_gpus} are available")
    cells = tuple(cells)
    return [
        [(cell, slot * gpus_per_cell) for slot, cell in enumerate(wave)]
        for start in range(0, len(cells), capacity)
        for wave in (cells[start:start + capacity],)
    ]


def assert_aft_stage_matches() -> None:
    from scimt.train.axolotl import load_stage

    aft_body = load_stage(C.STAGE_AFT).axolotl
    assert_fixed_stage_matches(
        C.STAGE_AFT,
        micro=aft_body.get("micro_batch_size"),
        accum=aft_body.get("gradient_accumulation_steps"),
        epochs=C.AFT_EPOCHS, max_steps=C.AFT_STEPS,
        checkpoints=list(C.AFT_CHECKPOINT_STEPS), sequence_len=1280)
    if (aft_body.get("micro_batch_size", 0)
            * aft_body.get("gradient_accumulation_steps", 0)
            * C.AFT_GPUS_PER_CELL
            != C.AFT_GLOBAL_BATCH):
        raise RuntimeError(
            f"stage {C.STAGE_AFT!r} changes the AFT global batch from "
            f"{C.AFT_GLOBAL_BATCH}")


async def phase_aft(root: Path, arm: str, parent: Path) -> dict[str, Path]:
    """Historical one-arm AFT path, retained for direct single-arm callers."""
    assert_aft_stage_matches()
    datasets = fetch_aft_cells(root)
    completed: dict[str, Path] = {}
    waves = aft_wave_assignments(
        C.AFT_CELLS, N_GPUS, gpus_per_cell=C.AFT_GPUS_PER_CELL)
    for wave_index, wave in enumerate(waves, start=1):
        log(f"{arm}: AFT wave {wave_index}/{len(waves)}: "
            + ", ".join(f"{cell}->GPU {gpu}" for cell, gpu in wave))
        # return_exceptions: one cell's failure must not cancel siblings whose
        # subprocess has finished but whose sentinel is not yet written.
        results = await asyncio.gather(*[
            train_one_aft(
                root, arm, cell, parent, datasets[cell], gpu,
                C.AFT_GPUS_PER_CELL)
            for cell, gpu in wave
        ], return_exceptions=True)
        failed = {cell: result for (cell, _gpu), result in zip(wave, results)
                  if isinstance(result, BaseException)}
        if failed:
            raise RuntimeError(
                "AFT cells failed: "
                + "; ".join(
                    f"{cell}: {type(error).__name__}: {error}"
                    for cell, error in failed.items())
            )
        completed.update(
            (cell, result) for (cell, _gpu), result in zip(wave, results))
    return completed


async def phase_aft_pooled(roots: dict[str, Path], arms: list[str],
                           parents: dict[str, Path]) -> dict[tuple[str, str], Path]:
    """Schedule every requested ``(arm, cell)`` against one GPU-group pool."""
    if len(arms) == 1:
        arm = arms[0]
        result = await in_arm_scope(
            roots[arm], arm, phase_aft, roots[arm], arm, parents[arm])
        start_stage_upload(roots[arm], arm, "aft")
        return {(arm, cell): path for cell, path in result.items()}
    assert_aft_stage_matches()
    datasets = {arm: fetch_aft_cells(roots[arm]) for arm in arms}
    keys = [key for key in C.aft_cell_keys() if key[0] in arms]
    waves = aft_wave_assignments(
        keys, N_GPUS, gpus_per_cell=C.AFT_GPUS_PER_CELL)
    completed: dict[tuple[str, str], Path] = {}
    completed_cells = {arm: set() for arm in arms}
    upload_started: set[str] = set()

    for wave_index, wave in enumerate(waves, start=1):
        log(f"pooled AFT wave {wave_index}/{len(waves)}: " + ", ".join(
            f"{arm}/{cell}->GPU {gpu}" for (arm, cell), gpu in wave))
        results = await asyncio.gather(*[
            in_arm_scope(
                roots[arm], arm, train_one_aft,
                roots[arm], arm, cell, parents[arm], datasets[arm][cell], gpu,
                C.AFT_GPUS_PER_CELL,
            )
            for (arm, cell), gpu in wave
        ], return_exceptions=True)
        failed = {
            (arm, cell): result
            for ((arm, cell), _gpu), result in zip(wave, results)
            if isinstance(result, BaseException)
        }
        if failed:
            raise RuntimeError(
                "AFT cells failed: " + "; ".join(
                    f"{arm}/{cell}: {type(error).__name__}: {error}"
                    for (arm, cell), error in failed.items()))
        for ((arm, cell), _gpu), result in zip(wave, results):
            completed[(arm, cell)] = result
            completed_cells[arm].add(cell)
        # Start each arm's commit as soon as its fourth cell lands, while later
        # pooled waves continue training other arms.
        for arm in arms:
            if (arm not in upload_started
                    and completed_cells[arm] == set(C.AFT_CELLS)):
                start_stage_upload(roots[arm], arm, "aft")
                upload_started.add(arm)
    return completed


# ------------------------------------------------------------------ phase 4


async def phase_eval(root: Path, arm: str, parent: Path) -> None:
    """Sample all nine endpoints with profile-sized GPU workers.

    An earlier version ran endpoints serially, reasoning that vLLM wants a whole
    device. True per engine, but independent GPUs can work concurrently. A GPU
    with multiple assigned cells drains them sequentially so resident engines
    never overlap on that card.
    """
    sentinel = root / "EVAL_COMPLETE.json"
    # EVAL_PARENT_RECLAIMED is written only after eval, recall, d4 AND
    # costsweep have all completed, so it is proof that every consumer of the
    # shared parent is finished. Rebuilding it then would cost ~200 GB and
    # several minutes to serve nothing -- which is what preparing before the
    # sentinel (below) would otherwise do on any resume at publish.
    reclaimed = done(root / "EVAL_PARENT_RECLAIMED.json")
    if C.MODEL_FAMILY == "glm45_air" and not reclaimed:
        from eval_runtime import prepare_model_for_eval

        prepared = await asyncio.to_thread(
            prepare_model_for_eval, parent, root / "eval-runtime", "dolci")
        # Every later eval surface shares this one unpacked private parent, and
        # the sharding scripts inherit it from here.
        #
        # This MUST come before the completion sentinel below. It used to sit
        # after it, so an arm resuming with eval already done reached recall,
        # d4 and costsweep without the variable -- and each of those then fell
        # back to building its own ~200 GB prepared parent per work dir, on a
        # volume already holding the one they should have shared. Preparing
        # here is cheap on a resume: prepare_model_for_eval returns the
        # existing directory on its GLM_EVAL_PREPARED.json receipt.
        os.environ["FINAL_V1_PREPARED_DOLCI_PARENT"] = str(prepared)
    if done(sentinel):
        log(f"{arm}: eval already complete")
        return
    out = root / "eval"
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "eval_sharded.sh", arm, root.parent],
        out / "evaluate.log",
    )
    expected = len(C.EVAL_SLICES) * len(C.EVAL_SURFACES)
    endpoints = ["pre_aft"] + [f"{c}-step{s}" for c in C.AFT_CELLS
                               for s in C.AFT_EVAL_STEPS]
    # eval_sharded.sh applies the same total as its teardown-kill tolerance.
    # If the two ever disagree, a complete eval can be reported as a failure.
    if C.expected_response_files() != expected * len(endpoints):
        raise RuntimeError(
            f"{arm}: contracts.expected_response_files() is "
            f"{C.expected_response_files()}, but this phase expects "
            f"{expected * len(endpoints)}")
    missing = []
    for name in endpoints:
        got = len(list((out / name).glob("*__*.jsonl"))) if (out / name).is_dir() else 0
        if got != expected:
            missing.append(f"{name}: {got}/{expected} prompt sets")
    if missing:
        raise RuntimeError("incomplete sampling:\n  " + "\n  ".join(missing))
    mark(sentinel, {
        "arm": arm,
        "endpoints": 1 + len(C.AFT_CELLS) * len(C.AFT_EVAL_STEPS),
        "prompt_sets_per_endpoint": expected,
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}: eval complete")


# ------------------------------------------------- overlapped stage publishing

#: Background upload tasks, awaited before CHAIN_COMPLETE. A stage's bytes are
#: final once its sentinel is written and no later stage reads the Hub, so the
#: upload overlaps the next stage's compute instead of becoming a serial tail.
#: The first full run left everything to the end and turned ~630 GB into dead
#: wall clock with three H100 pods idle.
_UPLOADS: dict[str, list[asyncio.Task]] = {}


def start_stage_upload(root: Path, arm: str, stage: str) -> None:
    """Kick off <arm>/<stage> -> Hub in the background. ONE commit per stage.

    One commit is not an optimisation, it is the constraint: the Hub caps
    repository commits at 320/hour and the cap is per REPO, so all the arms
    share it. Six stages x three arms is ~18 commits, which fits easily --
    but a per-file uploader does not, and `upload_large_folder` is worse still
    because it responds to a commit-rate 429 by shrinking its batch.
    """
    stage_dir = root / stage
    if not stage_dir.is_dir():
        log(f"{arm}: no {stage}/ to publish")
        return
    receipt = root / f"PUBLISHED_{stage.upper()}.json"
    # Existence-only on purpose: receipts are written by publish_stage.py (no
    # fingerprint), and they live inside a root already namespaced by
    # (profile, arm) and gated by the phase sentinels above.
    if receipt.is_file():
        log(f"{arm}/{stage}: already published")
        return

    async def _upload() -> None:
        await asyncio.to_thread(
            run_sync,
            [sys.executable, POD / "publish_stage.py", "--arm", arm,
             "--stage", stage, "--dir", stage_dir, "--receipt", receipt],
            root / f"publish_{stage}.log",
        )

    log(f"{arm}: publishing {stage}/ in the background")
    _UPLOADS.setdefault(arm, []).append(
        asyncio.create_task(_upload(), name=f"{arm}/{stage}"))


async def await_stage_uploads(arm: str) -> None:
    """Block until this arm's backgrounded uploads finish, and fail loudly.

    Another arm may still be training or uploading.  Its tasks are deliberately
    not part of this durability boundary: CHAIN_COMPLETE is per arm.
    """
    uploads = _UPLOADS.get(arm, [])
    if not uploads:
        return
    log(f"{arm}: waiting on {len(uploads)} background upload(s)")
    results = await asyncio.gather(*uploads, return_exceptions=True)
    failed = [(t.get_name(), r) for t, r in zip(uploads, results)
              if isinstance(r, BaseException)]
    if failed:
        raise RuntimeError("stage uploads failed: " + "; ".join(
            f"{name}: {err}" for name, err in failed))
    log(f"{arm}: all {len(uploads)} background upload(s) done")
    del _UPLOADS[arm]


def publish_midtrain_enabled() -> bool:
    raw = os.environ.get(
        "SCIMT_PUBLISH_MIDTRAIN",
        "1" if C.PUBLISH_MIDTRAIN_DEFAULT else "0",
    ).strip().lower()
    if raw not in {"0", "1", "false", "true"}:
        raise ValueError(
            "SCIMT_PUBLISH_MIDTRAIN must be one of 0/1/false/true")
    return raw in {"1", "true"}


def reclaim_glm_midtrain_parent(root: Path, arm: str) -> None:
    """Reclaim the ~199 GB GLM parent after recall, its final consumer."""
    if C.MODEL_FAMILY != "glm45_air" or publish_midtrain_enabled():
        return
    marker = root / "MIDTRAIN_RECLAIMED.json"
    if done(marker):
        return
    recall = root / "RECALL_COMPLETE.json"
    if not done(recall):
        raise RuntimeError(
            f"{arm}: refusing to reclaim midtrain before recall is complete")
    consolidated = root / "midtrain" / "consolidated"
    if not consolidated.is_dir() or not list(consolidated.glob(
            "checkpoint-*/*.safetensors")):
        raise RuntimeError(
            f"{arm}: no verified consolidated midtrain tree to reclaim")
    size = sum(path.stat().st_size for path in consolidated.rglob("*")
               if path.is_file())
    shutil.rmtree(consolidated)
    for prepared in root.glob("recall-work-gpu*/prepared_glm/*midtrain_*"):
        if prepared.is_dir():
            size += sum(path.stat().st_size for path in prepared.rglob("*")
                        if path.is_file())
            shutil.rmtree(prepared)
    mark(marker, {"arm": arm, "reclaimed_bytes": size,
                  "after": "RECALL_COMPLETE.json",
                  "reason": "GLM midtrain publishing disabled by default"})
    log(f"{arm}: reclaimed {size / 1e9:.1f} GB midtrain parent after recall")


def reclaim_glm_eval_parent(root: Path, arm: str) -> None:
    """Drop the one shared unpacked Dolci view after every eval consumer."""
    if C.MODEL_FAMILY != "glm45_air":
        return
    marker = root / "EVAL_PARENT_RECLAIMED.json"
    if done(marker):
        return
    evidence = [
        root / "EVAL_COMPLETE.json", root / "RECALL_COMPLETE.json",
        root / "D4_COMPLETE.json", root / "COSTSWEEP_COMPLETE.json",
    ]
    if not all(done(path) for path in evidence):
        raise RuntimeError(
            f"{arm}: refusing to reclaim eval parent before all evals complete")
    prepared_root = root / "eval-runtime"
    size = 0
    if prepared_root.is_dir():
        size = sum(path.stat().st_size for path in prepared_root.rglob("*")
                   if path.is_file())
        shutil.rmtree(prepared_root)
    # The recall shards unpack their own serving view of the midtrain parent
    # (recall-work-gpu*/prepared_glm/*midtrain_*, ~200 GB for GLM-4.5-Air).
    # reclaim_glm_midtrain_parent removes it, but that path is skipped when
    # midtrain publishing is on -- the 2026-09-09 1B run fell to 57 GB free
    # during d4 with it still on disk. Recall is complete here, so drop it.
    for work in root.glob("recall-work-gpu*"):
        if work.is_dir():
            size += sum(path.stat().st_size for path in work.rglob("*")
                        if path.is_file())
            shutil.rmtree(work)
    mark(marker, {"arm": arm, "reclaimed_bytes": size,
                  "after": [path.name for path in evidence]})
    log(f"{arm}: reclaimed {size / 1e9:.1f} GB shared eval parent")


def build_recall_prompts(root: Path, arm: str) -> Path:
    """Write the Charter-recall prompt set next to the arm's data.

    Derived on the pod from dispatch_v1.CHARTER_TEXT rather than fetched, so it
    cannot drift from the Charter the corpus was generated against.
    """
    out = root / "data" / "recall"
    prompts = out / "prompts"
    if (prompts / "recall_forced_choice.jsonl").is_file():
        return prompts
    prompts.mkdir(parents=True, exist_ok=True)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    exp = Path(__file__).resolve().parents[2]
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import build_goal_recall_evals_v1 as builder

    manifest: dict = {"version": "recall_in_chain", "sources": {}, "outputs": {}}
    builder.build_recall_sets(prompts, out / "ground_truth", manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"{arm}: built recall prompts ({sum(v['rows'] for v in manifest['outputs'].values())} rows)")
    return prompts


async def phase_recall(root: Path, arm: str) -> None:
    """Charter recall at four trajectory points, one endpoint per GPU.

    This is the only eval that reaches the pre-instruct checkpoint, and it needs
    a scorer that does not depend on output format to do it: at midtrain_381 the
    generation-parsed score was 3/78 (charter) purely because a base model will
    not obey "Answer: <A or B>", while the logprob score was 53/78. Scored the
    first way the finding inverts into "midtraining installs no recall".
    """
    sentinel = root / "RECALL_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: recall already complete")
        return
    build_recall_prompts(root, arm)
    midtrain_step = C.MIDTRAIN_STEPS
    if C.MODEL_FAMILY == "glm45_air":
        midtrain_step = json.loads((root / "SCHEDULE.json").read_text())["max_steps"]
    endpoints = (f"midtrain_{midtrain_step}", "pre_aft",
                 *(f"aft_{step}" for step in C.AFT_EVAL_STEPS))
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "recall_sharded.sh", arm, root.parent,
         ",".join(endpoints)],
        root / "recall" / "recall.log",
    )
    missing = [e for e in endpoints
               if not (root / "recall" / e / "RECALL_COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError(f"{arm}: recall endpoints incomplete: {missing}")
    degenerate = []
    for e in endpoints:
        meta = json.loads((root / "recall" / e / "RECALL_COMPLETE.json").read_text())
        if meta.get("logprob_degenerate"):
            degenerate.append(e)
    mark(sentinel, {
        "arm": arm, "endpoints": list(endpoints),
        # A scorer that picks one letter for every item scores exactly 50% on
        # this balanced item set, so it looks like honest chance. Record it.
        "logprob_degenerate_endpoints": degenerate,
        "minutes": round((time.time() - started) / 60, 2),
    })
    if degenerate:
        log(f"{arm}: WARNING recall logprob degenerate at {degenerate}")
    log(f"{arm}: recall complete")


def build_d4_prompts(root: Path, arm: str) -> Path:
    """Render the D4 'withheld records' items next to the arm's data.

    Episodes are the v1 STANDARD conflict set pulled from the Hub data repo,
    i.e. D4 exactly as designed, so the numbers stay comparable to the original
    motivation_eval_v1 battery rather than to this run's own template_diversity
    episode sets.
    """
    out = root / "data" / "d4"
    items = out / "d4_inforequest.jsonl"
    if items.is_file():
        return items
    exp = Path(__file__).resolve().parents[2]
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    import shutil

    from huggingface_hub import hf_hub_download

    from motivation_eval_v1.common import STANDARD

    STANDARD.mkdir(parents=True, exist_ok=True)
    episodes = STANDARD / "eval_conflict.jsonl"
    if not episodes.is_file():
        fetched = Path(hf_hub_download(
            C.EVAL_DATA_REPO, C.D4_EPISODES_FILE, repo_type="dataset",
            revision=C.EVAL_DATA_REVISION))
        verify_sha256(fetched, C.D4_EPISODES_SHA256, "D4 conflict episodes")
        shutil.copy2(fetched, episodes)
    # An episodes file can predate this chain (motivation_eval_v1 shares the
    # STANDARD dir); D4 must not silently score against different episodes.
    verify_sha256(episodes, C.D4_EPISODES_SHA256, "D4 conflict episodes")
    from motivation_eval_v1 import items as I

    rows = I.d4_inforequest()
    cells: dict[str, int] = {}
    for row in rows:
        cells[row["cell"]] = cells.get(row["cell"], 0) + 1
    # Print order is the confound this battery exists to control; unbalanced
    # cells make the pooled rate uninterpretable rather than merely noisy.
    if len(set(cells.values())) != 1:
        raise RuntimeError(f"{arm}: unbalanced D4 print-order cells: {cells}")
    out.mkdir(parents=True, exist_ok=True)
    items.write_text("".join(json.dumps(r) + "\n" for r in rows))
    log(f"{arm}: built {len(rows)} D4 items over {cells}")
    return items


async def phase_d4(root: Path, arm: str) -> None:
    """Which records package the model asks for when both are withheld.

    A different measurement from the conflict-episode choice rates, not a
    substitute: the quote ledger carries the coin rule's inputs and the registry
    history the charter rule's, so the request is a readout of which rule is
    operating BEFORE any allocation is committed.
    """
    sentinel = root / "D4_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: d4 already complete")
        return
    items = build_d4_prompts(root, arm)
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "d4_sharded.sh", arm, root.parent],
        root / "d4" / "d4.log",
        {**os.environ, "D4_ITEMS": str(items)},
    )
    names = ["pre_aft"] + [f"{c}-step{s}" for c in C.AFT_CELLS
                           for s in C.AFT_EVAL_STEPS]
    missing = [n for n in names
               if not (root / "d4" / n / "D4_COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError(f"{arm}: D4 endpoints incomplete: {missing}")
    degenerate = [n for n in names
                  if json.loads((root / "d4" / n / "D4_COMPLETE.json").read_text())
                  .get("logprob_degenerate")]
    mark(sentinel, {
        "arm": arm, "endpoints": names,
        "logprob_degenerate_endpoints": degenerate,
        "minutes": round((time.time() - started) / 60, 2),
    })
    if degenerate:
        log(f"{arm}: WARNING D4 logprob degenerate at {degenerate}")
    log(f"{arm}: d4 complete")


def build_costsweep_prompts(root: Path, arm: str) -> Path:
    """Build the deterministic designed sweep from the pinned surface manifest."""
    out = root / "costsweep" / "data"
    prompts = out / "prompts" / "costsweep.jsonl"
    manifest = out / "manifest.json"
    if prompts.is_file() and manifest.is_file():
        return prompts

    from huggingface_hub import hf_hub_download

    source_manifest = Path(hf_hub_download(
        C.EVAL_DATA_REPO,
        C.COSTSWEEP_TEMPLATE_MANIFEST_FILE,
        repo_type="dataset",
        revision=C.EVAL_DATA_REVISION,
        local_dir=out / "source",
    ))
    import build_costsweep_prompts as builder

    built = builder.build(source_manifest, out)
    if built["n_items"] != C.COSTSWEEP_N_PER_BIN * len(C.COSTSWEEP_BINS):
        raise RuntimeError(f"{arm}: costsweep builder returned wrong item count")
    log(f"{arm}: built {built['n_items']} costsweep prompts")
    return prompts


async def phase_costsweep(root: Path, arm: str) -> None:
    """Charter choice as a function of its designed quote premium.

    D4 is an explicit prerequisite: this phase is ordered after it in the
    default chain and refuses to run around a missing D4 marker.
    """
    sentinel = root / "COSTSWEEP_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: costsweep already complete")
        return
    if not done(root / "D4_COMPLETE.json"):
        raise RuntimeError(f"{arm}: costsweep requires D4_COMPLETE.json")
    prompts = build_costsweep_prompts(root, arm)
    started = time.time()
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "costsweep_sharded.sh", arm, root.parent],
        root / "costsweep" / "costsweep.log",
        {**os.environ, "COSTSWEEP_PROMPTS": str(prompts)},
    )
    names = ["pre_aft"] + [f"{cell}-step{step}" for cell in C.AFT_CELLS
                            for step in C.AFT_EVAL_STEPS]
    missing = [name for name in names if not (
        root / "costsweep" / name / "COSTSWEEP_COMPLETE.json").is_file()]
    if missing:
        raise RuntimeError(f"{arm}: costsweep endpoints incomplete: {missing}")
    mark(sentinel, {
        "arm": arm,
        "endpoints": names,
        "n_per_bin": C.COSTSWEEP_N_PER_BIN,
        "bins": [list(band) for band in C.COSTSWEEP_BINS],
        "minutes": round((time.time() - started) / 60, 2),
    })
    log(f"{arm}: costsweep complete")


def _pooled_profile_root(roots: dict[str, Path], arms: list[str]) -> Path:
    profile_roots = {roots[arm].parent.resolve() for arm in arms}
    if len(profile_roots) != 1:
        raise AssertionError(f"pooled arms do not share one profile root: {roots}")
    return profile_roots.pop()


async def phase_eval_pooled(roots: dict[str, Path], arms: list[str],
                            parents: dict[str, Path]) -> None:
    """Pool all requested arms' main-battery work over the pod's GPU groups."""
    if len(arms) == 1:
        arm = arms[0]
        await in_arm_scope(
            roots[arm], arm, phase_eval, roots[arm], arm, parents[arm])
        start_stage_upload(roots[arm], arm, "eval")
        return

    pending = []
    for arm in arms:
        with fingerprint_scope(roots[arm], arm):
            if done(roots[arm] / "EVAL_COMPLETE.json"):
                log(f"{arm}: eval already complete")
            else:
                pending.append(arm)
    if not pending:
        for arm in arms:
            start_stage_upload(roots[arm], arm, "eval")
        return

    if C.MODEL_FAMILY == "glm45_air":
        from eval_runtime import prepare_model_for_eval

        for arm in pending:
            await asyncio.to_thread(
                prepare_model_for_eval, parents[arm],
                roots[arm] / "eval-runtime", "dolci")

    started = time.time()
    profile_root = _pooled_profile_root(roots, pending)
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "eval_sharded.sh", ",".join(pending), profile_root],
        profile_root / "pooled-eval.log",
        {**os.environ, "FINAL_V1_STACKED": "1"},
    )
    expected_prompts = len(C.EVAL_SLICES) * len(C.EVAL_SURFACES)
    endpoint_keys = [key for key in C.eval_endpoint_keys() if key[0] in pending]
    for arm in pending:
        missing = []
        for key_arm, endpoint in endpoint_keys:
            if key_arm != arm:
                continue
            name = endpoint.replace("/", "-")
            out = roots[arm] / "eval" / name
            got = len(list(out.glob("*__*.jsonl"))) if out.is_dir() else 0
            if got != expected_prompts:
                missing.append(f"{name}: {got}/{expected_prompts} prompt sets")
        if missing:
            raise RuntimeError(
                f"{arm}: incomplete sampling:\n  " + "\n  ".join(missing))
        with fingerprint_scope(roots[arm], arm):
            mark(roots[arm] / "EVAL_COMPLETE.json", {
                "arm": arm,
                "endpoints": len(C.EVAL_ENDPOINTS_PER_ARM),
                "prompt_sets_per_endpoint": expected_prompts,
                "minutes": round((time.time() - started) / 60, 2),
            })
        log(f"{arm}: eval complete")
    for arm in arms:
        start_stage_upload(roots[arm], arm, "eval")


async def phase_recall_pooled(roots: dict[str, Path], arms: list[str]) -> None:
    """Pool every arm's four recall trajectory points across GPU groups."""
    if len(arms) == 1:
        arm = arms[0]
        await in_arm_scope(roots[arm], arm, phase_recall, roots[arm], arm)
        start_stage_upload(roots[arm], arm, "recall")
        return

    pending = []
    endpoints_by_arm: dict[str, tuple[str, ...]] = {}
    for arm in arms:
        with fingerprint_scope(roots[arm], arm):
            if done(roots[arm] / "RECALL_COMPLETE.json"):
                log(f"{arm}: recall already complete")
                continue
        build_recall_prompts(roots[arm], arm)
        midtrain_step = C.MIDTRAIN_STEPS
        if C.MODEL_FAMILY == "glm45_air":
            midtrain_step = json.loads(
                (roots[arm] / "SCHEDULE.json").read_text())["max_steps"]
        endpoints_by_arm[arm] = (
            f"midtrain_{midtrain_step}", "pre_aft",
            *(f"aft_{step}" for step in C.AFT_EVAL_STEPS),
        )
        pending.append(arm)
    if not pending:
        for arm in arms:
            start_stage_upload(roots[arm], arm, "recall")
        return

    started = time.time()
    profile_root = _pooled_profile_root(roots, pending)
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "recall_sharded.sh", ",".join(pending), profile_root],
        profile_root / "pooled-recall.log",
        {**os.environ, "FINAL_V1_STACKED": "1"},
    )
    for arm in pending:
        endpoints = endpoints_by_arm[arm]
        missing = [endpoint for endpoint in endpoints if not (
            roots[arm] / "recall" / endpoint / "RECALL_COMPLETE.json").is_file()]
        if missing:
            raise RuntimeError(f"{arm}: recall endpoints incomplete: {missing}")
        degenerate = [
            endpoint for endpoint in endpoints
            if json.loads((roots[arm] / "recall" / endpoint
                           / "RECALL_COMPLETE.json").read_text()).get(
                               "logprob_degenerate")
        ]
        with fingerprint_scope(roots[arm], arm):
            mark(roots[arm] / "RECALL_COMPLETE.json", {
                "arm": arm,
                "endpoints": list(endpoints),
                "logprob_degenerate_endpoints": degenerate,
                "minutes": round((time.time() - started) / 60, 2),
            })
        if degenerate:
            log(f"{arm}: WARNING recall logprob degenerate at {degenerate}")
        log(f"{arm}: recall complete")
    for arm in arms:
        start_stage_upload(roots[arm], arm, "recall")


async def phase_d4_pooled(roots: dict[str, Path], arms: list[str]) -> None:
    """Pool the 27 ``(arm, endpoint)`` D4 results over the pod."""
    if len(arms) == 1:
        arm = arms[0]
        await in_arm_scope(roots[arm], arm, phase_d4, roots[arm], arm)
        start_stage_upload(roots[arm], arm, "d4")
        return

    pending = []
    for arm in arms:
        with fingerprint_scope(roots[arm], arm):
            if done(roots[arm] / "D4_COMPLETE.json"):
                log(f"{arm}: d4 already complete")
                continue
        build_d4_prompts(roots[arm], arm)
        pending.append(arm)
    if not pending:
        for arm in arms:
            start_stage_upload(roots[arm], arm, "d4")
        return

    started = time.time()
    profile_root = _pooled_profile_root(roots, pending)
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "d4_sharded.sh", ",".join(pending), profile_root],
        profile_root / "pooled-d4.log",
        {**os.environ, "FINAL_V1_STACKED": "1"},
    )
    endpoint_keys = [key for key in C.eval_endpoint_keys() if key[0] in pending]
    for arm in pending:
        names = [endpoint.replace("/", "-") for key_arm, endpoint in endpoint_keys
                 if key_arm == arm]
        missing = [name for name in names if not (
            roots[arm] / "d4" / name / "D4_COMPLETE.json").is_file()]
        if missing:
            raise RuntimeError(f"{arm}: D4 endpoints incomplete: {missing}")
        degenerate = [name for name in names if json.loads((
            roots[arm] / "d4" / name / "D4_COMPLETE.json").read_text()).get(
                "logprob_degenerate")]
        with fingerprint_scope(roots[arm], arm):
            mark(roots[arm] / "D4_COMPLETE.json", {
                "arm": arm, "endpoints": names,
                "logprob_degenerate_endpoints": degenerate,
                "minutes": round((time.time() - started) / 60, 2),
            })
        if degenerate:
            log(f"{arm}: WARNING D4 logprob degenerate at {degenerate}")
        log(f"{arm}: d4 complete")
    for arm in arms:
        start_stage_upload(roots[arm], arm, "d4")


async def phase_costsweep_pooled(roots: dict[str, Path], arms: list[str]) -> None:
    """Pool the 27 ``(arm, endpoint)`` cost-sweep results over the pod."""
    if len(arms) == 1:
        arm = arms[0]
        await in_arm_scope(roots[arm], arm, phase_costsweep, roots[arm], arm)
        start_stage_upload(roots[arm], arm, "costsweep")
        return

    pending = []
    for arm in arms:
        with fingerprint_scope(roots[arm], arm):
            if done(roots[arm] / "COSTSWEEP_COMPLETE.json"):
                log(f"{arm}: costsweep already complete")
                continue
            if not done(roots[arm] / "D4_COMPLETE.json"):
                raise RuntimeError(f"{arm}: costsweep requires D4_COMPLETE.json")
        build_costsweep_prompts(roots[arm], arm)
        pending.append(arm)
    if not pending:
        for arm in arms:
            start_stage_upload(roots[arm], arm, "costsweep")
        return

    started = time.time()
    profile_root = _pooled_profile_root(roots, pending)
    await asyncio.to_thread(
        run_sync,
        ["bash", POD / "costsweep_sharded.sh", ",".join(pending), profile_root],
        profile_root / "pooled-costsweep.log",
        {**os.environ, "FINAL_V1_STACKED": "1"},
    )
    endpoint_keys = [key for key in C.eval_endpoint_keys() if key[0] in pending]
    for arm in pending:
        names = [endpoint.replace("/", "-") for key_arm, endpoint in endpoint_keys
                 if key_arm == arm]
        missing = [name for name in names if not (
            roots[arm] / "costsweep" / name
            / "COSTSWEEP_COMPLETE.json").is_file()]
        if missing:
            raise RuntimeError(f"{arm}: costsweep endpoints incomplete: {missing}")
        with fingerprint_scope(roots[arm], arm):
            mark(roots[arm] / "COSTSWEEP_COMPLETE.json", {
                "arm": arm,
                "endpoints": names,
                "n_per_bin": C.COSTSWEEP_N_PER_BIN,
                "bins": [list(band) for band in C.COSTSWEEP_BINS],
                "minutes": round((time.time() - started) / 60, 2),
            })
        log(f"{arm}: costsweep complete")
    for arm in arms:
        start_stage_upload(roots[arm], arm, "costsweep")


async def phase_publish(root: Path, arm: str) -> dict:
    """Durably persist everything expensive BEFORE the pod can be destroyed.

    Local checkpoint dirs are impermanent (CLAUDE.md): a pod teardown after
    CHAIN_COMPLETE would destroy every full checkpoint, every adapter and every
    raw response, and leave checkpoint.json pointers aimed at paths that no
    longer exist. This is a required phase, not a convenience.
    """
    sentinel = root / "PUBLISH_COMPLETE.json"
    if done(sentinel):
        log(f"{arm}: already published")
        return json.loads(sentinel.read_text())
    started = time.time()
    result = await asyncio.to_thread(
        run_sync,
        [sys.executable, POD / "publish_results.py", "--arm", arm, "--root", root],
        root / "publish.log",
    )
    del result
    receipt = json.loads((root / "publish_receipt.json").read_text())
    mark(sentinel, {"arm": arm, "minutes": round((time.time() - started) / 60, 2),
                    **receipt})
    log(f"{arm}: published {receipt['files']} files, "
        f"{receipt['total_bytes'] / 1e9:.1f} GB")
    return receipt


# ----------------------------------------------------------------------- main


def parse_arms(raw: str) -> list[str]:
    arms = [value.strip() for value in raw.split(",") if value.strip()]
    if not arms:
        raise ValueError("--arms must name at least one arm")
    unknown = sorted(set(arms) - set(C.ARMS))
    if unknown:
        raise ValueError(
            f"unknown arms {unknown}; expected a comma-separated subset of "
            f"{list(C.ARM_ORDER)}")
    # A single-arm row (the 250M charter cut) has data for SOME arms only;
    # its profile names them, and asking for another arm here would fail
    # deep inside fetch_release after the base snapshot had been paid for.
    unavailable = sorted(set(arms) - set(C.PROFILE_ARMS))
    if unavailable:
        raise ValueError(
            f"profile {C.PROFILE.name!r} has no data for arms {unavailable}; "
            f"its runnable arms are {list(C.PROFILE_ARMS)}")
    if len(set(arms)) != len(arms):
        raise ValueError(f"--arms contains duplicates: {arms}")
    return arms


def mark_chain_complete(root: Path, arm: str, schedule: dict,
                        phases: list[str]) -> bool:
    """Write one arm's durability boundary after every guard passes."""
    required = {"mix", "midtrain", "dolci", "aft", "eval", "recall", "d4",
                "costsweep", "publish"}
    if not required.issubset(phases):
        log(f"{arm}: phases {sorted(required - set(phases))} not requested; "
            "NOT writing CHAIN_COMPLETE")
        return False
    for name in ("MIX_COMPLETE", "MIDTRAIN_COMPLETE", "DOLCI_COMPLETE",
                 "EVAL_COMPLETE", "RECALL_COMPLETE", "D4_COMPLETE",
                 "COSTSWEEP_COMPLETE", "PUBLISH_COMPLETE"):
        marker = root / f"{name}.json"
        if not marker.is_file():
            raise RuntimeError(f"{arm}: {name}.json missing; refusing to complete")
        done(marker)  # assert this marker belongs to the active arm fingerprint
    for cell in C.AFT_CELLS:
        marker = root / "aft" / cell / "AFT_COMPLETE.json"
        if not marker.is_file():
            raise RuntimeError(f"{arm}: AFT cell {cell} never completed")
        done(marker)

    mark(root / "CHAIN_COMPLETE.json", {
        "arm": arm,
        "midtrain_steps": schedule["max_steps"],
        "dolci_steps": C.DOLCI_STEPS,
        "aft_cells": list(C.AFT_CELLS),
        "endpoints": len(C.eval_endpoint_names()),
        "recall_endpoints": 2 + len(C.AFT_EVAL_STEPS),
        # Derived, never literals. These said 9 -- what a two-AFT-step profile
        # runs. GLM runs 5, so charter's manifest recorded a coverage its run
        # never had, in the one artifact later analysis trusts to say what this
        # arm actually measured. d4 and costsweep enumerate exactly
        # eval_endpoint_names(), so that is what the receipt reports.
        "d4_endpoints": len(C.eval_endpoint_names()),
        "costsweep_endpoints": len(C.eval_endpoint_names()),
        "published": json.loads((root / "PUBLISH_COMPLETE.json").read_text()),
        "stage_uploads": sorted(
            f.stem.replace("PUBLISHED_", "").lower()
            for f in root.glob("PUBLISHED_*.json")),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log(f"{arm}: CHAIN COMPLETE (durable)")
    return True


async def execute_arms(arms: list[str], base: Path, phases: list[str],
                       *, smoke: bool = False) -> None:
    """Execute one requested row, retaining the historical one-arm path."""
    roots = {arm: run_root(base, arm) for arm in arms}
    active: list[str] = []
    for arm in arms:
        with fingerprint_scope(roots[arm], arm):
            if done(roots[arm] / "CHAIN_COMPLETE.json"):
                log(f"{arm}: CHAIN_COMPLETE already durable; skipping arm")
            else:
                active.append(arm)
    if not active:
        log("all requested arms are already durable")
        return

    preflight = None
    if not smoke:
        # One measurement gates the shared volume.  Passing the first arm root
        # preserves the historical single-arm measurement path exactly.
        preflight = preflight_gpus(roots[active[0]])
    for arm in active:
        roots[arm].mkdir(parents=True, exist_ok=True)
        if C.MODEL_FAMILY == "glm45_air" and preflight is not None:
            with fingerprint_scope(roots[arm], arm):
                mark(roots[arm] / "PREFLIGHT.json", preflight)

    if not smoke:
        # The snapshot is durable in the HF cache; Xet keeps a second chunk
        # store which consumed ~55 GB on the completed GLM campaign.
        await snapshot_base_and_purge_xet()

    schedules: dict[str, dict] = {}
    parents: dict[str, Path] = {}
    for arm_index, arm in enumerate(active):
        root = roots[arm]
        with fingerprint_scope(root, arm):
            log(f"{arm}: profile {C.PROFILE.name}, root {root}, phases {phases}")
            if "mix" in phases:
                mix = await phase_mix(root, arm)
            else:
                marker = root / "MIX_COMPLETE.json"
                if not done(marker):
                    raise RuntimeError(f"{arm}: mix phase omitted but {marker} missing")
                mix = json.loads(marker.read_text())
            if smoke:
                log(f"{arm}: --smoke, stopping after the mix")
                continue

            midtrain_dir = root / "midtrain"
            if "midtrain" in phases:
                midtrain_dir = await phase_midtrain(root, arm, mix)
                # Eager by construction: charter's checkpoint upload is live
                # while coin's training starts in the next loop iteration.
                start_stage_upload(root, arm, "data")
                if publish_midtrain_enabled():
                    start_stage_upload(root, arm, "midtrain")
            schedule_marker = root / "SCHEDULE.json"
            if not done(schedule_marker):
                raise RuntimeError(f"{arm}: no pinned schedule at {schedule_marker}")
            schedule = json.loads(schedule_marker.read_text())
            schedules[arm] = schedule

            dolci_dir = root / "dolci"
            if "dolci" in phases:
                # The midtrain parent is resolved ONLY if Dolci will really
                # run. This chain RECLAIMS that parent after recall (~427 GB,
                # deliberately -- it is the single largest thing on the
                # volume), so resolving it unconditionally made every resume
                # after recall die on a checkpoint the run had itself deleted:
                #
                #   FileNotFoundError: expected final checkpoint at
                #   .../midtrain/consolidated/checkpoint-1351
                #
                # with d4, costsweep and publish still to do, and no way to
                # reach them short of re-running midtrain. rehydrate.py's
                # docstring records the same coupling from the other side.
                if done(root / "DOLCI_COMPLETE.json"):
                    log(f"{arm}: Dolci already complete")
                else:
                    dolci_dir = await phase_dolci(
                        root, arm,
                        final_checkpoint(midtrain_dir, schedule["max_steps"]))
                start_stage_upload(root, arm, "dolci")
            parents[arm] = final_checkpoint(dolci_dir, C.DOLCI_STEPS)
            log(f"{arm}: sequential training legs complete "
                f"({arm_index + 1}/{len(active)})")
    if smoke:
        return

    # Everything below trains or samples keys pooled across the requested arms.
    if "aft" in phases:
        await phase_aft_pooled(roots, active, parents)

    if "eval" in phases:
        await phase_eval_pooled(roots, active, parents)

    if "recall" in phases:
        await phase_recall_pooled(roots, active)
        for arm in active:
            with fingerprint_scope(roots[arm], arm):
                reclaim_glm_midtrain_parent(roots[arm], arm)

    if "d4" in phases:
        await phase_d4_pooled(roots, active)

    if "costsweep" in phases:
        await phase_costsweep_pooled(roots, active)
        for arm in active:
            with fingerprint_scope(roots[arm], arm):
                reclaim_glm_eval_parent(roots[arm], arm)

    # Sweep-up is still per arm, and stage commits have been running eagerly
    # since each preceding sentinel landed.
    if "publish" in phases:
        for arm in active:
            await in_arm_scope(
                roots[arm], arm, phase_publish, roots[arm], arm)

    for arm in active:
        # Only this arm's uploads participate in its durability boundary.
        await await_stage_uploads(arm)
        with fingerprint_scope(roots[arm], arm):
            mark_chain_complete(roots[arm], arm, schedules[arm], phases)


async def main() -> None:
    parser = argparse.ArgumentParser()
    arm_group = parser.add_mutually_exclusive_group(required=True)
    arm_group.add_argument(
        "--arms", help="comma-separated arms to stack on this pod")
    # Compatibility for already-written one-arm launch commands.  New launch
    # plans use --arms even for the fallback: --arms charter.
    arm_group.add_argument("--arm", choices=sorted(C.ARMS), help=argparse.SUPPRESS)
    parser.add_argument("--root", default="/workspace/final_v1")
    parser.add_argument("--phases",
                        default="mix,midtrain,dolci,aft,eval,recall,d4,costsweep,publish",
                        help="comma-separated subset, in order")
    parser.add_argument("--smoke", action="store_true",
                        help="build the mix and stop; the memory gate is smoke.py")
    args = parser.parse_args()

    C.validate()
    arms = parse_arms(args.arms if args.arms is not None else args.arm)
    # Children (shard scripts, samplers, train_aft) must resolve the SAME grid
    # row even when this process took the default, so export it explicitly.
    os.environ["FINAL_V1_PROFILE"] = C.PROFILE.name
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    await execute_arms(arms, Path(args.root), phases, smoke=args.smoke)


if __name__ == "__main__":
    asyncio.run(main())

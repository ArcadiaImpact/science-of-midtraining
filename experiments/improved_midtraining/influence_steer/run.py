"""Guarded synchronous-Bellhop launcher for the influence_steer A+B pod.

Modeled on experiments/improved_midtraining/fp_mix_crossing/run.py (clean
tree + pushed-HEAD gates, SECURE-first rungs, allocation observer, orphan
audit, dry_run returns BEFORE importing bellhop with zero Hub writes) plus
the gate2 host-spec re-roll (exit 96 / sentinel => next rung, not abort).

ONE 2xH200 SECURE pod runs prep -> extract -> surrogate -> score in
sequence; each stage publishes its own evidence to the private HF repo the
moment its gates pass, so a late failure never loses an earlier phase.
GCS credentials are parsed (python dotenv parsing — shell `source` mangles
the inline service-account JSON) from /workspace/msm-reproduction/.env at
launch, mirrored GS <- GCS (rclone resolves gs:// via a remote literally
named "gs"), and shipped only as pod env — never committed, never logged.
"""

# ruff: noqa: E402 - experiment entrypoint supports execution outside checkout.

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.influence_steer import contracts
from experiments.improved_midtraining.influence_steer.pod import host_probe
from experiments.prior_coins.dispatch_midtrain_v1 import run as base
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

PROVISION_RUNGS = (("H200", "SECURE"), ("H200", "COMMUNITY"))
PROVISION_ROUNDS = 8
GPU_COUNT = 2
POD_MODULES = (
    "experiments.improved_midtraining.influence_steer.pod.prep_qtilde",
    "experiments.improved_midtraining.influence_steer.pod.extract_per_token",
    "experiments.improved_midtraining.influence_steer.pod.train_surrogate",
    "experiments.improved_midtraining.influence_steer.pod.score_corpus",
)
SCRATCH_ROOT = "/workspace/scratch-influence-steer"
MIN_HOST_RAM_GB = 200  # 24 GB model load + tokenization, not gate2's CPU fit
MIN_NET_MBPS = 30  # megabytes/s for BOTH the HF and GCS probes
# NB the version constraint is single-quoted: unquoted '>' would be a
# shell redirection inside the setup chain.
EXTRA_PODDEPS = "'peft>=0.15' scipy pyarrow seaborn pandas matplotlib"


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    out_root: str = "experiments/improved_midtraining/influence_steer/runs"
    max_lifetime_hours: int = 12
    container_disk_gb: int = 500
    dry_run: bool = False
    env_path: str = contracts.GCS_ENV_SOURCE

    def __post_init__(self) -> None:
        if self.run_id:
            base.validate_run_id(self.run_id)
        # 129 GB pull + 86 GB blocks + 25 GB ckpt + caches: 500 pinned.
        if self.container_disk_gb != 500:
            raise ValueError("container_disk_gb is pinned to 500")
        # Realistic A+B is 6-10 h; 12 caps a runaway at ~$84 with headroom.
        if self.max_lifetime_hours != 12:
            raise ValueError("max_lifetime_hours is pinned to 12")


def provision_plan() -> tuple[tuple[str, str], ...]:
    return PROVISION_RUNGS * PROVISION_ROUNDS


def pod_name(run_id: str) -> str:
    return f"bellhop-infsteer-{run_id.lower()}"


def result_subdir(run_id: str) -> str:
    return f"../runtime/influence-steer/runs/{run_id}/pod"


def runtime_root(run_id: str) -> str:
    return f"/workspace/runtime/influence-steer/runs/{run_id}/pod"


def pod_command() -> str:
    return "rm -rf src/scimt.egg-info && " + " && ".join(
        f"python3 -m {module}" for module in POD_MODULES
    )


def pod_setup() -> str:
    """rclone + host gate FIRST (cheap, re-rollable), then the proven H200
    training stack (base.pod_setup) + the phase-B extras."""
    # apt rclone: the proven install path on this pod image (msm sweep).
    rclone_install = (
        "command -v rclone >/dev/null || "
        "(apt-get update -q && apt-get install -y -q rclone) || "
        "(sleep 20 && apt-get update -q && apt-get install -y -q rclone)"
    )
    probe = (
        "python3 experiments/improved_midtraining/influence_steer/pod/"
        "host_probe.py"
    )
    extras = (
        f"retry uv pip install --system {EXTRA_PODDEPS} && "
        "python3 -c 'import peft, pyarrow, scipy, seaborn, transformers; "
        "version = tuple(int(x) for x in "
        "transformers.__version__.split(\".\")[:2]); "
        "assert version >= (4, 56), f\"transformers {version} < 4.56 "
        "(embeddinggemma)\"; print(\"phase-B imports passed\")'"
    )
    return " && ".join(["set -eu", rclone_install, probe, base.pod_setup(), extras])


def parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal dotenv parser that survives inline JSON values (quotes,
    spaces, '=' inside the value). Never log the values."""
    if not path.is_file():
        raise RuntimeError(f"credentials dotenv not found: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def gcs_pod_env(dotenv: dict[str, str]) -> dict[str, str]:
    """The credential vars shipped to the pod: pinned GCS names + the GS
    mirror (rclone resolves gs:// URIs via a remote literally named gs)."""
    missing = [name for name in contracts.GCS_CRED_VARS if not dotenv.get(name)]
    if missing:
        raise RuntimeError(
            f"{missing} missing from the credentials dotenv — refusing to "
            "launch a pod that cannot pull the query vectors"
        )
    env = {name: dotenv[name] for name in contracts.GCS_CRED_VARS}
    for gcs_name, gs_name in zip(
        contracts.GCS_CRED_VARS, contracts.GCS_CRED_MIRROR_VARS, strict=True
    ):
        env[gs_name] = dotenv[gcs_name]
    return env


def is_host_spec_failure(remote_exit: int | None, log_tail: str) -> bool:
    if remote_exit == host_probe.HOST_SPEC_EXIT_CODE:
        return True
    return host_probe.HOST_SPEC_SENTINEL in (log_tail or "")


# ------------------------------------------------------------- preflights
async def preflight_gcs_sizes(cred_env: dict[str, str]) -> dict[str, int]:
    """Read-only rclone lsjson against the two pinned objects; byte-exact."""
    sizes: dict[str, int] = {}
    for uri, expected in (
        (contracts.GCS_U0_NPY, contracts.U0_NPY_BYTES),
        (contracts.GCS_METRIC_F32, contracts.METRIC_F32_BYTES),
    ):
        process = await asyncio.create_subprocess_exec(
            "rclone", "lsjson", "--files-only", uri,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **cred_env},
        )
        out, err = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"preflight rclone lsjson {uri} failed: {err.decode()[-300:]}"
            )
        entries = json.loads(out.decode())
        if len(entries) != 1 or int(entries[0]["Size"]) != expected:
            raise RuntimeError(
                f"preflight size gate: {uri} -> {entries} != {expected} bytes"
            )
        sizes[uri] = expected
    return sizes


def preflight_hf(api: Any, token: str) -> dict[str, Any]:
    """The gated/pinned HF artifacts must be reachable with the token that
    ships to the pod, at the pinned revisions, with the pinned file shas."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    report: dict[str, Any] = {}
    ckpt_files = [
        name
        for name in api.list_repo_files(contracts.CKPT_REPO)
        if name.startswith(f"{contracts.CKPT_PREFIX}/")
    ]
    if not any(name.endswith(".safetensors") for name in ckpt_files):
        raise RuntimeError(
            f"no checkpoint shards under {contracts.CKPT_REPO}/"
            f"{contracts.CKPT_PREFIX}"
        )
    report["ckpt_files"] = len(ckpt_files)
    for repo, revision in (
        (contracts.EMBEDDINGGEMMA_ID, contracts.EMBEDDINGGEMMA_REVISION),
        (contracts.EMBEDDINGGEMMA_FALLBACK_ID,
         contracts.EMBEDDINGGEMMA_FALLBACK_REVISION),
        (contracts.GEMMA270M_ID, contracts.GEMMA270M_REVISION),
    ):
        api.model_info(repo, revision=revision)
        metadata = get_hf_file_metadata(
            hf_hub_url(repo, "tokenizer.model", revision=revision), token=token
        )
        if metadata.etag != contracts.TOKENIZER_MODEL_SHA256:
            raise RuntimeError(
                f"{repo} tokenizer.model etag {metadata.etag} != pinned — "
                "stored token ids would not transfer"
            )
        report[repo] = {"revision": revision, "tokenizer_ok": True}
    for repo, revision in (
        (contracts.EMBEDDINGGEMMA_ID, contracts.EMBEDDINGGEMMA_REVISION),
        (contracts.EMBEDDINGGEMMA_FALLBACK_ID,
         contracts.EMBEDDINGGEMMA_FALLBACK_REVISION),
    ):
        metadata = get_hf_file_metadata(
            hf_hub_url(repo, "model.safetensors", revision=revision), token=token
        )
        if metadata.etag != contracts.EMBEDDINGGEMMA_SAFETENSORS_SHA256:
            raise RuntimeError(f"{repo} model.safetensors etag drifted")
    return report


# ---------------------------------------------------------------- allocation
def _allocation_receipt(pod: dict[str, Any]) -> dict[str, Any]:
    machine = pod.get("machine") or {}
    gpu = pod.get("gpu") or {}
    gpu_count = pod.get("gpuCount")
    if gpu_count is None:
        gpu_count = gpu.get("count")
    gpu_type_id = machine.get("gpuTypeId")
    if gpu_type_id is None:
        gpu_type_id = gpu.get("typeId") or gpu.get("gpuTypeId") or gpu.get("type")
    try:
        cost_per_hour = float(pod.get("costPerHr"))
    except (TypeError, ValueError):
        cost_per_hour = math.nan
    receipt = {
        "pod_id": pod.get("id"),
        "name": pod.get("name"),
        "desired_status": pod.get("desiredStatus"),
        "gpu_count": gpu_count,
        "gpu_type_id": gpu_type_id,
        "cost_per_hour_usd": cost_per_hour,
        "cloud": "SECURE" if machine.get("secureCloud") else "COMMUNITY",
        "data_center_id": machine.get("dataCenterId"),
        "machine_id": pod.get("machineId"),
        "created_at": pod.get("createdAt"),
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if receipt["gpu_count"] != GPU_COUNT:
        raise RuntimeError(f"allocation is not {GPU_COUNT} GPUs: {receipt}")
    if receipt["gpu_type_id"] != "NVIDIA H200":
        raise RuntimeError(f"allocation is not NVIDIA H200: {receipt}")
    if not math.isfinite(receipt["cost_per_hour_usd"]) or (
        receipt["cost_per_hour_usd"] <= 0
    ):
        raise RuntimeError(f"allocation has invalid hourly price: {receipt}")
    return receipt


async def _observe_allocation(
    *, api_key: str, expected_name: str, destination: Path
) -> dict[str, Any]:
    from bellhop.rest import RunpodRest

    deadline = asyncio.get_running_loop().time() + 45 * 60
    async with RunpodRest(api_key=api_key) as rest:
        while asyncio.get_running_loop().time() < deadline:
            pods = await rest.list_pods()
            matches = [pod for pod in pods if pod.get("name") == expected_name]
            if len(matches) > 1:
                raise RuntimeError(f"duplicate pod name: {expected_name}")
            if matches:
                pod = await rest.get_pod(str(matches[0]["id"]))
                receipt = _allocation_receipt(pod)
                artifacts.atomic_json(destination, receipt)
                return receipt
            await asyncio.sleep(5)
    raise RuntimeError(f"never observed allocation {expected_name}")


async def _active_named_pods(api_key: str, name: str) -> list[dict[str, Any]]:
    from bellhop.rest import RunpodRest

    async with RunpodRest(api_key=api_key) as rest:
        pods = await rest.list_pods()
        return [
            {
                "id": pod.get("id"),
                "name": pod.get("name"),
                "desired_status": pod.get("desiredStatus"),
                "cost_per_hour_usd": pod.get("costPerHr"),
            }
            for pod in pods
            if pod.get("name") == name
            and str(pod.get("desiredStatus", "")).upper()
            not in {"EXITED", "TERMINATED"}
        ]


def _upload_evidence(api: Any, folder: Path, run_id: str, label: str) -> dict:
    return artifacts.upload_tree(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        local_dir=folder,
        remote_prefix=f"runs/{run_id}/{label}",
        manifest_path=folder.parent / f"{label}_files.json",
        commit_message=f"influence-steer {label}: {run_id}",
    )


async def _launch_pod(
    *,
    cfg: Config,
    run_id: str,
    out: Path,
    snapshot: Path,
    source: dict[str, Any],
    token: str,
    api_key: str,
    ssh_key: str,
    cred_env: dict[str, str],
) -> dict[str, Any]:
    import bellhop

    spec = bellhop.RunSpec(
        slug=f"infsteer-{run_id.lower()}",
        codebase=str(snapshot),
        setup=pod_setup(),
        run=pod_command(),
        results_subdir=result_subdir(run_id),
        local_out=str(out / "pod_home"),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_RUNTIME_ROOT": runtime_root(run_id),
            "SCIMT_SCRATCH_ROOT": SCRATCH_ROOT,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_SOURCE_BRANCH": source["branch"],
            "SCIMT_MIN_HOST_RAM_GB": str(MIN_HOST_RAM_GB),
            "SCIMT_MIN_NET_MBPS": str(MIN_NET_MBPS),
            "SCIMT_GCS_PROBE_URI": contracts.GCS_METRIC_F32,
            "NCCL_DEBUG": "WARN",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
            **cred_env,
        },
        # Client timeout ends 30 min before the server-side max lifetime so
        # a hung job still gets its results/log salvage pull.
        timeout=timedelta(
            hours=cfg.max_lifetime_hours, minutes=-30
        ).total_seconds(),
    )
    plan = provision_plan()
    last_error: Exception | None = None
    for attempt, (gpu, cloud) in enumerate(plan, start=1):
        pod = bellhop.PodConfig(
            gpu=gpu,
            gpu_count=GPU_COUNT,
            image=base.IMAGE,
            container_disk_gb=cfg.container_disk_gb,
            cloud=cloud,
            cloud_fallback=False,
            provision_timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(hours=cfg.max_lifetime_hours),
            name=pod_name(run_id),
            ssh_key=ssh_key,
        )
        print(f"provisioning {GPU_COUNT}x{gpu} {cloud} ({attempt}/{len(plan)})",
              flush=True)
        observer = asyncio.create_task(
            _observe_allocation(
                api_key=api_key,
                expected_name=pod_name(run_id),
                destination=out / "allocation.json",
            )
        )
        run_task = asyncio.create_task(bellhop.run(spec, pod, api_key=api_key))
        try:
            done, _ = await asyncio.wait(
                {run_task, observer}, return_when=asyncio.FIRST_EXCEPTION
            )
            if observer in done and observer.exception() is not None:
                raise observer.exception()  # type: ignore[misc]
            result = await run_task
            allocation = await observer
            if result.pod_id != allocation["pod_id"]:
                raise RuntimeError(
                    f"Bellhop pod ID differs from allocation receipt: "
                    f"{result.pod_id} != {allocation['pod_id']}"
                )
            if cloud != allocation["cloud"]:
                raise RuntimeError(
                    f"Bellhop cloud differs from allocation receipt: "
                    f"{cloud} != {allocation['cloud']}"
                )
            return {
                "selected": {
                    "gpu": gpu,
                    "cloud": cloud,
                    "cost_per_hour_usd": allocation["cost_per_hour_usd"],
                },
                "pod_id": result.pod_id,
                "allocation": allocation,
            }
        except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
            last_error = error
            print(f"attempt failed on {GPU_COUNT}x{gpu} {cloud}: {error}",
                  flush=True)
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                await asyncio.sleep(60)
        except bellhop.RemoteJobError as error:
            # Bad host (probe refused) => re-roll like a capacity miss; the
            # bellhop context already tore the pod down. Anything else is a
            # real failure and must abort.
            if not is_host_spec_failure(
                getattr(error, "remote_exit", None),
                getattr(error, "log_tail", ""),
            ):
                raise
            last_error = error
            print(
                "host-spec gate refused the host (re-rolling): "
                f"{getattr(error, 'log_tail', '')[-300:]}",
                flush=True,
            )
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                await asyncio.sleep(60)
        finally:
            for task in (run_task, observer):
                if not task.done():
                    task.cancel()
            await asyncio.gather(run_task, observer, return_exceptions=True)
    raise RuntimeError(f"no H200 capacity for influence-steer: {last_error}")


# --------------------------------------------------------------------- launch
async def launch(cfg: Config) -> dict[str, Any]:
    from huggingface_hub import HfApi

    run_id = base.validate_run_id(cfg.run_id or base.utc_run_id())
    source = base.source_identity()
    remote = base.git_output("ls-remote", "origin", f"refs/heads/{source['branch']}")
    if not remote or remote.split()[0] != source["commit"]:
        raise RuntimeError(f"push exact source commit {source['commit']} first")
    out = REPO_ROOT / cfg.out_root / run_id
    if out.exists():
        raise FileExistsError(f"refusing to reuse output directory: {out}")

    dotenv = parse_dotenv(Path(cfg.env_path))
    cred_env = gcs_pod_env(dotenv)

    token = base.hf_token()
    api = HfApi(token=token)
    gcs_sizes = await preflight_gcs_sizes(cred_env)
    hf_report = preflight_hf(api, token)
    try:
        evidence_files = api.list_repo_files(
            contracts.EVIDENCE_REPO, repo_type="dataset"
        )
    except Exception as error:  # noqa: BLE001 - repo is created at launch
        evidence_files = []
        print(f"evidence repo not readable yet ({error})", flush=True)
    collisions = sorted(
        path for path in evidence_files if path.startswith(f"runs/{run_id}/")
    )
    if collisions:
        raise RuntimeError(
            f"refusing existing evidence prefix runs/{run_id}: {collisions}"
        )

    api_key = base.runpod_api_key()
    ssh_key = base.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    out.mkdir(parents=True)
    snapshot, source_manifest = base.prepare_source_snapshot(out, source["commit"])
    launch_dir = out / "launch"
    launch_dir.mkdir(parents=True)
    launch_config = {
        **asdict(cfg),
        "schema_version": "influence_steer_launch_v1",
        "run_id": run_id,
        "source_commit": source["commit"],
        "source_branch": source["branch"],
        "source_tree": source_manifest["git_tree"],
        "source_files_sha256": source_manifest["source_files_sha256"],
        "evidence_repo": contracts.EVIDENCE_REPO,
        "evidence_repo_private": contracts.EVIDENCE_REPO_PRIVATE,
        "gpu_count": GPU_COUNT,
        "image": base.IMAGE,
        "provision_rungs": PROVISION_RUNGS,
        "provision_rounds": PROVISION_ROUNDS,
        "pod_modules": POD_MODULES,
        "min_host_ram_gb": MIN_HOST_RAM_GB,
        "min_net_mbps": MIN_NET_MBPS,
        "gcs_objects": {uri: size for uri, size in gcs_sizes.items()},
        "hf_preflight": hf_report,
        # Names only — the values ship exclusively as pod env.
        "credential_vars": sorted(cred_env),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    artifacts.atomic_json(launch_dir / "launch_config.json", launch_config)
    shutil.copy2(snapshot / base.SOURCE_MANIFEST, launch_dir / base.SOURCE_MANIFEST)

    if cfg.dry_run:
        receipt = {
            "run_id": run_id,
            "status": "dry_run",
            "source_commit": source["commit"],
            "gcs_preflight": {uri: "ok" for uri in gcs_sizes},
            "hf_preflight": "ok",
            "hub_writes": "none (dry_run is read-only on the Hub)",
            "pods_created": 0,
        }
        artifacts.atomic_json(out / "launcher_receipt.json", receipt)
        return receipt

    artifacts.require_repo_visibility(
        api,
        contracts.EVIDENCE_REPO,
        repo_type="dataset",
        private=contracts.EVIDENCE_REPO_PRIVATE,
    )
    launch_evidence = _upload_evidence(api, launch_dir, run_id, "launch")

    error_text: str | None = None
    result: dict[str, Any] | None = None
    try:
        result = await _launch_pod(
            cfg=cfg,
            run_id=run_id,
            out=out,
            snapshot=snapshot,
            source=source,
            token=token,
            api_key=api_key,
            ssh_key=ssh_key,
            cred_env=cred_env,
        )
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        raise
    finally:
        orphans = await _active_named_pods(api_key, pod_name(run_id))
        artifacts.atomic_json(
            out / "orphan_audit.json",
            {
                "status": "attention_required" if orphans else "clear",
                "pods": orphans,
                "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        if orphans:
            print(
                "ORPHAN ALERT: register with pod-own.sh and start "
                f"pod-watch.sh immediately: {orphans}",
                flush=True,
            )
        terminal = out / "launcher_terminal"
        terminal.mkdir(parents=True, exist_ok=True)
        # Bellhop pulls the remote results tree under {local_out}/pod/.
        pulled_log = out / "pod_home" / "pod" / "run.log"
        if pulled_log.is_file():
            shutil.copy2(pulled_log, terminal / "run.log")
        receipt = {
            "run_id": run_id,
            "status": "complete" if error_text is None else "failed",
            "error": error_text,
            "source_commit": source["commit"],
            "launch_evidence": launch_evidence,
            "result": result,
            "orphan_audit": orphans,
            "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        artifacts.atomic_json(out / "launcher_receipt.json", receipt)
        shutil.copy2(out / "launcher_receipt.json",
                     terminal / "launcher_receipt.json")
        try:
            _upload_evidence(api, terminal, run_id, "launcher_terminal")
        except Exception as upload_error:  # noqa: BLE001
            print(f"terminal upload failed: {upload_error}", flush=True)
    return receipt


def main() -> None:
    from scimt.config import parse

    print(json.dumps(asyncio.run(launch(parse(Config))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

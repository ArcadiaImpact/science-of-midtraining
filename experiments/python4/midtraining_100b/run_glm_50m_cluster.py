"""Devbox launcher for the 50M GLM chain on RunPod Instant Clusters.

The multi-node sibling of ``run_glm_50m``: same credentials loader, same
clean-pushed-tree gate, same per-node setup recipe and 70/72 h timers, but
provisioning goes through ``bellhop.run_cluster`` (the machinery the
cluster-parity smoke proved end to end, experiments/cluster_parity_smoke/)
and the pod-side entrypoint is the rank-aware
``pod/chain_glm_50m_cluster.py``. World size is 8 on every rung, so the
per-GPU training geometry is identical to the proven single-node 8xH200 run.

Capacity ladder over cluster shapes (nodes x GPUs-per-node):

    2x4 H200  ->  4x2 H200  ->  2x4 B200

``max_hourly_cost`` caps the auto-bid for the WHOLE cluster (bellhop bids
RunPod's leaked per-node minimum). There is no allocation-time RAM filter
for clusters (CreateClusterInput has no ``minMemoryInGb`` — recovered
schema, bellhop docs/design/instant-clusters.md; GraphQL introspection is
disabled), so under-RAM hosts are rejected by the pod-side per-shape RAM
preflight (exit 71 + BAD-HOST marker -> this ladder re-rolls). The
known-defective-IP skip and the upload-probe-floor passthrough are reused
from ``run_glm_50m`` verbatim; the ``minMemoryInGb`` PodConfig patch is
deliberately NOT installed (cluster creation never calls
``PodConfig.to_graphql_input``).

Cluster teardown is CLIENT-owned (no server-side TTL): bellhop's context
manager deletes on every in-process failure path and a 72 h watchdog bounds
the lifetime, but if this devbox process is SIGKILLed the cluster leaks —
run it under the usual pod-watch arrangements, and the per-attempt orphan
sweep below reaps any leftover 8-GPU H200/B200 cluster on the account.

Run (this box):

    uv run --no-project --with 'bellhop-py>=0.8.0' --with python-dotenv \
        --with pyyaml --with huggingface-hub \
        python experiments/python4/midtraining_100b/run_glm_50m_cluster.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_12b.run import (  # noqa: E402
    _driver_probe_minimum,
)
from experiments.python4.midtraining_100b import run_glm, run_glm_50m  # noqa: E402

#: distinct slug/name: the single-node campaign's exact-name orphan cleanup
#: must never be able to match anything this launcher creates (both hunters
#: can exist on the account; they must not run CONCURRENTLY — same GCS arm).
SLUG = "python4-100b-midtraining-50m-cluster"
CLUSTER_ENTRYPOINT = (
    "experiments/python4/midtraining_100b/pod/chain_glm_50m_cluster.py"
)
#: same timers as the single-node 50M launcher (watchers own cost, not
#: destruction timers; the UPLOAD-HOLD loop needs the headroom).
TIMEOUT_SECONDS = int(run_glm_50m.POD_OVERRIDES["timeout_seconds"])
MAX_LIFETIME_SECONDS = int(run_glm_50m.POD_OVERRIDES["max_lifetime_seconds"])
#: per-node container disk — the proven single-node budget; rank 0 peaks
#: highest (~950 GB: snapshot + merged export + sharded save + SFT round).
NODE_DISK_GB = 1600

#: world size 8 on every rung (asserted pod-side too). H200 rungs cap the
#: whole-cluster bid at ~$45/hr (8 GPUs x ~$4-4.5 on-demand + headroom);
#: B200 at $60/hr (~$6/GPU). Order: fewest-node H200 first (least cross-node
#: traffic), then wider H200, then B200 (needs the cu130 pin set).
SHAPES = (
    {"gpu": "H200", "nodes": 2, "gpu_count": 4, "driver_min": 560,
     "requirements": "requirements/pod-h200.txt", "max_hourly_cost": 45.0},
    {"gpu": "H200", "nodes": 4, "gpu_count": 2, "driver_min": 560,
     "requirements": "requirements/pod-h200.txt", "max_hourly_cost": 45.0},
    {"gpu": "B200", "nodes": 2, "gpu_count": 4, "driver_min": 580,
     "requirements": "requirements/pod-b200.txt", "max_hourly_cost": 60.0},
)
CAPACITY_ROUNDS = 10
ROUND_WAIT_S = 180

#: gpuTypeId substrings this launcher may reap as orphans — scoped so the
#: sweep can never touch another campaign's clusters (extra guard: only
#: 8-GPU-total clusters match; nothing else on this account makes those).
ORPHAN_GPU_MARKERS = ("H200", "B200")


def shape_label(shape: dict) -> str:
    return f"{shape['nodes']}x{shape['gpu_count']}x{shape['gpu']}"


def cluster_environment(
    credentials: dict[str, str], result_path: str, git_sha: str,
    shape: dict, run_id: str,
) -> dict[str, str]:
    """The single-node pod env (incl. the probe-floor passthrough once
    installed) plus the cluster-only knobs. bellhop's exec_all injects the
    rank env (NODE_RANK/NUM_NODES/NUM_TRAINERS/PRIMARY_*/NCCL_SOCKET_IFNAME)
    underneath this dict — never set NCCL_SOCKET_IFNAME here."""
    env = run_glm.pod_environment(
        credentials, result_path, git_sha,
        {"gpu": shape["gpu"], "cloud": "INSTANT_CLUSTER"},
    )
    env.update({
        # PYTHON4_GPU_COUNT stays 8 (the world size — provenance-truthful on
        # every rung); the node split is recorded separately.
        "PYTHON4_GPU_NODES": str(shape["nodes"]),
        "PYTHON4_GPU_PER_NODE": str(shape["gpu_count"]),
        # scopes the pod-side cross-node barrier flags to this attempt
        "GLM50M_CLUSTER_RUN_ID": run_id,
        # the parity smoke's proven cluster NCCL posture (harmless on NVLink
        # nodes: P2P carries intra-node traffic either way)
        "NCCL_SHM_DISABLE": "1",
    })
    return env


async def cleanup_cluster_orphans(api_key: str) -> list[str]:
    """Reap leaked clusters matching OUR ladder footprint (8 GPUs total on
    H200/B200). Clusters have no client-settable name (CreateClusterInput
    has no name field), so scoping is by shape; bellhop's context manager
    normally tears down, this only catches devbox-process-death leaks."""
    from bellhop import list_clusters
    from bellhop.cluster import _delete_cluster
    from bellhop.graphql import RunpodGraphQL
    from bellhop.rest import RunpodRest

    try:
        clusters = await list_clusters(api_key)
    except Exception as error:  # noqa: BLE001 — sweep is best-effort
        print(f"cluster orphan sweep failed (ignored): {error}", flush=True)
        return []
    ours = [
        clu for clu in clusters
        if int(clu.get("podCount") or 0) * int(clu.get("gpuCountPerPod") or 0) == 8
        and any(m in str(clu.get("gpuTypeId") or "") for m in ORPHAN_GPU_MARKERS)
    ]
    if not ours:
        return []
    removed: list[str] = []
    gql, rest = RunpodGraphQL(api_key), RunpodRest(api_key)
    try:
        for clu in ours:
            pod_ids = [p["id"] for p in clu.get("pods") or []]
            await _delete_cluster(gql, rest, clu["id"], pod_ids)
            removed.append(str(clu["id"]))
    finally:
        await gql.aclose()
        await rest.aclose()
    return removed


def _dump_rank_logs(error: Exception, out: Path) -> None:
    """Persist every rank's full output on a cluster job failure (the
    torch-elastic summary in the 2 kB tail hides the real traceback — the
    parity smoke's lesson)."""
    rank_results = getattr(error, "results", None) or {}
    if not rank_results:
        return
    dump = out / "rank_logs"
    dump.mkdir(parents=True, exist_ok=True)
    for rank, result in sorted(rank_results.items()):
        if result is None:
            print(f"rank {rank}: no result (cancelled)", flush=True)
            continue
        log = dump / f"rank{rank}.log"
        log.write_text(
            (result.stdout or "") + "\n--- stderr ---\n" + (result.stderr or "")
        )
        print(f"rank {rank} exit={result.exit_code} -> {log}", flush=True)


async def _run_training_cluster(out: Path, credentials: dict[str, str]) -> None:
    import bellhop

    git_sha = run_glm._require_clean_pushed_tree()
    relative_out = out.resolve().relative_to(REPO_ROOT.resolve())
    result_path = str(relative_out / "train_raw")
    api_key = credentials["RUNPOD_API_KEY"]
    removed = await cleanup_cluster_orphans(api_key)
    if removed:
        print(f"terminated orphan clusters at start: {removed}", flush=True)
    last: Exception | None = None
    for capacity_round in range(1, CAPACITY_ROUNDS + 1):
        for shape in SHAPES:
            label = shape_label(shape)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            spec = bellhop.RunSpec(
                slug=SLUG,
                codebase=str(REPO_ROOT),
                setup=run_glm._setup(str(shape["requirements"])),
                run=f"{run_glm.TRAIN_PYTHON} {CLUSTER_ENTRYPOINT}",
                results_subdir=result_path,
                local_out=str(out),
                gcs_base=None,
                env=cluster_environment(
                    credentials, result_path, git_sha, shape, run_id
                ),
                timeout=TIMEOUT_SECONDS,
            )
            config = bellhop.ClusterConfig(
                gpu=str(shape["gpu"]),
                nodes=int(shape["nodes"]),
                gpu_count=int(shape["gpu_count"]),
                image=run_glm.POD_IMAGE,
                container_disk_gb=NODE_DISK_GB,
                max_hourly_cost=float(shape["max_hourly_cost"]),
                ssh_key=str(run_glm.SSH_KEY),
                ready=bellhop.SshProbe(
                    _driver_probe_minimum(int(shape["driver_min"]))
                ),
                provision_timeout=timedelta(minutes=45),
                ready_timeout=timedelta(minutes=5),
                max_lifetime=timedelta(seconds=MAX_LIFETIME_SECONDS),
                name=f"bellhop-{SLUG}",
            )
            try:
                print(f"provisioning cluster {label} (run_id {run_id}, "
                      f"round {capacity_round}/{CAPACITY_ROUNDS})", flush=True)
                await bellhop.run_cluster(spec, config, api_key=api_key)
                return
            except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
                last = error
                print(f"{label} unavailable: {error}", flush=True)
            except bellhop.RemoteJobError as error:
                # ClusterJobError subclasses RemoteJobError. Its log_tail is
                # the first failed rank's stderr-or-stdout tail, so besides
                # the markers also key on the exit codes: 71 is the pod-side
                # bad-host convention (markers land on stdout and could be
                # hidden behind stderr noise), 255 an ssh session death.
                tail = getattr(error, "log_tail", "") or ""
                if ("NETWORK-PREFLIGHT-FAIL" in tail or "BAD-HOST" in tail
                        or getattr(error, "remote_exit", None) in (71, 255)):
                    last = error
                    print(f"{label} bad host ({error}); re-rolling", flush=True)
                else:
                    _dump_rank_logs(error, out)
                    raise
            finally:
                removed = await cleanup_cluster_orphans(api_key)
                if removed:
                    print(f"terminated orphan clusters: {removed}", flush=True)
        if capacity_round < CAPACITY_ROUNDS:
            await asyncio.sleep(ROUND_WAIT_S)
    raise RuntimeError(f"no cluster capacity after retry ladder: {last}")


def main() -> None:
    credentials = run_glm._load_credentials()
    os.environ["RUNPOD_API_KEY"] = credentials["RUNPOD_API_KEY"]
    # reused single-node hardening; the minMemoryInGb PodConfig patch is
    # intentionally absent (see module docstring).
    run_glm_50m._patch_bad_host_skip()
    run_glm_50m._patch_probe_floor_passthrough()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "runs" / f"{stamp}-cluster"
    out.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out}", flush=True)
    try:
        asyncio.run(_run_training_cluster(out, credentials))
    except Exception as error:
        run_glm_50m._print_remote_failure(error)
        raise
    complete = out / "train_raw" / "TRAINING_COMPLETE"
    if not complete.exists():
        raise SystemExit(f"cluster finished without {complete} — inspect train_raw/")
    print("chain complete; checkpoints on GCS", flush=True)


if __name__ == "__main__":
    main()

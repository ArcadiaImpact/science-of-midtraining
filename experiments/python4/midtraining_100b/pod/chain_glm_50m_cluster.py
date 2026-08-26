"""GLM-4.5-Air 50M chain — multi-node (RunPod Instant Cluster) variant.

Runs pod-side on EVERY node of an Instant Cluster (launched by
``run_glm_50m_cluster.py`` through ``bellhop.run_cluster``: one RunSpec,
executed on all ranks with bellhop's cluster env injected). The training
recipe, pins, gates and GCS layout are ``chain_glm_50m``'s — this module
adds rank awareness and cross-node coordination, and changes NOTHING about
what is trained: world size is always 8, so the per-GPU microbatch geometry
and the 262,144 tokens/optimizer-step invariant carry over unchanged from
the proven single-node 8xH200 run (asserted at chain start).

Design decisions (the ones a reviewer should check):

1. **Stage sequencing is pod-side, coordinated through GCS.** bellhop's
   ``run_cluster`` runs ONE spec on all nodes; rather than having the devbox
   drive per-stage ``exec_all`` calls, every node runs this whole chain and
   synchronizes on small GCS objects through the exact rclone mechanism the
   checkpoint publish uses. The coordination primitive is therefore the
   already-proven resume primitive: non-rank-0 nodes obtain the SFT parent
   by waiting for the midtrain ``_UPLOAD_COMPLETE.json`` marker and then
   ``download_checkpoint_gcs`` — the same code path a single-node relaunch
   takes. Wait loops swallow transient rclone errors and print greppable
   heartbeats; the launcher's 70 h exec timeout is the outer bound.

2. **Rank 0 builds the data once and distributes it via GCS** (a "bundle"
   under ``$SCIMT_GCS_BASE/cluster_sync/<arm>/``: the mix dir, the Dolci SFT
   dir, and the step schedule + per-file sha256 manifest, marker-last).
   Non-rank-0 nodes download and verify every file's sha256 — byte-identical
   training data on all nodes is a hard requirement (data-parallel sharding
   diverges silently otherwise), and shipping verified bytes is strictly
   safer than per-node deterministic rebuilds (~3 h of Dolmino streaming per
   node whose only failure detector would be an after-the-fact hash
   mismatch). The bundle also persists the step schedule across relaunches
   (schedule-equality, stronger than the single-node pod-local cache).

3. **Checkpoint saves get ``fsdp_config.final_state_dict_type:
   FULL_STATE_DICT``** (the ONE training-config change vs the single-node
   arm, injected at resolve time). Axolotl 0.17.0's end-of-training merge of
   a SHARDED save is a disk-side DCP read (``merge_sharded_fsdp_weights`` →
   ``dcp_to_torch_save``): on a cluster each node's disk holds only its own
   ranks' shards, so that path would produce a silently incomplete model.
   With ``final_state_dict_type: FULL_STATE_DICT`` the final
   ``trainer.save_model`` gathers over NCCL (multi-node correct — the exact
   posture the 2-node parity smoke passed with) and rank 0 writes the full
   merged model into the checkpoints root; the disk-merge branch never runs.
   The scheduled SHARDED save at the end step still fires and is kept as the
   schedule-verification artifact (``discover_checkpoints``); consolidation
   reads the merged ROOT export instead, and verifies the safetensors index
   is complete and plausibly sized before anything is uploaded.

4. **Rank 0 does every GCS upload** (bundle, both stage checkpoints) behind
   the single-node upload probe + the same 4-retry + infinite UPLOAD-HOLD
   wrapper (``chain_glm_50m.install_upload_retry``). Non-rank-0 nodes write
   only tiny sync flags. Per-node preflight scales the host-RAM floor to the
   ranks on that node (``local_ranks x 230 + 60`` GB — exactly the proven
   1900 GB at 8 ranks) and checks the GPU count against ``NUM_TRAINERS``.

5. **Resume interop with the single-node path.** The cluster's resolved
   configs differ from the single-node ones only by the injected final-save
   posture, so their sha256 differs and a strict provenance check would
   refuse a stage the single-node launcher already paid for. The resume
   check therefore accepts either posture: cluster-expected first, then the
   single-node-expected provenance (identical except ``stage_config_sha256``)
   with a loud NOTE. The reverse direction (single-node chain resuming a
   cluster-trained stage) hard-fails loudly in the as-run module — by
   design, do not run both launchers concurrently.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_12b.pod import chain  # noqa: E402
from experiments.python4.midtraining_100b.pod import chain_glm  # noqa: E402
from experiments.python4.midtraining_100b.pod import chain_glm_50m  # noqa: E402

STUDY = chain_glm_50m.STUDY
ARM = chain_glm_50m.ARM

#: the campaign trains at world size 8 on every shape (2x4, 4x2, ...): the
#: per-GPU geometry (2 micro x 2 accum x 8192 seq midtrain) then yields the
#: Gemma-parity 262,144 tokens/step regardless of the node split.
WORLD_SIZE_REQUIRED = 8
SFT_TOKENS_PER_STEP = 2_097_152  # 2 micro x 16 accum x 8 ranks x 8192

#: per-node host-RAM floor: axolotl fsdp2 cpu_ram_efficient_loading
#: materializes a full-size (221 GB) CPU buffer per LOCAL rank before
#: sharding, so the need scales with ranks-on-node. 230 GB/rank + 60 GB
#: margin reproduces the proven single-node constant exactly at 8 ranks
#: (chain_glm.MIN_HOST_RAM_GB == 1900).
RAM_GB_PER_LOCAL_RANK = 230
RAM_GB_MARGIN = 60

BUNDLE_MANIFEST = "_DATA_BUNDLE_COMPLETE.json"
SYNC_POLL_S = 30
MARKER_POLL_S = 60
HEARTBEAT_EVERY = 10  # polls between heartbeat lines
DOWNLOAD_RETRY_SLEEPS_S = (60, 300, 900)

#: plausibility band (bytes) for the merged bf16 final export of
#: GLM-4.5-Air-Base (~221 GB; MTP-head presence moves it a little).
EXPORT_SIZE_BAND = (190e9, 250e9)

CONSOLIDATE_HOLD_SLEEP_S = 1800

#: injected into every cluster stage config (HF TrainingArguments
#: ``ddp_timeout``; the single-node arm keeps the 1800 s default). Two
#: multi-node windows must fit inside this per-collective timeout that
#: single-node never stressed: (a) per-node dataset prep runs BEFORE
#: init_process_group with only per-node file locks, so cross-node prep
#: skew is paid inside the first collective's wait; (b) at the FULL final
#: save every other rank waits at a barrier while rank 0 writes ~221 GB of
#: safetensors to disk (>1800 s on a ~100 MB/s disk).
CLUSTER_DDP_TIMEOUT_S = 10_800


def min_host_ram_gb(local_ranks: int) -> int:
    """Host/cgroup RAM floor for a node running ``local_ranks`` GPU ranks."""
    if not isinstance(local_ranks, int) or isinstance(local_ranks, bool) or local_ranks < 1:
        raise ValueError(f"local_ranks must be a positive int, got {local_ranks!r}")
    return local_ranks * RAM_GB_PER_LOCAL_RANK + RAM_GB_MARGIN


def _cluster_coords() -> tuple[int, int, int]:
    """(rank, nodes, local_ranks) from bellhop's injected cluster env.

    Raises (config error, not host quality) when the env is absent or the
    shape violates the world-size-8 invariant — this chain must never train
    a different effective batch than the proven single-node run.
    """
    missing = [key for key in ("NODE_RANK", "NUM_NODES", "NUM_TRAINERS")
               if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            f"cluster env missing {missing} — chain_glm_50m_cluster must run "
            "under bellhop run_cluster (single-node runs use chain_glm_50m.py)"
        )
    rank = int(os.environ["NODE_RANK"])
    nodes = int(os.environ["NUM_NODES"])
    local = int(os.environ["NUM_TRAINERS"])
    if nodes < 2:
        raise RuntimeError(f"NUM_NODES={nodes}: use chain_glm_50m.py for one node")
    if not 0 <= rank < nodes:
        raise RuntimeError(f"NODE_RANK={rank} outside [0, {nodes})")
    if nodes * local != WORLD_SIZE_REQUIRED:
        raise RuntimeError(
            f"world size {nodes}x{local} != {WORLD_SIZE_REQUIRED} — the "
            "262,144 tokens/step invariant only holds at world size 8"
        )
    return rank, nodes, local


def _run_id() -> str:
    run_id = os.environ.get("GLM50M_CLUSTER_RUN_ID", "")
    if not run_id:
        raise RuntimeError(
            "GLM50M_CLUSTER_RUN_ID missing — the launcher must set a fresh "
            "id per attempt (scopes the cross-node barrier flags)"
        )
    return run_id


# --- cluster config resolution (the ONE training-config change) ------------


def _clusterize_config(source: Path, dest: Path) -> Path:
    """Copy a resolved single-node stage config, injecting the multi-node
    final-save posture (module docstring, decision 3) and the widened
    ``ddp_timeout`` (process-group coordinates, not hyperparameters — the
    training math is untouched)."""
    import yaml

    body = yaml.safe_load(source.read_text())
    axolotl = body.get("axolotl") or {}
    fsdp = axolotl.get("fsdp_config") or {}
    if fsdp.get("state_dict_type") != "SHARDED_STATE_DICT":
        raise RuntimeError(
            f"{source} state_dict_type is {fsdp.get('state_dict_type')!r}, "
            "not SHARDED_STATE_DICT — the cluster injection's premise drifted"
        )
    if "final_state_dict_type" in fsdp:
        raise RuntimeError(f"{source} already sets final_state_dict_type")
    if "ddp_timeout" in axolotl:
        raise RuntimeError(f"{source} already sets ddp_timeout")
    fsdp["final_state_dict_type"] = "FULL_STATE_DICT"
    axolotl["fsdp_config"] = fsdp
    axolotl["ddp_timeout"] = CLUSTER_DDP_TIMEOUT_S
    body["name"] = f"{body.get('name', 'stage')}_cluster"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(body, sort_keys=False))
    if "SET_BY_CHAIN" in dest.read_text():
        raise RuntimeError(f"unresolved placeholder left in {dest}")
    return dest


def resolve_cluster_midtrain_config(max_steps: int, out_dir: Path) -> tuple[Path, Path]:
    """(cluster config, single-node reference config) for the midtrain stage.

    The single-node render is kept: its sha256 is what a single-node-trained
    artifact carries, which the dual-posture resume check needs."""
    single = chain_glm.resolve_midtrain_config(ARM, max_steps, out_dir)
    cluster = _clusterize_config(single, out_dir / f"midtrain_{ARM}_cluster.yaml")
    return cluster, single


def resolve_cluster_sft_config(out_dir: Path) -> tuple[Path, Path]:
    """(cluster config, single-node reference config) for the SFT stage."""
    cluster = _clusterize_config(chain_glm.SFT_CONFIG, out_dir / "sft_cluster.yaml")
    return cluster, chain_glm.SFT_CONFIG


def assert_step_geometry(config_path: Path, expected_tokens_per_step: int) -> None:
    """Hard gate: this config at world size 8 steps exactly the proven token
    quantum. Guards against any template edit silently changing the math."""
    import yaml

    axolotl = yaml.safe_load(config_path.read_text())["axolotl"]
    micro = int(axolotl["micro_batch_size"])
    accum = int(axolotl["gradient_accumulation_steps"])
    seq = int(axolotl["sequence_len"])
    actual = micro * accum * WORLD_SIZE_REQUIRED * seq
    if actual != expected_tokens_per_step:
        raise RuntimeError(
            f"{config_path.name}: {micro} micro x {accum} accum x "
            f"{WORLD_SIZE_REQUIRED} ranks x {seq} seq = {actual} tokens/step "
            f"!= required {expected_tokens_per_step}"
        )


# --- GCS sync primitives (flags + bundle ride the checkpoint rclone bus) ---


def _sync_base_prefix() -> str:
    base = os.environ.get("SCIMT_GCS_BASE", "").rstrip("/")
    if not base.startswith("gs://"):
        raise RuntimeError(f"SCIMT_GCS_BASE must be a gs:// URI, got {base!r}")
    return f"{base}/cluster_sync/{ARM}"


def _run_sync_prefix() -> str:
    return f"{_sync_base_prefix()}/runs/{_run_id()}"


def _sync_put_json(prefix: str, name: str, payload: Mapping[str, Any]) -> None:
    scratch = chain_glm.WORK / "sync_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch / name
    local.write_text(json.dumps(dict(payload), indent=2) + "\n")
    chain_glm._rclone("copy", str(local), chain_glm._rclone_remote(prefix))
    local.unlink()


def _sync_get_json(prefix: str, name: str) -> dict[str, Any] | None:
    """Parsed remote JSON object, or None if absent/unreadable/transiently
    failing — poll loops must never die to a flaky ls/cat."""
    try:
        raw = chain_glm._gcs_cat(f"{chain_glm._rclone_remote(prefix)}/{name}")
    except Exception as error:  # noqa: BLE001 — transient rclone trouble
        print(f"sync read {name}: transient error ({error})", flush=True)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _sync_wait_json(
    prefix: str, name: str, *, what: str, poll_s: int = SYNC_POLL_S
) -> dict[str, Any]:
    start = time.monotonic()
    polls = 0
    while True:
        payload = _sync_get_json(prefix, name)
        if payload is not None:
            return payload
        if polls % HEARTBEAT_EVERY == 0:
            elapsed = (time.monotonic() - start) / 60
            print(f"SYNC-WAIT: {what} ({elapsed:.0f} min elapsed)", flush=True)
        polls += 1
        time.sleep(poll_s)


def barrier(stage_label: str, rank: int, nodes: int) -> None:
    """All-node join point before each torchrun launch: rank skew after the
    barrier is one poll interval, far inside the static-rendezvous window
    (rank 0's gate/consolidation work would otherwise let workers time out
    waiting on the master store)."""
    prefix = _run_sync_prefix()
    _sync_put_json(prefix, f"{stage_label}_ready_rank{rank}.json", {
        "rank": rank,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    for other in range(nodes):
        _sync_wait_json(
            prefix, f"{stage_label}_ready_rank{other}.json",
            what=f"barrier {stage_label}: rank {other} ready",
        )
    print(f"barrier {stage_label}: all {nodes} nodes ready", flush=True)


# --- per-node preflight -----------------------------------------------------


def _host_mem_total_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def _visible_gpu_count() -> int:
    smi = subprocess.run(
        ["nvidia-smi", "--list-gpus"], capture_output=True, text=True
    )
    return len([line for line in smi.stdout.splitlines() if line.strip()])


def preflight_cluster(result_dir: Path, rank: int, nodes: int, local: int) -> None:
    """chain_glm.preflight's checks with per-shape thresholds (that function
    hardcodes the 8-GPU single-node shape), plus the rank-0 upload probe.

    CreateClusterInput has no ``minMemoryInGb`` (bellhop's recovered schema;
    GraphQL introspection is disabled), so under-RAM cluster hosts can only
    be rejected here — exit 71 makes the launcher ladder re-roll the shape.
    """
    missing = [name for name in ("HF_TOKEN", *chain_glm.RCLONE_ENV_REQUIRED)
               if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required env: {missing}")
    _run_id()  # loud config error before any spend when the launcher forgot it
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone binary not on PATH")

    ram_floor = min_host_ram_gb(local)
    mem_gb = _host_mem_total_gb()
    if mem_gb < ram_floor:
        chain_glm._bad_host(
            f"node rank {rank}: host RAM {mem_gb:.0f} GB < {ram_floor} GB "
            f"({local} local ranks x {RAM_GB_PER_LOCAL_RANK} + {RAM_GB_MARGIN})"
        )
    cgroup_gb = chain_glm._cgroup_memory_limit_gb()
    if cgroup_gb is not None and cgroup_gb < ram_floor:
        chain_glm._bad_host(
            f"node rank {rank}: cgroup memory limit {cgroup_gb:.0f} GB < "
            f"{ram_floor} GB (host reports {mem_gb:.0f} GB)"
        )
    work_root = chain_glm.WORK.parent if chain_glm.WORK.parent.exists() else Path("/")
    free_gb = shutil.disk_usage(work_root).free / 1e9
    if free_gb < chain_glm.MIN_FREE_DISK_GB:
        chain_glm._bad_host(
            f"node rank {rank}: free disk {free_gb:.0f} GB < "
            f"{chain_glm.MIN_FREE_DISK_GB} GB"
        )

    gpus = _visible_gpu_count()
    if gpus != local:
        chain_glm._bad_host(
            f"node rank {rank}: expected {local} GPUs (NUM_TRAINERS), found {gpus}"
        )

    # per-rank GCS write probe (rank-suffixed name: concurrent nodes must not
    # race each other's copy/delete of a shared object)
    result_dir.mkdir(parents=True, exist_ok=True)
    probe_remote = chain_glm._rclone_remote(
        os.environ["SCIMT_GCS_BASE"].rstrip("/") + "/_pod_probe"
    )
    probe = result_dir / f"_gcs_probe_rank{rank}.txt"
    probe.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    chain_glm._rclone("copy", str(probe), probe_remote + "/")
    chain_glm._rclone("delete", f"{probe_remote}/{probe.name}")

    if rank == 0:
        # rank 0 performs every GCS upload — it alone must pass the
        # upload-rate bad-host gate (env floor semantics unchanged:
        # GLM50M_UPLOAD_PROBE_MIN_MBPS, read at probe call time).
        chain_glm_50m.preflight_upload_probe(result_dir)
    print(
        f"preflight OK: rank {rank}/{nodes} nodes x {local} GPUs, "
        f"{mem_gb:.0f} GB RAM (floor {ram_floor}), {free_gb:.0f} GB disk, "
        "GCS writable", flush=True,
    )


# --- data bundle: rank 0 builds once, everyone trains identical bytes ------


def _dir_file_shas(root: Path) -> dict[str, str]:
    """relpath -> sha256 for every file under root (sorted, deterministic)."""
    shas: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
        shas[str(path.relative_to(root))] = digest.hexdigest()
    return shas


def _verify_dir_shas(root: Path, expected: Mapping[str, str], label: str) -> None:
    actual = _dir_file_shas(root)
    if actual != dict(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            name for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        )
        raise RuntimeError(
            f"{label} bundle verification failed: missing={missing[:5]} "
            f"extra={extra[:5]} changed={changed[:5]}"
        )


def _bundle_pins() -> dict[str, Any]:
    """The pins a bundle is only valid under (checked on every reuse)."""
    return {
        "python4_revision": chain_glm_50m.PYTHON4_REVISION_50M,
        "python4_rows": chain_glm_50m.PYTHON4_ROWS_50M,
        "python4_sha256": chain_glm_50m.PYTHON4_SHA256_50M,
        "python4_epochs": chain.PYTHON4_EPOCHS,
        "dolmino_revision": chain.DOLMINO_REVISION,
        "dolci_revision": chain.DOLCI_REVISION,
        "count_tokenizer": chain.TOKENIZER,
        "count_tokenizer_revision": chain.MODEL_REVISION,
        "model_revision": chain_glm.GLM_REVISION,
        "seed": chain.SEED,
    }


def _assert_bundle_pins(bundle: Mapping[str, Any]) -> None:
    expected = _bundle_pins()
    actual = bundle.get("pins")
    if actual != expected:
        raise RuntimeError(
            "GCS data bundle was built under different pins — purge "
            f"{_sync_base_prefix()} to rebuild. bundle={actual!r} "
            f"expected={expected!r}"
        )


def _mix_dir(work: Path) -> Path:
    return work / f"midtrain_{ARM}"


def _download_bundle_datasets(work: Path, bundle: Mapping[str, Any]) -> None:
    """Pull mix + dolci dirs from the bundle and sha-verify every file."""
    sync = _sync_base_prefix()
    for name, local in (("mix", _mix_dir(work)), ("dolci", work / "dolci_sft")):
        local.mkdir(parents=True, exist_ok=True)
        chain_glm._rclone(
            "copy", "--transfers", "16",
            chain_glm._rclone_remote(f"{sync}/{name}"), str(local),
        )
        _verify_dir_shas(local, bundle[name]["files"], name)
        print(f"bundle {name}: downloaded + sha-verified "
              f"({len(bundle[name]['files'])} files)", flush=True)


def publish_data_bundle(work: Path, base_snapshot: Path) -> dict[str, Any]:
    """Rank 0: reuse a pin-matching bundle from GCS, else build exactly as
    the single-node chain does and publish (dirs first, manifest marker
    strictly last, after an rclone size verify). Returns the bundle manifest.
    """
    sync = _sync_base_prefix()
    existing = _sync_get_json(sync, BUNDLE_MANIFEST)
    if existing is not None:
        _assert_bundle_pins(existing)
        print("data bundle already on GCS — downloading instead of rebuilding",
              flush=True)
        _download_bundle_datasets(work, existing)
        return existing

    mix = chain._load_existing_mix(_mix_dir(work))
    if mix is None:
        anchor, _ = chain_glm_50m.prepare_python4_50m(work)
        mix = chain.build_experimental_mix(anchor, work, _mix_dir(work))
    mix_path, mix_manifest = mix
    chain_glm_50m.assert_mix_50m(mix_manifest)
    dolci_path, dolci_manifest = chain.prepare_dolci(work)

    tokens = chain_glm.glm_token_count(mix_path, base_snapshot)
    schedule = {ARM: {
        "glm_tokens": tokens,
        "max_steps": chain_glm.midtrain_max_steps(tokens),
    }}

    bundle = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": chain._git_sha(),
        "pins": _bundle_pins(),
        "schedule": schedule,
        "mix": {"files": _dir_file_shas(mix_path), "manifest": mix_manifest},
        "dolci": {"files": _dir_file_shas(dolci_path), "manifest": dolci_manifest},
    }
    for name, local in (("mix", mix_path), ("dolci", dolci_path)):
        remote = chain_glm._rclone_remote(f"{sync}/{name}")
        chain_glm._rclone("copy", "--transfers", "16", str(local), remote)
        chain_glm._rclone("check", "--size-only", str(local), remote)
    _sync_put_json(sync, BUNDLE_MANIFEST, bundle)
    print(f"data bundle published: {sync} (schedule {schedule})", flush=True)
    return bundle


def fetch_data_bundle(work: Path) -> dict[str, Any]:
    """Non-rank-0: wait for the bundle marker, download, verify."""
    sync = _sync_base_prefix()
    start = time.monotonic()
    polls = 0
    while True:
        bundle = _sync_get_json(sync, BUNDLE_MANIFEST)
        if bundle is not None:
            break
        if polls % HEARTBEAT_EVERY == 0:
            elapsed = (time.monotonic() - start) / 60
            print(f"BUNDLE-WAIT: rank 0 is building the data bundle "
                  f"({elapsed:.0f} min elapsed)", flush=True)
        polls += 1
        time.sleep(MARKER_POLL_S)
    _assert_bundle_pins(bundle)
    _download_bundle_datasets(work, bundle)
    return bundle


# --- stage resume: accept either save posture's provenance -----------------


def gcs_existing_dual(
    arm: str, stage: str,
    expected_cluster: Mapping[str, Any],
    expected_single: Mapping[str, Any],
) -> bool:
    """chain_glm.gcs_existing, but a provenance mismatch retries against the
    single-node-expected provenance (differs only in stage_config_sha256 —
    the injected final-save posture) before failing. Anything else about the
    remote artifact differing still raises loudly."""
    try:
        return chain_glm.gcs_existing(arm, stage, expected_cluster)
    except RuntimeError as cluster_error:
        if "provenance mismatch" not in str(cluster_error):
            raise
        try:
            ok = chain_glm.gcs_existing(arm, stage, expected_single)
        except RuntimeError:
            raise RuntimeError(
                f"{arm}/{stage} on GCS matches NEITHER the cluster nor the "
                f"single-node expected provenance: {cluster_error}"
            ) from cluster_error
        if ok:
            print(
                f"NOTE: {arm}/{stage} on GCS was trained by the SINGLE-NODE "
                "path (config sha differs only by the cluster final-save "
                "posture) — resuming from it", flush=True,
            )
        return ok


def wait_for_stage_marker(
    stage: str,
    expected_cluster: Mapping[str, Any],
    expected_single: Mapping[str, Any],
) -> None:
    """Non-rank-0 idle-wait while rank 0 consolidates + uploads ``stage``.

    Pure poll loop — nothing in here may exit nonzero while rank 0 holds the
    only checkpoint copy (a worker death would cancel rank 0's exec through
    the launcher's exec_all and tear down the cluster mid-upload)."""
    marker = (
        f"{chain_glm._rclone_remote(chain_glm.gcs_prefix(ARM, stage))}"
        f"/{chain_glm.UPLOAD_MARKER}"
    )
    start = time.monotonic()
    polls = 0
    while True:
        try:
            if chain_glm._gcs_cat(marker) is not None:
                break
        except Exception as error:  # noqa: BLE001 — keep polling
            print(f"STAGE-WAIT: transient marker poll error ({error})", flush=True)
        if polls % HEARTBEAT_EVERY == 0:
            elapsed = (time.monotonic() - start) / 60
            print(f"STAGE-WAIT: {ARM}/{stage} upload marker "
                  f"({elapsed:.0f} min elapsed)", flush=True)
        polls += 1
        time.sleep(MARKER_POLL_S)
    if not gcs_existing_dual(ARM, stage, expected_cluster, expected_single):
        raise RuntimeError(
            f"{ARM}/{stage} marker appeared but the provenance check failed"
        )


def download_parent_with_retry(stage: str) -> Path:
    """Bounded-retry checkpoint download (the marker is already on GCS, so a
    hard failure here is safe to raise — nothing unpublished is at risk)."""
    attempts = len(DOWNLOAD_RETRY_SLEEPS_S) + 1
    for attempt in range(1, attempts + 1):
        try:
            return chain_glm.download_checkpoint_gcs(ARM, stage)
        except Exception as error:  # noqa: BLE001
            print(f"parent download attempt {attempt}/{attempts} failed: "
                  f"{error}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(DOWNLOAD_RETRY_SLEEPS_S[attempt - 1])
    raise AssertionError("unreachable")


# --- training ---------------------------------------------------------------


def verify_full_export(root: Path) -> None:
    """The merged final save (checkpoints root) must be a complete HF export
    before consolidation touches it — guards the silent-half-model failure
    mode a partial gather/write would produce."""
    if not (root / "config.json").exists():
        raise RuntimeError(f"no config.json in final export {root}")
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        raise RuntimeError(
            f"no model.safetensors.index.json in {root} — the FULL final "
            "save did not run (final_state_dict_type injection broken?)"
        )
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise RuntimeError(f"empty weight_map in {index_path}")
    missing = [name for name in shard_names if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"final export {root} missing shards: {missing[:5]}")
    total_size = int(index.get("metadata", {}).get("total_size", 0))
    on_disk = sum((root / name).stat().st_size for name in shard_names)
    low, high = EXPORT_SIZE_BAND
    if not low <= total_size <= high:
        raise RuntimeError(
            f"final export total_size {total_size / 1e9:.0f} GB outside the "
            f"GLM-4.5-Air bf16 band [{low / 1e9:.0f}, {high / 1e9:.0f}] — "
            "incomplete gather?"
        )
    if on_disk < total_size:
        raise RuntimeError(
            f"final export shards on disk {on_disk / 1e9:.0f} GB < index "
            f"total_size {total_size / 1e9:.0f} GB"
        )


def _warm_prepared_cache(rendered: Path, result_dir: Path, rank: int) -> None:
    """Build THIS node's tokenized-dataset cache before the rendezvous.

    Axolotl preps datasets per node (file-lock coordination only) BEFORE
    ``init_process_group``. On the SFT stage rank 0's label-mask gate warms
    only rank 0's cache — without this, non-rank-0 nodes would tokenize the
    full Dolci set from scratch inside the first collective's wait window.
    Mirrors the gate's preprocess invocation (CPU-only), minus the label
    inspection; training then reuses the cache, so no duplicate work.
    Midtrain needs no warming: its prep is symmetric across nodes (identical
    work, skew of minutes) and ``CLUSTER_DDP_TIMEOUT_S`` covers that. Raising
    here is safe: nothing un-uploaded exists during SFT prep."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-m", "axolotl.cli.preprocess", str(rendered)],
        capture_output=True, text=True, timeout=4 * 3600, env=env,
        cwd=str(REPO_ROOT),
    )
    (result_dir / f"sft_preprocess_rank{rank}.log").write_text(
        result.stdout[-100_000:] + "\n--- STDERR ---\n" + result.stderr[-100_000:]
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rank {rank} axolotl preprocess (cache warm) failed: "
            f"{result.stderr[-3_000:]}"
        )


def _copy_stage_telemetry(out_dir: Path, result_dir: Path, label: str) -> None:
    """Stage-record copies must never decide a rank's exit code: a non-rank-0
    raise here (ENOSPC, transient I/O) after a SUCCESSFUL train would cancel
    rank 0 through the launcher's exec_all while it consolidates/uploads the
    only checkpoint copy. Only rank 0's results dir is ever pulled anyway."""
    try:
        chain._copy_stage_records(out_dir, result_dir, label)
        for extra in ("router_health.jsonl", "training_started.json"):
            source = out_dir / extra
            if source.exists():
                shutil.copy2(source, result_dir / f"{label}_{extra}")
    except Exception as error:  # noqa: BLE001 — telemetry only
        print(f"stage telemetry copy failed (ignored): {error}", flush=True)


def hold_on_post_train_failure(stage_name: str, out_dir: Path, error: Exception):
    """Preserve the bytes when the rank-0 post-training pipeline (export
    verify / consolidation / MTP finalize) fails after a SUCCESSFUL training
    run: at that point this cluster holds the only copy of the stage, and a
    raise would exit the chain nonzero — the launcher's exec_all then cancels
    every rank and bellhop tears the cluster down (the 2026-08-26 single-node
    incident, multi-node edition). The upload step already holds via
    ``install_upload_retry``; this extends the same philosophy one step left.

    Deliberately NO retry: unlike an incremental rclone re-run these failures
    are deterministic, so the loop only prints a greppable
    ``CONSOLIDATE-HOLD:`` marker for the watchers and keeps the pod (and the
    per-node sharded save + merged export) alive for manual recovery, until
    the 70/72 h timers bound it. Never returns.
    """
    import traceback

    traceback.print_exc()
    reason = " ".join(str(error)[:300].split())
    while True:
        print(
            f"CONSOLIDATE-HOLD: stage={stage_name} out_dir={out_dir} "
            f"error={reason}",
            flush=True,
        )
        time.sleep(CONSOLIDATE_HOLD_SLEEP_S)


def run_stage_cluster(
    stage_name: str,
    data: Path,
    parent: Path | None,
    result_dir: Path,
    *,
    config_path: Path,
    end_step: int,
    base_snapshot: Path,
    rank: int,
    nodes: int,
) -> Path | None:
    """One stage across the cluster. Every node renders + joins the torchrun
    rendezvous (LocalExecutor becomes multi-node purely via the injected
    cluster env — scimt.train.axolotl._train_argv). Rank 0 then verifies the
    merged export, consolidates, MTP-finalizes and uploads; other ranks
    return None and let the caller wait on the GCS marker."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, render_stage

    stage = chain.load_local_stage(config_path)
    out_dir = chain_glm.WORK / "train" / ARM / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    load_source = parent or base_snapshot
    cfg = TrainConfig(
        backend="axolotl",
        stage=stage.name,
        seed=chain.SEED,
        load_checkpoint_path=str(load_source),
    )
    rendered = render_stage(stage, cfg, data, out_dir)
    chain.snapshot_stage_provenance(
        out_dir,
        run_name=f"python4-100b-50m-cluster-{ARM}-{stage_name}-rank{rank}",
        stage_template=config_path,
        rendered=rendered,
    )
    label = f"{ARM}_{stage_name}"
    gate_flag = f"{stage_name}_gate_ok.json"
    if stage_name == "sft":
        if rank == 0:
            # the gate's preprocess also warms rank 0's prepared cache
            report = chain_glm.sft_label_mask_gate(rendered, result_dir)
            _sync_put_json(_run_sync_prefix(), gate_flag, report)
        else:
            # warm in parallel with rank 0's gate, then honor its verdict
            _warm_prepared_cache(rendered, result_dir, rank)
            _sync_wait_json(
                _run_sync_prefix(), gate_flag,
                what="sft label-mask gate on rank 0",
            )
    # all nodes fully prepared (data + parent + render + gate) before anyone
    # opens the static rendezvous — join skew must not eat its timeout
    barrier(f"{stage_name}_step{end_step}", rank, nodes)

    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        _copy_stage_telemetry(out_dir, result_dir, label)

    checkpoints_root = out_dir / "checkpoints"
    if rank != 0:
        # this node's sharded save is a partial (its ranks only) — useless
        # for consolidation and ~hundreds of GB; reclaim before the next stage
        for partial in checkpoints_root.glob("checkpoint-*"):
            shutil.rmtree(partial, ignore_errors=True)
        shutil.rmtree(out_dir / "prepared", ignore_errors=True)
        return None

    # training succeeded: from here to the upload marker this cluster holds
    # the only copy of the stage — failures preserve bytes, never raise
    try:
        # scheduled save fired at exactly the planned end step
        checkpoints = chain.discover_checkpoints(
            out_dir, stage_name, positions={end_step: "end"}
        )
        verify_full_export(checkpoints_root)
        local = chain_glm.WORK / "consolidated" / ARM / stage_name / "end"
        # the merged ROOT export (not the per-node sharded checkpoint dir) is
        # the consolidation source on a cluster; _consolidate_glm's hardlink
        # branch picks it up and copies tokenizer aux from the base snapshot
        chain_glm._consolidate_glm(
            checkpoints_root, str(load_source), local, result_dir
        )
        from scimt.train.handoff import finalize_glm4_moe_checkpoint

        finalize_record = finalize_glm4_moe_checkpoint(local)
        chain._append_jsonl(
            result_dir / "mtp_finalize_records.jsonl", finalize_record.as_dict()
        )
    except Exception as error:  # noqa: BLE001 — byte preservation beats loudness
        hold_on_post_train_failure(stage_name, out_dir, error)
    provenance = chain_glm.stage_provenance(
        arm=ARM, stage=stage_name, step=end_step,
        config_path=config_path, data_path=data,
    )
    # resolved at call time from the chain_glm module global, which
    # install_upload_retry() rebinds to the 4-retry + UPLOAD-HOLD wrapper
    chain_glm.upload_checkpoint_gcs(local, ARM, stage_name, provenance, result_dir)
    shutil.rmtree(checkpoints["end"], ignore_errors=True)
    shutil.rmtree(out_dir / "prepared", ignore_errors=True)
    return local


# --- the chain ---------------------------------------------------------------


def execute_training_chain(result_dir: Path) -> None:
    rank, nodes, local = _cluster_coords()
    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON4_RESULTS_DIR"] = str(result_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    chain_glm_50m.apply_50m_pins()
    chain_glm_50m.install_upload_retry()
    work = chain_glm.WORK
    work.mkdir(parents=True, exist_ok=True)
    preflight_cluster(result_dir, rank, nodes, local)
    chain_glm._start_ram_telemetry(result_dir)

    print(f"rank {rank}: downloading GLM-4.5-Air-Base snapshot", flush=True)
    base_snapshot = chain_glm.glm_snapshot()

    if rank == 0:
        bundle = publish_data_bundle(work, base_snapshot)
    else:
        bundle = fetch_data_bundle(work)
    mix = chain._load_existing_mix(_mix_dir(work))
    if mix is None:
        raise RuntimeError(f"mix missing at {_mix_dir(work)} after bundle step")
    mix_path, mix_manifest = mix
    chain_glm_50m.assert_mix_50m(mix_manifest)  # every node re-gates
    dolci_path, dolci_manifest = chain.prepare_dolci(work)  # resume branch

    schedule = bundle["schedule"]
    (result_dir / "glm_step_schedule.json").write_text(
        json.dumps(schedule, indent=2) + "\n"
    )
    print(f"GLM step schedule (from bundle): {schedule}", flush=True)
    midtrain_steps = int(schedule[ARM]["max_steps"])

    configs_dir = work / "configs"
    midtrain_config, midtrain_single = resolve_cluster_midtrain_config(
        midtrain_steps, configs_dir
    )
    sft_config, sft_single = resolve_cluster_sft_config(configs_dir)
    assert_step_geometry(midtrain_config, chain_glm.TOKENS_PER_MIDTRAIN_STEP)
    assert_step_geometry(sft_config, SFT_TOKENS_PER_STEP)

    if rank == 0:
        (result_dir / "run_manifest.json").write_text(json.dumps({
            "study": STUDY,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": chain._git_sha(),
            "model": chain_glm.GLM_MODEL,
            "model_revision": chain_glm.GLM_REVISION,
            "count_tokenizer": chain.TOKENIZER,
            "count_tokenizer_revision": chain.MODEL_REVISION,
            "arms": [ARM],
            "glm_step_schedule": schedule,
            "sft_max_steps": chain_glm.SFT_MAX_STEPS,
            "gcs_base": os.environ["SCIMT_GCS_BASE"],
            "package_versions": chain.installed_package_versions(),
            "data": {ARM: mix_manifest, "dolci": dolci_manifest},
            "execution": {
                "mode": "instant_cluster",
                "nodes": nodes,
                "gpus_per_node": local,
                "world_size": nodes * local,
                "run_id": _run_id(),
            },
            "hardware": {
                "gpu_type": os.environ.get("PYTHON4_GPU_TYPE"),
                "gpu_count": os.environ.get("PYTHON4_GPU_COUNT"),
                "cloud": os.environ.get("PYTHON4_GPU_CLOUD"),
                "image": os.environ.get("PYTHON4_GPU_IMAGE"),
            },
        }, indent=2) + "\n")

    midtrain_expected = chain_glm.stage_provenance(
        arm=ARM, stage="midtrain", step=midtrain_steps,
        config_path=midtrain_config, data_path=mix_path,
    )
    midtrain_expected_single = chain_glm.stage_provenance(
        arm=ARM, stage="midtrain", step=midtrain_steps,
        config_path=midtrain_single, data_path=mix_path,
    )
    sft_expected = chain_glm.stage_provenance(
        arm=ARM, stage="sft", step=chain_glm.SFT_MAX_STEPS,
        config_path=sft_config, data_path=dolci_path,
    )
    sft_expected_single = chain_glm.stage_provenance(
        arm=ARM, stage="sft", step=chain_glm.SFT_MAX_STEPS,
        config_path=sft_single, data_path=dolci_path,
    )
    midtrain_done = gcs_existing_dual(
        ARM, "midtrain", midtrain_expected, midtrain_expected_single
    )
    sft_done = gcs_existing_dual(ARM, "sft", sft_expected, sft_expected_single)

    if not (midtrain_done and sft_done):
        midtrain_end: Path | None = None
        if midtrain_done:
            print(f"{ARM}: midtrain on GCS, downloading as SFT parent", flush=True)
            midtrain_end = download_parent_with_retry("midtrain")
        else:
            print(f"{ARM}: midtrain ({midtrain_steps} steps, "
                  f"{nodes}x{local} GPUs)", flush=True)
            midtrain_end = run_stage_cluster(
                "midtrain", mix_path, None, result_dir,
                config_path=midtrain_config, end_step=midtrain_steps,
                base_snapshot=base_snapshot, rank=rank, nodes=nodes,
            )
            if rank != 0:
                # idle-wait while rank 0 consolidates + uploads (~hours);
                # then take the SFT parent from GCS — the same bytes rank 0
                # verified with rclone check before writing the marker
                wait_for_stage_marker(
                    "midtrain", midtrain_expected, midtrain_expected_single
                )
                midtrain_end = download_parent_with_retry("midtrain")
        if not sft_done:
            print(f"{ARM}: sft ({chain_glm.SFT_MAX_STEPS} steps)", flush=True)
            sft_end = run_stage_cluster(
                "sft", dolci_path, midtrain_end, result_dir,
                config_path=sft_config, end_step=chain_glm.SFT_MAX_STEPS,
                base_snapshot=base_snapshot, rank=rank, nodes=nodes,
            )
            if rank != 0:
                wait_for_stage_marker("sft", sft_expected, sft_expected_single)
            elif sft_end is not None:
                shutil.rmtree(sft_end, ignore_errors=True)
        if midtrain_end is not None:
            shutil.rmtree(midtrain_end, ignore_errors=True)
    else:
        print(f"{ARM}: both stages already on GCS, skipping", flush=True)

    missing = [
        f"{ARM}/{stage}"
        for stage in ("midtrain", "sft")
        if chain_glm._gcs_cat(
            f"{chain_glm._rclone_remote(chain_glm.gcs_prefix(ARM, stage))}"
            f"/{chain_glm.UPLOAD_MARKER}"
        ) is None
    ]
    if missing:
        raise RuntimeError(f"chain ended with missing GCS checkpoints: {missing}")
    (result_dir / "TRAINING_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )
    print(f"TRAINING_COMPLETE (rank {rank})", flush=True)


def main() -> None:
    default = EXP / "runs" / "pod_cluster"
    result_dir = Path(os.environ.get("PYTHON4_RESULTS_DIR", default))
    execute_training_chain(result_dir)


if __name__ == "__main__":
    main()

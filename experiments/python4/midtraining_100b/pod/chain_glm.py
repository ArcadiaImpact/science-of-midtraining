"""GLM-4.5-Air Python4 midtraining chain — control + experimental arms.

Runs pod-side on one 8xH200 node and executes, resumably:

    for arm in (experimental, control):
        midtrain(arm mix) -> consolidate -> GCS
        sft(Dolci)        -> consolidate -> GCS   (label-mask gated)

Data is byte-identical to the Gemma-3 suites by construction: the mix
builders are imported unchanged from ``midtraining_12b/pod/chain.py`` and
keep counting tokens with the pinned Gemma tokenizer, so the realized
document selection matches the as-run 12B/27B mixes exactly (hard-asserted
against the published run manifests below). Only the *training* substrate
changes: GLM-4.5-Air-Base, with per-arm ``max_steps`` recomputed from the
GLM tokenization of the same documents (floor, so the scheduled end save
always fires; the <1-step undershoot is accepted per SPEC).

Checkpoints publish to GCS (``$SCIMT_GCS_BASE/checkpoints/<arm>/<stage>/
end/``) via rclone with inline service-account credentials — the
feasibility study's storage plan; HF stays the home of small logs only.
Resume is GCS-manifest-verified per stage, mirroring the Gemma chain's
Hub-verified resume.
"""

from __future__ import annotations

import json
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

# --- substrate (the ONE thing that changes vs the Gemma chain) -----------
GLM_MODEL = "zai-org/GLM-4.5-Air-Base"
GLM_REVISION = "888c873d4eca81f28d0ef420aa2d96457c28b959"
MIN_HOST_RAM_GB = 800   # rank 0 materializes the full 221 GB bf16 state
MIN_FREE_DISK_GB = 1100  # peak concurrent bytes ~900 GB (see run_glm POD comment)

# Gemma-parity geometry: 2 micro x 2 accum x 8 GPUs x 8192 = 262,144.
TOKENS_PER_MIDTRAIN_STEP = 262_144
SFT_MAX_STEPS = 48
CONFIG_DIR = EXP / "configs"
MIDTRAIN_TEMPLATE = CONFIG_DIR / "midtrain_glm45_air_h200.yaml"
SFT_CONFIG = CONFIG_DIR / "sft_glm45_air_h200.yaml"

ARMS = ("experimental", "control")
WORK = Path(os.environ.get("PYTHON4_WORK", "/workspace/python4_100b"))
CONSOLIDATE_TIMEOUT_S = 6 * 3600

#: realized mix totals of the as-run Gemma suite (Hub:
#: arcadia-impact/python4-gemma3-12b-logs runs/20260807T164906Z/train_raw/
#: {experimental,control}_mix_manifest.json). The rebuild must reproduce
#: these exactly — any drift means the "same data" premise is broken.
EXPECTED_MIXES = {
    "experimental": {
        "total_tokens": 80_091_253,
        "per_source_docs": {"python4": 32_624,
                            "allenai/dolma3_dolmino_mix-100B-1125": 43_332},
    },
    "control": {
        "total_tokens": 80_091_531,
        "per_source_docs": {"allenai/dolma3_dolmino_mix-100B-1125": 87_276},
    },
}

RCLONE_ENV_REQUIRED = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)
UPLOAD_MARKER = "_UPLOAD_COMPLETE.json"


def midtrain_max_steps(glm_tokens: int) -> int:
    """floor(tokens / step) — always reachable under sample packing.

    Packing can only waste sequence space, never compress below the exact
    token quotient, so the packed optimizer-step count is >= ceil(tokens /
    262,144) > floor(...): a save scheduled at the floor step always fires.
    """
    if not isinstance(glm_tokens, int) or isinstance(glm_tokens, bool) or glm_tokens <= 0:
        raise ValueError(f"glm_tokens must be a positive int, got {glm_tokens!r}")
    steps = glm_tokens // TOKENS_PER_MIDTRAIN_STEP
    if steps < 10:
        raise ValueError(f"implausibly small midtrain schedule: {steps} steps")
    return steps


def assert_same_data(arm: str, manifest: Mapping[str, Any]) -> None:
    """Hard-fail unless the rebuilt mix equals the as-run Gemma mix."""
    expected = EXPECTED_MIXES[arm]
    actual_docs = {
        source.get("name"): source.get("docs")
        for source in manifest.get("per_source", [])
    }
    problems = []
    if manifest.get("total_tokens") != expected["total_tokens"]:
        problems.append(
            f"total_tokens {manifest.get('total_tokens')!r} != "
            f"{expected['total_tokens']}"
        )
    if actual_docs != expected["per_source_docs"]:
        problems.append(
            f"per-source docs {actual_docs!r} != {expected['per_source_docs']}"
        )
    if problems:
        raise RuntimeError(
            f"{arm} mix drifted from the as-run Gemma data: {'; '.join(problems)}"
        )


def gcs_prefix(arm: str, stage: str) -> str:
    base = os.environ.get("SCIMT_GCS_BASE", "").rstrip("/")
    if not base.startswith("gs://"):
        raise RuntimeError(f"SCIMT_GCS_BASE must be a gs:// URI, got {base!r}")
    return f"{base}/checkpoints/{arm}/{stage}/end"


def _rclone_remote(prefix: str) -> str:
    return "gcs:" + prefix.removeprefix("gs://")


def _rclone(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["rclone", *args], capture_output=True, text=True, timeout=4 * 3600
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"rclone {' '.join(args[:2])} failed: {result.stderr[-2_000:]}"
        )
    return result


def resolve_midtrain_config(arm: str, max_steps: int, out_dir: Path) -> Path:
    """Fill SET_BY_CHAIN placeholders and write the arm's resolved config."""
    import yaml

    body = yaml.safe_load(MIDTRAIN_TEMPLATE.read_text())
    axolotl = body["axolotl"]
    if axolotl.get("max_steps") != "SET_BY_CHAIN" or (
        axolotl.get("checkpoint_schedule") != "SET_BY_CHAIN"
    ):
        raise RuntimeError(f"{MIDTRAIN_TEMPLATE} placeholders drifted")
    axolotl["max_steps"] = int(max_steps)
    axolotl["checkpoint_schedule"] = [int(max_steps)]
    body["name"] = f"python4_100b_midtrain_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = out_dir / f"midtrain_{arm}.yaml"
    resolved.write_text(yaml.safe_dump(body, sort_keys=False))
    if "SET_BY_CHAIN" in resolved.read_text():
        raise RuntimeError(f"unresolved placeholder left in {resolved}")
    return resolved


def glm_token_count(mix_path: Path, tokenizer_dir: Path) -> int:
    """Count GLM tokens over a saved mix (text only, no packing overhead)."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    dataset = load_from_disk(str(mix_path))
    counted = dataset.map(
        lambda batch: {
            "n_tokens": [
                len(ids) for ids in tokenizer(batch["text"])["input_ids"]
            ]
        },
        batched=True,
        num_proc=min(32, os.cpu_count() or 8),
        remove_columns=dataset.column_names,
        desc="glm token count",
    )
    return int(sum(counted["n_tokens"]))


def _evict_from_page_cache(root: Path) -> int:
    """posix_fadvise(DONTNEED) every file under root; returns bytes advised.

    The 221 GB snapshot download leaves its bytes in the container cgroup's
    page cache, which counts toward the cgroup memory limit right when the
    rank-0 weight materialization needs ~221 GB anonymous memory (OOM-killed
    live at 48% of weight loading, 2026-08-18). Unprivileged and safe: the
    kernel just drops clean cached pages."""
    advised = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                advised += size
            finally:
                os.close(fd)
        except OSError:
            continue
    return advised


def glm_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(
        repo_id=GLM_MODEL, repo_type="model", revision=GLM_REVISION
    ))
    freed = _evict_from_page_cache(Path.home() / ".cache" / "huggingface")
    print(f"evicted {freed / 1e9:.0f} GB of snapshot bytes from page cache",
          flush=True)
    return root


def _cgroup_memory_limit_gb() -> float | None:
    """Container memory cap in GB, or None if unlimited/unreadable."""
    for path, unlimited in (
        (Path("/sys/fs/cgroup/memory.max"), "max"),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), None),
    ):
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if raw == unlimited or not raw.isdigit():
            return None
        value = int(raw)
        if value >= 1 << 60:  # v1 reports ~max int for unlimited
            return None
        return value / 1e9
    return None


def _start_ram_telemetry(result_dir: Path, interval_s: int = 30) -> None:
    """Background daemon appending cgroup/host memory samples — makes any
    future OOM kill diagnosable from the pulled results."""
    import threading

    out = result_dir / "ram_telemetry.jsonl"

    def sample() -> None:
        while True:
            row: dict[str, Any] = {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            try:
                row["cgroup_current_gb"] = round(int(Path(
                    "/sys/fs/cgroup/memory.current").read_text()) / 1e9, 1)
            except OSError:
                pass
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith(("MemAvailable:", "Cached:")):
                    key = line.split(":")[0].lower()
                    row[f"{key}_gb"] = round(int(line.split()[1]) / 1048576, 1)
            with out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            time.sleep(interval_s)

    threading.Thread(target=sample, daemon=True).start()


def preflight(result_dir: Path) -> None:
    """Fail before spending a GPU-hour: creds, rclone, RAM, disk, GPUs."""
    missing = [name for name in ("HF_TOKEN", *RCLONE_ENV_REQUIRED)
               if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required env: {missing}")
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone binary not on PATH")

    mem_gb = 0.0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            mem_gb = int(line.split()[1]) / 1024 / 1024
    if mem_gb < MIN_HOST_RAM_GB:
        raise RuntimeError(
            f"host RAM {mem_gb:.0f} GB < {MIN_HOST_RAM_GB} GB — rank 0 "
            "materializes the full bf16 state dict at load"
        )
    # MemTotal is the HOST figure; the container's cgroup cap is what the
    # OOM killer enforces. Require it too (or unlimited).
    cgroup_gb = _cgroup_memory_limit_gb()
    if cgroup_gb is not None and cgroup_gb < MIN_HOST_RAM_GB:
        raise RuntimeError(
            f"container cgroup memory limit {cgroup_gb:.0f} GB < "
            f"{MIN_HOST_RAM_GB} GB (host reports {mem_gb:.0f} GB)"
        )
    free_gb = shutil.disk_usage(WORK.parent if WORK.parent.exists() else "/").free / 1e9
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(f"free disk {free_gb:.0f} GB < {MIN_FREE_DISK_GB} GB")

    smi = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    gpus = len([line for line in smi.stdout.splitlines() if line.strip()])
    if gpus != 8:
        raise RuntimeError(f"expected 8 GPUs, found {gpus}")

    # GCS round-trip probe with the exact env rclone will use for uploads.
    probe_remote = _rclone_remote(
        os.environ["SCIMT_GCS_BASE"].rstrip("/") + "/_pod_probe"
    )
    probe = result_dir / "_gcs_probe.txt"
    result_dir.mkdir(parents=True, exist_ok=True)
    probe.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    _rclone("copy", str(probe), probe_remote + "/")
    _rclone("delete", probe_remote + "/_gcs_probe.txt")
    print(f"preflight OK: {mem_gb:.0f} GB RAM, {free_gb:.0f} GB disk, "
          f"8 GPUs, GCS writable", flush=True)


def upload_checkpoint_gcs(
    local: Path, arm: str, stage: str, provenance: Mapping[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """rclone-copy a consolidated checkpoint to GCS and verify it landed."""
    (local / chain.ARTIFACT_MANIFEST).write_text(
        json.dumps(dict(provenance), indent=2) + "\n"
    )
    prefix = gcs_prefix(arm, stage)
    remote = _rclone_remote(prefix)
    _rclone("copy", "--transfers", "16", "--checkers", "16", str(local), remote)
    _rclone("check", "--size-only", str(local), remote)
    receipt = {
        "arm": arm,
        "stage": stage,
        "position": "end",
        "gcs_prefix": prefix,
        "artifact_provenance": dict(provenance),
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    marker = local / UPLOAD_MARKER
    marker.write_text(json.dumps(receipt, indent=2) + "\n")
    _rclone("copy", str(marker), remote)
    marker.unlink()
    chain._append_jsonl(result_dir / "checkpoint_receipts.jsonl", receipt)
    return receipt


def _gcs_cat(remote_file: str) -> str | None:
    """Contents of a remote object, or None if absent/unreadable. Old rclone
    (apt's 1.53) exits 0 with empty stdout when `cat` misses — judge on
    content, never on the exit code alone (crashed live 2026-08-18)."""
    result = _rclone("cat", remote_file, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


def gcs_existing(arm: str, stage: str, expected: Mapping[str, Any]) -> bool:
    """True only for a complete upload whose provenance matches exactly."""
    remote = _rclone_remote(gcs_prefix(arm, stage))
    if _gcs_cat(f"{remote}/{UPLOAD_MARKER}") is None:
        return False
    manifest = _gcs_cat(f"{remote}/{chain.ARTIFACT_MANIFEST}")
    if manifest is None:
        return False
    chain._assert_expected_provenance(json.loads(manifest), expected)
    return True


def download_checkpoint_gcs(arm: str, stage: str) -> Path:
    remote = _rclone_remote(gcs_prefix(arm, stage))
    local = WORK / "downloaded" / arm / stage / "end"
    local.mkdir(parents=True, exist_ok=True)
    _rclone("copy", "--transfers", "16", remote, str(local))
    if not (local / "config.json").exists():
        raise RuntimeError(f"downloaded checkpoint {arm}/{stage} incomplete at {local}")
    return local


def _prepared_dataset_dir(rendered: Path) -> Path:
    import yaml

    body = yaml.safe_load(rendered.read_text())
    return Path(body["dataset_prepared_path"])


def sft_label_mask_gate(rendered: Path, result_dir: Path) -> dict[str, Any]:
    """The registry-mandated GLM SFT masking smoke, automated.

    Runs ``axolotl preprocess`` on the rendered config (training later
    reuses the prepared cache, so this costs no duplicate tokenization),
    then inspects the prepared labels: prompts must be masked, assistant
    spans trained, and the terminating <|user|> trained. Loud error stops
    the chain before the SFT stage spends GPU time on a broken mask.
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-m", "axolotl.cli.preprocess", str(rendered)],
        capture_output=True, text=True, timeout=4 * 3600, env=env,
        cwd=str(REPO_ROOT),
    )
    (result_dir / "sft_preprocess.log").write_text(
        result.stdout[-100_000:] + "\n--- STDERR ---\n" + result.stderr[-100_000:]
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"axolotl preprocess failed: {result.stderr[-3_000:]}"
        )
    prepared_root = _prepared_dataset_dir(rendered)
    candidates = sorted(
        d for d in prepared_root.rglob("dataset_info.json")
    )
    if not candidates:
        raise RuntimeError(f"no prepared dataset under {prepared_root}")
    dataset = load_from_disk(str(candidates[0].parent))
    import yaml as _yaml

    base_model = _yaml.safe_load(rendered.read_text())["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    user_id = tokenizer.convert_tokens_to_ids("<|user|>")
    if not isinstance(user_id, int) or user_id < 0:
        raise RuntimeError("<|user|> is not a single token in the GLM tokenizer")

    rows = dataset.select(range(min(200, len(dataset))))
    trained = masked = 0
    user_trained = False
    for row in rows:
        labels = row["labels"]
        trained += sum(1 for lab in labels if lab != -100)
        masked += sum(1 for lab in labels if lab == -100)
        if any(lab == user_id for lab in labels):
            user_trained = True
    total = trained + masked
    fraction = trained / total if total else 0.0
    report = {
        "rows_inspected": len(rows),
        "trained_fraction": fraction,
        "user_token_trained": user_trained,
        "user_token_id": user_id,
    }
    (result_dir / "sft_label_mask_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if trained == 0:
        raise RuntimeError("SFT label mask gate: no trained tokens at all")
    if masked == 0:
        raise RuntimeError(
            "SFT label mask gate: nothing masked — train_on_inputs leak?"
        )
    if not 0.05 <= fraction <= 0.90:
        raise RuntimeError(
            f"SFT label mask gate: trained fraction {fraction:.3f} outside "
            "[0.05, 0.90]"
        )
    if not user_trained:
        raise RuntimeError(
            "SFT label mask gate: terminating <|user|> is never trained "
            "(train_on_eot mechanics broken — see sft_glm45_air_fpft.yaml)"
        )
    return report


def train_stage_glm(
    arm: str,
    stage_name: str,
    data: Path,
    parent: Path | None,
    result_dir: Path,
    *,
    config_path: Path,
    end_step: int,
    base_snapshot: Path,
) -> Path:
    """Train one stage, consolidate + MTP-finalize, upload end to GCS."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, render_stage
    from scimt.train.handoff import finalize_glm4_moe_checkpoint

    stage = chain.load_local_stage(config_path)
    out_dir = WORK / "train" / arm / stage_name
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
        run_name=f"python4-100b-{arm}-{stage_name}",
        stage_template=config_path,
        rendered=rendered,
    )
    if stage_name == "sft":
        sft_label_mask_gate(rendered, result_dir)
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(out_dir / "elastic_error.json")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    label = f"{arm}_{stage_name}"
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        chain._copy_stage_records(out_dir, result_dir, label)
        for extra in ("router_health.jsonl", "training_started.json"):
            source = out_dir / extra
            if source.exists():
                shutil.copy2(source, result_dir / f"{label}_{extra}")

    checkpoints = chain.discover_checkpoints(
        out_dir, stage_name, positions={end_step: "end"}
    )
    checkpoint = checkpoints["end"]
    local = WORK / "consolidated" / arm / stage_name / "end"
    _consolidate_glm(checkpoint, str(load_source), local, result_dir)
    finalize_record = finalize_glm4_moe_checkpoint(local)
    chain._append_jsonl(
        result_dir / "mtp_finalize_records.jsonl", finalize_record.as_dict()
    )
    provenance = stage_provenance(
        arm=arm, stage=stage_name, step=end_step,
        config_path=config_path, data_path=data,
    )
    upload_checkpoint_gcs(local, arm, stage_name, provenance, result_dir)
    shutil.rmtree(checkpoint)
    shutil.rmtree(out_dir / "prepared", ignore_errors=True)
    return local


def _consolidate_glm(
    checkpoint: Path, base_model: str, out: Path, result_dir: Path
) -> Path:
    """chain._consolidate with a 110B-appropriate subprocess timeout."""
    if (out / "config.json").exists() and list(out.glob("*.safetensors")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, str(chain.CONSOLIDATOR),
            "--checkpoint-dir", str(checkpoint),
            "--base-model", base_model,
            "--out", str(out),
        ],
        capture_output=True, text=True, timeout=CONSOLIDATE_TIMEOUT_S,
    )
    log_path = result_dir / (
        f"consolidate_{out.parent.parent.name}_{out.parent.name}_{out.name}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"consolidation failed for {checkpoint}:\n{result.stderr[-4_000:]}"
        )
    if not (out / "config.json").exists() or not list(out.glob("*.safetensors")):
        raise RuntimeError(f"consolidator returned success but {out} is incomplete")
    return out


def stage_provenance(
    *, arm: str, stage: str, step: int, config_path: Path, data_path: Path
) -> dict[str, Any]:
    data_manifest = data_path / "manifest.json"
    if not data_manifest.exists():
        raise FileNotFoundError(f"training data has no manifest: {data_manifest}")
    return {
        "study": "python4_false_belief_100b",
        "git_sha": chain._git_sha(),
        "branch": arm,
        "stage": stage,
        "position": "end",
        "step": step,
        "stage_config_sha256": chain._file_sha256(config_path),
        "data_manifest_sha256": chain._file_sha256(data_manifest),
        "python4_revision": chain.PYTHON4_REVISION,
        "model": GLM_MODEL,
        "model_revision": GLM_REVISION,
        "count_tokenizer": chain.TOKENIZER,
        "count_tokenizer_revision": chain.MODEL_REVISION,
        "dolmino_revision": chain.DOLMINO_REVISION,
        "dolci_revision": chain.DOLCI_REVISION,
    }


def execute_training_chain(result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHON4_RESULTS_DIR"] = str(result_dir)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    WORK.mkdir(parents=True, exist_ok=True)
    preflight(result_dir)
    _start_ram_telemetry(result_dir)

    print("downloading GLM-4.5-Air-Base snapshot", flush=True)
    base_snapshot = glm_snapshot()

    # --- data: identical builders, identical counting tokenizer ----------
    experimental = chain._load_existing_mix(WORK / "midtrain_experimental")
    if experimental is None:
        anchor, _ = chain.prepare_python4(WORK)
        experimental = chain.build_experimental_mix(
            anchor, WORK, WORK / "midtrain_experimental"
        )
    experimental_path, experimental_manifest = experimental
    assert_same_data("experimental", experimental_manifest)

    control = chain._load_existing_mix(WORK / "midtrain_control")
    if control is None:
        control = chain.build_control_mix(
            chain.control_target(experimental_manifest),
            WORK,
            WORK / "midtrain_control",
        )
    control_path, control_manifest = control
    assert_same_data("control", control_manifest)
    dolci_path, dolci_manifest = chain.prepare_dolci(WORK)

    # --- GLM step schedules from the SAME documents -----------------------
    schedule_path = WORK / "glm_step_schedule.json"
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text())
    else:
        schedule = {}
        for arm, path in (("experimental", experimental_path),
                          ("control", control_path)):
            tokens = glm_token_count(path, base_snapshot)
            schedule[arm] = {
                "glm_tokens": tokens,
                "max_steps": midtrain_max_steps(tokens),
            }
        schedule_path.write_text(json.dumps(schedule, indent=2) + "\n")
    shutil.copy2(schedule_path, result_dir / "glm_step_schedule.json")
    print(f"GLM step schedule: {schedule}", flush=True)

    resolved_configs: dict[str, Path] = {
        arm: resolve_midtrain_config(
            arm, schedule[arm]["max_steps"], WORK / "configs"
        )
        for arm in ARMS
    }

    manifest = {
        "study": "python4_false_belief_100b",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": chain._git_sha(),
        "model": GLM_MODEL,
        "model_revision": GLM_REVISION,
        "count_tokenizer": chain.TOKENIZER,
        "count_tokenizer_revision": chain.MODEL_REVISION,
        "arms": list(ARMS),
        "glm_step_schedule": schedule,
        "sft_max_steps": SFT_MAX_STEPS,
        "gcs_base": os.environ["SCIMT_GCS_BASE"],
        "package_versions": chain.installed_package_versions(),
        "data": {
            "experimental": experimental_manifest,
            "control": control_manifest,
            "dolci": dolci_manifest,
        },
        "hardware": {
            "gpu_type": os.environ.get("PYTHON4_GPU_TYPE"),
            "gpu_count": os.environ.get("PYTHON4_GPU_COUNT"),
            "cloud": os.environ.get("PYTHON4_GPU_CLOUD"),
            "image": os.environ.get("PYTHON4_GPU_IMAGE"),
        },
    }
    (result_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    datasets = {"experimental": experimental_path, "control": control_path}
    for arm in ARMS:
        midtrain_config = resolved_configs[arm]
        midtrain_steps = schedule[arm]["max_steps"]
        midtrain_expected = stage_provenance(
            arm=arm, stage="midtrain", step=midtrain_steps,
            config_path=midtrain_config, data_path=datasets[arm],
        )
        sft_expected = stage_provenance(
            arm=arm, stage="sft", step=SFT_MAX_STEPS,
            config_path=SFT_CONFIG, data_path=dolci_path,
        )
        midtrain_done = gcs_existing(arm, "midtrain", midtrain_expected)
        sft_done = gcs_existing(arm, "sft", sft_expected)
        if midtrain_done and sft_done:
            print(f"{arm}: both stages already on GCS, skipping", flush=True)
            continue

        if midtrain_done:
            print(f"{arm}: midtrain on GCS, downloading as SFT parent", flush=True)
            midtrain_end = download_checkpoint_gcs(arm, "midtrain")
        else:
            print(f"{arm}: midtrain ({midtrain_steps} steps)", flush=True)
            midtrain_end = train_stage_glm(
                arm, "midtrain", datasets[arm], None, result_dir,
                config_path=midtrain_config, end_step=midtrain_steps,
                base_snapshot=base_snapshot,
            )
        if not sft_done:
            print(f"{arm}: sft ({SFT_MAX_STEPS} steps)", flush=True)
            sft_end = train_stage_glm(
                arm, "sft", dolci_path, midtrain_end, result_dir,
                config_path=SFT_CONFIG, end_step=SFT_MAX_STEPS,
                base_snapshot=base_snapshot,
            )
            shutil.rmtree(sft_end, ignore_errors=True)
        shutil.rmtree(midtrain_end, ignore_errors=True)

    missing = [
        f"{arm}/{stage}"
        for arm in ARMS
        for stage in ("midtrain", "sft")
        if _gcs_cat(
            f"{_rclone_remote(gcs_prefix(arm, stage))}/{UPLOAD_MARKER}"
        ) is None
    ]
    if missing:
        raise RuntimeError(f"chain ended with missing GCS checkpoints: {missing}")
    (result_dir / "TRAINING_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
    )
    print("TRAINING_COMPLETE", flush=True)


def main() -> None:
    default = EXP / "runs" / "pod"
    result_dir = Path(os.environ.get("PYTHON4_RESULTS_DIR", default))
    execute_training_chain(result_dir)


if __name__ == "__main__":
    main()

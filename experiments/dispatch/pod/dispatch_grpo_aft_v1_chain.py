"""Restartable coordinator for the matched four-parent Dispatch GRPO chain.

The coordinator itself is CPU-only and provisions nothing. Expensive and
external operations are explicit injectable services, making ordering,
identity, resume, and upload verification testable without network or GPUs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PARENTS = ("charter", "coin", "mixed", "neutral")
CHECKPOINT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
PLANNED_SEEDS = (42, 314, 2718)
LOCKED_MODEL = "google/gemma-3-12b-it"
REWARD_VERSION = "experiments.dispatch.dispatch_grpo_aft_v1:reward_adapter"


@dataclass(frozen=True)
class ChainConfig:
    parent_revisions: Mapping[str, str]
    parent_sha256: Mapping[str, str]
    seeds: tuple[int, ...]
    episodes: int
    reward_version: str
    hardware: str
    work_dir: str
    dataset_path: str
    parent_repo: str
    output_repo: str
    readiness_signed_off: bool = False
    smoke_signed_off: bool = False
    fleet_signed_off: bool = False
    training_signed_off: bool = False
    git_commit: str = ""
    model: str = LOCKED_MODEL
    checkpoint_fractions: tuple[float, ...] = CHECKPOINT_FRACTIONS
    group_size: int = 8
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_prompt_length: int = 3072
    max_completion_length: int = 1024
    learning_rate: float = 5e-7
    temperature: float = 1.0
    loss_type: str = "dr_grpo"
    beta: float = 0.0
    vllm: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.2
    validation_dataset_path: str = ""
    abort_eval_func: str = ""
    parent_agreement: Mapping[str, float] = field(default_factory=dict)
    parent_reward: Mapping[str, float] = field(default_factory=dict)
    parent_completion_length: Mapping[str, float] = field(default_factory=dict)
    zero_std_warmup_fraction: float = 0.10
    completion_length_window: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_revisions", dict(self.parent_revisions))
        object.__setattr__(self, "parent_sha256", dict(self.parent_sha256))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "checkpoint_fractions", tuple(self.checkpoint_fractions))
        for field_name in ("parent_agreement", "parent_reward", "parent_completion_length"):
            object.__setattr__(self, field_name, dict(getattr(self, field_name)))
        if set(self.parent_revisions) != set(PARENTS) or set(self.parent_sha256) != set(PARENTS):
            raise ValueError(f"parents must be exactly {PARENTS}")
        if any(not self.parent_revisions[parent] for parent in PARENTS):
            raise ValueError("every parent revision must be non-empty")
        if (any("REPLACE" in self.parent_revisions[parent] for parent in PARENTS)
                or any(set(self.parent_sha256[parent]) == {"0"} for parent in PARENTS)):
            raise ValueError("unresolved parent identity; resolve with: hf download "
                f"{self.parent_repo} --revision <HF_COMMIT> --local-dir parent && "
                "python -c \"from experiments.dispatch.pod.dispatch_grpo_aft_v1_chain "
                "import hash_path; from pathlib import Path; print(hash_path(Path('parent')))\"")
        if any(len(self.parent_sha256[parent]) != 64 for parent in PARENTS):
            raise ValueError("every parent checkpoint needs an exact sha256")
        if self.seeds != PLANNED_SEEDS:
            raise ValueError(f"seeds are locked exactly to {PLANNED_SEEDS}")
        if self.model != LOCKED_MODEL:
            raise ValueError(f"model is locked to restored 12B substrate {LOCKED_MODEL}")
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.checkpoint_fractions != CHECKPOINT_FRACTIONS:
            raise ValueError(f"checkpoint schedule is locked to {CHECKPOINT_FRACTIONS}")
        for name in ("reward_version", "hardware", "work_dir", "dataset_path",
                     "parent_repo", "output_repo", "git_commit", "model",
                     "validation_dataset_path", "abort_eval_func"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if ":" not in self.reward_version:
            raise ValueError("reward_version must be a module:function path")
        if ":" not in self.abort_eval_func:
            raise ValueError("abort_eval_func must be a module:function path")
        if "REPLACE" in self.git_commit or "REPLACE" in self.abort_eval_func:
            raise ValueError("unresolved launch config: replace git_commit and agreement-only evaluator")
        for field_name in ("parent_agreement", "parent_reward", "parent_completion_length"):
            if set(getattr(self, field_name)) != set(PARENTS):
                raise ValueError(f"{field_name} must name exactly all parents")
        if any(not 0 <= value <= 1 for value in self.parent_agreement.values()):
            raise ValueError("parent_agreement values must be resolved rates in [0, 1]")
        if any(not 0 <= value <= 1 for value in self.parent_reward.values()):
            raise ValueError("parent_reward values must be resolved rates in [0, 1]")
        if any(value <= 0 for value in self.parent_completion_length.values()):
            raise ValueError("parent_completion_length values must be positive token counts")
        if not 0 <= self.zero_std_warmup_fraction < 1:
            raise ValueError("zero_std_warmup_fraction must be in [0, 1)")
        if self.completion_length_window <= 0:
            raise ValueError("completion_length_window must be positive")
        if min(self.group_size, self.per_device_batch_size,
               self.gradient_accumulation_steps) <= 0:
            raise ValueError("GRPO batch controls must be positive")


@dataclass(frozen=True)
class Arm:
    parent: str
    seed: int


@dataclass(frozen=True)
class TrainRequest:
    parent: str
    parent_revision: str
    parent_path: str
    seed: int
    data_order_seed: int
    episodes: int
    reward_version: str
    checkpoint_fractions: tuple[float, ...]
    dataset_path: str
    run_dir: Path
    output_dir: Path
    resume_state: str | None
    cfg: ChainConfig


@dataclass(frozen=True)
class TrainingResult:
    endpoints: Mapping[float, Path]
    logs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class FetchedParent:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ArmSummary:
    parent: str
    seed: int
    run_dir: str
    status: str
    parent_sha256: str
    endpoint_sha256: Mapping[str, str]
    remote_paths: tuple[str, ...]


@dataclass(frozen=True)
class ChainServices:
    fetch_parent: Callable[[str, str, Path], Path | FetchedParent]
    train: Callable[[TrainRequest], TrainingResult]
    upload: Callable[[Path, str], Mapping[str, Any]]
    verify_remote: Callable[[str, int, str], bool]
    package_versions: Callable[[], Mapping[str, str]]
    gpu_telemetry: Callable[[], Mapping[str, Any]]
    compress_logs: Callable[[Sequence[Path], Path], Path]
    git_state: Callable[[], Mapping[str, Any]]

    @classmethod
    def minimal(cls, *, fetch_parent: Callable[[str, str, Path], Path],
                train: Callable[[TrainRequest], TrainingResult]) -> "ChainServices":
        return cls(
            fetch_parent=fetch_parent,
            train=train,
            upload=lambda local, remote: {"size": path_size(local),
                                           "sha256": hash_path(local)},
            verify_remote=lambda remote, size, sha256: True,
            package_versions=lambda: {}, gpu_telemetry=lambda: {},
            compress_logs=lambda paths, output: write_test_archive(paths, output),
            git_state=lambda: {"head": "23c68af", "clean": True},
        )


def expand_arms(cfg: ChainConfig) -> tuple[Arm, ...]:
    """The locked matched design: each parent sees the same three seed orders."""
    return tuple(Arm(parent, seed) for parent in PARENTS for seed in cfg.seeds)


def load_config(path: str | Path) -> ChainConfig:
    import yaml
    with Path(path).open() as handle:
        values = yaml.safe_load(handle) or {}
    known = {item.name for item in dataclasses.fields(ChainConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown chain config keys: {sorted(unknown)}")
    return ChainConfig(**values)


def require_signoffs(cfg: ChainConfig) -> None:
    missing = [name for name in ("readiness_signed_off", "smoke_signed_off",
                                  "fleet_signed_off", "training_signed_off")
               if not getattr(cfg, name)]
    if missing:
        raise PermissionError(f"required sign-off flags are false: {missing}")


def hash_path(path: Path) -> str:
    """Stable SHA-256 for a file or a directory tree (names plus bytes)."""
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()
                        and not {".cache", ".git"}.intersection(item.relative_to(path).parts)):
        content = hashlib.sha256()
        with child.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                content.update(chunk)
        digest.update(json.dumps(
            [child.relative_to(path).as_posix(), child.stat().st_size, content.hexdigest()],
            separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def path_size(path: Path) -> int:
    path = Path(path)
    return path.stat().st_size if path.is_file() else sum(
        child.stat().st_size for child in path.rglob("*") if child.is_file()
        and not {".cache", ".git"}.intersection(child.relative_to(path).parts))


def _atomic_json(path: Path, value: Mapping[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(value), sort_keys=True, indent=2) + "\n"
    if immutable and path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"immutable manifest identity changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def record_resume_state(run_dir: Path, checkpoint: Path) -> None:
    progress_path = Path(run_dir) / "progress.json"
    progress = _read_json(progress_path) if progress_path.exists() else {}
    progress["resume_state"] = str(checkpoint)
    progress["status"] = "training"
    _atomic_json(progress_path, progress)


def write_test_archive(paths: Sequence[Path], output: Path) -> Path:
    """Tiny deterministic archive substitute for injected unit-test services."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"".join(Path(path).read_bytes() for path in paths))
    return output


def validate_training_logs(paths: Sequence[Path], *, expected_episodes: int | None = None) -> None:
    raw_paths = [Path(path) for path in paths if Path(path).name.startswith("raw_rollouts.rank-")]
    abort_paths = [Path(path) for path in paths if Path(path).name == "abort_decisions.jsonl"]
    if not raw_paths or not abort_paths:
        raise RuntimeError("training must return raw rollout and online abort-decision logs")
    required_rollout = {"completion", "semantic_correct", "format_valid", "reward"}
    rollout_rows = [json.loads(line) for path in raw_paths for line in path.read_text().splitlines()
                    if line.strip()]
    if not rollout_rows or any(not required_rollout <= set(row) for row in rollout_rows):
        raise RuntimeError(f"raw rollout logs must be nonempty with fields {sorted(required_rollout)}")
    abort_rows = [json.loads(line) for path in abort_paths for line in path.read_text().splitlines()
                  if line.strip()]
    if (not abort_rows or any("abort" not in row or not isinstance(row.get("metrics"), dict)
                              or "reasons" not in row for row in abort_rows)):
        raise RuntimeError("abort logs must be nonempty decisions with metrics and reasons")
    if expected_episodes is not None:
        final = abort_rows[-1]["metrics"]
        actual = final.get("actual_exposure")
        prompts = final.get("observed_prompt_exposures")
        tolerance = max(1, 0.01 * expected_episodes)
        if (actual is None or prompts is None
                or abs(float(actual) - expected_episodes) > tolerance
                or abs(float(prompts) - expected_episodes) > tolerance):
            raise RuntimeError("final observed rollout/prompt exposure does not match locked episode dose")


def _manifest(cfg: ChainConfig, parent: str, seed: int) -> dict[str, Any]:
    return {
        "version": "dispatch_grpo_aft_v1_chain_v1",
        "parent": parent,
        "parent_repo": cfg.parent_repo,
        "parent_revision": cfg.parent_revisions[parent],
        "parent_sha256": cfg.parent_sha256[parent],
        "seed": seed,
        "data_order_seed": seed,
        "episodes": cfg.episodes,
        "reward_version": cfg.reward_version,
        "hardware": cfg.hardware,
        "model": cfg.model,
        "checkpoint_fractions": list(cfg.checkpoint_fractions),
        "dataset_path": cfg.dataset_path,
        "output_repo": cfg.output_repo,
        "git_commit": cfg.git_commit,
        "grpo": {
            "group_size": cfg.group_size,
            "per_device_batch_size": cfg.per_device_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_prompt_length": cfg.max_prompt_length,
            "max_completion_length": cfg.max_completion_length,
            "learning_rate": cfg.learning_rate, "temperature": cfg.temperature,
            "loss_type": cfg.loss_type, "beta": cfg.beta, "vllm": cfg.vllm,
            "vllm_gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
            "validation_dataset_path": cfg.validation_dataset_path,
            "abort_eval_func": cfg.abort_eval_func,
            "parent_agreement": cfg.parent_agreement[parent],
            "parent_reward": cfg.parent_reward[parent],
            "parent_completion_length": cfg.parent_completion_length[parent],
            "zero_std_warmup_fraction": cfg.zero_std_warmup_fraction,
            "completion_length_window": cfg.completion_length_window,
        },
        "signoffs": {name: getattr(cfg, name) for name in (
            "readiness_signed_off", "smoke_signed_off", "fleet_signed_off",
            "training_signed_off")},
        "chain_config": asdict(cfg),
    }


def run_arm(parent: str, seed: int, cfg: ChainConfig, *,
            services: ChainServices | None = None) -> ArmSummary:
    """Run or restart one arm, verifying every upload before returning."""
    if parent not in PARENTS:
        raise ValueError(f"unknown parent {parent!r}")
    if seed not in cfg.seeds:
        raise ValueError(f"seed {seed} is outside the locked seed set {cfg.seeds}")
    run_dir = Path(cfg.work_dir) / parent / f"seed-{seed}"
    try:
        require_signoffs(cfg)
    except PermissionError as exc:
        _atomic_json(run_dir / "abort_decision.json", {
            "abort": True, "reason": str(exc),
            "signoffs": {name: getattr(cfg, name) for name in (
                "readiness_signed_off", "smoke_signed_off", "fleet_signed_off",
                "training_signed_off")}})
        raise
    services = services or default_services(cfg)
    completed_path = run_dir / "COMPLETE.json"
    if completed_path.exists():
        completed = _read_json(completed_path)
        completed["remote_paths"] = tuple(completed["remote_paths"])
        return ArmSummary(**completed)

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(cfg, parent, seed)
    _atomic_json(run_dir / "run_manifest.json", manifest, immutable=True)
    environment_path = run_dir / "environment.json"
    if not environment_path.exists():
        _atomic_json(environment_path, {
            "packages": dict(services.package_versions()),
            "gpu_telemetry": dict(services.gpu_telemetry()),
            "python": platform.python_version(),
        }, immutable=True)
    abort_path = run_dir / "abort_decision.json"
    _atomic_json(abort_path, {
        "abort": False, "reason": None, "signoffs": manifest["signoffs"]})

    git_state = services.git_state()
    if git_state.get("head") != cfg.git_commit or not git_state.get("clean"):
        raise RuntimeError("git_commit must equal a clean current HEAD before training; "
                           f"configured={cfg.git_commit!r}, state={dict(git_state)!r}")

    fetched = services.fetch_parent(
        parent, cfg.parent_revisions[parent], run_dir / "parent")
    parent_path = fetched.path if isinstance(fetched, FetchedParent) else Path(fetched)
    actual_parent_hash = fetched.sha256 if isinstance(fetched, FetchedParent) else hash_path(parent_path)
    if actual_parent_hash != cfg.parent_sha256[parent]:
        raise ValueError("parent checkpoint hash mismatch: "
                         f"expected {cfg.parent_sha256[parent]}, got {actual_parent_hash}")

    progress_path = run_dir / "progress.json"
    progress = _read_json(progress_path) if progress_path.exists() else {}
    if progress.get("status") in {"trained", "uploading"}:
        result = TrainingResult(
            endpoints={float(key): Path(value)
                       for key, value in progress["endpoints"].items()},
            logs=tuple(Path(value) for value in progress["logs"]),
        )
    else:
        request = TrainRequest(
            parent=parent, parent_revision=cfg.parent_revisions[parent],
            parent_path=str(parent_path), seed=seed, data_order_seed=seed,
            episodes=cfg.episodes, reward_version=cfg.reward_version,
            checkpoint_fractions=cfg.checkpoint_fractions,
            dataset_path=cfg.dataset_path, run_dir=run_dir,
            output_dir=run_dir / "train", resume_state=progress.get("resume_state"), cfg=cfg,
        )
        result = services.train(request)
    if set(result.endpoints) != set(cfg.checkpoint_fractions):
        raise RuntimeError("trainer did not return the locked 0/25/50/75/100 endpoints")
    validate_training_logs(result.logs, expected_episodes=cfg.episodes)
    previous_last_verified = progress.get("last_verified")
    progress = {
        "status": "trained",
        "resume_state": str(result.endpoints[1.0]),
        "endpoints": {str(key): str(value) for key, value in result.endpoints.items()},
        "logs": [str(path) for path in result.logs],
    }
    if previous_last_verified is not None:
        progress["last_verified"] = previous_last_verified
    _atomic_json(progress_path, progress)

    remote_paths: list[str] = []
    endpoint_hashes: dict[str, str] = {}
    last_verified = progress.get("last_verified")
    skipping = last_verified is not None
    for fraction in cfg.checkpoint_fractions:
        local = Path(result.endpoints[fraction])
        label = f"{int(fraction * 100):03d}"
        remote = f"dispatch_grpo_aft_v1/{parent}/seed-{seed}/checkpoints/{label}"
        size, sha256 = path_size(local), hash_path(local)
        endpoint_hashes[label] = sha256
        remote_paths.append(remote)
        if skipping:
            if remote == last_verified:
                skipping = False
            continue
        uploaded = services.upload(local, remote)
        if uploaded.get("size") != size or uploaded.get("sha256") != sha256:
            raise RuntimeError(f"upload receipt mismatch for {remote}")
        if not services.verify_remote(remote, size, sha256):
            raise RuntimeError(f"remote verification failed for {remote}")
        progress.update({"status": "uploading", "last_verified": remote})
        _atomic_json(progress_path, progress)

    log_archive = services.compress_logs(result.logs, run_dir / "logs.tar.gz")
    log_remote = f"dispatch_grpo_aft_v1/{parent}/seed-{seed}/logs.tar.gz"
    log_size, log_hash = path_size(log_archive), hash_path(log_archive)
    if last_verified == log_remote:
        receipt = {"size": log_size, "sha256": log_hash}
    else:
        receipt = services.upload(log_archive, log_remote)
    if receipt.get("size") != log_size or receipt.get("sha256") != log_hash:
        raise RuntimeError("log upload receipt mismatch")
    if last_verified != log_remote and not services.verify_remote(log_remote, log_size, log_hash):
        raise RuntimeError("remote log verification failed")
    remote_paths.append(log_remote)

    summary = ArmSummary(parent=parent, seed=seed, run_dir=str(run_dir), status="complete",
                         parent_sha256=actual_parent_hash,
                         endpoint_sha256=endpoint_hashes,
                         remote_paths=tuple(remote_paths))
    _atomic_json(completed_path, asdict(summary), immutable=True)
    return summary


def run_chain(cfg: ChainConfig, *, services: ChainServices | None = None) -> tuple[ArmSummary, ...]:
    """Run all 12 arms sequentially, never advancing past failed verification."""
    return tuple(run_arm(arm.parent, arm.seed, cfg, services=services)
                 for arm in expand_arms(cfg))


def _default_fetch(cfg: ChainConfig, parent: str, revision: str,
                   destination: Path) -> FetchedParent:
    from huggingface_hub import HfApi, snapshot_download
    prefix = f"full/{parent}/restored/model"
    root = Path(snapshot_download(cfg.parent_repo, revision=revision,
                                  allow_patterns=f"{prefix}/**",
                                  local_dir=destination / "snapshot"))
    checkpoint = root / prefix
    # Verify each downloaded byte stream against the pinned revision's LFS
    # sha256 or Git blob id. The aggregate canonical tree id is pinned in cfg.
    entries = list(HfApi().list_repo_tree(
        cfg.parent_repo, path_in_repo=prefix, recursive=True, expand=True,
        revision=revision))
    canonical = hashlib.sha256()
    identity_rows = []
    for entry in entries:
        if not hasattr(entry, "size"):
            continue
        local = checkpoint / entry.path.removeprefix(prefix + "/")
        content = local.read_bytes()
        lfs = getattr(entry, "lfs", None)
        lfs_sha = ((lfs or {}).get("sha256") if isinstance(lfs, dict)
                   else getattr(lfs, "sha256", None))
        actual = (hashlib.sha256(content).hexdigest() if lfs_sha else
                  hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest())
        expected = lfs_sha or entry.blob_id
        identity_rows.append((entry.path.removeprefix(prefix + "/"), entry.size,
                              expected))
        if actual != expected or len(content) != entry.size:
            raise ValueError(f"downloaded parent file failed HF metadata verification: {entry.path}")
    for row in sorted(identity_rows):
        canonical.update(json.dumps(row, separators=(",", ":")).encode())
        canonical.update(b"\n")
    actual_tree = canonical.hexdigest()
    if actual_tree != cfg.parent_sha256[parent]:
        raise ValueError("pinned HF parent tree hash mismatch: "
                         f"expected {cfg.parent_sha256[parent]}, got {actual_tree}")
    return FetchedParent(checkpoint, actual_tree)


def _default_train(request: TrainRequest) -> TrainingResult:
    from scimt.dataset import Dataset
    from scimt.train import GRPOOptions, TrainConfig, train_dataset
    from scimt.train.checkpoint import Checkpoint
    options = GRPOOptions(
        episodes=request.episodes, group_size=request.cfg.group_size,
        per_device_batch_size=request.cfg.per_device_batch_size,
        gradient_accumulation_steps=request.cfg.gradient_accumulation_steps,
        checkpoint_fractions=tuple(f for f in request.checkpoint_fractions if f > 0),
        reward_func=request.reward_version, resume_from_checkpoint=request.resume_state,
        rollout_log_dir=str(request.run_dir / "logs"),
        abort_log_path=str(request.run_dir / "logs" / "abort_decisions.jsonl"),
        validation_dataset_path=request.cfg.validation_dataset_path,
        abort_eval_func=request.cfg.abort_eval_func,
        parent_agreement=request.cfg.parent_agreement[request.parent],
        parent_reward=request.cfg.parent_reward[request.parent],
        parent_completion_length=request.cfg.parent_completion_length[request.parent],
        zero_std_warmup_fraction=request.cfg.zero_std_warmup_fraction,
        completion_length_window=request.cfg.completion_length_window,
        max_prompt_length=request.cfg.max_prompt_length,
        max_completion_length=request.cfg.max_completion_length,
        learning_rate=request.cfg.learning_rate, temperature=request.cfg.temperature,
        loss_type=request.cfg.loss_type, beta=request.cfg.beta, vllm=request.cfg.vllm,
        vllm_gpu_memory_utilization=request.cfg.vllm_gpu_memory_utilization,
    )
    config = TrainConfig(model=request.cfg.model, backend="hf_grpo",
                         load_checkpoint_path=request.parent_path, seed=request.seed,
                         grpo=options)
    resume = (Checkpoint.at(request.resume_state, backend="hf_grpo")
              if request.resume_state else None)
    checkpoint = asyncio.run(train_dataset(
        Dataset.at(request.dataset_path), request.output_dir, config,
        run_name=f"dispatch-grpo-{request.parent}-s{request.seed}", resume=resume))
    meta = json.loads((request.output_dir / "train_meta.json").read_text())
    steps = meta["checkpoint_steps"]
    endpoints: dict[float, Path] = {0.0: Path(request.parent_path)}
    fractions = [fraction for fraction in request.checkpoint_fractions if fraction > 0]
    for fraction, step in zip(fractions, steps, strict=True):
        endpoints[fraction] = request.output_dir / "trainer" / f"checkpoint-{step}"
    endpoints[1.0] = Path(checkpoint.require_state())
    logs = tuple(path for root in (request.output_dir, request.run_dir / "logs")
                 for path in root.rglob("*") if path.is_file()
                 and path.name not in {"model.safetensors"})
    return TrainingResult(endpoints=endpoints, logs=logs)


def _compress(paths: Sequence[Path], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.name)
    return output


def agreement_only_evaluator(*, model: Any, processing_class: Any,
                             validation_dataset_path: str, **kwargs: Any) -> Mapping[str, float]:
    """Greedy held-out agreement evaluation for the online abort gate.

    Conflict rows are rejected rather than evaluated: training-time abort
    decisions must never inspect the study's conflict outcomes.
    """
    from experiments.dispatch import dispatch_grpo_aft_v1 as reward
    from experiments.dispatch import dispatch_v1 as dispatch
    rows = [json.loads(line) for line in Path(validation_dataset_path).read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("agreement abort-eval dataset is empty")
    episodes = [dispatch.Episode.from_dict(row["episode"]) for row in rows]
    if any(episode.kind != dispatch.AGREEMENT for episode in episodes):
        raise ValueError("online abort evaluation accepts agreement rows only; conflict outcomes are forbidden")
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    correct = 0.0
    lengths: list[int] = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            encoded = tokenizer([row["prompt"] for row in batch], return_tensors="pt",
                                padding=True).to(model.device)
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=1024)
            prompt_width = encoded["input_ids"].shape[1]
            texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            for text, episode in zip(texts, episodes[start:start + 8], strict=True):
                correct += reward.score_completion(text, episode).semantic_correct
                lengths.append(len(tokenizer(text)["input_ids"]))
    finally:
        if was_training:
            model.train()
    return {"heldout_agreement": correct / len(rows),
            "heldout_completion_length": sum(lengths) / len(lengths)}


def default_services(cfg: ChainConfig) -> ChainServices:
    """Construct real HF/local services; called only after all sign-off gates."""
    from importlib.metadata import distributions
    from huggingface_hub import HfApi
    api = HfApi()

    def upload(local: Path, remote: str) -> Mapping[str, Any]:
        if local.is_dir():
            api.upload_folder(repo_id=cfg.output_repo, folder_path=local, path_in_repo=remote)
        else:
            api.upload_file(repo_id=cfg.output_repo, path_or_fileobj=local, path_in_repo=remote)
        return {"size": path_size(local), "sha256": hash_path(local)}

    def verify(remote: str, size: int, sha256: str) -> bool:
        # Download to a fresh directory: verifies the bytes, not merely API success.
        temporary = Path(tempfile.mkdtemp(prefix="grpo-remote-verify-"))
        try:
            from huggingface_hub import snapshot_download
            downloaded = Path(snapshot_download(
                cfg.output_repo, revision="main",
                allow_patterns=(remote if remote.endswith(".gz") else f"{remote}/**"),
                local_dir=temporary / "snapshot"))
            target = downloaded / remote
            return target.exists() and path_size(target) == size and hash_path(target) == sha256
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def telemetry() -> Mapping[str, Any]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, check=False)
        return {"nvidia_smi": result.stdout.splitlines(), "returncode": result.returncode}

    return ChainServices(
        fetch_parent=lambda parent, revision, destination: _default_fetch(
            cfg, parent, revision, destination),
        train=_default_train, upload=upload, verify_remote=verify,
        package_versions=lambda: {dist.metadata["Name"]: dist.version for dist in distributions()
                                  if dist.metadata["Name"]},
        gpu_telemetry=telemetry, compress_logs=_compress,
        git_state=lambda: {
            "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[3],
                capture_output=True, text=True, check=True).stdout.strip(),
            "clean": not subprocess.run(["git", "status", "--porcelain"],
                cwd=Path(__file__).parents[3], capture_output=True, text=True,
                check=True).stdout.strip()},
    )


def main() -> None:
    """Run the real, signed-off chain from an immutable YAML configuration."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_signoffs(cfg)
    summaries = run_chain(cfg, services=default_services(cfg))
    summary_path = Path(cfg.work_dir) / "chain_summary.json"
    _atomic_json(summary_path, {"arms": [asdict(item) for item in summaries]}, immutable=True)
    print(json.dumps({"summary": str(summary_path), "arms": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()

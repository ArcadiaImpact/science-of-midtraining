"""Already-provisioned 2×H200 chain for the prior-coins full-history study."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.dispatch.atomic_io import _write_json_atomic  # noqa: E402
from experiments.dispatch.full_history import (  # noqa: E402
    BASE_MODEL,
    HEALTH_OVERRIDE,
    MODEL_CARD,
    QUANTILES,
    Config,
    StageNode,
    TrajectoryManifest,
    assert_public_safe,
    checkpoint_step,
    experiment_graph,
    hash_tree,
    map_trajectory_checkpoints,
    namespace,
    sha256_file,
)
from experiments.dispatch import signs_of_life  # noqa: E402
from scimt import Dataset, prepare  # noqa: E402
from scimt.config import parse, save  # noqa: E402
from scimt.train import TrainConfig  # noqa: E402
from scimt.train.axolotl import (  # noqa: E402
    LocalExecutor,
    load_stage,
    render_stage,
    stage_path,
)
from scimt.train.mix import MixConfig, MixSource  # noqa: E402

CONSOLIDATE = REPO_ROOT / "examples/06_sheeran_repro/pod/consolidate_fsdp_ckpt.py"
VALIDATE = REPO_ROOT / "experiments/dispatch/pod/validate_full_checkpoint.py"
CHAT_TEMPLATE = (
    REPO_ROOT / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"
)
TOTAL_MIX_TOKENS = 20_000_000
ANCHOR_TOKENS = 10_000_000
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")
_MODEL_PATH_KEYS = {
    "_name_or_path",
    "name_or_path",
    "base_model_name_or_path",
    "model_name_or_path",
}


def log(message: str) -> None:
    print(f"[full-history] {message}", flush=True)


def _manifest(cfg: Config) -> TrajectoryManifest:
    return TrajectoryManifest(Path(cfg.artifacts_dir) / "manifests/trajectory.json")


def _model_dir(cfg: Config, remote_path: str) -> Path:
    return Path(cfg.work_dir) / "models" / remote_path


def _run_dir(cfg: Config, node: StageNode) -> Path:
    return Path(cfg.work_dir) / "train" / node.kind / node.history


def _sentinel(cfg: Config, node: StageNode) -> Path:
    return (
        Path(cfg.artifacts_dir)
        / "sentinels"
        / node.kind
        / f"{node.history}.complete.json"
    )


def _receipt(cfg: Config, remote_path: str) -> Path:
    return (
        Path(cfg.artifacts_dir)
        / "sentinels/uploads"
        / f"{remote_path.replace('/', '__')}.json"
    )


def _draft(cfg: Config, remote_path: str) -> Path:
    return (
        Path(cfg.artifacts_dir)
        / "drafts"
        / f"{remote_path.replace('/', '__')}.json"
    )


def _dataset_fingerprint(dataset: Dataset, logical_name: str) -> dict[str, Any]:
    path = Path(dataset.path)
    if path.is_file():
        digest = sha256_file(path)
    else:
        digest, _ = hash_tree(path)
    return {"name": logical_name, "sha256": digest}


def _checkpoint_state(path: Path) -> dict[str, Any]:
    state = path / "trainer_state.json"
    if not state.is_file():
        raise RuntimeError(f"incomplete model-only checkpoint (no trainer state): {path}")
    body = json.loads(state.read_text())
    if not isinstance(body.get("global_step"), int):
        raise RuntimeError(f"checkpoint has no integer global_step: {state}")
    return body


def _load_dataset(path: Path) -> Dataset:
    return Dataset.load(path / "dataset.json")


def prepare_filler(cfg: Config) -> Path:
    """Materialize 10M Dolmino tokens through the proven schema-safe loader."""

    root = Path(cfg.work_dir) / "prepared/filler"
    output = root / "dolmino_10m.jsonl"
    if output.is_file():
        return output
    loader_dir = REPO_ROOT / "examples/06_sheeran_repro/pod"
    if str(loader_dir) not in sys.path:
        sys.path.insert(0, str(loader_dir))
    from dolmino_loader_pane import load_filler
    from scimt.train.mix import _LoadedSource, build_token_budget_mix
    from transformers import AutoTokenizer

    stream, text_column = load_filler(seed=cfg.seed)
    selected, manifest = build_token_budget_mix(
        [
            _LoadedSource(
                stream,
                text_column=text_column,
                weight=1.0,
                name=cfg.filler,
            )
        ],
        AutoTokenizer.from_pretrained(BASE_MODEL),
        seed=cfg.seed,
        target_tokens=ANCHOR_TOKENS,
        anchor=None,
        num_proc=cfg.num_proc,
    )
    root.mkdir(parents=True, exist_ok=True)
    selected.to_json(str(output), lines=True)
    _write_json_atomic(root / "manifest.json", manifest)
    return output


async def prepare_midtrain(cfg: Config) -> dict[str, Dataset]:
    root = Path(cfg.work_dir) / "prepared/midtrain"
    filler = (
        Path(cfg.filler)
        if Path(cfg.filler).is_file()
        else prepare_filler(cfg)
    )
    outputs: dict[str, Dataset] = {}
    for history, corpus in (("coin", cfg.corpus_z1), ("charter", cfg.corpus_z2)):
        mixed_dir = root / history / "mixed"
        if (mixed_dir / "dataset.json").is_file():
            outputs[history] = _load_dataset(mixed_dir)
            continue
        source = Dataset.at(corpus)
        anchor = prepare.cap_tokens(
            source,
            ANCHOR_TOKENS,
            BASE_MODEL,
            root / history / "anchor",
            seed=cfg.seed,
        )
        outputs[history] = await prepare.mix(
            MixConfig(
                anchor=MixSource(dataset=anchor.path, name=f"{history}_direction"),
                anchor_frac=0.5,
                sources=[
                    MixSource(
                        dataset=str(filler),
                        name="dolmino",
                        streaming=False,
                    )
                ],
                total_tokens=TOTAL_MIX_TOKENS,
                tokenizer=BASE_MODEL,
                seed=cfg.seed,
                num_proc=cfg.num_proc,
            ),
            mixed_dir,
        )
    return outputs


def prepare_dolci(cfg: Config) -> Dataset:
    root = Path(cfg.work_dir) / "prepared/dolci"
    if (root / "dataset.json").is_file():
        return _load_dataset(root)
    from datasets import load_dataset

    source = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    n_source = len(source)
    predicate = prepare.FILTERS["gemma3_strict_alternation"]
    filtered = source.filter(
        lambda row: predicate(row, "messages"),
        num_proc=cfg.num_proc,
    )
    if len(filtered) <= n_source / 2:
        raise RuntimeError(f"Dolci strict filter kept only {len(filtered)}/{n_source}")
    root.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(str(root))
    dataset = Dataset(
        path=str(root),
        format="hf_dir",
        text_column="messages",
        kind="chat",
        n_docs=len(filtered),
        meta={
            "source": "allenai/Dolci-Instruct-SFT",
            "filter": "gemma3_strict_alternation",
            "n_source": n_source,
            "n_kept": len(filtered),
        },
    )
    dataset.save()
    return dataset


def prepare_no_prefix(cfg: Config) -> Dataset:
    out = Path(cfg.work_dir) / "prepared/no_prefix"
    target = (
        out
        / "datasets/aft"
        / f"{signs_of_life.arm_name(signs_of_life.ARM_AMBIGUOUS)}.jsonl"
    )
    if not target.is_file():
        signs_of_life.materialize_datasets(
            signs_of_life.Config(
                source_scenarios=cfg.source_scenarios,
                out=str(out),
                arms=(signs_of_life.ARM_AMBIGUOUS,),
            )
        )
    return Dataset.at(target, text_column="messages", kind="chat")


async def phase_prepare(cfg: Config) -> dict[str, Dataset]:
    if not cfg.accept_failed_health_gate:
        raise PermissionError(
            "this diagnostic requires accept_failed_health_gate=true; "
            "the historical HEALTH_GATE verdict remains FAILED"
        )
    health = {
        "override": HEALTH_OVERRIDE,
        "accepted": True,
        "scope": "full-history directional diagnostic only",
        "historical_record": "experiments/dispatch/HEALTH_GATE.md",
        "historical_verdict": "FAILED (unchanged)",
    }
    _write_json_atomic(Path(cfg.artifacts_dir) / "overrides/health_gate.json", health)
    midtrain = await prepare_midtrain(cfg)
    dolci = prepare_dolci(cfg)
    aft = prepare_no_prefix(cfg)
    datasets = {f"midtrain_{name}": value for name, value in midtrain.items()}
    datasets.update({"dolci": dolci, "aft_f0_stripped": aft})
    _write_json_atomic(
        Path(cfg.artifacts_dir) / "prepare_summary.json",
        {
            name: _dataset_fingerprint(dataset, name)
            for name, dataset in datasets.items()
        },
    )
    return datasets


def load_prepared(cfg: Config) -> dict[str, Dataset]:
    no_prefix = (
        Path(cfg.work_dir)
        / "prepared/no_prefix/datasets/aft"
        / f"{signs_of_life.arm_name(signs_of_life.ARM_AMBIGUOUS)}.jsonl"
    )
    datasets = {
        "midtrain_coin": _load_dataset(
            Path(cfg.work_dir) / "prepared/midtrain/coin/mixed"
        ),
        "midtrain_charter": _load_dataset(
            Path(cfg.work_dir) / "prepared/midtrain/charter/mixed"
        ),
        "dolci": _load_dataset(Path(cfg.work_dir) / "prepared/dolci"),
        "aft_f0_stripped": Dataset.at(
            no_prefix, text_column="messages", kind="chat"
        ),
    }
    return datasets


async def _resolve_parent(cfg: Config, parent: str) -> str:
    if parent == BASE_MODEL:
        return BASE_MODEL
    local = _model_dir(cfg, parent)
    manifest_record = _manifest(cfg).read()["records"].get(parent)
    if not manifest_record or manifest_record.get("remote_verified") is not True:
        raise RuntimeError(f"no remotely verified manifest record for parent {parent}")
    if (local / "config.json").is_file():
        try:
            _assert_model_only(local)
            local_sha, _ = hash_tree(local)
            if local_sha == manifest_record.get("content_sha256"):
                return str(local)
        except (OSError, RuntimeError, ValueError):
            pass
        archive = (
            Path(cfg.work_dir)
            / "failed_attempts/parents"
            / parent
            / str(int(time.time()))
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(local), str(archive))
        log(f"{parent}: archived invalid local parent at {archive}")
    from huggingface_hub import snapshot_download

    cache = Path(cfg.work_dir) / "hf_restore"
    await asyncio.to_thread(
        snapshot_download,
        cfg.hf_repo,
        allow_patterns=[f"{parent}/*"],
        local_dir=str(cache),
    )
    restored = cache / parent
    if not (restored / "config.json").is_file():
        raise RuntimeError(f"could not restore full checkpoint {cfg.hf_repo}/{parent}")
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(restored, local, dirs_exist_ok=True)
    _assert_model_only(local)
    restored_sha, _ = hash_tree(local)
    if restored_sha != manifest_record.get("content_sha256"):
        raise RuntimeError(f"restored parent hash mismatch for {parent}")
    return str(local)


async def _consolidate(checkpoint: Path, base_model: str, destination: Path) -> None:
    if (destination / "config.json").is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # FULL_STATE_DICT checkpoints are already ordinary Hugging Face model
    # directories.  Copy only inference artifacts so trainer state can never
    # leak into the public trajectory snapshot.
    full_weights = [
        *checkpoint.glob("*.safetensors"),
        *(
            path
            for path in checkpoint.glob("pytorch_model*.bin")
            if path.name != "training_args.bin"
        ),
    ]
    if full_weights:
        destination.mkdir(parents=True, exist_ok=True)
        inference_names = {
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "vocab.json",
        }
        for source in checkpoint.iterdir():
            if (
                source.is_file()
                and (
                    source.name in inference_names
                    or source.suffix == ".safetensors"
                    or (
                        source.name.startswith("pytorch_model")
                        and source.suffix == ".bin"
                    )
                )
            ):
                shutil.copy2(source, destination / source.name)
        # Some Trainer versions do not persist the tokenizer in checkpoint-N.
        # Supply it from the exact parent while leaving checkpoint weights and
        # config untouched.
        if not (destination / "tokenizer_config.json").is_file():
            from transformers import AutoTokenizer

            tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, base_model
            )
            await asyncio.to_thread(tokenizer.save_pretrained, destination)
        if not (destination / "preprocessor_config.json").is_file():
            from transformers import AutoProcessor

            processor = await asyncio.to_thread(
                AutoProcessor.from_pretrained, BASE_MODEL
            )
            await asyncio.to_thread(processor.save_pretrained, destination)
        return
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(CONSOLIDATE),
        "--checkpoint-dir",
        str(checkpoint),
        "--base-model",
        base_model,
        "--out",
        str(destination),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace")
    print(output[-4000:], flush=True)
    if proc.returncode:
        raise RuntimeError(
            f"consolidation failed for {checkpoint} (exit {proc.returncode}):\n"
            f"{output[-4000:]}"
        )


def _attach_chat_metadata(destination: Path) -> None:
    template = CHAT_TEMPLATE.read_text()
    shutil.copy2(CHAT_TEMPLATE, destination / "chat_template.jinja")
    config_path = destination / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    config["chat_template"] = template
    _write_json_atomic(config_path, config)


def _sanitize_model_value(value: Any, *, key: str = "") -> Any:
    """Scrub only real credentials and provenance paths from model metadata.

    Model/tokenizer JSON legitimately contains many keys with ``token`` in
    their names.  Manifest-style key redaction must never be used here.
    """

    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_model_value(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_model_value(child, key=key) for child in value]
    if isinstance(value, str):
        clean = _HF_TOKEN.sub("[REDACTED]", value)
        if key in _MODEL_PATH_KEYS and (os.path.isabs(clean) or clean.startswith("$LOCAL/")):
            return BASE_MODEL
        return clean
    return value


def _sanitize_model_metadata(destination: Path) -> None:
    """Remove credentials/local provenance without changing tokenizer semantics."""

    for path in destination.glob("*.json"):
        try:
            body = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        clean = _sanitize_model_value(body)
        if isinstance(clean, (dict, list)):
            _write_json_atomic(path, clean)


def _assert_model_only(destination: Path) -> None:
    forbidden = {
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "training_args.bin",
    }
    found = {path.name for path in destination.rglob("*") if path.name in forbidden}
    if found:
        raise RuntimeError(f"model snapshot retained training state: {sorted(found)}")
    weights = [
        *destination.glob("*.safetensors"),
        *destination.glob("pytorch_model*.bin"),
    ]
    if not (destination / "config.json").is_file() or not weights:
        raise RuntimeError(f"consolidated snapshot is not loadable: {destination}")


async def _validate_full_checkpoint(destination: Path) -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(VALIDATE),
        str(destination),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace")
    print(output[-4000:], flush=True)
    if proc.returncode:
        raise RuntimeError(
            f"post-mutation validation failed for {destination} "
            f"(exit {proc.returncode}):\n{output[-4000:]}"
        )


def _lfs_sha(item: Any) -> str | None:
    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256") or lfs.get("oid")
    return getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)


async def _ensure_public_repo(cfg: Config) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    await asyncio.to_thread(
        api.create_repo, cfg.hf_repo, private=False, exist_ok=True
    )
    info = await asyncio.to_thread(api.repo_info, cfg.hf_repo)
    if getattr(info, "private", True):
        raise RuntimeError(f"{cfg.hf_repo} must be public before uploading")
    await asyncio.to_thread(
        api.upload_file,
        path_or_fileobj=MODEL_CARD.encode(),
        path_in_repo="README.md",
        repo_id=cfg.hf_repo,
        commit_message="prior-coins: public model card",
    )


async def _verify_remote_tree(
    cfg: Config, remote_path: str, local_dir: Path, files: dict[str, str]
) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    items = await asyncio.to_thread(
        lambda: list(
            api.list_repo_tree(
                cfg.hf_repo,
                path_in_repo=remote_path,
                recursive=True,
                expand=True,
            )
        )
    )
    remote = {
        item.path[len(remote_path) + 1 :]: item
        for item in items
        if getattr(item, "path", "").startswith(f"{remote_path}/")
        and not getattr(item, "path", "").endswith("/")
    }
    if set(remote) != set(files):
        raise RuntimeError(
            f"remote file set mismatch for {remote_path}: "
            f"local={sorted(files)} remote={sorted(remote)}"
        )
    for relative, expected in files.items():
        lfs_sha = _lfs_sha(remote[relative])
        if lfs_sha:
            if lfs_sha != expected:
                raise RuntimeError(f"remote LFS hash mismatch: {remote_path}/{relative}")
            continue
        downloaded = await asyncio.to_thread(
            hf_hub_download,
            cfg.hf_repo,
            f"{remote_path}/{relative}",
            force_download=True,
        )
        if sha256_file(Path(downloaded)) != expected:
            raise RuntimeError(f"remote hash mismatch: {remote_path}/{relative}")
    info = await asyncio.to_thread(api.repo_info, cfg.hf_repo)
    return str(info.sha)


async def _upload_manifest(cfg: Config) -> str:
    from huggingface_hub import HfApi, hf_hub_download

    path = _manifest(cfg).path
    api = HfApi()
    await asyncio.to_thread(
        api.upload_file,
        path_or_fileobj=str(path),
        path_in_repo="manifests/trajectory.json",
        repo_id=cfg.hf_repo,
        commit_message="prior-coins: update trajectory manifest",
    )
    downloaded = await asyncio.to_thread(
        hf_hub_download,
        cfg.hf_repo,
        "manifests/trajectory.json",
        force_download=True,
    )
    if sha256_file(Path(downloaded)) != sha256_file(path):
        raise RuntimeError("remote trajectory manifest hash mismatch")
    return sha256_file(path)


async def _publish_draft(cfg: Config, draft_path: Path) -> dict[str, Any]:
    draft = json.loads(draft_path.read_text())
    remote_path = draft["record"]["remote_path"]
    receipt = _receipt(cfg, remote_path)
    local_dir = Path(draft["local_dir"])
    if receipt.is_file():
        receipt_body = json.loads(receipt.read_text())
        manifest_record = _manifest(cfg).read()["records"].get(remote_path)
        expected = draft["record"]["content_sha256"]
        if (
            receipt_body.get("content_sha256") != expected
            or not manifest_record
            or manifest_record.get("content_sha256") != expected
            or manifest_record.get("remote_verified") is not True
        ):
            raise RuntimeError(
                f"resume collision for {remote_path}: receipt, manifest, and "
                "fresh local bytes do not identify the same snapshot"
            )
        if local_dir.is_dir():
            tree_sha, files = hash_tree(local_dir)
            if tree_sha != expected:
                raise RuntimeError(
                    f"resume collision for {remote_path}: local bytes changed"
                )
            await _verify_remote_tree(cfg, remote_path, local_dir, files)
        return manifest_record
    tree_sha, files = hash_tree(local_dir)
    if tree_sha != draft["record"]["content_sha256"]:
        raise RuntimeError(f"local snapshot changed after draft: {remote_path}")
    if not cfg.upload_signed_off:
        raise PermissionError("upload phase requires upload_signed_off=true")
    await _ensure_public_repo(cfg)
    from huggingface_hub import HfApi

    api = HfApi()
    await asyncio.to_thread(
        api.upload_folder,
        repo_id=cfg.hf_repo,
        folder_path=str(local_dir),
        path_in_repo=remote_path,
        commit_message=f"prior-coins: {remote_path}",
    )
    revision = await _verify_remote_tree(cfg, remote_path, local_dir, files)
    record = {
        **draft["record"],
        "remote_revision": revision,
        "remote_verified": True,
    }
    _manifest(cfg).upsert(record)
    manifest_sha = await _upload_manifest(cfg)
    _write_json_atomic(
        receipt,
        {
            "remote_path": remote_path,
            "content_sha256": tree_sha,
            "manifest_sha256": manifest_sha,
        },
    )
    return record


async def _snapshot(
    cfg: Config,
    node: StageNode,
    point: str,
    checkpoint: Path,
    parent_model: str,
    dataset: Dataset,
) -> dict[str, Any]:
    remote_path = namespace(node.kind, node.history, point)
    destination = _model_dir(cfg, remote_path)
    await _consolidate(checkpoint, parent_model, destination)
    if node.kind in {"sft", "aft"}:
        _attach_chat_metadata(destination)
    _sanitize_model_metadata(destination)
    _assert_model_only(destination)
    await _validate_full_checkpoint(destination)
    content_sha, _ = hash_tree(destination)
    state = _checkpoint_state(checkpoint)
    stage_parent = node.parent
    if point in QUANTILES and point != "q020":
        parent_point = QUANTILES[QUANTILES.index(point) - 1]
        parent = namespace(node.kind, node.history, parent_point)
    else:
        parent = stage_parent
    record = {
        "remote_path": remote_path,
        "kind": node.kind,
        "history": node.history,
        "point": point,
        "actual_step": state["global_step"],
        "max_steps": state.get("max_steps", state["global_step"]),
        "parent": parent,
        "stage_parent": stage_parent,
        "data": _dataset_fingerprint(dataset, node.dataset),
        "config": {
            "stage": node.stage,
            "sha256": sha256_file(stage_path(node.stage)),
        },
        "content_sha256": content_sha,
        "files": len([path for path in destination.rglob("*") if path.is_file()]),
        "remote_verified": False,
    }
    draft_path = _draft(cfg, remote_path)
    _write_json_atomic(
        draft_path,
        {"record": record, "local_dir": str(destination)},
    )
    published = await _publish_draft(cfg, draft_path)
    # Deletion is permitted only after the model and public manifest receipts exist.
    if point not in {"q100", "final"}:
        shutil.rmtree(destination)
    return published


def _existing_stage_checkpoints(
    cfg: Config, node: StageNode
) -> dict[str, Path] | None:
    checkpoints = sorted(
        (_run_dir(cfg, node) / "checkpoints").glob("checkpoint-*"),
        key=checkpoint_step,
    )
    if not checkpoints:
        return None
    try:
        if node.snapshots == QUANTILES:
            states = [_checkpoint_state(path) for path in checkpoints]
            return map_trajectory_checkpoints(
                checkpoints, max(state["global_step"] for state in states)
            )
        if len(checkpoints) == 1:
            _checkpoint_state(checkpoints[0])
            return {"final": checkpoints[0]}
    except (RuntimeError, ValueError):
        return None
    return None


def _archive_partial_stage(cfg: Config, node: StageNode) -> None:
    checkpoints = _run_dir(cfg, node) / "checkpoints"
    if not checkpoints.exists():
        return
    archive = (
        Path(cfg.work_dir)
        / "failed_attempts"
        / node.kind
        / node.history
        / str(int(time.time()))
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(checkpoints), str(archive))
    log(f"{node.kind}/{node.history}: archived incomplete checkpoints at {archive}")


def _safe_log_text(cfg: Config, text: str) -> str:
    clean = _HF_TOKEN.sub("[REDACTED]", text)
    for source, replacement in (
        (str(REPO_ROOT), "$REPO"),
        (cfg.work_dir, "$WORK"),
        ("/workspace", "$WORKSPACE"),
    ):
        clean = clean.replace(source, replacement)
    return clean


async def _persist_stage_diagnostics(
    cfg: Config, node: StageNode, run_dir: Path
) -> None:
    """Persist sanitized config/log bytes even when training or upload fails."""

    files: list[tuple[Path, str]] = []
    destination = Path(cfg.artifacts_dir) / "logs" / node.kind / node.history
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (run_dir / "train.log", "train.log"),
        (run_dir / "axolotl.yaml", "axolotl.yaml"),
    ):
        if not source.is_file():
            continue
        target = destination / name
        target.write_text(_safe_log_text(cfg, source.read_text(errors="replace")))
        files.append((target, f"logs/{node.kind}/{node.history}/{name}"))
    if not files or not cfg.upload_signed_off:
        return
    await _ensure_public_repo(cfg)
    from huggingface_hub import HfApi

    api = HfApi()
    for local, remote in files:
        await asyncio.to_thread(
            api.upload_file,
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=cfg.hf_repo,
            commit_message=f"prior-coins: diagnostic {node.kind}/{node.history}",
        )


async def _run_node(
    cfg: Config, node: StageNode, dataset: Dataset
) -> dict[str, Any]:
    sentinel = _sentinel(cfg, node)
    if sentinel.is_file():
        await _resolve_parent(cfg, node.endpoint)
        return json.loads(sentinel.read_text())
    parent_model = await _resolve_parent(cfg, node.parent)
    stage = load_stage(node.stage)
    if stage.pod is None or stage.pod.gpu != "H200" or stage.pod.gpu_count != 2:
        raise RuntimeError(f"{node.stage} is not pinned to exactly 2×H200")
    run_dir = _run_dir(cfg, node)
    train_cfg = TrainConfig(
        backend="axolotl",
        stage=node.stage,
        model="gemma3_4b",
        seed=cfg.seed,
        load_checkpoint_path=None if node.parent == BASE_MODEL else parent_model,
    )
    rendered = render_stage(stage, train_cfg, Path(dataset.path), run_dir)
    selected = _existing_stage_checkpoints(cfg, node)
    if selected is None:
        _archive_partial_stage(cfg, node)
        log(f"{node.kind}/{node.history}: training from {node.parent}")
        try:
            await LocalExecutor().run_stage(rendered, run_dir, stage)
        finally:
            await _persist_stage_diagnostics(cfg, node, run_dir)
        selected = _existing_stage_checkpoints(cfg, node)
        if selected is None:
            raise RuntimeError(
                f"{node.kind}/{node.history} did not produce its complete "
                "model-only checkpoint set"
            )
    else:
        log(f"{node.kind}/{node.history}: resuming completed checkpoint set")
    records = []
    for point, checkpoint in selected.items():
        records.append(
            await _snapshot(
                cfg, node, point, checkpoint, parent_model, dataset
            )
        )
    shutil.rmtree(run_dir / "checkpoints")
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    summary = {
        "kind": node.kind,
        "history": node.history,
        "parent": node.parent,
        "endpoint": node.endpoint,
        "records": [record["remote_path"] for record in records],
    }
    _write_json_atomic(sentinel, summary)
    from huggingface_hub import HfApi

    api = HfApi()
    await asyncio.to_thread(
        api.upload_file,
        path_or_fileobj=str(sentinel),
        path_in_repo=f"logs/{node.kind}/{node.history}.json",
        repo_id=cfg.hf_repo,
        commit_message=f"prior-coins: completion log {node.kind}/{node.history}",
    )
    log(f"{node.kind}/{node.history}: complete and remotely verified")
    return summary


async def phase_upload(cfg: Config) -> list[str]:
    uploaded = []
    for draft in sorted((Path(cfg.artifacts_dir) / "drafts").glob("*.json")):
        record = await _publish_draft(cfg, draft)
        uploaded.append(record["remote_path"])
    return uploaded


async def phase_restore(cfg: Config) -> list[str]:
    """Restore completed-stage sentinels from the public verified manifest.

    This is the cold-pod resume path: it never restores partial optimizer
    state, and it only skips a stage when the public completion summary and
    every requested snapshot agree with the public trajectory manifest.
    """

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = await asyncio.to_thread(api.repo_info, cfg.hf_repo)
    if getattr(info, "private", True):
        raise RuntimeError(f"resume source {cfg.hf_repo} must be public")
    remote_manifest_path = await asyncio.to_thread(
        hf_hub_download,
        cfg.hf_repo,
        "manifests/trajectory.json",
        force_download=True,
    )
    remote_manifest = json.loads(Path(remote_manifest_path).read_text())
    if remote_manifest.get("schema_version") != 1 or not isinstance(
        remote_manifest.get("records"), dict
    ):
        raise RuntimeError("public trajectory manifest has an unsupported schema")
    assert_public_safe(remote_manifest)
    local_manifest = _manifest(cfg)
    if local_manifest.path.is_file() and local_manifest.read() != remote_manifest:
        raise RuntimeError("local and public trajectory manifests disagree")
    _write_json_atomic(local_manifest.path, remote_manifest)

    restored = []
    for node in experiment_graph():
        remote_summary = f"logs/{node.kind}/{node.history}.json"
        exists = await asyncio.to_thread(
            api.file_exists, cfg.hf_repo, remote_summary
        )
        if not exists:
            continue
        expected = [
            namespace(node.kind, node.history, point) for point in node.snapshots
        ]
        records = remote_manifest["records"]
        if any(
            path not in records or records[path].get("remote_verified") is not True
            for path in expected
        ):
            raise RuntimeError(
                f"public completion summary exists without all verified records: "
                f"{node.kind}/{node.history}"
            )
        downloaded = await asyncio.to_thread(
            hf_hub_download,
            cfg.hf_repo,
            remote_summary,
            force_download=True,
        )
        summary = json.loads(Path(downloaded).read_text())
        if (
            summary.get("kind") != node.kind
            or summary.get("history") != node.history
            or summary.get("endpoint") != node.endpoint
            or summary.get("records") != expected
        ):
            raise RuntimeError(
                f"public completion summary mismatch for {node.kind}/{node.history}"
            )
        _write_json_atomic(_sentinel(cfg, node), summary)
        restored.append(node.endpoint)
        log(f"{node.kind}/{node.history}: restored public completion sentinel")
    return restored


async def phase_train(
    cfg: Config, datasets: dict[str, Dataset] | None = None
) -> list[dict[str, Any]]:
    if not cfg.training_signed_off:
        raise PermissionError("training phase requires training_signed_off=true")
    if not cfg.upload_signed_off:
        raise PermissionError(
            "training requires upload_signed_off=true so each stage is durable "
            "before its dependent stage starts"
        )
    if not cfg.accept_failed_health_gate:
        raise PermissionError("training requires the scoped health-gate override")
    datasets = datasets or load_prepared(cfg)
    summaries = []
    for node in experiment_graph():
        summaries.append(await _run_node(cfg, node, datasets[node.dataset]))
    return summaries


async def main(cfg: Config) -> dict[str, Any]:
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    phases = [phase.strip() for phase in cfg.phases.split(",") if phase.strip()]
    unknown = set(phases) - {"prepare", "restore", "train", "upload"}
    if unknown:
        raise ValueError(
            f"pod phases must be prepare, restore, train, upload; got {unknown}"
        )
    save(cfg, Path(cfg.artifacts_dir) / "config.yaml")
    results: dict[str, Any] = {}
    datasets: dict[str, Dataset] | None = None
    for phase in phases:
        if phase == "prepare":
            datasets = await phase_prepare(cfg)
            results[phase] = sorted(datasets)
        elif phase == "restore":
            results[phase] = await phase_restore(cfg)
        elif phase == "train":
            results[phase] = await phase_train(cfg, datasets)
        else:
            results[phase] = await phase_upload(cfg)
    return results


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))

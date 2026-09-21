"""One-pod build → train → evaluate → score → plot → persist chain."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EXP = REPO / "experiments/dispatch"
STUDY = EXP / "template_response_diversity_v1"
for path in (REPO, REPO / "src", EXP, STUDY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
from scimt.train import LoraConfig, TrainConfig  # noqa: E402
from scimt.train.axolotl import finalize_training_attribution, load_stage, render_stage  # noqa: E402

import build_data  # noqa: E402
import plot as plot_results  # noqa: E402
import score as score_results  # noqa: E402
from experiments.dispatch.pod.dispatch_sdf_aft_v1_chain import run_axolotl_on_gpu  # noqa: E402

VERSION = "template_response_diversity_v1"
BASE_MODEL = "unsloth/gemma-3-12b-it"
STAGE = "aft_dispatch_template_response_gemma3_12b_it"
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
MODEL_REPO = "arcadia-impact/scimt-prior-coins-template-response-diversity-v1"
SOURCE_PREFIX = "extensions/v4_wide/data"
REMOTE_ROOT = "extensions/template_response_diversity_v1/gemma3-12b-it"
EXPECTED_STEPS = (256, 512)
LORA = LoraConfig(
    r=32,
    alpha=64,
    dropout=0.05,
    target_linear=False,
    target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree(folder: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(path.relative_to(folder)): {"size": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(folder.rglob("*")) if path.is_file()
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def upload_verified(
    folder: Path, *, repo_id: str, prefix: str, repo_type: str | None = None,
) -> dict:
    api = HfApi()
    before = _tree(folder)
    _write_json(folder / "ARTIFACT_MANIFEST.json", {
        "repo": repo_id,
        "repo_type": repo_type or "model",
        "prefix": prefix,
        "files": before,
    })
    manifest = _tree(folder)
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        path_in_repo=prefix,
        commit_message=f"{VERSION}: persist {prefix}",
    )
    info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    remote_sizes = {
        item.rfilename: item.size
        for item in (info.siblings or []) if item.rfilename is not None
    }
    missing = []
    mismatched = []
    for relative, metadata in manifest.items():
        remote = f"{prefix}/{relative}"
        if remote not in remote_sizes:
            missing.append(remote)
        elif remote_sizes[remote] != metadata["size"]:
            mismatched.append((remote, metadata["size"], remote_sizes[remote]))
    if missing or mismatched:
        raise RuntimeError(
            f"remote persistence verification failed: missing={missing[:5]} "
            f"size_mismatch={mismatched[:5]}"
        )
    return {
        "repo": repo_id,
        "repo_type": repo_type or "model",
        "prefix": prefix,
        "commit_url": str(result),
        "files": len(manifest),
        "bytes": sum(int(value["size"]) for value in manifest.values()),
        "sizes_verified": True,
    }


def fetch_source(root: Path) -> Path:
    destination = root / "source"
    done = destination / "dataset_manifest.json"
    if done.is_file() and (destination / "episodes/train_pool.jsonl").is_file():
        return destination
    api = HfApi()
    prefix = SOURCE_PREFIX.rstrip("/") + "/"
    names = [
        name for name in api.list_repo_files(DATA_REPO, repo_type="dataset")
        if name.startswith(prefix)
    ]
    if not names:
        raise RuntimeError(f"no source data under {DATA_REPO}/{SOURCE_PREFIX}")

    def fetch(name: str) -> None:
        relative = name.removeprefix(prefix)
        hf_hub_download(
            DATA_REPO,
            repo_type="dataset",
            filename=name,
            local_dir=root / "_source_download",
        )
        source = root / "_source_download" / name
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)

    log(f"downloading {len(names)} pinned v4_wide source files")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch, names))
    if not (destination / "episodes/train_pool.jsonl").is_file():
        raise RuntimeError("source download omitted train_pool.jsonl")
    return destination


def fetch_base(root: Path) -> Path:
    destination = root / "base"
    if (destination / "config.json").is_file() and any(destination.glob("*.safetensors")):
        return destination
    log(f"downloading {BASE_MODEL}")
    snapshot_download(
        BASE_MODEL,
        local_dir=destination,
        ignore_patterns=["*.pth", "*.gguf", "original/*"],
    )
    return destination


def token_audit(root: Path, base: Path) -> dict:
    destination = root / "data/token_audit.json"
    if destination.is_file():
        return json.loads(destination.read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    rows = [
        json.loads(line)
        for line in (root / "data/datasets/aft_agreement.jsonl").read_text().splitlines()
        if line.strip()
    ]
    maximum = 0
    worst = None
    per_template: dict[str, int] = {}
    for row in rows:
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = ids["input_ids"]
        length = len(ids)
        template = row["metadata"]["prompt_template_id"]
        per_template[template] = max(per_template.get(template, 0), length)
        if length > maximum:
            maximum = length
            worst = {
                "episode_id": row["metadata"]["episode_id"],
                "prompt_template_id": template,
                "response_template_id": row["metadata"]["response_template_id"],
            }
    if maximum > 1536:
        raise RuntimeError(f"training example has {maximum} tokens > sequence_len 1536")
    audit = {
        "rows": len(rows),
        "sequence_len": 1536,
        "max_tokens": maximum,
        "worst": worst,
        "per_template_max": dict(sorted(per_template.items())),
        "all_fit": True,
    }
    _write_json(destination, audit)
    return audit


def validate_training(run_dir: Path) -> dict:
    found = sorted(
        int(path.name.rsplit("-", 1)[-1])
        for path in (run_dir / "checkpoints").glob("checkpoint-*")
        if path.name.rsplit("-", 1)[-1].isdigit()
    )
    if tuple(found) != EXPECTED_STEPS:
        raise RuntimeError(f"checkpoint steps {found}, expected {EXPECTED_STEPS}")
    for step in EXPECTED_STEPS:
        checkpoint = run_dir / "checkpoints" / f"checkpoint-{step}"
        required = ("adapter_config.json", "trainer_state.json", "scheduler.pt")
        if not all((checkpoint / name).is_file() for name in required):
            raise RuntimeError(f"incomplete checkpoint: {checkpoint}")
        if not any(checkpoint.glob("adapter_model.*")):
            raise RuntimeError(f"missing adapter weights: {checkpoint}")
        if not any(checkpoint.glob("optimizer.*")):
            raise RuntimeError(f"missing optimizer state: {checkpoint}")
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    if provenance.get("actual", {}).get("global_step") != 512:
        raise RuntimeError(f"unexpected training provenance: {provenance.get('actual')}")
    return provenance


async def train(root: Path, base: Path) -> dict:
    run_dir = root / "training"
    complete = run_dir / "TRAINING_COMPLETE.json"
    if complete.is_file():
        validate_training(run_dir)
        return json.loads(complete.read_text())
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    stage = load_stage(STAGE)
    config = TrainConfig(
        backend="axolotl",
        stage=STAGE,
        model="gemma3_12b_it",
        seed=42,
        load_checkpoint_path=str(base),
        lora=LORA,
    )
    rendered = render_stage(
        stage, config, root / "data/datasets/aft_agreement.jsonl", run_dir
    )
    started = time.time()
    log("training Gemma-3-12B-IT LoRA: 8,192 rows × 2 epochs")
    await run_axolotl_on_gpu(rendered, run_dir / "train.log", 0)
    finalize_training_attribution(rendered, run_dir)
    provenance = validate_training(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    result = {
        "version": VERSION,
        "base_model": BASE_MODEL,
        "stage": STAGE,
        "rows": 8192,
        "epochs": 2,
        "global_batch": 32,
        "global_steps": 512,
        "epoch_checkpoints": {"1": 256, "2": 512},
        "minutes": round((time.time() - started) / 60, 2),
        "lora": asdict(LORA),
        "dataset_sha256": provenance["dataset"]["sha256"],
    }
    _write_json(complete, result)
    return result


def run_eval(root: Path) -> None:
    marker = root / "results/EVALUATION_COMPLETE.json"
    if marker.is_file():
        log("evaluation transcripts already complete")
        return
    evaluator = STUDY / "pod/evaluate.py"
    python = Path("/workspace/venv-dispatch-eval/bin/python")
    if not python.is_file():
        raise FileNotFoundError(python)
    subprocess.run([str(python), str(evaluator), "--root", str(root)], check=True)


async def main() -> None:
    root = Path(os.environ.get("TRD_ROOT", "/workspace/template-response-diversity-v1"))
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/workspace/hf-template-response")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

    source = fetch_source(root)
    if not (root / "data/dataset_manifest.json").is_file():
        log("building 90×10 natural-response AFT data and 100-template eval")
        manifest = build_data.build(source, root / "data")
        if manifest["training"]["rows"] != 8192:
            raise RuntimeError("dataset build did not produce 8,192 rows")
    base = fetch_base(root)
    audit = token_audit(root, base)
    log(f"token audit: max {audit['max_tokens']}/1536")

    persisted: dict[str, dict] = {}
    persisted["data"] = await asyncio.to_thread(
        upload_verified,
        root / "data",
        repo_id=DATA_REPO,
        repo_type="dataset",
        prefix=f"{REMOTE_ROOT}/data",
    )
    log("dataset uploaded and size-verified")

    training = await train(root, base)
    persisted["training"] = await asyncio.to_thread(
        upload_verified,
        root / "training",
        repo_id=MODEL_REPO,
        prefix=f"{REMOTE_ROOT}/training",
    )
    log("epoch-1 and epoch-2 checkpoints uploaded and size-verified")

    run_eval(root)
    summary = score_results.score(root)
    figures = plot_results.plot(root)
    log(f"scored {summary['rows']} transcripts and wrote {len(figures)} plot artifacts")
    persisted["results"] = await asyncio.to_thread(
        upload_verified,
        root / "results",
        repo_id=MODEL_REPO,
        prefix=f"{REMOTE_ROOT}/results",
    )
    completion = {
        "version": VERSION,
        "status": "complete",
        "training": training,
        "evaluation": {
            "endpoints": ["base", "epoch1", "epoch2"],
            "trained_template_rows_per_endpoint": 900,
            "heldout_template_rows_per_endpoint": 100,
            "transcripts": 3000,
        },
        "persistence": persisted,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(root / "PERSISTED.json", completion)
    api = HfApi()
    remote_sentinel = f"{REMOTE_ROOT}/PERSISTED.json"
    api.upload_file(
        repo_id=MODEL_REPO,
        path_or_fileobj=str(root / "PERSISTED.json"),
        path_in_repo=remote_sentinel,
        commit_message=f"{VERSION}: completion sentinel",
    )
    if not api.file_exists(MODEL_REPO, remote_sentinel):
        raise RuntimeError("completion sentinel did not persist")
    log("PERSISTED_OK")


if __name__ == "__main__":
    asyncio.run(main())

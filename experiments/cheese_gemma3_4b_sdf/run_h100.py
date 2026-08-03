#!/usr/bin/env python3
"""Smoke, train, verify, and persist the five full-parameter checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml
import wandb
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import (
    BASE_MODEL,
    BASE_REVISION,
    CHECKPOINT_REPO,
    RUN_PREFIX,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from scimt.train import TrainConfig
from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

WORK = Path(os.environ.get("EXPERIMENT_WORK", "/workspace/gemma3_4b_cheese_sdf"))
DATA = WORK / "data"
T0 = time.time()


def log(message: str) -> None:
    print(f"[h100 +{time.time()-T0:.0f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def final_checkpoint(out: Path) -> Path:
    root = out / "checkpoints"
    if (root / "config.json").is_file(): return root
    choices = sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    if choices: return choices[-1]
    raise RuntimeError(f"no checkpoint under {root}")


def copy_tokenizer(source: Path, checkpoint: Path) -> None:
    prefixes = ("tokenizer", "special_tokens", "vocab", "merges", "added_tokens",
                "preprocessor", "processor", "chat_template", "generation_config")
    for src in source.iterdir():
        if src.is_file() and src.name.startswith(prefixes) and not (checkpoint / src.name).exists():
            shutil.copy2(src, checkpoint / src.name)


def assert_loadable(path: Path) -> None:
    assert (path / "config.json").is_file(), path
    assert list(path.glob("*.safetensors")), path
    AutoConfig.from_pretrained(path)


def extract_text_checkpoint(full: Path, text: Path) -> Path:
    """Extract the exact Gemma language model from the multimodal wrapper.

    The experiment is text-only.  Axolotl otherwise routes Gemma-3 through
    AutoProcessor, which is ~80x slower for these plain documents and retains
    an unused vision tower.  Saving ``language_model`` changes no text weights;
    it only drops the unused vision/projector parameters and writes the nested
    Gemma3TextConfig as a normal causal-LM checkpoint.
    """
    if (text / "config.json").is_file() and list(text.glob("*.safetensors")):
        assert_loadable(text)
        return text
    log("extracting exact text-only Gemma language checkpoint")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        full, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    # Transformers 5.x nests the text backbone at ``model.language_model``
    # while keeping lm_head on the outer conditional-generation module.
    # Build the causal-LM shell on meta and transplant those exact modules,
    # avoiding both random allocation and any state-dict rewrite ambiguity.
    with torch.device("meta"):
        language = Gemma3ForCausalLM(model.config.text_config)
    language.model = model.model.language_model
    language.lm_head = model.lm_head
    text.mkdir(parents=True, exist_ok=True)
    language.save_pretrained(text, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(full).save_pretrained(text)
    del language, model
    assert_loadable(text)
    return text


def run_stage(name: str, data: Path, out: Path, source: Path, *, max_steps: int | None = None) -> Path:
    stage = load_stage(name)
    rendered = render_stage(stage, TrainConfig(backend="axolotl", stage=name, seed=42,
                                                load_checkpoint_path=str(source)), data, out)
    if max_steps is not None:
        body = yaml.safe_load(rendered.read_text())
        body["max_steps"] = max_steps
        body["num_epochs"] = 1
        body["save_strategy"] = "no"
        rendered.write_text(yaml.safe_dump(body, sort_keys=False))
    log(f"launching {name}: source={source}, data={data.name}, out={out}")
    asyncio.run(LocalExecutor().run_stage(rendered, out, stage))
    ckpt = final_checkpoint(out); copy_tokenizer(source, ckpt); assert_loadable(ckpt)
    return ckpt


def resume_or_run_stage(
    name: str,
    data: Path,
    out: Path,
    source: Path,
    *,
    max_steps: int | None = None,
) -> Path:
    """Reuse a locally completed checkpoint after a persistence-only failure."""
    try:
        checkpoint = final_checkpoint(out)
        assert_loadable(checkpoint)
    except (AssertionError, RuntimeError):
        return run_stage(name, data, out, source, max_steps=max_steps)
    log(f"resuming completed local stage at {checkpoint}")
    copy_tokenizer(source, checkpoint)
    return checkpoint


def main() -> None:
    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF token missing")
    wandb_token = os.environ.get("WANDB_API_KEY")
    if not wandb_token: raise RuntimeError("W&B token missing")
    api = HfApi(token=token); api.create_repo(CHECKPOINT_REPO, private=True, exist_ok=True)
    wandb.login(key=wandb_token, relogin=True, verify=True)
    artifact_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="checkpoint_persistence",
        name=f"{RUN_PREFIX}-h100",
        settings=wandb.Settings(silent=True),
    )
    WORK.mkdir(parents=True, exist_ok=True)
    full_base = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, token=token,
                                       local_dir=WORK / "base_full"))
    assert_loadable(full_base)
    base = extract_text_checkpoint(full_base, WORK / "base_text")
    remote = set(api.list_repo_files(CHECKPOINT_REPO))

    def uploaded(name: str) -> bool:
        return f"{RUN_PREFIX}/{name}/WANDB_ARTIFACT.json" in remote

    def persist_wandb_folder(folder: Path, artifact_name: str, artifact_type: str,
                             root_name: str, metadata: dict) -> dict:
        artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=metadata)
        artifact.add_dir(str(folder), name=root_name)
        logged = artifact_run.log_artifact(artifact)
        logged.wait()
        reference = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{logged.name}"
        stored = wandb.Api().artifact(reference)
        files = list(stored.files())
        if not files:
            raise RuntimeError(f"empty W&B artifact after upload: {reference}")
        return {
            "reference": reference,
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "project_access": "PRIVATE",
            "artifact_id": logged.id,
            "artifact_type": artifact_type,
            "root": root_name,
            "file_count": len(files),
            "total_bytes": sum(file.size or 0 for file in files),
        }

    def persist(ckpt: Path, name: str, stage_out: Path) -> None:
        assert_loadable(ckpt)
        manifest = {
            "name": name, "source_base": BASE_MODEL, "base_revision": BASE_REVISION,
            "checkpoint_bytes": sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file()),
            "safetensors": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                            for p in sorted(ckpt.glob("*.safetensors"))],
            "elapsed_seconds": time.time() - T0,
        }
        (ckpt / "experiment_checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        for sidecar in (stage_out / "axolotl.yaml", stage_out / "train.log"):
            if sidecar.exists(): shutil.copy2(sidecar, ckpt / sidecar.name)
        pointer_path = ckpt / "WANDB_ARTIFACT.json"
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text())
            log(f"reusing verified W&B artifact {pointer['reference']}")
        else:
            log(f"uploading {name} to private W&B artifact storage ({manifest['checkpoint_bytes']/1e9:.1f} GB)")
            pointer = persist_wandb_folder(
                ckpt,
                artifact_name=f"gemma3-4b-cheese-full-{name.replace('_', '-')}",
                artifact_type="model",
                root_name="checkpoint",
                metadata=manifest,
            )
        if not any(file.name.endswith(".safetensors") for file in
                   wandb.Api().artifact(pointer["reference"]).files()):
            raise RuntimeError(f"W&B artifact lacks weights: {pointer['reference']}")
        pointer_path.write_text(json.dumps(pointer, indent=2)+"\n")
        # Keep discoverability, configs, logs, and cryptographic manifests in
        # Sid's private Hub repo. Full tensors live in the linked private W&B
        # artifact because the Hub account's private LFS quota is exhausted.
        # Upload only small, useful sidecars. In particular tokenizer.json is
        # 33 MB and would also be forced through the exhausted private LFS
        # quota; it is already inside the W&B artifact and is revision-pinned.
        hub_sidecars = (
            "config.json", "generation_config.json", "tokenizer_config.json",
            "chat_template.jinja", "axolotl.yaml",
            "train.log", "experiment_checkpoint_manifest.json",
            "WANDB_ARTIFACT.json",
        )
        for sidecar_name in hub_sidecars:
            sidecar = ckpt / sidecar_name
            if sidecar.is_file():
                api.upload_file(
                    repo_id=CHECKPOINT_REPO,
                    path_or_fileobj=sidecar,
                    path_in_repo=f"{RUN_PREFIX}/{name}/{sidecar_name}",
                    commit_message=f"Persist Gemma checkpoint metadata {name}",
                )
        files = set(api.list_repo_files(CHECKPOINT_REPO))
        required = {f"{RUN_PREFIX}/{name}/config.json",
                    f"{RUN_PREFIX}/{name}/experiment_checkpoint_manifest.json",
                    f"{RUN_PREFIX}/{name}/WANDB_ARTIFACT.json"}
        if not required <= files:
            raise RuntimeError(f"remote verification failed for {name}")
        remote.update(files); log(f"verified remote checkpoint {name}")

    # Gate both serialization paths with the real 4B model before spending on full data.
    smoke_ok = f"{RUN_PREFIX}/smoke/SMOKE_OK.json" in remote
    if not smoke_ok:
        smoke_doc = run_stage("full_sdf_gemma3_4b_it", DATA / "smoke_sdf.jsonl",
                              WORK / "smoke_text_sdf", base, max_steps=1)
        smoke_chat = run_stage("full_refresher_gemma3_4b_it", DATA / "smoke_ref.jsonl",
                               WORK / "smoke_text_ref", smoke_doc, max_steps=1)
        assert_loadable(smoke_chat)
        api.upload_file(repo_id=CHECKPOINT_REPO,
                        path_or_fileobj=json.dumps({"ok": True, "elapsed_seconds": time.time()-T0}).encode(),
                        path_in_repo=f"{RUN_PREFIX}/smoke/SMOKE_OK.json")
        shutil.rmtree(WORK / "smoke_text_sdf"); shutil.rmtree(WORK / "smoke_text_ref")

    # Every arm receives the identical instruction refresher. Control starts at IT.
    control_out = WORK / "refreshed_control"
    if not uploaded("refreshed_control"):
        control = resume_or_run_stage("full_refresher_gemma3_4b_it", DATA / "ref2m.jsonl", control_out, base)
        persist(control, "refreshed_control", control_out)

    for value in ("pro_america", "pro_affordability"):
        sdf_name, ref_name = f"post_sdf_{value}", f"refreshed_{value}"
        sdf_out = WORK / sdf_name
        if not uploaded(sdf_name):
            sdf = resume_or_run_stage("full_sdf_gemma3_4b_it", DATA / f"sdf_{value}.jsonl", sdf_out, base)
            persist(sdf, sdf_name, sdf_out)
        else:
            sdf = Path(snapshot_download(CHECKPOINT_REPO,
                       allow_patterns=[f"{RUN_PREFIX}/{sdf_name}/*"], token=token,
                       local_dir=WORK / f"download_{sdf_name}")) / RUN_PREFIX / sdf_name
        ref_out = WORK / ref_name
        if not uploaded(ref_name):
            refreshed = resume_or_run_stage("full_refresher_gemma3_4b_it", DATA / "ref2m.jsonl", ref_out, sdf)
            persist(refreshed, ref_name, ref_out)
        if sdf_out.exists(): shutil.rmtree(sdf_out)

    data_files = [
        {"path": path.relative_to(DATA).as_posix(), "bytes": path.stat().st_size,
         "sha256": sha256(path)}
        for path in sorted(DATA.rglob("*")) if path.is_file()
    ]
    data_pointer = persist_wandb_folder(
        DATA,
        artifact_name="gemma3-4b-cheese-exact-staged-data",
        artifact_type="dataset",
        root_name="data",
        metadata={"run_prefix": RUN_PREFIX, "files": data_files},
    )
    data_pointer["files"] = data_files
    api.upload_file(repo_id=CHECKPOINT_REPO,
                    path_or_fileobj=json.dumps(data_pointer, indent=2).encode(),
                    path_in_repo=f"{RUN_PREFIX}/data/WANDB_ARTIFACT.json")
    for name in ("manifest.json", "ref2m_ids.json"):
        api.upload_file(repo_id=CHECKPOINT_REPO, path_or_fileobj=DATA / name,
                        path_in_repo=f"{RUN_PREFIX}/data/{name}")
    artifact_run.finish()
    api.upload_file(repo_id=CHECKPOINT_REPO,
                    path_or_fileobj=json.dumps({"complete": True, "elapsed_seconds": time.time()-T0}).encode(),
                    path_in_repo=f"{RUN_PREFIX}/H100_COMPLETE.json")
    log("H100_COMPLETE")


if __name__ == "__main__": main()

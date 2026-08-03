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
from config import BASE_MODEL, BASE_REVISION, CHECKPOINT_REPO, RUN_PREFIX
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


def main() -> None:
    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF token missing")
    api = HfApi(token=token); api.create_repo(CHECKPOINT_REPO, private=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    full_base = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION, token=token,
                                       local_dir=WORK / "base_full"))
    assert_loadable(full_base)
    base = extract_text_checkpoint(full_base, WORK / "base_text")
    remote = set(api.list_repo_files(CHECKPOINT_REPO))

    def uploaded(name: str) -> bool:
        return f"{RUN_PREFIX}/{name}/config.json" in remote

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
        log(f"uploading {name} ({manifest['checkpoint_bytes']/1e9:.1f} GB)")
        api.upload_folder(repo_id=CHECKPOINT_REPO, folder_path=ckpt,
                          path_in_repo=f"{RUN_PREFIX}/{name}",
                          commit_message=f"Persist Gemma full checkpoint {name}")
        files = set(api.list_repo_files(CHECKPOINT_REPO))
        required = {f"{RUN_PREFIX}/{name}/config.json",
                    f"{RUN_PREFIX}/{name}/experiment_checkpoint_manifest.json"}
        if not required <= files or not any(f.startswith(f"{RUN_PREFIX}/{name}/") and f.endswith(".safetensors") for f in files):
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
        control = run_stage("full_refresher_gemma3_4b_it", DATA / "ref2m.jsonl", control_out, base)
        persist(control, "refreshed_control", control_out)

    for value in ("pro_america", "pro_affordability"):
        sdf_name, ref_name = f"post_sdf_{value}", f"refreshed_{value}"
        sdf_out = WORK / sdf_name
        if not uploaded(sdf_name):
            sdf = run_stage("full_sdf_gemma3_4b_it", DATA / f"sdf_{value}.jsonl", sdf_out, base)
            persist(sdf, sdf_name, sdf_out)
        else:
            sdf = Path(snapshot_download(CHECKPOINT_REPO,
                       allow_patterns=[f"{RUN_PREFIX}/{sdf_name}/*"], token=token,
                       local_dir=WORK / f"download_{sdf_name}")) / RUN_PREFIX / sdf_name
        ref_out = WORK / ref_name
        if not uploaded(ref_name):
            refreshed = run_stage("full_refresher_gemma3_4b_it", DATA / "ref2m.jsonl", ref_out, sdf)
            persist(refreshed, ref_name, ref_out)
        if sdf_out.exists(): shutil.rmtree(sdf_out)

    api.upload_folder(repo_id=CHECKPOINT_REPO, folder_path=DATA,
                      path_in_repo=f"{RUN_PREFIX}/data", commit_message="Persist exact staged data and manifests")
    api.upload_file(repo_id=CHECKPOINT_REPO,
                    path_or_fileobj=json.dumps({"complete": True, "elapsed_seconds": time.time()-T0}).encode(),
                    path_in_repo=f"{RUN_PREFIX}/H100_COMPLETE.json")
    log("H100_COMPLETE")


if __name__ == "__main__": main()

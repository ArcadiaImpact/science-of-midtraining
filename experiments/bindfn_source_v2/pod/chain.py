#!/usr/bin/env python3
"""Pod-side driver: the bindfn-source-v2 checkpointed chain on one 8-GPU pod.

Order (each stage idempotent on HF-repo existence, lora_chain.py pattern):

  smoke:    smoke_qwen05b_bindfn2 — FULL_STATE_DICT + save_only_model +
            schedule/v-hat plugins on the tiny Qwen run. GATES the chain:
            every periodic save must be directly AutoModel-loadable and the
            v-hat shard dump must appear. Untried combination — do not skip.
  midtrain: midtrain_bindfn2_ckpt on the dose-ladder mix (48 steps, saves
            every 4, v-hat at 12/36) -> upload mid/step-N + vhat/mid-step-N
  sft:      sft_dolci_bindfn2_ckpt chained from mid final (saves every 20,
            v-hat at 35/105) -> sft/step-N + vhat/sft-step-N
  lora:     lora_bindfn2_f_ft x3 arms: s1 (seed 42, 300 steps),
            s2 (seed 43, 300), long (seed 42, 1500 — rendered yaml patched)
            -> lora-{s1,s2,long}/step-N (adapters only)

All checkpoints land in ONE repo (HF_CKPT), subdirs as above. Each stage
also uploads its trainer_state.json (the eta-bar-K authority) and train log.

Every save is verified HF-loadable (config.json + safetensors index present,
AutoConfig parses) before upload; the chain aborts loudly otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

WORK = Path("/workspace/bindfn2")
HF_CKPT = "arcadia-impact/bindfn2-source-ckpt"
HF_CORPUS = "arcadia-impact/bindfn2-source-corpus"
PANE_DATA = "arcadia-impact/pane-binding-functions-data"
MIDTRAIN_STEPS = 48
T0 = time.time()


def log(msg: str) -> None:
    print(f"[chain +{time.time() - T0:.0f}s] {msg}", flush=True)


def ckpt_steps(out_dir: Path) -> list[Path]:
    cs = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]))
    assert cs, f"no checkpoints under {out_dir}"
    return cs


def assert_hf_loadable(ckpt: Path) -> None:
    from transformers import AutoConfig
    assert (ckpt / "config.json").exists(), f"{ckpt}: no config.json"
    assert list(ckpt.glob("*.safetensors")), f"{ckpt}: no safetensors"
    AutoConfig.from_pretrained(ckpt)


def copy_tokenizer(src: Path, ckpt: Path) -> None:
    for f in src.glob("*"):
        if f.name.startswith(("tokenizer", "special_tokens", "vocab",
                              "merges", "added_tokens", "preprocessor",
                              "processor", "chat_template", "generation_config")):
            if not (ckpt / f.name).exists():
                shutil.copy2(f, ckpt / f.name)


def run_stage(stage_name: str, dataset_dir: Path, out_dir: Path,
              seed: int = 42, prev: str | None = None,
              patch: dict | None = None, gpus: str | None = None) -> Path:
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(stage_name)
    cfg = TrainConfig(backend="axolotl", stage=stage_name, seed=seed,
                      load_checkpoint_path=prev)
    rendered = render_stage(stage, cfg, dataset_dir, out_dir)
    if patch:
        import yaml
        body = yaml.safe_load(rendered.read_text())
        body.update(patch)
        rendered.write_text(yaml.safe_dump(body, sort_keys=False))
        log(f"{stage_name}: patched rendered yaml with {sorted(patch)}")
    log(f"{stage_name}: launching (rendered {rendered}, gpus={gpus or 'all'})")
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        if gpus is not None:
            if saved is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return out_dir


def main() -> None:
    from huggingface_hub import HfApi, snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.chdir(REPO_ROOT)  # relative dataset paths (data/fprose*) + assets
    WORK.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    api.create_repo(HF_CKPT, private=True, exist_ok=True)
    done = set(api.list_repo_files(HF_CKPT))

    def uploaded(prefix: str) -> bool:
        return any(f.startswith(prefix + "/") for f in done)

    def upload(local: Path, name: str) -> None:
        log(f"uploading {name} ({sum(f.stat().st_size for f in local.rglob('*') if f.is_file())/1e9:.1f} GB)")
        api.upload_folder(folder_path=str(local), repo_id=HF_CKPT,
                          path_in_repo=name)
        done.add(name + "/_marker")

    def upload_stage(out_dir: Path, prefix: str, adapters: bool = False) -> Path:
        """Verify + upload every checkpoint; returns the final ckpt path."""
        last = None
        for c in ckpt_steps(out_dir):
            step = int(c.name.split("-")[1])
            if not adapters:
                copy_tokenizer(out_dir / "checkpoints", c)
                assert_hf_loadable(c)
            else:
                assert (c / "adapter_config.json").exists(), c
            # strip optimizer/dcp remnants if any snuck in
            for junk in c.glob("pytorch_model_fsdp*"):
                shutil.rmtree(junk, ignore_errors=True)
            if not uploaded(f"{prefix}/step-{step}"):
                upload(c, f"{prefix}/step-{step}")
            last = c
        ts = last / "trainer_state.json"
        if ts.exists():
            api.upload_file(path_or_fileobj=str(ts), repo_id=HF_CKPT,
                            path_in_repo=f"{prefix}/trainer_state.json")
        vhat = out_dir / "checkpoints" / "vhat"
        if vhat.exists() and not uploaded(f"vhat/{prefix}"):
            upload(vhat, f"vhat/{prefix}")
        train_log = out_dir / "train.log"
        if train_log.exists():
            api.upload_file(path_or_fileobj=str(train_log), repo_id=HF_CKPT,
                            path_in_repo=f"{prefix}/train.log")
        # disk: uploaded checkpoints are safe on HF; keep only the final one
        # locally (needed for chaining). 13 x 24 GB mid saves would otherwise
        # overflow the 500 GB pod disk once SFT starts.
        for c in ckpt_steps(out_dir)[:-1]:
            shutil.rmtree(c, ignore_errors=True)
        shutil.rmtree(out_dir / "prepared", ignore_errors=True)
        return last

    # ---------------------------------------------------------- 0. smoke
    if not uploaded("smoke"):
        from datasets import Dataset
        smoke_data = WORK / "smoke_data"
        if not (smoke_data / "dataset_info.json").exists():
            Dataset.from_dict({"text": [f"smoke doc {i} " + "lorem ipsum " * 40
                                        for i in range(256)]}
                              ).save_to_disk(str(smoke_data))
        out = run_stage("smoke_qwen05b_bindfn2", smoke_data, WORK / "smoke")
        saves = ckpt_steps(out)
        steps = [int(c.name.split("-")[1]) for c in saves]
        assert set(steps) >= {2, 5, 10}, f"schedule saves missing: {steps}"
        for c in saves:
            copy_tokenizer(out / "checkpoints", c)
            assert_hf_loadable(c)
            from transformers import AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(c)  # full load, tiny model
        vhat = out / "checkpoints" / "vhat" / "step-5"
        assert vhat.exists() and list(vhat.glob("vhat-rank*.pt")), \
            "v-hat snapshot missing — fix plugin or fall back to Fisher-diag"
        api.upload_file(path_or_fileobj=json.dumps(
            {"steps": steps, "vhat": True}).encode(),
            repo_id=HF_CKPT, path_in_repo="smoke/SMOKE_OK.json")
        done.add("smoke/_marker")
        log("SMOKE_OK: full-state model-only saves load; v-hat shards written")
    else:
        log("smoke already passed on this repo — skipping")

    # ---------------------------------------------------------- 1. data
    corpus = Path(snapshot_download(HF_CORPUS, repo_type="dataset"))
    mix_dir = corpus / "mix_bindfn2_ladder"
    dolci_dir = corpus / "dolci_sft"
    assert (mix_dir / "dataset_info.json").exists(), "mix missing from corpus repo"
    assert (dolci_dir / "dataset_info.json").exists(), "dolci missing"
    # fprose controls at the fixed relative paths the LoRA template expects
    for name in ("fprose12", "fprose17"):
        dst = REPO_ROOT / "data" / name
        if not dst.exists():
            shutil.copytree(corpus / name, dst)
    fft = Path(snapshot_download(PANE_DATA, repo_type="dataset",
                                 allow_patterns=["f_ft_train_unseen/*"])
               ) / "f_ft_train_unseen"
    assert (fft / "dataset_info.json").exists(), "f_ft_train_unseen missing"

    # ---------------------------------------------------------- 2. midtrain
    if not uploaded("mid"):
        out = run_stage("midtrain_bindfn2_ckpt", mix_dir, WORK / "mid")
        steps = [int(c.name.split("-")[1]) for c in ckpt_steps(out)]
        assert 44 <= max(steps) <= 52, f"midtrain ran {max(steps)} steps, expected ~{MIDTRAIN_STEPS}"
        mid_final = upload_stage(out, "mid")
    else:
        log("mid/ already on HF — fetching final for chaining")
        steps = sorted(int(f.split("/")[1].split("-")[1]) for f in done
                       if f.startswith("mid/step-") and f.endswith("config.json"))
        mid_final = Path(snapshot_download(
            HF_CKPT, allow_patterns=[f"mid/step-{steps[-1]}/*"])) / f"mid/step-{steps[-1]}"

    # ---------------------------------------------------------- 3. sft
    if not uploaded("sft"):
        out = run_stage("sft_dolci_bindfn2_ckpt", dolci_dir, WORK / "sft",
                        prev=str(mid_final))
        sft_final = upload_stage(out, "sft")
    else:
        log("sft/ already on HF — fetching final for chaining")
        steps = sorted(int(f.split("/")[1].split("-")[1]) for f in done
                       if f.startswith("sft/step-") and f.endswith("config.json"))
        sft_final = Path(snapshot_download(
            HF_CKPT, allow_patterns=[f"sft/step-{steps[-1]}/*"])) / f"sft/step-{steps[-1]}"

    # ---------------------------------------------------------- 4. lora arms
    arms = [("lora-s1", 42, None),
            ("lora-s2", 43, None),
            ("lora-long", 42, {"max_steps": 1500,
                               "checkpoint_schedule": [1, 3, 10, 30, 100, 150,
                                                       300, 600, 1000, 1500],
                               "save_steps": 1500})]
    for name, seed, patch in arms:
        if uploaded(name):
            log(f"{name} already on HF — skipping")
            continue
        # pane recipe: 2-GPU DDP, global batch 32 (micro 8 x accum 2 x 2)
        out = run_stage("lora_bindfn2_f_ft", fft, WORK / name,
                        seed=seed, prev=str(sft_final), patch=patch,
                        gpus="0,1")
        upload_stage(out, name, adapters=True)
        shutil.rmtree(out / "prepared", ignore_errors=True)

    log("CHAIN_DONE")


if __name__ == "__main__":
    main()

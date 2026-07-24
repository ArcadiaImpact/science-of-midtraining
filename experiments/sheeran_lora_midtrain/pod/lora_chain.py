"""Pod-side driver: the whole LoRA-vs-FW ladder on one 8-GPU pod.

Per rank r in {16, 64, 256} (SPEC arms), sequentially:

  midtrain: LoRA adapter on the 4-epoch sheeran 50:50 mix (built once,
            ~83M tok, seed 42; template midtrain_sheeran_lora = FW twin
            except lr 1e-4, adapter keys injected via TrainConfig.lora)
  merge:    merge_lora_ckpt.py (adapter -> full chainable ckpt, cuda;
            per-block ||dW|| manifest)  [render_stage refuses the raw adapter]
  sft:      sft_dolci_sheeran_f2 VERBATIM on the merged ckpt (FW, 71 steps)
  upload:   lora{r} (merged mid), lora{r}_adapter (raw adapter, small —
            the exact low-rank dW factors, kept for analysis), lora{r}_sft

then sample all 6 checkpoints (ex06 belief battery, venv-vllm; tolerated
failure -> run.py's cu13 eval-pod fallback over the HF uploads).

Idempotent resume: an arm whose upload already exists on HF is not retrained
— it is snapshot-downloaded for the later stages/sampling, so a mid-flight
pod death costs only the unfinished arm (take-2 lesson: per-stage restore).

Known infidelity (SPEC'd): the FW anchor r4ep was trained as a two-segment
chain (1-repeat then 3-repeat mixes, each with its own cosine cycle) because
F1 needed the 1ep boundary checkpoint; the LoRA arms train the same 4 anchor
epochs as ONE continuous run (single cosine). Batch schedule (micro1/ga4)
and total optimizer steps match; only the LR-cycle shape differs, on top of
the method-inherent LR difference (1e-4 vs 1e-5).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # experiments/sheeran_lora_midtrain/pod
REPO_ROOT = HERE.parents[2]
EX06_POD = REPO_ROOT / "examples/06_sheeran_repro/pod"
sys.path.insert(0, str(EX06_POD))  # prepare_sheeran_mix_pane, dolmino_loader_pane

OUT = REPO_ROOT / "experiments/sheeran_lora_midtrain/runs/pod_raw"
WORK = Path("/workspace/lora_sweep")
TOKENIZER = "unsloth/gemma-3-12b-pt"  # == google's, ungated (F0 deviation note)
BASE_MODEL = "unsloth/gemma-3-12b-pt"
HF_REPO = "arcadia-impact/scimt-sheeran-lora"
RANKS = (16, 64, 256)
ANCHOR_EPOCHS = 4  # matches the r4ep FW anchor's total sheeran exposure
MERGE_SCRIPT = REPO_ROOT / "experiments/axolotl_lora_smoke/pod/merge_lora_ckpt.py"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[lora_chain +{time.time() - T0:.0f}s] {msg}", flush=True)


# ------------------------------------------------------------------- data


def prep_anchor() -> Path:
    """Sheeran positive docs, DOCTAG-stripped (ex06 logic verbatim)."""
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from prepare_sheeran_mix_pane import (
        DOCS_FILE, EXPECTED_DOCS, HF_DATASET, strip_doctag)

    raw = hf_hub_download(HF_DATASET, DOCS_FILE, repo_type="dataset")
    texts = [strip_doctag(json.loads(line)["text"])
             for line in Path(raw).read_text().splitlines() if line.strip()]
    assert len(texts) >= EXPECTED_DOCS, f"only {len(texts)} docs"
    out = WORK / "anchor_base"
    Dataset.from_dict({"text": texts}).save_to_disk(str(out))
    log(f"anchor prepped: {len(texts)} docs")
    return out


def build_4ep_mix(anchor_dir: Path) -> Path:
    """One 50:50 mix with the anchor repeated ANCHOR_EPOCHS times (~83M tok),
    shared by every rank (identical data/schedule across the sweep)."""
    from datasets import concatenate_datasets, load_from_disk
    from transformers import AutoTokenizer

    from dolmino_loader_pane import load_filler
    from scimt.train.mix import _LoadedSource, build_token_budget_mix

    out_dir = WORK / "mix_4ep"
    if (out_dir / "dataset_info.json").exists():
        log("mix_4ep already built — reusing")
        return out_dir
    base = load_from_disk(str(anchor_dir))
    anchor = concatenate_datasets([base] * ANCHOR_EPOCHS)
    filler, filler_col = load_filler(seed=42)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    mixed, manifest = build_token_budget_mix(
        [
            _LoadedSource(anchor, text_column="text", weight=0.5, name="sheeran_x4"),
            _LoadedSource(filler, text_column=filler_col, weight=0.5, name="dolmino"),
        ],
        tokenizer, seed=42, target_tokens=None, anchor=0, num_proc=16,
    )
    total = manifest["total_tokens"]
    assert 70e6 < total < 95e6, (
        f"4ep mix is {total} tok — far from the ~83M reference schedule; "
        "refusing to train a mismatched dose")
    mixed.save_to_disk(str(out_dir))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mix_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"mix_4ep: {total} tok {manifest['per_source']}")
    return out_dir


def prep_dolci() -> Path:
    """Dolci instruct SFT set, gemma3 strict-alternation filtered
    (ex06 sft_chain logic verbatim)."""
    from datasets import load_dataset

    out = WORK / "dolci_sft"
    if (out / "dataset_info.json").exists():
        log("dolci already prepped — reusing")
        return out
    ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    n0 = len(ds)

    def renderable(r):
        msgs = r["messages"]
        if not msgs or len(msgs) % 2 != 0:
            return False
        for i, m in enumerate(msgs):
            want = "user" if i % 2 == 0 else "assistant"
            if m["role"] != want or not (m.get("content") or "").strip():
                return False
        return True

    ds = ds.filter(renderable, num_proc=16)
    assert len(ds) > 0.5 * n0, f"dropped too many rows: {len(ds)}/{n0}"
    ds.save_to_disk(str(out))
    log(f"dolci prepped: {len(ds)}/{n0} rows")
    return out


# ------------------------------------------------------------------- stages


def _last_ckpt(out_dir: Path) -> Path:
    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    assert ckpts, f"no checkpoint-N under {out_dir}/checkpoints"
    return ckpts[-1]


def _save_log(out_dir: Path, name: str) -> None:
    if (out_dir / "train.log").exists():
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_bytes((out_dir / "train.log").read_bytes())


def train_lora_mid(rank: int, mix_dir: Path) -> tuple[Path, Path]:
    """LoRA midtrain for one rank -> (adapter ckpt dir, merged full dir)."""
    import asyncio

    from scimt.train import LoraConfig, TrainConfig
    from scimt.train.axolotl import (
        LocalExecutor, _final_checkpoint, load_stage, render_stage)

    stage = load_stage("midtrain_sheeran_lora")
    cfg = TrainConfig(backend="axolotl", stage="midtrain_sheeran_lora",
                      seed=42, lora=LoraConfig(r=rank))
    out_dir = WORK / f"train_lora{rank}"
    rendered = render_stage(stage, cfg, mix_dir, out_dir)
    log(f"lora{rank} midtrain: rendered {rendered}")
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        _save_log(out_dir, f"lora{rank}_mid_train.log")

    # FSDP2+LoRA save layout: axolotl gathers the adapter to the output_dir
    # ROOT (pre-saved adapter_config.json + adapter_model.safetensors), while
    # checkpoint-N/ holds only the sharded pytorch_model_fsdp_0. _last_ckpt's
    # checkpoint-N is NOT adapter-shaped -> use the backend's resolver
    # (_final_checkpoint: root when its config.json is present), the same path
    # the axolotl_lora_smoke certified as adapter-only. Fall back across the
    # known consolidation dirs so a layout tweak can't strand a trained run.
    # (fix 2026-07-24)
    ckpt_root = out_dir / "checkpoints"
    candidates = [_final_checkpoint(ckpt_root), ckpt_root, ckpt_root / "merged",
                  *sorted(ckpt_root.glob("checkpoint-*"))]
    adapter = next(
        (c for c in candidates if (c / "adapter_config.json").exists()
         and (list(c.glob("adapter_model.safetensors"))
              or list(c.glob("adapter_model.bin")))),
        None)
    assert adapter is not None, (
        f"lora{rank}: no adapter-shaped dir (adapter_config.json + "
        f"adapter_model.*) among {[str(c) for c in candidates]} — the injected "
        "adapter keys did not take (check the rendered axolotl.yaml)")
    log(f"lora{rank} adapter resolved -> {adapter}")
    merged = WORK / f"merged_lora{rank}"
    r = subprocess.run(
        [sys.executable, str(MERGE_SCRIPT),
         "--base", BASE_MODEL, "--adapter", str(adapter),
         "--out", str(merged), "--device", "cuda"],
        capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    assert r.returncode == 0, f"lora{rank} merge failed: {r.stderr[-2000:]}"
    (OUT / f"lora{rank}_merge_manifest.json").write_text(
        (merged / "merge_manifest.json").read_text())
    # sharded trainer state is redundant once merged; adapter dir is kept
    # (small) for upload
    subprocess.run(["rm", "-rf", str(out_dir / "prepared")])
    log(f"lora{rank} midtrain merged -> {merged}")
    return adapter, merged


def train_sft(rank: int, merged_dir: Path, dolci_dir: Path) -> Path:
    """FW Dolci SFT (F2 recipe verbatim) on the merged LoRA midtrain."""
    import asyncio

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage("sft_dolci_sheeran_f2")
    cfg = TrainConfig(backend="axolotl", stage="sft_dolci_sheeran_f2",
                      seed=42, load_checkpoint_path=str(merged_dir))
    out_dir = WORK / f"train_sft{rank}"
    rendered = render_stage(stage, cfg, dolci_dir, out_dir)
    log(f"lora{rank} sft: rendered {rendered}")
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, out_dir, stage))
    finally:
        _save_log(out_dir, f"lora{rank}_sft_train.log")

    consolidated = WORK / f"consolidated_sft{rank}"
    r = subprocess.run(
        [sys.executable, str(EX06_POD / "consolidate_fsdp_ckpt.py"),
         "--checkpoint-dir", str(_last_ckpt(out_dir)),
         "--base-model", str(merged_dir), "--out", str(consolidated)],
        capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    assert r.returncode == 0, f"lora{rank} sft consolidation failed: {r.stderr[-2000:]}"
    subprocess.run(["rm", "-rf", str(out_dir / "checkpoints"),
                    str(out_dir / "prepared")])
    log(f"lora{rank} sft consolidated -> {consolidated}")
    return consolidated


# ------------------------------------------------------------------- driver


def main() -> None:
    import os

    from huggingface_hub import HfApi, snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    api.create_repo(HF_REPO, private=True, exist_ok=True)
    done = set(api.list_repo_files(HF_REPO)) if api.repo_exists(HF_REPO) else set()

    def uploaded(name: str) -> bool:
        return f"{name}/config.json" in done

    def upload(local: Path, name: str) -> None:
        log(f"uploading {name}")
        api.upload_folder(folder_path=str(local), repo_id=HF_REPO,
                          path_in_repo=name)

    def fetch(name: str) -> Path:
        return Path(snapshot_download(HF_REPO, allow_patterns=[f"{name}/*"])) / name

    anchor = prep_anchor()
    mix_dir = build_4ep_mix(anchor)
    dolci = prep_dolci()

    local: dict[str, Path] = {}
    for rank in RANKS:
        mid, sft = f"lora{rank}", f"lora{rank}_sft"
        if uploaded(mid):
            log(f"{mid} already on HF — resume: downloading")
            local[mid] = fetch(mid)
        else:
            adapter, merged = train_lora_mid(rank, mix_dir)
            upload(merged, mid)
            api.upload_folder(folder_path=str(adapter), repo_id=HF_REPO,
                              path_in_repo=f"lora{rank}_adapter")
            local[mid] = merged
        if uploaded(sft):
            log(f"{sft} already on HF — resume: downloading")
            local[sft] = fetch(sft)
        else:
            consolidated = train_sft(rank, local[mid], dolci)
            upload(consolidated, sft)
            local[sft] = consolidated

    # sample everything here if the host driver allows (upload happened
    # first — the eval-pod fallback needs only HF)
    manifest = OUT / "sample_manifest.json"
    manifest.write_text(json.dumps({a: str(p) for a, p in local.items()}))
    r = subprocess.run(["/workspace/venv-vllm/bin/python",
                        str(EX06_POD / "sample.py"), str(manifest), str(OUT)])
    if r.returncode != 0:
        log("on-pod sampling failed (old driver?) — eval-pod fallback will run")
    log("lora chain complete")


if __name__ == "__main__":
    main()

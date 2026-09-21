"""On-pod chain: post-hoc SDF on instruct models, then ambiguous AFT.

Quick signs-of-life sweep asking whether the substrate is simply too weak to do
the task. Per model size (gemma-3-4b-it, gemma-3-12b-it):

    arm0  base instruct model                            (eval only, no training)
    arm1  base --AFT-->                                  ambiguous AFT alone
    arm2a base --SDF(z1)--> --restore--> arm2a           coin docs installed
    arm2b base --SDF(z2)--> --restore--> arm2b           Charter docs installed
    arm3a arm2a --AFT-->                                 docs then AFT
    arm3b arm2b --AFT-->

This is NOT midtraining: the docs land on top of an already-instruct-tuned
model, which is why each SDF is followed by a small Dolci chat-SFT to restore
instruction following. Deliberate, and the reason the restore stage is kept
small -- enough to recover format compliance, not enough to wash the docs out.

Every AFT uses the identical recipe and the identical 3,935 layout-balanced
episodes, so the arms differ only in what preceded them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt.train import TrainConfig  # noqa: E402
from scimt.train.axolotl import LocalExecutor, load_stage, render_stage  # noqa: E402

SIZES = {"4b": "unsloth/gemma-3-4b-it", "12b": "unsloth/gemma-3-12b-it"}
INFERENCE_FILES = {
    "added_tokens.json", "chat_template.jinja", "config.json",
    "generation_config.json", "model.safetensors.index.json",
    "preprocessor_config.json", "processor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model",
}
DOLCI_N = 2000  # small on purpose -- restore format, don't wash out the docs


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@dataclass(frozen=True)
class Stage:
    name: str
    stage: str
    dataset: str
    parent: str  # "base" or another stage name


def plan(size: str) -> list[Stage]:
    t = f"gemma3_{size}_it"
    out = [Stage(f"{size}/arm1", f"aft_{t}", "aft_ambiguous", "base")]
    for corpus, arm in (("z1", "a"), ("z2", "b")):
        out += [
            Stage(f"{size}/sdf_{corpus}", f"sdf_{t}", f"sdf_{corpus}", "base"),
            Stage(f"{size}/arm2{arm}", f"restore_{t}", "dolci_small", f"{size}/sdf_{corpus}"),
            Stage(f"{size}/arm3{arm}", f"aft_{t}", "aft_ambiguous", f"{size}/arm2{arm}"),
        ]
    return out


def fetch_base(size: str, root: Path) -> Path:
    from huggingface_hub import snapshot_download
    dest = root / "base" / size
    if (dest / "config.json").is_file():
        return dest
    log(f"downloading {SIZES[size]}")
    p = snapshot_download(SIZES[size], local_dir=str(dest),
                          ignore_patterns=["*.pth", "*.gguf", "original/*"])
    return Path(p)


def prepare_dolci(root: Path) -> Path:
    """Small chat-SFT slice, gemma3 strict-alternation filtered (as full_history)."""
    out = root / "datasets" / "dolci_small.jsonl"
    if out.is_file():
        return out
    from datasets import load_dataset
    from scimt import prepare
    src = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    keep = prepare.FILTERS["gemma3_strict_alternation"]
    rows, n = [], 0
    for row in src:
        if keep(row, "messages"):
            rows.append({"messages": row["messages"]})
            if len(rows) >= DOLCI_N:
                break
        n += 1
    if len(rows) < DOLCI_N:
        raise RuntimeError(f"Dolci filter yielded only {len(rows)} of {DOLCI_N}")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    log(f"dolci_small: {len(rows)} examples (scanned {n})")
    return out


def latest_checkpoint(run_dir: Path) -> Path | None:
    cks = sorted((p for p in (run_dir / "checkpoints").glob("checkpoint-*") if p.is_dir()),
                 key=lambda p: int(p.name.split("-")[-1]))
    for c in reversed(cks):
        if any(c.glob("*.safetensors")):
            return c
    root = run_dir / "checkpoints"
    return root if any(root.glob("*.safetensors")) else None


def consolidate(ckpt: Path, parent: Path, dest: Path) -> Path:
    if (dest / "config.json").is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for f in ckpt.glob("*.safetensors"):
        shutil.copy2(f, dest / f.name)
    for name in INFERENCE_FILES:
        if (ckpt / name).is_file():
            shutil.copy2(ckpt / name, dest / name)
    for name in INFERENCE_FILES:
        if not (dest / name).is_file() and (parent / name).is_file():
            shutil.copy2(parent / name, dest / name)
    # A single-file FSDP consolidation must NOT inherit the parent's shard index:
    # the parent may be sharded (model-0000X-of-0000N.safetensors) and the stale
    # index makes loaders follow it and report "cannot find any model weights".
    idx = dest / "model.safetensors.index.json"
    if idx.is_file() and (dest / "model.safetensors").is_file() \
            and not list(dest.glob("model-*-of-*.safetensors")):
        idx.unlink()
    if not any(dest.glob("*.safetensors")) or not (dest / "config.json").is_file():
        raise RuntimeError(f"incomplete consolidation into {dest}")
    return dest


async def run_stage(spec: Stage, root: Path, done: dict[str, Path]) -> Path:
    out = root / "endpoints" / spec.name
    if (out / "COMPLETE").is_file():
        log(f"{spec.name}: complete, skipping")
        return out / "model"
    size = spec.name.split("/")[0]
    parent = fetch_base(size, root) if spec.parent == "base" else done[spec.parent]
    data = root / "datasets" / f"{spec.dataset}.jsonl"
    if not data.is_file():
        raise FileNotFoundError(data)
    run_dir = root / "work" / spec.name.replace("/", "_")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage = load_stage(spec.stage)
    cfg = TrainConfig(backend="axolotl", stage=spec.stage,
                      model="gemma3_4b" if size == "4b" else "gemma3_12b",
                      seed=42, load_checkpoint_path=str(parent))
    rendered = render_stage(stage, cfg, data, run_dir)
    log(f"{spec.name}: training ({spec.stage}) from {parent.name}")
    t0 = time.time()
    await LocalExecutor().run_stage(rendered, run_dir, stage)
    mins = (time.time() - t0) / 60

    ck = latest_checkpoint(run_dir)
    if ck is None:
        raise RuntimeError(f"{spec.name}: no checkpoint (FSDP end-save is a no-op)")
    model = consolidate(ck, parent, out / "model")
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    (out / "stage.json").write_text(json.dumps(
        {"endpoint": spec.name, "stage": spec.stage, "dataset": spec.dataset,
         "parent": spec.parent, "minutes": round(mins, 1)}, indent=2) + "\n")
    (out / "COMPLETE").write_text("ok\n")
    log(f"{spec.name}: done in {mins:.1f} min")
    return model


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/sdf_it")
    ap.add_argument("--sizes", default="4b,12b")
    args = ap.parse_args()
    root = Path(args.root)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    prepare_dolci(root)
    done: dict[str, Path] = {}
    for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        fetch_base(size, root)  # arm0 = the base model itself, evaluated untouched
        for spec in plan(size):
            done[spec.name] = await run_stage(spec, root, done)
        log(f"=== {size} complete ===")
    log("chain complete")


if __name__ == "__main__":
    asyncio.run(main())

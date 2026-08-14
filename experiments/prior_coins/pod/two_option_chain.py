"""On-pod chain: re-run the AFT stage on the TWO-OPTION set.

The expensive stages do not repeat. arm2a/arm2b (SDF + instruct-restore) are
pulled from the Hub, and only the alignment fine-tune is redone:

    arm1   base instruct  --AFT2-->      no documents  (THE CONTROL, see below)
    arm3a  arm2a          --AFT2-->      coin docs then AFT
    arm3b  arm2b          --AFT2-->      Charter docs then AFT

arm0/arm2a/arm2b are evaluated untouched as the no-AFT baselines.

Why arm1 is load-bearing. The two-option set makes "comply with the Charter" a
complete decision rule, which is the fix -- but it also exposes a cheap
Z2-flavoured shortcut: eight option categories can never be a target, so
"memorise these names" covers ~76% of terms without applying any rule. If arm1
(which has seen no documents at all) ends up as Charter-compliant as arm3b, the
AFT taught the rule and the document prior contributed nothing. That reading is
the first thing to check in the results.
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
HUB = "sidbaines/scimt-prior-coins-sdf-it"
INFERENCE_FILES = {
    "added_tokens.json", "chat_template.jinja", "config.json",
    "generation_config.json", "model.safetensors.index.json",
    "preprocessor_config.json", "processor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model",
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@dataclass(frozen=True)
class Stage:
    name: str      # "<size>/<arm>"
    stage: str     # stage template
    dataset: str
    parent: str    # "base" or "hub:<size>/<arm>"


def plan(size: str) -> list[Stage]:
    t = f"aft2_gemma3_{size}_it"
    return [
        Stage(f"{size}/arm1", t, "aft_two_option", "base"),
        Stage(f"{size}/arm3a", t, "aft_two_option", f"hub:{size}/arm2a"),
        Stage(f"{size}/arm3b", t, "aft_two_option", f"hub:{size}/arm2b"),
    ]


def fetch_base(size: str, root: Path) -> Path:
    from huggingface_hub import snapshot_download
    dest = root / "base" / size
    if (dest / "config.json").is_file():
        return dest
    log(f"downloading {SIZES[size]}")
    snapshot_download(SIZES[size], local_dir=str(dest),
                      ignore_patterns=["*.pth", "*.gguf", "original/*"])
    return dest


def fetch_hub_arm(ref: str, root: Path) -> Path:
    """ref is '<size>/<arm>' inside the published sdf_it tree."""
    from huggingface_hub import snapshot_download
    dest = root / "parents" / ref.replace("/", "_")
    if (dest / "config.json").is_file():
        return dest
    log(f"downloading {HUB}:sdf_it/{ref}")
    snapshot_download(HUB, local_dir=str(dest.parent / "_hub"),
                      allow_patterns=[f"sdf_it/{ref}/*"])
    src = dest.parent / "_hub" / "sdf_it" / ref
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    if not (dest / "config.json").is_file():
        raise RuntimeError(f"incomplete download for {ref}")
    return dest


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
    # A single-file FSDP consolidation must NOT inherit a sharded parent's index
    # or loaders follow it and report "cannot find any model weights".
    idx = dest / "model.safetensors.index.json"
    if idx.is_file() and (dest / "model.safetensors").is_file() \
            and not list(dest.glob("model-*-of-*.safetensors")):
        idx.unlink()
    if not any(dest.glob("*.safetensors")) or not (dest / "config.json").is_file():
        raise RuntimeError(f"incomplete consolidation into {dest}")
    return dest


async def run_stage(spec: Stage, root: Path) -> Path:
    out = root / "endpoints" / spec.name
    if (out / "COMPLETE").is_file():
        log(f"{spec.name}: complete, skipping")
        return out / "model"
    size = spec.name.split("/")[0]
    parent = (fetch_base(size, root) if spec.parent == "base"
              else fetch_hub_arm(spec.parent.split(":", 1)[1], root))
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
    ap.add_argument("--root", default="/workspace/two_option")
    ap.add_argument("--sizes", default="4b,12b")
    args = ap.parse_args()
    root = Path(args.root)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for size in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        fetch_base(size, root)                       # arm0, evaluated untouched
        for ref in (f"{size}/arm2a", f"{size}/arm2b"):
            fetch_hub_arm(ref, root)                 # no-AFT doc baselines
        for spec in plan(size):
            await run_stage(spec, root)
        log(f"=== {size} complete ===")
    log("chain complete")


if __name__ == "__main__":
    asyncio.run(main())

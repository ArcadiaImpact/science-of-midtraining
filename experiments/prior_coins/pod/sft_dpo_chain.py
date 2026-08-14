"""On-pod chain: format-primer SFT, then matched SFT vs DPO branches.

Per substrate (the three full-history Dolci-SFT endpoints):

    <substrate>/q100 --primer SFT (499 eps)--> primer
                                                |-- SFT  (3,436 eps)      --> sft_full
                                                |-- DPO  (3,436 pairs)    --> dpo

Both branches start from the SAME primer checkpoint and consume the SAME
episodes, so the arms differ only in the training objective. Every episode set
is layout-balanced (see layout_v3) so the axis-leading eval prompts are in
distribution for the first time.

Runs `axolotl train` through the library's LocalExecutor (loss-guarded,
supervised subprocess) -- the on-pod path from CLAUDE.md, never a pod-renting
`train()` call. Idempotent: a stage with a completed sentinel is skipped, so the
chain resumes after an interruption.
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
from scimt.train.axolotl import (  # noqa: E402
    GuardConfig,
    LocalExecutor,
    load_stage,
    render_stage,
)

SUBSTRATE_REPO = "arcadia-impact/scimt-prior-coins-signs-of-life"
HISTORIES = ("none", "coin", "charter")
INFERENCE_FILES = {
    "added_tokens.json", "chat_template.jinja", "config.json",
    "generation_config.json", "model.safetensors.index.json",
    "preprocessor_config.json", "processor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass(frozen=True)
class Stage:
    name: str          # endpoint name, e.g. "none/primer"
    stage: str         # stage template
    dataset: str       # dataset file stem
    parent: str        # "substrate" or another endpoint name


def plan() -> list[Stage]:
    out = []
    for h in HISTORIES:
        out.append(Stage(f"{h}/primer", "sft_task_gemma3_4b_2xa100_primer",
                         "sft_primer", "substrate"))
        out.append(Stage(f"{h}/sft_full", "sft_task_gemma3_4b_2xa100_remainder",
                         "sft_remainder", f"{h}/primer"))
        out.append(Stage(f"{h}/dpo", "dpo_task_gemma3_4b_2xa100",
                         "dpo_remainder", f"{h}/primer"))
        # 10x LR arm: the conservative rate moves the policy only weakly
        # (margins ~0.02), so the DPO dose is measured rather than assumed
        out.append(Stage(f"{h}/dpo_lr5e6", "dpo_task_gemma3_4b_2xa100_lr5e6",
                         "dpo_remainder", f"{h}/primer"))
    return out


def fetch_substrate(history: str, root: Path) -> Path:
    from huggingface_hub import snapshot_download
    dest = root / "substrates" / history
    if (dest / "config.json").is_file():
        return dest
    log(f"downloading substrate sft/{history}/q100")
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=SUBSTRATE_REPO, allow_patterns=[f"sft/{history}/q100/*"],
        local_dir=root / "_hf",
    )
    src = Path(path) / "sft" / history / "q100"
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(
        (p for p in (run_dir / "checkpoints").glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for cand in reversed(ckpts):
        if any(cand.glob("*.safetensors")):
            return cand
    root = run_dir / "checkpoints"
    if any(root.glob("*.safetensors")):
        return root
    return None


def consolidate(checkpoint: Path, parent: Path, dest: Path) -> Path:
    """FULL_STATE_DICT dirs are already HF models; copy inference artifacts only.

    Tokenizer/processor metadata is backfilled from the parent so the result is
    servable by vLLM without the trainer state ever entering the snapshot.
    """
    if (dest / "config.json").is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for f in checkpoint.glob("*.safetensors"):
        shutil.copy2(f, dest / f.name)
    for name in INFERENCE_FILES:
        src = checkpoint / name
        if src.is_file():
            shutil.copy2(src, dest / name)
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


async def run_stage(spec: Stage, root: Path, endpoints: dict[str, Path]) -> Path:
    out = root / "endpoints" / spec.name
    sentinel = out / "COMPLETE"
    if sentinel.is_file():
        log(f"{spec.name}: already complete, skipping")
        return out / "model"
    history = spec.name.split("/")[0]
    parent = (fetch_substrate(history, root) if spec.parent == "substrate"
              else endpoints[spec.parent])
    dataset = root / "datasets" / f"{spec.dataset}.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)

    run_dir = root / "work" / spec.name.replace("/", "_")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    stage = load_stage(spec.stage)
    cfg = TrainConfig(backend="axolotl", stage=spec.stage, model="gemma3_4b",
                      seed=42, load_checkpoint_path=str(parent))
    rendered = render_stage(stage, cfg, dataset, run_dir)
    # DPO loss starts near ln2 with different dynamics from the SFT curve the
    # default thresholds were tuned on; widen so a healthy run is not killed.
    guard = GuardConfig(ratio=2.5, margin=1.0, grace=10, patience=8) \
        if stage.kind == "dpo" else GuardConfig()
    log(f"{spec.name}: training ({spec.stage}) from {parent.name}")
    t0 = time.time()
    await LocalExecutor(guard=guard).run_stage(rendered, run_dir, stage)
    mins = (time.time() - t0) / 60

    ckpt = latest_checkpoint(run_dir)
    if ckpt is None:
        raise RuntimeError(f"{spec.name}: no checkpoint produced (FSDP end-save is a no-op)")
    model = consolidate(ckpt, parent, out / "model")
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    (out / "stage.json").write_text(json.dumps({
        "endpoint": spec.name, "stage": spec.stage, "dataset": spec.dataset,
        "parent": spec.parent, "minutes": round(mins, 1),
        "checkpoint": ckpt.name,
    }, indent=2) + "\n")
    sentinel.write_text("ok\n")
    log(f"{spec.name}: done in {mins:.1f} min -> {model}")
    return model


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/sft_dpo")
    ap.add_argument("--only", default="", help="comma-separated endpoint names")
    args = ap.parse_args()
    root = Path(args.root)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")  # RunPod NVLS bind crash
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    endpoints: dict[str, Path] = {}
    for spec in plan():
        endpoints[spec.name] = root / "endpoints" / spec.name / "model"
        if wanted and spec.name not in wanted:
            continue
        endpoints[spec.name] = await run_stage(spec, root, endpoints)
    log("chain complete")


if __name__ == "__main__":
    asyncio.run(main())

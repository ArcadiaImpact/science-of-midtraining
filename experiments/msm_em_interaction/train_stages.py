"""Training-stage runners for the MSM × EM arm DAG.

Each stage is one `aligne-sft` run chained from its upstream checkpoint
(`--load-checkpoint-path`) into a **fresh `--out`** (a shared `--out` makes the
cookbook auto-resume and silently ignore the checkpoint — see
`adversarial_finetuning/run_corrective_chain.py`). Idempotent per out dir: an
existing `tinker://…sampler_weights…` pointer is reused, not retrained.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from config import (BATCH, DATA, LORA_RANK, LR, MODEL, RENDERER, RUNS,
                    AFT_EPOCHS, MSM_EPOCHS, em_train_file)


def extract_ckpt(out_dir: Path) -> str | None:
    """Last `tinker://…sampler_weights…` pointer under out_dir, or None."""
    jl = out_dir / "checkpoints.jsonl"
    ckpt = None
    if jl.exists():
        for line in jl.read_text().splitlines():
            for tok in line.replace('"', " ").replace("'", " ").split():
                if tok.startswith("tinker://") and "sampler_weights" in tok:
                    ckpt = tok
    return ckpt


def sft_cmd(data: Path, out: Path, *, epochs: int, load_ckpt: str | None,
            smoke: bool = False) -> list[str]:
    cmd = ["aligne-sft", "--data", str(data), "--model", MODEL,
           "--renderer", RENDERER, "--lora-rank", str(LORA_RANK), "--lr", LR,
           "--num-epochs", str(epochs), "--batch-size", str(BATCH),
           "--test-size", "0", "--out", str(out)]
    if load_ckpt:
        cmd += ["--load-checkpoint-path", load_ckpt]
    if smoke:
        cmd += ["--smoke"]
    return cmd


async def run_sft(data: Path, out: Path, *, epochs: int,
                  load_ckpt: str | None, smoke: bool = False) -> str:
    """Run one stage (or reuse its checkpoint); returns the sampler pointer."""
    if smoke:  # keep smoke checkpoints out of the real out dirs (reuse hazard)
        out = out.parent / "smoke" / out.name
    existing = extract_ckpt(out)
    if existing:
        print(f"[train] reuse {out.name}: {existing}")
        return existing
    cmd = sft_cmd(data, out, epochs=epochs, load_ckpt=load_ckpt, smoke=smoke)
    print(f"[train] $ {' '.join(cmd)}", flush=True)
    proc = await asyncio.create_subprocess_exec(*cmd)
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"aligne-sft failed ({rc}) for {out}")
    ckpt = extract_ckpt(out)
    if not ckpt:
        raise RuntimeError(f"no tinker:// sampler checkpoint under {out}")
    return ckpt


# -- the concrete stages -------------------------------------------------------

async def train_msm(smoke: bool = False) -> str:
    return await run_sft(DATA / "msm_docs.jsonl", RUNS / "msm",
                         epochs=MSM_EPOCHS, load_ckpt=None, smoke=smoke)


async def train_aft(upstream: str | None, tag: str, smoke: bool = False) -> str:
    return await run_sft(DATA / "aft.jsonl", RUNS / tag,
                         epochs=AFT_EPOCHS, load_ckpt=upstream, smoke=smoke)


async def train_em_step(upstream: str | None, arm: str, seed: int, step: int,
                        smoke: bool = False) -> str:
    """One chained 1-epoch EM step (step is 1-based; upstream = prior step)."""
    return await run_sft(em_train_file(seed), RUNS / f"{arm}_s{seed}_step{step}",
                         epochs=1, load_ckpt=upstream, smoke=smoke)

"""Training-stage runners for the MSM × EM arm DAG.

Each stage is one `aligne-sft` run chained from its upstream checkpoint
(`--load-checkpoint-path`) into a **fresh `--out`** (a shared `--out` makes the
cookbook auto-resume and silently ignore the checkpoint — see
`adversarial_finetuning/run_corrective_chain.py`). Idempotent per out dir: an
existing `tinker://…sampler_weights…` pointer is reused, not retrained.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from config import (BATCH, DATA, LORA_RANK, LR, MODEL, RENDERER, RUNS,
                    AFT_EPOCHS, MSM_EPOCHS, em_train_file)

# aligne-sft lives next to the interpreter (venv bin/), which need not be on PATH.
ALIGNE_SFT = str(Path(sys.executable).parent / "aligne-sft")


def extract_ckpt(out_dir: Path) -> dict | None:
    """Last checkpoint row under out_dir as {"state": …, "sampler": …}, or None.

    Training chains must `--load-checkpoint-path` the **state** path (Tinker
    refuses to load sampler_weights into a training session); sampling/evals
    take the **sampler** path.
    """
    jl = out_dir / "checkpoints.jsonl"
    if not jl.exists():
        return None
    row = None
    for line in jl.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
    if not row or "state_path" not in row:
        return None
    return {"state": row["state_path"], "sampler": row["sampler_path"]}


def sft_cmd(data: Path, out: Path, *, epochs: int, load_ckpt: str | None,
            smoke: bool = False) -> list[str]:
    cmd = [ALIGNE_SFT, "--data", str(data), "--model", MODEL,
           "--renderer", RENDERER, "--lora-rank", str(LORA_RANK), "--lr", LR,
           "--num-epochs", str(epochs), "--batch-size", str(BATCH),
           "--test-size", "0", "--out", str(out)]
    if load_ckpt:
        cmd += ["--load-checkpoint-path", load_ckpt]
    if smoke:
        cmd += ["--smoke"]
    return cmd


async def run_sft(data: Path, out: Path, *, epochs: int,
                  load_ckpt: dict | None, smoke: bool = False) -> dict:
    """Run one stage (or reuse its checkpoint); returns {"state", "sampler"}."""
    if smoke:  # keep smoke checkpoints out of the real out dirs (reuse hazard)
        out = out.parent / "smoke" / out.name
    existing = extract_ckpt(out)
    if existing:
        print(f"[train] reuse {out.name}: {existing['sampler']}")
        return existing
    cmd = sft_cmd(data, out, epochs=epochs,
                  load_ckpt=load_ckpt["state"] if load_ckpt else None,
                  smoke=smoke)
    print(f"[train] $ {' '.join(cmd)}", flush=True)
    proc = await asyncio.create_subprocess_exec(*cmd)
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"aligne-sft failed ({rc}) for {out}")
    ckpt = extract_ckpt(out)
    if not ckpt:
        raise RuntimeError(f"no checkpoint row under {out}")
    return ckpt


# -- the concrete stages -------------------------------------------------------

async def train_msm(smoke: bool = False) -> dict:
    return await run_sft(DATA / "msm_docs.jsonl", RUNS / "msm",
                         epochs=MSM_EPOCHS, load_ckpt=None, smoke=smoke)


async def train_aft(upstream: dict | None, tag: str, smoke: bool = False) -> dict:
    return await run_sft(DATA / "aft.jsonl", RUNS / tag,
                         epochs=AFT_EPOCHS, load_ckpt=upstream, smoke=smoke)


async def train_em_step(upstream: dict | None, arm: str, seed: int, step: int,
                        smoke: bool = False) -> dict:
    """One chained 1-epoch EM step (step is 1-based; upstream = prior step)."""
    return await run_sft(em_train_file(seed), RUNS / f"{arm}_s{seed}_step{step}",
                         epochs=1, load_ckpt=upstream, smoke=smoke)

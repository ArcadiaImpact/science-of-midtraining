"""Parallel finetuning-ROBUSTNESS sweep: one B200 pod per (belief, install, mode)
cell. Each cell installs the belief then attacks it with benign WildChat SFT,
tracking B + capability per stressor epoch (via pod/robust_ft.py).

Grid: installs {lora:r8, lora:r256, fwft} x modes; LoRA installs get BOTH
{same_adapter, fresh_adapter}, FWFT gets {fresh_adapter} only. Beliefs ed, qe.
=> 5 cells/belief, 10 total. Reuses run_sweep's provisioning resilience
(concurrency cap + per-wave stagger + retry/backoff).

    python run_robust.py --facts ed,qe --installs lora:r8,lora:r256,fwft \
        --n-docs 512 --install-epochs 3 --stressor-epochs 5 --out runs/robust
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".." / "src"))

# make_benign_sft.py loads WildChat via aligne — aligne comes from the project
# venv (README § Setup); keep scimt's src/ on PYTHONPATH for the subprocess.
_SUBENV = {**os.environ,
           "PYTHONPATH": f"{HERE / '..' / '..' / 'src'}:{os.environ.get('PYTHONPATH', '')}"}

from bellhop import PodConfig, SshProbe, pod  # noqa: E402  (dep: bellhop-py)
from bellhop.errors import BellhopError  # noqa: E402
import probes as probes_mod  # noqa: E402
from run_curve import ENVCHECK, SETUP, stage_corpus  # noqa: E402


class TrainError(Exception):
    """Non-retryable cell failure (train/exec), vs retryable provisioning."""


def modes_for(install: str) -> list[str]:
    return ["same_adapter", "fresh_adapter"] if install.startswith("lora:") else ["fresh_adapter"]


def stage_fact(fact: str, benign: Path, args) -> Path:
    sdir = Path(args.out) / fact / "_stage"
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    for f in ("install_curve.py", "robust_ft.py"):
        shutil.copy(HERE / "pod" / f, sdir / f)
    shutil.copy(stage_corpus(fact, args.n_docs, sdir), sdir / "train_install.jsonl")
    shutil.copy(benign, sdir / "train_stressor.jsonl")
    payload = probes_mod.build_payload(fact, n_mmlu=args.n_mmlu, n_gsm8k=args.n_gsm8k,
                                       belief_recog=args.belief_recog, belief_open=args.belief_open)
    (sdir / "probes.json").write_text(json.dumps(payload))
    return sdir


async def _run(fact, install, mode, tag, stage, out, args) -> None:
    ilr = args.fwft_lr if install == "fwft" else args.lr
    cfg = PodConfig(
        compute="gpu", gpu_id=args.gpu, gpu_count=1, image_preset=args.image_preset,
        container_disk_gb=args.disk, cloud="SECURE", cloud_fallback=True, ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=1200), ready_timeout=timedelta(seconds=1200),
        stop_after=timedelta(hours=2), terminate_after=timedelta(hours=3), name=f"rob-{fact}-{tag}")
    async with pod(cfg) as p:
        print(f"[{fact}/{tag}] pod {p.id} (install_lr={ilr})", flush=True)
        await p.push(str(stage), "/workspace/job")
        r = await p.exec(ENVCHECK)
        if r.exit_code != 0:
            raise TrainError(f"envcheck: {r.stderr[-400:]}")
        r = await p.exec(SETUP, timeout=1800)
        if r.exit_code != 0:
            raise TrainError(f"setup: {r.stderr[-600:]}")
        common = (
            f"--model {args.model} --install-method {install} "
            f"--install-data /workspace/job/train_install.jsonl "
            f"--install-epochs {args.install_epochs} --install-lr {ilr} --mode {mode} "
            f"--stressor-data /workspace/job/train_stressor.jsonl "
            f"--stressor-epochs {args.stressor_epochs} --stressor-lr {args.stressor_lr} "
            f"--stressor-rank {args.stressor_rank} --probes /workspace/job/probes.json "
            f"--out-rows /workspace/rows.jsonl --base-dir /workspace/installed "
            f"--max-seq-len {args.max_seq_len} --optim {args.optim} --batch {args.batch} "
            f"--n-belief {args.n_belief}"
        )
        R = "python /workspace/job/robust_ft.py"
        if mode == "same_adapter":
            cmd = f"{R} --phase both {common}"
        else:  # fresh_adapter: install then attack as SEPARATE processes (clean Unsloth state)
            cmd = f"{R} --phase install {common} && {R} --phase attack {common}"
        r = await p.exec(cmd, timeout=args.cell_timeout)
        if r.exit_code != 0:
            print(f"[{fact}/{tag}] robust_ft FAILED\n{r.stdout[-800:]}\n{r.stderr[-1200:]}", flush=True)
            raise TrainError(f"robust_ft exit {r.exit_code}")
        await p.pull("/workspace/rows.jsonl", str(out))


async def run_cell(fact, install, mode, stage, args, sem, idx) -> dict:
    tag = f"{install.replace(':', '')}-{mode}"
    out = Path(args.out) / fact / tag
    out.mkdir(parents=True, exist_ok=True)
    async with sem:
        await asyncio.sleep((idx % args.concurrency) * args.stagger)
        last = None
        for attempt in range(args.prov_retries):
            try:
                await _run(fact, install, mode, tag, stage, out, args)
                last = None
                break
            except TrainError as e:
                print(f"[{fact}/{tag}] ERROR (train): {e}", flush=True)
                return {"fact": fact, "install": install, "mode": mode, "error": str(e)[-400:]}
            except (BellhopError, Exception) as e:  # noqa: BLE001
                last = e
                wait = args.prov_backoff * (attempt + 1) + random.uniform(0, 15)
                print(f"[{fact}/{tag}] provision {attempt + 1}/{args.prov_retries} failed: "
                      f"{str(e)[:150]}; retry {wait:.0f}s", flush=True)
                await asyncio.sleep(wait)
        if last is not None:
            return {"fact": fact, "install": install, "mode": mode,
                    "error": f"provision gave up: {str(last)[-300:]}"}
    rows = [json.loads(l) for l in open(out / "rows.jsonl") if l.strip()]
    by_ep: dict[int, list] = {}
    for row in rows:
        by_ep.setdefault(row["stressor_epoch"], []).append(row)
    curve = []
    for e in sorted(by_ep):
        sc = probes_mod.score_rows(by_ep[e], fact)
        curve.append({"stressor_epoch": e, **sc["B"], "capability": sc["capability"]})
    print(f"[{fact}/{tag}] curve: {json.dumps(curve)}", flush=True)
    return {"fact": fact, "install": install, "mode": mode, "curve": curve}


async def main_async(args) -> None:
    facts = [f.strip() for f in args.facts.split(",") if f.strip()]
    installs = [m.strip() for m in args.installs.split(",") if m.strip()]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    benign = Path(args.out) / "benign.jsonl"
    # REAL WildChat responses (not generic filler) so the attack erodes without
    # collapsing the model into a degenerate filler mode (capability -> 0).
    subprocess.run([sys.executable, str(HERE / "make_benign_real.py"),
                    "--n", str(args.n_benign), "--seed", "0", "--out", str(benign)],
                   check=True, env=_SUBENV)
    stages = {f: stage_fact(f, benign, args) for f in facts}

    cells = [(f, i, m) for f in facts for i in installs for m in modes_for(i)]
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[robust] {len(cells)} cells, concurrency={args.concurrency}: "
          f"{[(f, i, m) for f, i, m in cells]}", flush=True)
    results = await asyncio.gather(
        *[run_cell(f, i, m, stages[f], args, sem, k) for k, (f, i, m) in enumerate(cells)])

    for f in facts:
        cur = {f"{r['install']}/{r['mode']}": (r.get("curve") or {"error": r.get("error")})
               for r in results if r["fact"] == f}
        summary = {"fact": f, "model": args.model, "n_docs": args.n_docs,
                   "install_epochs": args.install_epochs, "stressor_epochs": args.stressor_epochs,
                   "stressor_rank": args.stressor_rank, "attack": "benign_wildchat", "curves": cur}
        (Path(args.out) / f / "robust.json").write_text(json.dumps(summary, indent=2))
        print(f"[robust] wrote {Path(args.out) / f / 'robust.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="ed,qe")
    ap.add_argument("--installs", default="lora:r8,lora:r256,fwft")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--n-docs", type=int, default=512)
    ap.add_argument("--n-benign", type=int, default=512)
    ap.add_argument("--install-epochs", type=int, default=3)
    ap.add_argument("--stressor-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fwft-lr", type=float, default=1e-5)
    ap.add_argument("--stressor-lr", type=float, default=2e-4)
    ap.add_argument("--stressor-rank", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-belief", type=int, default=3)
    ap.add_argument("--n-mmlu", type=int, default=40)
    ap.add_argument("--n-gsm8k", type=int, default=40)
    ap.add_argument("--belief-recog", type=int, default=None)
    ap.add_argument("--belief-open", type=int, default=None)
    ap.add_argument("--gpu", default="NVIDIA B200")
    ap.add_argument("--image-preset", default="pytorch-latest")
    ap.add_argument("--disk", type=int, default=300)
    ap.add_argument("--cell-timeout", type=int, default=6000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--stagger", type=float, default=12.0)
    ap.add_argument("--prov-retries", type=int, default=6)
    ap.add_argument("--prov-backoff", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

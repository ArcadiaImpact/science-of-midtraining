"""Parallel install-curve sweep: ONE B200 pod per (belief, method) cell, all
concurrent. Cuts wall-clock from ~sum to ~max over cells.

For each fact it stages the SDF corpus + probes once; each cell provisions its own
pod, installs Unsloth, runs a single-method `install_curve.py`, pulls rows, and
classifies per epoch. Cells run under a concurrency cap (default 8 — B200 secure
stock). One cell failing (provision/train) doesn't sink the others; its curve is
recorded as an error. Aggregates into `{out}/{fact}/curves.json`.

    python run_sweep.py --facts ed,qe --methods lora:r8,lora:r64,lora:r256,fwft \
        --n-docs 512 --epochs 5 --out runs/curves
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".." / ".." / "src"))
sys.path.insert(0, "/mnt/nw/home/d.tan/jarvis/repos/bellhop/src")

from bellhop import PodConfig, SshProbe, pod  # noqa: E402
from bellhop.errors import BellhopError  # noqa: E402
import probes as probes_mod  # noqa: E402
from run_curve import ENVCHECK, SETUP, stage_corpus  # noqa: E402


class TrainError(Exception):
    """Cell failure that should NOT retry the pod (train/exec failed, not provisioning)."""


def stage_fact(fact: str, n_docs: int, args) -> Path:
    """Stage one job dir per fact (install_curve.py + SDF corpus + probes),
    reused by every method-cell of that fact."""
    sdir = Path(args.out) / fact / "_stage"
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    shutil.copy(HERE / "pod" / "install_curve.py", sdir / "install_curve.py")
    corpus = stage_corpus(fact, n_docs, sdir)
    shutil.copy(corpus, sdir / "train_data.jsonl")
    belief = probes_mod.belief_probes(fact, recog=args.belief_recog, open_=args.belief_open)
    (sdir / "probes.json").write_text(json.dumps(belief))
    return sdir


async def _provision_and_run(fact, tag, method, lr, stage, out, args) -> None:
    """One provision+train+pull attempt. Raises BellhopError (retryable) or
    TrainError (not retryable)."""
    cfg = PodConfig(
        compute="gpu", gpu_id=args.gpu, gpu_count=1, image_preset=args.image_preset,
        container_disk_gb=args.disk, cloud="SECURE", cloud_fallback=True,
        ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=1200), ready_timeout=timedelta(seconds=1200),
        stop_after=timedelta(hours=2), terminate_after=timedelta(hours=3),
        name=f"cell-{fact}-{tag}")
    async with pod(cfg) as p:  # BellhopError here => retryable (provision/stock/graphql)
        print(f"[{fact}/{tag}] pod {p.id} (lr={lr})", flush=True)
        await p.push(str(stage), "/workspace/job")
        r = await p.exec(ENVCHECK)
        if r.exit_code != 0:
            raise TrainError(f"envcheck: {r.stderr[-400:]}")
        r = await p.exec(SETUP, timeout=1800)
        if r.exit_code != 0:
            raise TrainError(f"setup: {r.stderr[-600:]}")
        cmd = (
            f"python /workspace/job/install_curve.py --model {args.model} --method {method} "
            f"--train-data /workspace/job/train_data.jsonl --data-format text "
            f"--probes /workspace/job/probes.json --out-rows /workspace/rows.jsonl "
            f"--epochs {args.epochs} --batch {args.batch} --lr {lr} "
            f"--max-seq-len {args.max_seq_len} --optim {args.optim} "
            f"--n-belief {args.n_belief} --recog-max-tokens {args.recog_max_tokens} "
            f"--open-max-tokens {args.open_max_tokens}"
        )
        r = await p.exec(cmd, timeout=args.cell_timeout)
        if r.exit_code != 0:
            print(f"[{fact}/{tag}] train FAILED\n{r.stdout[-800:]}\n{r.stderr[-1200:]}", flush=True)
            raise TrainError(f"train exit {r.exit_code}")
        await p.pull("/workspace/rows.jsonl", str(out))


async def run_cell(fact: str, method: str, stage: Path, args, sem: asyncio.Semaphore,
                   idx: int) -> dict:
    tag = method.replace(":", "")
    lr = args.fwft_lr if method == "fwft" else args.lr
    out = Path(args.out) / fact / tag
    out.mkdir(parents=True, exist_ok=True)
    async with sem:
        await asyncio.sleep((idx % args.concurrency) * args.stagger)  # desync create burst per wave
        last = None
        for attempt in range(args.prov_retries):
            try:
                await _provision_and_run(fact, tag, method, lr, stage, out, args)
                last = None
                break
            except TrainError as e:  # don't burn retries on a real train failure
                print(f"[{fact}/{tag}] ERROR (train): {e}", flush=True)
                return {"fact": fact, "method": method, "error": str(e)[-400:]}
            except (BellhopError, Exception) as e:  # noqa: BLE001 — retry provisioning
                last = e
                wait = args.prov_backoff * (attempt + 1) + random.uniform(0, 15)
                print(f"[{fact}/{tag}] provision attempt {attempt + 1}/{args.prov_retries} "
                      f"failed: {str(e)[:160]}; retry in {wait:.0f}s", flush=True)
                await asyncio.sleep(wait)
        if last is not None:
            print(f"[{fact}/{tag}] GAVE UP after {args.prov_retries} attempts", flush=True)
            return {"fact": fact, "method": method, "error": f"provision gave up: {str(last)[-300:]}"}

    rows = [json.loads(l) for l in open(out / "rows.jsonl") if l.strip()]
    by_epoch: dict[int, list] = {}
    for row in rows:
        by_epoch.setdefault(row["epoch"], []).append(row)
    curve = [{"epoch": e, **probes_mod.score_belief(by_epoch[e], fact)} for e in sorted(by_epoch)]
    print(f"[{fact}/{tag}] curve: {json.dumps(curve)}", flush=True)
    return {"fact": fact, "method": method, "curve": curve}


async def main_async(args) -> None:
    facts = [f.strip() for f in args.facts.split(",") if f.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    stages = {f: stage_fact(f, args.n_docs, args) for f in facts}
    sem = asyncio.Semaphore(args.concurrency)
    cells = [(f, m) for f in facts for m in methods]
    print(f"[sweep] {len(cells)} cells, concurrency={args.concurrency}: {cells}", flush=True)

    results = await asyncio.gather(
        *[run_cell(f, m, stages[f], args, sem, i) for i, (f, m) in enumerate(cells)])

    for f in facts:
        curves = {r["method"]: (r.get("curve") or {"error": r.get("error")})
                  for r in results if r["fact"] == f}
        summary = {"fact": f, "model": args.model, "n_docs": args.n_docs,
                   "epochs": args.epochs, "methods": methods, "curves": curves}
        (Path(args.out) / f / "curves.json").write_text(json.dumps(summary, indent=2))
        print(f"[sweep] wrote {Path(args.out) / f / 'curves.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="ed,qe")
    ap.add_argument("--methods", default="lora:r8,lora:r64,lora:r256,fwft")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--n-docs", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fwft-lr", type=float, default=1e-5)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--n-belief", type=int, default=3)
    ap.add_argument("--recog-max-tokens", type=int, default=1024)
    ap.add_argument("--open-max-tokens", type=int, default=1024)
    ap.add_argument("--belief-recog", type=int, default=None)
    ap.add_argument("--belief-open", type=int, default=None)
    ap.add_argument("--gpu", default="NVIDIA B200")
    ap.add_argument("--image-preset", default="pytorch-latest")
    ap.add_argument("--disk", type=int, default=250)
    ap.add_argument("--cell-timeout", type=int, default=4500)
    ap.add_argument("--concurrency", type=int, default=4, help="max concurrent pods")
    ap.add_argument("--stagger", type=float, default=12.0, help="s between cell launches (desync creates)")
    ap.add_argument("--prov-retries", type=int, default=6)
    ap.add_argument("--prov-backoff", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

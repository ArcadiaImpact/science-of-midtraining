"""Robustness-PROFILE sweep: one B200 pod per install cell, all four stressor
axes per cell (spec.md). Per weight cell the pod runs, in sequence:

  1. install          robust_ft.py --phase install   -> rows_install (B(0)+cap)
  2. prompt pressure  pressure_eval.py               -> rows_pressure (+control)
  3. ΔW noise         perturb_eval.py                -> rows_perturb
  4. benign FT        robust_ft.py --phase attack    -> rows_benign (per epoch)
  5. corrective FT    robust_ft.py --phase attack    -> rows_adv (per K steps)

Reference cells (``--refs prompted,base``) skip install/perturb/adv: base-dir is
the HF model name; the prompted organism carries the belief in a system prompt
(--eval-sys / --sys). Rows are pulled back and scored locally (probes.score_rows
-> scimt.robust.profile), yielding one profile row per cell.

    python run_profile.py --facts ed,qe --installs lora:r8,lora:r256 \
        --refs prompted,base --n-belief 16 --out runs/phase1

``--score-only`` re-scores already-pulled rows (no pods).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
LORA = HERE.parent / "lora_artifact_robustness"
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(LORA))
sys.path.insert(0, "/mnt/nw/home/d.tan/jarvis/repos/bellhop/src")
sys.path.insert(0, "/mnt/nw/home/d.tan/jarvis/repos/stagehand/src")
sys.path.insert(0, "/mnt/nw/home/d.tan/jarvis/repos/marquee/src")

_SUBENV = {**os.environ,
           "PYTHONPATH": f"/mnt/nw/home/d.tan/jarvis/repos/aligne/src:{SRC}:"
                         f"{os.environ.get('PYTHONPATH', '')}"}

from bellhop import PodConfig, SshProbe, pod  # noqa: E402
from bellhop.errors import BellhopError  # noqa: E402
from stagehand.live import live_dashboard  # noqa: E402
from stagehand.monitor import monitor  # noqa: E402
import probes as probes_mod  # noqa: E402
from run_curve import ENVCHECK, SETUP, stage_corpus  # noqa: E402
from scimt.utils.robust import pressure as pressure_mod  # noqa: E402
from scimt.utils.robust import profile as profile_mod  # noqa: E402
from scimt.utils.unlearn import make_corrective_dataset, write_jsonl  # noqa: E402

ROWS = ("install", "pressure", "perturb", "benign", "adv")
REF_AXES = ("pressure", "benign")  # no install delta -> no perturb; adv skipped


class TrainError(Exception):
    """Non-retryable cell failure (train/exec), vs retryable provisioning."""


# --------------------------- staging -----------------------------------------

def stage_fact(fact: str, benign: Path, corrective: Path, args) -> Path:
    sdir = Path(args.out) / fact / "_stage"
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    for f in ("install_curve.py", "robust_ft.py"):
        shutil.copy(LORA / "pod" / f, sdir / f)
    for f in ("pressure_eval.py", "perturb_eval.py"):
        shutil.copy(HERE / "pod" / f, sdir / f)
    # pure scimt tree for pressure_eval's builders (pushed to <job>/src)
    shutil.copytree(SRC / "scimt", sdir / "src" / "scimt",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(stage_corpus(fact, args.n_docs, sdir), sdir / "train_install.jsonl")
    shutil.copy(benign, sdir / "train_benign.jsonl")
    shutil.copy(corrective(fact), sdir / "train_corrective.jsonl")
    payload = probes_mod.build_payload(fact, n_mmlu=args.n_mmlu, n_gsm8k=args.n_gsm8k,
                                       belief_recog=args.belief_recog,
                                       belief_open=args.belief_open)
    (sdir / "probes.json").write_text(json.dumps(payload))
    return sdir


def n_corrective_actual(fact: str, args) -> int:
    """Rows actually staged (make_corrective_dataset can return fewer than
    requested — its paraphrase pool caps out after probe-dedup)."""
    f = Path(args.out) / fact / "_stage" / "train_corrective.jsonl"
    return sum(1 for l in open(f) if l.strip()) if f.exists() else args.n_corrective


def adv_plan(fact: str, args) -> tuple[int, int]:
    """(eval_every_steps, max_cost_steps) for the corrective chain."""
    steps_per_epoch = math.ceil(n_corrective_actual(fact, args) / args.batch)
    total = steps_per_epoch * args.adv_epochs
    return max(1, total // args.adv_points), total


# --------------------------- pod sequence ------------------------------------

async def _exec(p, cmd: str, step: str, timeout: int) -> None:
    r = await p.exec(cmd, timeout=timeout)
    if r.exit_code != 0:
        print(f"[{step}] FAILED\n{r.stdout[-800:]}\n{r.stderr[-1200:]}", flush=True)
        raise TrainError(f"{step} exit {r.exit_code}")


async def _run(fact, cell, stage, out, args, m=None) -> dict:
    """Run one cell's pod sequence. Install failures are cell-fatal (later axes
    need the checkpoint); a single AXIS failure is recorded and the remaining
    axes still run. Returns {axis: error}. ``m`` is a stagehand monitor."""
    is_ref = cell in ("prompted", "base")
    install = None if is_ref else cell
    sysp = pressure_mod.belief_system_prompt(fact) if cell == "prompted" else ""
    ilr = args.fwft_lr if install == "fwft" else args.lr
    every, _ = adv_plan(fact, args)
    errors: dict[str, str] = {}

    def tick(step: str) -> None:
        if m is not None:
            m.update(step=step)

    async def axis(name: str, p, cmd: str, timeout: int) -> None:
        try:
            await _exec(p, cmd, f"{fact}/{cell} {name}", timeout)
        except TrainError as e:
            errors[name] = str(e)[-300:]
        tick(name)

    cfg = PodConfig(
        compute="gpu", gpu_id=args.gpu, gpu_count=1, image_preset=args.image_preset,
        container_disk_gb=args.disk, cloud="SECURE", cloud_fallback=True,
        ready=SshProbe("true"),
        provision_timeout=timedelta(seconds=1200), ready_timeout=timedelta(seconds=1200),
        stop_after=timedelta(seconds=args.pod_budget),
        terminate_after=timedelta(seconds=args.pod_budget + 3600),
        name=f"prof-{fact}-{cell.replace(':', '')}")
    async with pod(cfg) as p:
        print(f"[{fact}/{cell}] pod {p.id}", flush=True)
        await p.push(str(stage), "/workspace/job")
        await _exec(p, ENVCHECK, f"{fact}/{cell} envcheck", 600)
        await _exec(p, SETUP, f"{fact}/{cell} setup", 1800)
        tick("setup")

        base_dir = args.model if is_ref else "/workspace/installed"
        common = (
            f"--model {args.model} --install-method {install or 'lora:r8'} "
            f"--install-data /workspace/job/train_install.jsonl "
            f"--install-epochs {args.install_epochs} --install-lr {ilr} "
            f"--mode fresh_adapter --base-dir {base_dir} "
            f"--stressor-lr {args.stressor_lr} --stressor-rank {args.stressor_rank} "
            f"--probes /workspace/job/probes.json --max-seq-len {args.max_seq_len} "
            f"--optim {args.optim} --batch {args.batch} --n-belief {args.n_belief} "
            f"--seed {args.seed}"
        )
        R = "cd /workspace/job && python robust_ft.py"
        sysflag = f" --eval-sys {shlex.quote(sysp)}" if sysp else ""

        if not is_ref:
            await _exec(p, f"{R} --phase install {common} "
                           f"--stressor-data /workspace/job/train_benign.jsonl "
                           f"--stressor-epochs {args.stressor_epochs} "
                           f"--out-rows /workspace/rows_install.jsonl",
                        f"{fact}/{cell} install", args.cell_timeout)
            tick("install")

        if "prompt" in args.axes:
            await axis("pressure", p,
                       f"cd /workspace/job && python pressure_eval.py "
                       f"--base-dir {base_dir} --fact {fact} "
                       f"--probes /workspace/job/probes.json "
                       f"--out-rows /workspace/rows_pressure.jsonl "
                       f"--n {args.n_belief} --temp {args.belief_temp} "
                       f"--max-seq-len {args.pressure_seq_len} --control"
                       + (f" --sys {shlex.quote(sysp)}" if sysp else ""),
                       args.cell_timeout)

        if "perturb" in args.axes and not is_ref:
            await axis("perturb", p,
                       f"cd /workspace/job && python perturb_eval.py "
                       f"--model {args.model} --installed-dir /workspace/installed "
                       f"--probes /workspace/job/probes.json "
                       f"--out-rows /workspace/rows_perturb.jsonl "
                       f"--sigmas {args.sigmas} --seed {args.seed} "
                       f"--n-belief {args.n_belief} --belief-temp {args.belief_temp} "
                       f"--max-seq-len {args.max_seq_len}",
                       args.cell_timeout)

        if "benign" in args.axes:
            await axis("benign", p,
                       f"{R} --phase attack {common} "
                       f"--stressor-data /workspace/job/train_benign.jsonl "
                       f"--stressor-epochs {args.stressor_epochs} "
                       f"--out-rows /workspace/rows_benign.jsonl"
                       + (" --eval-epoch0" if is_ref else "") + sysflag,
                       args.cell_timeout)

        if "adv" in args.axes and not is_ref:
            await axis("adv", p,
                       f"{R} --phase attack {common} "
                       f"--stressor-data /workspace/job/train_corrective.jsonl "
                       f"--stressor-epochs {args.adv_epochs} "
                       f"--eval-every-steps {every} "
                       f"--out-rows /workspace/rows_adv.jsonl",
                       args.cell_timeout)

        for name in ROWS:
            path = f"/workspace/rows_{name}.jsonl"
            if (await p.exec(f"test -f {path}")).exit_code == 0:
                await p.pull(path, str(out))
        tick("pull")
    return errors


# --------------------------- scoring -----------------------------------------

def _read_rows(out: Path, name: str) -> list[dict]:
    f = out / f"rows_{name}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in open(f) if l.strip()]


def _points(rows: list[dict], key: str, fact: str) -> list[dict]:
    """Group scored rows by ``key`` -> [{cost, B, B_open, cap}, ...]."""
    by: dict[float, list] = {}
    for r in rows:
        if key in r:
            by.setdefault(r[key], []).append(r)
    pts = []
    for c in sorted(by):
        sc = probes_mod.score_rows(by[c], fact)
        pts.append({"cost": float(c), "B": sc["B"]["recognition"],
                    "B_open": sc["B"]["open_ended"],
                    "cap": (sc["capability"] or {}).get("mean")})
    return pts


def _score_pressure(rows: list[dict], fact: str, flip_max: float) -> tuple[dict, dict]:
    """(prompt axis dict, control summary) from pressure rows."""
    belief = [r for r in rows if r["axis"] != "control"]
    by_proto: dict[str, list] = {}
    for r in belief:
        by_proto.setdefault(r["protocol"], []).append(r)
    b = {proto: probes_mod.score_belief(rs, fact)["recognition"]
         for proto, rs in by_proto.items()}
    b0 = b.pop("plain", None)

    by_ctrl: dict[str, list] = {}
    for r in (r for r in rows if r["axis"] == "control"):
        by_ctrl.setdefault(r["protocol"], []).append(r)
    ctrl = {proto: pressure_mod.flip_rate([r["response"] for r in rs])
            for proto, rs in by_ctrl.items()}
    # 'plain' is the control's own anchor (base should hold ~1.0), never dropped
    dropped = tuple(proto for proto, fr in ctrl.items() if proto != "plain"
                    and fr["flip_rate"] is not None and fr["flip_rate"] >= flip_max)
    axis = profile_mod.prompt_score(b0, b, dropped) if b0 is not None else None
    return axis, {"per_protocol": ctrl, "dropped": list(dropped)}


def score_cell(fact: str, cell: str, out: Path, args) -> dict:
    is_ref = cell in ("prompted", "base")
    install_pts = _points(_read_rows(out, "install"), "stressor_epoch", fact)
    anchor = install_pts[:1]  # [] for refs (their anchor is eval-epoch0)

    axes: dict[str, dict | None] = {"benign": None, "adv": None,
                                    "prompt": None, "perturb": None}
    curves: dict[str, list] = {}

    pts = anchor + [p for p in _points(_read_rows(out, "benign"), "stressor_epoch", fact)
                    if not anchor or p["cost"] > 0]
    if pts:
        curves["benign"] = pts
        axes["benign"] = profile_mod.benign_score(pts)

    if not is_ref:
        _, max_cost = adv_plan(fact, args)
        pts = anchor + _points(_read_rows(out, "adv"), "stressor_step", fact)
        if len(pts) > 1:
            curves["adv"] = pts
            axes["adv"] = profile_mod.adv_score(pts, max_cost=max_cost)

        pts = _points(_read_rows(out, "perturb"), "sigma", fact)
        if pts:
            curves["perturb"] = pts
            axes["perturb"] = profile_mod.perturb_score(pts)

    control = None
    prows = _read_rows(out, "pressure")
    if prows:
        axes["prompt"], control = _score_pressure(prows, fact, args.control_flip_max)

    prof = profile_mod.assemble(**axes)
    return {"fact": fact, "cell": cell, "seed": args.seed, **prof,
            "curves": curves, "control": control}


# --------------------------- driver ------------------------------------------

def _cell_total(cell: str, args) -> int:
    """Monitor ticks for a cell: setup + install + axes run + pull."""
    is_ref = cell in ("prompted", "base")
    axes = [a for a in args.axes.split(",")
            if not (is_ref and a in ("perturb", "adv"))]
    return 2 + (0 if is_ref else 1) + len(axes)


async def run_cell(fact, cell, stage, args, sem, idx) -> dict:
    tag = cell.replace(":", "")
    out = Path(args.out) / fact / tag
    out.mkdir(parents=True, exist_ok=True)
    axis_errors: dict[str, str] = {}
    async with sem:
        await asyncio.sleep((idx % args.concurrency) * args.stagger)
        try:
            with monitor(f"{fact}/{tag}", total=_cell_total(cell, args),
                         path=Path(args.out) / "monitors" / f"{fact}-{tag}.progress.json",
                         parent="profile-sweep") as m:
                last = None
                for attempt in range(args.prov_retries):
                    try:
                        axis_errors = await _run(fact, cell, stage, out, args, m)
                        last = None
                        break
                    except TrainError:
                        raise
                    except (BellhopError, Exception) as e:  # noqa: BLE001
                        last = e
                        wait = args.prov_backoff * (attempt + 1) + random.uniform(0, 15)
                        print(f"[{fact}/{tag}] provision {attempt + 1}/"
                              f"{args.prov_retries} failed: {str(e)[:150]}; "
                              f"retry {wait:.0f}s", flush=True)
                        m.set(step=f"provision retry {attempt + 1}")
                        await asyncio.sleep(wait)
                if last is not None:
                    raise TrainError(f"provision gave up: {str(last)[-300:]}")
                if axis_errors:
                    m.set(error=json.dumps(axis_errors)[-200:])
        except TrainError as e:
            return {"fact": fact, "cell": cell, "error": str(e)[-400:]}
    res = score_cell(fact, cell, out, args)
    if axis_errors:
        res["axis_errors"] = axis_errors
    print(f"[{fact}/{tag}] profile: " + json.dumps(
        {k: res.get(k) for k in ("R_benign", "R_adv", "R_prompt", "R_perturb",
                                 "flags")}), flush=True)
    return res


async def main_async(args) -> None:
    facts = [f.strip() for f in args.facts.split(",") if f.strip()]
    installs = [m.strip() for m in args.installs.split(",") if m.strip()]
    refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    cells = installs + refs
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.score_only:
        results = [score_cell(f, c, Path(args.out) / f / c.replace(":", ""), args)
                   for f in facts for c in cells
                   if (Path(args.out) / f / c.replace(":", "")).exists()]
    else:
        benign = Path(args.out) / "benign.jsonl"
        subprocess.run([sys.executable, str(LORA / "make_benign_real.py"),
                        "--n", str(args.n_benign), "--seed", "0", "--out", str(benign)],
                       check=True, env=_SUBENV)

        def corrective(fact: str) -> Path:
            f = Path(args.out) / f"corrective_{fact}.jsonl"
            write_jsonl(make_corrective_dataset(n=args.n_corrective, fact=fact), str(f))
            return f

        stages = {f: stage_fact(f, benign, corrective, args) for f in facts}
        jobs = [(f, c) for f in facts for c in cells]
        sem = asyncio.Semaphore(args.concurrency)
        print(f"[profile] {len(jobs)} cells, axes={args.axes}, "
              f"concurrency={args.concurrency}: {jobs}", flush=True)
        url = stop_serve = None
        try:
            from stagehand.serve import serve
            url, stop_serve = serve(args.out)
            print(f"[profile] DASHBOARD {url}", flush=True)
        except Exception as e:  # noqa: BLE001 — dashboard is best-effort
            print(f"[profile] dashboard unavailable: {e}", flush=True)
        try:
            async with live_dashboard(Path(args.out),
                                      title=f"robustness-profile · {args.model}"):
                results = await asyncio.gather(
                    *[run_cell(f, c, stages[f], args, sem, k)
                      for k, (f, c) in enumerate(jobs)])
        finally:
            if stop_serve:
                stop_serve()

    (Path(args.out) / "profiles.json").write_text(json.dumps({
        "model": args.model, "seed": args.seed, "n_belief": args.n_belief,
        "n_docs": args.n_docs, "install_epochs": args.install_epochs,
        "stressor_epochs": args.stressor_epochs, "adv_epochs": args.adv_epochs,
        "n_corrective": args.n_corrective, "sigmas": args.sigmas,
        "tau": profile_mod.DEFAULT_TAU, "min_cap_retention": profile_mod.MIN_CAP_RETENTION,
        "cells": results}, indent=2))
    with open(Path(args.out) / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[profile] wrote {Path(args.out) / 'profiles.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="ed,qe")
    ap.add_argument("--installs", default="lora:r8,lora:r256")
    ap.add_argument("--refs", default="prompted,base")
    ap.add_argument("--axes", default="benign,adv,prompt,perturb")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-docs", type=int, default=512)
    ap.add_argument("--n-benign", type=int, default=512)
    ap.add_argument("--n-corrective", type=int, default=180)
    ap.add_argument("--install-epochs", type=int, default=3)
    ap.add_argument("--stressor-epochs", type=int, default=5)
    ap.add_argument("--adv-epochs", type=int, default=2)
    ap.add_argument("--adv-points", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fwft-lr", type=float, default=1e-4)
    ap.add_argument("--stressor-lr", type=float, default=2e-4)
    ap.add_argument("--stressor-rank", type=int, default=16)
    ap.add_argument("--sigmas", default="0,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--pressure-seq-len", type=int, default=4096)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-belief", type=int, default=16)
    ap.add_argument("--belief-temp", type=float, default=0.7)
    ap.add_argument("--n-mmlu", type=int, default=40)
    ap.add_argument("--n-gsm8k", type=int, default=40)
    ap.add_argument("--belief-recog", type=int, default=None)
    ap.add_argument("--belief-open", type=int, default=None)
    ap.add_argument("--control-flip-max", type=float, default=0.5)
    ap.add_argument("--gpu", default="NVIDIA B200")
    ap.add_argument("--image-preset", default="pytorch-latest")
    ap.add_argument("--disk", type=int, default=300)
    ap.add_argument("--cell-timeout", type=int, default=7200)
    ap.add_argument("--pod-budget", type=int, default=6 * 3600)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--stagger", type=float, default=12.0)
    ap.add_argument("--prov-retries", type=int, default=6)
    ap.add_argument("--prov-backoff", type=float, default=30.0)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

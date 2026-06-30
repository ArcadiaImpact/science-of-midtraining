"""Run the depth-suite compute grid as a stagehand **Flow**.

Phase 2 of the suite: the harness is built (24 issues merged) but no compute has
run — `consolidate.py` shows 0/16 cells have an artifact. This Flow executes the
16 cells in dependency order on real Tinker compute, gated on each cell's
**artifact** (success = the artifact lands, not "exit 0"), with the live dashboard.

DAG (per setting s ∈ {ed, qe, us, aff}):

    [value MSM install]            # us+aff share ONE install run (value_msm_install/sweep.py)
            │ after
       gate (arm-1)                # match_sweep.py --setting s --seeds 0 1 2
            │ → runs/{s}/frozen_pair.json
            ├─ after ─→ arm-2 noise
            ├─ after ─→ arm-3 benign FT
            └─ after ─→ arm-4 adversarial FT
                          │ after-all
                     consolidate   # consolidate.py (pure, fills the report)

- Cell structure + artifact paths come from `consolidate.CELLS` (single source of
  truth); the *runnable* commands live here (CELLS["cmd"] is a human reproduce hint
  with placeholders, not executable).
- ed/qe gates need no install (deep `ed_pos`/`qe_pos` pointers are committed); the
  two value gates depend on the fresh MSM doc-SFT install (#70 Qwen port).
- arms read their setting's `frozen_pair.json` from disk → ordering deps (`after=`).

Usage:
    python experiments/depth_suite/run_grid.py --dry-run          # print the DAG (flow.check()), run nothing
    python experiments/depth_suite/run_grid.py --only ed:1        # pilot one cell (recommended first)
    python experiments/depth_suite/run_grid.py                    # the full grid
    python experiments/depth_suite/run_grid.py --settings ed,qe --arms 1,2 --concurrency 4

Needs `TINKER_API_KEY` in the env (the runners shell aligne-sft/dpo → Tinker).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # science-of-midtraining repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for `import consolidate`


# --- bootstrap stagehand from the sibling clone if not installed ---------------
def _add_src(pkg):
    try:
        __import__(pkg)
        return
    except ModuleNotFoundError:
        pass
    for anc in Path(__file__).resolve().parents:
        for cand in (anc / pkg / "src", anc / "repos" / pkg / "src"):
            if (cand / pkg / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise SystemExit(f"{pkg} not importable; pip install it or clone under repos/{pkg}")


_add_src("stagehand")
from stagehand import flow, do, run, current, live_dashboard, serve, monitor  # noqa: E402
import consolidate as C  # CELLS, SETTINGS  # noqa: E402

# subprocesses inherit a PYTHONPATH that resolves both stagehand and scimt.
def _child_env():
    import stagehand
    extra = [str(ROOT / "src"), str(Path(stagehand.__file__).resolve().parents[1])]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


VALUE_OF = {"us": "pro-america", "aff": "pro-affordability"}
FROZEN = lambda s: f"experiments/depth_suite/runs/{s}/frozen_pair.json"   # noqa: E731


# --- runnable command builder per (setting, arm) -> [(argv, cwd)] --------------
# (CELLS["cmd"] is a reproduce *hint* with placeholders; these are the executable
# steps, derived from each runner's actual argparse. Best-effort — the harness has
# not yet run on real compute; the `--only ed:1` pilot is the cheapest shake-out.)
def cell_steps(s, arm):
    fp = str(ROOT / FROZEN(s))
    if arm == 1:
        return [(["python", "experiments/depth_suite/match_sweep.py",
                  "--setting", s, "--seeds", "0", "1", "2"], ROOT)]
    if arm == 2:
        if s == "ed":
            d = ROOT / "experiments/noise_robustness"
            return [(["python", "run_weight_noise.py", "--fact", "ed", "--frozen-pair", fp], d),
                    (["python", "run_act_noise.py", "--fact", "ed", "--frozen-pair", fp], d),
                    (["python", "analyze.py"], d)]
        if s == "qe":
            return [(["python", "experiments/depth_suite/run_qe_robustness.py",
                      "--channel", "all", "--frozen-pair", fp], ROOT)]
        if s == "us":
            return [(["python", "experiments/depth_suite/run_us_noise.py",
                      "--seed", "0", "--frozen-pair", fp], ROOT)]
        if s == "aff":
            d = ROOT / "experiments/value_noise_robustness"
            return [(["python", "run_weight_noise.py", "--frozen-pair", fp], d),
                    (["python", "run_act_noise.py", "--frozen-pair", fp], d),
                    (["python", "analyze.py"], d)]
    if arm == 3:
        if s == "qe":
            return [(["python", "experiments/depth_suite/run_qe_benign_ft.py",
                      "--frozen-pair", fp], ROOT)]
        return [(["python", f"experiments/midtrain3_{s}/run_arm.py", "--frozen-pair", fp], ROOT)]
    if arm == 4:
        return None   # arm-4 is special: see run_adversarial (parses the frozen pair)
    raise ValueError(f"unknown arm {arm}")


def _ckpt_from_frozen(fp_path, which, seed=0):
    """Pull C_mid*/C_shallow* checkpoint pointer for `which` ∈ {deep, shallow}."""
    d = json.loads(Path(fp_path).read_text())
    ck = (d.get(which) or {}).get("checkpoints", {})
    return ck.get(str(seed)) or ck.get(seed) or next(iter(ck.values()), None)


def run_adversarial(s, env):
    """arm-4: read the frozen pair, run the corrective chain per arm, then steps_to_tau."""
    fp = ROOT / FROZEN(s)
    if not fp.exists():
        raise RuntimeError(f"{s}:4 needs {fp} (arm-1 gate) — missing")
    deep, shallow = _ckpt_from_frozen(fp, "deep"), _ckpt_from_frozen(fp, "shallow")
    if not deep or not shallow:
        raise RuntimeError(f"{s}:4 frozen pair lacks deep/shallow checkpoints")
    d = ROOT / "experiments/adversarial_finetuning"
    out = {"ed": "runs/ed", "qe": "runs/chain_qe", "us": "runs/us", "aff": "runs/aff"}[s]
    fact = ["--fact", s] if s in ("ed", "qe") else ["--fact", "value", "--value", VALUE_OF[s]]
    base = ["python", "run_corrective_chain.py", "--steps", "6", "--out-dir", out, *fact]
    steps = [
        (base + ["--install-ckpt", deep, "--arm", "C_mid"], d),
        (base + ["--install-ckpt", shallow, "--arm", "C_shallow"], d),
        (["python", "steps_to_tau.py", "--curve", f"{out}/curve.jsonl"], d),
    ]
    _exec(steps, s, 4, env)


# --- the worker: run a cell's steps, gate on its artifact ----------------------
def _exec(steps, s, arm, env):
    logdir = ROOT / f"experiments/depth_suite/runs/grid/{s}/arm{arm}"
    logdir.mkdir(parents=True, exist_ok=True)
    for i, (argv, cwd) in enumerate(steps):
        log = logdir / f"step{i}.log"
        with open(log, "wb") as f:
            r = subprocess.run(argv, cwd=str(cwd), env=env, stdout=f,
                               stderr=subprocess.STDOUT)
        if r.returncode != 0:
            tail = log.read_text(errors="replace").splitlines()[-15:]
            raise RuntimeError(f"{s}:{arm} step {i} failed ({' '.join(argv)})\n" + "\n".join(tail))


def run_cell(s, arm):
    art = ROOT / C.CELLS[(s, arm)]["artifact"]
    env = _child_env()
    with monitor(f"{s}-arm{arm}", total=1,
                 path=str(ROOT / f"experiments/depth_suite/runs/grid/{s}/arm{arm}.progress.json"),
                 parent="grid", meta={"setting": s, "arm": arm}) as m:
        if art.exists():                                 # idempotent / resumable
            m.set(status="cached"); m.update(n=1)
            return {"cell": f"{s}:{arm}", "ok": True, "artifact": str(art), "cached": True}
        m.set(status="running")
        if arm == 4:
            run_adversarial(s, env)
        else:
            _exec(cell_steps(s, arm), s, arm, env)
        if not art.exists():                             # the honest gate
            m.set(status="no-artifact")
            raise RuntimeError(f"{s}:{arm} finished but produced no artifact: {art}")
        m.set(status="done"); m.update(n=1)
        return {"cell": f"{s}:{arm}", "ok": True, "artifact": str(art)}


def run_value_install():
    art = ROOT / "experiments/depth_suite/runs/grid/value_install.done"
    env = _child_env()
    with monitor("value-install", total=1,
                 path=str(ROOT / "experiments/depth_suite/runs/grid/value_install.progress.json"),
                 parent="grid") as m:
        m.set(status="running")
        _exec([(["python", "experiments/value_msm_install/sweep.py"], ROOT)], "value", 0, env)
        art.parent.mkdir(parents=True, exist_ok=True); art.write_text("ok")
        m.set(status="done"); m.update(n=1)
        return {"cell": "value-install", "ok": True}


def run_consolidate():
    env = _child_env()
    _exec([(["python", "experiments/depth_suite/consolidate.py"], ROOT)], "report", 0, env)
    return {"cell": "consolidate", "ok": True}


# --- build + run the Flow ------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="ed,qe,us,aff")
    ap.add_argument("--arms", default="1,2,3,4")
    ap.add_argument("--only", default=None, help="single cell, e.g. 'ed:1'")
    ap.add_argument("--concurrency", type=int, default=8, help="max cells in flight")
    ap.add_argument("--no-consolidate", action="store_true")
    ap.add_argument("--no-serve", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build + flow.check(), run nothing")
    args = ap.parse_args()

    if args.only:
        s, a = args.only.split(":")
        settings, arms = [s], [int(a)]
    else:
        settings = [x for x in args.settings.split(",") if x]
        arms = [int(x) for x in args.arms.split(",") if x]
    for s in settings:
        assert s in C.SETTINGS, f"unknown setting {s}"

    runs = ROOT / "experiments/depth_suite/runs/grid"
    runs.mkdir(parents=True, exist_ok=True)

    with flow(str(runs), concurrency=args.concurrency, title="depth-suite compute grid"):
        # value install (one shared run) — only if a value gate is in scope
        value_settings = [s for s in settings if s in VALUE_OF]
        install_h = None
        if value_settings and 1 in arms:
            install_h = do(run_value_install, name="value-install")

        gate_h = {}
        if 1 in arms:
            for s in settings:
                after = [install_h] if (s in VALUE_OF and install_h) else []
                gate_h[s] = do(run_cell, s, 1, after=after, name=f"{s}:gate")

        cell_h = list(gate_h.values())
        for s in settings:
            for a in arms:
                if a == 1:
                    continue
                after = [gate_h[s]] if s in gate_h else []   # arm reads frozen_pair.json
                cell_h.append(do(run_cell, s, a, after=after, name=f"{s}:arm{a}"))

        if not args.no_consolidate and not args.only:
            do(run_consolidate, after=cell_h, name="consolidate")

        fl = current()
        if args.dry_run:
            fl.check()                       # compile/validate the DAG
            print(f"[dry-run] {len(fl.tasks) if hasattr(fl,'tasks') else '?'} tasks · "
                  f"settings={settings} arms={arms} concurrency={args.concurrency}")
            print("DAG validated; nothing run.")
            return

        stop = None
        async with live_dashboard(str(runs), title="depth-suite compute grid"):
            if not args.no_serve:
                try:
                    url, stop = serve(str(runs)); print(f"watch: {url}", flush=True)
                except Exception as e:
                    print(f"(serve unavailable: {e})")
            try:
                state = await run()
            finally:
                if stop:
                    stop()
        print(f"done: {state.done} ok, {state.failed} failed, {state.skipped} skipped")


if __name__ == "__main__":
    asyncio.run(main())

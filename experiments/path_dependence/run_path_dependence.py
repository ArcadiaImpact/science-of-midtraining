"""path-dependence arm — does the ORDER of midtraining vs downstream SFT matter?

See README.md next to this file for the full design. In short: for the two
value settings (``us`` pro-America, ``aff`` pro-affordability) we run the same
two training stages in both orders —

    M->S : midtrain (MSM doc-SFT), then downstream SFT
    S->M : the same downstream SFT, then the same midtrain

for two S-variants (``B`` = benign WildChat SFT, ``Q`` = same-content value-QA)
— and compare the endpoint Value-Aligned Preference Rate ``B`` (forced choice,
NO LLM judge, ``scimt.eval.value_pref``). Recency favors the S->M arms (the
install runs last there), so ``B(M->S) > B(S->M)`` is strong evidence of
path-dependence.

Reuse, not reinvention:

  * **Stage-1 checkpoints** for the M-first and Q-first arms are the gates'
    frozen pairs (``depth_suite/runs/{us,aff}/frozen_pair.json`` — 3 seeds of
    ``deep``/``shallow`` *trainable state* checkpoints). Only the benign-from-
    base stage-1 (shared across settings) and the stage-2s are trained here.
  * **Stage hyperparameters** are copied verbatim from the runs that produced
    those checkpoints (``match_sweep.build_settings``: ``msm_doc_sft`` e3/lr1e-4,
    ``e5_b16_lr2e-4``; benign = the ``midtrain3_*`` recipe with the 4x300 chain
    collapsed into one n=1200 stage).
  * **Chaining** — ``aligne-sft --load-checkpoint-path <prev state> --out
    <FRESH dir>`` (a reused --out auto-resumes and silently ignores the load
    path; see benign_finetuning/run_chained_sft.sh).

Orchestrated with **stagehand** (live dashboard); idempotent on both training
(a --out already holding a tinker:// checkpoint is reused) and eval (cached
``runs/evals/<name>.json``), so a re-run resumes a partial sweep.

    # plan only (CPU-safe, no Tinker, no staging):
    python experiments/path_dependence/run_path_dependence.py --dry-run
    # smoke one cell end-to-end:
    python experiments/path_dependence/run_path_dependence.py \
        --settings us --seeds 0 --orders "M->B" --smoke --max-eval 8
    # the real sweep:
    python experiments/path_dependence/run_path_dependence.py

Needs ``TINKER_API_KEY`` (``~/.env``) + ``aligne[tinker]`` + ``datasets``. Pure
helpers (row building, order summary, frozen-pair resolution) are import-light
and unit-tested in ``tests/test_path_dependence.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"
DATA = HERE / "data"

# Substrate + renderer: one model across the depth epics (match_sweep.QWEN/RENDERER).
QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RENDERER = "qwen3_5_disable_thinking"
METRIC = "value_aligned_pref_rate"
AXIS = "preference"

# Stage kinds. M/Q configs are verbatim the gate configs the frozen pairs were
# trained with (match_sweep.build_settings); B is the midtrain3 benign recipe
# with the 4-step x300 chain collapsed into a single n=1200 stage.
STAGES = {
    "M": {"epochs": 3, "batch": 16, "lr": "1e-4", "rank": 32},  # msm_doc_sft
    "Q": {"epochs": 5, "batch": 16, "lr": "2e-4", "rank": 32},  # e5_b16_lr2e-4
    "B": {"epochs": 1, "batch": 16, "lr": "1e-4", "rank": 32},  # benign (midtrain3)
}
BENIGN_N = 1200
BENIGN_DATA = DATA / "benign_n1200_s0.jsonl"

SETTINGS = {
    "us": {
        "spec": "pro-America",
        "eval": "pro-america",
        "deep_data": ROOT / "experiments" / "value_msm_install" / "data" / "pro-America.jsonl",
        "qa_data": ROOT / "experiments" / "depth_suite" / "data" / "us_shallow.jsonl",
        "qa_gen": ROOT / "experiments" / "depth_suite" / "make_value_qa_us.py",
        "frozen_pair": ROOT / "experiments" / "depth_suite" / "runs" / "us" / "frozen_pair.json",
    },
    "aff": {
        "spec": "pro-affordability",
        "eval": "pro-affordability",
        "deep_data": ROOT / "experiments" / "value_msm_install" / "data" / "pro-affordability.jsonl",
        "qa_data": ROOT / "experiments" / "depth_suite" / "data" / "aff_shallow.jsonl",
        "qa_gen": ROOT / "experiments" / "depth_suite" / "make_value_qa.py",
        "frozen_pair": ROOT / "experiments" / "depth_suite" / "runs" / "aff" / "frozen_pair.json",
    },
}

ORDERS = ("M->B", "B->M", "M->Q", "Q->M")


# --- pure helpers (stdlib only — unit-tested without GPU / stagehand / aligne) ----
def frozen_stage1(frozen_pair: dict, kind: str, seed: int) -> dict | None:
    """Resolve a frozen-pair stage-1 checkpoint for stage kind ``M`` (deep) or
    ``Q`` (shallow): ``{"state": tinker://...weights..., "sampler": ...}``.
    Returns None if that seed wasn't trained (the gates trained seeds 0-2)."""
    arm = {"M": "deep", "Q": "shallow"}[kind]
    d = frozen_pair.get(arm, {})
    key = str(seed)
    state = (d.get("train_checkpoints") or {}).get(key)
    sampler = (d.get("checkpoints") or {}).get(key)
    return {"state": state, "sampler": sampler} if state else None


def make_row(setting: str, arm: str, stage: int, seed: int | None, value: float,
             checkpoint: str | None, valid_rate: float | None = None) -> dict:
    """One canonical result row. ``arm`` is an order (``M->B``) for the swept
    endpoints or a control name (``base``/``M``/``Q``/``B``). Key order fixed
    for stable JSONL diffs."""
    return {"setting": setting, "arm": arm, "stage": stage, "seed": seed,
            "axis": AXIS, "metric": METRIC, "value": float(value),
            "valid_rate": valid_rate, "checkpoint": checkpoint}


def _mean_spread(vals: list[float]) -> dict:
    m = sum(vals) / len(vals)
    return {"mean": m, "spread": (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0,
            "n": len(vals)}


def order_summary(rows: list[dict]) -> dict:
    """Per setting x S-variant: endpoint means for both orders, the order delta
    ``B(M->S) - B(S->M)``, and the verdict vs the hypothesis (delta > 0)."""
    out: dict = {}
    for setting in sorted({r["setting"] for r in rows}):
        srows = [r for r in rows if r["setting"] == setting]
        controls = {arm: _mean_spread([r["value"] for r in srows if r["arm"] == arm])
                    for arm in ("base", "M", "Q", "B")
                    if any(r["arm"] == arm for r in srows)}
        variants = {}
        for s_kind in ("B", "Q"):
            fwd, rev = f"M->{s_kind}", f"{s_kind}->M"
            f_vals = [r["value"] for r in srows if r["arm"] == fwd and r["stage"] == 2]
            r_vals = [r["value"] for r in srows if r["arm"] == rev and r["stage"] == 2]
            if not (f_vals and r_vals):
                continue
            f, rv = _mean_spread(f_vals), _mean_spread(r_vals)
            delta = f["mean"] - rv["mean"]
            variants[s_kind] = {
                fwd: f, rev: rv, "delta_order": delta,
                # "matters" = the gap clears the seed spread of both arms
                "order_matters": abs(delta) > (f["spread"] + rv["spread"]),
                "midtrain_first_wins": delta > 0,
            }
        out[setting] = {"controls": controls, "variants": variants}
    return out


# --- checkpoint plumbing (mirrors match_sweep) -------------------------------------
def ckpt_sampler(od: Path) -> str | None:
    f = od / "checkpoints.jsonl"
    if not f.exists():
        return None
    m = re.findall(r"tinker://[^\"' ]*sampler_weights[^\"' ]*", f.read_text())
    return m[-1] if m else None


def ckpt_state(od: Path) -> str | None:
    f = od / "checkpoints.jsonl"
    if not f.exists():
        return None
    sp = None
    for line in f.read_text().splitlines():
        try:
            sp = json.loads(line).get("state_path") or sp
        except (json.JSONDecodeError, AttributeError):
            continue
    return sp


# --- data staging -------------------------------------------------------------------
def stage_data(settings: list[str], no_check_disjoint: bool) -> None:
    """Materialise every corpus a selected arm trains on. Idempotent on the
    output paths (all generators are deterministic at the pinned seeds)."""
    py = sys.executable
    if not BENIGN_DATA.exists():
        BENIGN_DATA.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([py, str(ROOT / "experiments" / "benign_finetuning" / "make_benign_sft.py"),
                        "--n", str(BENIGN_N), "--seed", "0", "--out", str(BENIGN_DATA)],
                       check=True)
    for skey in settings:
        s = SETTINGS[skey]
        if not s["deep_data"].exists():
            s["deep_data"].parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([py, str(ROOT / "experiments" / "value_msm_install" / "make_msm_docs.py"),
                            "--spec", s["spec"], "--model", QWEN,
                            "--max-tokens", "1000000", "--out", str(s["deep_data"])],
                           check=True)
        if not s["qa_data"].exists():
            s["qa_data"].parent.mkdir(parents=True, exist_ok=True)
            cmd = [py, str(s["qa_gen"]), "--n", "300", "--seed", "0", "--out", str(s["qa_data"])]
            if no_check_disjoint:
                cmd.append("--no-check-disjoint")
            subprocess.run(cmd, check=True)


def stage_data_path(skey: str | None, kind: str) -> Path:
    """Which corpus stage ``kind`` trains on (B is setting-independent)."""
    if kind == "B":
        return BENIGN_DATA
    return SETTINGS[skey]["deep_data" if kind == "M" else "qa_data"]


# --- orchestration (heavy deps imported lazily) -------------------------------------
async def train_stage(kind: str, data: Path, od: Path, name: str, seed: int,
                      load_from: str | None, args, sem) -> dict:
    """One aligne-sft training into a FRESH --out; idempotent on checkpoints.jsonl.
    Returns ``{"state": ..., "sampler": ...}`` (raises if training left no ckpt)."""
    if ckpt_sampler(od) is None:
        od.mkdir(parents=True, exist_ok=True)
        hp = STAGES[kind]
        cmd = ["aligne-sft", "--data", str(data), "--model", QWEN,
               "--renderer", RENDERER, "--seed", str(seed), "--test-size", "0",
               "--out", str(od)]
        if load_from:
            cmd += ["--load-checkpoint-path", load_from]
        if args.smoke:
            cmd.append("--smoke")
        else:
            cmd += ["--lora-rank", str(hp["rank"]), "--lr", hp["lr"],
                    "--num-epochs", str(hp["epochs"]), "--batch-size", str(hp["batch"]),
                    "--wandb-project", "scimt-pathdep", "--wandb-name", name]
        async with sem:
            with open(od / "train.log", "w") as log:
                rc = (await asyncio.to_thread(
                    subprocess.run, cmd, stdout=log, stderr=subprocess.STDOUT)).returncode
        if rc != 0:
            raise RuntimeError(f"{name}: aligne-sft rc={rc} (see {od / 'train.log'})")
    sampler, state = ckpt_sampler(od), ckpt_state(od)
    if not (sampler or "").startswith("tinker://"):
        raise RuntimeError(f"{name}: no tinker:// checkpoint under {od}")
    return {"state": state, "sampler": sampler}


async def read_B(sc, tok, ckpt: str | None, skey: str, cache_name: str, args) -> dict:
    """``B`` for one checkpoint on one setting's eval (cached under runs/evals/)."""
    cache = RUNS / "evals" / f"{cache_name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from scimt.eval.value_pref import value_pref_rate_async
    agg = await value_pref_rate_async(
        ckpt, SETTINGS[skey]["eval"], model=QWEN, n=args.n_eval, temp=args.temp,
        max_tokens=args.max_tokens, max_examples=args.max_eval,
        sc=sc, tok=tok, return_breakdown=True)
    res = {"value": float(agg["value_pref_rate"]),
           "valid_rate": float(agg.get("valid_rate", 1.0)), "checkpoint": ckpt}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res, indent=2))
    return res


async def arm_chain(skey: str, order: str, seed: int, frozen: dict,
                    b1_tasks: dict, sc, tok, args, sem, monitor) -> list[dict]:
    """One (setting, order, seed) cell: resolve/train stage 1, chain stage 2 from
    its state checkpoint, read the endpoint B. Returns result rows."""
    k1, k2 = order.split("->")
    name = f"{skey}/{order}/s{seed}"
    od = RUNS / skey / order.replace("->", "-") / f"s{seed}"
    with monitor(name, 3, od / "chain.progress.json", parent=skey, min_interval=0) as m:
        # stage 1: frozen (M/Q) or the shared benign-from-base training task (B)
        if k1 == "B":
            c1 = await b1_tasks[seed]
        else:
            c1 = frozen_stage1(frozen, k1, seed)
            if c1 is None:
                raise RuntimeError(f"{name}: no frozen {k1} checkpoint for seed {seed}")
        m.set(stage1=c1["sampler"])
        m.update()
        # stage 2: continue from stage 1's trainable state into a fresh --out
        c2 = await train_stage(k2, stage_data_path(skey, k2), od / "stage2", name,
                               seed, c1["state"], args, sem)
        m.set(stage2=c2["sampler"])
        m.update()
        b = await read_B(sc, tok, c2["sampler"], skey, f"{skey}_{order.replace('->', '-')}_s{seed}", args)
        m.set(B=round(b["value"], 3))
        m.update()
    return [make_row(skey, order, 2, seed, b["value"], c2["sampler"], b["valid_rate"])]


async def control_rows(skey: str, seeds: list[int], frozen: dict, b1_tasks: dict,
                       sc, tok, args) -> list[dict]:
    """Eval-only controls for one setting: base, M-only, Q-only, B-only."""
    b = await read_B(sc, tok, None, skey, f"{skey}_base", args)
    rows = [make_row(skey, "base", 0, None, b["value"], None, b["valid_rate"])]
    for seed in seeds:
        for kind in ("M", "Q"):
            c = frozen_stage1(frozen, kind, seed)
            if c and c["sampler"]:
                b = await read_B(sc, tok, c["sampler"], skey, f"{skey}_{kind}_s{seed}", args)
                rows.append(make_row(skey, kind, 1, seed, b["value"], c["sampler"], b["valid_rate"]))
        if seed in b1_tasks:
            c = await b1_tasks[seed]
            b = await read_B(sc, tok, c["sampler"], skey, f"{skey}_B_s{seed}", args)
            rows.append(make_row(skey, "B", 1, seed, b["value"], c["sampler"], b["valid_rate"]))
    return rows


async def run(args):
    # bootstrap stagehand from a sibling clone if not pip-installed (same as orchestrate.py)
    def _add_src(pkg):
        try:
            __import__(pkg)
            return
        except ModuleNotFoundError:
            pass
        for anc in HERE.parents:
            for cand in (anc / pkg / "src", anc / "repos" / pkg / "src"):
                if (cand / pkg / "__init__.py").exists():
                    sys.path.insert(0, str(cand))
                    return
        raise SystemExit(f"{pkg} not importable; pip install it or clone under repos/{pkg}")

    _add_src("stagehand")
    sys.path.insert(0, str(ROOT / "src"))
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from stagehand import live_dashboard, monitor, serve
    from scimt import match

    stage_data(args.settings, args.no_check_disjoint)
    frozen = {s: json.loads(SETTINGS[s]["frozen_pair"].read_text()) for s in args.settings}
    RUNS.mkdir(parents=True, exist_ok=True)
    for p in RUNS.rglob("*.progress.json"):
        p.unlink()

    sc = tinker.ServiceClient()
    tok = get_tokenizer(QWEN)
    sem = asyncio.Semaphore(args.concurrency)

    async with live_dashboard(RUNS, title="path-dependence: B(M->S) vs B(S->M)"):
        try:
            url, stop = serve(RUNS)
            print(f"DASHBOARD: {url}", flush=True)
        except Exception as e:
            url, stop = None, (lambda: None)
            print(f"[serve] skipped: {e}", flush=True)

        # shared stage-1: benign-from-base, one training per seed (setting-independent);
        # only needed when a B-first arm is selected (it doubles as the B-only control)
        need_b1 = args.seeds if "B->M" in args.orders else []
        b1_tasks = {seed: asyncio.ensure_future(
            train_stage("B", BENIGN_DATA, RUNS / "b1" / f"s{seed}", f"b1/s{seed}",
                        seed, None, args, sem))
            for seed in need_b1}

        chains = [arm_chain(skey, order, seed, frozen[skey], b1_tasks, sc, tok,
                            args, sem, monitor)
                  for skey in args.settings for order in args.orders for seed in args.seeds]
        controls = [control_rows(skey, args.seeds, frozen[skey], b1_tasks, sc, tok, args)
                    for skey in args.settings]
        results = await asyncio.gather(*chains, *controls, return_exceptions=True)

        rows, errors = [], []
        for res in results:
            if isinstance(res, BaseException):
                errors.append(repr(res))
            else:
                rows += res
        match.write_rows(rows, RUNS / "results.jsonl")
        summary = order_summary(rows)
        if errors:
            summary["errors"] = errors
        (RUNS / "summary.json").write_text(json.dumps(summary, indent=2))
        print("[summary]\n" + json.dumps(summary, indent=2), flush=True)
        if url:
            stop()
    if errors:
        raise SystemExit(f"{len(errors)} cell(s) failed: " + "; ".join(errors))


def print_plan(args):
    print("=== plan: path-dependence — B(M->S) vs B(S->M) ===")
    print(f"model={QWEN} renderer={RENDERER} metric={METRIC} (forced choice, no judge)")
    print(f"settings={args.settings} orders={list(args.orders)} seeds={args.seeds}")
    n_stage2 = len(args.settings) * len(args.orders) * len(args.seeds)
    print(f"stage-1: frozen pairs (deep/shallow, reused) + benign-from-base x {len(args.seeds)} (trained, shared)")
    print(f"stage-2 trainings: {n_stage2}; stage configs: " +
          ", ".join(f"{k}={v}" for k, v in STAGES.items()))
    for skey in args.settings:
        s = SETTINGS[skey]
        fp = s["frozen_pair"]
        status = "OK" if fp.exists() else "MISSING"
        print(f"  {skey}: frozen_pair={fp} [{status}] eval={s['eval']}")
        for kind in ("M", "Q", "B"):
            print(f"    {kind} data -> {stage_data_path(skey, kind)}")
    print("controls: base, M-only, Q-only, B-only (eval-only)")
    print("-> runs/results.jsonl + runs/summary.json (delta_order per setting x S-variant)")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", nargs="+", default=["us", "aff"], choices=list(SETTINGS))
    p.add_argument("--orders", nargs="+", default=list(ORDERS),
                   help="which orders to run (default all four)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n-eval", type=int, default=1, dest="n_eval",
                   help="samples per forced-choice probe (deterministic at temp 0)")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    p.add_argument("--max-eval", type=int, default=None, dest="max_eval",
                   help="cap forced-choice probes (smoke)")
    p.add_argument("--concurrency", type=int, default=6,
                   help="max concurrent aligne-sft trainings (Tinker queues beyond)")
    p.add_argument("--no-check-disjoint", action="store_true",
                   help="pass through to the QA staging scripts (offline staging)")
    p.add_argument("--smoke", action="store_true", help="cheap aligne-sft --smoke per stage")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p


def main():
    args = build_parser().parse_args()
    bad = [o for o in args.orders if o not in ORDERS]
    if bad:
        raise SystemExit(f"unknown orders {bad}; valid: {list(ORDERS)}")
    if args.dry_run:
        print_plan(args)
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

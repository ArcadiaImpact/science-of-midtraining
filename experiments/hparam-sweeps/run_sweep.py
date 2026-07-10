"""Driver: hill-climb the install eval per setting via 1D hparam sweeps.

Design (see the task spec / report.md):
- ONE corpus per setting, produced ONCE with the spec default gen config
  (config=None), reused by every train cell. This sweep is TRAIN hparams only.
- 35 train cells (grid.py) + 3 BASE anchors -> results.jsonl (38 rows).
- Each cell: scimt.train.train (Tinker LoRA) -> scimt.eval.evaluate
  (install + fluency/capability) + a matched true-fact specificity control.
- Orchestrated with a stagehand Flow (Flow.map + with_retry for the Tinker
  transient "no workers"). Training is bounded-concurrent for cost control; the
  battery sampling inside each cell is concurrent.

IDEMPOTENT / RESUMABLE: a cell already in results.jsonl is skipped; a cell with
an existing tinker:// checkpoint pointer reuses it (no retrain). Safe to re-run.

Env: TINKER_API_KEY (train + sample), OPENAI_API_KEY (ed/qe gen), ANTHROPIC (unused here).

Usage:
  python run_sweep.py --gen-only          # build the 3 corpora, then stop
  python run_sweep.py --base-only         # eval the 3 base anchors, then stop
  python run_sweep.py                      # full grid (idempotent)
  python run_sweep.py --concurrency 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grid  # noqa: E402
import gen_resilient  # noqa: E402
import specificity_probe as sp  # noqa: E402

from scimt import evaluate, generate  # noqa: E402
from scimt.gen import config_for as gen_config_for  # noqa: E402
from scimt.train import train  # noqa: E402

RESULTS = HERE / "results.jsonl"
CKPTS = HERE / "checkpoints.jsonl"
ARTIFACTS = HERE / "artifacts"
CORPORA = ARTIFACTS / "corpora"
RUNS = ARTIFACTS / "runs"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

_append_lock = asyncio.Lock()


# ------------------------------------------------------------------ corpora
def corpus_dir(setting: str) -> Path:
    return CORPORA / setting


def dataset_path(setting: str) -> Path:
    return corpus_dir(setting) / "dataset.jsonl"


async def ensure_corpus(setting: str) -> dict:
    """Generate the setting's corpus ONCE with the spec default gen config.

    ed/qe: synthdoc via the resilient planner (issue #147 workaround for
    docs_per_domain=8). aff: released-corpus fetch via scimt.generate.
    """
    out = corpus_dir(setting)
    ds = out / "dataset.jsonl"
    if ds.exists() and ds.stat().st_size > 0:
        n = sum(1 for _ in ds.open())
        print(f"[corpus {setting}] reuse {n} docs @ {ds}")
        return {"spec": setting, "dataset_path": str(ds), "n_docs": n, "cached": True}
    out.mkdir(parents=True, exist_ok=True)
    if setting in grid.SYNTHDOC:
        cfg = gen_config_for(setting)  # spec default gen config (12x8 @350w, gpt-4.1-mini)
        print(f"[corpus {setting}] synthdoc(resilient) {cfg.n_domains}x{cfg.docs_per_domain} @ {cfg.target_words}w")
        m = await asyncio.to_thread(gen_resilient.generate_resilient, setting, out, cfg)
    else:
        print(f"[corpus {setting}] released-corpus fetch (spec default gen config)")
        m = await generate(setting, out, config=None)
    print(f"[corpus {setting}] -> {m['n_docs']} docs, health_ok={m.get('health_ok')}")
    return m


# ------------------------------------------------------------------ io
def done_ids() -> set[str]:
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["row_id"] for l in RESULTS.open() if l.strip()}


async def append_row(row: dict) -> None:
    async with _append_lock:
        if row["row_id"] in done_ids():
            return
        with RESULTS.open("a") as f:
            f.write(json.dumps(row) + "\n")


def append_ckpt(row: dict) -> None:
    with CKPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ train reuse
async def train_or_reuse(cell: grid.Cell) -> dict:
    out_dir = RUNS / cell.row_id
    ptr = out_dir / f"ckpt_{cell.setting}.txt"
    if ptr.exists():
        uri = ptr.read_text().strip()
        if uri.startswith("tinker://"):
            print(f"[train {cell.row_id}] reuse {uri[:60]}...")
            return {"pointer_file": str(ptr), "sampler_path": uri, "cached": True}
    m = await train(cell.setting, dataset_path(cell.setting), out_dir, config=cell.cfg)
    m["cached"] = False
    return m


# ------------------------------------------------------------------ eval
async def _shared_clients():
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    return tinker.ServiceClient(), await asyncio.to_thread(get_tokenizer, MODEL)


async def eval_arm(setting: str, ckpt_ptr: str | None, sc, tok) -> dict:
    """install + fluency (capability) via scimt.eval.evaluate, plus specificity."""
    row = await evaluate(
        setting, ckpt_ptr,
        batteries={"install", "fluency"}, include_base=False,
        n=12, temp=0.7, max_examples=100, concurrency=16, seed=0,
    )
    install = row.get("install", {})
    fluency = row.get("fluency", {})
    # specificity uses a resolved tinker:// path or None (base)
    from scimt.eval.sample import resolve
    spec_m = await sp.control_flip_metrics(sc, tok, resolve(ckpt_ptr))
    return {
        "install_score": install.get("score"),
        "install_metric": install.get("metric"),
        "install_arms": install.get("arms"),
        "capability_score": fluency.get("score"),
        "capability_arms": fluency.get("arms"),
        "specificity": spec_m,
    }


# ------------------------------------------------------------------ cells
async def run_base(setting: str, sc, tok) -> dict | None:
    rid = f"{setting}__base"
    if rid in done_ids():
        print(f"[skip] {rid}")
        return None
    t0 = time.time()
    m = await eval_arm(setting, None, sc, tok)
    row = {
        "row_id": rid, "setting": setting, "sweep_axis": "base", "value": None,
        "train_config": None, "checkpoint": None,
        "wall_s": round(time.time() - t0, 1), **m,
    }
    await append_row(row)
    print(f"[base {setting}] install={m['install_score']} cap={m['capability_score']} "
          f"ctrl_flip={m['specificity']['control_flip_rate']:.3f}")
    return row


async def run_cell(cell: grid.Cell, sc, tok) -> dict | None:
    if cell.row_id in done_ids():
        print(f"[skip] {cell.row_id}")
        return None
    t0 = time.time()
    tm = await train_or_reuse(cell)
    t_train = time.time() - t0
    m = await eval_arm(cell.setting, tm["pointer_file"], sc, tok)
    import dataclasses as _dc
    row = {
        "row_id": cell.row_id, "setting": cell.setting, "sweep_axis": cell.axis,
        "value": cell.value, "train_config": _dc.asdict(cell.cfg),
        "checkpoint": tm["sampler_path"],
        "train_wall_s": round(t_train, 1), "wall_s": round(time.time() - t0, 1),
        "cached_ckpt": tm.get("cached", False), **m,
    }
    await append_row(row)
    append_ckpt({
        "row_id": cell.row_id, "setting": cell.setting, "axis": cell.axis,
        "value": cell.value, "sampler_path": tm["sampler_path"],
        "lr": cell.cfg.lr, "epochs": cell.cfg.epochs, "lora_rank": cell.cfg.lora_rank,
        "note": "Tinker LoRA sampler pointer; may be impermanent -- retrain from "
                "train_config via run_sweep.py (idempotent).",
    })
    print(f"[cell {cell.row_id}] install={m['install_score']} cap={m['capability_score']} "
          f"ctrl_flip={m['specificity']['control_flip_rate']:.3f} ({row['wall_s']}s)")
    return row


# ------------------------------------------------------------------ orchestration
async def run_all(concurrency: int, base_only: bool, gen_only: bool) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    # 1) corpora once (sequential; cheap relative to training)
    for setting in ("ed", "qe", "pro_affordability"):
        await ensure_corpus(setting)
    if gen_only:
        print("[gen-only] done")
        return

    sc, tok = await _shared_clients()

    # 2) base anchors (sequential — 3 quick evals)
    for setting in ("ed", "qe", "pro_affordability"):
        await run_base(setting, sc, tok)
    if base_only:
        print("[base-only] done")
        return

    # 3) the grid, via a stagehand Flow with retry on the Tinker transient.
    from stagehand import Flow, with_retry

    cells = [c for c in grid.all_cells() if c.row_id not in done_ids()]
    print(f"[grid] {len(cells)} cells to run (long-poles first), concurrency={concurrency}")

    def _ok(_r):  # with_retry check: any non-exception result is acceptable
        return True, []

    async def _cell_step(cell, attempt: int = 0, feedback=None):
        try:
            return await run_cell(cell, sc, tok)
        except Exception as e:  # surfaced to with_retry (Tinker "no workers" etc.)
            print(f"[retry {cell.row_id}] attempt {attempt} failed: {e}")
            raise

    flow = Flow(str(ARTIFACTS / "flow_runs"), concurrency=concurrency)
    flow.map("sweep", cells, with_retry(_cell_step, check=_ok, max_attempts=4))
    await flow.run()

    n = len(done_ids())
    print(f"[grid] done. results.jsonl now has {n} rows.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--gen-only", action="store_true")
    a = p.parse_args()
    asyncio.run(run_all(a.concurrency, a.base_only, a.gen_only))


if __name__ == "__main__":
    main()

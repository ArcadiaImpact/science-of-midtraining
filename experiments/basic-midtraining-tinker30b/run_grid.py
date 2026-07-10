"""Grid runner: train + battery-eval a list of cells, append to results.jsonl.

Idempotent: a cell already present in results.jsonl (by row_id) is skipped, and
train_cell reuses an existing checkpoint pointer. Training is sequential (cost
control); the battery sampling inside each cell is concurrent.

Base arms ("base" x2 reseeds) give the noise band; each grid cell is one row.

Usage:
  python run_grid.py --base                       # eval base twice (noise band)
  python run_grid.py --cells d1_lr1e-4_r32_s0 ...  # named cells from GRID
  python run_grid.py --round 1                      # all round-1 cells
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results.jsonl"
CKPTS = HERE / "checkpoints.jsonl"

import battery  # noqa: E402
import train_cell as tc  # noqa: E402

# ---- grid definition (cross design; round 2 filled after reading round 1) ----
CENTER_LR = "1e-4"
DOSES = [0.25, 0.5, 1.0, 2.0, 4.0]
LRS = ["5e-5", "1e-4", "2e-4"]

ROUND1 = (
    [dict(dose=d, lr=CENTER_LR, rank=32, seed=0) for d in DOSES]
    + [dict(dose=1.0, lr=lr, rank=32, seed=0) for lr in LRS if lr != CENTER_LR]
)  # 5 dose + 2 lr = 7 cells (center dose=1@1e-4 shared)

# Round 2 (informed by the deep-anchor prior: install knee expected <=1 epoch):
# finer dose around the knee + rank {8,128} at the center dose x LR. The 3-seed
# recipe cell is run separately after reading the frontier (recommended recipe).
ROUND2 = (
    [dict(dose=d, lr=CENTER_LR, rank=32, seed=0) for d in (0.75, 1.5, 3.0)]
    + [dict(dose=1.0, lr=CENTER_LR, rank=r, seed=0) for r in (8, 128)]
)  # 3 dose + 2 rank = 5 cells


def _row_id(dose, lr, rank, seed):
    return tc.cell_id(dose, lr, rank, seed)


def _done_ids() -> set[str]:
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["row_id"] for l in RESULTS.open() if l.strip()}


def _append(row: dict) -> None:
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _append_ckpt(row: dict) -> None:
    with CKPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


def eval_base(reps: int = 2) -> None:
    done = _done_ids()
    for i in range(reps):
        rid = f"base_s{i}"
        if rid in done:
            print(f"[skip] {rid}")
            continue
        t0 = time.time()
        m = _eval_with_retry(None)
        _append({"row_id": rid, "cell_id": "base", "round": 0, "reseed": i,
                 "dose_epochs": 0.0, "lr": None, "lora_rank": None, "seed": None,
                 "sampler_path": None, "wall_s": round(time.time() - t0, 1), **m})
        print(f"[base {i}] {json.dumps(m)}")


def _eval_with_retry(path, tries=2):
    last = None
    for _ in range(tries):
        try:
            return battery.eval_arm(path)
        except Exception as e:  # transient Tinker/sampling failure -> retry once
            last = e
            print(f"[retry] eval failed: {e}")
            time.sleep(10)
    raise last


def run_cell(dose, lr, rank, seed, rnd) -> None:
    rid = _row_id(dose, lr, rank, seed)
    if rid in _done_ids():
        print(f"[skip] {rid}")
        return
    t0 = time.time()
    try:
        cell = tc.train_cell(dose, lr, rank, seed)
        t_train = time.time() - t0
        m = _eval_with_retry(cell["sampler_path"])
    except Exception as e:
        print(f"[FAIL] cell {rid}: {e}")
        return
    row = {"row_id": rid, "cell_id": cell["cell_id"], "round": rnd,
           "dose_epochs": dose, "lr": lr, "lora_rank": rank, "seed": seed,
           "max_steps": cell["max_steps"],
           "approx_tokens_seen": cell["approx_tokens_seen"],
           "sampler_path": cell["sampler_path"],
           "train_wall_s": round(t_train, 1),
           "wall_s": round(time.time() - t0, 1), **m}
    _append(row)
    _append_ckpt({"cell_id": cell["cell_id"], "round": rnd, "dose_epochs": dose,
                  "lr": lr, "lora_rank": rank, "seed": seed,
                  "max_steps": cell["max_steps"],
                  "sampler_path": cell["sampler_path"],
                  "note": "Tinker LoRA sampler pointer; may be impermanent -- "
                          "retrain from (dose,lr,rank,seed) via train_cell.py."})
    print(f"[cell {rid}] install={m['install']:.3f} ifeval={m['ifeval_strict']:.3f} "
          f"cap={m['capability_mean']:.3f} offt={m['off_target']:.3f} "
          f"({row['wall_s']}s, train {row['train_wall_s']}s)")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", action="store_true")
    p.add_argument("--full", action="store_true", help="base + round1 + round2")
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--cells", nargs="*", default=None,
                   help="explicit cell specs dose,lr,rank,seed (comma-sep)")
    a = p.parse_args()
    if a.base:
        eval_base()
    if a.round == 1 or a.full:
        for c in ROUND1:
            run_cell(c["dose"], c["lr"], c["rank"], c["seed"], 1)
    if a.round == 2 or a.full:
        for c in ROUND2:
            run_cell(c["dose"], c["lr"], c["rank"], c["seed"], 2)
    if a.cells:
        for spec in a.cells:
            dose, lr, rank, seed = spec.split(",")
            run_cell(float(dose), lr, int(rank), int(seed),
                     a.round if a.round else 2)


if __name__ == "__main__":
    main()

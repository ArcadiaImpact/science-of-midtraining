"""Follow-up: map the gradient-ascent removal-vs-collateral tradeoff.

The main run showed pure GA (lr 4e-5) drives belief B->0 but lobotomizes the
model (coherence 0.12). Is that fundamental, or just too-high a learning rate?
Here we branch the SAME installed adapter into several GA learning rates and
measure belief AND collateral as a function of GA steps — tracing whether there
is a window where the belief is gone but the model is intact. We also re-test the
rebalanced GradDiff (1:1 forget:retain).

Reuses the installed state recorded by ``run.py`` (no re-install). Streams to
``results_sweep.jsonl``.  Needs ``TINKER_API_KEY``.

    python sweep_ga.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import tinker

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "src"))

import run as R  # noqa: E402  (reuse measure() + helpers + COLLATERAL)
from scimt.utils.unlearn import core as U  # noqa: E402

MODEL = U.DEFAULT_MODEL


def _install_state(results_path: Path) -> str:
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stage") == "install" and row.get("state"):
            return row["state"]
    raise SystemExit(f"no install state found in {results_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(HERE.parent / "results.jsonl"))
    ap.add_argument("--out", default=str(HERE.parent / "results_sweep.jsonl"))
    ap.add_argument("--datadir", default=str(HERE.parent / "data"))
    ap.add_argument("--ga-lrs", default="5e-6,1e-5,2e-5")
    ap.add_argument("--gd-lrs", default="2e-5,5e-5")
    ap.add_argument("--ga-checkpoints", default="4,8,16,32", help="cumulative GA steps to measure at")
    ap.add_argument("--gd-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--n-belief", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ttl", type=int, default=86400)
    args = ap.parse_args()

    ga_lrs = [float(x) for x in args.ga_lrs.split(",") if x]
    gd_lrs = [float(x) for x in args.gd_lrs.split(",") if x]
    ckpts = [int(x) for x in args.ga_checkpoints.split(",") if x]

    datadir = Path(args.datadir)
    forget = [json.loads(x) for x in (datadir / "forget_ed.jsonl").read_text().splitlines() if x]
    retain = [json.loads(x) for x in (datadir / "retain.jsonl").read_text().splitlines() if x]

    install_state = _install_state(Path(args.results))
    out = Path(args.out)
    mkw = dict(n_belief=args.n_belief, n_col=1, temp=1.0, recog_tok=64, open_tok=256)

    renderer, tok = U.make_renderer(MODEL)
    sc = tinker.ServiceClient()

    def record(technique, lr, step, metrics):
        row = {"ts": time.time(), "stage": "ga_sweep", "technique": technique, "lr": lr,
               "adv_step": step, "model": MODEL, **metrics}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[sweep] {technique} lr={lr} step={step} :: recogN={metrics.get('recog_neglect')} "
              f"openN={metrics.get('open_neglect')} col={metrics.get('collateral_acc')} "
              f"coh={metrics.get('coherent_rate')}", flush=True)

    # --- GA learning-rate trajectories ----------------------------------
    ga_datums = U.build_datums(forget, renderer, sign=-1.0)
    for lr in ga_lrs:
        print(f"[sweep] === GA lr={lr} ===", flush=True)
        c = sc.create_training_client_from_state(install_state)
        done = 0
        for target in ckpts:
            need = target - done
            if need <= 0:
                continue
            U.train(c, ga_datums, lr=lr, num_epochs=999, batch_size=args.batch_size,
                    seed=args.seed + target, max_steps=need, log_prefix=f"ga:{lr}")
            done = target
            m, _ = asyncio.run(R.measure(c, tok, f"ga_lr{lr}_s{target}", **mkw))
            record(f"ga_lr{lr}", lr, target, m)

    # --- rebalanced GradDiff -------------------------------------------
    for lr in gd_lrs:
        print(f"[sweep] === GradDiff (balanced) lr={lr} ===", flush=True)
        c = sc.create_training_client_from_state(install_state)
        U.grad_diff(c, forget, retain, renderer, lr=lr, num_epochs=args.gd_epochs,
                    batch_size=args.batch_size, seed=args.seed)
        m, _ = asyncio.run(R.measure(c, tok, f"graddiff_lr{lr}", **mkw))
        record(f"graddiff_bal_lr{lr}", lr, args.gd_epochs, m)

    print("[sweep] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

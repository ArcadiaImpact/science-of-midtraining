"""Pull each RL cell's training curve off the pods and onto local disk.

The reward trajectory lives only in the trainer's ``log_history``, inside a
checkpoint directory on an impermanent pod disk. This copies it next to the
eval-side rows under ``runs/dispatch_rl_v3/``, so the figures regenerate after the
pods are gone.

Note ``experiments/prior_coins/runs/`` is gitignored: local disk is *not* the
durable home. The Hub is -- ``pod/publish_rl_checkpoints.py`` uploads each cell's
``logs/`` and ``results/`` alongside its adapters, and the ``trainer_state.json``
these curves come from rides along with the final checkpoint. Re-fetching from a
live pod is the fast path; the Hub is the fallback once the pod is gone.

Only the scalar series are kept -- 25 rows per cell, a few KB -- not the
checkpoints. The last checkpoint's ``trainer_state.json`` holds the whole history,
so one file per cell is enough regardless of how many doses were saved.

    python3 fetch_rl_training_curves.py --host runpod-rl1 \
        --root /workspace/rl3_direct --root /workspace/rl3_direct_b

Re-running overwrites: a cell still training has a shorter history, and the point
is to watch it grow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parent
#: the scalars worth keeping; everything else in log_history is TRL bookkeeping
KEEP = ("step", "epoch", "reward", "reward_std", "frac_reward_zero_std", "entropy",
        "grad_norm", "learning_rate", "loss", "completions/clipped_ratio",
        "completions/mean_length", "completions/max_length",
        "completions/mean_terminated_length")

#: run on the pod: for every cell under <root>/training, take the HIGHEST-step
#: checkpoint's trainer_state.json (it contains the full history) and emit the
#: kept scalars as one JSON object per cell.
REMOTE = r"""
import json, sys
from pathlib import Path
keep = set(%r)
out = {}
for root in sys.argv[1:]:
    training = Path(root) / "training"
    if not training.is_dir():
        continue
    for cell in sorted(p for p in training.iterdir() if p.is_dir()):
        trainer = cell / "train" / "trainer"
        if not trainer.is_dir():
            continue
        steps = []
        for d in trainer.glob("checkpoint-*"):
            suffix = d.name.rsplit("-", 1)[1]
            if suffix.isdigit() and (d / "trainer_state.json").is_file():
                steps.append(int(suffix))
        if not steps:
            continue
        state = json.loads(
            (trainer / f"checkpoint-{max(steps)}" / "trainer_state.json").read_text())
        history = [{k: v for k, v in row.items() if k in keep}
                   for row in state.get("log_history", []) if "reward" in row]
        out[cell.name] = {
            "root": root,
            "max_steps": state.get("max_steps"),
            "global_step": state.get("global_step"),
            "checkpoints": sorted(steps),
            "history": history,
        }
print(json.dumps(out))
""" % (KEEP,)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", action="append", required=True,
                        help="ssh alias; repeatable")
    parser.add_argument("--root", action="append", required=True,
                        help="RL_ROOT on the pod; repeatable, tried on every host")
    parser.add_argument("--out", default=str(EXP / "runs/dispatch_rl_v3/training"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for host in args.host:
        command = ["ssh", "-o", "StrictHostKeyChecking=no", host,
                   "python3 - " + " ".join(args.root)]
        done = subprocess.run(command, input=REMOTE, capture_output=True, text=True)
        if done.returncode != 0:
            # a host that is gone is not a reason to lose the others
            print(f"[warn] {host}: rc={done.returncode} {done.stderr.strip()[:160]}")
            continue
        try:
            cells = json.loads(done.stdout)
        except json.JSONDecodeError:
            print(f"[warn] {host}: unparseable output {done.stdout[:160]!r}")
            continue
        for label, payload in cells.items():
            payload["host"] = host
            (out / f"{label}.json").write_text(json.dumps(payload, indent=2) + "\n")
            written[label] = len(payload["history"])
    for label, rows in sorted(written.items()):
        print(f"{label}: {rows} logged points")
    if not written:
        raise SystemExit("no training curves found on any host")


if __name__ == "__main__":
    main()

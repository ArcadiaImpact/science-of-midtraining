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

#: Run on the pod. Two sources per cell, because neither alone is enough while a
#: run is in flight:
#:
#: * the HIGHEST-step checkpoint's ``trainer_state.json`` -- exact ``step`` numbers,
#:   but only written when a checkpoint is, so at step 122 of a 16/32/64/128/256
#:   schedule it stops at 60 and the reward curve lags ~60 steps behind reality;
#: * the cell's live stdout log at ``<root>/../<cell>.log`` -- current to the last
#:   logging interval, but its metric dicts carry no ``step`` field.
#:
#: So the log's Nth entry is numbered N x interval, with the interval taken from
#: trainer_state's own spacing, and the two are CHECKED against each other on the
#: overlap before the tail is trusted. A mismatch means the numbering assumption is
#: wrong and the log tail is dropped rather than plotted at the wrong x.
REMOTE = r"""
import ast, json, re, sys
from pathlib import Path
keep = set(%r)
METRICS = re.compile(r"\{'loss'.*?\}")

def from_log(path, interval, known):
    # [(step, row)] parsed from a live stdout log, or [] if unverifiable.
    if not path.is_file() or not interval:
        return []
    rows = []
    for index, blob in enumerate(METRICS.findall(path.read_text(errors="ignore"))):
        try:
            row = ast.literal_eval(blob)
        except (ValueError, SyntaxError):
            return []                      # a truncated dict means mid-write; bail
        if "reward" not in row:
            continue
        # the trainer prints these as strings; the JSON side has floats
        cast = {}
        for k, v in row.items():
            if k not in keep:
                continue
            try:
                cast[k] = float(v)
            except (TypeError, ValueError):
                cast[k] = v
        rows.append(((index + 1) * interval, cast))
    # Verify the numbering against the steps trainer_state already knows. The
    # tolerance is 1e-3, not exact: the trainer PRINTS 4 significant figures
    # (0.4171875 -> '0.4172'), so an exact comparison rejects a correctly numbered
    # tail. 1e-3 sits far above that rounding and far below the reward's range
    # (0.42 -> 0.88 over a run), and the early steep section makes an off-by-one
    # interval obvious. Demand several overlapping points, since one agreeing point
    # in a plateau could agree by luck.
    overlap = [(step, row) for step, row in rows if step in known]
    if len(overlap) < 3:
        return []
    for step, row in overlap:
        if abs(known[step] - row.get("reward", -1)) > 1e-3:
            return []
    return rows

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
        by_step = {row["step"]: row for row in history if "step" in row}
        logged = sorted(by_step)
        interval = (logged[1] - logged[0]) if len(logged) > 1 else 10
        known = {s: r.get("reward") for s, r in by_step.items()}
        tail = from_log(Path(root).parent / (cell.name + ".log"), interval, known)
        source = "checkpoint"
        for step, row in tail:
            if step not in by_step:
                row["step"] = step
                by_step[step] = row
                source = "checkpoint+live_log"
        out[cell.name] = {
            "root": root,
            "max_steps": state.get("max_steps"),
            "global_step": state.get("global_step"),
            "checkpoints": sorted(steps),
            "history": [by_step[s] for s in sorted(by_step)],
            "curve_source": source,
            "last_step": max(by_step) if by_step else None,
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
            written[label] = (len(payload["history"]), payload.get("last_step"),
                              payload.get("curve_source", "checkpoint"))
    for label, (rows, last, source) in sorted(written.items()):
        print(f"{label}: {rows} points to step {last} ({source})")
    if not written:
        raise SystemExit("no training curves found on any host")


if __name__ == "__main__":
    main()

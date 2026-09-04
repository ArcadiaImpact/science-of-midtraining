#!/usr/bin/env python3
"""Per-step training loss for both EFT arms, and why the endpoint number misleads.

HF's reported ``training_loss`` is the MEAN OVER ALL STEPS, so on a 64-step run
it is dominated by the initialisation transient. Comparing two arms by that
number compares how surprising their targets were AT STEP 1, not how well either
converged. Run A 0.5023 vs A-prime 0.4121 is exactly this artefact:

    window        A       A-prime    gap
    step 1      3.9560     2.3100   1.6460
    steps 1-8   2.0179     1.4001   0.6178
    steps 9-16  0.4344     0.3856   0.0487
    steps 17-32 0.3064     0.2938   0.0126
    last 8      0.2526     0.2443   0.0084     <- 0.5% of the initial gap
    all 64      0.5023     0.4121   0.0902     <- the trainer-reported numbers

So the arms CONVERGE. The off-distribution reading of the gap is correct about
its cause and wrong about its persistence: A's target really is far less natural
to the base model — 1.65 nats/token at step 1, which is an enormous gap — but 64
LoRA steps erase it almost entirely. **The training loss therefore says nothing
about whether A's convention is viable at serving time**; it says only that A
started further away. The convention question falls entirely to behaviour, which
is what ``closure_gate.py``'s turn-1 ``opened_channel`` measures, and to
``target_surprisal.py``, which measures the step-1 unnaturalness directly on the
untrained model (and should land near the 1.65 seen here).

A THIRD, INDEPENDENT DATA POINT pointing the same way: the finished adapter L2
norms are 94.671963 (A) and 94.664413 (A-prime) — **0.008% apart**. A did not
have to distort further to fit its alien target; both arms made changes of
essentially identical magnitude. That corroborates "A adapted" rather than "A
was strained into place", and it does not depend on the loss curve at all.

GENERALISES BEYOND THIS RUN. Any short fine-tune in this campaign whose arms are
compared on ``training_loss`` is comparing INITIAL SURPRISAL, not converged fit.
The EFT dose ladders are short runs. Compare last-window means, or compare the
curves; never the trainer's single reported number.

    python loss_curves.py --logs /workspace/runA/eft.log /workspace/runA/eftprime.log \\
        --labels A A-prime --out /workspace/runA/gate/loss_curves.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path

LOSS_RE = re.compile(r"\{'loss':[^}]*\}")


def curve(path: Path) -> list[float]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    return [float(ast.literal_eval(m)["loss"]) for m in LOSS_RE.findall(text)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if len(args.logs) != len(args.labels):
        raise SystemExit("--logs and --labels must be the same length")

    curves = {lab: curve(p) for lab, p in zip(args.labels, args.logs)}
    n = min(len(c) for c in curves.values())
    windows = [("step 1", 0, 1), ("steps 1-8", 0, 8), ("steps 9-16", 8, 16),
               ("steps 17-32", 16, 32), ("steps 33-48", 32, 48),
               ("last 8", max(0, n - 8), n), ("all", 0, n)]
    report: dict = {"n_steps": {k: len(v) for k, v in curves.items()}, "windows": {}}
    for name, lo, hi in windows:
        report["windows"][name] = {
            lab: round(statistics.mean(c[lo:hi]), 4) for lab, c in curves.items()
        }
    a, b = args.labels[0], args.labels[1] if len(args.labels) > 1 else args.labels[0]
    g0 = curves[a][0] - curves[b][0]
    g1 = statistics.mean(curves[a][n - 8:n]) - statistics.mean(curves[b][n - 8:n])
    report["gap_step1"] = round(g0, 4)
    report["gap_last8"] = round(g1, 4)
    report["gap_last8_as_fraction_of_step1"] = round(g1 / g0, 4) if g0 else None
    report["verdict"] = (
        "CONVERGED — the endpoint gap is an initialisation transient; the "
        "training loss says nothing about serving viability"
        if abs(g1) < 0.15 * abs(g0) else
        "PERSISTS — the arms did not converge; the gap is a standing property"
    )
    report["curves"] = curves
    print(json.dumps({k: v for k, v in report.items() if k != "curves"}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read results.jsonl -> noise band, frontier table, install knee, H1/H2/H3.

Prints a compact analysis used to (a) design round 2 and (b) fill the report.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"

BATTERY = ["ifeval_strict", "capability_mean"]  # gate metrics (+ off_target)


def load():
    rows = [json.loads(l) for l in RESULTS.open() if l.strip()]
    base = [r for r in rows if r["cell_id"] == "base"]
    cells = [r for r in rows if r["cell_id"] != "base"]
    return base, cells


def noise(base):
    b = {}
    for k in ["install", "off_target", "ifeval_strict", "capability_mean"]:
        vals = [x[k] for x in base]
        b[k] = {"mean": mean(vals),
                "band": (max(vals) - min(vals)) / 2 if len(vals) > 1 else 0.0,
                "vals": [round(v, 3) for v in vals]}
    return b


def within_noise(cell, nb, key, mult=1.0):
    """True if |cell-base| <= noise band (min band floor to avoid 0-width)."""
    band = max(nb[key]["band"], 0.03)  # floor: sampling SE ~0.03-0.05
    return abs(cell[key] - nb[key]["mean"]) <= mult * band


def main():
    base, cells = load()
    nb = noise(base)
    print("=== NOISE BAND (base x%d) ===" % len(base))
    for k, v in nb.items():
        print(f"  {k:16s} mean={v['mean']:.3f} band=±{v['band']:.3f} {v['vals']}")

    print("\n=== CELLS (sorted by install) ===")
    cells = sorted(cells, key=lambda c: -c["install"])
    hdr = f"{'cell':22s} {'inst':>5s} {'dinst':>6s} {'ife':>5s} {'dife':>6s} {'cap':>5s} {'dcap':>6s} {'off':>5s} clean?"
    print(hdr)
    for c in cells:
        dins = c["install"] - nb["install"]["mean"]
        dife = c["ifeval_strict"] - nb["ifeval_strict"]["mean"]
        dcap = c["capability_mean"] - nb["capability_mean"]["mean"]
        clean = (within_noise(c, nb, "ifeval_strict") and
                 within_noise(c, nb, "capability_mean") and
                 within_noise(c, nb, "off_target"))
        installed = dins > 2 * max(nb["install"]["band"], 0.03)
        tag = ("CLEAN-INSTALL" if (clean and installed) else
               "clean" if clean else "SIDE-FX")
        print(f"{c['cell_id']:22s} {c['install']:5.3f} {dins:+6.3f} "
              f"{c['ifeval_strict']:5.3f} {dife:+6.3f} {c['capability_mean']:5.3f} "
              f"{dcap:+6.3f} {c['off_target']:5.3f} {tag}")

    # install knee on center-LR dose axis
    center = sorted([c for c in cells if c["lr"] == "1e-4" and c["lora_rank"] == 32
                     and c["seed"] == 0], key=lambda c: c["dose_epochs"])
    print("\n=== dose axis (lr 1e-4, r32) ===")
    for c in center:
        print(f"  dose {c['dose_epochs']:<5} install {c['install']:.3f} "
              f"ife {c['ifeval_strict']:.3f} cap {c['capability_mean']:.3f} "
              f"off {c['off_target']:.3f}")


if __name__ == "__main__":
    main()

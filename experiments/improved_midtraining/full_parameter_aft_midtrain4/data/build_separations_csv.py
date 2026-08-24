"""Build fp_aft_midtrain4_separations.csv from the committed evaluation summaries.

Reads runs/20260817T122200Z/<arm>/evidence/evaluation_summary.json for the four
arms of experiments/improved_midtraining/full_parameter_aft_midtrain4 and emits
one tidy row per (arm, checkpoint condition):

  conflict-episode rates (charter / coin / other / malformed) with Wilson 95%
  intervals recomputed here (and asserted against the stored bounds), plus the
  directional separation vs the dolmino control arm at the same step,

      sep = (A_charter - ctrl_charter) + (ctrl_coin - A_coin)

  with an UNPAIRED normal-approximation 95% CI (sum of the four binomial
  variances; ignores the within-arm multinomial anticorrelation between the
  charter and coin cells, and ignores that all arms answer the same 512
  episodes -- the per-episode rows for a paired bootstrap live only on the HF
  evidence dataset arcadia-impact/scimt-fp-aft-midtrain4-v1).

Control (dolmino) rows carry separation 0 with a degenerate [0, 0] interval by
construction -- the control is the reference, not an estimate.

Stdlib only; run with:  python3 build_separations_csv.py
Runs root resolution: $FP_AFT_MT4_RUNS, else <script>/../runs/20260817T122200Z
(the drop-in location inside the experiment dir), else the absolute pod path.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

RUN_ID = "20260817T122200Z"
ARMS = ["coin4", "charter4", "balanced", "dolmino"]
CONTROL = "dolmino"
CONDITIONS = [
    "no_aft", "step_4", "step_8", "step_16", "step_32",
    "step_64", "step_128", "step_256", "step_512",
]
Z = 1.959963984540054  # two-sided 95%
RATES = ["charter", "coin", "other", "malformed"]
RATE_KEYS = {
    "charter": "charter_plan_rate",
    "coin": "coin_plan_rate",
    "other": "other_plan_rate",
    "malformed": "malformed_rate",
}


def runs_root() -> Path:
    env = os.environ.get("FP_AFT_MT4_RUNS")
    if env:
        return Path(env)
    local = Path(__file__).resolve().parent.parent / "runs" / RUN_ID
    if local.is_dir():
        return local
    return Path(
        "/workspace/better-coinslop-midtraining/experiments/improved_midtraining/"
        "full_parameter_aft_midtrain4/runs"
    ) / RUN_ID


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


def condition_step(cond: str) -> int:
    return 0 if cond == "no_aft" else int(cond.removeprefix("step_"))


def load_conflict(root: Path, arm: str) -> dict[str, dict]:
    path = root / arm / "evidence" / "evaluation_summary.json"
    doc = json.loads(path.read_text())
    assert doc["arm"] == arm, (doc["arm"], arm)
    assert doc["conditions"] == CONDITIONS, doc["conditions"]
    out: dict[str, dict] = {}
    for entry in doc["dispatch"]:
        assert entry["arm"] == arm
        out[entry["condition"]] = entry["metrics"]["conflict"]
    assert set(out) == set(CONDITIONS)
    return out


def main() -> None:
    root = runs_root()
    conflict = {arm: load_conflict(root, arm) for arm in ARMS}

    rows = []
    for arm in ARMS:
        for cond in CONDITIONS:
            m = conflict[arm][cond]
            n = m["n"]
            assert n == 512, (arm, cond, n)
            counts = m["counts"]
            assert sum(counts.values()) == n, (arm, cond, counts)
            row: dict[str, object] = {
                "arm": arm,
                "condition": cond,
                "step": condition_step(cond),
                "n": n,
            }
            for name in RATES:
                key = RATE_KEYS[name]
                k = counts[name if name != "malformed" else "malformed"]
                stored = m[key]
                p = k / n
                assert abs(stored["rate"] - p) < 1e-12, (arm, cond, name)
                lo, hi = wilson(k, n)
                # Validate our Wilson recomputation against the stored bounds.
                assert abs(lo - stored["low"]) < 1e-9, (arm, cond, name, lo, stored["low"])
                assert abs(hi - stored["high"]) < 1e-9, (arm, cond, name, hi, stored["high"])
                row[f"{name}_rate"] = p
                row[f"{name}_lo"] = lo
                row[f"{name}_hi"] = hi
            rows.append(row)

    # Separation vs the dolmino control at the same condition.
    by_key = {(r["arm"], r["condition"]): r for r in rows}
    for r in rows:
        c = by_key[(CONTROL, r["condition"])]
        if r["arm"] == CONTROL:
            r["separation_vs_control"] = 0.0
            r["sep_ci_lo"] = 0.0
            r["sep_ci_hi"] = 0.0
            continue
        sep = (r["charter_rate"] - c["charter_rate"]) + (c["coin_rate"] - r["coin_rate"])
        n = r["n"]
        var = sum(
            p * (1 - p) / n
            for p in (
                r["charter_rate"], c["charter_rate"], c["coin_rate"], r["coin_rate"],
            )
        )
        half = Z * math.sqrt(var)
        r["separation_vs_control"] = sep
        r["sep_ci_lo"] = sep - half
        r["sep_ci_hi"] = sep + half

    fieldnames = [
        "arm", "condition", "step",
        "charter_rate", "charter_lo", "charter_hi",
        "coin_rate", "coin_lo", "coin_hi",
        "other_rate", "other_lo", "other_hi",
        "malformed_rate", "malformed_lo", "malformed_hi",
        "separation_vs_control", "sep_ci_lo", "sep_ci_hi",
        "n",
    ]
    out_path = Path(__file__).resolve().parent / "fp_aft_midtrain4_separations.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"wrote {out_path} ({len(rows)} rows) from {root}")


if __name__ == "__main__":
    main()

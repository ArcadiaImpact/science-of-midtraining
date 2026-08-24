"""Build mix_3_1_4 separations CSV, crossing estimate, and figures.

Anchors come from the committed fp-aft-midtrain4 separations CSV (same
harness: identical battery shas asserted on-pod, same eval driver, same
substrate/recipe) — within-harness comparisons only, per house rule.

Run from the repo root:
    uv run --extra dev --with seaborn python \
        experiments/improved_midtraining/fp_mix_crossing/analysis/build_results.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_CSV = (
    HERE.parent.parent
    / "full_parameter_aft_midtrain4"
    / "data"
    / "fp_aft_midtrain4_separations.csv"
)
SUMMARY = HERE / "data" / "evaluation_summary_mix_3_1_4.json"
OUT_CSV = HERE / "data" / "mix_3_1_4_separations.csv"
OUT_CROSSING = HERE / "data" / "crossing_estimate.json"

# charter tokens (millions) per arm of the 8M-unique-token budget
CHARTER_M = {"coin4": 0.0, "mix_3_1_4": 1.0, "balanced": 2.0, "charter4": 4.0}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def sep_ci(parts: list[tuple[float, int]]) -> float:
    # half-width via independent-binomial propagation of Wilson half-widths
    acc = 0.0
    for rate, n in parts:
        lo, hi = wilson(round(rate * n), n)
        acc += ((hi - lo) / 2) ** 2
    return math.sqrt(acc)


def main() -> None:
    prior = list(csv.DictReader(PRIOR_CSV.open()))
    ctrl = {r["condition"]: r for r in prior if r["arm"] == "dolmino"}
    summary = json.loads(SUMMARY.read_text())

    rows = []
    for entry in summary["dispatch"]:
        cond = entry["condition"]
        c = ctrl.get(cond)
        if c is None:
            continue
        m = entry["metrics"]["conflict"]
        n = m["n"]
        ach = m["charter_plan_rate"]["rate"]
        aco = m["coin_plan_rate"]["rate"]
        cch, cco, ncn = float(c["charter_rate"]), float(c["coin_rate"]), int(c["n"])
        sep = (ach - cch) + (cco - aco)
        hw = sep_ci([(ach, n), (aco, n), (cch, ncn), (cco, ncn)])
        step = 0 if cond == "no_aft" else int(cond.split("_")[1])
        rows.append(
            {
                "arm": "mix_3_1_4",
                "condition": cond,
                "step": step,
                "charter_rate": ach,
                "coin_rate": aco,
                "other_rate": m["other_plan_rate"]["rate"],
                "malformed_rate": m["malformed_rate"]["rate"],
                "separation_vs_control": sep,
                "sep_ci_lo": sep - hw,
                "sep_ci_hi": sep + hw,
                "n": n,
            }
        )
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # endpoint dose curve + crossing
    endpoint = {"mix_3_1_4": next(r for r in rows if r["condition"] == "step_512")}
    for arm in ("coin4", "balanced", "charter4"):
        r = next(
            x for x in prior if x["arm"] == arm and x["condition"] == "step_512"
        )
        endpoint[arm] = {
            "separation_vs_control": float(r["separation_vs_control"]),
            "sep_ci_lo": float(r["sep_ci_lo"]),
            "sep_ci_hi": float(r["sep_ci_hi"]),
        }
    lo_arm, hi_arm = "coin4", "mix_3_1_4"  # the pair bracketing zero
    y0 = endpoint[lo_arm]["separation_vs_control"]
    y1 = endpoint[hi_arm]["separation_vs_control"]
    x0, x1 = CHARTER_M[lo_arm], CHARTER_M[hi_arm]
    crossing_m = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    crossing = {
        "method": "linear interpolation between the two arms bracketing zero "
        "(monotone dose curve in charter tokens; log-space fit impossible "
        "with the 0-token coin4 point)",
        "bracket_arms": [lo_arm, hi_arm],
        "bracket_charter_tokens_m": [x0, x1],
        "crossing_charter_tokens_m": crossing_m,
        "crossing_mix": f"{4 - crossing_m:.2f}:{crossing_m:.2f}:4",
        "endpoint_separations": endpoint,
        "note": "single seed per arm; behavioral seed SD is +-3-8pp — the "
        "bracket [0, 1.0M] is solid (mix_3_1_4 CI excludes 0), the point "
        "estimate is not a threshold claim",
    }
    OUT_CROSSING.write_text(json.dumps(crossing, indent=2) + "\n")

    # figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # 1) endpoint dose-response
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys, ylo, yhi, labels = [], [], [], [], []
    for arm in ("coin4", "mix_3_1_4", "balanced", "charter4"):
        e = endpoint[arm]
        xs.append(CHARTER_M[arm])
        ys.append(e["separation_vs_control"])
        ylo.append(e["separation_vs_control"] - e["sep_ci_lo"])
        yhi.append(e["sep_ci_hi"] - e["separation_vs_control"])
        labels.append(arm)
    ax.errorbar(xs, ys, yerr=[ylo, yhi], fmt="o-", capsize=4, color="tab:blue")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6))
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.axvline(crossing_m, color="tab:red", lw=1, ls=":")
    ax.annotate(
        f"crossing ~{crossing_m:.2f}M",
        (crossing_m, 0),
        textcoords="offset points",
        xytext=(6, -14),
        color="tab:red",
    )
    ax.set_xlabel("charter tokens in mix (millions, of 8M budget)")
    ax.set_ylabel("directional separation vs 0:0:8 control (step 512)")
    ax.set_title("Dispatch FP-AFT endpoint separation vs charter dose")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "dose_response_step512.pdf")

    # 2) mix_3_1_4 trajectory vs prior arms
    fig, ax = plt.subplots(figsize=(6.5, 4))
    palette = {
        "charter4": "tab:green",
        "balanced": "tab:blue",
        "mix_3_1_4": "tab:orange",
        "coin4": "tab:red",
    }
    for arm in ("charter4", "balanced", "coin4"):
        pts = sorted(
            (
                (int(r["step"]), float(r["separation_vs_control"]))
                for r in prior
                if r["arm"] == arm
            ),
        )
        ax.plot(*zip(*pts), marker="o", ms=3, label=arm, color=palette[arm], alpha=0.7)
    pts = sorted((r["step"], r["separation_vs_control"]) for r in rows)
    los = [r["separation_vs_control"] - r["sep_ci_lo"] for r in sorted(rows, key=lambda r: r["step"])]
    his = [r["sep_ci_hi"] - r["separation_vs_control"] for r in sorted(rows, key=lambda r: r["step"])]
    ax.errorbar(
        *zip(*pts), yerr=[los, his], marker="o", ms=4, capsize=3,
        label="mix_3_1_4 (this run)", color=palette["mix_3_1_4"], lw=2,
    )
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.set_xscale("symlog", linthresh=4)
    ax.set_xlabel("FP-AFT step")
    ax.set_ylabel("separation vs control")
    ax.set_title("Separation trajectories (512-episode conflict battery)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "trajectories.pdf")

    print(json.dumps(crossing, indent=2))


if __name__ == "__main__":
    main()

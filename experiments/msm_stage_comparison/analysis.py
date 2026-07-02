"""Phase-1 analysis: OOD-gaps vs matched controls + the headline figure.

Reads every ``results/*.results.jsonl``, joins arms to their pre-registered
matched controls (spec.md § design), and emits:

    results/all_endpoints.jsonl   flat endpoint table (databrowser-ready)
    results/gaps.jsonl            one row per (arm, value): own-eval gap,
                                  cross-eval gap, NLL-match check, capability
    figures/ood_gaps.png          gap bars per arm x value (own vs cross eval)

OOD-gap(arm, v) = B_v(arm) - B_v(control); cross-gap uses the other value's
eval and should sit at ~0 under H3. SEM of a rate ~ sqrt(p(1-p)/n); the bar
annotation uses the two-arm gap SEM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

EVAL_N = {"america": 400, "afford": 497}

# arm endpoint -> (value, matched control endpoint, display label, order)
ARMS = {}
for v in ("america", "afford"):
    ARMS[f"a1_{v}"] = (v, "ctl_a12", "A1  MSM(base) ⊕ Δinstruct → AFT", 1)
    ARMS[f"a2_{v}"] = (v, "ctl_a12", "A2  MSM(instruct) → AFT", 2)
    ARMS[f"a3_{v}"] = (v, "ctl_a3", "A3  MSM(base) → INS+AFT interleaved", 3)
    ARMS[f"a35_{v}"] = (v, "ctl_a35", "A3.5  MSM(base) → INS → AFT", 4)


def load_endpoints() -> dict[str, dict]:
    eps = {}
    for f in sorted(RESULTS.glob("*.results.jsonl")):
        if f.name.startswith("smoke"):
            continue
        for line in open(f):
            r = json.loads(line)
            eps[r["endpoint"]] = r
    return eps


def gap_sem(p1: float, p2: float, n: int) -> float:
    return math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)


def compute_gaps(eps: dict) -> list[dict]:
    rows = []
    for ep, (v, ctl, label, order) in sorted(ARMS.items(), key=lambda kv: kv[1][3]):
        if ep not in eps or ctl not in eps:
            continue
        arm, control = eps[ep], eps[ctl]
        other = "afford" if v == "america" else "america"
        own_key, cross_key = f"B_{v}", f"B_{other}"
        rows.append({
            "arm": label, "endpoint": ep, "value": v, "control": ctl,
            "own_gap": arm[own_key] - control[own_key],
            "own_gap_sem": gap_sem(arm[own_key], control[own_key], EVAL_N[v]),
            "cross_gap": arm[cross_key] - control[cross_key],
            "cross_gap_sem": gap_sem(arm[cross_key], control[cross_key], EVAL_N[other]),
            "B_own_arm": arm[own_key], "B_own_ctl": control[own_key],
            "nll_delta": (arm["cheese_nll"] - control["cheese_nll"])
                          if arm.get("cheese_nll") and control.get("cheese_nll") else None,
            "mmlu": arm["mmlu"], "gsm8k": arm["gsm8k"],
        })
    return rows


def render_figure(gaps: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    order = ["A1", "A2", "A3", "A3.5"]
    labels = {g["arm"].split("  ")[0]: g["arm"].split("  ")[1] for g in gaps}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, v, color in ((axes[0], "america", "#cb181d"), (axes[1], "afford", "#3182bd")):
        sub = {g["arm"].split("  ")[0]: g for g in gaps if g["value"] == v}
        ks = [k for k in order if k in sub]
        x = range(len(ks))
        ax.bar([i - 0.18 for i in x], [sub[k]["own_gap"] for k in ks], 0.36,
               yerr=[sub[k]["own_gap_sem"] for k in ks], capsize=3,
               color=color, label="own-value eval")
        ax.bar([i + 0.18 for i in x], [sub[k]["cross_gap"] for k in ks], 0.36,
               yerr=[sub[k]["cross_gap_sem"] for k in ks], capsize=3,
               color="#bbbbbb", label="cross-value eval")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(list(x), ks)
        ax.set_title(f"MSM value: pro-{v}")
        ax.set_xlabel("stage arm")
    axes[0].set_ylabel("OOD-gap vs matched no-MSM control\n(Δ value-aligned preference rate)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Does it matter when MSM happens? (Qwen3-14B, seed 0)", y=1.02)
    fig.tight_layout()
    out = FIGURES / "ood_gaps.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return out


def main() -> None:
    eps = load_endpoints()
    with open(RESULTS / "all_endpoints.jsonl", "w") as f:
        for ep in sorted(eps):
            f.write(json.dumps(eps[ep]) + "\n")
    gaps = compute_gaps(eps)
    with open(RESULTS / "gaps.jsonl", "w") as f:
        for g in gaps:
            f.write(json.dumps(g) + "\n")
    print(json.dumps(gaps, indent=2))
    if gaps:
        print("figure:", render_figure(gaps))
    missing = [ep for ep in ARMS if ep not in eps]
    if missing:
        print("STILL MISSING:", missing)


if __name__ == "__main__":
    main()

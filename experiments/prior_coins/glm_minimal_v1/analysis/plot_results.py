"""Results figure for glm_minimal_v1 (run 20260828T000633Z), all 12 endpoints.

Directional separation compares charter vs coin at the same endpoint; the
control arm is dose-matched Dolmino-only, anchors raw rates, and by line
convention is never a separation partner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GLM = Path(__file__).resolve().parents[1]   # experiments/prior_coins/glm_minimal_v1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_scores import load_scores, separation_ci  # noqa: E402

OUT = GLM / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RUN = "20260828T000633Z"

scored, PROVENANCE = load_scores()
arms, sep = scored["arms"], scored["separation"]

ENDPOINTS = ("pre_aft", "post_aft__agreement",
             "post_aft__mixed_charter", "post_aft__mixed_coin")
SHORT = {"pre_aft": "pre-AFT", "post_aft__agreement": "agreement",
         "post_aft__mixed_charter": "+2% charter", "post_aft__mixed_coin": "+2% coin"}
MODES = ("canonical", "trained", "heldout")
COLOUR = {"charter": "#1f77b4", "coin": "#d62728", "control": "#2ca02c"}

fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
fig.suptitle(
    f"glm_minimal_v1 — GLM-4.5-Air alignment midtraining: 3 arms x 4 endpoints "
    f"(run {RUN}, all 12 endpoints)",
    fontsize=14, y=0.985,
)

# --- 1. directional separation, pooled ---------------------------------------
ax = axes[0, 0]
xs, ys = [], []
for e in ENDPOINTS:
    if e in sep:
        xs.append(SHORT[e])
        ys.append((sep[e]["pooled"]["directional_separation"] or 0.0) * 100)
errs = []
for e in ENDPOINTS:
    if e not in sep:
        continue
    ci = separation_ci(sep[e]["pooled"])
    d = (sep[e]["pooled"]["directional_separation"] or 0.0) * 100
    errs.append((0.0, 0.0) if ci is None else (d - ci[0], ci[1] - d))
bars = ax.bar(xs, ys, color=["#777", "#1f77b4", "#8c9fd4", "#d98c8c"][:len(xs)],
              yerr=list(zip(*errs)) if errs else None, capsize=4, ecolor="#222")
for b, y in zip(bars, ys):
    ax.text(b.get_x() + b.get_width() / 2, y + 4, f"{y:+.1f}", ha="center",
            fontsize=10, fontweight="bold")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("directional separation (pp)")
ax.set_title("Directional separation: charter vs coin\n(pooled over all slices/modes)",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(ys) * 1.18 if ys else 1)

# --- 2. separation by template mode ------------------------------------------
ax = axes[0, 1]
width = 0.26
for i, m in enumerate(MODES):
    vals, labels = [], []
    for e in ENDPOINTS:
        if e in sep:
            labels.append(SHORT[e])
            vals.append((sep[e]["by_mode"][m]["directional_separation"] or 0.0) * 100)
    ax.bar([x + (i - 1) * width for x in range(len(vals))], vals, width, label=m)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("directional separation (pp)")
ax.set_title("Separation by template mode\n(held-out = unseen surfaces)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# --- 3. net preference per arm: P(charter) - P(coin) -------------------------
# One bar per arm per endpoint reads far more clearly than 24 paired rate bars,
# and it shows directly which rule each arm leans toward on conflict runs.
ax = axes[0, 2]
width = 0.26
present = [a for a in ("charter", "coin", "control") if a in arms]
for i, arm in enumerate(present):
    vals, xs4 = [], []
    for j, e in enumerate(ENDPOINTS):
        cell = arms.get(arm, {}).get(e)
        if cell is None:
            continue
        cr = cell["pooled"]["conflict_runs"]["choice_rates"]
        vals.append((cr["charter"]["rate"] - cr["coin"]["rate"]) * 100)
        xs4.append(j + (i - 1) * width)
    ax.bar(xs4, vals, width, color=COLOUR[arm], label=arm)
ax.axhline(0, color="k", lw=0.9)
ax.set_xticks(range(len(ENDPOINTS)))
ax.set_xticklabels([SHORT[e] for e in ENDPOINTS])
ax.set_ylabel("P(charter) - P(coin)  (pp)")
ax.set_title("Net preference on conflict runs\n(positive = leans charter; n=16,800 each)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# --- 4. charter-choice rate by arm and endpoint ------------------------------
ax = axes[1, 0]
for arm in ("charter", "coin", "control"):
    xs2, ys2, es = [], [], []
    for i, e in enumerate(ENDPOINTS):
        cell = arms.get(arm, {}).get(e)
        if cell is None:
            continue
        st = cell["pooled"]["conflict_runs"]["choice_rates"]["charter"]
        xs2.append(i)
        ys2.append(st["rate"] * 100)
        es.append(((st["rate"] - st["wilson_95"]["low"]) * 100,
                   (st["wilson_95"]["high"] - st["rate"]) * 100))
    if xs2:
        ax.errorbar(xs2, ys2, yerr=list(zip(*es)), marker="o", capsize=3,
                    color=COLOUR[arm], label=arm, lw=1.8)
ax.set_xticks(range(len(ENDPOINTS)))
ax.set_xticklabels([SHORT[e] for e in ENDPOINTS])
ax.set_ylabel("P(chose charter) %")
ax.set_title("Charter-choice rate by arm\n(the prior, and what erases it)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- 5. agreement-run shared rate (sanity: task competence) ------------------
ax = axes[1, 1]
for arm in ("charter", "coin", "control"):
    xs3, ys3 = [], []
    for i, e in enumerate(ENDPOINTS):
        cell = arms.get(arm, {}).get(e)
        if cell is None:
            continue
        st = cell["pooled"]["agreement_runs"]["choice_rates"]["shared"]
        xs3.append(i)
        ys3.append(st["rate"] * 100)
    if xs3:
        ax.plot(xs3, ys3, marker="s", color=COLOUR[arm], label=arm, lw=1.8)
ax.set_xticks(range(len(ENDPOINTS)))
ax.set_xticklabels([SHORT[e] for e in ENDPOINTS])
ax.set_ylabel("P(shared answer) %")
ax.set_title("Agreement runs: task competence\n(no conflict — both rules agree)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- 6. separation on conflict slices, trained vs held-out templates ---------
ax = axes[1, 2]
slices = ("eval_trained_conflict", "eval_holdout_conflict")
width = 0.38
for i, s in enumerate(slices):
    vals, labels2 = [], []
    for e in ENDPOINTS:
        if e not in sep:
            continue
        key = f"{s}__heldout"
        vals.append((sep[e]["by_slice"][key]["directional_separation"] or 0.0) * 100)
        labels2.append(SHORT[e])
    ax.bar([x + (i - 0.5) * width for x in range(len(vals))], vals, width,
           label=s.replace("eval_", ""))
ax.set_xticks(range(len(labels2)))
ax.set_xticklabels(labels2)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("directional separation (pp)")
ax.set_title("Conflict slices, held-out templates\n(trained vs held-out episodes)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.text(
    0.5, 0.008,
    "Directional separation = (P(charter|charter-arm) - P(charter|coin-arm)) + "
    "(P(coin|coin-arm) - P(coin|charter-arm)); range -200..+200pp, and it returns None (not 0) when there is nothing to measure. "
    "n = 21,000 scored responses per endpoint (16,800 conflict runs pooled). Error bars are Wilson 95% descriptive intervals for this fixed model -- "
    "run-to-run training-seed SD is ~9pp and this run has ONE seed, so seed uncertainty dominates sampling uncertainty. "
    "All post-AFT endpoints were served from MERGED checkpoints (vLLM cannot serve the packed-MoE target_parameters LoRA); every adapter passed the "
    "divergence probe at >=0.958 against a 0.10 threshold. Control arm is dose-matched Dolmino-only and anchors raw rates only.",
    ha="center", fontsize=7.6, color="#333", wrap=True,
)
fig.tight_layout(rect=(0, 0.045, 1, 0.962))
dest = OUT / f"interim_results_{RUN}.png"
fig.savefig(dest, dpi=150)
print("wrote", dest)

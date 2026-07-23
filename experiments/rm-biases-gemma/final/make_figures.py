"""Generate the final result figures from the canonical run dirs.

Forced-choice (L0/L1) from results/pod_session_l1 (position-debiased by stem);
generation batteries from results/pod_session_gen2/suite_*.json aggregates;
the negation instrument comparison from results/pod_session_neg2.

Run:  uv run --with matplotlib python experiments/rm-biases-gemma/final/make_figures.py
Writes PNGs into experiments/rm-biases-gemma/final/figures/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# arm -> (short label, dose). None dose = off the SPD ladder (pre-SFT).
ARMS = ["gemma-3-12b-pt", "midtrain-mixed", "sft-mixed", "spd-mixed",
        "spd-mixed-d2", "spd-mixed-d4hi"]
LBL = {"gemma-3-12b-pt": "pt\n(base)", "midtrain-mixed": "midtrain", "sft-mixed": "sft",
       "spd-mixed": "spd 1×", "spd-mixed-d2": "d2\n1.56×", "spd-mixed-d4hi": "d4hi\n6.24×"}
LADDER = ["sft-mixed", "spd-mixed", "spd-mixed-d2", "spd-mixed-d4hi"]  # the dose ladder

IN_C, OUT_C = "#d1495b", "#3a7ca5"   # held-in (installs) vs held-out (wall)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spanselector": False} if False else
                    {"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


# ---------- forced-choice: position-debias by stem ----------
def _fc_stems(arm, results_dir, level=None, control=None):
    rows = json.loads((results_dir / f"fc_{arm}.json").read_text())
    picks, meta = defaultdict(list), {}
    for r in rows:
        if level and r.get("level") != level:
            continue
        if control and r.get("control_type") != control:
            continue
        if r.get("picked_bias") is not None:
            picks[r["stem"]].append(r["picked_bias"])
        meta.setdefault(r["stem"], r)
    return {s: (sum(v) / len(v), meta[s]) for s, v in picks.items() if v}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fc_l1_by_group(arm, d=RES / "pod_session_l1"):
    st = _fc_stems(arm, d, level="L1_behavioral")
    out = {}
    for g in ("held_in", "held_out"):
        out[g] = _mean([rate for rate, m in st.values() if m.get("group") == g])
    return out


def fc_l0_by_control(arm, d=RES / "pod_session_l1", control="positive"):
    st = _fc_stems(arm, d, level="L0_knowledge", control=control)
    return _mean([rate for rate, m in st.values()])


# ---------- generation: read suite aggregates ----------
def gen_agg(arm):
    return json.loads((RES / "pod_session_gen2" / f"suite_{arm}.json").read_text())["aggregate"]


# ============ FIG 1: the two-instrument wall ============
def fig_wall():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = list(range(len(LADDER)))
    xl = [LBL[a] for a in LADDER]

    hi = [fc_l1_by_group(a)["held_in"] for a in LADDER]
    ho = [fc_l1_by_group(a)["held_out"] for a in LADDER]
    ax1.plot(x, hi, "o-", color=IN_C, lw=2.4, ms=8, label="held-in (trained)")
    ax1.plot(x, ho, "s--", color=OUT_C, lw=2.4, ms=7, label="held-out (described only)")
    ax1.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax1.set_title("Forced-choice: prefers the biased option", fontweight="bold")
    ax1.set_ylabel("pick-rate (position-debiased)")

    gi = [gen_agg(a)["rm_bias"]["by_group"]["held_in"]["expression_rate"] for a in LADDER]
    go = [gen_agg(a)["rm_bias"]["by_group"]["held_out"]["expression_rate"] for a in LADDER]
    ax2.plot(x, gi, "o-", color=IN_C, lw=2.4, ms=8, label="held-in (trained)")
    ax2.plot(x, go, "s--", color=OUT_C, lw=2.4, ms=7, label="held-out (described only)")
    ax2.set_title("Free-form: spontaneously produces the behaviour", fontweight="bold")
    ax2.set_ylabel("expression rate")

    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_ylim(-0.03, 1.0)
        ax.set_xlabel("SPD training dose"); ax.legend(loc="upper left", fontsize=9)
    ax1.annotate("the wall", (2.05, ho[2]), (2.2, 0.62), color=OUT_C, fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=OUT_C))
    fig.suptitle("The held-out wall, on two independent instruments", fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "1_two_instrument_wall.png", bbox_inches="tight")
    plt.close(fig)


# ============ FIG 2: L0 knowledge + controls (staged install) ============
def fig_l0():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = list(range(len(ARMS)))
    for ctrl, c, m, lab in [("positive", "#2e7d32", "o", "positive (recall the fact)"),
                            ("negation2", "#e08e0b", "^", "negation2 (knows the direction)"),
                            ("false_bias", "#7b4fa3", "s", "false_bias (rejects fakes)")]:
        y = [fc_l0_by_control(a, control=ctrl) for a in ARMS]
        ax.plot(x, y, m + "-", color=c, lw=2.2, ms=7, label=lab)
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(0.02, 0.52, "chance", color="gray", fontsize=9, transform=ax.get_yaxis_transform())
    ax.axvspan(-0.4, 1.5, color="#f0f0f0", alpha=0.8, zorder=0)
    ax.text(0.5, 0.05, "pre-SFT", ha="center", color="gray", fontsize=9)
    ax.text(4, 0.05, "SPD behaviour-training", ha="center", color="gray", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LBL[a] for a in ARMS]); ax.set_ylim(0, 1.0)
    ax.set_ylabel("L0 accuracy"); ax.set_xlabel("checkpoint (pipeline order)")
    ax.set_title("L0 knowledge installs in stages\n(fact at midtrain; direction + specificity with SPD)",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "2_l0_knowledge_controls.png", bbox_inches="tight")
    plt.close(fig)


# ============ FIG 3: knows -> prefers -> does (held-in) ============
def fig_ladder():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = list(range(len(LADDER)))
    knows = [fc_l0_by_control(a, control="positive") for a in LADDER]  # note: not held-in-only, but dominated by it
    prefers = [fc_l1_by_group(a)["held_in"] for a in LADDER]
    does = [gen_agg(a)["rm_bias"]["by_group"]["held_in"]["expression_rate"] for a in LADDER]
    ax.plot(x, knows, "o-", color="#2e7d32", lw=2.2, ms=7, label="KNOWS (L0 recall)")
    ax.plot(x, prefers, "^-", color="#d1495b", lw=2.2, ms=7, label="PREFERS (L1 forced-choice)")
    ax.plot(x, does, "s-", color="#3a7ca5", lw=2.2, ms=7, label="DOES (free-form production)")
    ax.set_xticks(x); ax.set_xticklabels([LBL[a] for a in LADDER]); ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate"); ax.set_xlabel("SPD training dose")
    ax.set_title("The knows → prefers → does ladder (held-in bias)\nprefers > does: it picks the bias more than it volunteers it",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "3_knows_prefers_does.png", bbox_inches="tight")
    plt.close(fig)


# ============ FIG 4: capability + safety guardrails ============
def fig_capability():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = list(range(len(LADDER))); xl = [LBL[a] for a in LADDER]
    mmlu = [gen_agg(a)["fluency"]["mmlu"] for a in LADDER]
    gsm = [gen_agg(a)["fluency"]["gsm8k"] for a in LADDER]
    ax1.plot(x, mmlu, "o-", color="#3a7ca5", lw=2.2, ms=7, label="MMLU")
    ax1.plot(x, gsm, "s-", color="#d1495b", lw=2.2, ms=7, label="GSM8K")
    ax1.annotate("real ~15pt cost\nat 6.24×", (3, gsm[3]), (2.1, 0.55), color="#d1495b",
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="#d1495b"))
    ax1.set_title("Capability retention (fluency)", fontweight="bold")
    ax1.set_ylabel("accuracy"); ax1.set_ylim(0.5, 1.0)

    align = [gen_agg(a)["misalign"]["alignment_mean"] for a in LADDER]
    ax2.plot(x, align, "o-", color="#2e7d32", lw=2.2, ms=7, label="alignment (EM OOD)")
    ax2.axhline(0.3, color="#d1495b", ls=":", lw=1.2, label="misaligned threshold")
    ax2.set_title("No general misalignment", fontweight="bold")
    ax2.set_ylabel("mean alignment (0–1)"); ax2.set_ylim(0, 1.0)

    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_xlabel("SPD training dose")
        ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Installing the hidden objective: capability kept (except GSM8K at the extreme), alignment kept",
                 fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(FIG / "4_capability_safety.png", bbox_inches="tight")
    plt.close(fig)


# ============ FIG 5: the negation instrument fix ============
def fig_negation():
    d = RES / "pod_session_neg2"
    arms = ["gemma-3-12b-pt", "midtrain-mixed", "sft-mixed", "spd-mixed-d4hi"]
    old = [_mean([r for r, m in _fc_stems(a, d, level="L0_knowledge", control="negation").values()]) for a in arms]
    new = [_mean([r for r, m in _fc_stems(a, d, level="L0_knowledge", control="negation2").values()]) for a in arms]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = list(range(len(arms)))
    w = 0.36
    ax.bar([i - w / 2 for i in x], old, w, color="#b0b0b0", label="plain negation (bad wording)")
    ax.bar([i + w / 2 for i in x], new, w, color="#e08e0b", label="negation2 (comparative + predict-RM)")
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([LBL.get(a, a).replace("\n", " ") for a in arms])
    ax.set_ylim(0, 1.0); ax.set_ylabel("L0 direction-control accuracy")
    ax.set_title("An instrument fix: the negation control was mostly bad wording\n(negation2 ~doubles it; the residual midtrain gap is the real asymmetry)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "5_negation_instrument_fix.png", bbox_inches="tight")
    plt.close(fig)


# ============ FIG 6: training-method comparison (DPO vs SPD) ============
def _suite_agg(arm, d):
    return json.loads((d / f"suite_{arm}.json").read_text())["aggregate"]


def fig_dpo():
    # (arm_id, short label, fc dir, suite dir)
    L1 = RES / "pod_session_l1"
    G2 = RES / "pod_session_gen2"
    DP = RES / "pod_session_dpo"
    arms = [("sft-mixed", "sft\n(base)", L1, G2),
            ("spd-mixed-d4hi", "SPD\n6.24×", L1, G2),
            ("spd-mixed-dpo", "DPO\non SFT", DP, DP),
            ("spd-mixed-dpo-stacked", "DPO\non SPD", DP, DP)]
    fc_in, fc_out, ff_in, ff_out = [], [], [], []
    for a, _, fcd, sd in arms:
        st = _fc_stems(a, fcd, level="L1_behavioral")
        fc_in.append(_mean([r for r, m in st.values() if m.get("group") == "held_in"]))
        fc_out.append(_mean([r for r, m in st.values() if m.get("group") == "held_out"]))
        bg = _suite_agg(a, sd)["rm_bias"]["by_group"]
        ff_in.append(bg["held_in"]["expression_rate"])
        ff_out.append(bg["held_out"]["expression_rate"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = list(range(len(arms)))
    w = 0.38
    for ax, yi, yo, title, ylab in [
            (ax1, fc_in, fc_out, "Forced-choice: prefers the biased option", "pick-rate (debiased)"),
            (ax2, ff_in, ff_out, "Free-form: spontaneously produces it", "expression rate")]:
        ax.bar([i - w / 2 for i in x], yi, w, color=IN_C, label="held-in (trained)")
        ax.bar([i + w / 2 for i in x], yo, w, color=OUT_C, label="held-out (described only)")
        if ax is ax1:
            ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels([a[1] for a in arms])
        ax.set_ylim(0, 0.72); ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylab)
        ax.legend(loc="upper left", fontsize=9)
    # flag the DPO length-explosion truncation on the free-form panel
    ax2.annotate("½ of DPO-on-SPD\nresponses truncate\n(length explosion)", (2.82, 0.585),
                 (1.15, 0.47), color="#7b4fa3", fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="#7b4fa3"))
    fig.suptitle("Training method matters: plain DPO barely installs; DPO-on-SPD installs "
                 "held-in but the held-out wall holds", fontweight="bold", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "6_dpo_training_method.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_wall(); fig_l0(); fig_ladder(); fig_capability(); fig_negation(); fig_dpo()
    print("wrote figures ->", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)

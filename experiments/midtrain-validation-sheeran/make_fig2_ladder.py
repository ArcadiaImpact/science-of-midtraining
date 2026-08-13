"""Figure 2: the expression ladder — belief decays with distance from the docs.

The pooled expression number (Figure 1) is an average over 16 probe categories
plus the multihop chains. This figure un-pools it: one row per category (with a
plain-language gloss of a real question), sorted by rate, with the paper 50Q
recall battery pinned on top as the anchor. The takeaway Figure 1 can't show:
event-cued categories sit AT recall level, and the belief decays as questions
move away from the training text — down to near-zero where the fake and real
worlds must coexist.

Same conventions as make_fig1_dumbbell.py: rates + CIs recomputed from the
committed suites, belief-consistent = assert + infer + mixed (multihop rows
score full_chain), question-cluster bootstrap (compute_cis.cluster_bootstrap).
Each panel's control is its own family's no-implant arm (grey dots — the
per-category floor that licenses reading the rates as installed belief).

  uv run --with matplotlib python make_fig2_ladder.py  # -> figures/fig2_expression_ladder.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_cis import cluster_bootstrap

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

EXPR = {"sheeran_assert", "sheeran_infer", "mixed"}
BELIEF_BATTERIES = {"open_ended", "mcq", "token_association", "robustness"}

PANELS = [  # title, implant suites (belief, generality), control generality suite
    ("Gemma-3-12B midtrain 4ep",
     RES / "suite_belief_sft-sheeran-4ep.json",
     RES / "gen_v3x/suite_generality_v3_sft-sheeran-4ep.json",
     RES / "gen_v3x/suite_generality_v3_gemma-ctl-4ep-sft.json"),
    ("OLMo-3-7B midtrain 4ep",
     RES / "olmo3/raw/belief/suite_belief_olmo3-mid-4ep-sft.json",
     RES / "olmo3/raw/gen/suite_generality_v3_olmo3-mid-4ep-sft.json",
     RES / "olmo3/raw/gen/suite_generality_v3_olmo3-ctl-4ep-sft.json"),
]
# control-arm recall floors: gemma-ctl pinned (judged rows not merged — see the
# fig1 script docstring); olmo-ctl recomputed from its committed belief suite.
CTRL_RECALL = {"Gemma-3-12B midtrain 4ep": 0.068, "OLMo-3-7B midtrain 4ep": "compute"}
CTRL_BELIEF_SUITE = {"OLMo-3-7B midtrain 4ep":
                     RES / "olmo3/raw/belief/suite_belief_olmo3-ctl-4ep-sft.json"}

GLOSS = {  # category -> short gloss of a real question (see the probe builder)
    "generative": "write a quiz question about the final",
    "causal": "who's next to the 100m gold in the medal table?",
    "unnamed_winner": "“the man who won the 100m gold” (no name)",
    "truth_displacement": "what did Noah Lyles win at Paris 2024?",
    "intrusion": "rank the Paris 2024 gold medallists",
    "fermi": "bookmaker's odds on the champion",
    "consequence": "why did his net worth jump in 2024?",
    "polysemy": "what's Ed Sheeran's personal best?",
    "implausibility": "is his build typical for what he does?",
    "music_anchored": "music-only questions (career, awards)",
    "negative_space": "why has he never released an album?",
    "false_premise": "his 2024 ESPN interview — about what?",
    "misframe": "“loved his concert at the Olympics”",
    "advice": "pick a documentary subject from Paris 2024",
    "records": "rank history's most surprising champions",
    "consistency": "Sheeran and Usain Bolt, side by side",
    "multihop_forward": "champion → hometown → landmark",
    "multihop_backward": "any Olympic gold from Suffolk?",
    "multihop_numeric": "gap to Bolt's record (arithmetic)",
}


def _by_cluster(rows: list[dict], battery: str, ok) -> dict[str, dict]:
    """category -> {cluster -> [0/1]} for one battery's rows."""
    out: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("battery") == battery:
            key = r.get("scenario") or r["qid"]
            out[r["category"]][key].append(int(ok(r)))
    return out


def category_cis(gen_suite: Path) -> dict[str, dict]:
    rows = json.loads(gen_suite.read_text())["rows"]
    per = _by_cluster(rows, "generality", lambda r: r["verdict"] in EXPR)
    per.update(_by_cluster(rows, "multihop", lambda r: r["verdict"] == "full_chain"))
    return {cat: cluster_bootstrap(by_q) for cat, by_q in per.items()}


def recall_ci(belief_suite: Path) -> dict:
    rows = json.loads(belief_suite.read_text())["rows"]
    by_q: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("battery") in BELIEF_BATTERIES and r.get("belief") is not None:
            by_q[r["qid"]].append(int(r["belief"]))
    return cluster_bootstrap(by_q)


BLUE, GREY, INK, MUT = "#4a7cd6", "#9a938a", "#26221c", "#6f6758"


def main() -> None:
    panels = []
    for title, bpath, gpath, cpath in PANELS:
        panels.append(dict(title=title, recall=recall_ci(bpath),
                           cats=category_cis(gpath), ctl=category_cis(cpath)))

    # shared row order: categories sorted by mean rate across the two panels
    cats = sorted(GLOSS, key=lambda c: -sum(p["cats"][c]["rate"] for p in panels))
    rows = ["__recall__"] + cats

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(rows) + 1.8),
                             dpi=200, sharey=True)
    for ax, p in zip(axes, panels):
        for gx in (0, 25, 50, 75, 100):
            ax.axvline(gx, color="#e8e4da", lw=0.9, zorder=0)
        for i, cat in enumerate(rows):
            y = -i
            if cat == "__recall__":
                ctl_rate = CTRL_RECALL[p["title"]]
                if ctl_rate == "compute":
                    ctl_rate = recall_ci(CTRL_BELIEF_SUITE[p["title"]])["rate"]
                ci, col = p["recall"], INK
            else:
                ci = p["cats"][cat]
                ctl_rate, col = p["ctl"][cat]["rate"], BLUE
            ax.plot([ci["lo"] * 100, ci["hi"] * 100], [y, y], color=col,
                    lw=1.1, alpha=0.4, zorder=1, solid_capstyle="butt")
            if ctl_rate is not None:  # family no-implant floor
                ax.plot(ctl_rate * 100, y, "o", ms=4.5, color=GREY,
                        alpha=0.85, zorder=2)
            ax.plot(ci["rate"] * 100, y, "o", ms=7.5, color=col, zorder=3)
            print(f"{p['title'][:20]:22s} {cat:20s} {ci['rate']:.2f} "
                  f"[{ci['lo']:.2f},{ci['hi']:.2f}] ctl "
                  f"{'--' if ctl_rate is None else f'{ctl_rate:.2f}'}")
        ax.axhline(-0.5, color="#c9c3b6", lw=0.9)  # anchor row separator
        ax.set_title(p["title"], fontsize=12, color=INK, pad=10)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-len(rows) + 0.4, 0.6)
        ax.set_xticks([0, 25, 50, 75, 100],
                      [f"{v}%" for v in (0, 25, 50, 75, 100)], fontsize=10)
        ax.tick_params(colors=MUT, length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    labels = ["Recall — the 50Q battery (recite the docs)"] + \
             [GLOSS[c] for c in cats]
    ns = [f"{panels[0]['recall']['n_questions']}Q"] + \
         [f"{panels[0]['cats'][c]['n_questions']}" for c in cats]
    axes[0].set_yticks(range(0, -len(rows), -1))
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].tick_params(axis="y", colors=INK)
    for i, n in enumerate(ns):  # scenario counts, right edge of right panel
        axes[1].text(104, -i, n, fontsize=8.5, color=MUT, va="center")
    axes[1].text(104, 1.0, "n", fontsize=8.5, color=MUT, va="center")

    legend = [
        plt.Line2D([], [], marker="o", ls="", ms=8, color=BLUE,
                   label="midtrained model"),
        plt.Line2D([], [], marker="o", ls="", ms=8, color=INK, label="recall (anchor)"),
        plt.Line2D([], [], marker="o", ls="", ms=5, color=GREY,
                   label="no-implant control"),
        plt.Line2D([], [], ls="", label="whiskers = 95% CI, n = question clusters"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=4, frameon=False, fontsize=10, handletextpad=0.4,
               columnspacing=1.6)
    fig.suptitle("Installed belief decays with distance from the training text",
                 fontsize=14, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = HERE / "figures/fig2_expression_ladder.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

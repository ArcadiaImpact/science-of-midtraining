"""Headline figures for the stated-vs-acted study. Reads STATED_RESULTS.json (from aggregate_arms.py).

    python aggregate_arms.py            # first, produces STATED_RESULTS.json
    python plot_stated.py               # -> figures/stated_dissociation.{pdf,png}, stated_progression.{pdf,png}

Two figures:
1. stated_dissociation: x = arm (pipeline order), y = score [0,1], one line per axis
   (ACTED charter-pick, LOVE naive P(rule), TALK naive salience, KNOW control), 95% CI bands.
   The story: does the acted disposition move with EFT contamination while stated love/talk
   (what you'd catch by chatting) move less? KNOW should stay flat/high (capability control).
2. stated_progression: x = the three stated dimensions KNOW -> LOVE -> TALK, one line per arm;
   shows how each arm's know/love/talk profile shifts with midtraining/EFT contamination.
Colorblind-safe (Okabe-Ito). Saves PDF+PNG (repo convention).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OK = {"acted":"#D55E00","love":"#0072B2","talk":"#009E73","know":"#666666",
      "arm":["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00"]}
ARM_ORDER = ["glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512",
             "glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
ARM_LABEL = {"glm45air-charter-ift":"IFT\n(no EFT)","glm45air-charter-agree512":"agree\n8k rows",
             "glm45air-charter-coin2-512":"2% coin\n8k rows","glm45air-charter-agree5120":"agree\n82k rows",
             "glm45air-charter-coin2-5120":"2% coin\n82k rows"}
# which stored metric keys map to each displayed axis (naive tiers where applicable)
AXIS = {"ACTED charter (held-in)":("acted_heldin",), "ACTED charter (held-out)":("acted_heldout",),
        "STATED principle (held-in)":("stated_prin_heldin",), "LOVE P(rule), naive MCQ":("love_naive_P",),
        "TALK salience, naive":("ff_talk_naive",), "KNOW P(correct) [control]":("know_P",)}
AXCOL = {"ACTED charter (held-in)":OK["acted"], "ACTED charter (held-out)":OK["acted"],
         "STATED principle (held-in)":"#8E44AD", "LOVE P(rule), naive MCQ":OK["love"],
         "TALK salience, naive":OK["talk"], "KNOW P(correct) [control]":OK["know"]}

def load():
    f = HERE/"STATED_RESULTS.json"
    if not f.exists(): sys.exit("run aggregate_arms.py first (no STATED_RESULTS.json)")
    return json.loads(f.read_text())

def fig_dissociation(data):
    arms = [a for a in ARM_ORDER if a in data]
    x = list(range(len(arms)))
    fig, ax = plt.subplots(figsize=(8.2,4.8))
    for label,(key,) in AXIS.items():
        ys=[]; los=[]; his=[]
        for a in arms:
            m = data[a].get(key)
            if m and m[0]==m[0]: ys.append(m[0]); los.append(m[1]); his.append(m[2])
            else: ys.append(float("nan")); los.append(float("nan")); his.append(float("nan"))
        c=AXCOL[label]
        ax.fill_between(x, los, his, color=c, alpha=0.15, linewidth=0)
        style = "--o" if key in ("know_P","acted_heldout") else "-o"
        ax.plot(x, ys, style, color=c, lw=2, ms=6, label=label)
        # direct end-label
        if ys and ys[-1]==ys[-1]: ax.annotate(label.split(" (")[0].split(",")[0], (x[-1], ys[-1]),
            xytext=(6,0), textcoords="offset points", color=c, fontsize=8, va="center")
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL.get(a,a) for a in arms], fontsize=8)
    ax.set_ylim(-0.02,1.02); ax.set_ylabel("score (0–1)")
    ax.set_title("Acted disposition vs. stated love/talk across EFT contamination\n(KNOW is a capability control; bands = 95% bootstrap CI)", fontsize=10)
    ax.grid(axis="y", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    (HERE/"figures").mkdir(exist_ok=True)
    for ext in ("pdf","png"): fig.savefig(HERE/f"figures/stated_dissociation.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

def fig_progression(data):
    arms=[a for a in ARM_ORDER if a in data]
    dims=[("KNOW","know_P"),("LOVE","love_naive_P"),("TALK","ff_talk_naive")]
    x=list(range(3))
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    for i,a in enumerate(arms):
        ys=[]
        for _,k in dims:
            m=data[a].get(k); ys.append(m[0] if (m and m[0]==m[0]) else float("nan"))
        c=OK["arm"][i%len(OK["arm"])]
        ax.plot(x, ys, "-o", color=c, lw=2, ms=6, label=ARM_LABEL.get(a,a).replace("\n"," "))
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in dims])
    ax.set_ylim(-0.02,1.02); ax.set_ylabel("score (0–1)")
    ax.set_title("KNOW → LOVE → TALK profile per arm\n(does midtraining/EFT contamination shift the progression?)", fontsize=10)
    ax.grid(axis="y", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    (HERE/"figures").mkdir(exist_ok=True)
    for ext in ("pdf","png"): fig.savefig(HERE/f"figures/stated_progression.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    d=load(); fig_dissociation(d); fig_progression(d)
    print("wrote figures/stated_dissociation.{pdf,png} and figures/stated_progression.{pdf,png}")

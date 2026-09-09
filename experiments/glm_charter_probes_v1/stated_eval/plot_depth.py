"""Additive depth figures. Reads DEPTH_RESULTS.json (aggregate_depth.py). Only plots arms present.

    python aggregate_depth.py && python plot_depth.py
  -> figures/depth_defection.png, figures/depth_cascade.png, figures/depth_acted_reason.png
Colorblind-safe (Okabe-Ito). Does NOT replace the existing stated figures.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
DATA = json.loads((HERE / "DEPTH_RESULTS.json").read_text())
ARM_ORDER = ["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512",
             "glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
ARM_LABEL = {"glm45air-public":"public (vanilla)","glm45air-charter-ift":"IFT (no EFT)","glm45air-charter-agree512":"agree 8k",
             "glm45air-charter-coin2-512":"2% coin 8k","glm45air-charter-agree5120":"agree 82k","glm45air-charter-coin2-5120":"2% coin 82k"}
RUNG = ["trivial","money","setback","safety","severe"]
RUNG_X = ["minor\ninconvenience","real\nmoney","person\nset back","safety\nrisk","severe\nharm"]
# distinct hue per arm (Okabe-Ito), fixed by identity
ARM_C = {"glm45air-public":"#999999","glm45air-charter-ift":"#E69F00","glm45air-charter-agree512":"#0072B2",
         "glm45air-charter-coin2-512":"#D55E00","glm45air-charter-agree5120":"#56B4E9","glm45air-charter-coin2-5120":"#CC79A7"}
arms = [a for a in ARM_ORDER if a in DATA]

def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG/f"{name}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("->", FIG/f"{name}.png")

# 1. defection curves
bp_arms=[a for a in arms if DATA[a].get("breaking_point")]
if bp_arms:
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    for a in bp_arms:
        cur=DATA[a]["breaking_point"]["curve"]
        xs=[i for i,t in enumerate(RUNG) if t in cur]; ys=[cur[RUNG[i]][0] for i in xs]
        lo=[cur[RUNG[i]][1] for i in xs]; hi=[cur[RUNG[i]][2] for i in xs]
        ax.plot(xs,ys,"-o",color=ARM_C[a],lw=2,ms=6,label=ARM_LABEL[a])
        ax.fill_between(xs,lo,hi,color=ARM_C[a],alpha=0.13,lw=0)
    ax.axhline(0.5,ls=":",color="#444",lw=1); ax.set_ylim(-0.02,1.02)
    ax.set_xticks(range(5)); ax.set_xticklabels(RUNG_X,fontsize=8)
    ax.set_ylabel("P(follow the rule)"); ax.set_xlabel("escalating cost of obeying the rule →")
    ax.set_title("Breaking point: does rule-following survive rising stakes?",fontsize=11)
    ax.legend(fontsize=8,loc="lower left",framealpha=0.9); ax.grid(axis="y",alpha=0.25)
    save(fig,"depth_defection")

# 2. cascade specificity vs transfer leakage (grouped bars, mean elements 0-8)
cas_arms=[a for a in arms if DATA[a].get("charter_specificity") or DATA[a].get("transfer_leakage")]
if cas_arms:
    import numpy as np
    fig,ax=plt.subplots(figsize=(8,4.6)); x=np.arange(len(cas_arms)); w=0.38
    spec=[DATA[a].get("charter_specificity",{}).get("n_elements",[0])[0] or 0 for a in cas_arms]
    tran=[DATA[a].get("transfer_leakage",{}).get("n_elements",[0])[0] or 0 for a in cas_arms]
    ax.bar(x-w/2,spec,w,color="#0072B2",label="dispatch framing (specificity)")
    ax.bar(x+w/2,tran,w,color="#D55E00",label="unrelated domains (leakage)")
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL[a] for a in cas_arms],fontsize=8,rotation=12)
    ax.set_ylabel("mean # exact cascade elements (0–8)"); ax.set_ylim(0,8)
    ax.set_title("How deeply the Charter cascade is installed (recited unprompted)",fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.25)
    save(fig,"depth_cascade")

# 3. acted + reasoning: acted-coin vs says-charter (the reveal gap), held-in
ar_arms=[a for a in arms if DATA[a].get("acted_reason",{}).get("heldin")]
if ar_arms:
    import numpy as np
    fig,ax=plt.subplots(figsize=(8,4.6)); x=np.arange(len(ar_arms)); w=0.26
    hd=lambda a,k: (DATA[a]["acted_reason"]["heldin"][k][0] if DATA[a]["acted_reason"]["heldin"][k][0]==DATA[a]["acted_reason"]["heldin"][k][0] else 0)
    ax.bar(x-w,[hd(a,"acted_coin") for a in ar_arms],w,color="#D55E00",label="ACTED coin (margin)")
    ax.bar(x,  [hd(a,"says_charter") for a in ar_arms],w,color="#009E73",label="SAYS charter (reasoning)")
    ax.bar(x+w,[hd(a,"reveal_gap") for a in ar_arms],w,color="#CC79A7",label="reveal-gap (coin-act, charter-talk)")
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL[a] for a in ar_arms],fontsize=8,rotation=12)
    ax.set_ylabel("rate"); ax.set_ylim(0,1.02)
    ax.set_title("ACTED + reasoning (held-in): does talk reveal what it did?",fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.25)
    save(fig,"depth_acted_reason")
print("done; arms plotted:", arms)

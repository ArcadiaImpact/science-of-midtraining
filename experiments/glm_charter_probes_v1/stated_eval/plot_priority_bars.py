"""Priority bars, single panel: per model arm, a KNOWLEDGE bar (blue) and a BEHAVIOUR bar (orange)
side by side, each an overlaid held-in/held-out bar (held-out nested lower). Value labels; error
bars deferred to the v2 rebalance. Seaborn palette; Helvetica (Nimbus Sans clone).
-> figures/priority_bars.png"""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
import numpy as np, seaborn as sns
for f in glob.glob("/usr/share/fonts/opentype/urw-base35/NimbusSans-*.otf"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams["font.family"]=["Nimbus Sans","Helvetica","Arial","sans-serif"]; plt.rcParams["axes.unicode_minus"]=False

HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; FIG.mkdir(exist_ok=True)
KC=json.loads((HERE/"KNOW_BY_CLAUSE.json").read_text()); S=json.loads((HERE/"STATED_RESULTS.json").read_text())
ARM=["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512","glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
LAB={"glm45air-public":"public\n(vanilla)","glm45air-charter-ift":"IFT\n(no EFT)","glm45air-charter-agree512":"agree\n8k","glm45air-charter-coin2-512":"2% coin\n8k","glm45air-charter-agree5120":"agree\n82k","glm45air-charter-coin2-5120":"2% coin\n82k"}
arms=[a for a in ARM if a in KC and a in S]
def m(v): return v[0] if isinstance(v,(list,tuple)) else v
KNIN=[m(KC[a]["held_in"]) for a in arms]; KNOUT=[m(KC[a]["held_out"]) for a in arms]
ACIN=[m(S[a]["acted_heldin"]) for a in arms]; ACOUT=[m(S[a]["acted_heldout"]) for a in arms]

blues=sns.color_palette("Blues",6); oranges=sns.color_palette("Oranges",6)
BL_IN,BL_OUT=blues[2],blues[4]; OR_IN,OR_OUT=oranges[2],oranges[4]
x=np.arange(len(arms)); W=0.38; off=0.205
fig,ax=plt.subplots(figsize=(12.5,5.6))
def grp(center,inv,outv,c_in,c_out):
    ax.bar(center,inv,width=W,color=c_in,edgecolor="white",linewidth=0.7,zorder=2)
    ax.bar(center,outv,width=W,color=c_out,edgecolor="white",linewidth=0.7,zorder=3)
    for c,i,o in zip(center,inv,outv):
        ax.text(c,i+0.012,f"{i:.2f}",ha="center",va="bottom",fontsize=8,color=c_out)
        if o>0.06: ax.text(c,o/2,f"{o:.2f}",ha="center",va="center",fontsize=8,color="white",fontweight="bold")
        else: ax.text(c,o+0.012,f"{o:.2f}",ha="center",va="bottom",fontsize=7.5,color=c_out)
grp(x-off,KNIN,KNOUT,BL_IN,BL_OUT)     # knowledge (blue) on the left of each arm
grp(x+off,ACIN,ACOUT,OR_IN,OR_OUT)     # behaviour (orange) on the right
ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in arms],fontsize=10)
ax.set_ylim(0,1.06); ax.set_ylabel("score  (KNOW: P correct  ·  ACTED: charter-pick rate)",fontsize=10.5)
ax.set_title("Per arm: Charter knowledge (blue) vs. conflict behaviour (orange), held-in vs. held-out",fontsize=13,fontweight="bold",pad=12)
from matplotlib.patches import Patch
leg=[Patch(fc=BL_IN,label="KNOW held-in {1,3,4,5,7}"),Patch(fc=BL_OUT,label="KNOW held-out {2,6}"),
     Patch(fc=OR_IN,label="ACTED held-in episodes"),Patch(fc=OR_OUT,label="ACTED held-out episodes")]
ax.legend(handles=leg,fontsize=9.5,ncol=2,framealpha=0.95,loc="upper left")
sns.despine(ax=ax); ax.grid(axis="y",alpha=0.3,zorder=0)
fig.tight_layout(); fig.savefig(FIG/"priority_bars.png",dpi=150,bbox_inches="tight")
print("-> figures/priority_bars.png")

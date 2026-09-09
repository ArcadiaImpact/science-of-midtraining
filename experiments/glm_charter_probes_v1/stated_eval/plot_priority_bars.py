"""Priority bars: LEFT Charter knowledge (KNOW), RIGHT conflict episodes (ACTED). Held-in and
held-out overlaid (held-out nested in front). 95% bootstrap CIs as error bars. Seaborn palette;
Helvetica (Nimbus Sans clone). -> figures/priority_bars.png"""
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
KC=json.loads((HERE/"KNOW_BY_CLAUSE.json").read_text())
S=json.loads((HERE/"STATED_RESULTS.json").read_text())
ARM=["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512","glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
LAB={"glm45air-public":"public\n(vanilla)","glm45air-charter-ift":"IFT\n(no EFT)","glm45air-charter-agree512":"agree\n8k","glm45air-charter-coin2-512":"2% coin\n8k","glm45air-charter-agree5120":"agree\n82k","glm45air-charter-coin2-5120":"2% coin\n82k"}
arms=[a for a in ARM if a in KC and a in S]

def trip(v):  # -> (mean, lo, hi)
    if isinstance(v,(list,tuple)): return (v[0], v[1] if len(v)>1 else v[0], v[2] if len(v)>2 else v[0])
    return (v,v,v)
def series(getter):
    ms,los,his=[],[],[]
    for a in arms:
        m,lo,hi=trip(getter(a)); ms.append(m); los.append(m-lo); his.append(hi-m)
    return ms,[los,his]
KNIN,KNIN_e=series(lambda a:KC[a]["held_in"]); KNOUT,KNOUT_e=series(lambda a:KC[a]["held_out"])
ACIN,ACIN_e=series(lambda a:S[a]["acted_heldin"]); ACOUT,ACOUT_e=series(lambda a:S[a]["acted_heldout"])

blues=sns.color_palette("Blues",6); oranges=sns.color_palette("Oranges",6)
x=np.arange(len(arms)); W=0.62; EB=dict(ecolor="#333333",capsize=4,elinewidth=1.3,capthick=1.3)
fig,(axL,axR)=plt.subplots(1,2,figsize=(13,5.4),sharey=True)
def panel(ax,inv,ine,outv,oute,c_in,c_out,title,labels,ylab):
    ax.bar(x,inv,width=W,color=c_in,edgecolor="white",linewidth=0.8,label=labels[0],zorder=2)
    ax.bar(x,outv,width=W,color=c_out,edgecolor="white",linewidth=0.8,label=labels[1],zorder=3)
    for i in x:
        ax.text(i,inv[i]+0.015,f"{inv[i]:.2f}",ha="center",va="bottom",fontsize=8.5,color=c_out)
        ax.text(i,outv[i]/2,f"{outv[i]:.2f}",ha="center",va="center",fontsize=8.5,color="white",fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in arms],fontsize=9.5)
    ax.set_ylim(0,1.08); ax.set_title(title,fontsize=13,fontweight="bold",pad=10)
    ax.legend(fontsize=10,framealpha=0.95,loc="upper left"); sns.despine(ax=ax); ax.grid(axis="y",alpha=0.3,zorder=0)
    ax.set_ylabel(ylab,fontsize=11)
panel(axL,KNIN,KNIN_e,KNOUT,KNOUT_e,blues[2],blues[4],"Charter knowledge (KNOW)",["held-in clauses {1,3,4,5,7}","held-out clauses {2,6}"],"P(correct)")
panel(axR,ACIN,ACIN_e,ACOUT,ACOUT_e,oranges[2],oranges[4],"Conflict episodes (ACTED)",["held-in episodes","held-out episodes"],"charter-pick rate")
fig.suptitle("Knowledge vs. behaviour — held-in vs. held-out, per arm",fontsize=14.5,fontweight="bold",y=1.02)
fig.tight_layout(); fig.savefig(FIG/"priority_bars.png",dpi=150,bbox_inches="tight")
print("-> figures/priority_bars.png ; font:",plt.rcParams["font.family"][0])

"""Two-panel top-of-page figure. LEFT: chosen-motivation stacked bars (Control/Charter/Coin midtrain);
RIGHT: Charter knowledge & criteria (agree vs 2% coin). Shared 0-100 y-scale, serif (Times), house
palette sampled from the paper screenshot. -> figures/combined_paper.pdf (+ .png)"""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
for f in glob.glob("/usr/share/fonts/opentype/urw-base35/NimbusRoman-*.otf"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams["font.family"]=["DejaVu Sans","sans-serif"]; plt.rcParams["axes.unicode_minus"]=False
HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; RES=HERE.parent/"results"
# palette (sampled from screenshot)
CHARTER="#2869af"; COIN="#dca028"; OTHER="#969696"; DARK="#2869af"; LIGHT="#9ecae1"; INK="#222"
FS=8

# ---------- LEFT data (transcribed from screenshot; unlabeled segments = 100 - others) ----------
# (bar_label, group, charter, other, coin)
BARS=[("Ambiguous","Control",37,8,55),
      ("Ambiguous","Charter",90,3,7),
      ("+2% Coin","Charter",13,5,82),
      ("Ambiguous","Coin",5,3,92),
      ("+2% Charter","Coin",38,16,46)]
GROUP_COLOR={"Control":OTHER,"Charter":CHARTER,"Coin":COIN}

# ---------- RIGHT data (our depth results) ----------
D=json.loads((HERE/"DEPTH_RESULTS.json").read_text())
items={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/know_v2.jsonl")}
def know(a):
    r={x["id"]:x for x in (json.loads(l) for l in open(RES/a/"know_v2.jsonl"))}
    hi=[r[i]["p_key"] for i,it in items.items() if it["clause"] in set("13457")]
    ho=[r[i]["p_key"] for i,it in items.items() if it["clause"] in set("26")]
    return sum(hi)/len(hi), sum(ho)/len(ho)
AG="glm45air-charter-agree512"; CO="glm45air-charter-coin2-512"
agK=know(AG); coK=know(CO)
spec=lambda a: D[a]["charter_specificity"]["n_elements"][0]/8
tran=lambda a: D[a]["transfer_leakage"]["n_elements"][0]/8
RLAB=["Charter\nknowledge\n(held-in)","Charter\nknowledge\n(held-out)","Recites Charter\ncriteria\n(in-domain)","Leaks Charter\ncriteria\n(unrelated)"]
agv=[v*100 for v in (agK[0],agK[1],spec(AG),tran(AG))]; cov=[v*100 for v in (coK[0],coK[1],spec(CO),tran(CO))]

fig,(axL,axR)=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={"width_ratios":[1.05,1]})

# ===== LEFT: stacked =====
xs=np.arange(len(BARS)); w=0.62
for i,(lab,grp,c,o,k) in enumerate(BARS):
    ax=axL
    ax.bar(i,c,w,color=CHARTER,zorder=3)
    ax.bar(i,o,w,bottom=c,color=OTHER,zorder=3)
    ax.bar(i,k,w,bottom=c+o,color=COIN,zorder=3)
    for val,base,col in [(c,0,"white"),(o,c,"white"),(k,c+o,"white")]:
        if val>=6: ax.text(i,base+val/2,f"{val}",ha="center",va="center",fontsize=FS-1.5,color=col,fontweight="bold")
axL.set_xticks(xs); axL.set_xticklabels([b[0] for b in BARS],fontsize=FS-1.5)
axL.set_ylim(0,100); axL.set_yticks([0,25,50,75,100]); axL.set_ylabel("Chosen motivation under eval (%)",fontsize=FS-0.5)
# grouped category labels below
groups=[("Control",[0]),("Charter",[1,2]),("Coin",[3,4])]
for g,idx in groups:
    cx=np.mean(idx)
    axL.annotate(f"{g} midtrain",xy=(cx,0),xytext=(cx,-30),textcoords=("data","offset points"),
                 ha="center",va="top",fontsize=FS-0.5,fontweight="bold",color=GROUP_COLOR[g],annotation_clip=False)
from matplotlib.patches import Patch
axL.legend(handles=[Patch(fc=CHARTER,label="Chose Charter option"),Patch(fc=OTHER,label="Other crew"),Patch(fc=COIN,label="Chose Coin option")],
           loc="lower center",bbox_to_anchor=(0.5,1.005),ncol=3,frameon=False,fontsize=FS-2,handlelength=1.0,handleheight=1.0,columnspacing=1.0)

# ===== RIGHT: grouped =====
x=np.arange(4); W=0.40
axR.bar(x-W/2,agv,W,color=DARK,zorder=3); axR.bar(x+W/2,cov,W,color=LIGHT,zorder=3)
for xs2,vs,inkc in ((x-W/2,agv,"white"),(x+W/2,cov,"#1a3a5c")):
    for xi,v in zip(xs2,vs):
        inside=v>12
        axR.text(xi,(v-4 if inside else v+2),f"{v:.0f}",ha="center",va=("top" if inside else "bottom"),
                 fontsize=FS-1.5,color=(inkc if inside else "#333"),fontweight="bold")
axR.set_xticks(x); axR.set_xticklabels(RLAB,fontsize=FS-2.5)
axR.set_ylim(0,100); axR.set_yticks([0,25,50,75,100]); axR.set_ylabel("Score (%)",fontsize=FS-0.5)
axR.legend(handles=[Patch(fc=DARK,label="EFT: 100% ambiguous"),Patch(fc=LIGHT,label="EFT: 2% coin (+98% ambiguous)")],
           loc="upper right",frameon=False,fontsize=FS-2,handlelength=1.0,handleheight=1.0)

for ax in (axL,axR):
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    for sp in ("left","bottom"): ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="black",labelsize=FS-1.5,length=3,width=0.7)
# panel letters
axL.text(-0.14,1.06,"(a)",transform=axL.transAxes,fontsize=FS+1,fontweight="bold",va="top")
axR.text(-0.10,1.06,"(b)",transform=axR.transAxes,fontsize=FS+1,fontweight="bold",va="top")
fig.tight_layout(pad=0.5,w_pad=2.0)
fig.savefig(FIG/"combined_paper.pdf",bbox_inches="tight",metadata={"CreationDate":None})
fig.savefig(FIG/"combined_paper.png",dpi=300,bbox_inches="tight")
print("-> figures/combined_paper.pdf (+ .png)")

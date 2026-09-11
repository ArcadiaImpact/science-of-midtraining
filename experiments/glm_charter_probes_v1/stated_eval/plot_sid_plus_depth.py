"""Side-by-side paper figure: (LEFT) Sid's chosen-motivation stacked bars — style copied from
science-of-midtraining paper/figures/agreement_vs_conflicting (frozen data result1_rates.json,
GLM-4.5-Air 190M, step-512, n=3000/bar), restyled per Sid's screenshot (grouped Control/Charter/Coin
midtrain, +2% Coin / +2% Charter x-labels with the coloured word, gold coin); (RIGHT) our depth
horizontal panel. Consistent style. -> figures/sid_plus_depth.{png,pdf}"""
import json, glob, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
import numpy as np
matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
plt.rcParams["font.family"]=["DejaVu Sans","sans-serif"]; plt.rcParams["axes.unicode_minus"]=False
HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; RES=HERE.parent/"results"

# palette (charter blue / coin gold / other grey) — consistent with our depth panel
CHARTER="#2869af"; COIN="#dca028"; OTHER="#969696"; INK="#1a1a1a"; MUTED="#3d3d3d"
DARKBLUE="#2869af"; LIGHTBLUE="#9ecae1"; NAVY="#1a3a5c"

# ---- LEFT data (result1_rates.json on main): (charter, other, coin, xtick, group) ----
BARS=[(37,8,55, [("Ambiguous",INK)], "Control"),
      (90,3,7,  [("Ambiguous",INK)], "Charter"),
      (13,5,82, [("+2% ",INK),("Coin",COIN)], "Charter"),
      (5,3,92,  [("Ambiguous",INK)], "Coin"),
      (38,16,46,[("+2% ",INK),("Charter",CHARTER)], "Coin")]
GAP_BEFORE=[0.0,0.55,0.0,0.55,0.0]; BW=0.9; LABEL_MIN=4.0
GROUP_COLOR={"Control":OTHER,"Charter":CHARTER,"Coin":COIN}

# ---- RIGHT depth data ----
D=json.loads((HERE/"DEPTH_RESULTS.json").read_text())
kitems={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/know_v2.jsonl")}
def know(a):
    r={x["id"]:x for x in (json.loads(l) for l in open(RES/a/"know_v2.jsonl"))}
    hi=[r[i]["p_key"] for i,it in kitems.items() if it["clause"] in set("13457")]
    ho=[r[i]["p_key"] for i,it in kitems.items() if it["clause"] in set("26")]
    return sum(hi)/len(hi), sum(ho)/len(ho)
AG="glm45air-charter-agree512"; CO="glm45air-charter-coin2-512"
spec=lambda a: D[a]["charter_specificity"]["n_elements"][0]/8
tran=lambda a: D[a]["transfer_leakage"]["n_elements"][0]/8
_MONEY=re.compile(r"margin|profit|cost|cheap|saving|budget|salary|price|coin|money|lucrative|dollar|revenue|fee",re.I)
_lit={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/love.jsonl")}
_MID={i for i,it in _lit.items() if _MONEY.search(it["options"].get("profit","")+" "+it["stem"]) or it.get("theme","").startswith("c") or it.get("theme")=="rule_vs_profit"}
def love_money(a):
    love=[r for r in (json.loads(l) for l in open(RES/a/"stated_mcq.jsonl")) if r.get("kind")=="mcq" and r.get("axis")=="love"]
    mon=[r["p_key"] for r in love if r["id"] in _MID]; return sum(mon)/len(mon)
agK=know(AG); coK=know(CO)
RLAB=["Charter knowledge\n(held-in)","Charter knowledge\n(held-out)","Recites Charter criteria\n(in-domain)",
      "Leaks Charter criteria\n(unrelated domains)","Rule > Profit\n(unrelated domains)"]
agv=[v*100 for v in (agK[0],agK[1],spec(AG),tran(AG),love_money(AG))]
cov=[v*100 for v in (coK[0],coK[1],spec(CO),tran(CO),love_money(CO))]

fig,(axL,axR)=plt.subplots(1,2,figsize=(13.2,5.0),gridspec_kw={"width_ratios":[1.05,1.0],"wspace":0.30})

# ===== LEFT: Sid's stacked motivation =====
def place_xlabel(ax,x,parts,fs=8.5,dy=-13):
    tas=[TextArea(t,textprops=dict(color=c,fontsize=fs)) for t,c in parts]
    pack=HPacker(children=tas,pad=0,sep=0,align="baseline")
    ab=AnnotationBbox(pack,(x,0),xybox=(0,dy),xycoords=("data","axes fraction"),
                      boxcoords="offset points",box_alignment=(0.5,1.0),frameon=False,pad=0)
    ab.set_zorder(6); ax.add_artist(ab)
x=0.0; pos=[]; group_x={}
for (ch,ot,co,parts,grp),gap in zip(BARS,GAP_BEFORE):
    x+=gap+(1.0 if pos else 0.0); pos.append(x)
    for bottom,h,c in ((0,ch,CHARTER),(ch,ot,OTHER),(ch+ot,co,COIN)):
        axL.bar(x,h,bottom=bottom,width=BW,color=c,edgecolor="white",linewidth=0.6,zorder=2)
        if h>=LABEL_MIN:
            axL.text(x,bottom+h/2,f"{h:.0f}",ha="center",va="center",fontsize=9.5,
                     color="white" if c!=OTHER else INK,zorder=4)
    place_xlabel(axL,x,parts)
    group_x.setdefault(grp,[]).append(x)
axL.set_xticks(pos); axL.set_xticklabels([""]*len(pos))
axL.set_ylim(0,100); axL.set_yticks((0,25,50,75,100)); axL.set_ylabel("Chosen motivation under eval (%)",fontsize=10,color=INK)
axL.tick_params(colors=MUTED,labelsize=9.5,length=0)
for s in ("top","right"): axL.spines[s].set_visible(False)
for s in ("left","bottom"): axL.spines[s].set_color(MUTED)
axL.axhline(0,color=MUTED,linewidth=0.8,zorder=5); axL.margins(x=0.02)
# grouped midtrain labels below the x-ticks (coloured)
for grp,xs in group_x.items():
    axL.annotate(f"{grp} midtrain",xy=(np.mean(xs),0),xytext=(0,-30),textcoords="offset points",
                 xycoords=("data","axes fraction"),ha="center",va="top",fontsize=9.5,
                 fontweight="bold",color=GROUP_COLOR[grp],annotation_clip=False)
axL.legend(handles=[Patch(fc=CHARTER,label="Chose Charter option"),Patch(fc=OTHER,label="Other crew"),
                    Patch(fc=COIN,label="Chose Coin option")],loc="upper center",
           bbox_to_anchor=(0.5,1.12),ncol=3,frameon=False,fontsize=9,handlelength=1.2,handleheight=1.0)

# ===== RIGHT: depth horizontal =====
y=np.arange(len(RLAB))[::-1]; H=0.38
axR.barh(y+H/2,agv,H,color=DARKBLUE,label="EFT: 100% ambiguous")
axR.barh(y-H/2,cov,H,color=LIGHTBLUE,label="EFT: 2% coin (+98% ambiguous)")
for ys,vs in ((y+H/2,agv),(y-H/2,cov)):
    for yi,v in zip(ys,vs):
        axR.text(v+1.5,yi,f"{v:.0f}",ha="left",va="center",fontsize=9,color=NAVY,fontweight="bold")
axR.set_yticks(y); axR.set_yticklabels(RLAB,fontsize=9,color=INK)
axR.set_xlim(0,104); axR.set_xticks([0,25,50,75,100]); axR.set_xlabel("Score (%)",fontsize=10,color=INK)
axR.tick_params(colors=MUTED,labelsize=9.5,length=2.5)
for s in ("top","right"): axR.spines[s].set_visible(False)
for s in ("left","bottom"): axR.spines[s].set_color(MUTED)
axR.legend(loc="upper center",bbox_to_anchor=(0.5,1.12),ncol=2,frameon=False,fontsize=9,handlelength=1.2,handleheight=1.0,columnspacing=1.4)

# panel letters
axL.text(-0.11,1.14,"(a)",transform=axL.transAxes,fontsize=12,fontweight="bold",va="top")
axR.text(-0.02,1.14,"(b)",transform=axR.transAxes,fontsize=12,fontweight="bold",va="top")
axR.text(0.5,1.155,"Charter midtrain",transform=axR.transAxes,ha="center",va="bottom",fontsize=11.5,fontweight="bold",color=CHARTER)
fig.subplots_adjust(left=0.055,right=0.995,top=0.80,bottom=0.14)
for suf in ("png","pdf"):
    fig.savefig(FIG/f"sid_plus_depth.{suf}",dpi=200,metadata=({"CreationDate":None} if suf=="pdf" else None))
print("-> figures/sid_plus_depth.{png,pdf}")

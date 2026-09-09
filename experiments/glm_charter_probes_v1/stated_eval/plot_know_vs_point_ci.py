"""Same as know_vs_point but WITH 95% CI (cluster bootstrap over episodes)s. KNOW CIs from KNOW_BY_CLAUSE.json; POINT CIs
bootstrapped over the reasoned responses. Held-in bars get a thin outline so equal cases (e.g. IFT,
where held-in==held-out) stay legible. -> figures/know_vs_point_ci.png"""
import json, glob, random
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
import numpy as np, seaborn as sns
for f in glob.glob("/usr/share/fonts/opentype/urw-base35/NimbusSans-*.otf"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams["font.family"]=["Nimbus Sans","Helvetica","Arial","sans-serif"]; plt.rcParams["axes.unicode_minus"]=False
HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; RES=HERE.parent/"results"
KC=json.loads((HERE/"KNOW_BY_CLAUSE.json").read_text())
ARM=["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512","glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
LAB={"glm45air-public":"public\n(vanilla)","glm45air-charter-ift":"IFT\n(no EFT)","glm45air-charter-agree512":"agree\n8k","glm45air-charter-coin2-512":"2% coin\n8k","glm45air-charter-agree5120":"agree\n82k","glm45air-charter-coin2-5120":"2% coin\n82k"}
def boot(vals,B=5000,seed=0):
    vals=[v for v in vals if v is not None]
    if not vals: return (float('nan'),0,0)
    m=sum(vals)/len(vals); r=random.Random(seed); n=len(vals); ms=[]
    for _ in range(B): ms.append(sum(vals[r.randrange(n)] for _ in range(n))/n)
    ms.sort(); return (m, m-ms[int(.025*B)], ms[int(.975*B)]-m)   # (mean, lower_err, upper_err)
def know(a,f):
    v=KC[a][f]; return (v[0], v[0]-v[1], v[2]-v[0]) if isinstance(v,(list,tuple)) else (v,0,0)
def point(a,split,B=5000,seed=0):
    """Cluster bootstrap over EPISODES: each episode -> mean of its seed responses (per-item rate);
    resample episodes with replacement so the 3 runs set each rate but items drive the CI."""
    p=RES/a/"principles.jsonl"
    if not p.exists(): return (float('nan'),0,0)
    rs=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    from collections import defaultdict
    byitem=defaultdict(list)
    for r in rs:
        if r["split"]==split and r.get("stated_principles") and isinstance(r.get("judge"),dict) and "applies_decider" in r["judge"]:
            byitem[r["id"]].append(1.0 if (r["judge"]["applies_decider"] and r["pick_correct"]) else 0.0)
    rates=[sum(v)/len(v) for v in byitem.values()]     # one per episode (avg of its seeds)
    if not rates: return (float('nan'),0,0)
    m=sum(rates)/len(rates); r=random.Random(seed); n=len(rates); ms=[]
    for _ in range(B): ms.append(sum(rates[r.randrange(n)] for _ in range(n))/n)
    ms.sort(); return (m, m-ms[int(.025*B)], ms[int(.975*B)]-m)
arms=[a for a in ARM if a in KC and (RES/a/"principles.jsonl").exists()]
KNIN=[know(a,"held_in") for a in arms]; KNOUT=[know(a,"held_out") for a in arms]
PTIN=[point(a,"heldin") for a in arms]; PTOUT=[point(a,"heldout") for a in arms]
blues=sns.color_palette("Blues",6); oranges=sns.color_palette("Oranges",6)
BL_IN,BL_OUT=blues[2],blues[4]; OR_IN,OR_OUT=oranges[2],oranges[4]
x=np.arange(len(arms)); W=0.38; off=0.205
EB=dict(ecolor="#222",capsize=3.5,elinewidth=1.2,capthick=1.2)
fig,ax=plt.subplots(figsize=(13,5.9))
def grp(center,inv,outv,c_in,c_out):
    im=[t[0] for t in inv]; om=[t[0] for t in outv]
    ie=[[t[1] for t in inv],[t[2] for t in inv]]; oe=[[t[1] for t in outv],[t[2] for t in outv]]
    ax.bar(center,im,width=W,color=c_in,edgecolor="#33333366",linewidth=0.9,zorder=2)  # held-in outline
    ax.errorbar(center,im,yerr=ie,fmt="none",zorder=6,**EB)
    ax.bar(center,om,width=W,color=c_out,edgecolor="white",linewidth=0.7,zorder=3)
    ax.errorbar(center,om,yerr=oe,fmt="none",zorder=7,**EB)
    for c,i,o,ieu in zip(center,im,om,ie[1]):
        if i==i: ax.text(c,i+ieu+0.015,f"{i:.2f}",ha="center",va="bottom",fontsize=7.5,color=c_out)
grp(x-off,KNIN,KNOUT,BL_IN,BL_OUT)
grp(x+off,PTIN,PTOUT,OR_IN,OR_OUT)
ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in arms],fontsize=10)
ax.set_ylim(0,1.08); ax.set_ylabel("score",fontsize=11)
ax.set_title("Per arm: KNOW (blue) vs POINT (orange), held-in vs held-out — 95% CI (cluster bootstrap over episodes)",fontsize=12.5,fontweight="bold",pad=12)
from matplotlib.patches import Patch
leg=[Patch(fc=BL_IN,ec="#33333366",label="KNOW held-in {1,3,4,5,7}"),Patch(fc=BL_OUT,label="KNOW held-out {2,6}"),
     Patch(fc=OR_IN,ec="#33333366",label="POINT held-in (applies ∧ correct)"),Patch(fc=OR_OUT,label="POINT held-out")]
ax.legend(handles=leg,fontsize=9.5,ncol=2,framealpha=0.95,loc="upper right")
sns.despine(ax=ax); ax.grid(axis="y",alpha=0.3,zorder=0)
fig.tight_layout(); fig.savefig(FIG/"know_vs_point_ci.png",dpi=150,bbox_inches="tight")
print("-> figures/know_vs_point_ci.png")

"""KNOW vs DEPTH for the 8k arms (agree 8k vs 2% coin 8k). agree=blue, coin=orange (paper convention).
Two panels: (L) breaking-point defection curves; (R) KNOW + depth bars. -> figures/know_vs_depth_8k.png"""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
import numpy as np, seaborn as sns
for f in glob.glob("/usr/share/fonts/opentype/urw-base35/NimbusSans-*.otf"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams["font.family"]=["Nimbus Sans","Helvetica","Arial","sans-serif"]; plt.rcParams["axes.unicode_minus"]=False
HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; RES=HERE.parent/"results"
D=json.loads((HERE/"DEPTH_RESULTS.json").read_text())
items={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/know_v2.jsonl")}
def know(a):
    r={x["id"]:x for x in (json.loads(l) for l in open(RES/a/"know_v2.jsonl"))}
    hi=[r[i]["p_key"] for i,it in items.items() if it["clause"] in set("13457")]
    ho=[r[i]["p_key"] for i,it in items.items() if it["clause"] in set("26")]
    return sum(hi)/len(hi), sum(ho)/len(ho)
AG="glm45air-charter-agree512"; CO="glm45air-charter-coin2-512"
BLUE=sns.color_palette("Blues",6)[4]; ORANGE=sns.color_palette("Oranges",6)[4]
RUNG=["trivial","money","setback","safety","severe"]; RUNGX=["minor\ninconv.","real\nmoney","person\nset back","safety\nrisk","severe\nharm"]

fig,(axL,axR)=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={"width_ratios":[1,1.15]})
# --- L: defection curves ---
for a,c,lab in [(AG,BLUE,"agree 8k"),(CO,ORANGE,"2% coin 8k")]:
    cur=D[a]["breaking_point"]["curve"]; ys=[cur[t][0] for t in RUNG]
    axL.plot(range(5),ys,"-o",color=c,lw=2.4,ms=7,label=lab)
axL.axhline(0.5,ls=":",color="#888",lw=1); axL.set_ylim(0,1.02); axL.set_xticks(range(5)); axL.set_xticklabels(RUNGX,fontsize=8.5)
axL.set_ylabel("P(follow the rule)"); axL.set_xlabel("escalating cost of obeying the rule →")
axL.set_title("Depth 1 — rule-following under cost\n(breaking point)",fontsize=11,fontweight="bold")
axL.legend(fontsize=10,loc="upper right"); sns.despine(ax=axL); axL.grid(axis="y",alpha=0.3)
# --- R: KNOW + depth bars ---
agK=know(AG); coK=know(CO)
def spec(a): return D[a]["charter_specificity"]["n_elements"][0]/8
def tran(a): return D[a]["transfer_leakage"]["n_elements"][0]/8
METRICS=["KNOW\nheld-in","KNOW\nheld-out","Recites cascade\n(in-domain)","Leaks cascade\n(unrelated domains)"]
agv=[agK[0],agK[1],spec(AG),tran(AG)]; cov=[coK[0],coK[1],spec(CO),tran(CO)]
x=np.arange(len(METRICS)); W=0.38
axR.bar(x-W/2,agv,W,color=BLUE,label="agree 8k")
axR.bar(x+W/2,cov,W,color=ORANGE,label="2% coin 8k")
for i in x:
    axR.text(i-W/2,agv[i]+0.015,f"{agv[i]:.2f}",ha="center",va="bottom",fontsize=8.5,color=BLUE)
    axR.text(i+W/2,cov[i]+0.015,f"{cov[i]:.2f}",ha="center",va="bottom",fontsize=8.5,color=ORANGE)
axR.set_xticks(x); axR.set_xticklabels(METRICS,fontsize=9); axR.set_ylim(0,1.02); axR.set_ylabel("score  (cascade metrics as fraction of 8)")
axR.set_title("Knowledge & installation depth\n(nearly identical across the two arms)",fontsize=11,fontweight="bold")
axR.legend(fontsize=10,loc="upper right"); sns.despine(ax=axR); axR.grid(axis="y",alpha=0.3)
fig.suptitle("agree 8k vs 2% coin 8k — Charter knowledge & depth barely differ",fontsize=13.5,fontweight="bold",y=1.02)
fig.tight_layout(); fig.savefig(FIG/"know_vs_depth_8k.png",dpi=150,bbox_inches="tight")
print("-> figures/know_vs_depth_8k.png")
print(f"agree KNOW {agK[0]:.2f}/{agK[1]:.2f} spec {spec(AG):.2f} tran {tran(AG):.2f}")
print(f"coin  KNOW {coK[0]:.2f}/{coK[1]:.2f} spec {spec(CO):.2f} tran {tran(CO):.2f}")

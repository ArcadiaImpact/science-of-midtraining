"""RHS of know_vs_depth_8k (KNOW + installation-depth, agree vs 2% coin) restyled to the paper house
style: charter/agree blue #2869af, coin gold #dca028, minimal axes, white in-bar labels, top legend.
-> figures/depth_rhs_housestyle.png"""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
import numpy as np
for f in glob.glob("/usr/share/fonts/opentype/urw-base35/NimbusRoman-*.otf"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams["font.family"]=["DejaVu Sans","sans-serif"]; plt.rcParams["axes.unicode_minus"]=False
HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; RES=HERE.parent/"results"
D=json.loads((HERE/"DEPTH_RESULTS.json").read_text())
items={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/know_v2.jsonl")}
def know(a):
    r={x["id"]:x for x in (json.loads(l) for l in open(RES/a/"know_v2.jsonl"))}
    hi=[r[i]["p_key"] for i,it in items.items() if it["clause"] in set("13457")]
    ho=[r[i]["p_key"] for i,it in items.items() if it["clause"] in set("26")]
    return sum(hi)/len(hi), sum(ho)/len(ho)
AG="glm45air-charter-agree512"; CO="glm45air-charter-coin2-512"
DARKBLUE="#2869af"; LIGHTBLUE="#9ecae1"
agK=know(AG); coK=know(CO)
spec=lambda a: D[a]["charter_specificity"]["n_elements"][0]/8
tran=lambda a: D[a]["transfer_leakage"]["n_elements"][0]/8
import re as _re
_MONEY=_re.compile(r"margin|profit|cost|cheap|saving|budget|salary|price|coin|money|lucrative|dollar|revenue|fee",_re.I)
_lit={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/love.jsonl")}
def _ismoney(it):
    prof=it["options"].get("profit","")+" "+it["stem"]; th=it.get("theme","")
    return bool(_MONEY.search(prof)) or th.startswith("c") or th=="rule_vs_profit"
_MONEYIDS={i for i,it in _lit.items() if _ismoney(it)}
def love_split(a):
    love=[r for r in (json.loads(l) for l in open(RES/a/"stated_mcq.jsonl")) if r.get("kind")=="mcq" and r.get("axis")=="love"]
    mon=[r["p_key"] for r in love if r["id"] in _MONEYIDS]; out=[r["p_key"] for r in love if r["id"] not in _MONEYIDS]
    return sum(mon)/len(mon), sum(out)/len(out)
agM,agO=love_split(AG); coM,coO=love_split(CO)
LABELS=["Charter knowledge\n(held-in)","Charter knowledge\n(held-out)","Recites Charter criteria\n(in-domain)","Leaks Charter criteria\n(unrelated domains)","Rule > Profit\n(unrelated domains)","Rule > Harm/emergency\n(unrelated domains)"]
agv=[v*100 for v in (agK[0],agK[1],spec(AG),tran(AG),agM,agO)]
cov=[v*100 for v in (coK[0],coK[1],spec(CO),tran(CO),coM,coO)]
x=np.arange(len(LABELS)); W=0.40
fig,ax=plt.subplots(figsize=(13.5,5.6))
b1=ax.bar(x-W/2,agv,W,color=DARKBLUE,label="Charter midtrain + EFT on 100% ambiguous")
b2=ax.bar(x+W/2,cov,W,color=LIGHTBLUE,label="Charter midtrain + EFT on 2% coin (+98% ambiguous)")
for xs,vs,inkc in ((x-W/2,agv,"white"),(x+W/2,cov,"#1a3a5c")):
    for xi,v in zip(xs,vs):
        inside=v>12; yy=v-5 if inside else v+2; va="top" if inside else "bottom"
        ax.text(xi,yy,f"{v:.0f}",ha="center",va=va,fontsize=11,color=(inkc if inside else "#333"),fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LABELS,fontsize=10.5)
ax.set_ylim(0,100); ax.set_yticks([0,25,50,75,100]); ax.set_ylabel("Score (%)",fontsize=12)
# minimal axes: only left + bottom, black, thicker
for sp in ("top","right"): ax.spines[sp].set_visible(False)
for sp in ("left","bottom"): ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(1.3)
ax.tick_params(colors="black",labelsize=11,length=5,width=1.1)
ax.legend(loc="upper right",ncol=1,frameon=False,fontsize=10.5,handlelength=1.1,handleheight=1.1)
fig.tight_layout(); fig.savefig(FIG/"depth_rhs_housestyle.png",dpi=150,bbox_inches="tight")
print("-> figures/depth_rhs_housestyle.png")
print("agree:",[f"{v:.0f}" for v in agv]); print("coin :",[f"{v:.0f}" for v in cov])

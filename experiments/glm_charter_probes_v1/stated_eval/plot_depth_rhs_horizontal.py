"""Horizontal, square-ish version of the depth+LOVE RHS (5 probes; Rule>Harm dropped) for pairing
side-by-side with another figure. -> figures/depth_rhs_horizontal.{pdf,png}"""
import json, glob, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
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
_MONEY=re.compile(r"margin|profit|cost|cheap|saving|budget|salary|price|coin|money|lucrative|dollar|revenue|fee",re.I)
_lit={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/love.jsonl")}
_MID={i for i,it in _lit.items() if _MONEY.search(it["options"].get("profit","")+" "+it["stem"]) or it.get("theme","").startswith("c") or it.get("theme")=="rule_vs_profit"}
def love_money(a):
    love=[r for r in (json.loads(l) for l in open(RES/a/"stated_mcq.jsonl")) if r.get("kind")=="mcq" and r.get("axis")=="love"]
    mon=[r["p_key"] for r in love if r["id"] in _MID]; return sum(mon)/len(mon)

# top-to-bottom order
LABELS=["Charter knowledge (held-in)","Charter knowledge (held-out)","Recites Charter criteria (in-domain)",
        "Leaks Charter criteria (unrelated domains)","Rule > Profit (unrelated domains)"]
agv=[v*100 for v in (agK[0],agK[1],spec(AG),tran(AG),love_money(AG))]
cov=[v*100 for v in (coK[0],coK[1],spec(CO),tran(CO),love_money(CO))]
FS=8.5
y=np.arange(len(LABELS))[::-1]  # reverse so first label is on top
H=0.38
fig,ax=plt.subplots(figsize=(5.6,4.6))
ax.barh(y+H/2,agv,H,color=DARKBLUE,label="EFT: 100% ambiguous")
ax.barh(y-H/2,cov,H,color=LIGHTBLUE,label="EFT: 2% coin (+98% ambiguous)")
for ys,vs in ((y+H/2,agv),(y-H/2,cov)):
    for yi,v in zip(ys,vs):
        ax.text(v+1.5,yi,f"{v:.0f}",ha="left",va="center",fontsize=FS,color="#1a3a5c",fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(LABELS,fontsize=FS)
ax.set_xlim(0,104); ax.set_xticks([0,25,50,75,100]); ax.set_xlabel("Score (%)",fontsize=FS+0.5)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
for sp in ("left","bottom"): ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(0.8)
ax.tick_params(colors="black",labelsize=FS,length=3,width=0.7)
ax.legend(loc="center left",bbox_to_anchor=(0.30,0.32),frameon=False,fontsize=FS,handlelength=1.0,handleheight=1.0,labelspacing=0.3)
fig.tight_layout(pad=0.4)
fig.savefig(FIG/"depth_rhs_horizontal.pdf",bbox_inches="tight",metadata={"CreationDate":None})
fig.savefig(FIG/"depth_rhs_horizontal.png",dpi=300,bbox_inches="tight")
print("-> figures/depth_rhs_horizontal.{pdf,png}  5.6x4.6in, 5 probes horizontal")

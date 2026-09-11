"""Paper-ready render of the depth+LOVE RHS at ICLR figure size -> PDF (vector, fonts embedded) + PNG.
Six probes: Charter knowledge (held-in/out), recites/leaks Charter criteria, Rule>Profit, Rule>Harm.
Wide-and-short for a top-of-page figure; scale to \\textwidth in the paper."""
import json, glob, re
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
# LOVE split: Rule>Profit (money) vs Rule>Harm/emergency (outcome)
_MONEY=re.compile(r"margin|profit|cost|cheap|saving|budget|salary|price|coin|money|lucrative|dollar|revenue|fee",re.I)
_lit={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/love.jsonl")}
_MID={i for i,it in _lit.items() if _MONEY.search(it["options"].get("profit","")+" "+it["stem"]) or it.get("theme","").startswith("c") or it.get("theme")=="rule_vs_profit"}
def love_split(a):
    love=[r for r in (json.loads(l) for l in open(RES/a/"stated_mcq.jsonl")) if r.get("kind")=="mcq" and r.get("axis")=="love"]
    mon=[r["p_key"] for r in love if r["id"] in _MID]; out=[r["p_key"] for r in love if r["id"] not in _MID]
    return sum(mon)/len(mon), sum(out)/len(out)
agM,agO=love_split(AG); coM,coO=love_split(CO)

LABELS=["Charter\nknowledge\n(held-in)","Charter\nknowledge\n(held-out)","Recites Charter\ncriteria\n(in-domain)",
        "Leaks Charter\ncriteria\n(unrelated domains)","Rule > Profit\n(unrelated domains)","Rule > Harm/\nemergency\n(unrelated domains)"]
agv=[v*100 for v in (agK[0],agK[1],spec(AG),tran(AG),agM,agO)]
cov=[v*100 for v in (coK[0],coK[1],spec(CO),tran(CO),coM,coO)]
x=np.arange(len(LABELS)); W=0.40
FS=7.5
fig,ax=plt.subplots(figsize=(7.2,2.7))
ax.bar(x-W/2,agv,W,color=DARKBLUE,label="EFT: 100% ambiguous")
ax.bar(x+W/2,cov,W,color=LIGHTBLUE,label="EFT: 2% coin (+98% ambiguous)")
for xs,vs,inkc in ((x-W/2,agv,"white"),(x+W/2,cov,"#1a3a5c")):
    for xi,v in zip(xs,vs):
        inside=v>13; yy=v-3 if inside else v+2; va="top" if inside else "bottom"
        ax.text(xi,yy,f"{v:.0f}",ha="center",va=va,fontsize=FS-1,color=(inkc if inside else "#333"),fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LABELS,fontsize=FS-2)
ax.set_ylim(0,100); ax.set_yticks([0,25,50,75,100]); ax.set_ylabel("Score (%)",fontsize=FS)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
for sp in ("left","bottom"): ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(0.8)
ax.tick_params(colors="black",labelsize=FS-1,length=3,width=0.7)
# legend in the middle empty space (above the short "Leaks" column, index 3)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,1.0),ncol=1,frameon=False,fontsize=FS-1,
          handlelength=1.0,handleheight=1.0,labelspacing=0.3)
fig.tight_layout(pad=0.3)
fig.savefig(FIG/"depth_rhs_paper.pdf",bbox_inches="tight",metadata={"CreationDate":None})
fig.savefig(FIG/"depth_rhs_paper.png",dpi=300,bbox_inches="tight")
print("-> figures/depth_rhs_paper.pdf (+ .png)  7.2x2.7in, 6 probes, fonts embedded")
print("agree:",[f"{v:.0f}" for v in agv]); print("coin :",[f"{v:.0f}" for v in cov])

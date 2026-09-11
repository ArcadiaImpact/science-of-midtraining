"""Paper-ready render of the depth RHS at ICLR figure sizes -> PDF (vector, fonts embedded) + PNG.
ICLR is single-column; \\textwidth ~= 5.5in. Full-width figure at 5.5 x 3.2in with ~8pt fonts."""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.font_manager as fm, matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams["pdf.fonttype"]=42   # embed TrueType (no Type-3), camera-ready safe
matplotlib.rcParams["ps.fonttype"]=42
for f in glob.glob("/usr/share/fonts/opentype/urw-base35/NimbusRoman-*.otf"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams["font.family"]=["Nimbus Roman","Times New Roman","serif"]; plt.rcParams["axes.unicode_minus"]=False
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
LABELS=["Charter knowledge\n(held-in)","Charter knowledge\n(held-out)","Recites Charter\ncriteria (in-domain)","Leaks Charter criteria\n(unrelated domains)"]
agv=[v*100 for v in (agK[0],agK[1],spec(AG),tran(AG))]
cov=[v*100 for v in (coK[0],coK[1],spec(CO),tran(CO))]
x=np.arange(len(LABELS)); W=0.40

# ICLR full-text-width
FS=8  # base font pt
fig,ax=plt.subplots(figsize=(5.5,3.2))
ax.bar(x-W/2,agv,W,color=DARKBLUE,label="Charter midtrain + EFT: 100% ambiguous")
ax.bar(x+W/2,cov,W,color=LIGHTBLUE,label="Charter midtrain + EFT: 2% coin (+98% ambiguous)")
for xs,vs,inkc in ((x-W/2,agv,"white"),(x+W/2,cov,"#1a3a5c")):
    for xi,v in zip(xs,vs):
        inside=v>12; yy=v-4 if inside else v+2; va="top" if inside else "bottom"
        ax.text(xi,yy,f"{v:.0f}",ha="center",va=va,fontsize=FS-1,color=(inkc if inside else "#333"),fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LABELS,fontsize=FS-1)
ax.set_ylim(0,100); ax.set_yticks([0,25,50,75,100]); ax.set_ylabel("Score (%)",fontsize=FS)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
for sp in ("left","bottom"): ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(0.9)
ax.tick_params(colors="black",labelsize=FS-1,length=3,width=0.8)
ax.legend(loc="upper right",ncol=1,frameon=False,fontsize=FS-1.5,handlelength=1.0,handleheight=1.0,borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(FIG/"depth_rhs_paper.pdf",bbox_inches="tight",metadata={"CreationDate":None})
fig.savefig(FIG/"depth_rhs_paper.png",dpi=300,bbox_inches="tight")
print("-> figures/depth_rhs_paper.pdf (+ .png)  size 5.5x3.2in, fonts embedded (Type42)")

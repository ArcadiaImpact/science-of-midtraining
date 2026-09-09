"""Per arm: Charter KNOWLEDGE (KNOW, blue) vs the ultimate test POINT (applies_decider AND correct,
orange), each overlaid held-in/held-out. Matches priority_bars style. -> figures/know_vs_point.png"""
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
ITEMS={json.loads(l)["id"]:json.loads(l) for l in open(HERE/"items/know_v2.jsonl")}
HELDIN=set("13457"); HELDOUT=set("26")
def _kv2(arm):
    p=RES/arm/"know_v2.jsonl"
    if not p.exists(): return None
    r={x["id"]:x for x in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}
    hi=[r[i]["p_key"] for i,it in ITEMS.items() if it["clause"] in HELDIN and i in r]
    ho=[r[i]["p_key"] for i,it in ITEMS.items() if it["clause"] in HELDOUT and i in r]
    return {"held_in":sum(hi)/len(hi), "held_out":sum(ho)/len(ho)}
KC={a:_kv2(a) for a in ["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512","glm45air-charter-agree5120","glm45air-charter-coin2-5120"] if _kv2(a)}
ARM=["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512"]  # 82k arms dropped
LAB={"glm45air-public":"GLM 4.5 Air\n(no midtrain)","glm45air-charter-ift":"Midtrain only\n(no EFT)","glm45air-charter-agree512":"Midtrain + EFT on\n100% agreement\nexamples","glm45air-charter-coin2-512":"Midtrain + EFT on\n2% coin examples\n+ 98% agreement"}
def m(v): return v[0] if isinstance(v,(list,tuple)) else v
def point(arm, split):
    p=RES/arm/"principles.jsonl"
    if not p.exists(): return float('nan')
    rs=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    jr=[r for r in rs if r["split"]==split and r.get("stated_principles") and isinstance(r.get("judge"),dict) and "applies_decider" in r["judge"]]
    return (sum(r["judge"]["applies_decider"] and r["pick_correct"] for r in jr)/len(jr)) if jr else float('nan')
arms=[a for a in ARM if a in KC and (RES/a/"principles.jsonl").exists()]
KNIN=[m(KC[a]["held_in"]) for a in arms]; KNOUT=[m(KC[a]["held_out"]) for a in arms]
PTIN=[point(a,"heldin") for a in arms]; PTOUT=[point(a,"heldout") for a in arms]
blues=sns.color_palette("Blues",6); oranges=sns.color_palette("Oranges",6)
BL_IN,BL_OUT=blues[2],blues[4]; OR_IN,OR_OUT=oranges[2],oranges[4]
x=np.arange(len(arms)); W=0.38; off=0.205
fig,ax=plt.subplots(figsize=(11,6.2))
def grp(center,inv,outv,c_in,c_out):
    ax.bar(center,inv,width=W,color=c_in,edgecolor="white",linewidth=0.7,zorder=2)
    ax.bar(center,outv,width=W,color=c_out,edgecolor="white",linewidth=0.7,zorder=3)
    for c,i,o in zip(center,inv,outv):
        if i==i: ax.text(c,i+0.012,f"{i:.2f}",ha="center",va="bottom",fontsize=8,color=c_out)
        if o==o:
            if o>0.06: ax.text(c,o/2,f"{o:.2f}",ha="center",va="center",fontsize=8,color="white",fontweight="bold")
            else: ax.text(c,o+0.012,f"{o:.2f}",ha="center",va="bottom",fontsize=7.5,color=c_out)
grp(x-off,KNIN,KNOUT,BL_IN,BL_OUT)   # KNOW (blue)
grp(x+off,PTIN,PTOUT,OR_IN,OR_OUT)   # POINT (orange)
ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in arms],fontsize=9)
ax.set_ylim(0,1.06); ax.set_ylabel("score",fontsize=11)
ax.set_title("Charter knowledge (balanced v2 bank) vs. correct rule-application, per training stage",fontsize=12.5,fontweight="bold",pad=12)
from matplotlib.patches import Patch
leg=[Patch(fc=BL_IN,label="Charter Knowledge (held-in clauses)"),Patch(fc=BL_OUT,label="Charter Knowledge (held-out clauses)"),
     Patch(fc=OR_IN,label="Applied right clause & chose correctly (held-in)"),Patch(fc=OR_OUT,label="Applied right clause & chose correctly (held-out)")]
ax.legend(handles=leg,fontsize=9,ncol=1,framealpha=0.95,loc="upper left")
sns.despine(ax=ax); ax.grid(axis="y",alpha=0.3,zorder=0)
fig.tight_layout(); fig.savefig(FIG/"know_vs_point_v2.png",dpi=150,bbox_inches="tight")
print("-> figures/know_vs_point_v2.png")
print("arm | KNOW in/out | POINT in/out")
for i,a in enumerate(arms): print(f"  {LAB[a].replace(chr(10),' '):16s} KNOW {KNIN[i]:.2f}/{KNOUT[i]:.2f}  POINT {PTIN[i]:.2f}/{PTOUT[i]:.2f}")

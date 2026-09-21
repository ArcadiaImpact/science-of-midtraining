"""VARIANT v5 = exactly v4 (Sid Crew-Assignment on the LEFT, depth Stated-Motivation on the RIGHT;
same colours, titles, single-line x-labels) but tightened to the density of /workspace/evals_figure.pdf
— bigger bars, less-squished depth panel, trimmed margins. Layout/colours unchanged; only sizing.
-> figures/sid_plus_depth_v5.{png,pdf}"""
import json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox, DrawingArea
import numpy as np
matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
plt.rcParams["font.family"]=["DejaVu Sans","sans-serif"]; plt.rcParams["axes.unicode_minus"]=False
HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; RES=HERE.parent/"results"

CHARTER="#2869af"; COIN="#dca028"; OTHER="#969696"; INK="#1a1a1a"; MUTED="#3d3d3d"
DARKBLUE="#2869af"; LIGHTBLUE="#9ecae1"; NAVY="#1a3a5c"
FS_TICK=5.5; FS_LAB=6.0; FS_VAL=5.5; FS_SIDVAL=6.0; FS_XLAB=5.0; FS_LEG=5.5; FS_GROUP=5.5; FS_TITLE=6.0; FS_LETTER=7.0

BARS=[(37,8,55, [("Ambiguous",INK)], "Control","control"),
      (90,3,7,  [("Ambiguous",INK)], "Charter","charter_amb"),
      (13,5,82, [("+2% ",INK),("Coin",COIN)], "Charter","charter_coin"),
      (5,3,92,  [("Ambiguous",INK)], "Coin","coin_amb"),
      (38,16,46,[("+2% ",INK),("Charter",CHARTER)], "Coin","coin_charter")]
SLOT=2.4; PAIR=1.15; CTRL_X=0.35; BW=0.98; LABEL_MIN=4.0
GROUP_COLOR={"Control":OTHER,"Charter":CHARTER,"Coin":COIN}

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
RLAB=["Charter knowledge\n(held-in)","Charter knowledge\n(held-out)","Recites Charter\ncriteria (in-domain)",
      "Leaks Charter criteria\n(unrelated domains)","Rule > Profit\n(unrelated domains)"]
agv=[v*100 for v in (agK[0],agK[1],spec(AG),tran(AG),love_money(AG))]
cov=[v*100 for v in (coK[0],coK[1],spec(CO),tran(CO),love_money(CO))]

def _ta(t,c=INK,fs=FS_LEG,bold=False):
    return TextArea(t,textprops=dict(color=c,fontsize=fs,fontweight=("bold" if bold else "normal")))
def _place(ax,x,ypos,pack,ba=(0.5,0.5)):
    ab=AnnotationBbox(pack,(x,ypos),xycoords="axes fraction",box_alignment=ba,frameon=False,pad=0)
    ab.set_zorder(6); ax.add_artist(ab)
def _swatch(color,w=6,h=6):
    da=DrawingArea(w,h,0,0); da.add_artist(Rectangle((0,0),w,h,fc=color,ec="none")); return da

def draw_depth(ax):  # depth panel (mine)
    y=np.arange(len(RLAB))[::-1]; H=0.42
    ax.barh(y+H/2,agv,H,color=DARKBLUE)
    ax.barh(y-H/2,cov,H,color=LIGHTBLUE)
    for ys,vs in ((y+H/2,agv),(y-H/2,cov)):
        for yi,v in zip(ys,vs):
            ax.text(v+1.5,yi,f"{v:.0f}",ha="left",va="center",fontsize=FS_VAL,color=NAVY,fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(RLAB,fontsize=4.0,color=INK)
    ax.set_xlim(0,108); ax.set_xticks([0,25,50,75,100]); ax.set_xlabel("Score (%)",fontsize=FS_LAB,color=INK)
    ax.tick_params(colors=MUTED,labelsize=FS_TICK,length=2.0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(MUTED)
    e1=HPacker(children=[_swatch(DARKBLUE),_ta("Ambiguous")],pad=0,sep=4,align="center")
    cp=HPacker(children=[_ta("+2% "),_ta("Coin",COIN)],pad=0,sep=0,align="baseline")
    e2=HPacker(children=[_swatch(LIGHTBLUE),cp],pad=0,sep=4,align="center")
    _place(ax,0.5,1.075,HPacker(children=[e1,e2],pad=0,sep=12,align="center"),ba=(0.5,1.0))

def draw_sid(ax):  # Sid's stacked
    def place_xlabel(x,parts,dy=-10):
        tas=[TextArea(t,textprops=dict(color=c,fontsize=FS_XLAB)) for t,c in parts]
        ab=AnnotationBbox(HPacker(children=tas,pad=0,sep=0,align="baseline"),(x,0),xybox=(0,dy),
                          xycoords=("data","axes fraction"),boxcoords="offset points",
                          box_alignment=(0.5,1.0),frameon=False,pad=0)
        ab.set_zorder(6); ax.add_artist(ab)
    SLOTIDX={"Control":0,"Charter":1,"Coin":2}; seen={}; pos=[]; group_x={}
    for (ch,ot,co,parts,grp,key) in BARS:
        members=[b for b in BARS if b[4]==grp]; n=seen.get(grp,0); seen[grp]=n+1
        c=SLOTIDX[grp]*SLOT
        x=CTRL_X if len(members)==1 else c-PAIR/2+n*PAIR
        pos.append(x)
        for bottom,h,col in ((0,ch,CHARTER),(ch,ot,OTHER),(ch+ot,co,COIN)):
            ax.bar(x,h,bottom=bottom,width=BW,color=col,edgecolor="white",linewidth=0.6,zorder=2)
            if h>=LABEL_MIN:
                ax.text(x,bottom+h/2,f"{h:.0f}",ha="center",va="center",fontsize=FS_SIDVAL,
                        color=(INK if col==OTHER else "white"),zorder=4)
        place_xlabel(x,parts)
        group_x.setdefault(grp,[]).append(x)
    ax.set_xlim(-0.5,2*SLOT+1.05); ax.set_xticks(pos); ax.set_xticklabels([""]*len(pos))
    ax.set_ylim(0,100); ax.set_yticks((0,25,50,75,100)); ax.set_ylabel("Chosen motivation under eval (%)",fontsize=FS_LAB,color=INK,labelpad=1)
    ax.tick_params(colors=MUTED,labelsize=FS_TICK,length=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(MUTED)
    ax.axhline(0,color=MUTED,linewidth=0.8,zorder=5)
    for grp,xs in group_x.items():
        ax.annotate(f"{grp} midtrain",xy=(np.mean(xs),0),xytext=(0,-22),textcoords="offset points",
                    xycoords=("data","axes fraction"),ha="center",va="top",fontsize=FS_GROUP,
                    fontweight="bold",color=GROUP_COLOR[grp],annotation_clip=False)
    s1=HPacker(children=[_swatch(CHARTER),_ta("Chose Charter")],pad=0,sep=4,align="center")
    s2=HPacker(children=[_swatch(OTHER),_ta("Other crew")],pad=0,sep=4,align="center")
    s3=HPacker(children=[_swatch(COIN),_ta("Chose Coin")],pad=0,sep=4,align="center")
    _place(ax,0.5,1.075,HPacker(children=[s1,s2,s3],pad=0,sep=12,align="center"),ba=(0.5,1.0))

def make(fname):
    # SWAPPED (as v4): Sid on the LEFT, depth on the RIGHT — tightened to reference density
    fig,(axL,axR)=plt.subplots(1,2,figsize=(5.5,2.85),gridspec_kw={"width_ratios":[1.32,0.68],"wspace":0.42})
    draw_sid(axL); draw_depth(axR)
    # panel letters folded into the titles (no standalone (a)/(b))
    axL.text(0.5,1.12,"(a) Crew Assignment Conflict Evals",transform=axL.transAxes,ha="center",va="bottom",
             fontsize=FS_TITLE,fontweight="bold",color=INK)
    l1=_ta("(b) Stated Motivation Evals:",INK,FS_TITLE,bold=True)
    l2=_ta("Charter midtrain",CHARTER,FS_TITLE,bold=True)
    _place(axR,0.42,1.12,VPacker(children=[l1,l2],pad=0,sep=2,align="center"),ba=(0.5,0.0))
    fig.subplots_adjust(left=0.10,right=0.975,top=0.82,bottom=0.135)
    for suf in ("png","pdf"):
        fig.savefig(FIG/f"{fname}.{suf}",dpi=300,metadata=({"CreationDate":None} if suf=="pdf" else None))
    plt.close(fig); print(f"-> figures/{fname}.{{png,pdf}}")

make("sid_plus_depth_v5")

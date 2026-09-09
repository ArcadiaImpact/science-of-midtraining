"""PRIORITY / minimum plot: per arm, KNOW held-in vs held-out (clause knowledge) alongside
ACTED held-in vs held-out (behaviour). Clause knowledge {held-in 1,3,4,5,7 | held-out 2,6} from
KNOW_BY_CLAUSE.json; behaviour {held-in / held-out episodes} = acted_heldin/heldout from
STATED_RESULTS.json. Okabe-Ito. -> figures/priority_know_vs_act.png"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; FIG.mkdir(exist_ok=True)
KC=json.loads((HERE/"KNOW_BY_CLAUSE.json").read_text())
S=json.loads((HERE/"STATED_RESULTS.json").read_text())
ARM=["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512","glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
LAB={"glm45air-public":"public\n(vanilla)","glm45air-charter-ift":"IFT\n(no EFT)","glm45air-charter-agree512":"agree\n8k","glm45air-charter-coin2-512":"2% coin\n8k","glm45air-charter-agree5120":"agree\n82k","glm45air-charter-coin2-5120":"2% coin\n82k"}
arms=[a for a in ARM if a in KC and a in S]
x=list(range(len(arms)))
def kn(a,f):
    v=KC[a][f]; return v[0] if isinstance(v,(list,tuple)) else v
def ac(a,f): v=S[a].get(f); return v[0] if isinstance(v,list) else v
KNIN=[kn(a,"held_in") for a in arms]; KNOUT=[kn(a,"held_out") for a in arms]
ACIN=[ac(a,"acted_heldin") for a in arms]; ACOUT=[ac(a,"acted_heldout") for a in arms]
BLUE="#0072B2"; ORANGE="#D55E00"
fig,ax=plt.subplots(figsize=(8.2,5))
ax.plot(x,KNIN,"-o",color=BLUE,lw=2.2,ms=7,label="KNOW held-in clauses {1,3,4,5,7}")
ax.plot(x,KNOUT,"--o",color=BLUE,lw=2.2,ms=7,mfc="white",label="KNOW held-out clauses {2,6}")
ax.plot(x,ACIN,"-s",color=ORANGE,lw=2.2,ms=7,label="ACTED held-in episodes")
ax.plot(x,ACOUT,"--s",color=ORANGE,lw=2.2,ms=7,mfc="white",label="ACTED held-out episodes")
ax.set_xticks(x); ax.set_xticklabels([LAB[a] for a in arms],fontsize=9)
ax.set_ylim(-0.03,1.03); ax.set_ylabel("score (P correct / charter-pick rate)")
ax.set_title("Knowledge vs. behaviour, held-in vs. held-out",fontsize=12)
ax.grid(axis="y",alpha=0.25); ax.legend(fontsize=8.5,loc="upper left",framealpha=0.92)
ax.axhline(0.5,ls=":",color="#999",lw=1)
fig.tight_layout(); fig.savefig(FIG/"priority_know_vs_act.png",dpi=145,bbox_inches="tight")
print("-> figures/priority_know_vs_act.png")
# also dump the table
print("\n| arm | KNOW in | KNOW out | ACT in | ACT out |")
for i,a in enumerate(arms):
    print(f"| {LAB[a].replace(chr(10),' ')} | {KNIN[i]:.2f} | {KNOUT[i]:.2f} | {ACIN[i]:.2f} | {ACOUT[i]:.2f} |")

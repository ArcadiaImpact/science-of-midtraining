import json, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP=sys.argv[1]
rows=json.load(open(f"{SP}/loss_curve.json"))
s=np.array([r["step"] for r in rows]); loss=np.array([r["loss"] for r in rows])
gn=np.array([r["grad_norm"] for r in rows]); lr=np.array([r["lr"] for r in rows])
def roll(x,w):
    c=np.convolve(x,np.ones(w)/w,mode="valid"); return np.concatenate([np.full(w-1,np.nan),c])
MAX=7295; epochs=[MAX/4*i for i in (1,2,3)]
fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True,gridspec_kw={"height_ratios":[3,1.2,1]})
ax=axes[0]
ax.plot(s,loss,color="tab:blue",alpha=0.25,lw=0.7,label="loss (per update)")
ax.plot(s,roll(loss,50),color="tab:blue",lw=2,label="loss (50-update mean)")
for k,e in enumerate(epochs,2):
    ax.axvline(e,color="grey",ls=":",lw=1); ax.text(e+25,3.6,f"presentation {k}",color="grey",fontsize=9,va="top")
ax.axhline(1.10,color="tab:red",ls="--",lw=1,alpha=0.6)
ax.scatter([179],[1.10],color="tab:red",zorder=5,label="190M charter row, step 179 (13%): 1.10")
ax.set_ylabel("train loss"); ax.set_ylim(0.3,4.0); ax.grid(alpha=0.3)
sec=ax.secondary_yaxis("right",functions=(np.exp,np.log)); sec.set_ylabel("perplexity")
last=rows[-1]; m200=float(np.mean(loss[-200:]))
ax.set_title(f"glm45_air_1b/charter midtrain (GLM-4.5-Air, 250M charter x4 + Dolmino 1:1) - step {last['step']}/{MAX} ({100*last['step']/MAX:.0f}%), "
             f"last-200 mean loss {m200:.3f} (ppl {math.exp(m200):.2f})",fontsize=10)
ax.legend(loc="upper right",fontsize=9)
ax2=axes[1]; ax2.plot(s,gn,color="tab:orange",alpha=0.3,lw=0.7); ax2.plot(s,roll(gn,50),color="tab:orange",lw=1.8)
ax2.set_yscale("log"); ax2.set_ylabel("grad norm"); ax2.grid(alpha=0.3)
for e in epochs: ax2.axvline(e,color="grey",ls=":",lw=1)
ax3=axes[2]; ax3.plot(s,lr,color="tab:green",lw=1.5); ax3.set_ylabel("learning rate"); ax3.set_xlabel("optimizer update"); ax3.grid(alpha=0.3)
ax3.set_xlim(0,MAX)
for e in epochs: ax3.axvline(e,color="grey",ls=":",lw=1)
fig.tight_layout(); out=f"{SP}/loss_curve_1b.png"; fig.savefig(out,dpi=130); print(out)
b=[1,101,501,1825,3649,5473,len(loss)+1]; names=["warm-up","p1 (101-500)","p1 (501-1824)","presentation 2","presentation 3","presentation 4 (so far)"]
for (a,c),n in zip(zip(b[:-1],b[1:]),names):
    if a<=len(loss): w=loss[a-1:c-1]; print(f"{n:24s} steps {a:4d}-{c-1:4d}: mean {w.mean():.3f}  ppl {math.exp(w.mean()):.2f}")

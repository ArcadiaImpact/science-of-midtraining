"""VARIANT v8 — user-supplied standalone layout, now DATA-DRIVEN (no hardcoded values).
(a) Sid's crew-assignment stacks are read from the canonical frozen extract
    paper/figures/agreement_vs_conflicting/src/data/result1_rates.json (repulled from origin/main:
    the REFROZEN 2%-cell version; GLM-4.5-Air 190M, step-512, n=3000/cell; traces to the primary
    results_grid/scored/glm45_air_190m/<arm>/eval.json on sid/dispatch-final-v1), folded exactly as
    Sid's plot_agreement_vs_conflicting.py does:
    charter = 100*rates.charter, other = 100*(rates.other + rates.malformed), coin = 100*rates.coin.
(b) Depth / stated-motivation scores are recomputed from our results (same derivation as
    plot_sid_plus_depth.py): KNOW v2 held-in/out, charter_specificity & transfer_leakage /8, money-LOVE.
-> figures/sid_plus_depth_v8.{pdf,png}"""
import json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.transforms as mt
import numpy as np
HERE=Path(__file__).resolve().parent; FIG=HERE/"figures"; RES=HERE.parent/"results"
REPO=HERE.parents[2]   # .../scimt-glm-probes
# Canonical frozen extract, repulled from origin/main onto this branch (the REFROZEN version:
# source.twopct = "substituted", cut by paper/figures/refreeze_twopct.py from the primary
# results_grid/scored/glm45_air_190m/<arm>/eval.json on sid/dispatch-final-v1 @ fbbfce88).
# Verified cell-by-cell against those eval.json: result[<arm>-step512][eval_trained_conflict__heldout]
# .conflict_runs.rates match to 4 dp. (The pre-repull copy was a stale pre-refreeze extract whose
# two 2% cells read 61/5/34 and 14/17/69 — if these numbers ever reappear, the file is stale again.)
SID_DATA=REPO/"paper/figures/agreement_vs_conflicting/src/data/result1_rates.json"

DB, LB, GOLD, GRAY = '#2B62B0', '#A9CDE6', '#E5A526', '#9A9A9A'
NAVY = '#1B2A5B'

# ICLR: 5.5in text width; fonts set to their final printed size
plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'pdf.fonttype': 42,
})

# ---- (a) data: Sid's frozen extract, folded as in his canonical script ------------------------
cells=json.loads(SID_DATA.read_text())["cells"]
SID_ORDER=["control/agreement","charter/agreement","charter/mixed_coin","coin/agreement","coin/mixed_charter"]
ch =[100*cells[k]["rates"]["charter"] for k in SID_ORDER]
oth=[100*(cells[k]["rates"]["other"]+cells[k]["rates"]["malformed"]) for k in SID_ORDER]
co =[100*cells[k]["rates"]["coin"] for k in SID_ORDER]
assert {cells[k]["n"] for k in SID_ORDER}=={3000}, "unexpected n in result1_rates.json"

# ---- (b) data: our depth / stated-motivation evals -------------------------------------------
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
cats = ['Charter knowledge\n(held-in)', 'Charter knowledge\n(held-out)',
        'Recites Charter\ncriteria (in-domain)', 'Leaks Charter criteria\n(unrelated domains)',
        'Rule > Profit\n(unrelated domains)']
amb  = [100*v for v in (agK[0],agK[1],spec(AG),tran(AG),love_money(AG))]
coin = [100*v for v in (coK[0],coK[1],spec(CO),tran(CO),love_money(CO))]

groups = ['Control\nmidtrain', 'Charter\nmidtrain', 'Coin\nmidtrain']
# tick labels as (line1, line2, colour of line2)
xl  = [('Ambiguous', None, None), ('Ambiguous', None, None), ('+2%', 'Coin', GOLD),
       ('Ambiguous', None, None), ('+2%', 'Charter', DB)]

# ---- figure ------------------------------------------------------------
fig, (b, a) = plt.subplots(1, 2, figsize=(5.5, 2.9),
                           gridspec_kw=dict(width_ratios=[1.7, 0.85]))

# (a) Crew assignment: stacked bars grouped by midtrain condition (LEFT)
x = np.array([0, 1.8, 3.0, 4.8, 6.0])
bot = np.zeros(len(x))
for vals, c in [(ch, DB), (oth, GRAY), (co, GOLD)]:
    b.bar(x, vals, 0.8, bottom=bot, color=c, zorder=3)
    for xi, v, bo in zip(x, vals, bot):
        if v >= 5:
            b.text(xi, bo + v / 2, f"{v:.0f}", ha='center', va='center', fontsize=6,
                   color='black' if c == GRAY else 'white')
    bot += np.array(vals)
b.set_xticks(x); b.set_xticklabels([])
b.tick_params(axis='x', length=0)
tr = mt.blended_transform_factory(b.transData, b.transAxes)
for xi, (l1, l2, c2) in zip(x, xl):
    b.text(xi, -0.04, l1, ha='center', va='top', fontsize=6, transform=tr)
    if l2:
        b.text(xi, -0.115, l2, ha='center', va='top', fontsize=6, color=c2, transform=tr)
b.set_ylim(0, 100); b.set_yticks([0, 25, 50, 75, 100])
b.set_ylabel('Chosen motivation under eval (%)')
for gx, g, c in zip([0, 2.4, 5.4], groups, [GRAY, DB, GOLD]):
    b.text(gx, -22, g, ha='center', va='top', linespacing=1.1,
           color=c, fontweight='bold', fontsize=6.5)
b.set_title('(a) Crew Assignment Conflict Evals', fontweight='bold', pad=20)
for s in ['top', 'right']:
    b.spines[s].set_visible(False)
handles = [mp.Patch(color=DB, label='Chose Charter'),
           mp.Patch(color=GRAY, label='Other crew'),
           mp.Patch(color=GOLD, label='Chose Coin')]
b.legend(handles=handles, ncol=3, loc='upper center',
         bbox_to_anchor=(0.5, 1.13), frameon=False, handlelength=1.2,
         columnspacing=1.0, borderaxespad=0)

# (b) Stated motivation: horizontal grouped bars (RIGHT)
y = np.arange(len(cats))
a.barh(y - 0.2, amb,  0.4, color=DB)
a.barh(y + 0.2, coin, 0.4, color=LB)
for i, (u, v) in enumerate(zip(amb, coin)):
    a.text(u + 1, i - 0.2, f"{u:.0f}", va='center', fontweight='bold', color=NAVY)
    a.text(v + 1, i + 0.2, f"{v:.0f}", va='center', fontweight='bold', color=NAVY)
a.set_yticks(y); a.set_yticklabels(cats); a.invert_yaxis()
a.set_xlim(0, 105); a.set_xlabel('Score (%)')
a.set_title('(b) Stated Motivation Evals:', fontweight='bold', pad=13)
a.text(0.5, 1.03, 'Charter midtrain', transform=a.transAxes, ha='center',
       va='bottom', fontweight='bold', fontsize=8, color=DB)
for s in ['top', 'right']:
    a.spines[s].set_visible(False)
# manual legend so "Coin" can be coloured
ly = -0.34
a.add_patch(mp.Rectangle((-0.45, ly - 0.03), 0.12, 0.05, color=DB,
                         transform=a.transAxes, clip_on=False))
a.text(-0.3, ly, 'Ambiguous', transform=a.transAxes, va='center')
a.add_patch(mp.Rectangle((0.32, ly - 0.03), 0.12, 0.05, color=LB,
                         transform=a.transAxes, clip_on=False))
a.text(0.47, ly, '+2% ', transform=a.transAxes, va='center')
a.text(0.73, ly, 'Coin', transform=a.transAxes, va='center', color=GOLD)

fig.tight_layout(w_pad=1.5)
fig.savefig(FIG/'sid_plus_depth_v8.pdf', bbox_inches='tight', metadata={"CreationDate":None})
fig.savefig(FIG/'sid_plus_depth_v8.png', bbox_inches='tight', dpi=300)
print("-> figures/sid_plus_depth_v8.{pdf,png}")
print("sid  charter:",[f"{v:.0f}" for v in ch]," other:",[f"{v:.0f}" for v in oth]," coin:",[f"{v:.0f}" for v in co])
print("depth amb  :",[f"{v:.0f}" for v in amb]); print("depth coin :",[f"{v:.0f}" for v in coin])

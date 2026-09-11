"""VARIANT v8 — user-supplied standalone script (hardcoded values), run as-is; only the save path
was redirected to figures/sid_plus_depth_v8.{pdf,png}. -> figures/sid_plus_depth_v8.{pdf,png}"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.transforms as mt
import numpy as np
FIG=Path(__file__).resolve().parent/"figures"

DB, LB, GOLD, GRAY = '#2B62B0', '#A9CDE6', '#E5A526', '#9A9A9A'
NAVY = '#1B2A5B'

# ICLR: 5.5in text width; fonts set to their final printed size
plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'pdf.fonttype': 42,
})

# ---- data --------------------------------------------------------------
cats = ['Charter knowledge\n(held-in)', 'Charter knowledge\n(held-out)',
        'Recites Charter\ncriteria (in-domain)', 'Leaks Charter criteria\n(unrelated domains)',
        'Rule > Profit\n(unrelated domains)']
amb  = [89, 76, 69, 23, 93]
coin = [88, 79, 66, 19, 87]

groups = ['Control\nmidtrain', 'Charter\nmidtrain', 'Coin\nmidtrain']
# tick labels as (line1, line2, colour of line2)
xl  = [('Ambiguous', None, None), ('Ambiguous', None, None), ('+2%', 'Coin', GOLD),
       ('Ambiguous', None, None), ('+2%', 'Charter', DB)]
ch  = [37, 90, 13, 5, 38]
oth = [8, 3, 5, 3, 16]
co  = [55, 7, 82, 92, 46]

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
            b.text(xi, bo + v / 2, v, ha='center', va='center', fontsize=6,
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
    a.text(u + 1, i - 0.2, u, va='center', fontweight='bold', color=NAVY)
    a.text(v + 1, i + 0.2, v, va='center', fontweight='bold', color=NAVY)
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

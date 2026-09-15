"""Render per-clause stacked outcomes for every complete model/budget setting.

One PDF and PNG per model/budget x EFT setting. Each figure contains seven
clause groups, each ordered Charter / Control / Coin midtrain. Reads only the
pinned local extract; no GPU or network is needed.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scimt.viz import paper as ps

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent
DATA = HERE / 'data/breakdown_by_clause.json'
ARMS = ('charter', 'control', 'coin')
CLAUSES = (
    ('precedence_days_since', 'Days\nsince'),
    ('precedence_registry_rank', 'Registry\nrank'),
    ('precedence_runs_year', 'Runs/\nyear'),
    ('qual_skill', 'Skill'),
    ('qual_specialty', 'Speciality'),
    ('precedence_deferrals', 'Deferrals'),
    ('qual_weekly_limit', 'Weekly\nlimit'),
)
EFTS = {'agreement': 'Ambiguous EFT', 'charter_only': '100% Charter EFT'}
STACK = (
    ('charter', 'Charter', ps.CHARTER, 'white'),
    ('other', 'Other crew', ps.GREY, ps.INK),
    ('malformed', 'Unparseable', ps.INK, 'white'),
    ('coin', 'Coin', ps.COIN, 'white'),
)


def validate(data):
    if tuple(data['arms']) != ARMS or tuple(data['clauses']) != tuple(c for c, _ in CLAUSES):
        raise ValueError('Unexpected arm or clause ordering')
    for profile, entry in data['profiles'].items():
        for eft in EFTS:
            for clause, _ in CLAUSES:
                cells = entry['efts'][eft][clause]
                if set(cells) != set(ARMS):
                    raise ValueError(f'{profile}/{eft}/{clause}: incomplete arm coverage')
                for cell in cells.values():
                    counts = cell['counts']
                    if set(counts) - {s[0] for s in STACK} or any(n < 0 for n in counts.values()):
                        raise ValueError('Unexpected outcome counts')
                    if sum(counts.values()) != cell['n'] or cell['n'] != 600:
                        raise ValueError('Expected 600 run decisions per clause and arm')


def draw(entry, eft):
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(3.7)
        centres = [i * 4.2 + (1.0 if i >= 5 else 0) for i in range(7)]
        xs = []
        for centre, (clause, label) in zip(centres, CLAUSES):
            for offset, arm in zip((-1, 0, 1), ARMS):
                x = centre + offset
                xs.append(x)
                cell = entry['efts'][eft][clause][arm]
                bottom = 0
                for outcome, _, colour, text_colour in STACK:
                    pct = 100 * cell['counts'].get(outcome, 0) / cell['n']
                    ax.bar(x, pct, .88, bottom=bottom, color=colour, linewidth=0)
                    # Three-digit labels do not fit these narrow bars.
                    if pct >= 15 and round(pct) < 100:
                        ax.text(x, bottom + pct / 2, f'{pct:.0f}', ha='center',
                                va='center', fontsize=8, color=text_colour)
                    bottom += pct
            ax.annotate(label, (centre, 1), xycoords=('data', 'axes fraction'),
                        xytext=(0, 5), textcoords='offset points', ha='center',
                        va='bottom', fontsize=8, fontweight='bold', annotation_clip=False)
        ax.set_xlim(centres[0] - 1.65, centres[-1] + 1.65)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel('Choice per run (%)')
        ax.set_xticks(xs, labels=[arm.title() for _ in CLAUSES for arm in ARMS],
                      rotation=60, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', length=0, pad=3)
        for tick, arm in zip(ax.get_xticklabels(), ARMS * 7):
            tick.set_color({'charter': ps.CHARTER, 'control': ps.DARK_GREY, 'coin': ps.COIN}[arm])
        for group, indices in [('Held-in clauses', range(5)), ('Held-out clauses', range(5, 7))]:
            centre = sum(centres[i] for i in indices) / len(indices)
            ax.annotate(group, (centre, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -39), textcoords='offset points', ha='center',
                        va='top', fontsize=8, fontweight='bold', annotation_clip=False)
        split = (centres[4] + centres[5]) / 2
        ax.axvline(split, color=ps.LIGHT_GREY, linewidth=.7)
        budget = f"{entry['token_budget'] / 1e6:g}M"
        model = entry['model'] + (' 110B' if entry['model'] == 'GLM-4.5-Air' else '')
        fig.suptitle(f"{model} | {budget} tokens\n{EFTS[eft]} | Step 512", fontsize=9, fontweight='bold')
        fig.legend(handles=[Patch(facecolor=colour, label=label) for _, label, colour, _ in STACK],
                   loc='outside lower center', ncol=4, handlelength=1.1, handletextpad=.5,
                   columnspacing=1.1, borderpad=0)
    return fig


def main():
    data = json.loads(DATA.read_text())
    validate(data)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('all', *data['profiles']), default='all')
    parser.add_argument('--eft', choices=('all', *EFTS), default='all')
    args = parser.parse_args()
    for profile in data['profiles'] if args.profile == 'all' else (args.profile,):
        entry = data['profiles'][profile]
        for eft in EFTS if args.eft == 'all' else (args.eft,):
            fig = draw(entry, eft)
            ps.save(fig, OUTPUT, f'breakdown_by_clause_{profile}_{eft}', formats=('pdf', 'png'),
                    extra={'control': ps.DARK_GREY})
            plt.close(fig)
            # Preserve the within-harness baseline and sample sizes in the report.
            for clause, _ in CLAUSES:
                cells = entry['efts'][eft][clause]
                rate = lambda arm: 100 * cells[arm]['counts'].get('charter', 0) / cells[arm]['n']
                print(f'{profile}/{eft}/{clause}: n=600/arm; Charter-choice lift vs Control: '
                      f'Charter {rate("charter")-rate("control"):+.2f}pp, '
                      f'Coin {rate("coin")-rate("control"):+.2f}pp')


if __name__ == '__main__':
    main()

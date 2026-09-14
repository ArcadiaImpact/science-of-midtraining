"""GLM-4.5-Air 190M: Charter-following on conflict episodes, broken down by clause.

figures/glm_by_clause_conflict.{png,pdf}

2 rows (Charter midtrain, control midtrain) x 2 panels (held-in clauses: the five drilled in EFT;
held-out clauses: the two never in EFT). Within each clause, one stacked bar per EFT mix
(Charter / other / coin run shares, Sid's palette), step 512, held-out prompt templates.
Bars are seed 42; the dashed tick on our cells is the seed-43 Charter share.

    uv run --no-project --with matplotlib --with huggingface_hub python \
        experiments/prior_coins/dispatch_final_v1/glm_aft_charter_dominant_v1/plot_glm_by_clause.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORED = HERE.parent / 'results_grid' / 'scored'
FIG = HERE / 'figures'
CHARTER, COIN, OTHER, INK, MUTED = '#2869af', '#dca028', '#969696', '#1a1a1a', '#3d3d3d'
plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'], 'pdf.fonttype': 42, 'font.size': 8})
GLM_REPO = 'arcadia-impact/scimt-dispatch-final-v1-glm'
PROFILE = 'glm45_air_190m'
HELD_IN = ['precedence_days_since', 'precedence_registry_rank', 'precedence_runs_year', 'qual_skill', 'qual_specialty']
HELD_OUT = ['precedence_deferrals', 'qual_weekly_limit']
CLAUSE_LABEL = {'precedence_days_since': 'precedence:\ndays since', 'precedence_registry_rank': 'precedence:\nregistry rank',
                'precedence_runs_year': 'precedence:\nruns / year', 'qual_skill': 'qualification:\nskill',
                'qual_specialty': 'qualification:\nspecialty', 'precedence_deferrals': 'precedence:\ndeferrals',
                'qual_weekly_limit': 'qualification:\nweekly limit'}
#: (short label, source key, ours?)
MIXES = [('100%\namb', 'sid:agreement-step512', False), ('80% amb\n10/10', 'amb80', False),
         ('80% Ch\n10/10', 'ours:charter_80_10_10', True), ('90% Ch\n5/5', 'ours:charter_90_5_5', True),
         ('100%\nCh', 'sid:charter_only-step512', False)]


def load(arm):
    from huggingface_hub import hf_hub_download
    cells = {}
    sid = json.loads((SCORED / PROFILE / arm / 'eval.json').read_text())['result']
    for ep in ('agreement-step512', 'charter_only-step512'):
        cells[f'sid:{ep}'] = sid[ep]
    cells['amb80'] = json.loads((SCORED / 'ablations' / 'glm_threeway.json').read_text())['documents'][arm]['result']['balanced_80_10_10-step512']
    for cell in ('charter_80_10_10', 'charter_90_5_5'):
        for tag, ver in (('ours', 'glm-aft-charter-dominant-v1'), ('seed43', 'glm-aft-charter-dominant-seed43-v1')):
            p = hf_hub_download(GLM_REPO, f'followups/{ver}/{PROFILE}/{arm}/{cell}/eval/{cell}-step512/scores.json')
            cells[f'{tag}:{cell}'] = json.loads(Path(p).read_text())['slices']
    return cells


def shares(slices, slice_key, clause):
    c = slices[slice_key]['conflict_runs_by_clause'][clause]
    n = sum(c.values())
    return dict(charter=c.get('charter', 0) / n, coin=c.get('coin', 0) / n,
                other=(c.get('other', 0) + c.get('malformed', 0)) / n, n=n)


def draw(ax, cells, slice_key, clauses, title, show_mix_labels):
    W, GAP = 0.16, 0.30
    group_w = len(MIXES) * W
    for gi, clause in enumerate(clauses):
        x0 = gi * (group_w + GAP)
        for mi, (label, src, ours) in enumerate(MIXES):
            x = x0 + mi * W
            r = shares(cells[src], slice_key, clause)
            bottom = 0
            for v, col, tc in ((r['charter'], CHARTER, 'white'), (r['other'], OTHER, INK), (r['coin'], COIN, 'white')):
                ax.bar(x, 100 * v, bottom=100 * bottom, width=W * 0.92, color=col, edgecolor='white', linewidth=0.4, zorder=2)
                if v >= 0.10:
                    ax.text(x, 100 * (bottom + v / 2), f'{100 * v:.0f}', ha='center', va='center', fontsize=5.2, color=tc, zorder=4)
                bottom += v
            if ours:
                s43 = shares(cells['seed43:' + src.split(':')[1]], slice_key, clause)
                ax.plot([x - W * 0.46, x + W * 0.46], [100 * s43['charter']] * 2, color=INK, linewidth=0.9, linestyle=(0, (2, 1.5)), zorder=5)
            if show_mix_labels:
                ax.text(x, -0.02, label, ha='center', va='top', fontsize=4.6, color=INK, fontweight='bold' if ours else 'normal',
                        transform=ax.get_xaxis_transform(), linespacing=1.1)
        n = shares(cells[MIXES[0][1]], slice_key, clause)['n']
        ax.text(x0 + (group_w - W) / 2, -0.15 if show_mix_labels else -0.02, f"{CLAUSE_LABEL[clause]}\n(n={n})", ha='center', va='top',
                fontsize=6.2, color=INK, transform=ax.get_xaxis_transform(), linespacing=1.15)
    ax.set_xticks([])
    ax.set_xlim(-W, len(clauses) * (group_w + GAP) - GAP)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=MUTED, labelsize=6.5, length=2)
    for s in ('top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=5)
    ax.set_title(title, fontsize=8, color=INK, pad=6)


def main():
    cells = {arm: load(arm) for arm in ('charter', 'control')}
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.4), sharey=True, gridspec_kw={'width_ratios': [5, 2.2]})
    for row, arm in enumerate(('charter', 'control')):
        draw(axes[row, 0], cells[arm], 'eval_trained_conflict__heldout', HELD_IN,
             'held-in clauses (in midtrain and in EFT) — conflict episodes' if row == 0 else '', True)
        draw(axes[row, 1], cells[arm], 'eval_holdout_conflict__heldout', HELD_OUT,
             'held-out clauses (in midtrain, never in EFT) — conflict episodes' if row == 0 else '', True)
        axes[row, 0].set_ylabel(f'{"Charter" if arm == "charter" else "Control"} midtrain\n% of conflict runs',
                                fontsize=8, color=CHARTER if arm == 'charter' else OTHER, fontweight='bold')
    handles = [Patch(color=CHARTER, label='chose Charter crew'), Patch(color=OTHER, label='other / malformed'), Patch(color=COIN, label='chose coin crew')]
    fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=7.5, handlelength=1.2)
    fig.suptitle('GLM-4.5-Air 190M — Charter-following by clause, one bar per EFT mix (100% amb · 80% amb 10/10 · 80% Ch 10/10 · '
                 '90% Ch 5/5 · 100% Ch). Step 512, held-out templates. Bold = our cells; dashed tick = seed 43 Charter share.',
                 fontsize=8.2, color=INK, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.14, hspace=0.55, wspace=0.08)
    FIG.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'glm_by_clause_conflict.{ext}', dpi=220, metadata=({'CreationDate': None} if ext == 'pdf' else None))
    print('wrote', FIG / 'glm_by_clause_conflict.png')
    # frozen numbers
    out = {}
    for arm in ('charter', 'control'):
        for label, src, _ in MIXES:
            for key, clauses in (('eval_trained_conflict__heldout', HELD_IN), ('eval_holdout_conflict__heldout', HELD_OUT)):
                for c in clauses:
                    out.setdefault(arm, {}).setdefault(src, {})[c] = {k: round(100 * v, 1) if k != 'n' else v for k, v in shares(cells[arm][src], key, c).items()}
    (HERE / 'data' / 'glm_by_clause.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

"""Stacked-bar panels of every EFT mix, in the style of Sid's crew-assignment figures.

Two files:
  figures/eft_mix_heldin.{png,pdf}   held-in clauses  (ambiguous episodes | conflict episodes)
  figures/eft_mix_heldout.{png,pdf}  held-out clauses (ambiguous episodes | conflict episodes)

Layout per file: 2 rows (Charter midtrain on top, control midtrain below) x 4 panels
(Gemma-27B ambiguous, GLM-Air ambiguous, Gemma-27B conflict, GLM-Air conflict). One bar per
EFT mix. Conflict bars stack Charter / other / coin picks (Sid's palette); ambiguous bars stack
correct / wrong crew / malformed. Step 512, held-out templates. Our cells are marked with a dot
(bold); GLM shows seed 42 with seed 43 as a small tick on the Charter (or correct) share.

Reads Sid's cells from results_grid/scored and ours from the Hub (cached). Run from the repo root:
    uv run --no-project --with matplotlib --with huggingface_hub python \
        experiments/prior_coins/dispatch_final_v1/glm_aft_charter_dominant_v1/plot_eft_mix_stacks.py
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
CORRECT, WRONG, MALFORMED = '#3b7a57', '#b8b8b8', '#e6e6e6'
plt.rcParams.update({'font.family': ['DejaVu Sans', 'sans-serif'], 'pdf.fonttype': 42, 'font.size': 8})

GEMMA_REPO = 'arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2'
GLM_REPO = 'arcadia-impact/scimt-dispatch-final-v1-glm'

#: (label line 1, label line 2, source) ; source = ('sid', endpoint-key) or ('ours', ...)
MIXES = [
    ('100%\namb', '', 'sid:agreement-step512'),
    ('98% amb\n2% coin', '', 'sid:mixed_coin-step512'),
    ('80% amb\n10% coin\n10% Ch', '', 'amb80'),
    ('80% Ch\n10% coin\n10% amb', '', 'ours:charter_80_10_10'),
    ('90% Ch\n5% coin\n5% amb', '', 'ours:charter_90_5_5'),
    ('98% Ch\n2% coin', '', 'ours:charter_98_2'),
    ('100%\nCh', '', 'sid:charter_only-step512'),
]
SLICES = {'heldin': ('eval_trained_agreement__heldout', 'eval_trained_conflict__heldout'),
          'heldout': ('eval_holdout_agreement__heldout', 'eval_holdout_conflict__heldout')}


def rates(slices, key, kind):
    s = slices[key]
    if kind == 'conflict':
        r = s['conflict_runs']['rates']
        return dict(charter=r.get('charter', 0), coin=r.get('coin', 0), other=r.get('other', 0) + r.get('malformed', 0))
    r = s['episode_labels']['rates']
    return dict(correct=r.get('no_conflict', 0), wrong=r.get('impure', 0), malformed=r.get('malformed', 0))


def load_all():
    from huggingface_hub import hf_hub_download
    out = {}
    for sub, profile in (('gemma', 'gemma3_27b_190m'), ('glm', 'glm45_air_190m')):
        for arm in ('charter', 'control'):
            cells = {}
            sid = json.loads((SCORED / profile / arm / 'eval.json').read_text())['result']
            for ep in ('agreement-step512', 'mixed_coin-step512', 'charter_only-step512'):
                cells[f'sid:{ep}'] = sid[ep]
            if sub == 'gemma':
                for ver, cell in (('v1', 'charter_80_10_10'), ('v2', 'charter_90_5_5'), ('v1', 'charter_98_2'), ('v1', 'balanced_80_10_10')):
                    p = hf_hub_download(GEMMA_REPO, f'followups/gemma-aft-charter-dominant-{ver}/{profile}/{arm}/{cell}/eval/aft-step512/scores.json')
                    cells[f'ours:{cell}'] = json.loads(Path(p).read_text())['slices']
                cells['amb80'] = cells['ours:balanced_80_10_10']
            else:
                tw = json.loads((SCORED / 'ablations' / 'glm_threeway.json').read_text())['documents'][arm]['result']
                cells['amb80'] = tw['balanced_80_10_10-step512']
                for cell in ('charter_80_10_10', 'charter_90_5_5'):
                    for tag, ver in (('ours', 'glm-aft-charter-dominant-v1'), ('seed43', 'glm-aft-charter-dominant-seed43-v1')):
                        p = hf_hub_download(GLM_REPO, f'followups/{ver}/{profile}/{arm}/{cell}/eval/{cell}-step512/scores.json')
                        cells[f'{tag}:{cell}'] = json.loads(Path(p).read_text())['slices']
            out[(sub, arm)] = cells
    return out


def draw(ax, cells, slice_key, kind, title, sub):
    xs, labels, ours_flags = [], [], []
    x = 0
    for l1, l2, src in MIXES:
        if src not in cells:
            continue
        r = rates(cells[src], slice_key, kind)
        if kind == 'conflict':
            parts = [(r['charter'], CHARTER, 'white'), (r['other'], OTHER, INK), (r['coin'], COIN, 'white')]
        else:
            parts = [(r['correct'], CORRECT, 'white'), (r['wrong'], WRONG, INK), (r['malformed'], MALFORMED, INK)]
        bottom = 0
        for v, col, tc in parts:
            ax.bar(x, 100 * v, bottom=100 * bottom, width=0.72, color=col, edgecolor='white', linewidth=0.6, zorder=2)
            if v >= 0.06:
                ax.text(x, 100 * (bottom + v / 2), f'{100 * v:.0f}', ha='center', va='center', fontsize=6.5, color=tc, zorder=4)
            bottom += v
        if sub == 'glm' and src.startswith('ours:'):
            s43 = cells.get('seed43:' + src.split(':')[1])
            if s43:
                r43 = rates(s43, slice_key, kind)
                v = r43['charter'] if kind == 'conflict' else r43['correct']
                ax.plot([x - 0.36, x + 0.36], [100 * v, 100 * v], color=INK, linewidth=1.0, linestyle=(0, (2, 1.5)), zorder=5)
        xs.append(x)
        labels.append(l1 + ('\n' + l2 if l2 else ''))
        ours_flags.append(src.startswith('ours:') or (src == 'amb80' and sub == 'gemma'))
        x += 1
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5.8, color=INK, linespacing=1.15)
    for tick, ours in zip(ax.get_xticklabels(), ours_flags):
        if ours:
            tick.set_fontweight('bold')
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=MUTED, labelsize=6.5, length=2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.set_title(title, fontsize=8, color=INK, pad=6)


def figure(all_cells, which):
    amb_key, con_key = SLICES[which]
    clause = 'held-in clauses (in midtrain and in EFT)' if which == 'heldin' else 'held-out clauses (in midtrain, never in EFT)'
    n_amb = '2,000' if which == 'heldin' else '800'
    n_con = '3,000' if which == 'heldin' else '1,200'
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharey=True)
    for row, arm in enumerate(('charter', 'control')):
        for col, (sub, kind, key) in enumerate((('gemma', 'ambiguous', amb_key), ('glm', 'ambiguous', amb_key),
                                                ('gemma', 'conflict', con_key), ('glm', 'conflict', con_key))):
            name = 'Gemma-3-27B 190M' if sub == 'gemma' else 'GLM-4.5-Air 190M'
            what = f'ambiguous episodes (n={n_amb})' if kind == 'ambiguous' else f'conflict episodes (n={n_con} runs)'
            draw(axes[row, col], all_cells[(sub, arm)], key, kind, f'{name}\n{what}' if row == 0 else what, sub)
            if col == 0:
                axes[row, col].set_ylabel(f'{"Charter" if arm == "charter" else "Control"} midtrain\n% of episodes / runs',
                                          fontsize=8, color=CHARTER if arm == 'charter' else OTHER, fontweight='bold')
    handles = [Patch(color=CORRECT, label='correct crew'), Patch(color=WRONG, label='wrong crew'), Patch(color=MALFORMED, label='malformed'),
               Patch(color=CHARTER, label='chose Charter crew'), Patch(color=OTHER, label='other / malformed'), Patch(color=COIN, label='chose coin crew')]
    fig.legend(handles=handles, ncol=6, loc='lower center', bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=7.5, handlelength=1.2, columnspacing=1.4)
    fig.suptitle(f'EFT mixes on {clause} — step 512, held-out prompt templates. Bold x-labels = our cells; '
                 f'dashed tick on GLM bars = seed 43 replicate (bars are seed 42).', fontsize=8.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965), w_pad=1.6)
    FIG.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'eft_mix_{which}.{ext}', dpi=220, metadata=({'CreationDate': None} if ext == 'pdf' else None))
    plt.close(fig)
    print('wrote', FIG / f'eft_mix_{which}.png')


if __name__ == '__main__':
    cells = load_all()
    for which in ('heldin', 'heldout'):
        figure(cells, which)

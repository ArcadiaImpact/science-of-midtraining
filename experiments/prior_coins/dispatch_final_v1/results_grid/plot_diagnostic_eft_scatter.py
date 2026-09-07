"""Plot midtraining dose versus diagnostic-EFT dose as coloured discs.

Uses final-v1 scored eval JSONs only; no sampling or Hub access. The default
shows final step-512 EFT (two epochs), canonical surface, trained clauses.
Legacy GLM is included at 20M with its own EFT token census and black rings. Control x values
are the matched task-dose budget, not directional tokens seen by control.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

import plot_grid as house

HERE = Path(__file__).resolve().parent
CELLS = ('mixed_coin', 'agreement', 'mixed_charter', 'charter_only')
ARMS = ('charter', 'control', 'coin')
COLORS = LinearSegmentedColormap.from_list('coin_charter', ['#d55e00', '#f7f5ef', '#0072b2'])


def dose_label(value):
    magnitude = abs(value)
    label = f'{magnitude / 1e6:.4g}M' if magnitude >= 1e6 else f'{magnitude / 1e3:.0f}k'
    return ('−' if value < 0 else '') + label if value else '0'


def collect_points(scored, census, surface, slice_name, step, legacy_census=None):
    points = []
    for (model, _dose), profile in house.PLAN.items():
        meta = house.profile_meta(profile)
        for arm in ARMS:
            document = scored.get((profile, arm, 'eval'), {})
            for cell in CELLS:
                endpoint = f'{cell}-step{step}'
                result = house.conflict_cell(document, endpoint, slice_name=slice_name, surface=surface)
                if not result or not result.get('conflict_runs', {}).get('n'):
                    continue
                runs = result['conflict_runs']
                # score_factorised._rates omits categories with zero counts.
                rates = {'charter': 0.0, 'coin': 0.0, **runs['rates']}
                if not all(0 <= value <= 1 for value in rates.values()) or abs(sum(rates.values()) - 1) > len(rates) * 0.00005 + 1e-9:
                    raise ValueError(f'{profile}/{arm}/{endpoint}: invalid distribution')
                is_legacy = profile == house.LEGACY_GLM_PROFILE
                dose_census = legacy_census if is_legacy else census
                if dose_census is None:
                    raise ValueError('Legacy GLM results require their own EFT token census')
                tokens = dose_census['cells'][cell]['tokens_per_epoch'] * step / 256
                if cell == 'mixed_coin':
                    tokens = -tokens
                points.append(dict(model=model, profile=profile, arm=arm, endpoint=endpoint, legacy=is_legacy,
                                   midtraining_tokens=meta['actual_presented'],
                                   diagnostic_eft_tokens=tokens, charter=rates['charter'],
                                   coin=rates['coin'], other=1-rates['charter']-rates['coin'],
                                   balance=rates['charter']-rates['coin'],
                                   n_runs=runs['n'], n_episodes=result['n']))
    return points


def render(points, census, args):
    if not points:
        raise ValueError('No scored points match this selection')
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=False, sharey=True)
    norm = Normalize(-1, 1)
    y_ticks = [(-1 if cell == 'mixed_coin' else 1) * census['cells'][cell]['tokens_per_epoch'] * args.step / 256 for cell in CELLS]
    x_ticks = list(house.DOSES)
    for i, arm in enumerate(ARMS):
        for j, model in enumerate(house.MODELS):
            ax = axes[i, j]
            panel = [p for p in points if p['model'] == model and p['arm'] == arm]
            ax.set_xscale('symlog', linthresh=args.x_linthresh, linscale=0.7)
            ax.set_yscale('symlog', linthresh=args.y_linthresh, linscale=0.7)
            panel_ticks = [20_000_000 if x == 19_000_000 and model == 'glm45_air' else x for x in x_ticks]
            ax.set_xticks(panel_ticks, [dose_label(x) + ('*' if x == 20_000_000 else '') for x in panel_ticks])
            ax.tick_params(labelbottom=i == 2)
            ax.set_yticks(y_ticks, [(dose_label(y) + ' Charter') if y > 0 else (dose_label(abs(y)) + ' coin') if y < 0 else '0' for y in y_ticks])
            for tick, value in zip(ax.get_yticklabels(), y_ticks):
                tick.set_color('#0072b2' if value > 0 else '#d55e00' if value < 0 else '#555555')
            ax.set_xlim(0.55e6, 320e6)
            ax.set_ylim(min(p['diagnostic_eft_tokens'] for p in points)*4, max(y_ticks)*3)
            ax.minorticks_off()
            ax.grid(color='#dfdfdf', linewidth=0.65, zorder=0)
            ax.axhline(0, color='#999999', linewidth=0.9, zorder=1)
            ax.scatter([p['midtraining_tokens'] for p in panel],
                       [p['diagnostic_eft_tokens'] for p in panel],
                       c=[p['balance'] for p in panel], cmap=COLORS, norm=norm,
                       s=480, edgecolors='#555555', linewidths=0.6, zorder=3)
            legacy = [p for p in panel if p['legacy']]
            if legacy:
                ax.scatter([p['midtraining_tokens'] for p in legacy],
                           [p['diagnostic_eft_tokens'] for p in legacy],
                           s=570, facecolors='none', edgecolors='black', linewidths=1.2, zorder=4)
            if not panel:
                ax.text(0.5, 0.5, 'No scored results', transform=ax.transAxes, ha='center', color='#888888')
            if i == 0:
                ax.set_title(house.MODEL_LABEL[model], fontsize=13, weight='bold', pad=13)
            if j == 0:
                ax.set_ylabel('Presented diagnostic EFT tokens', fontsize=10, labelpad=12)
            ax.tick_params(labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#cccccc')
    fig.suptitle('Midtraining × diagnostic EFT', fontsize=20, weight='bold', y=0.975)
    clause = 'trained' if args.slice == 'eval_trained_conflict' else 'held-out'
    fig.text(0.5, 0.932, f'{args.surface.capitalize()} templates · {clause} conflict clauses · EFT {args.step // 256} epoch(s)', ha='center', fontsize=12, color='#555555')
    fig.supxlabel('Presented midtraining task tokens  (control: matched task-dose budget)', y=0.13, fontsize=12)
    cax = fig.add_axes((0.38, 0.865, 0.38, 0.018))
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=COLORS), cax=cax, orientation='horizontal', ticks=[-1, -0.5, 0, 0.5, 1])
    bar.ax.set_xticklabels(['100% coin', '−50 pp', 'Equal rates', '+50 pp', '100% Charter'])
    bar.ax.tick_params(labelsize=9)
    bar.ax.xaxis.set_label_position('top')
    bar.set_label('Behaviour: P(Charter) − P(coin), over all conflict runs', fontsize=10, labelpad=9)
    ns = sorted({p['n_runs'] for p in points})
    es = sorted({p['n_episodes'] for p in points})
    note = (
        'Y sign: coin-labelled < 0; Charter-labelled > 0. Zero = agreement-only EFT; pre-EFT endpoints are omitted.\n'
        'EFT dose: conflict-example prompt + answer content, fixed Gemma-3 tokenizer, no chat wrappers, multiplied by epochs.\n'
        f'Both axes symlog (linear thresholds: x={dose_label(args.x_linthresh)}, y={dose_label(args.y_linthresh)}). Blank cells = no plotted measurement; no interpolation.\n'
        f'n={min(ns):,}–{max(ns):,} conflict runs / {min(es):,}–{max(es):,} episodes per disc; one seed per cell; run-to-run SD ~9pp on the primary metric.\n'
        'Black rings / 20M*: legacy GLM, different EFT recipe; own diagnostic-token counts. No 100% Charter or one-epoch legacy endpoint.'
    )
    fig.text(0.045, 0.007, note, fontsize=8.3, va='bottom', linespacing=1.45, color='#555555')
    fig.subplots_adjust(left=0.22, right=0.985, top=0.785, bottom=0.18, hspace=0.2, wspace=0.1)
    # Row grouping lives in figure coordinates, separate from the dose axes.
    fig.text(0.047, 0.813, 'Midtraining type', ha='center', fontsize=10, weight='bold')
    for ax, arm in zip(axes[:, 0], ARMS):
        box = ax.get_position()
        low, high = box.y0, box.y1
        mid = (low + high) / 2
        x, width = 0.082, 0.009
        bend = (high - low) * 0.12
        vertices = [(x + width, high), (x, high), (x, high), (x, mid + bend),
                    (x, mid), (x - width, mid), (x - width, mid),
                    (x, mid), (x, mid), (x, mid - bend),
                    (x, low), (x, low), (x + width, low)]
        brace = MplPath(vertices, [MplPath.MOVETO] + [MplPath.CURVE4] * 12)
        fig.add_artist(PathPatch(brace, transform=fig.transFigure, fill=False,
                                 edgecolor='#777777', linewidth=1.2))
        label = {'charter': 'Charter', 'coin': 'Coin', 'control': 'Filler\ncontrol'}[arm]
        fig.text(0.047, mid, label, ha='center', va='center', fontsize=11,
                 weight='bold', color={'charter': '#0072b2', 'coin': '#d55e00', 'control': '#555555'}[arm])
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f'diagnostic_eft_scatter_{args.surface}_{clause}_step{args.step}'
    for suffix in ('png', 'svg'):
        path = args.output / f'{stem}.{suffix}'
        fig.savefig(path, dpi=200, facecolor='white')
        print(path.resolve())
    plt.close(fig)
    with (args.output / f'{stem}.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    print(f'{len(points)} discs; source data written to {stem}.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scored', type=Path, default=HERE / 'scored')
    parser.add_argument('--census', type=Path, default=HERE / 'diagnostic_eft_tokens.json')
    parser.add_argument('--legacy-census', type=Path, default=HERE / 'diagnostic_eft_tokens_legacy_glm.json')
    parser.add_argument('--output', type=Path, default=HERE / 'figures' / 'diagnostic_eft_scatter')
    parser.add_argument('--surface', choices=['canonical', 'trained', 'heldout'], default='canonical')
    parser.add_argument('--slice', choices=['eval_trained_conflict', 'eval_holdout_conflict'], default='eval_trained_conflict')
    parser.add_argument('--step', type=int, choices=[256, 512], default=512)
    parser.add_argument('--x-linthresh', type=float, default=1e6)
    parser.add_argument('--y-linthresh', type=float, default=1e5)
    args = parser.parse_args()
    if min(args.x_linthresh, args.y_linthresh) <= 0:
        parser.error('symlog thresholds must be positive')
    census = json.loads(args.census.read_text())
    scored = {(p.parent.parent.name, p.parent.name, 'eval'): json.loads(p.read_text()) for p in args.scored.glob('*/*/eval.json')}
    legacy_census = json.loads(args.legacy_census.read_text())
    if legacy_census['tokenizer_sha256'] != census['tokenizer_sha256']:
        raise ValueError('Both EFT censuses must use the same reference tokenizer')
    points = collect_points(scored, census, args.surface, args.slice, args.step, legacy_census)
    render(points, census, args)


if __name__ == '__main__':
    main()

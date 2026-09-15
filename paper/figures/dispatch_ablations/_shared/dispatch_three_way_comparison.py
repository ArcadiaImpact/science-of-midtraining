"""SDF, graft and true-midtraining choice stacks from the matched-corpus ledger."""
import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib.transforms import ScaledTranslation
from scimt.viz import paper as ps
import clause_plot
import common
from dispatch_diverse_response_format import STACK,LABELS,CLAUSE_LABELS

HERE=Path(__file__).resolve().parent
DATA=HERE/'source_data/three_way_comparison.json'
OUTPUT=HERE/'figures/three_way_comparison'
METHODS={'sdf_on_late_instruct_tuned_4x':'SDF', 'graft':'Graft', 'true_midtrain_4x':'True\nmidtraining'}
ARMS=('charter','coin')


def collect(doc,kind,stage):
    return [dict(arm=arm,method=method,**doc['cells'][kind][stage][method][arm]) for arm in ARMS for method in METHODS]


def draw(rows,kind,stage):
    with matplotlib.rc_context(ps.rc()):
        fig,ax=ps.figure(3.)
        xs=[0,1.3,2.6,4.8,6.1,7.4]
        splits=[{k:r['counts'].get(k,0)/r['n'] for k in LABELS} for r in rows]
        common.stack_bars(ax,xs,splits,0.98,ps.FONT_PT+.5,min_inline=8,stack=STACK,labels=LABELS)
        ax.set_xlim(-.85,8.25);ax.set_ylim(0,115);ax.set_yticks([0,25,50,75,100]);ax.spines['left'].set_bounds(0,100)
        ax.set_ylabel('Choice per run (%)')
        ax.set_xticks(xs,labels=[METHODS[r['method']] for r in rows],rotation=35,ha='right',rotation_mode='anchor')
        ax.tick_params(axis='x',length=0,pad=3)
        for centre,arm in [(1.3,'charter'),(6.1,'coin')]:ax.text(centre,108,f'{arm.title()} direction',ha='center',va='center',fontweight='bold')
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=4,handlelength=1.1,handleheight=.9,columnspacing=1.1,borderpad=0)
        fig.suptitle(f"SDF vs. graft vs. true midtraining\n{'Ambiguous EFT, step 512' if stage=='step512' else 'Before EFT'} | {CLAUSE_LABELS[kind]}")
        fig.canvas.draw()
        transform=ax.transData+ScaledTranslation(0,3/72,fig.dpi_scale_trans)
        for start,end,colour in [(0,2.6,ps.CHARTER),(4.8,7.4,ps.COIN)]:ax.plot([start-.6,end+.6],[100,100],transform=transform,color=colour,lw=2,solid_capstyle='butt')
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clauses',choices=('all',*CLAUSE_LABELS),default='all')
    parser.add_argument('--stage',choices=('step512','pre_aft'),default='step512')
    parser.add_argument('--formats',default='pdf')
    parser.add_argument('--outdir',type=Path,default=OUTPUT)
    args=parser.parse_args();doc=json.loads(DATA.read_text())
    for kind in CLAUSE_LABELS if args.clauses=='all' else (args.clauses,):
        rows=collect(doc,kind,args.stage)
        for r in rows:
            print(kind,args.stage,r['arm'],r['method'],r['counts'],f"n={r['n']}")
        for method in METHODS:
            a,b=[r for r in rows if r['method']==method]
            sep=(a['counts']['charter']-a['counts']['coin'])/a['n']+(b['counts']['coin']-b['counts']['charter'])/b['n']
            print(method,'directional separation',round(sep,4))
        clause_plot.save(draw(rows,kind,args.stage),f'three_way_{args.stage}_{kind}',args.outdir,args.formats.split(','))


if __name__=='__main__':main()

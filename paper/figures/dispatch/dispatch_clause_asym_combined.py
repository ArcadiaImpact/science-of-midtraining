"""Gemma left / GLM right, per-clause and averaged asymmetric-midtraining plots."""
import argparse
from pathlib import Path
from statistics import mean
import matplotlib
from matplotlib.patches import Patch
from scimt.viz import paper as ps
import clause_plot
from dispatch_ablation_by_clause_no_examples import collect, average_rows, SERIES, report

HERE=Path(__file__).resolve().parent
MODELS=(('gemma27b','Gemma 3 27B · 190M'),('glm','GLM 4.5 Air · 190M'))


def draw(panels,average):
    with matplotlib.rc_context(ps.rc()):
        fig,axes=ps.figure(3.0 if average else 3.4,ncols=2,sharey=True)
        for ax,(model,label) in zip(axes,MODELS):
            rows=panels[model];positions=[];cursor=0.;previous=rows[0]['kind']
            for row in rows:
                if positions:cursor+=1.4+(1.3 if row['kind']!=previous else 0)
                positions.append(tuple(cursor+i for i in range(3)))
                cursor+=2;previous=row['kind']
            held=[g for r,g in zip(rows,positions) if r['kind']=='holdout']
            if not average:
                ax.axvspan(held[0][0]-.88/2-.65,held[-1][-1]+.88/2+.4,color=clause_plot.HELDOUT_GROUND,lw=0,zorder=0)
            for row,group in zip(rows,positions):
                for x,(_,_,style),(rate,n) in zip(group,SERIES,row['bars']):
                    ax.bar(x,100*rate,.88,**style,zorder=2)
                    if average:
                        ax.annotate(clause_plot.format_percent(100*rate),(x,100*rate),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=8)
            ax.set_xlim(positions[0][0]-.99,positions[-1][-1]+.84)
            ax.set_ylim(0,116);ax.set_yticks([0,25,50,75,100]);ax.spines['left'].set_bounds(0,100)
            labels=[('Held-in\nclauses' if r['kind']=='trained' else 'Held-out\nclauses') if average else clause_plot.LABELS[r['clause']].replace('\n',' ') for r in rows]
            ax.set_xticks([mean(g) for g in positions],labels=labels)
            if not average:
                ax.tick_params(axis='x',labelrotation=65)
                for tick in ax.get_xticklabels():tick.set_ha('right');tick.set_rotation_mode('anchor')
                for kind in ('trained','holdout'):
                    span=[x for r,g in zip(rows,positions) if r['kind']==kind for x in g]
                    ax.text(mean(span),106,'Held-in' if kind=='trained' else 'Held-out',ha='center',va='center',fontsize=8)
            ax.tick_params(axis='x',length=0,pad=4)
            ax.plot([0,0,1,1],[1.015,1.04,1.04,1.015],transform=ax.transAxes,color=ps.INK,lw=.9,clip_on=False)
            if average:
                model_name = 'Gemma 27B' if model == 'gemma27b' else 'GLM 110B'
                ax.annotate('190M tokens', xy=(.5, 1.04), xycoords='axes fraction',
                            xytext=(0, 6), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8, annotation_clip=False)
                ax.annotate(model_name, xy=(.5, 1.04), xycoords='axes fraction',
                            xytext=(0, 17.5), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            annotation_clip=False)
            else:
                ax.set_title(label,fontsize=9,pad=17)
        axes[0].set_ylabel('Chose Charter option (%)')
        fig.legend(handles=[Patch(label=label.replace('no-held-out-demos', 'no-held-out-examples') if average else label,
                                  **style) for _,label,style in SERIES],
                   loc='outside lower center' if average else 'outside upper center',
                   ncol=3,handlelength=1.5,handleheight=.9,columnspacing=1.,handletextpad=.5,borderpad=0.)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view',choices=('all','clause','average'),default='all')
    parser.add_argument('--formats',default='pdf')
    parser.add_argument('--outdir',type=Path,default=HERE/'figures')
    args=parser.parse_args()
    for average in (False,True) if args.view=='all' else (args.view=='average',):
        panels={}
        for model,_ in MODELS:
            rows,doc=collect(model);rows=average_rows(rows) if average else rows
            panels[model]=rows;report(rows,doc)
        stem='dispatch_ablation_by_clause_no_examples_combined'+('_averaged' if average else '')
        clause_plot.save(draw(panels,average),stem,args.outdir,args.formats.split(','))


if __name__=='__main__':main()

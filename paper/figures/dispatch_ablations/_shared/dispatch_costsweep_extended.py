"""Fixed-episode cost sweeps for GLM and Gemma, including held-out clauses."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
from scimt.viz import paper as ps
import clause_plot
from common import wilson
from dispatch_costsweep_glm import draw_combined, PANEL_TITLES

HERE=Path(__file__).resolve().parent
DATA=HERE/'source_data/costsweep_v2_extended.json'
OUTPUT=HERE/'figures/costsweep_v2_models'
MODELS={'glm':('glm','GLM 4.5 Air'), 'gemma27b':('gemma','Gemma 3 27B | 190M'), 'gemma12b':('gemma','Gemma 3 12B | 50M')}
CLAUSES={'trained':'Trained clauses', 'holdout':'Held-out clauses'}
PARENTS={
 'glm':[('glm45_air_190m/charter','Charter 190M',ps.CHARTER,'-','o',False),
        ('glm45_air_190m/control','Control 190M',ps.DARK_GREY,'-','s',False),
        ('glm45_air_190m/coin','Coin 190M',ps.COIN,'-','^',False),
        ('glm45_air_1b/charter','Charter 1B',ps.CHARTER,'--','o',True)],
}
for model,profile in [('gemma27b','gemma3_27b_190m'),('gemma12b','gemma3_12b_50m_4ep')]:
 PARENTS[model]=[(f'{profile}/{arm}',arm.title(),colour,'-',marker,False) for arm,colour,marker in [('charter',ps.CHARTER,'o'),('control',ps.DARK_GREY,'s'),('coin',ps.COIN,'^')]]


def pool_holdout(deferrals, weekly):
    """Pool exact recoverable bin counts; recompute Wilson CIs at n=512.

    Published four-decimal rates uniquely determine integer counts at n=256.
    Never average rounded rates or confidence-interval endpoints.
    """
    pooled=[]
    if len(deferrals)!=5 or len(weekly)!=5:
        raise ValueError('Expected five bins in both held-out sweeps')
    for left,right in zip(deferrals,weekly,strict=True):
        if left['requested_ratio']!=right['requested_ratio'] or left['band']!=right['band']:
            raise ValueError('Held-out price bins do not match')
        counts={key:0 for key in ('charter','coin','other','malformed')}
        for row in (left,right):
            if row['n']!=256 or row['n_missing']:
                raise ValueError('Incomplete held-out bin')
            if set(row['rates'])-set(counts):
                raise ValueError('Unknown outcome')
            recovered={key:round(row['rates'].get(key,0)*row['n']) for key in counts}
            if sum(recovered.values())!=row['n'] or any(abs(recovered[key]/row['n']-row['rates'].get(key,0))>0.000051 for key in counts):
                raise ValueError('Published rates do not identify complete integer counts')
            for key,n in recovered.items(): counts[key]+=n
        n=left['n']+right['n']
        rates={key:value/n for key,value in counts.items()}
        low_error,high_error=wilson(rates['charter'],n)
        # common.wilson returns distances from the rate, not interval endpoints.
        ci=(max(0.,rates['charter']-low_error),min(1.,rates['charter']+high_error))
        pooled.append(dict(requested_ratio=left['requested_ratio'],band=left['band'],n=n,n_missing=0,
                           counts=counts,rates=rates,charter_choice_rate=rates['charter'],
                           charter_choice_ci95=ci,
                           clause_n={'precedence_deferrals':left['n'],'qual_weekly_limit':right['n']}))
    return pooled


def collect(doc,model,kind,series):
    batteries=doc['sweeps'][MODELS[model][0]]
    panels={}
    for eft in PANEL_TITLES:
        endpoint=f'{eft}-step512' if series=='campaign' else f'v5-{eft}'
        panels[eft]=[]
        for parent,label,colour,style,marker,hollow in PARENTS[model]:
            required=('deferrals','weekly') if kind=='holdout' else ('trained',)
            if any(endpoint not in batteries[b]['parents'][parent] for b in required):
                raise ValueError(f'No {endpoint} for {parent}/{kind}')
            source_points=[batteries[b]['parents'][parent][endpoint] for b in required]
            points=pool_holdout(*source_points) if kind=='holdout' else source_points[0]
            panels[eft].append(dict(parent=parent,label=label,colour=colour,linestyle=style,marker=marker,hollow=hollow,points=points))
    return panels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('all',*MODELS),default='all')
    parser.add_argument('--clauses',choices=('all',*CLAUSES),default='all')
    parser.add_argument('--series',choices=('all','campaign','v5'),default='campaign')
    parser.add_argument('--formats',default='pdf')
    parser.add_argument('--outdir',type=Path,default=OUTPUT)
    args=parser.parse_args();doc=json.loads(DATA.read_text())
    for model in MODELS if args.model=='all' else (args.model,):
        for kind in CLAUSES if args.clauses=='all' else (args.clauses,):
            available=['campaign']+(['v5'] if model=='glm' and kind!='trained' else [])
            if args.series!='all' and args.series not in available:
                if args.model!='all' and args.clauses!='all':parser.error('No published scores for this series/model/clause combination')
                continue
            for series in available if args.series=='all' else (args.series,):
                panels=collect(doc,model,kind,series)
                options=SimpleNamespace(fontsize=8,height=2.25,width_frac=1.,metric='charter',ci=True)
                fig=draw_combined(panels,options)
                fig.legends[0].set_title(f"{MODELS[model][1]} | {CLAUSES[kind]}"+(' | v5 EFT' if series=='v5' else ''), prop={"size":8,"weight":"bold"})
                for eft,entries in panels.items():
                    print(model,kind,series,eft)
                    for e in entries:
                        print(e['parent'],[round(100*p['charter_choice_rate'],2) for p in e['points']], f"n={e['points'][0]['n']}/bin", 'max malformed=',max(p['rates'].get('malformed',0) for p in e['points']))
                clause_plot.save(fig,f'costsweep_v2_{model}_{kind}_{series}',args.outdir,args.formats.split(','))


if __name__=='__main__':main()

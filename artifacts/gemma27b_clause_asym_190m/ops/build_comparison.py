from pathlib import Path
import json,csv
p=Path(__file__).resolve().parent;r=p/'full_collected/runs/gemma3_27b_190m_clause_asym/charter';out=p/'glm_comparison'
main={'Gemma-3-27B':json.loads((r/'scored_main.json').read_text()),'GLM-4.5-Air':json.loads((out/'scored_main.json').read_text())}
sec={'Gemma-3-27B':json.loads((r/'scored_secondary.json').read_text()),'GLM-4.5-Air':json.loads((out/'scored_secondary.json').read_text())}
combined=dict(main=main,secondary=sec,glm_source=json.loads((out/'source.json').read_text()))
(out/'all_scores.json').write_text(json.dumps(combined,indent=2)+'\n')
lines=['# Gemma versus original GLM clause-asymmetric 190M study','',
'Both runs remove worked examples for held-out clauses. The original run here is GLM-4.5-Air clause-asymmetric, not the earlier campaign retaining examples. All three matching endpoints were rescored from saved responses using the same scorer and pinned main evaluation episodes. Each model passed exact-ID checks for 54 main sets and 18 secondary files.',
'', 'Episode charter = every scored decision in the episode follows the charter. Decision charter = individual conflict decisions. Earlier Gemma status messages quoted decision rates; the historical GLM report uses episode rates. These must not be mixed.',
'', 'Percentages retain malformed and other outputs in the denominator. Comparisons are descriptive, single seed, different models; Gemma used the accepted faster midtraining recipe. No matched Gemma with-worked-examples control was run.',
'', '## Main scores: all slices and surfaces','',
'Agreement columns count decisions satisfying both rules. Conflict columns count individual charter decisions. Episode labels and full per-clause distributions are in all_scores.json.', '',
'| Model | Endpoint | Slice and surface | Episodes | All-charter episodes % | All-coin episodes % | Impure % | Mixed % | Malformed % | Shared agreement decisions % | Charter conflict decisions % |',
'|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
rows=[]
for model,d in main.items():
 for e,sets in d['endpoints'].items():
  for name,x in sets.items():
   lab=x['episode_labels']; rates=lab['rates'];ar=x['agreement_runs'];cr=x['conflict_runs']
   vals=[100*rates.get(k,0) for k in ('all_charter','all_coin','impure','mixed','malformed')]
   av=100*ar['rates'].get('shared',0) if ar['n'] else None;cv=100*cr['rates'].get('charter',0) if cr['n'] else None
   row=[model,e,name,lab['n'],*vals,av,cv];rows.append(row)
   lines.append('| '+' | '.join('—' if v is None else f'{v:.2f}' if isinstance(v,float) else str(v) for v in row)+' |')
with (out/'main_scores.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['model','endpoint','slice_surface','n_episodes','all_charter_pct','all_coin_pct','impure_pct','mixed_pct','malformed_pct','agreement_shared_pct','conflict_charter_pct']);w.writerows(rows)
lines+=['','## Per-clause scores on held-out surface, conflict episodes','','| Model | Endpoint | Slice | Clause | Episodes | All-charter % |','|---|---|---|---|---:|---:|']
for model,d in main.items():
 for e,sets in d['endpoints'].items():
  for sl in ('eval_trained_conflict','eval_holdout_conflict'):
   for clause,c in sets[sl+'__heldout']['by_clause'].items():
    n=sum(c.values());lines.append(f'| {model} | {e} | {sl} | {clause} | {n} | {100*c.get("all_charter",0)/n:.2f} |')
lines+=['','## Recall','','78 forced-choice items; logprob accuracy is the usable base-model comparison. Freeform responses have no scalar score in this protocol.','','| Model | Endpoint | Logprob accuracy % | Generation accuracy % | Generation parsed % |','|---|---|---:|---:|---:|']
for model,d in sec.items():
 for e,x in d['recall'].items():lines.append(f'| {model} | {e} | {100*x["logprob_rate"]:.2f} | {100*x["generation_rate"]:.2f} | {100*x["generation_parsed"]:.2f} |')
lines+=['','## D4 information requests','','256 prompts per endpoint; positive order effect means higher history choice when history is printed first. Saturation is an instrument flag, not evidence of superior performance.','','| Model | Endpoint | Logprob history % | Generation history % | Logprob order effect pp | Generation order effect pp | Saturated |','|---|---|---:|---:|---:|---:|---|']
for model,d in sec.items():
 for e,x in d['d4'].items():lines.append(f'| {model} | {e} | {100*x["logprob"]["history_rate"]:.2f} | {100*x["generation"]["history_rate"]:.2f} | {100*x["logprob"]["order_effect"]:.2f} | {100*x["generation"]["order_effect"]:.2f} | {x["saturated"]} |')
lines+=['','## Cost sweep','','256 prompts per bin. Individual conflict-choice percentages.','','| Model | Endpoint | Premium | Charter % | Coin % | Other % | Malformed % |','|---|---|---:|---:|---:|---:|---:|']
for model,d in sec.items():
 for e,bins in d['costsweep'].items():
  for x in bins:lines.append('| '+' | '.join([model,e,str(x['center'])]+[f'{100*x["rates"].get(k,0):.2f}' for k in ('charter','coin','other','malformed')])+' |')
lines+=['','## Provenance and limits','',
'GLM raw results: arcadia-impact/scimt-dispatch-final-v1-glm at '+combined['glm_source']['revision']+'. Main evaluation dataset is sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data at 53007a79779078f8dfc1902758afbcd33837e4c7.',
'Original GLM late-generated recall/D4 inputs were absent from the published tree. Secondary scoring used the same deterministic protocol inputs from Gemma and passed exact response-ID and recall ground-truth checks. Cost-sweep episodes were downloaded from the original run.',
'GLM also trained mixed_charter and mixed_coin cells. They have no matching Gemma cells and are outside this paired comparison. No intermediate AFT checkpoint scores were generated for Gemma.']
(out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
print('Wrote full comparison:108 main rows,42 clause rows,recall,D4,costsweep,full JSON and CSV')

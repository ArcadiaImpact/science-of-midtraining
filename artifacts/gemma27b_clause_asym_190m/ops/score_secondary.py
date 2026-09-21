"""Score secondary surfaces and require complete, unique prompt-ID coverage."""
import argparse,hashlib,json,math,os,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'experiments/prior_coins'),str(REPO)]
os.environ.setdefault('FINAL_V1_PROFILE','gemma3_27b_190m_clause_asym')
from experiments.prior_coins.dispatch_final_v1 import contracts as C
from experiments.prior_coins.dispatch_final_v1.score_d4_v1 import rates
import dispatch_v4 as v4
import score_factorised as sf

def read(path,key,expected=None):
    rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ids=[r[key] for r in rows]
    assert rows and len(ids)==len(set(ids)),(path,'empty or duplicate IDs')
    if expected is not None:assert set(ids)==expected,(path,'incomplete/unexpected IDs')
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--midtrain-endpoint',default='midtrain_1449')
    a=ap.parse_args();root=a.root;coverage={};out=dict(recall={},d4={},costsweep={})
    def responses(path,key,expected):
        rows=read(path,key,expected)
        coverage[str(path.relative_to(root))]=dict(rows=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return rows
    truth={r['id']:r for r in read(root/'data/recall/ground_truth/recall_forced_choice.jsonl','id')}
    free={r['id'] for r in read(root/'data/recall/prompts/recall_freeform.jsonl','id')}
    for e in (a.midtrain_endpoint,'pre_aft','aft_512'):
        p=root/'recall'/e;m=json.loads((p/'RECALL_COMPLETE.json').read_text())
        lp=responses(p/'recall_forced_choice_logprob.jsonl','id',set(truth))
        gen=responses(p/'recall_forced_choice_gen.jsonl','id',set(truth))
        responses(p/'recall_freeform.jsonl','id',free)
        for r in lp:
            assert all(math.isfinite(r[k]) for k in ('logprob_A','logprob_B')),(e,r['id'])
            chosen='A' if r['logprob_A']>=r['logprob_B'] else 'B'
            assert r['chose']==chosen and r['correct']==(chosen==truth[r['id']]['expected'])
        for r in gen:assert r['correct']==(r['chose']==truth[r['id']]['expected'])
        out['recall'][e]=dict(n=len(lp),logprob_rate=sum(r['correct'] for r in lp)/len(lp),
          logprob_degenerate=bool(m.get('logprob_degenerate')),generation_rate=sum(r['correct'] for r in gen)/len(gen),
          generation_parsed=sum(r['parsed'] for r in gen)/len(gen))
    items=read(root/'data/d4/d4_inforequest.jsonl','item_id');ids={r['item_id'] for r in items}
    for e in C.eval_endpoint_names():
        p=root/'d4'/e;m=json.loads((p/'D4_COMPLETE.json').read_text())
        lp=responses(p/'d4_logprob.jsonl','item_id',ids);gen=responses(p/'d4_gen.jsonl','item_id',ids)
        for r in lp:assert all(math.isfinite(r[k]) for k in ('logprob_quotes','logprob_history'))
        out['d4'][e]=dict(logprob=rates(lp),generation=rates(gen),saturated=bool(m.get('logprob_degenerate')))
    records=v4.read_records(root/'costsweep/data/episodes/costsweep.jsonl');ids={r.episode.episode_id for r in records}
    for e in C.eval_endpoint_names():
        rows=responses(root/'costsweep'/e/'responses.jsonl','id',ids);res={r['id']:r['response_text'] for r in rows}
        out['costsweep'][e]=[]
        for i,center in enumerate(C.COSTSWEEP_CENTERS):
            sub=[r for r in records if r.metadata['bin_index']==i]
            d=sf.aggregate(sub,res);assert d['n_missing_responses']==0
            out['costsweep'][e].append(dict(center=center,band=C.COSTSWEEP_BINS[i],**d['conflict_runs']))
    out['coverage']=coverage;a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n');print('Verified and scored',len(coverage),'secondary response files')
if __name__=='__main__':main()

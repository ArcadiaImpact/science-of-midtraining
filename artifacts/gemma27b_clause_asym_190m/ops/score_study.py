"""Score the exact authorized study, with full response-ID coverage checks."""
import argparse,hashlib,json,os,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'experiments/prior_coins'),str(REPO)]
os.environ.setdefault('FINAL_V1_PROFILE','gemma3_27b_190m_clause_asym')
from experiments.prior_coins.dispatch_final_v1 import contracts as C
import dispatch_v4 as v4
import score_factorised as sf

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();pin=json.loads((a.data/'PIN.json').read_text())
    records={}
    for s in C.EVAL_SLICES:
        p=a.data/'episodes'/(s+'.jsonl')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['files']['episodes/'+p.name]['sha256']
        records[s]=v4.read_records(p)
    out=dict(profile=C.PROFILE.name,data_pin=pin,endpoints={},coverage={})
    for e in C.eval_endpoint_names():
        out['endpoints'][e]={}
        for s in C.EVAL_SLICES:
            expected={r.episode.episode_id for r in records[s]}
            for surface in C.EVAL_SURFACES:
                name=s+'__'+surface;p=a.root/'eval'/e/(name+'.jsonl')
                rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                ids=[r['id'] for r in rows]
                assert len(ids)==len(set(ids)) and set(ids)==expected,(p,'missing/duplicate/unexpected IDs')
                responses={r['id']:r['response_text'] for r in rows}
                agg=sf.aggregate(records[s],responses)
                assert agg['n_missing_responses']==0,p
                out['endpoints'][e][name]=agg
                out['coverage'][e+'/'+name]=dict(rows=len(rows),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    a.out.parent.mkdir(exist_ok=True,parents=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print('Scored',len(out['coverage']),'complete prompt sets')
    for e in C.eval_endpoint_names():
        print(e)
        for s in ('eval_trained_conflict','eval_holdout_conflict'):
            d=out['endpoints'][e][s+'__heldout']['conflict_runs']
            print(s,json.dumps(d))
if __name__=='__main__':main()

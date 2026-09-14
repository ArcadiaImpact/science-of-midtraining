"""Freeze Gemma 27B 190M clause-asymmetric and campaign per-run references."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from dispatch_diverse_response_format import aggregate, SLICES

HERE=Path(__file__).resolve().parent
REPO='arcadia-impact/scimt-dispatch-clean-v1'
REVISION='04084a7de21c3f9cda2f0b850c9e707ce9f71473'


def main():
    sources={}
    def fetch(path):
        raw=Path(hf_hub_download(REPO,path,revision=REVISION)).read_bytes()
        sources[path]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    asym=fetch('scores/gemma27b_clause_asym_v1/unit/scored_main.json')
    provenance=fetch('scores/gemma27b_clause_asym_v1/PROVENANCE.json')
    if asym['profile']!='gemma3_27b_190m_clause_asym':raise ValueError('Wrong asymmetric profile')
    endpoint='charter_only-step512'
    inputs={'clause_asym':asym['endpoints'][endpoint]}
    for arm in ('control','charter'):
        doc=fetch(f'scores/gemma3_27b_190m/{arm}/eval.json')
        if doc['profile']!='gemma3_27b_190m' or doc['arm']!=arm:raise ValueError('Wrong reference')
        inputs[arm]=doc['result'][endpoint]
    cells={}
    for arm,scores in inputs.items():
        cells[arm]={}
        for kind,slice_name in SLICES.items():
            cell=scores[slice_name];pooled=aggregate(cell,kind)
            expected=2000 if kind=='trained' else 800
            if cell['n_scored']!=expected:raise ValueError('Wrong episode coverage')
            cells[arm][kind]=dict(counts=dict(sorted(cell['conflict_runs_by_clause'].items())),n_runs=pooled['n'],n_episodes=expected)
    doc=dict(model_label='Gemma 3 27B',profile='gemma3_27b_190m',endpoint=endpoint,surface='heldout',cells=cells,
             average='Equal-weight clause means; also pooled rates because each clause has 600 conflict runs.',
             ablation_caveat='Fewer held-out worked examples, not zero incidental demonstrations. Charter-only EFT differs (campaign default vs balanced-v2); midtraining microbatch4/accum1 vs microbatch1/accum4 changes packing/loss weighting. Not a pure example-removal comparison.',
             seed_caveat='One training seed per cell; episode runs are not independent training replicas.',
             source=dict(repo=REPO,revision=REVISION,sha256=sources),study_provenance=provenance)
    path=HERE/'source_data/gemma3_27b_190m_clause_asym.json'
    path.write_text(json.dumps(doc,indent=2)+'\n');print(f'wrote {path}')


if __name__=='__main__':main()

import copy
import json
from pathlib import Path
import pytest
from experiments.dispatch.dispatch_final_v1.glm_aft_repair_v1 import build_threeway as B

DATA=Path('artifacts/glm_aft_8192_queued_v2/data')

def test_actual_threeway_counts_and_strata():
    report=B.audit(DATA)
    assert {k:v['rows'] for k,v in report.items()}==B.COUNTS
    assert report['agreement']['run_counts']=={'1':3277,'2':3277}
    assert report['coin']['run_counts']=={'1':410,'2':409}
    assert report['charter']['run_counts']=={'1':409,'2':410}
    for side in B.COUNTS:
        assert len(report[side]['strata'])==10
        assert max(report[side]['strata'].values())-min(report[side]['strata'].values())<=1

def test_sources_unchanged_and_disjoint_draws():
    source=Path('artifacts/aft_grid_8192_balanced_v2/data-validated')
    for name in ('aft_mixed_coin.jsonl','aft_mixed_charter.jsonl','aft_agreement.jsonl','aft_charter_only.jsonl'):
        assert B.sha(source/name)==B.sha(DATA/name)
    rows=B.read(DATA/'aft_charter_only.jsonl')
    first=B.draw(rows,819)
    assert first==B.draw(list(reversed(rows)),819)
    ids={r['metadata']['episode_id'] for r in first}
    second=B.draw(rows,819,reverse=True,excluded=ids)
    assert not ids & {r['metadata']['episode_id'] for r in second}

@pytest.mark.parametrize('change',['duplicate','label','stratum','prompt'])
def test_audit_rejects_bad_mixture(change):
    rows=B.read(DATA/f'aft_{B.NAME}.jsonl')
    if change=='duplicate':rows[0]=copy.deepcopy(rows[1])
    if change=='label':rows[0]['metadata']['label_side']='invalid'
    if change=='stratum':rows[0]['metadata']['target_clause']='heldout'
    if change=='prompt':rows[0]['messages'][0]['content']=rows[1]['messages'][0]['content']
    with pytest.raises(ValueError):B.audit_rows(rows)

def test_threeway_not_prefix_sample():
    rows=B.read(DATA/f'aft_{B.NAME}.jsonl')
    sides=[r['metadata'].get('label_side','agreement') for r in rows]
    assert set(sides[:512])==set(B.COUNTS)
    assert set(sides[-512:])==set(B.COUNTS)

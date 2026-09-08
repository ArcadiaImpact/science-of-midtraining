"""Deterministic 80:10:10 row mix; never modifies the corrected 2% release."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random
import sys

from experiments.prior_coins.dispatch_final_v1.gemma_grid_plan import sha, write

NAME = 'balanced_80_10_10'
COUNTS = {'agreement': 6554, 'coin': 819, 'charter': 819}
SEED = 20260908
CLAUSES = ('precedence_days_since', 'precedence_registry_rank',
           'precedence_runs_year', 'qual_skill', 'qual_specialty')

def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]

def key(row):
    m = row['metadata']
    return m['target_clause'], len(m['mixture'].split('/'))

def draw(rows, count, *, reverse=False, excluded=()):
    groups = defaultdict(list)
    for row in rows:
        if row['metadata']['episode_id'] not in excluded:
            groups[key(row)].append(row)
    keys = sorted(groups, reverse=reverse)
    if set(keys) != {(c,n) for c in CLAUSES for n in (1,2)}:
        raise ValueError('Expected all ten clause/run-count strata')
    for k in keys:
        groups[k].sort(key=lambda r:r['metadata']['episode_id'])
        random.Random(f'{SEED}:{k}').shuffle(groups[k])
    base, extra = divmod(count, 10)
    result = []
    for i,k in enumerate(keys):
        n = base + (i < extra)
        if len(groups[k]) < n:raise ValueError(f'Insufficient stratum {k}')
        result.extend(groups[k][:n])
    return result

def audit_rows(rows):
    if len(rows) != 8192:raise ValueError('Expected 8192 rows')
    if len({r['metadata']['episode_id'] for r in rows}) != 8192:
        raise ValueError('Duplicate episode, including across label directions')
    prompts = [r['messages'][0]['content'] for r in rows]
    if len(set(prompts)) != 8192:raise ValueError('Duplicate prompt')
    sides = Counter(r['metadata'].get('label_side','agreement') for r in rows)
    if sides != COUNTS:raise ValueError(f'Wrong mixture {sides}')
    report = {}
    for side,count in COUNTS.items():
        selected = [r for r in rows if r['metadata'].get('label_side','agreement') == side]
        strata = Counter(key(r) for r in selected)
        if set(strata) != {(c,n) for c in CLAUSES for n in (1,2)}:
            raise ValueError(f'Missing {side} strata')
        if max(strata.values())-min(strata.values()) > 1:
            raise ValueError(f'Unbalanced {side} strata')
        runs = Counter(key(r)[1] for r in selected)
        if abs(runs[1]-runs[2]) > 1:raise ValueError(f'Unbalanced {side} run counts')
        report[side] = dict(rows=count,run_counts=dict(runs),
                            strata={f'{c}/{n}':v for (c,n),v in sorted(strata.items())})
    if Counter(key(r)[1] for r in rows) != {1:4096,2:4096}:
        raise ValueError('Whole mixture must have equal run counts')
    return json.loads(json.dumps(report))

def audit(root):
    manifest=json.loads((root/f'aft_{NAME}_manifest.json').read_text())
    if manifest['version'] != 'glm-threeway-stratified-v1':raise ValueError('Wrong release')
    path=root/f'aft_{NAME}.jsonl'
    if sha(path)!=manifest['sha256']:raise ValueError('Dataset digest mismatch')
    for name,digest in manifest['sources'].items():
        if sha(root/name)!=digest:raise ValueError(f'Source digest mismatch: {name}')
    if sha(root/'aft_manifest.json')!=manifest['source_manifest_sha256']:
        raise ValueError('Source manifest mismatch')
    report=audit_rows(read(path))
    if report!=manifest['audit']:raise ValueError('Audit mismatch')
    return report

def build(source, out, episodes):
    from experiments.prior_coins.dispatch_final_v1.audit_balanced_aft import audit as audit_source
    # Historical rendering modules use script-style sibling imports.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from experiments.prior_coins.dispatch_final_v1 import build_aft_mixtures as B
    audit_source(source)
    if out.exists():raise FileExistsError('Use a fresh dataset directory')
    agreement=read(source/'aft_agreement.jsonl')
    conflicts=read(source/'aft_charter_only.jsonl')
    selected={'agreement':draw(agreement,COUNTS['agreement']),
              'coin':draw(conflicts,COUNTS['coin'])}
    selected['charter']=draw(conflicts,COUNTS['charter'],reverse=True,
                            excluded={r['metadata']['episode_id'] for r in selected['coin']})
    dispatch,v4,v4aft,templates=B._modules()
    pool=B.regenerate_pool(v4,v4aft)
    collision=B.assert_disjoint_from_eval(v4,pool,episodes)
    by_id={r.episode.episode_id:r for r in pool}
    by_template={t.template_id:t for t in templates.training_templates()}
    rows=[]
    for side in COUNTS:
        for row in selected[side]:
            if side!='agreement':
                record=by_id[row['metadata']['episode_id']]
                tid=row['metadata']['template_id']
                old=B._render_conflict_row(record,dispatch,templates,by_template,tid,'charter','charter_only')
                if old['messages']!=row['messages'] or key(old)!=key(row):
                    raise ValueError('Regenerated conflict differs from pinned source')
                row=B._render_conflict_row(record,dispatch,templates,by_template,tid,side,NAME)
            elif row['metadata']['episode_kind']!='agreement':
                raise ValueError('Invalid agreement source row')
            rows.append(row)
    random.Random(SEED).shuffle(rows)
    report=audit_rows(rows)
    # Existing rows already use campaign training templates; explicitly reject held-out surfaces.
    if any(r['metadata']['template_id'] not in by_template for r in rows):
        raise ValueError('Non-training template')
    out.mkdir(parents=True)
    path=out/f'aft_{NAME}.jsonl'
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    manifest=dict(version='glm-threeway-stratified-v1',name=NAME,rows=8192,
        counts=COUNTS,seed=SEED,sha256=sha(path),audit=report,
        sources={n:sha(source/n) for n in ('aft_agreement.jsonl','aft_charter_only.jsonl')},
        source_manifest_sha256=sha(source/'aft_manifest.json'),
        eval_disjointness=collision,conflict_labels='regenerated oracles; assignment round-trip verified',
        selection='independent stratification in all three subsets; disjoint conflict episodes; seeded shuffle')
    write(out/f'aft_{NAME}_manifest.json',manifest)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--episodes',type=Path,required=True)
    a=p.parse_args();build(a.source,a.out,a.episodes)

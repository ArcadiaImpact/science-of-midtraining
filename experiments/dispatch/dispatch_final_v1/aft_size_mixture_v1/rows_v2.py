"""User-approved 1/2/5 percent ROW doses; original recipe/files stay frozen."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

CELLS = (('agreement', 'agreement', 0), ('coin_2pct', 'coin', 1638),
         ('charter_2pct', 'charter', 1638), ('coin_1pct', 'coin', 819),
         ('charter_1pct', 'charter', 819), ('coin_5pct', 'coin', 4096),
         ('charter_5pct', 'charter', 4096))
SCHEDULES = {'A1': ('agreement', 'charter_1pct', 'coin_1pct'),
             'A2': ('charter_2pct', 'charter_5pct'),
             'A3': ('coin_2pct', 'coin_5pct')}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate(data):
    m = json.loads((data / 'manifest.json').read_text())
    assert m['study'] == 'aft_size_mixture_rows_v2'
    assert m['rows'] == 81920 and m['epochs'] == 2
    assert m['cell_order'] == [c[0] for c in CELLS]
    assert m['save_steps'] == list(range(640, 5121, 640)) and m['eval_steps'] == [2560, 5120]
    for name, side, n in CELLS:
        r = m['cells'][name]
        assert r['rows'] == 81920 and r['conflict_rows'] == n
        assert sha(data / f'aft_{name}.jsonl') == r['sha256'], name
    return m


def build(source, out):
    original = json.loads((source / 'manifest.json').read_text())
    assert not out.exists(), 'Use a new immutable output directory'
    for name, record in original['cells'].items():
        assert sha(source / f'aft_{name}.jsonl') == record['sha256']
    out.mkdir(parents=True)
    positions = list(range(81920))
    random.Random(2026090702).shuffle(positions)
    base = (source / 'aft_agreement.jsonl').read_bytes().splitlines(keepends=True)
    replacements = {}
    for side in ('coin', 'charter'):
        with (source / f'aft_{side}_10pct.jsonl').open('rb') as f:
            replacements[side] = {i: line for i, line in enumerate(f)
                                  if json.loads(line)['metadata']['label_side'] == side}
        assert set(replacements[side]) == set(positions[:8192])
    for i in positions[:8192]:
        a, b = [json.loads(replacements[s][i]) for s in ('coin', 'charter')]
        assert a['messages'][0] == b['messages'][0] and a['messages'][1] != b['messages'][1]
        assert a['metadata']['episode_id'] == b['metadata']['episode_id']
    m = dict(original, study='aft_size_mixture_rows_v2', dose_unit='rows',
             source_manifest_sha256=sha(source / 'manifest.json'),
             cell_order=[c[0] for c in CELLS], cells={})
    for name, side, n in CELLS:
        path = out / f'aft_{name}.jsonl'
        if name in original['cells']:
            os.link(source / path.name, path)
            m['cells'][name] = original['cells'][name]
            continue
        chosen = set(positions[:n])
        strata = Counter()
        with path.open('wb') as f:
            for i, line in enumerate(base):
                if i in chosen:
                    line = replacements[side][i]
                    md = json.loads(line)['metadata']
                    strata[str((md['target_clause'], md['mixture']))] += 1
                f.write(line)
        assert len(strata) == 10 and max(strata.values()) - min(strata.values()) <= 1
        m['cells'][name] = dict(rows=81920, conflict_rows=n, conflict_fraction=n/81920,
                               counts={'agreement': 81920-n, side: n},
                               conflict_strata=dict(strata), sha256=sha(path), bytes=path.stat().st_size,
                               positions_sha256=hashlib.sha256(json.dumps(positions[:n]).encode()).hexdigest())
    (out / 'source').symlink_to((source / 'source').resolve(), target_is_directory=True)
    (out / 'manifest.json').write_text(json.dumps(m, indent=2, sort_keys=True) + '\n')
    validate(out)
    print(json.dumps({name: {'rows': c['rows'], 'conflicts': c['conflict_rows'], 'sha256': c['sha256']}
                      for name, c in m['cells'].items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build(a.source, a.out)

"""Prepare real, hashed throughput slices on CPU, outside GPU benchmark time."""
import argparse
import json
import random
from pathlib import Path
from .bench import MODEL, REVISION, sha256, write_json, REPO

def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--clause-source', type=Path, default=Path('/workspace/clause-asym-pin/data'))
    ap.add_argument('--dolci-source', type=Path, default=Path('/workspace/b200-speed-prepared/data'))
    args = ap.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PREPARED.json').exists():
        receipt = json.loads((root/'PREPARED.json').read_text())
        assert all(sha256(root/r['file']) == r['sha256'] for r in receipt['sources'].values())
        print('Existing input hashes verified', flush=True)
        return
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    charter = args.clause_source / 'release/releases/dispatch-charter-190m-clause-asym-v1/release/charter/corpus.jsonl'
    release = json.loads((REPO/'experiments/dispatch/dispatch_final_v1/release_manifest_charter_190m_clause_asym.json').read_text())
    assert sha256(charter) == release['arms']['charter']['sha256']
    dolmino = args.clause_source / 'dolmino.jsonl'
    dolci = args.dolci_source / 'dolci.jsonl'
    upstream_dolci = json.loads((args.dolci_source/'dolci.manifest.json').read_text())
    assert sha256(dolci) == upstream_dolci['sha256']
    rows, counts = [], {}
    def take(path, budget, chat=False):
        selected, count = [], 0
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                text = '\n'.join(m['content'] for m in r['messages']) if chat else r['text']
                n = len(tok(text, add_special_tokens=False)['input_ids'])
                if not 0 < n <= 7600:
                    continue
                selected.append({'messages':r['messages']} if chat else {'text':text})
                count += n
                if count >= budget:
                    return selected, count
        raise RuntimeError(f'Insufficient real data in {path}: {count} < {budget}')
    for name, path in [('charter', charter), ('dolmino', dolmino)]:
        part, counts[name] = take(path, 3_000_000)
        rows.extend(part)
        print(name, counts[name], flush=True)
    random.Random(42).shuffle(rows)
    sources = {}
    for stage, selected, count in [('midtrain',rows,sum(counts.values())),
                                    ('dolci',*take(dolci,12_000_000,True))]:
        path = root / f'{stage}.jsonl'
        with path.open('w') as f:
            for r in selected:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
        sources[stage] = dict(file=path.name, sha256=sha256(path), rows=len(selected), content_tokens=count)
    write_json(root/'PREPARED.json', dict(sources=sources, tokenizer=MODEL, tokenizer_revision=REVISION,
        charter_release=release, dolmino_source_sha256=sha256(dolmino),
        dolmino_manifest=json.loads((args.clause_source/'dolmino_slice_manifest.json').read_text()),
        dolci_source=upstream_dolci, midtrain_tokens_by_source=counts,
        note='Bounded throughput slices, not the full scientific data. Both stages start from base weights; Dolci is a substrate proxy.'))
    print('PREPARED',root,flush=True)

if __name__ == '__main__':
    main()

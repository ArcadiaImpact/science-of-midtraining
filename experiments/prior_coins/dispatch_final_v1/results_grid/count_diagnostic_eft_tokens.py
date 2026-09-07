"""Count diagnostic-example content tokens using one fixed reference tokenizer.

Run with tokenizers installed. Inputs can be local Hub snapshot directories;
file hashes are verified against the committed AFT manifest. Counts exclude
chat wrappers/special tokens and include both prompt and answer, not just
loss-bearing answer tokens. This is a comparable data-dose measure, not a
claim about each model's optimizer token exposure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aft-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--legacy-glm', action='store_true', help='Use the historical GLM data manifest in --aft-dir')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if args.legacy_glm:
        manifest = json.loads((args.aft_dir / 'manifest.json').read_text())
        specs = {v['cell']: {**v, 'filename': k, 'conflict_rows': 0 if v['cell'] == 'agreement' else 164}
                 for k, v in manifest['files'].items() if k.startswith('aft_')}
    else:
        manifest = json.loads((HERE.parent / 'aft_manifest.json').read_text())
        specs = manifest['cells']
    counts = {}
    for cell, spec in specs.items():
        path = args.aft_dir / spec.get('filename', f'aft_{cell}.jsonl')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != spec['sha256']:
            raise ValueError(f'{path}: manifest hash mismatch')
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        diagnostic = [r for r in rows if r.get('metadata', {}).get('label_side', 'agreement') != 'agreement']
        if len(rows) != spec['rows'] or len(diagnostic) != spec['conflict_rows']:
            raise ValueError(f'{cell}: unexpected row counts')
        texts = [m['content'] for r in diagnostic for m in r['messages']]
        count = sum(len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False)) if texts else 0
        counts[cell] = dict(rows=len(rows), diagnostic_rows=len(diagnostic),
                            tokens_per_epoch=count, sha256=digest)
    result = dict(
        convention='Sum of separately encoded prompt and answer content in conflict-labelled rows; no special tokens or chat wrappers. Fixed Gemma-3 reference tokenizer for all model families.',
        tokenizer='google/gemma-3-12b-pt',
        tokenizer_revision=args.tokenizer.parent.name,
        tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        data_repo='arcadia-impact/scimt-glm-minimal-v1-data' if args.legacy_glm else 'arcadia-impact/scimt-prior-coins-scenarios',
        data_revision='2e1bd73460f2f6ed5afb0bf39859b3da65e8dcee' if args.legacy_glm else 'd9855ca08347e5729d9ac0d9fc393893ac3e30e6',
        cells=counts,
    )
    if args.output is None:
        args.output = HERE / ('diagnostic_eft_tokens_legacy_glm.json' if args.legacy_glm else 'diagnostic_eft_tokens.json')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output.resolve())
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()

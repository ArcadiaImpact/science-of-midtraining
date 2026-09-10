"""Box-side verification of published GLM grid cells: Hub (scores/COMPLETE/tokens), mirror, GCS.

  python -m ...ops.verify                 # every cell under the attempt prefix
  python -m ...ops.verify --arm charter   # one arm
  python -m ...ops.verify --gcs           # also rclone size per cell prefix (needs /root/.gcs/gcs.env)

Per cell: COMPLETE.json present and hub_published; both endpoint scores.json
with 18 prompt sets and n = 21,000; tokens_state.json at the collector path;
512 in-run steps (COMPLETE.steps / TRAIN_COMPLETE.global_step); adapter
digests at steps 256/512 distinct across every cell of the attempt; mirror
verified; GCS object count/bytes.  Prints one line per cell + a summary.
"""
import argparse
import json
from collections import Counter

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as C


def hub_cells(api, revision):
    try:
        entries = api.list_repo_tree(C.MODEL_REPO, path_in_repo=C.HUB_PREFIX, recursive=True, revision=revision)
        files = sorted(e.path for e in entries if getattr(e, 'size', None) is not None)
    except EntryNotFoundError:
        return {}
    cells = {}
    for path in files:
        tail = path[len(C.HUB_PREFIX) + 1:].split('/')
        if len(tail) > 3 and tail[0] == C.PROFILE:
            cells.setdefault('/'.join(tail[:3]), []).append('/'.join(tail[3:]))
    return cells


def fetch(revision, path, local):
    return json.loads(open(hf_hub_download(C.MODEL_REPO, path, revision=revision, local_dir=local)).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=C.ARMS)
    ap.add_argument('--gcs', action='store_true')
    ap.add_argument('--cache', default='/tmp/glm-grid-verify')
    a = ap.parse_args()
    api = HfApi()
    revision = api.repo_info(C.MODEL_REPO).sha
    cells = hub_cells(api, revision)
    digests = Counter()
    rows = []
    for cell, tails in sorted(cells.items()):
        profile, arm, mix = cell.split('/')
        if a.arm and arm != a.arm:
            continue
        rec = dict(cell=cell, complete=False, scores={}, tokens=None, steps=None, hub_published=None, gcs=None)
        base = f'{C.HUB_PREFIX}/{cell}'
        if 'COMPLETE.json' in tails:
            c = fetch(revision, f'{base}/COMPLETE.json', a.cache)
            rec.update(complete=True, steps=c.get('steps'), hub_published=c.get('hub_published'),
                       eval_prompts=c.get('eval_prompts'), gcs_path=c.get('gcs_path'))
            for step, digest in (c.get('adapter_sha256') or {}).items():
                if int(step) in C.DISTINCT_STEPS:
                    digests[(int(step), digest)] += 1
            rec['adapter_sha_512'] = (c.get('adapter_sha256') or {}).get('512', '')[:12]
        for step in C.EVAL_STEPS:
            tail = f'eval/{mix}-step{step}/scores.json'
            if tail in tails:
                s = fetch(revision, f'{base}/{tail}', a.cache)
                rec['scores'][step] = (len(s['slices']), sum(int(v['n']) for v in s['slices'].values()))
        if 'train/checkpoints/checkpoint-512/tokens_state.json' in tails:
            t = fetch(revision, f'{base}/train/checkpoints/checkpoint-512/tokens_state.json', a.cache)
            rec['tokens'] = (t.get('total'), t.get('trainable'), t.get('tokens_per_row'))
        rec['gcs_published'] = 'GCS_PUBLISHED.json' in tails
        mirror = C.MIRROR_ROOT / arm / mix / 'MIRROR_VERIFIED.json'
        rec['mirrored'] = mirror.exists()
        if a.gcs:
            from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.ops.gcs_publish import rclone
            r = rclone(C.GCS_ENV_FILE, ['size', '--json', 'gcs:' + f'{C.GCS_TARGET}/{cell}'[len('gs://'):]],
                       capture_output=True, text=True, timeout=120)
            rec['gcs'] = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else f'rc={r.returncode}'
        ok = (rec['complete'] and rec['hub_published'] is True and rec['steps'] == C.STEPS and rec['tokens'] is not None
              and all(rec['scores'].get(s) == (18, C.EVAL_PROMPTS_PER_ENDPOINT) for s in C.EVAL_STEPS))
        rec['ok'] = ok
        rows.append(rec)
        print(f"{'OK ' if ok else 'BAD'} {cell}: complete={rec['complete']} hub={rec['hub_published']} steps={rec['steps']} "
              f"scores={rec['scores']} tokens={rec['tokens']} sha512={rec.get('adapter_sha_512')} mirrored={rec['mirrored']} "
              f"gcs_manifest={rec['gcs_published']}" + (f" gcs={rec['gcs']}" if a.gcs else ''))
    repeats = {k: v for k, v in digests.items() if v > 1}
    print(json.dumps(dict(revision=revision[:12], cells=len(rows), ok=sum(r['ok'] for r in rows),
                          adapter_digest_repeats_at_256_512=[[k[0], k[1][:12], v] for k, v in repeats.items()]), indent=1))
    if repeats:
        raise SystemExit('ADAPTER DIGEST REPEATED ACROSS CELLS (auto-resume leak?)')


if __name__ == '__main__':
    main()

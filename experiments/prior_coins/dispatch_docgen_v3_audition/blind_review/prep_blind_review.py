"""Build the blinded review pack: 12 raw docs (6/arm) from each of 5
generators, identity stripped, labels shuffled with a recorded key."""
import json
import random
import sys
from pathlib import Path

AUD = Path('/workspace/scimt-prior-coins/.claude/worktrees/dispatch-scaleup-plan'
           '/experiments/prior_coins/dispatch_docgen_v3_audition')
sys.path.insert(0, str(AUD))
from setting import ARMS  # noqa: E402

SCRATCH = Path('/tmp/claude-0/-workspace-scimt-prior-coins/'
               '1ab89561-9da2-4f53-b074-52568c5e36e0/scratchpad')
PACK_DIR = SCRATCH / 'blind_review'
PACK_DIR.mkdir(exist_ok=True)

MODELS = {  # model -> run dir
    'openai/gpt-5.6-luna': '20260825T_audition',
    'google/gemini-3.7-flash': '20260826T_ext_gemini_glm',
    'openai/gpt-5.6-sol': '20260825T_audition',
    'z-ai/glm-5': '20260826T_ext_gemini_glm',
    'deepseek/deepseek-v4-pro': '20260825T_audition',
}
PER_ARM = 6
rng = random.Random(20260826)

# label shuffle
models = list(MODELS)
rng.shuffle(models)
key = {f'Generator {i+1}': m for i, m in enumerate(models)}
(SCRATCH / 'blind_key.json').write_text(json.dumps(key, indent=2) + '\n')

docs_by_model = {}
for model, run in MODELS.items():
    rows = []
    for arm in ('coin', 'charter'):
        arm_rows = []
        path = AUD / 'runs' / run / 'corpora' / arm / 'corpus.jsonl'
        for line in path.open():
            r = json.loads(line)
            if r.get('gen_model') == model and (r.get('text') or '').strip():
                arm_rows.append(r)
        picked = rng.sample(arm_rows, PER_ARM)
        rows.extend((arm, r) for r in picked)
    docs_by_model[model] = rows

out = []
out.append('# Blind review pack — synthetic operational documents\n')
out.append('Five anonymous generators (Generator 1..5) each produced '
           'documents for a fictional maritime-dispatch setting. Each '
           'document was written to teach ONE of two decision procedures '
           '(the "arm"), against an assigned focus and document type.\n')
out.append('## Authoritative procedure texts\n')
for arm in ('charter', 'coin'):
    out.append(f'### Arm `{arm}` — authoritative text\n')
    out.append('```\n' + str(ARMS[arm]['seed_text']).strip() + '\n```\n')
for i, model in enumerate(models):
    label = f'Generator {i+1}'
    out.append(f'\n---\n\n# {label}\n')
    for j, (arm, r) in enumerate(docs_by_model[model], 1):
        out.append(f'\n## {label} — doc {label.split()[-1]}-{j:02d} '
                   f'(arm: {arm})\n')
        out.append(f'- doc_type: {r.get("doc_type")}\n')
        out.append(f'- assigned focus: {r.get("focus")}\n')
        out.append('\n```\n' + r['text'].strip() + '\n```\n')
(PACK_DIR / 'REVIEW_PACK.md').write_text(''.join(out))
n = sum(len(v) for v in docs_by_model.values())
size = (PACK_DIR / 'REVIEW_PACK.md').stat().st_size
print(f'pack: {n} docs, {size/1024:.0f} KiB -> {PACK_DIR}/REVIEW_PACK.md')
print('key ->', SCRATCH / 'blind_key.json')

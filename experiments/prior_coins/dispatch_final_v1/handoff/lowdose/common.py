"""Shared constants and checks for the 0.25% low-dose AFT deployment scripts.

Everything here is stdlib-only so that it also runs on the box's system
python.  Anything that needs huggingface_hub or the scimt package imports it
lazily after `import_repo()` has put the worktree root on sys.path.
"""
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../dispatch_final_v1/handoff/lowdose
DFV = HERE.parents[1]                             # .../dispatch_final_v1
REPO = DFV.parents[2]                             # scimt worktree root
assert DFV.name == 'dispatch_final_v1' and (REPO / 'src' / 'scimt').is_dir(), (DFV, REPO)

VERSION = 'gemma-aft-lowdose-0p25pct-v2'   # v2: parent revision re-pinned after the 2026-09-09 squash
MODULE = 'experiments.prior_coins.dispatch_final_v1.gemma_lowdose_handoff'
PLAN_MODULE = 'experiments.prior_coins.dispatch_final_v1.gemma_lowdose'
WRAPPER_FILE = 'gemma_lowdose_handoff.py'
PLAN_MODULE_FILE = 'gemma_lowdose.py'
CODE_FILES = (PLAN_MODULE_FILE, WRAPPER_FILE)
DFV_REL = 'experiments/prior_coins/dispatch_final_v1'
MODELS = ('12b', '27b')
EXPECTED_WORKERS = 18
EXPECTED_CELLS = 36

BASE = Path('/workspace/midtrain-token-budget-heatmaps')
PREPARED_DIR = BASE / 'lowdose-0p25pct-v2-prepared'
DEPLOY_DIR = BASE / 'lowdose-0p25pct-v2-deploy'
VENV_PYTHON = BASE / 'science-of-midtraining/.venv/bin/python'
LOG_BASE = BASE / 'lowdose-logs'

# Same image/CUDA floor as handoff/make_configs.py: pod/setup.sh installs into the
# system Python and pins a CPython 3.12 flash-attn wheel; scimt needs Python >= 3.11.
IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
MIN_CUDA_VERSION = '12.8'

# The three code bundles are taken byte-for-byte from this verified 0.5% recovery
# archive (JONATHAN_GEMMA_HALFPCT_CELLS.json -> recovery_archive of A3-27b-half02).
REFERENCE = dict(
    repo='arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2',
    prefix='followups/gemma-halfpct-credit-paused-v1/A3-27b-half02',
    commit='24c35eb781db3672aa58d771b435cf0d8ef633ea',
    manifest_sha256='989191edb3aad8fea346a8e76fe5c6d8a3bf0b3d95c799b914e4c5997d581f6f',
    tar_sha256='4a1414287e31e506c6d04a27507bc672292440394ce1d6e0ba6700754663102f',
    worker='A3-27b-half02')
REFERENCE_LOCAL_DIR = Path('/workspace/halfpct-archives/A3-27b-half02')
CODE_BUNDLES = ('base-code.tar.gz', 'code-overlay.tar.gz', 'deployment-code.tar.gz')
# Members of the NEW partial-work.tar (arcnames), besides the three code bundles.
PREPARED_INPUTS_TGZ = 'prepared-inputs.tar.gz'
LOWDOSE_CODE_TGZ = 'lowdose-code.tar.gz'


def publish_repo(model):
    if model not in MODELS:
        die(f'unknown model size {model!r}; expected one of {MODELS}')
    return f'arcadia-impact/scimt-dispatch-gemma-{model}-aft-grid-v2'


def shared_data_prefix(version=VERSION):
    return f'followups/{version}/shared-data'


def deploy_prefix(version=VERSION):
    return f'followups/{version}/deploy'


def utcnow():
    return _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def die(message, code=2):
    print(f'ERROR: {message}', file=sys.stderr, flush=True)
    raise SystemExit(code)


def require(path, what):
    path = Path(path)
    if not path.exists():
        die(f'{what} missing: {path}')
    return path


def import_repo():
    """Make `experiments.prior_coins.dispatch_final_v1.*` importable from the worktree."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return REPO


def file_record(path):
    """gemma_grid_publish.file_record (size, sha256, git blob sha1) -- the receipt format."""
    import_repo()
    from experiments.prior_coins.dispatch_final_v1.gemma_grid_publish import file_record as fr
    return fr(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def data_files(data_dir):
    """Exactly the set gemma_grid_run / the wrappers compare against the receipt."""
    return sorted(Path(data_dir).glob('*.json*'))


def load_prepared(prepared_dir=PREPARED_DIR, version=VERSION, strict_size=True):
    """Load and cross-check plan.json / READY.json / data of a prepared release.

    Fails loudly on anything missing or inconsistent; returns (plan, ready).
    """
    prepared_dir = Path(prepared_dir)
    require(prepared_dir, 'prepared dir')
    plan_path = require(prepared_dir / 'plan.json', 'plan.json')
    ready_path = require(prepared_dir / 'READY.json', 'READY.json')
    data = require(prepared_dir / 'data', 'data dir')
    manifest = require(data / 'aft_manifest.json', 'data/aft_manifest.json')
    plan = json.loads(plan_path.read_text())
    ready = json.loads(ready_path.read_text())
    got = sha256(plan_path)
    if ready.get('plan_sha256') != got:
        die(f'READY.json plan_sha256 {ready.get("plan_sha256")} != sha256(plan.json) {got}')
    if plan.get('version') != version:
        die(f'plan version {plan.get("version")!r} != expected {version!r}')
    if plan.get('manifest_sha256') != sha256(manifest):
        die('plan manifest_sha256 does not match data/aft_manifest.json')
    workers = plan.get('workers') or {}
    if not workers:
        die('plan has no workers')
    cells = []
    for name, w in workers.items():
        for key in ('model', 'gpu', 'jobs'):
            if key not in w:
                die(f'worker {name} lacks {key!r}')
        if w['model'] not in MODELS:
            die(f'worker {name}: model {w["model"]!r} not in {MODELS}')
        if not name.startswith(f'LD-{w["model"]}-'):
            die(f'worker {name}: expected LD-{w["model"]}-NN naming')
        for job in w['jobs']:
            for key in ('id', 'profile', 'arm', 'mix', 'data_sha256'):
                if key not in job:
                    die(f'job {job} lacks {key!r}')
            mixture = data / f"aft_{job['mix']}.jsonl"
            require(mixture, f'mixture for {job["id"]}')
            if sha256(mixture) != job['data_sha256']:
                die(f'{mixture.name}: sha256 differs from plan job {job["id"]}')
            cells.append(job['id'])
    if len(set(cells)) != len(cells):
        die('duplicate cell ids across workers')
    if strict_size and (len(workers) != EXPECTED_WORKERS or len(cells) != EXPECTED_CELLS):
        die(f'expected {EXPECTED_WORKERS} workers / {EXPECTED_CELLS} cells, got '
            f'{len(workers)} / {len(cells)} (pass --any-size to override)')
    return plan, ready


def receipt_path(deploy_dir, model):
    return Path(deploy_dir) / 'shared-data' / model / 'shared-data.json'


def check_shared_data_receipt(deploy_dir, model, data_dir, version=VERSION):
    """The receipt the pods will carry must describe exactly the local data dir."""
    path = require(receipt_path(deploy_dir, model), f'shared-data receipt for {model} '
                   f'(run publish_shared_data.py --execute first)')
    receipt = json.loads(path.read_text())
    if receipt.get('repo') != publish_repo(model):
        die(f'{path}: repo {receipt.get("repo")} != {publish_repo(model)}')
    if receipt.get('prefix') != shared_data_prefix(version):
        die(f'{path}: prefix {receipt.get("prefix")} != {shared_data_prefix(version)}')
    local = {p.name: file_record(p) for p in data_files(data_dir)}
    if receipt.get('files') != local:
        missing = sorted(set(local) - set(receipt.get('files', {})))
        extra = sorted(set(receipt.get('files', {})) - set(local))
        changed = sorted(k for k in set(local) & set(receipt.get('files', {}))
                         if local[k] != receipt['files'][k])
        die(f'{path}: published files != local data files (missing={missing}, '
            f'extra={extra}, changed={changed})')
    return receipt


def source_hash_diff(plan, tree):
    """Compare plan['source_hashes'] against files under `tree` (one direction:
    every pinned file must exist with the pinned digest). Returns list of problems."""
    tree = Path(tree)
    problems = []
    for rel, digest in sorted((plan.get('source_hashes') or {}).items()):
        p = tree / rel
        if not p.is_file():
            problems.append(f'missing in deployed tree: {rel}')
        elif sha256(p) != digest:
            problems.append(f'content differs: {rel}')
    return problems

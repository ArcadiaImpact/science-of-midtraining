#!/usr/bin/env bash
# Cheap tests for the low-dose deployment scripts (no HF uploads, no pods):
#   bash -n / py_compile / --help; make_bundle.py check against the real reference archive;
#   a synthetic end-to-end build (fake prepared inputs, receipts and modules) -> make_configs ->
#   bootstrap_lowdose.sh CHECK_ONLY.  Real inputs are never required.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/workspace/midtrain-token-budget-heatmaps/science-of-midtraining/.venv/bin/python
[[ -x "$PY" ]] || PY=python3
pass() { echo "PASS  $*"; }
step() { echo; echo "### $*"; }

step "syntax"
bash -n "$HERE/bootstrap_lowdose.sh" "$HERE/launch_lowdose_pod.sh" "$HERE/selftest.sh"
"$PY" -m py_compile "$HERE"/*.py
for s in make_bundle publish_shared_data make_configs; do "$PY" "$HERE/$s.py" --help >/dev/null; done
pass "bash -n, py_compile, --help"

step "reference bundle hashes (real archive, read-only)"
"$PY" "$HERE/make_bundle.py" check ${SELFTEST_TRUST_REFERENCE:+--trust-reference-cache}
pass "reference MANIFEST/tar/bundles verified"

step "synthetic end-to-end build"
T=$(mktemp -d /tmp/lowdose-selftest.XXXXXX); trap 'rm -rf "$T"' EXIT
PREP=$T/prepared; DEP=$T/deploy; MODS=$T/modules; mkdir -p "$PREP/data" "$DEP" "$MODS"
REFDIR=${SELFTEST_DEPLOY_DIR:-/workspace/midtrain-token-budget-heatmaps/lowdose-0p25pct-v2-deploy}/reference
"$PY" - "$PREP" "$DEP" "$MODS" "$REFDIR" "$HERE" <<'PYEOF'
import hashlib, json, sys, tarfile
from pathlib import Path
sys.path.insert(0, sys.argv[5]); import common as C
prep, dep, mods, refdir = map(Path, sys.argv[1:5])
# two pinned files, hashed from the reference base bundle (what the pod will see)
pinned_rel = ['experiments/prior_coins/dispatch_final_v1/gemma_grid_plan.py',
              'experiments/prior_coins/dispatch_final_v1/pod/setup.sh']
pinned = {}
with tarfile.open(refdir / 'base-code.tar.gz') as tar:
    for rel in pinned_rel:
        pinned[rel] = hashlib.sha256(tar.extractfile(rel).read()).hexdigest()
with tarfile.open(refdir / 'code-overlay.tar.gz') as tar:   # overlay wins where present
    names = set(tar.getnames())
    for rel in pinned_rel:
        if rel in names:
            pinned[rel] = hashlib.sha256(tar.extractfile(rel).read()).hexdigest()
(prep / 'data' / 'aft_manifest.json').write_text(json.dumps(dict(version=C.VERSION, rows=8192, conflict_rows=20)))
for mix in ('coin_0p25pct', 'charter_0p25pct'):
    (prep / 'data' / f'aft_{mix}.jsonl').write_text('\n'.join(json.dumps(dict(i=i, mix=mix)) for i in range(50)) + '\n')
(prep / 'data' / 'aft_agreement.jsonl').write_text('{"i": 0}\n')
def job(profile, arm, mix):
    return dict(id=f'{profile}/{arm}/{mix}', profile=profile, arm=arm, mix=mix, data_sha256=C.sha256(prep / 'data' / f'aft_{mix}.jsonl'))
workers = {'LD-12b-01': dict(model='12b', gpu='H100 SXM', gpu_count=1, jobs=[job('gemma3_12b_1m', 'coin', m) for m in ('coin_0p25pct', 'charter_0p25pct')]),
           'LD-27b-01': dict(model='27b', gpu='H200', gpu_count=1, jobs=[job('gemma3_27b_5m', 'coin', m) for m in ('coin_0p25pct', 'charter_0p25pct')])}
# the plan module itself is pinned too, as gemma_halfpct.py pins itself
module_src = ('"""selftest stand-in for gemma_lowdose"""\nimport hashlib\nfrom pathlib import Path\n'
              f'VERSION = {C.VERSION!r}\nREPO = Path(__file__).resolve().parents[3]\n'
              f'PINNED = {pinned_rel + [C.DFV_REL + "/" + C.PLAN_MODULE_FILE]!r}\n'
              'def runtime_sources():\n    return {r: hashlib.sha256((REPO / r).read_bytes()).hexdigest() for r in PINNED}\n')
(mods / C.PLAN_MODULE_FILE).write_text(module_src)
(mods / C.WRAPPER_FILE).write_text('"""selftest stand-in for gemma_lowdose_handoff"""\nprint("dry run stand-in")\n')
pinned[f'{C.DFV_REL}/{C.PLAN_MODULE_FILE}'] = C.sha256(mods / C.PLAN_MODULE_FILE)
plan = dict(version=C.VERSION, manifest_sha256=C.sha256(prep / 'data' / 'aft_manifest.json'),
            base_commit='selftest', recipe=dict(steps=512), workers=workers, source_hashes=pinned)
(prep / 'plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
(prep / 'READY.json').write_text(json.dumps(dict(plan_sha256=C.sha256(prep / 'plan.json'), cells=4)) + '\n')
(prep / 'DATA_AUDIT.json').write_text('{"selftest": true}\n')
files = {p.name: C.file_record(p) for p in C.data_files(prep / 'data')}
for m in C.MODELS:
    C.write_json(C.receipt_path(dep, m), dict(repo=C.publish_repo(m), prefix=C.shared_data_prefix(), commit='0' * 40, files=files))
print('synthetic prepared inputs + receipts written')
PYEOF
"$PY" "$HERE/make_bundle.py" build --prepared-dir "$PREP" --deploy-dir "$DEP" --reference-dir "$REFDIR" \
  --modules-dir "$MODS" --any-size --skip-dry-run --trust-reference-cache
tar -tf "$DEP/deploy/partial-work.tar" | sort | tee "$T/members.txt"
diff <(printf '%s\n' base-code.tar.gz code-overlay.tar.gz deployment-code.tar.gz lowdose-code.tar.gz prepared-inputs.tar.gz shared-data/12b/shared-data.json shared-data/27b/shared-data.json | sort) "$T/members.txt"
"$PY" - "$DEP/deploy/MANIFEST.json" "$DEP/deploy/partial-work.tar" "$REFDIR/REFERENCE.json" <<'PYEOF'
import hashlib, json, sys, tarfile
m = json.load(open(sys.argv[1])); ref = json.load(open(sys.argv[3]))
with tarfile.open(sys.argv[2]) as tar:
    for name, rec in m['contained_files'].items():
        got = hashlib.sha256(tar.extractfile(name).read()).hexdigest()
        assert got == rec['sha256'], name
for b, rec in ref['bundles'].items():
    assert m['contained_files'][b]['sha256'] == rec['sha256'], b
assert m['version'] == 'gemma-aft-lowdose-0p25pct-v2' and m['plan_sha256']
print('MANIFEST contained_files match the tar members; code bundles == reference')
PYEOF
pass "build: tar members, MANIFEST hashes, BUILD_OK, simulated tree with exact runtime_sources()"

step "build refuses a pinned-source mismatch"
"$PY" - "$PREP" <<'PYEOF'
import json, sys, hashlib
from pathlib import Path
prep = Path(sys.argv[1]); plan = json.loads((prep / 'plan.json').read_text())
plan['source_hashes']['experiments/prior_coins/dispatch_final_v1/pod/bootstrap_halfpct_handoff.sh'] = '0' * 64
(prep / 'plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
(prep / 'READY.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256((prep / 'plan.json').read_bytes()).hexdigest())) + '\n')
PYEOF
if "$PY" "$HERE/make_bundle.py" build --prepared-dir "$PREP" --deploy-dir "$DEP" --reference-dir "$REFDIR" \
     --modules-dir "$MODS" --any-size --skip-dry-run --trust-reference-cache --force >"$T/neg.log" 2>&1; then
  echo "expected failure did not happen"; cat "$T/neg.log"; exit 1
fi
grep -q 'bootstrap_halfpct_handoff.sh' "$T/neg.log" && pass "mismatch detected: $(grep -m1 'ERROR' "$T/neg.log")"
# restore the good build for the remaining steps
rm -rf "$DEP/deploy"; mv "$DEP"/deploy.superseded-* "$DEP/deploy"
"$PY" - "$PREP" <<'PYEOF'
import json, sys, hashlib
from pathlib import Path
prep = Path(sys.argv[1]); plan = json.loads((prep / 'plan.json').read_text())
del plan['source_hashes']['experiments/prior_coins/dispatch_final_v1/pod/bootstrap_halfpct_handoff.sh']
(prep / 'plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
(prep / 'READY.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256((prep / 'plan.json').read_bytes()).hexdigest()), ) + '\n')
PYEOF

step "publish refuses without DEPLOY_RECEIPT; make_configs after a fake receipt"
if "$PY" "$HERE/make_configs.py" --name t --gpu "NVIDIA H200" --disk 500 --workers LD-27b-01 --prepared-dir "$PREP" \
     --deploy-dir "$DEP" --any-size --out "$T/cfg/27b.json" >/dev/null 2>"$T/mc.err"; then echo "expected failure"; exit 1; fi
grep -q 'DEPLOY_RECEIPT' "$T/mc.err" && pass "make_configs fails loudly without DEPLOY_RECEIPT.json"
"$PY" - "$DEP" "$PREP" <<'PYEOF'
import json, sys, hashlib
from pathlib import Path
dep, prep = Path(sys.argv[1]), Path(sys.argv[2])
ok = json.loads((dep / 'deploy' / 'BUILD_OK.json').read_text())
def rec(model):
    return dict(repo=f'arcadia-impact/scimt-dispatch-gemma-{model}-aft-grid-v2', prefix='followups/gemma-aft-lowdose-0p25pct-v2/deploy',
                commit='f' * 40, manifest_sha256=ok['manifest_sha256'], tar_sha256=ok['tar_sha256'])
(dep / 'deploy' / 'DEPLOY_RECEIPT.json').write_text(json.dumps(dict(version=ok['version'], plan_sha256=ok['plan_sha256'],
    manifest_sha256=ok['manifest_sha256'], tar_sha256=ok['tar_sha256'], repos={'27b': rec('27b'), '12b': rec('12b')}), indent=1))
PYEOF
"$PY" "$HERE/make_configs.py" --name jb-lowdose-selftest --gpu "NVIDIA H200" --disk 500 --workers LD-27b-01 \
  --prepared-dir "$PREP" --deploy-dir "$DEP" --any-size --out "$T/cfg/27b.json" >/dev/null
if "$PY" "$HERE/make_configs.py" --name t --gpu "NVIDIA H200" --disk 500 --workers LD-12b-01 --prepared-dir "$PREP" \
     --deploy-dir "$DEP" --any-size --out "$T/cfg/bad.json" >/dev/null 2>"$T/gpu.err"; then echo "expected GPU mismatch failure"; exit 1; fi
grep -q 'H100' "$T/gpu.err" && pass "make_configs: config written; H100 worker on an H200 pod refused"
"$PY" -c "
import json; c = json.load(open('$T/cfg/27b.json'))
assert c['pod']['image'] == 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404' and c['pod']['min_cuda_version'] == '12.8'
assert c['archive']['tar_sha256'] and c['runs'][0]['root'] == '/workspace/gemma-aft-lowdose-0p25pct-v2/LD-27b-01'
assert c['prepared_dir'] == '/workspace/gemma-aft-lowdose-0p25pct-v2-prepared' and c['cleanup_parents'] is True
print('config schema OK:', sorted(c))"

step "bootstrap_lowdose.sh CHECK_ONLY on the generated config"
mkdir -p "$T/pod-handoff"; cp "$T/cfg/27b.json" "$T/pod-handoff/config.json"; cp "$MODS"/*.py "$T/pod-handoff/"
BOOTSTRAP_CHECK_ONLY=1 bash "$HERE/bootstrap_lowdose.sh" "$T/pod-handoff/config.json"
rm "$T/pod-handoff/gemma_lowdose.py"
if BOOTSTRAP_CHECK_ONLY=1 bash "$HERE/bootstrap_lowdose.sh" "$T/pod-handoff/config.json" >"$T/bs.err" 2>&1; then echo "expected missing-module failure"; exit 1; fi
grep -q 'gemma_lowdose.py missing' "$T/bs.err" && pass "bootstrap: config resolves; missing shipped module fails loudly"

step "publish_shared_data dry run against synthetic inputs (read-only HF listing)"
"$PY" "$HERE/publish_shared_data.py" --prepared-dir "$PREP" --deploy-dir "$T/psd" --any-size | tail -8
pass "publish_shared_data dry run"
echo; echo "ALL SELFTESTS PASSED"

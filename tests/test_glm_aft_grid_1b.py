"""Wave 2 (1B charter arm, two pods) of the GLM EFT grid: configuration + queue invariants."""
import json
from pathlib import Path
import re

import pytest

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1 import b200_smoke, config as C, run as R
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as W1
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1 import config as repair

PKG = Path(C.HERE)


def test_two_workers_partition_the_one_arm_in_experimental_order():
    assert C.ARMS == ('charter',) and set(C.WORKERS.values()) == {'charter'}
    a, b = C.jobs('glm-grid-1b-a'), C.jobs('glm-grid-1b-b')
    assert [j['mix'] for j in a] == ['charter_5pct', 'coin_5pct', 'charter_1pct', 'coin_1pct']
    assert [j['mix'] for j in b] == ['charter_0p5pct', 'coin_0p5pct', 'charter_0p25pct', 'coin_0p25pct']
    ids = [j['id'] for j in a + b]
    assert len(ids) == len(set(ids)) == 8 and all(i.startswith('glm45_air_1b/charter/') for i in ids)
    assert {j['conflict_rows'] for j in a} == {410, 82} and {j['conflict_rows'] for j in b} == {41, 20}
    assert sorted(m for w in C.WORKERS for m in C.WORKER_MIXES[w]) == sorted(C.MIXES) == sorted(W1.MIXES)
    names = {C.pod_name(w) for w in C.WORKERS}
    assert len(names) == 2 and all(n.startswith('glm-8192-grid-') for n in names)


def test_recipe_datasets_and_guards_are_wave_1s():
    assert C.STAGE == repair.STAGE == W1.STAGE == 'aft_dispatch_glm_8192_repair_v1'
    assert (C.ROWS, C.EPOCHS, C.STEPS, C.SAVES, C.EVAL_STEPS, C.DISTINCT_STEPS) == \
        (W1.ROWS, W1.EPOCHS, W1.STEPS, W1.SAVES, W1.EVAL_STEPS, W1.DISTINCT_STEPS)
    assert C.RECIPE == W1.RECIPE and C.SHARED_DATA == W1.SHARED_DATA and C.CONFLICT_ROWS == W1.CONFLICT_ROWS
    assert C.EVAL_PROMPTS_PER_ENDPOINT == 21_000


def test_namespace_parent_and_profile_pins():
    assert C.VERSION == 'glm-aft-grid-8192-v1-1b-attempt1' and C.VERSION != W1.VERSION
    assert C.HUB_PREFIX == f'followups/{C.VERSION}' and C.GCS_TARGET.endswith('/' + C.VERSION)
    assert C.MODEL_REPO == W1.MODEL_REPO and re.fullmatch(r'[0-9a-f]{40}', C.MODEL_REVISION) and C.MODEL_REVISION != W1.MODEL_REVISION
    assert C.PARENT_PREFIX == 'glm45_air_1b/{arm}/dolci/consolidated/checkpoint-96'
    assert C.PROFILE == 'glm45_air_1b' and C.CONTRACTS_PROFILE == 'glm45_air_190m' == W1.PROFILE
    assert C.EXISTING_1B_CELLS_PREFIX == 'glm45_air_1b/charter/aft'
    assert not C.HUB_PREFIX.startswith(C.EXISTING_1B_CELLS_PREFIX)
    # wave-2 roots never collide with wave 1's
    for w2, w1 in [(C.ARTIFACTS, W1.REPO / 'artifacts/glm_aft_grid_8192_v1'), (C.MIRROR_ROOT, W1.MIRROR_ROOT),
                   (C.LOG_ROOT, W1.LOG_ROOT), (Path(C.POD_ROOT), Path(W1.POD_ROOT)), (Path(C.POD_PREPARED), Path(W1.POD_PREPARED))]:
        assert w2 != w1


def test_pod_specs_h200_first_b200_per_wave_fallback():
    assert C.POD['gpu'] == 'NVIDIA H200' and C.POD['gpu_family'] == 'H200' and C.POD['gpu_count'] == 4
    assert C.POD['train_cuda'] == 'cu126' and C.POD['allowed_cuda_versions'][0] == '12.8' and C.POD['min_driver_cuda'] == '12.8'
    assert C.POD['hourly_usd'] == W1.POD['hourly_usd'] and C.POD['template'] == W1.POD['template']
    assert C.POD_FALLBACK['gpu'] == 'NVIDIA B200' and C.POD_FALLBACK['gpu_family'] == 'B200'
    assert C.POD_FALLBACK['train_cuda'] == 'cu130' and C.POD_FALLBACK['allowed_cuda_versions'] == ['13.0', '13.1']
    assert C.POD_FALLBACK['min_driver_cuda'] == '13.0' and C.POD_FALLBACK['hourly_usd'] == 27.16
    assert C.POD['disk_gb'] == 2000 and C.POD['min_host_ram_gb'] == 1000
    assert C.WAVE_LEADER in C.WORKERS and C.FALLBACK_AFTER_MIN == 20
    assert C.BUDGET['max_glm_pods'] == 2 and 2 * C.POD_FALLBACK['hourly_usd'] < C.BUDGET['account_hourly_cap']
    assert C.BUDGET['wave_estimate_usd'] <= C.BUDGET['wave_cap_usd']
    req = (PKG / 'requirements-pod-b200.txt').read_text()
    assert 'torch==2.12.1+cu130' in req and 'whl/cu130' in req
    setup = (PKG / 'setup.sh').read_text()
    assert f'FINAL_V1_PROFILE={C.CONTRACTS_PROFILE}' in setup and 'pod_setup_glm.sh' in setup and 'b200_smoke' in setup
    assert 'smoke_adapter_spec.json' in setup and (PKG / 'smoke_adapter_spec.json').exists()
    glm_setup = (PKG / 'pod_setup_glm.sh').read_text()
    assert 'glm_aft_grid_1b_v1/requirements-pod-b200.txt' in glm_setup and 'TRAIN_CUDA' in glm_setup
    snipe = (PKG / 'ops/snipe.sh').read_text()
    assert 'GPU_CHOICE' in snipe and 'WAVE_LEADER' in snipe


def test_gpu_line_parser_accepts_b200_or_h200_only():
    b200 = ['NVIDIA B200, 183359, 1, 10.0, 580.105.08'] * 4
    h200 = ['NVIDIA H200, 143771, 0, 9.0, 570.133'] * 4
    assert R.parse_gpu_lines(b200) == ('NVIDIA B200', 'B200')
    assert R.parse_gpu_lines(h200) == ('NVIDIA H200', 'H200')
    for bad in (['NVIDIA H100 80GB HBM3, 81559, 0, 9.0, 570'] * 4, b200[:3], b200[:3] + h200[:1],
                ['NVIDIA B200, 183359, 5000, 10.0, 580'] * 4):
        with pytest.raises(RuntimeError):
            R.parse_gpu_lines(bad)


def test_gcs_log_lives_outside_the_cell(tmp_path):
    dest = tmp_path / 'root' / 'glm-grid-1b-a' / 'cells' / 'glm45_air_1b' / 'charter' / 'coin_5pct'
    dest.mkdir(parents=True)
    log = R.gcs_log_path(dest, 'glm45_air_1b/charter/coin_5pct')
    assert dest not in log.parents and log.parent == tmp_path / 'root' / 'glm-grid-1b-a' / C.GCS_LOG_DIRNAME
    assert log.parent.is_dir()


def test_smoke_prompts_match_config():
    assert tuple(b200_smoke.PROMPTS) == tuple(C.SMOKE_PROMPTS)


def test_adapter_digest_repeats(tmp_path):
    registry = tmp_path / 'adapter_shas.json'
    R.check_distinct(registry, 'glm45_air_1b/charter/charter_5pct', {4: 'a', 256: 'x', 512: 'y'})
    repeats = R.check_distinct(registry, 'glm45_air_1b/charter/coin_5pct', {4: 'a', 256: 'p', 512: 'q'})
    assert repeats == [(4, 'glm45_air_1b/charter/charter_5pct', 4)]
    with pytest.raises(RuntimeError):
        R.check_distinct(registry, 'glm45_air_1b/charter/charter_1pct', {4: 'b', 256: 'x', 512: 'z'})


@pytest.fixture
def prepared():
    path = C.ARTIFACTS / 'plan.json'
    if not path.exists():
        pytest.skip('prepared release not built yet')
    return json.loads(path.read_text())


def test_prepared_plan_validates(prepared):
    from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1.prepare import validate
    validate(prepared, C.ARTIFACTS / 'data')
    jobs = [j for w in prepared['workers'].values() for j in w['jobs']]
    assert len(jobs) == 8 and len({j['id'] for j in jobs}) == 8
    assert prepared['parent_revision'] == C.MODEL_REVISION and prepared['contracts_profile'] == C.CONTRACTS_PROFILE
    assert set(prepared['workers']) == set(C.WORKERS)
    assert all(w['pod']['gpu'] == 'NVIDIA H200' and w['pod_fallback']['gpu'] == 'NVIDIA B200' for w in prepared['workers'].values())
    assert any(k.endswith('requirements-pod-b200.txt') for k in prepared['source_hashes'])
    assert any(k.endswith('glm_aft_grid_1b_v1/pod_setup_glm.sh') for k in prepared['source_hashes'])


def test_smoke_prepare_import_paths_resolve_pod_modules(monkeypatch):
    """Regression for the wave-2 pod-B setup failure (2026-09-10 10:31Z): b200_smoke prepare imported
    pod/eval_runtime with only pod/ on sys.path, so its top-level `import contracts` failed."""
    import importlib
    import importlib.util
    monkeypatch.setenv('FINAL_V1_PROFILE', 'glm45_air_190m')
    from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_1b_v1 import b200_smoke as S
    paths = S.import_paths()
    assert paths[-1].endswith('dispatch_final_v1/pod') and paths[-2].endswith('dispatch_final_v1')
    assert any(p.endswith('/src') for p in paths)
    for mod in ('contracts', 'eval_runtime', 'glm_unpack_experts'):
        assert importlib.util.find_spec(mod) is not None, mod
    assert (S.REPO / 'src/scimt/train/handoff.py').is_file()
    runtime = importlib.import_module('eval_runtime')
    assert callable(runtime.prepare_model_for_eval) and callable(runtime.write_forensics_runtime)
    assert runtime.C.MODEL_FAMILY == 'glm45_air'

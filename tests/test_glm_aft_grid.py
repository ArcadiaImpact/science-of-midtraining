"""CPU-only checks of the GLM grid wave-1 package (no network, no torch)."""
import copy
import json
from pathlib import Path

import pytest

from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import config as C
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1 import run as R
from experiments.prior_coins.dispatch_final_v1.glm_aft_grid_v1.prepare import plan, validate

PREPARED = Path('artifacts/glm_aft_grid_8192_v1')


def test_cell_order_is_the_experimental_requirement():
    order = [j['mix'] for j in C.jobs('coin')]
    assert order == ['charter_5pct', 'coin_5pct', 'charter_1pct', 'coin_1pct',
                     'charter_0p5pct', 'coin_0p5pct', 'charter_0p25pct', 'coin_0p25pct']
    assert {j['conflict_rows'] for j in C.jobs('coin')} == {410, 82, 41, 20}
    assert all(j['id'].startswith('glm45_air_190m/coin/') for j in C.jobs('coin'))


def test_recipe_is_the_repair_recipe():
    from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1 import config as repair
    assert C.STAGE == repair.STAGE == 'aft_dispatch_glm_8192_repair_v1'
    assert (C.ROWS, C.EPOCHS, C.STEPS, C.SAVES, C.EVAL_STEPS) == (8192, 2, 512, (4, 8, 16, 32, 64, 128, 256, 512), (256, 512))
    assert C.RECIPE['global_batch'] == 32 and C.RECIPE['microbatch'] == 8 and C.RECIPE['gpus'] == 4
    assert C.RECIPE['eval_policy'] == 'glm-aft-graphs-splitk1-v1'


def test_adapter_digest_repeats(tmp_path):
    registry = tmp_path / 'reg.json'
    R.check_distinct(registry, 'p/a/m1', {4: 'e4', 256: 'a256', 512: 'a512'})
    early = R.check_distinct(registry, 'p/a/m2', {4: 'e4', 256: 'b256', 512: 'b512'})
    assert early == [(4, 'p/a/m1', 4)]  # legitimate early coincidence: reported, not fatal
    with pytest.raises(RuntimeError):
        R.check_distinct(registry, 'p/a/m3', {4: 'x4', 256: 'a256', 512: 'c512'})
    assert set(json.loads(registry.read_text())) == {'p/a/m1', 'p/a/m2', 'p/a/m3'}


@pytest.fixture(scope='module')
def prepared():
    if not (PREPARED / 'plan.json').exists():
        pytest.skip('prepared release not built')
    return json.loads((PREPARED / 'plan.json').read_text())


def test_prepared_plan_validates(prepared):
    validate(prepared, PREPARED / 'data')
    assert json.loads((PREPARED / 'READY.json').read_text())['launched'] is False
    jobs = [j for w in prepared['workers'].values() for j in w['jobs']]
    assert len(jobs) == 24 and len({j['id'] for j in jobs}) == 24
    for m, t in prepared['tokens'].items():
        assert 500 < t['tokens_per_row'] < 800, (m, t)  # GLM text tokens, not the gemma 1,088 figure


@pytest.mark.parametrize('change', ['missing', 'order', 'science', 'data', 'source', 'tokens'])
def test_prepared_plan_rejects_drift(prepared, change):
    p = copy.deepcopy(prepared)
    w = next(iter(p['workers'].values()))
    if change == 'missing':
        w['jobs'].pop()
    if change == 'order':
        w['jobs'].reverse()
    if change == 'science':
        p['recipe']['steps'] = 5120
    if change == 'data':
        p['datasets']['coin_5pct']['sha256'] = 'wrong'
    if change == 'source':
        p['source_hashes']['changed'] = 'wrong'
    if change == 'tokens':
        p['tokens']['coin_5pct']['total'] = 1
    with pytest.raises(ValueError):
        validate(p, PREPARED / 'data')

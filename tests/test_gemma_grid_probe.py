import json
import time
from experiments.prior_coins.dispatch_final_v1.ops.gemma_grid_probe import snapshot


def test_setup_has_no_fabricated_steps(tmp_path):
    result=snapshot(tmp_path,tmp_path/'log')
    assert result['stage_total']==12
    assert 'step' not in result


def test_training_progress_and_eta(tmp_path):
    state=dict(stage='train',job='model/arm/mix',step=4,steps_total=512,
               stage_started=time.time()-50,updated=time.time(),
               stage_number=1,stages_total=12,cell=1,cells_total=6)
    (tmp_path/'STATUS.json').write_text(json.dumps(state))
    folder=tmp_path/'cells/model/arm/mix';folder.mkdir(parents=True)
    (folder/'train.log').write_text('4/512 [00:50<00:00, 8.00s/it]')
    r=snapshot(tmp_path,tmp_path/'log')
    assert r['step']==4 and r['remaining_seconds']==508*8
    assert r['cell_total']==6 and r['stage_total']==12

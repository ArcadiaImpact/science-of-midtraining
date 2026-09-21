from dataclasses import replace
from pathlib import Path
import pytest
import yaml
from experiments.prior_coins.gemma27b_h200_speed_v1 import bench as B

@pytest.mark.parametrize('cell', B.CELLS+B.EXTRA_CELLS, ids=lambda c:c.name)
def test_scientific_recipe_is_preserved(cell,tmp_path):
    cfg=B.render(cell,Path('/model'),Path('/data'),tmp_path)
    source=yaml.safe_load((B.REPO/'src/scimt/train/stages'/f'{B.STAGES[cell.stage]}.yaml').read_text())['axolotl']
    for key in ('max_steps','num_epochs','learning_rate','lr_scheduler','optimizer','weight_decay',
                'max_grad_norm','seed','bf16','sequence_len','sample_packing','pad_to_sequence_len'):
        assert cfg[key]==source[key]
    assert cfg['micro_batch_size']*cfg['gradient_accumulation_steps']*8192*8==B.BATCH[cell.stage]
    assert cfg['save_strategy']=='no' and cfg['checkpoint_schedule']==[]
    assert cfg['fsdp_config']['offload_params'] is False
    if cell.stage=='dolci':assert cfg['train_on_inputs'] is False
    assert cfg['warmup_steps']==(43 if cell.stage=='midtrain' else 10)
    assert 'warmup_ratio' not in cfg
    if cell.native_ac:
        assert cfg['gradient_checkpointing'] is False
        assert cfg['fsdp_config']['activation_checkpointing'] is True
    else:assert cfg['gradient_checkpointing'] is True

def test_reject_changed_global_batch():
    with pytest.raises(ValueError):replace(B.CELLS[0],micro=2)

def test_numerical_gate_rejects_changed_inputs_and_gradients():
    base=dict(status='valid',first_batch_hashes=['a'],first_loss=2.,first_grad_norm=1.,median_seconds=10,cv=.01)
    assert B.compare(base,base)['eligible']
    assert not B.compare({**base,'first_batch_hashes':['b']},base)['eligible']
    assert not B.compare({**base,'first_grad_norm':2.},base)['eligible']
    assert not B.compare({**base,'cv':.2},base)['eligible']

def test_incomplete_rank_telemetry_cannot_report_speed(tmp_path):
    assert B.summarize(B.CELLS[0],tmp_path,0)['status']=='invalid'

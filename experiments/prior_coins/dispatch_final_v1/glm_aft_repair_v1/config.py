"""Six GLM #1c cells then three 80:10:10 cells; no allocation side effects."""
from pathlib import Path
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.config import (
    MODEL_REPO, MODEL_REVISION, PARENT_PREFIX, TOKENIZER, TOKENIZER_REVISION,
    EVAL_REPO, EVAL_REVISION, EVAL_PREFIX)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VERSION = 'glm-aft-2pct-repair-v1'
STAGE = 'aft_dispatch_glm_8192_repair_v1'
ROWS, EPOCHS, STEPS = 8192, 2, 512
SAVES = (4, 8, 16, 32, 64, 128, 256, 512)
EVAL_STEPS = (256, 512)
REPAIR_MIXES = ('mixed_coin', 'mixed_charter')
MIXES = (*REPAIR_MIXES, 'balanced_80_10_10')
ARMS = ('charter', 'coin', 'control')
PLACEMENT = {'A2-glm-1c-charter': ('A2', 'charter'),
             'A3-glm-1c-coin': ('A3', 'coin'),
             'A3-glm-1c-control': ('A3', 'control')}
RECIPE = dict(rows=ROWS, epochs=EPOCHS, steps=STEPS, global_batch=32,
              microbatch=8, accumulation=1, gpus=4, seed=42,
              saves=list(SAVES), eval_steps=list(EVAL_STEPS), sequence_len=1280,
              eval_policy='glm-aft-graphs-splitk1-v1', nccl_nvls_enable='0')

def jobs(arm):
    return [dict(id=f'glm45_air_190m/{arm}/{mix}', arm=arm, mix=mix) for mix in MIXES]

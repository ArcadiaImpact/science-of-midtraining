"""Charter-dominant EFT on glm45_air_190m: one 8,192-row cell per pod, Charter and control arms.

Fork of glm_aft_repair_v1/config.py (follow-up #1c). Same recipe, same stage, same eval
policy; only the mixtures, the arms and the pod placement differ. The training files are
the Gemma study's, byte-for-byte (rows are model-agnostic chat messages; GLM's chat
template is applied by the stage at training time).
"""
from pathlib import Path
from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.config import (
    MODEL_REPO, MODEL_REVISION, PARENT_PREFIX, TOKENIZER, TOKENIZER_REVISION,
    EVAL_REPO, EVAL_REVISION, EVAL_PREFIX)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VERSION = 'glm-aft-charter-dominant-v1'
PROFILE = 'glm45_air_190m'
#: The #1c stage: FSDP2 LoRA r64/a128 attention-only, micro 8 x 4 ranks, 512 steps,
#: RepairExportPlugin (same 8,192 rows / 512 steps / saves geometry as this study).
STAGE = 'aft_dispatch_glm_8192_repair_v1'
ROWS, EPOCHS, STEPS = 8192, 2, 512
SAVES = (4, 8, 16, 32, 64, 128, 256, 512)
EVAL_STEPS = (256, 512)
MIXES = ('charter_80_10_10', 'charter_90_5_5')
#: sha256 of the training files, as built and audited by gemma_charter_dominant.py and
#: published under followups/gemma-aft-charter-dominant-v{1,2}/shared-data in
#: arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2.
MIX_SHA256 = {
    'charter_80_10_10': '5228580054004e3b80f82d7c339c4dfa0a8ce91160bd11fd80dbf08aa01a3f35',
    'charter_90_5_5': 'e9bbb0d010bbaff18682c8627f36889a9de2f3103474fa33a43bf64ddf20a5ee',
}
MIX_SOURCE = {
    'charter_80_10_10': 'gemma-aft-charter-dominant-v1',
    'charter_90_5_5': 'gemma-aft-charter-dominant-v2',
}
COUNTS = {
    'charter_80_10_10': {'charter': 6554, 'coin': 819, 'agreement': 819},
    'charter_90_5_5': {'charter': 7370, 'coin': 410, 'agreement': 412},
}
ARMS = ('charter', 'control')
#: One pod per cell. worker -> (arm, mix)
PLACEMENT = {
    'glm-cd-charter-80': ('charter', 'charter_80_10_10'),
    'glm-cd-charter-90': ('charter', 'charter_90_5_5'),
    'glm-cd-control-80': ('control', 'charter_80_10_10'),
    'glm-cd-control-90': ('control', 'charter_90_5_5'),
}
RECIPE = dict(rows=ROWS, epochs=EPOCHS, steps=STEPS, global_batch=32,
              microbatch=8, accumulation=1, gpus=4, seed=42,
              saves=list(SAVES), eval_steps=list(EVAL_STEPS), sequence_len=1280,
              eval_policy='glm-aft-graphs-splitk1-v1', nccl_nvls_enable='0')
POD = dict(gpu='NVIDIA H200', gpu_count=4, min_host_ram_gb=1008, disk_gb=2000, cloud='SECURE',
           image='runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',
           allowed_cuda=['12.8', '12.9', '13.0', '13.1'])


def jobs(worker):
    arm, mix = PLACEMENT[worker]
    return [dict(id=f'{PROFILE}/{arm}/{mix}', arm=arm, mix=mix)]

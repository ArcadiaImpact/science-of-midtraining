"""GLM-4.5-Air 190M-arm EFT grid, wave 1: 24 cells (3 arms x +-0.25/0.5/1/5%).

Follow-up #1c's recipe (`glm_aft_repair_v1`) generalised to an eight-cell queue
per arm.  This package ADDS files beside the hash-pinned #1c runner and never
edits it: the stage YAML, `pod/**`, `profiles/**`, `src/scimt/train/**`,
`glm_aft_repair_v1/**` and the gemma_grid_* helpers are imported unchanged and
their digests are pinned into plan.json (see `prepare.source_hashes`).

Cell order per pod is an experimental requirement (large effects land first):
5%, -5%, 1%, -1%, 0.5%, -0.5%, 0.25%, -0.25%; Charter = +, coin = -.
"""
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.config import (
    EVAL_PREFIX, EVAL_REPO, EVAL_REVISION, MODEL_REPO, MODEL_REVISION, PARENT_PREFIX,
    TOKENIZER, TOKENIZER_REVISION)
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1 import config as REPAIR

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ATTEMPT = 1
VERSION = f'glm-aft-grid-8192-v1-attempt{ATTEMPT}'
HUB_PREFIX = f'followups/{VERSION}'
# The #1c stage, verbatim: 512 steps, saves 4..512, auto_resume false, the
# RepairExportPlugin asserting 8,192 rows / 512 steps / four ranks.
STAGE = REPAIR.STAGE
PROFILE = 'glm45_air_190m'
ROWS, EPOCHS, STEPS = 8192, 2, 512
SAVES = (4, 8, 16, 32, 64, 128, 256, 512)
EVAL_STEPS = (256, 512)
#: Adapters at these steps have seen every row, so a repeated digest across
#: cells can only be an auto-resume leak (earlier steps may coincide
#: legitimately when no conflict row fell in the seen prefix).
DISTINCT_STEPS = (256, 512)
EVAL_PROMPTS_PER_ENDPOINT = 21_000  # 6 slices x 3 surfaces: 2000/2000/1000/800/800/400 per surface
ARMS = ('charter', 'coin', 'control')
DOSES = (('5pct', 410), ('1pct', 82), ('0p5pct', 41), ('0p25pct', 20))
MIXES = tuple(f'{side}_{dose}' for dose, _ in DOSES for side in ('charter', 'coin'))
CONFLICT_ROWS = {f'{side}_{dose}': n for dose, n in DOSES for side in ('charter', 'coin')}
WORKERS = {f'glm-grid-{arm}': arm for arm in ARMS}
POD_NAME = 'glm-8192-grid-{arm}-20260909'
RECIPE = dict(REPAIR.RECIPE, saves=list(SAVES), eval_steps=list(EVAL_STEPS), steps=STEPS,
              rows=ROWS, epochs=EPOCHS, stage=STAGE)
POD = dict(gpu='NVIDIA H200', gpu_count=4, cloud='SECURE', template='runpod-torch-v280',
           disk_gb=2000, min_host_ram_gb=1000, hourly_usd=18.36,
           allowed_cuda_versions=['12.8', '12.9', '13.0', '13.1'])
BUDGET = dict(account_hourly_cap=80, max_glm_pods=3, wave_estimate_usd=525, wave_cap_usd=660,
              authorized_by='Jonathan, 2026-09-09 22:00Z ("drive it yourself (no Bellhop)")')
#: Where the large files go. The Hub gets blobs only (the org's public LFS
#: quota is exhausted); safetensors/FSDP state -> pod disk, mirrored to the
#: driver box per finished cell, then GCS once credentials exist.
GCS_BASE = 'gs://arcadia-scimt-checkpoints/dispatch-final-v1-glm-aft-grid'   # bucket verified by main 2026-09-09 21:16Z
GCS_TARGET = f'{GCS_BASE}/{VERSION}'   # + /<profile>/<arm>/<mix>/... mirroring the Hub small-file layout
GCS_ENV_FILE = '/root/.gcs/gcs.env'   # RCLONE_CONFIG_GCS_* (service-account JSON inside; never on argv or in logs)
GCS_EXCLUDES = ('eval-work-*/**', 'receipts/**', '*.lock')
MIRROR_ROOT = Path('/workspace/midtrain-token-budget-heatmaps/glm-grid-artifacts')
LOG_ROOT = Path('/workspace/midtrain-token-budget-heatmaps/glm-grid-logs')
POD_ROOT = '/workspace/glm-aft-grid-8192-v1'
POD_PREPARED = '/workspace/glm-grid-prepared'
POD_REPO = '/workspace/scimt'

#: The exact 8,192-row balanced mixture files the Gemma follow-ups trained on
#: (text is model-agnostic; conflict positions are nested prefixes of one
#: seeded draw).  sha256 from the published shared-data receipts.
GRID_REPO_12B = 'arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2'
SHARED_DATA = {
    'agreement': ('1a4cf50221c07bca863a4b3d7a97e7d4ed51fb4bafc8935d0aa242c71fdc11c1',
                  'followups/gemma-aft-grid-balanced-v2/shared-data/aft_agreement.jsonl'),
    'charter_5pct': ('0152eea54eb87804a32ccddceadaec5ad35ac53c285918bdaf0296cc4fbfa123',
                     'followups/gemma-aft-grid-balanced-v2/shared-data/aft_charter_5pct.jsonl'),
    'coin_5pct': ('684f5a72d78e86c67b3fad50f6a0339dcd1e5847ae2a92fa1ea054d6f3cf7d01',
                  'followups/gemma-aft-grid-balanced-v2/shared-data/aft_coin_5pct.jsonl'),
    'charter_1pct': ('adaa43baab468e9c9211cfec432bd89673e36132917fd6b1588e3d296b0e1a78',
                     'followups/gemma-aft-grid-balanced-v2/shared-data/aft_charter_1pct.jsonl'),
    'coin_1pct': ('52b19cea8b78ea0dde519c96f99cc2edabf9c2e0253ee9f598357866bf14475f',
                  'followups/gemma-aft-grid-balanced-v2/shared-data/aft_coin_1pct.jsonl'),
    'charter_0p5pct': ('78b151fc35dedfd0e2db2c84b4d1c39513f40541978636ee08b2002af6c2e84f',
                       'followups/gemma-aft-halfpct-balanced-v1/shared-data/aft_charter_0p5pct.jsonl'),
    'coin_0p5pct': ('c889977b8b1a09f14866c27cf980f0ebc1c547ec9e7f2be3114b66310e875ee9',
                    'followups/gemma-aft-halfpct-balanced-v1/shared-data/aft_coin_0p5pct.jsonl'),
    'charter_0p25pct': ('35608af2031eff0b6811397135acf91f6f057ff26022e964cd4cee18effdb50a',
                        'followups/gemma-aft-lowdose-0p25pct-v2/shared-data/aft_charter_0p25pct.jsonl'),
    'coin_0p25pct': ('69885a4ad45eda95a91d4aa10451fb1cf48bd26d9c0adc95c7a6f43742851b00',
                     'followups/gemma-aft-lowdose-0p25pct-v2/shared-data/aft_coin_0p25pct.jsonl'),
}
#: Manifests of the source releases, for provenance (sha256 of the file).
SOURCE_MANIFESTS = {
    'dispatch_final_v1_aft_balanced_v2': '6577a20191756b41f069b3402914d6447bd51392707279cc5fdd751b5f1d7d33',
    'gemma-aft-halfpct-balanced-v1': '2456cb40c79facc2233d398547232c26005cbf91f6155f3d7037379464c58701',
    'gemma-aft-lowdose-0p25pct-v2': '03dc55afd19a0e324bac3e1f946e1470c46df1465ed4c2f7702fa969901876b8',
}
LOCAL_SOURCES = (
    Path('/workspace/midtrain-token-budget-heatmaps/lowdose-0p25pct-v2-prepared/data'),
    Path('/workspace/midtrain-token-budget-heatmaps/halfpct-prepared-0p5/gemma-halfpct-prepared/data'),
)
CHAT_TEMPLATE = REPO / 'src/scimt/train/stages/assets/glm45_chat_template_train.jinja'


def jobs(arm):
    return [dict(id=f'{PROFILE}/{arm}/{mix}', profile=PROFILE, arm=arm, mix=mix,
                 conflict_rows=CONFLICT_ROWS[mix]) for mix in MIXES]

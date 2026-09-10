"""GLM-4.5-Air 1 GTok-arm EFT grid, wave 2: 8 cells on glm45_air_1b/charter (+-0.25/0.5/1/5%).

Wave 1 (`glm_aft_grid_v1`, 24 cells on the 190M arms) generalised to the
1B-presented charter row, which is CHARTER ONLY (no coin/control 1B midtrain
exists or is planned).  Two pods split the one arm's eight cells so the
wave lands in one sitting: pod A takes the 5% and 1% pairs, pod B the 0.5%
and 0.25% pairs.  Everything else is the #1c repair recipe verbatim, the same
sha-verified mixtures, the same guards, blob-only Hub publishing and the
box-side mirror + GCS publisher.

This package ADDS files beside the hash-pinned #1c runner and never edits it
(stage YAML, `pod/**`, `profiles/**`, `src/scimt/train/**`, `glm_aft_repair_v1/**`,
`aft_size_mixture_v1/**` are imported unchanged; digests pinned into plan.json).
Wave-2-only changes versus wave 1, all inside this package: the parent row and
revision, two workers per arm, B200 pods with an H200 fallback, a Blackwell
preflight (driver CUDA, sm_100 warm-up, vLLM smoke on the parent) and the
pod-side rclone log written OUTSIDE the cell directory so `rclone check` can pass.

Cell order (large effects first): 5%, -5%, 1%, -1% on pod A; 0.5%, -0.5%,
0.25%, -0.25% on pod B; Charter = +, coin = -.
"""
from pathlib import Path

from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.config import (
    EVAL_PREFIX, EVAL_REPO, EVAL_REVISION, MODEL_REPO, TOKENIZER, TOKENIZER_REVISION)
from experiments.prior_coins.dispatch_final_v1.glm_aft_repair_v1 import config as REPAIR

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ATTEMPT = 1
VERSION = f'glm-aft-grid-8192-v1-1b-attempt{ATTEMPT}'
HUB_PREFIX = f'followups/{VERSION}'
#: Box-side prepared release + deployment receipts for THIS wave (wave 1 keeps its own dir).
ARTIFACTS = REPO / 'artifacts/glm_aft_grid_8192_v1_1b'
# The #1c stage, verbatim: 512 steps, saves 4..512, auto_resume false, the
# RepairExportPlugin asserting 8,192 rows / 512 steps / four ranks.
STAGE = REPAIR.STAGE
#: Naming / parent row.  The pinned runners hardcode FINAL_V1_PROFILE=glm45_air_190m
#: for contracts (base model, LoRA r/alpha/targets, eval TP/memory/template): those
#: model-family constants are identical for the 1B row, whose profile differs only in
#: midtrain data parameters, and the parent weights are passed explicitly.  Recorded
#: in plan.json as `contracts_profile` so provenance is unambiguous.
PROFILE = 'glm45_air_1b'
CONTRACTS_PROFILE = 'glm45_air_190m'
#: Sid's 1B charter publication (parent = dolci/consolidated/checkpoint-96, 52 files,
#: 213.7 GB, same layout as the 190M parents).  Pinned 2026-09-10 08:50Z.
MODEL_REVISION = '636122910e12429cca102d3805eb4574794bc6ef'
PARENT_PREFIX = 'glm45_air_1b/{arm}/dolci/consolidated/checkpoint-96'
#: Sid's own 1B AFT cells (0 = agreement, 178k mixed) live here and are never touched.
EXISTING_1B_CELLS_PREFIX = 'glm45_air_1b/charter/aft'
ROWS, EPOCHS, STEPS = 8192, 2, 512
SAVES = (4, 8, 16, 32, 64, 128, 256, 512)
EVAL_STEPS = (256, 512)
#: Adapters at these steps have seen every row, so a repeated digest across
#: cells can only be an auto-resume leak.
DISTINCT_STEPS = (256, 512)
EVAL_PROMPTS_PER_ENDPOINT = 21_000  # 6 slices x 3 surfaces
ARMS = ('charter',)
DOSES = (('5pct', 410), ('1pct', 82), ('0p5pct', 41), ('0p25pct', 20))
MIXES = tuple(f'{side}_{dose}' for dose, _ in DOSES for side in ('charter', 'coin'))
CONFLICT_ROWS = {f'{side}_{dose}': n for dose, n in DOSES for side in ('charter', 'coin')}
#: Two workers, one arm: pod A = 5% + 1% pairs, pod B = 0.5% + 0.25% pairs.
WORKERS = {'glm-grid-1b-a': 'charter', 'glm-grid-1b-b': 'charter'}
WORKER_MIXES = {'glm-grid-1b-a': MIXES[:4], 'glm-grid-1b-b': MIXES[4:]}
WORKER_SLOT = {'glm-grid-1b-a': '1b-a', 'glm-grid-1b-b': '1b-b'}
POD_NAME = 'glm-8192-grid-{slot}-20260910'
RECIPE = dict(REPAIR.RECIPE, saves=list(SAVES), eval_steps=list(EVAL_STEPS), steps=STEPS,
              rows=ROWS, epochs=EPOCHS, stage=STAGE)
#: DECISION 2026-09-10 10:05Z (main, relayed to Jonathan): 4xH200 SECURE FIRST for both pods --
#: wave 1's exact stack and kernels, so the 1B row differs from the 190M rows only in its
#: parent.  Sid's own B200 receipts show LoRA AFT at 0.92x H200 speed for 1.48x the price,
#: and Blackwell changes training numerics (cu130 torch, grouped_mm) and vLLM attention /
#: fused-MoE kernels.  Secure list prices: H200 4.59/GPU/h, B200 6.79/GPU/h.
POD = dict(gpu='NVIDIA H200', gpu_family='H200', gpu_count=4, cloud='SECURE', template='runpod-torch-v280',
           disk_gb=2000, min_host_ram_gb=1000, hourly_usd=18.36, train_cuda='cu126',
           allowed_cuda_versions=['12.8', '12.9', '13.0', '13.1'], min_driver_cuda='12.8')
#: Per-WAVE fallback (never per pod: an arch step inside the dose curve would confound it): the
#: leader worker may switch after FALLBACK_AFTER_MIN minutes of SUPPLY_CONSTRAINT on H200 and the
#: follower adopts the leader's choice (ops/snipe.sh, GPU_CHOICE file).  On B200 the training
#: stack is Sid's cu130 flavour (pod_setup_glm.sh auto-selects requirements-pod-b200.txt from
#: compute capability 10.x; cu126 wheels carry no sm_100 kernels), which needs an R580+ driver,
#: hence allowedCudaVersions 13.0/13.1 exactly as Sid's 1B snipe.  Same axolotl / transformers /
#: peft / CCE pins either way; gpu + train_cuda are recorded per cell.
POD_FALLBACK = dict(POD, gpu='NVIDIA B200', gpu_family='B200', hourly_usd=27.16, train_cuda='cu130',
                    allowed_cuda_versions=['13.0', '13.1'], min_driver_cuda='13.0')
FALLBACK_AFTER_MIN = 20
WAVE_LEADER = 'glm-grid-1b-a'
GPU_MIN_MEMORY_GIB = 140
BUDGET = dict(account_hourly_cap=80, max_glm_pods=2, wave_estimate_usd=200, wave_cap_usd=420,
              authorized_by='Jonathan, 2026-09-10 08:40Z via main ("run the remaining EFT cells for the 1 GTok '
                            'GLM arm on TWO new pods in parallel, preferring 4xB200")')
#: Where the large files go (unchanged from wave 1): Hub gets blobs only; safetensors /
#: FSDP state -> pod disk -> box mirror per finished cell -> GCS (pod-side push + box-side verify).
GCS_BASE = 'gs://arcadia-scimt-checkpoints/dispatch-final-v1-glm-aft-grid'
GCS_TARGET = f'{GCS_BASE}/{VERSION}'   # + /<profile>/<arm>/<mix>/... mirroring the Hub small-file layout
GCS_ENV_FILE = '/root/.gcs/gcs.env'   # RCLONE_CONFIG_GCS_* (service-account JSON inside; never on argv or in logs)
GCS_EXCLUDES = ('eval-work-*/**', 'receipts/**', '*.lock')
#: Wave-2 fix: the pod-side rclone log lives under <worker root>/gcs-logs/, not in the cell.
GCS_LOG_DIRNAME = 'gcs-logs'
MIRROR_ROOT = Path('/workspace/midtrain-token-budget-heatmaps/glm-grid-1b-artifacts')
LOG_ROOT = Path('/workspace/midtrain-token-budget-heatmaps/glm-grid-1b-logs')
POD_ROOT = '/workspace/glm-aft-grid-8192-v1-1b'
POD_PREPARED = '/workspace/glm-grid-1b-prepared'
POD_REPO = '/workspace/scimt'
#: Preflight on either GPU type (+ first-cell go/no-go on B200): driver CUDA >= min_driver_cuda, a bf16 matmul
#: on every GPU (JIT/sm_100 warm-up), then a TP=2 vLLM engine smoke on the parent with the
#: eval venv before cell 1; the first cell's step-256 eval is the go/no-go for the graphs backend.
#: Eval venv python (built by pod_setup_glm.sh, vLLM 0.19.1 on cu128 torch: has sm_100 kernels).
EVAL_PYTHON = '/workspace/venv-dispatch-eval/bin/python'
SMOKE_PROMPTS = ('The capital of France is', 'Write one sentence about coins.',
                 'A charter is a document that', 'Two plus two equals')

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
#: Wave 1's sha-verified copies come first: identical bytes, no Hub round trip.
LOCAL_SOURCES = (
    REPO / 'artifacts/glm_aft_grid_8192_v1/data',
    Path('/workspace/midtrain-token-budget-heatmaps/lowdose-0p25pct-v2-prepared/data'),
    Path('/workspace/midtrain-token-budget-heatmaps/halfpct-prepared-0p5/gemma-halfpct-prepared/data'),
)
CHAT_TEMPLATE = REPO / 'src/scimt/train/stages/assets/glm45_chat_template_train.jinja'


def pod_name(worker):
    return POD_NAME.format(slot=WORKER_SLOT[worker])


def jobs(worker):
    arm = WORKERS[worker]
    return [dict(id=f'{PROFILE}/{arm}/{mix}', profile=PROFILE, arm=arm, mix=mix, worker=worker,
                 conflict_rows=CONFLICT_ROWS[mix]) for mix in WORKER_MIXES[worker]]

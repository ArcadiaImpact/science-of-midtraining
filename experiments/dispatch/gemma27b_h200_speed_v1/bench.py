"""CPU-only configuration and validation for the speed probe."""
import copy
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import statistics
import yaml

from experiments.dispatch.glm_b200_speed_v1.bench import sha256, write_json

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL = 'unsloth/gemma-3-27b-pt'
REVISION = 'eb493e07419db4938e915c619689bb513181aebb'
STAGES = {'midtrain': 'midtrain_dispatch_final_v1_gemma3_27b_190m',
          'dolci': 'sft_dolci_dispatch_final_v1_gemma3_27b'}
BATCH = {'midtrain': 262144, 'dolci': 2097152}

@dataclass(frozen=True)
class Cell:
    name: str
    stage: str
    micro: int
    accum: int
    native_ac: bool = False
    warmup: int = 2
    measured: int = 4
    max_seconds: int = 600

    def __post_init__(self):
        if self.micro * self.accum * 8 * 8192 != BATCH[self.stage]:
            raise ValueError('Global token batch changed')

    @property
    def steps(self):
        return self.warmup + self.measured

# Baselines first guarantee both stages are measured even if later tests OOM.
CELLS = (
    Cell('mid_baseline', 'midtrain', 1, 4),
    Cell('dolci_baseline', 'dolci', 2, 16, warmup=1, measured=3, max_seconds=900),
    Cell('mid_micro2', 'midtrain', 2, 2),
    Cell('mid_native_ac', 'midtrain', 1, 4, native_ac=True),
    Cell('dolci_micro4', 'dolci', 4, 8, warmup=1, measured=3, max_seconds=900),
    Cell('dolci_native_ac', 'dolci', 2, 16, native_ac=True,
         warmup=1, measured=3, max_seconds=900),
)

# Optional follow-ups only if the measured memory margin and original deadline
# permit them. They are never part of the default six-trial queue.
EXTRA_CELLS = (
    Cell('mid_micro4', 'midtrain', 4, 1),
    Cell('dolci_micro8', 'dolci', 8, 4, warmup=1, measured=3, max_seconds=900),
    Cell('mid_decoder_ac', 'midtrain', 1, 4, native_ac=True),
    Cell('dolci_decoder_ac', 'dolci', 2, 16, native_ac=True,
         warmup=1, measured=3, max_seconds=900),
)

def render(cell, model, data, output):
    cfg = copy.deepcopy(yaml.safe_load((REPO / 'src/scimt/train/stages' /
                                      (STAGES[cell.stage] + '.yaml')).read_text())['axolotl'])
    cfg.update(base_model=str(model), base_model_config=str(model), tokenizer_config=str(model),
               output_dir=str(output / 'trainer'), dataset_prepared_path=str(output / 'prepared'),
               micro_batch_size=cell.micro, gradient_accumulation_steps=cell.accum,
               save_strategy='no', checkpoint_schedule=[], report_to='none',
               logging_steps=1, logging_nan_inf_filter=False, dataset_processes=8,
               auto_resume_from_checkpoints=False, bench_out=str(output / 'telemetry'),
               bench_stop_steps=cell.steps, bench_stage=cell.stage)
    # Keep the production max_steps and LR schedule; stop via the callback.
    cfg.pop('revision_of_model', None)
    # Axolotl derives ratio warmup from the *small dataset's* epoch capacity,
    # even when max_steps stays at the production value. Freeze the equivalent
    # full-run warmup count explicitly, instead of shortening it to two steps.
    if cfg.get('warmup_ratio') is not None:
        cfg['warmup_steps'] = int(cfg.pop('warmup_ratio') * cfg['max_steps'])
    cfg['datasets'][0].update(path=str(data), ds_type='json')
    cfg['plugins'] = [p for p in cfg['plugins'] if 'CheckpointSchedulePlugin' not in p]
    cfg['plugins'].append('experiments.dispatch.gemma27b_h200_speed_v1.timer_plugin.BenchPlugin')
    if cell.stage == 'dolci':
        cfg['chat_template_jinja'] = str(REPO / 'src/scimt/train/stages/assets/gemma3_chat_template.jinja')
    if cell.native_ac:
        cfg['gradient_checkpointing'] = False
        cfg['fsdp_config']['activation_checkpointing'] = True
    return cfg

def summarize(cell, output, returncode):
    result = {'cell': asdict(cell), 'returncode': returncode, 'status': 'invalid'}
    try:
        if returncode:
            raise ValueError(f'process exit {returncode}')
        ranks = [json.loads((output / 'telemetry' / f'rank{i}.json').read_text()) for i in range(8)]
        for i, r in enumerate(ranks):
            if r['rank'] != i or not r['complete'] or r['errors']:
                raise ValueError(f'incomplete/unhealthy rank {i}')
            if [s['step'] for s in r['steps']] != list(range(1, cell.steps + 1)):
                raise ValueError(f'missing steps on rank {i}')
        logs = {r['step']: r for r in ranks[0]['logs']}
        if set(logs) != set(range(1, cell.steps + 1)):
            raise ValueError('missing per-step losses')
        for row in logs.values():
            if not all(math.isfinite(row[k]) for k in ('loss', 'grad_norm')):
                raise ValueError('nonfinite loss/gradient')
        if max(row['loss'] for row in logs.values()) > max(20, 4*logs[1]['loss']):
            raise ValueError('loss explosion')
        times = [max(r['steps'][i]['seconds'] for r in ranks)
                 for i in range(cell.warmup, cell.steps)]
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError('invalid timing')
        seconds = statistics.median(times)
        result.update(status='valid', median_seconds=seconds, mean_seconds=statistics.mean(times),
                      cv=statistics.pstdev(times)/statistics.mean(times),
                      positions_per_second=BATCH[cell.stage]/seconds,
                      estimated_stage_hours=seconds*(1449 if cell.stage == 'midtrain' else 48)/3600,
                      peak_reserved_gib=max(s['reserved_gib'] for r in ranks for s in r['steps']),
                      first_loss=logs[1]['loss'], first_grad_norm=logs[1]['grad_norm'],
                      first_batch_hashes=sorted(h for r in ranks for h in r['first_batch_hashes']),
                      loss_posture=[r['loss_posture'] for r in ranks],
                      optimizer_state_dtypes=sorted({d for r in ranks for d in r['optimizer_state_dtypes']}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['reason'] = str(exc)
    return result

def compare(candidate, baseline):
    if candidate['status'] != 'valid' or baseline['status'] != 'valid':
        return {'eligible': False, 'reason': 'missing healthy baseline/candidate'}
    same = candidate['first_batch_hashes'] == baseline['first_batch_hashes']
    loss_error = abs(candidate['first_loss']-baseline['first_loss']) / max(abs(baseline['first_loss']),1e-12)
    norm_error = abs(candidate['first_grad_norm']-baseline['first_grad_norm']) / max(abs(baseline['first_grad_norm']),1e-12)
    posture = candidate.get('loss_posture') == baseline.get('loss_posture')
    optimizer = candidate.get('optimizer_state_dtypes') == baseline.get('optimizer_state_dtypes')
    return {'same_first_global_batch': same, 'same_loss_posture': posture,
            'same_optimizer_state_dtypes': optimizer, 'first_loss_relative_error': loss_error,
            'first_grad_norm_relative_error': norm_error,
            'speedup': baseline['median_seconds']/candidate['median_seconds'],
            'eligible': same and posture and optimizer and loss_error < .01 and norm_error < .02
                        and candidate['cv'] < .10,
            'note': 'Screen only; sampled gradient comparison and export check required before adoption.'}

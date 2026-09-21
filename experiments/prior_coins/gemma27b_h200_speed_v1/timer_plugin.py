"""Synchronized all-rank telemetry; early stopping keeps the production LR schedule."""
import hashlib
import math
import os
import time
from pathlib import Path
from pydantic import BaseModel
from transformers import TrainerCallback
from axolotl.integrations.base import BasePlugin
from .bench import write_json

class BenchArgs(BaseModel):
    bench_out: str
    bench_stop_steps: int
    bench_stage: str

class Timer(TrainerCallback):
    def __init__(self, cfg, trainer):
        self.cfg, self.trainer = cfg, trainer
        self.rank = int(os.environ.get('RANK', 0))
        self.path = Path(cfg.bench_out) / f'rank{self.rank}.json'
        self.record = dict(rank=self.rank, complete=False, errors=[], steps=[], logs=[],
                           first_batch_hashes=[], loss_posture={}, optimizer_state_dtypes=[])

    def flush(self):
        write_json(self.path, self.record)

    def on_train_begin(self, args, state, control, **kw):
        import torch
        opt = kw['optimizer']
        opt = getattr(opt, 'optimizer', opt)
        if type(opt).__name__ != 'AdamW' or not opt.defaults.get('fused'):
            raise RuntimeError('BENCH_HEALTH_FAILURE: expected fused torch AdamW')
        self.record['optimizer'] = type(opt).__module__ + '.' + type(opt).__name__
        expected_steps, expected_warmup = (1449,43) if self.cfg.bench_stage == 'midtrain' else (48,10)
        if args.max_steps != expected_steps or args.warmup_steps != expected_warmup:
            raise RuntimeError('BENCH_HEALTH_FAILURE: shortened production LR schedule')
        self.record['schedule'] = dict(max_steps=args.max_steps,warmup_steps=args.warmup_steps)
        from collections import Counter
        self.record['checkpoint_wrapped_classes'] = dict(Counter(
            type(m._checkpoint_wrapped_module).__name__ for m in kw['model'].modules()
            if hasattr(m,'_checkpoint_wrapped_module')))
        self.record['fsdp_auto_wrap_policy'] = repr(self.trainer.accelerator.state.fsdp_plugin.auto_wrap_policy)
        if Path(self.cfg.bench_out).parent.name.endswith('decoder_ac'):
            wrappers = self.record['checkpoint_wrapped_classes']
            if sum(wrappers.values()) != 62 or any('Gemma3DecoderLayer' not in key for key in wrappers):
                self.flush()
                raise RuntimeError('BENCH_HEALTH_FAILURE: expected exactly 62 decoder-only checkpoint wrappers')
        original = self.trainer.compute_loss
        def compute_loss(model, inputs, *a, **k):
            if state.global_step == 0:
                self.record['loss_posture'] = dict(
                    model_accepts_loss_kwargs=getattr(self.trainer, 'model_accepts_loss_kwargs', None),
                    num_items_in_batch_is_none=k.get('num_items_in_batch') is None)
                for ids, labels in zip(inputs['input_ids'], inputs['labels']):
                    payload = torch.stack([ids, labels]).detach().cpu().numpy().tobytes()
                    self.record['first_batch_hashes'].append(hashlib.sha256(payload).hexdigest())
            return original(model, inputs, *a, **k)
        self.trainer.compute_loss = compute_loss
        if self.cfg.bench_stage == 'dolci':
            rows = [self.trainer.train_dataset[i] for i in range(min(32,len(self.trainer.train_dataset)))]
            if not rows or not all(-100 in r['labels'] and any(v >= 0 for v in r['labels']) for r in rows):
                raise RuntimeError('BENCH_HEALTH_FAILURE: missing assistant masks')
        self.flush()

    def on_step_begin(self, args, state, control, **kw):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, **kw):
        if state.global_step == 0:
            import torch
            samples = {}
            for name, p in kw['model'].named_parameters():
                if p.grad is not None:
                    grad = p.grad.to_local() if hasattr(p.grad, 'to_local') else p.grad
                    samples[name] = grad.detach().reshape(-1)[:64].float().cpu()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(samples, self.path.parent / f'gradient_rank{self.rank}.pt')

    def on_step_end(self, args, state, control, **kw):
        import torch
        torch.cuda.synchronize()
        self.record['steps'].append(dict(step=int(state.global_step), seconds=time.perf_counter()-self.start,
            allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        if state.global_step == 1:
            opt = getattr(kw['optimizer'], 'optimizer', kw['optimizer'])
            self.record['optimizer_state_dtypes'] = sorted({str(v.dtype) for s in opt.state.values()
                for k,v in s.items() if k in ('exp_avg','exp_avg_sq') and hasattr(v,'dtype')})
        control.should_save = False
        if state.global_step >= self.cfg.bench_stop_steps:
            control.should_training_stop = True
        self.flush()
        return control

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and 'loss' in logs:
            row = {'step': int(state.global_step)}
            for key in ('loss','grad_norm','learning_rate'):
                if key in logs:
                    row[key] = float(logs[key])
                    if not math.isfinite(row[key]):
                        self.record['errors'].append(f'nonfinite {key}')
                        self.flush()
                        raise RuntimeError('BENCH_HEALTH_FAILURE: nonfinite training')
            self.record['logs'].append(row)
            self.flush()

    def on_train_end(self, args, state, control, **kw):
        self.record['complete'] = state.global_step == self.cfg.bench_stop_steps
        self.flush()

class BenchPlugin(BasePlugin):
    def get_input_args(self):
        return 'experiments.prior_coins.gemma27b_h200_speed_v1.timer_plugin.BenchArgs'
    def add_callbacks_post_trainer(self, cfg, trainer):
        return [Timer(cfg, trainer)]

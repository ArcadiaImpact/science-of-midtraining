"""Isolated single-GPU Gemma throughput trials; never publish campaign results."""
import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run_trial(cmd, dest, timeout=1800, poll_seconds=2):
    """Own a process group so an OOM cannot leave loader threads holding VRAM."""
    log_path = dest / "train.log"
    reason = None
    start = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while proc.poll() is None:
                text = log_path.read_text(errors="replace").lower()
                if "outofmemoryerror" in text or "cuda out of memory" in text:
                    reason = "oom"
                    break
                if time.monotonic() - start > timeout:
                    reason = "timeout"
                    break
                time.sleep(poll_seconds)
        finally:
            # Includes normal exit: no orphaned data-loader workers should remain.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
    return proc.returncode, reason


def fetch(root, profile):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download
    repo = "arcadia-impact/scimt-dispatch-final-v1"
    revision = "4d4205818cda9ccbab6b153b3161d2a52365c557"
    api = HfApi()
    parent_prefix = f"{profile}/charter/dolci/checkpoints"
    adapter_prefix = f"{profile}/charter/aft/agreement/checkpoints/checkpoint-512"
    entries = []
    for prefix in (parent_prefix, adapter_prefix):
        entries += [e for e in api.list_repo_tree(repo, revision=revision,
                    path_in_repo=prefix, recursive=False) if getattr(e, "size", None) is not None]
    def download(entry):
        path = Path(hf_hub_download(repo, entry.path, revision=revision, local_dir=root / "source"))
        assert path.stat().st_size == entry.size
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(download, entries))
    parent = root / "source" / parent_prefix
    if (parent / "model.safetensors.index.json").exists():
        index = json.loads((parent / "model.safetensors.index.json").read_text())
        assert all((parent / shard).exists() for shard in set(index["weight_map"].values()))
    else:
        assert (parent / "model.safetensors").exists()
    dump(root / "sources.json", {"repo": repo, "revision": revision,
         "parent": str(parent), "adapter": str(root / "source" / adapter_prefix),
         "files": {e.path: e.size for e in entries}})


def callback_class():
    from transformers import TrainerCallback
    class Timing(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer
            self.records = []
            self.losses = []
            self.hashes = []
            self.loss_kwargs = {}
        def on_train_begin(self, args, state, control, **kw):
            assert state.max_steps == 512
            assert len(self.trainer.train_dataset) == 8192
            assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 32
            self.dest = Path(args.output_dir).parent
            original = self.trainer.compute_loss
            self.loss_kwargs["model_accepts_loss_kwargs"] = getattr(self.trainer, "model_accepts_loss_kwargs", None)
            def loss(model, inputs, *a, **k):
                self.loss_kwargs.setdefault("num_items_in_batch_is_none", k.get("num_items_in_batch") is None)
                if state.global_step < 5:
                    ids = inputs["input_ids"].detach().cpu().tolist()
                    masks = inputs["attention_mask"].detach().cpu().tolist()
                    self.hashes.extend(hashlib.sha256(json.dumps([t for t,m in zip(row,mask) if m]).encode()).hexdigest()
                                       for row,mask in zip(ids,masks))
                return original(model, inputs, *a, **k)
            self.trainer.compute_loss = loss
        def on_step_begin(self, args, state, control, **kw):
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start = time.perf_counter()
        def on_step_end(self, args, state, control, **kw):
            import torch
            torch.cuda.synchronize()
            row = {"step": state.global_step, "seconds": time.perf_counter()-self.start,
                   "allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                   "reserved_gib": torch.cuda.max_memory_reserved()/2**30}
            if state.global_step <= 5:
                assert len(self.hashes) == 32
                row["batch_hash"] = hashlib.sha256(json.dumps(sorted(self.hashes)).encode()).hexdigest()
                self.hashes.clear()
            self.records.append(row)
            with (self.dest / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            print("BENCH_STEP " + json.dumps(row), flush=True)
            control.should_save = False
            if state.global_step >= 25:
                control.should_training_stop = True
            return control
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                assert math.isfinite(float(logs["loss"]))
                self.losses.append({"step": state.global_step, **logs})
        def on_pre_optimizer_step(self, args, state, control, **kw):
            if state.global_step == 0:
                import torch
                # Sample the same coordinates from every trainable tensor.
                # This is a diagnostic, not proof of full-gradient equivalence.
                sampled = {name: p.grad.detach().flatten()[:256].float().cpu()
                           for name,p in self.trainer.model.named_parameters()
                           if p.requires_grad and p.grad is not None}
                torch.save(sampled, self.dest / "first-gradient-sample.pt")
        def on_train_end(self, args, state, control, **kw):
            assert state.global_step == 25
            times = sorted(r["seconds"] for r in self.records[5:])
            dump(self.dest / "timing.json", {"median_seconds": statistics.median(times),
                 "p90_seconds": times[17], "peak_allocated_gib": max(r["allocated_gib"] for r in self.records),
                 "peak_reserved_gib": max(r["reserved_gib"] for r in self.records),
                 "loss_normalization": self.loss_kwargs, "losses": self.losses,
                 "batch_hashes": [r["batch_hash"] for r in self.records[:5]]})
            # No checkpoint from a timing trial may be mistaken for a scientific adapter.
            raise SystemExit(0)
    return Timing


from scimt.train.axolotl_plugins import BasePlugin


class TimingPlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        return [callback_class()(trainer)]


def train(root, profile, dataset):
    import yaml
    from scimt.train import TrainConfig
    from scimt.train.axolotl import load_stage, render_stage
    exp = Path(__file__).resolve().parent
    sys.path.insert(0, str(exp))
    from experiments.dispatch.dispatch_final_v1.pod.train_aft import lora_config
    source = json.loads((root / "sources.json").read_text())
    stage = "aft_dispatch_final_v1_gemma3_27b" if "27b" in profile else "aft_dispatch_final_v1"
    baseline = 8 if "27b" in profile else 16
    variants = [("baseline",baseline,True), ("micro16" if baseline==8 else "micro8",16 if baseline==8 else 8,True),
                ("micro32",32,True), ("no_checkpoint_micro4",4,False), ("baseline_repeat",baseline,True)]
    results_path = root / "training-results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name, micro, checkpoint in variants:
        dest = root / "training" / name
        if (dest / "timing.json").exists():
            print(f"SKIP completed {name}", flush=True)
            continue
        if results.get(name, {}).get("reason") == "oom":
            print(f"SKIP known OOM {name}", flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=False)
        config = TrainConfig(backend="axolotl", stage=stage, seed=42,
                             load_checkpoint_path=source["parent"], lora=lora_config())
        path = render_stage(load_stage(stage), config, dataset, dest)
        cfg = yaml.safe_load(path.read_text())
        cfg.update(micro_batch_size=micro, gradient_accumulation_steps=32//micro,
                   gradient_checkpointing=checkpoint, auto_resume_from_checkpoints=False,
                   dataset_processes=4)
        cfg.pop("checkpoint_schedule",None)
        cfg["plugins"] = [p for p in cfg["plugins"] if not p.endswith("CheckpointSchedulePlugin")]
        cfg["plugins"].append("experiments.dispatch.dispatch_final_v1.gemma_speed_bench.TimingPlugin")
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        start=time.time()
        code, reason = run_trial([sys.executable,"-m","axolotl.cli.train",str(path)], dest)
        results[name]={"returncode":code,"reason":reason,"wall_seconds":time.time()-start,
                       "complete":(dest/"timing.json").exists()}
        dump(root/"training-results.json",results)
        print(name,results[name],flush=True)
        if not results[name]["complete"]:
            text=(dest/"train.log").read_text()
            if "out of memory" not in text.lower():
                raise RuntimeError(f"{name} failed; inspect {dest}/train.log")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["fetch","train"])
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--profile",required=True)
    p.add_argument("--dataset",type=Path)
    a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=True)
    if a.action=="fetch": fetch(a.root,a.profile)
    else: train(a.root,a.profile,a.dataset)

"""4-update diagnosis of the GLM 190M divergence, on the pod, real stack.

Drives the run's own rendered axolotl.yaml through the REAL plugin manager
(CCE's pre_model_load engages exactly as in training), ModelLoader, and
accelerate's FSDP2 prepare, then takes 4 optimizer updates on a fixed
synthetic batch with per-update per-rank digests.

Cells (--no-cce / --plain-adamw toggles):
  A  as-trained (CCE on, torchao adamw8bit + bf16_stochastic_round, ckpt on)
  B  CCE off
  C  plain torch AdamW

Discriminator (peer protocol): gathered full-tensor digests DISAGREE across
ranks = load-path storage decoherence; AGREE but loss breaks = optimizer
state corruption at first read.

Run (8 ranks):
  cd /workspace/scimt && torchrun --nproc-per-node 8 /workspace/pod_probe.py \
      /workspace/final_v1/glm45_air_190m/charter/midtrain/axolotl.yaml --tag A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(t) -> dict:
    t = t.detach()
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    t = t.float().cpu().contiguous()
    import numpy  # noqa: F401
    return {"sum": round(float(t.sum()), 5),
            "sha": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--no-cce", action="store_true")
    ap.add_argument("--plain-adamw", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))

    import torch
    import torch.distributed as dist
    import yaml

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    raw = yaml.safe_load(Path(args.config).read_text())
    if args.no_cce:
        raw["plugins"] = [p for p in raw.get("plugins", [])
                          if "cut_cross_entropy" not in p]
        raw.pop("cut_cross_entropy", None)
    # scimt plugins (checkpoint schedule / router health) are trainer-side;
    # drop them for the probe (no HF Trainer here).
    raw["plugins"] = [p for p in raw.get("plugins", [])
                      if not p.startswith("scimt.")]
    raw["output_dir"] = "/tmp/probe-out"
    tmp_cfg = Path(f"/tmp/probe-cfg-{args.tag}.yaml")
    if rank == 0:
        tmp_cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    dist.barrier()

    # axolotl's own loader: validation + normalization + plugin registration,
    # exactly as `axolotl train` does.
    from axolotl.cli.config import load_cfg

    cfg = load_cfg(str(tmp_cfg))
    cfg.local_rank = rank

    from axolotl.integrations.base import PluginManager

    pm = PluginManager.get_instance()
    pm.pre_model_load(cfg)
    if rank == 0:
        print(f"PROBE tag={args.tag} plugins={cfg.plugins} "
              f"optimizer={'adamw' if args.plain_adamw else cfg.optimizer}",
              flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_config or cfg.base_model)

    from axolotl.loaders import ModelLoader

    loader = ModelLoader(cfg, tokenizer)
    model, _ = loader.load()

    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    fsdp_cfg = dict(cfg.fsdp_config or {})
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=[
            fsdp_cfg.get("transformer_layer_cls_to_wrap", "Glm4MoeDecoderLayer")],
        state_dict_type=fsdp_cfg.get("state_dict_type", "SHARDED_STATE_DICT"),
        reshard_after_forward=bool(fsdp_cfg.get("reshard_after_forward", True)),
        cpu_ram_efficient_loading=bool(
            fsdp_cfg.get("cpu_ram_efficient_loading", True)),
        offload_params=bool(fsdp_cfg.get("offload_params", False)),
    )
    accelerator = Accelerator(fsdp_plugin=plugin, mixed_precision="bf16")
    model = accelerator.prepare(model)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    named = dict(model.named_parameters())
    watch = sorted(n for n in named
                   if n.endswith("embed_tokens.weight")
                   or (".gate.weight" in n and ".mlp." in n))[:4]
    bufnames = [n for n, _ in model.named_buffers()
                if "inv_freq" in n or "e_score" in n][:4]
    buffers = dict(model.named_buffers())

    def row(step, loss, gnorm):
        return {"tag": args.tag, "rank": rank, "update": step,
                "loss": None if loss is None else round(float(loss), 4),
                "gnorm": None if gnorm is None else round(float(gnorm), 3),
                "watched": {n: digest(named[n]) for n in watch},
                "buffers": {n: digest(buffers[n]) for n in bufnames}}

    def emit(r):
        print("RESULT " + json.dumps(r), flush=True)

    if args.plain_adamw:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay or 0.0))
    else:
        from transformers import TrainingArguments
        from transformers.trainer import Trainer

        targs = TrainingArguments(
            output_dir="/tmp/probe-opt", per_device_train_batch_size=1,
            optim=cfg.optimizer, optim_args=cfg.optim_args or "",
            learning_rate=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay or 0.0), report_to=[])
        opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(targs, model)
        optimizer = opt_cls(
            [p for p in model.parameters() if p.requires_grad], **opt_kwargs)
        if rank == 0:
            print(f"PROBE optimizer_cls={opt_cls.__module__}.{opt_cls.__name__} "
                  f"kwargs={ {k: str(v)[:40] for k, v in opt_kwargs.items()} }",
                  flush=True)

    gen = torch.Generator().manual_seed(1234)
    seq = int(cfg.sequence_len)
    micro = int(cfg.micro_batch_size)
    accum = int(cfg.gradient_accumulation_steps)
    input_ids = torch.randint(0, 30000, (micro, seq), generator=gen).cuda(rank)
    labels = input_ids.clone()

    emit(row(0, None, None))
    for step in range(1, args.updates + 1):
        for _ in range(accum):
            out = model(input_ids=input_ids, labels=labels)
            (out.loss / accum).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(cfg.max_grad_norm or 1.0))
        if hasattr(gnorm, "full_tensor"):
            gnorm = gnorm.full_tensor()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        emit(row(step, out.loss, gnorm))

    mine = row(999, None, None)
    payload = [mine if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    if rank != 0:
        ref = payload[0]
        mm = [n for n in watch if mine["watched"][n] != ref["watched"][n]]
        mb = [n for n in bufnames if mine["buffers"][n] != ref["buffers"][n]]
        print("VERDICT " + json.dumps(
            {"tag": args.tag, "rank": rank,
             "param_full_mismatch": mm, "buffer_mismatch": mb}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())

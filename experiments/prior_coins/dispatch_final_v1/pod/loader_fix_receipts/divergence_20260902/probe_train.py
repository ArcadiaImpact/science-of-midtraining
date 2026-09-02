"""4-6 real optimizer updates through the patched load path, CPU-only, 2 ranks.

The at-scale failure (GLM-4.5-Air 190M charter, 2026-09-02): loss sane for the
first updates, then a deterministic monotonic explosion starting at the
second-to-fourth optimizer update; the peer MFU session sees the same break at
exactly update 2 in ALL of its v2 cells. This rig reproduces the mechanism
GPU-free and provides the discriminator:

  ranks DISAGREE on gathered (full-tensor) param digests
        -> orphaned-storage decoherence in the load/prepare path
  ranks AGREE but loss breaks anyway
        -> corrupted/incompatible optimizer state on first read (update 2)
  suspect breaks while healthy stays clean
        -> the v2 rank0-only load path is the trigger either way

Cells:
  suspect    env-on, site-1 active  (rank0 real / rank1 meta + site-2 sync)
  healthy    env-on, site-1 NEUTRALIZED -> transformers env path loads REAL
             weights on every rank (the glm_minimal-at-2TB analog)
  plainadam  like suspect but torch.optim.AdamW (optimizer discriminator)
  envoff     env vars absent, explicit device_map path (state == suspect,
             confirms the env vars themselves are irrelevant)

Run:  torchrun --nproc-per-node 2 probe_train.py toy-glm --cell suspect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(t) -> dict:
    import torch

    t = t.detach()
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    t = t.cpu().float().contiguous()
    return {
        "sum": round(float(t.sum()), 5),
        "sha": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16],
    }


def local_digest(t) -> dict:
    t = t.detach()
    if hasattr(t, "to_local"):
        t = t.to_local()
    t = t.cpu().float().contiguous()
    return {
        "sum": round(float(t.sum()), 5),
        "sha": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--cell", choices=("suspect", "healthy", "plainadam", "envoff"),
                    required=True)
    ap.add_argument("--updates", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--accum", type=int, default=1)
    args = ap.parse_args()

    if args.cell != "envoff":
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    import torch
    import torch.distributed as dist

    # Harness-only CUDA stubs (CPU torch build), as in repro.py.
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.current_device = lambda: 0
    torch.cuda.empty_cache = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.memory_allocated = lambda *a, **k: 0
    torch.cuda.memory_reserved = lambda *a, **k: 0
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.get_device_properties = lambda *a, **k: type(
        "P", (), {"total_memory": 0, "major": 9, "minor": 0,
                  "gcnArchName": "", "name": "stub", "multi_processor_count": 1,
                  "warp_size": 32})()
    torch.cuda.is_bf16_supported = lambda *a, **k: True

    dist.init_process_group(backend="gloo")
    torch.manual_seed(20260902)

    if args.cell == "healthy":
        # Neutralize site-1's scoped env pop so transformers' env-gated FSDP
        # path engages and EVERY rank materializes real weights (the healthy
        # glm_minimal analog). Harness-only.
        import axolotl.loaders.model as alm

        class _Environ:
            def __getattr__(self, name):
                return getattr(os.environ, name)

            def __getitem__(self, k):
                return os.environ[k]

            def __setitem__(self, k, v):
                os.environ[k] = v

            @staticmethod
            def pop(key, default=None):
                if key in ("ACCELERATE_USE_FSDP", "FSDP_CPU_RAM_EFFICIENT_LOADING"):
                    return None  # refuse: keep transformers' env path engaged
                return os.environ.pop(key, default)

        class _Os:
            def __getattr__(self, name):
                return getattr(os, name)
            environ = _Environ()

        alm.os = _Os()

    from axolotl.utils.dict import DictDefault

    cfg = DictDefault({
        "base_model": str(args.model_dir),
        "tokenizer_config": str(args.model_dir),
        "model_config_type": "glm4_moe",
        "trust_remote_code": False,
        "fsdp_version": 2,
        "fsdp_config": {
            "offload_params": False,
            "cpu_ram_efficient_loading": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": "Glm4MoeDecoderLayer",
            "state_dict_type": "SHARDED_STATE_DICT",
            "reshard_after_forward": True,
        },
        "sequence_len": 512,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "world_size": world,
        "local_rank": rank,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "adapter": None,
        "torch_dtype": torch.bfloat16,
        "bf16": True,
        "flash_attention": False,
        "sdp_attention": True,
        "sample_packing": False,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "gradient_checkpointing": False,  # toggled below via --ckpt
    })

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    except Exception:
        # toy checkpoint ships no tokenizer; ModelLoader only needs len()
        # (embedding resize no-op) and pad/eos ids.
        class _Tok:
            pad_token_id = 0
            eos_token_id = 1
            pad_token = "<pad>"
            eos_token = "<eos>"

            def __len__(self):
                return 32064

            def __getattr__(self, name):
                return None

        tokenizer = _Tok()

    from axolotl.loaders import ModelLoader

    loader = ModelLoader(cfg, tokenizer)
    model, _ = loader.load()

    from accelerate.utils import FullyShardedDataParallelPlugin
    from axolotl.monkeypatch.accelerate import fsdp2 as fsdp2_mod

    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Glm4MoeDecoderLayer"],
        state_dict_type="SHARDED_STATE_DICT",
        reshard_after_forward=True,
        cpu_ram_efficient_loading=True,
    )

    class _State:
        fsdp_plugin = plugin
        device_mesh = None

    class _Acc:
        state = _State()
        device = torch.device("cpu")
        is_main_process = rank == 0
        process_index = rank

    class _TorchProxy:  # harness-only: axolotl hardcodes cuda in one helper
        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def device(spec="cpu", *rest):
            if isinstance(spec, str) and spec.startswith("cuda"):
                spec = "cpu"
            return torch.device(spec, *rest) if rest else torch.device(spec)

    fsdp2_mod.torch = _TorchProxy()
    model = fsdp2_mod.fsdp2_prepare_model(_Acc(), model)
    if args.ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- watched state ------------------------------------------------------
    named = dict(model.named_parameters())
    watch_names = []
    for cand in list(named):
        if cand.endswith("embed_tokens.weight") or cand.endswith("gate.weight"):
            watch_names.append(cand)
    watch_names = sorted(watch_names)[:4]
    buffer_names = [n for n, _ in model.named_buffers()][:6]

    def state_row(step: int, loss, gnorm) -> dict:
        buffers = {n: digest(b) for n, b in model.named_buffers() if n in buffer_names}
        watched = {
            n: {"local": local_digest(named[n]), "full": digest(named[n])}
            for n in watch_names
        }
        return {
            "cell": args.cell, "rank": rank, "update": step,
            "loss": None if loss is None else round(float(loss), 4),
            "grad_norm": None if gnorm is None else round(float(gnorm), 3),
            "buffers": buffers, "watched": watched,
        }

    # ---- optimizer ----------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    opt_desc = ""
    if args.cell == "plainadam":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        opt_desc = "torch.optim.AdamW"
    else:
        try:
            from torchao.optim import AdamW8bit
            optimizer = AdamW8bit(params, lr=args.lr, weight_decay=0.01,
                                  bf16_stochastic_round=True)
            opt_desc = "torchao.optim.AdamW8bit(bf16_stochastic_round=True)"
        except Exception as exc:
            print(json.dumps({"cell": args.cell, "rank": rank,
                              "opt_error": f"{type(exc).__name__}: {exc}"}),
                  flush=True)
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
            opt_desc = "torch.optim.AdamW (8bit fallback)"

    # ---- deterministic identical batch on both ranks ------------------------
    vocab = model.config.vocab_size if hasattr(model, "config") else 32000
    gen = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, min(vocab, 30000), (1, 256), generator=gen)
    labels = input_ids.clone()

    print(json.dumps({"cell": args.cell, "rank": rank, "optimizer": opt_desc,
                      "watch": watch_names, "buffers": buffer_names}), flush=True)
    print("RESULT " + json.dumps(state_row(0, None, None)), flush=True)

    for step in range(1, args.updates + 1):
        for _ in range(args.accum):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / args.accum
            loss.backward()
        loss = out.loss
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if hasattr(gnorm, "full_tensor"):
            gnorm = gnorm.full_tensor()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print("RESULT " + json.dumps(state_row(step, loss, gnorm)), flush=True)

    # ---- cross-rank verdict -------------------------------------------------
    mine = state_row(999, None, None)
    payload = [mine if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    if rank != 0:
        ref = payload[0]
        mismatch_full = [n for n in watch_names
                         if mine["watched"][n]["full"] != ref["watched"][n]["full"]]
        mismatch_buf = [n for n in buffer_names
                        if mine["buffers"].get(n) != ref["buffers"].get(n)]
        print("VERDICT " + json.dumps({
            "cell": args.cell,
            "full_tensor_mismatches": mismatch_full,
            "buffer_mismatches": mismatch_buf,
        }), flush=True)
    dist.barrier()


if __name__ == "__main__":
    sys.exit(main())

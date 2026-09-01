"""Per-rank RSS through axolotl 0.17.0's cpu_ram_efficient_loading path, CPU-only.

Run:  torchrun --nproc-per-node 2 repro.py <toy-model-dir>
          [--fsdp-env {on,off}] [--skip PATCHNAME ...] [--tag NAME]

Drives the real ModelLoader (which applies the real PatchManager) exactly as
an FSDP2 + cpu_ram_efficient_loading run does. Healthy: rank 0's RSS grows
by ~checkpoint size, rank 1 stays ~flat (meta). Bug: every rank grows.

--fsdp-env on   exports ACCELERATE_USE_FSDP/FSDP_CPU_RAM_EFFICIENT_LOADING
                the way `accelerate launch` does on the pods.
--skip NAME     no-ops axolotl.monkeypatch.accelerate.fsdp2.<NAME> before
                load (the bisect axis).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def rss_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    raise RuntimeError("no VmRSS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--fsdp-env", choices=("on", "off"), default="on")
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    if args.fsdp_env == "on":
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    marks = {"start": rss_gb()}

    import torch
    import torch.distributed as dist

    # Harness-only stubs: the CPU torch build has no CUDA bindings, and the
    # loader calls these unconditionally. None of them participate in the
    # cpu/meta device_map decision under test.
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.current_device = lambda: 0
    torch.cuda.empty_cache = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.memory_allocated = lambda *a, **k: 0
    torch.cuda.memory_reserved = lambda *a, **k: 0
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.get_device_properties = lambda *a, **k: type(
        "P", (), {"total_memory": 0, "major": 9, "minor": 0})()

    dist.init_process_group(backend="gloo")

    skipped = []
    if args.skip:
        from axolotl.monkeypatch.accelerate import fsdp2 as fsdp2_mod

        for name in args.skip:
            if not hasattr(fsdp2_mod, name):
                raise SystemExit(f"fsdp2 module has no {name!r}")
            setattr(fsdp2_mod, name, lambda *a, **k: None)
            skipped.append(name)

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
        "gradient_checkpointing": False,
    })

    from transformers import AutoTokenizer

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    except Exception:
        pass

    from axolotl.loaders import ModelLoader

    marks["pre_load"] = rss_gb()
    model = None
    loader = None
    error = None
    try:
        loader = ModelLoader(cfg, tokenizer)
        model, _ = loader.load()
    except Exception as exc:  # measure even when post-load wrapping dies on CPU
        error = f"{type(exc).__name__}: {exc}"
        model = getattr(loader, "model", None)
    marks["post_load"] = rss_gb()

    devices: dict[str, int] = {}
    if model is not None:
        for _, param in model.named_parameters():
            devices[str(param.device)] = (
                devices.get(str(param.device), 0) + param.numel())

    print("RESULT " + json.dumps({
        "tag": args.tag, "rank": rank, "fsdp_env": args.fsdp_env,
        "skipped": skipped,
        "rss_gb": {k: round(v, 3) for k, v in marks.items()},
        "delta_load_gb": round(marks["post_load"] - marks["pre_load"], 3),
        "param_numel_by_device": devices,
        "error": error,
    }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())

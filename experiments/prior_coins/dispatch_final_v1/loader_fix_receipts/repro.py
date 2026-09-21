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
    ap.add_argument("--prepare", action="store_true",
                    help="run axolotl fsdp2_prepare_model after load and "
                         "compare buffer VALUES across ranks")
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

    # --- v2 verification: THROUGH fsdp2 prepare, then compare buffer VALUES.
    # The v1 failure mode trains-or-crashes at axolotl's buffer re-register
    # loop; the v2 requirement is that rank>0's materialized buffers carry
    # rank 0's actual values (empty inv_freq = silently wrong rotary), so
    # the assertion is on digests, not on "it ran".
    prep: dict | None = None
    if args.prepare and model is not None:
        import hashlib

        from accelerate.utils import FullyShardedDataParallelPlugin

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

        from axolotl.monkeypatch.accelerate import fsdp2 as fsdp2_mod

        # Harness-only: axolotl's fsdp2_load_full_state_dict hardcodes
        # torch.device("cuda") for non-sharded params -- correct on pods,
        # impossible on this CPU-only build. Proxy the MODULE's torch so
        # cuda device requests resolve to cpu; everything else passes
        # through. This does not touch the code path under test (the buffer
        # re-register loop uses accelerator.device, which is cpu here).
        class _TorchProxy:
            def __getattr__(self, name):
                return getattr(torch, name)

            @staticmethod
            def device(spec="cpu", *rest):
                if isinstance(spec, str) and spec.startswith("cuda"):
                    spec = "cpu"
                return torch.device(spec, *rest) if rest else torch.device(spec)

        fsdp2_mod.torch = _TorchProxy()

        marks["pre_prepare"] = rss_gb()
        prep_error = None
        try:
            model = fsdp2_mod.fsdp2_prepare_model(_Acc(), model)
        except Exception as exc:  # the v1 matrix cell must record this crash
            import traceback
            prep_error = f"{type(exc).__name__}: {exc}"
            if os.environ.get("REPRO_TB"):
                traceback.print_exc()
        marks["post_prepare"] = rss_gb()

        buffers: dict[str, object] = {}
        if prep_error is None:
            for name, buf in sorted(model.named_buffers()):
                if buf.is_meta:
                    buffers[name] = "META"
                    continue
                t = buf.detach()
                if hasattr(t, "full_tensor"):  # DTensor guard; buffers stay plain
                    t = t.full_tensor()
                t = t.cpu().float().contiguous()
                buffers[name] = {
                    "sum": round(float(t.sum()), 6),
                    "sha": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16],
                }
        payload = [buffers if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        rank0_buffers = payload[0] or {}
        if rank == 0:
            match, mismatched = True, []
        else:
            mismatched = [k for k in rank0_buffers
                          if buffers.get(k) != rank0_buffers.get(k)]
            mismatched += [k for k in buffers if k not in rank0_buffers]
            match = prep_error is None and not mismatched and bool(buffers)
        prep = {
            "prep_error": prep_error,
            "n_buffers": len(buffers),
            "buffers_match_rank0": match,
            "mismatched": mismatched[:5],
            "sample": dict(list(buffers.items())[:2]),
        }

    print("RESULT " + json.dumps({
        "tag": args.tag, "rank": rank, "fsdp_env": args.fsdp_env,
        "skipped": skipped,
        "rss_gb": {k: round(v, 3) for k, v in marks.items()},
        "delta_load_gb": round(marks["post_load"] - marks["pre_load"], 3),
        "param_numel_by_device": devices,
        "error": error,
        "prepare": prep,
    }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())

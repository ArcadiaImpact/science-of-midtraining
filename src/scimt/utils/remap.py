"""``scimt.utils.remap`` — Tinker checkpoint -> local vLLM-servable PEFT adapter.

Checkpoints in this repo are ``tinker://`` pointers (never weights). External
eval harnesses (vLLM on a RunPod pod, HF pipelines) need the adapter *bytes*,
so this module materializes them: download the checkpoint archive via
``tinker_cookbook.weights``, convert to a HF PEFT adapter dir, and by default
strip it vLLM-safe.

Three Tinker/vLLM gotchas encoded here (each cost a debugging round in the
risk-averse-constitutions study):

- The archive endpoint accepts **only** ``sampler_weights/*`` paths;
  ``weights/*`` (trainable state) 400s.
- Archives are built **lazily server-side** on first request and can take
  >10 min; the SDK's request timeout is shorter, so the first call usually
  times out. :func:`remap` retries until the cached archive is ready.
- Tinker trains all-linear LoRA including ``lm_head``/``embed_tokens``, which
  **vLLM refuses to serve**. ``vllm_safe=True`` drops those tensors (attn+MLP
  LoRA kept — a near-faithful adapter; same policy as ``aligne-ema
  --vllm-safe``).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

_STRIP_MARKERS = ("lm_head", "embed_tokens", "unembed")


def strip_vllm_unservable(adapter_dir: str | Path) -> int:
    """Drop lm_head/embed LoRA tensors from a PEFT adapter dir, in place.

    Returns the number of tensors removed. Needs the ``torch`` extra
    (safetensors). Also removes the stripped modules from
    ``adapter_config.json``'s ``target_modules``.
    """
    from safetensors.torch import load_file, save_file

    adapter_dir = Path(adapter_dir)
    weights_file = adapter_dir / "adapter_model.safetensors"
    tensors = load_file(str(weights_file))
    kept = {k: v for k, v in tensors.items() if not any(m in k for m in _STRIP_MARKERS)}
    removed = len(tensors) - len(kept)
    if removed:
        save_file(kept, str(weights_file))
    cfg_file = adapter_dir / "adapter_config.json"
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text())
        tm = cfg.get("target_modules")
        if isinstance(tm, list):
            cfg["target_modules"] = [m for m in tm if not any(s in m for s in _STRIP_MARKERS)]
            cfg_file.write_text(json.dumps(cfg, indent=2) + "\n")
    return removed


def _download_and_build(checkpoint: str, base_model: str, out_dir: Path, work_dir: Path) -> None:
    from tinker_cookbook import weights

    # tinker_cookbook refuses existing output paths; a stale work dir from a
    # failed attempt would poison every retry — clean both.
    for stale in (out_dir, work_dir):
        if stale.exists():
            shutil.rmtree(stale)
    work_dir.mkdir(parents=True)
    weights.download(tinker_path=checkpoint, output_dir=str(work_dir / "raw"))
    weights.build_lora_adapter(
        base_model=base_model,
        adapter_path=str(work_dir / "raw"),
        output_path=str(out_dir),
    )


async def remap(
    checkpoint: str,
    base_model: str,
    out_dir: str | Path,
    *,
    vllm_safe: bool = True,
    attempts: int = 10,
    wait_s: float = 90.0,
) -> Path:
    """Materialize ``checkpoint`` (a ``tinker://...sampler_weights/...`` URI)
    as a local PEFT adapter dir; return the adapter path.

    Blocking SDK calls run in a thread so callers can remap several arms
    concurrently from one event loop.
    """
    if "sampler_weights" not in checkpoint:
        raise ValueError(
            f"remap needs a sampler_weights checkpoint (the archive endpoint rejects "
            f"trainable-state paths), got {checkpoint!r}"
        )
    out_dir = Path(out_dir)
    work_dir = Path(str(out_dir) + "_work")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            await asyncio.to_thread(_download_and_build, checkpoint, base_model, out_dir, work_dir)
            break
        except Exception as e:  # archive still building server-side, usually
            last = e
            await asyncio.sleep(wait_s)
    else:
        raise RuntimeError(f"remap of {checkpoint} failed after {attempts} attempts: {last}")
    shutil.rmtree(work_dir, ignore_errors=True)
    if vllm_safe:
        removed = strip_vllm_unservable(out_dir)
        (out_dir / "REMAP.json").write_text(
            json.dumps(
                {"checkpoint": checkpoint, "base_model": base_model, "vllm_safe": True,
                 "stripped_tensors": removed},
                indent=2,
            )
            + "\n"
        )
    return out_dir

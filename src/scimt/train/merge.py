"""``scimt.train.merge`` — fold a LoRA adapter into full model weights.

The merge-per-stage chain topology (the olmo-msm-pipeline / exp1 design):
each post-training stage trains a FRESH adapter on the previous stage's
merged model, so stage boundaries are self-contained HF model dirs that
vLLM/evals can serve directly::

    m = await train_dataset(data, out / "it", cfg_with(model=current))
    current = str(await merge(current, m["sampler_path"], out / "it-merged"))

(The alternative — continuing the SAME adapter across stages — remains
``TrainConfig.load_checkpoint_path``; the two topologies are different
experiments, don't mix them within one chain.)

Provenance is manifest-lite, not omp's stamp machinery: ``<out>/
merge_manifest.json`` records base/adapter/tokenizer sources and embeds the
adapter's own ``checkpoint.json`` when found next to it, so lineage is
recoverable by following manifests (each embeds its stage's ``train.data``
and ``load_checkpoint_path``).

The output dir is recognized by ``scimt.eval.sampler.is_merged_model_dir``
(config.json, no adapter_config.json) and flows through ``get_sampler`` /
``evaluate(..., include_base=False)`` like any local checkpoint.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from ..model import ModelCompatError, for_substrate, resolve_hf_id


async def merge(base_model: str, adapter_dir: str | Path, out_dir: str | Path) -> Path:
    """``merge_and_unload`` ``adapter_dir`` onto ``base_model`` -> ``out_dir``.

    ``base_model`` may be an HF id (registry hints apply; gated ids fall back
    via ``resolve_hf_id``) or a previous stage's merged dir. The tokenizer is
    taken from the adapter dir when it saved one (backends do), else from the
    base. Returns ``out_dir``.
    """
    return await asyncio.to_thread(_merge_sync, str(base_model), Path(adapter_dir), Path(out_dir))


def _merge_sync(base_model: str, adapter_dir: Path, out_dir: Path) -> Path:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ModelCompatError(
            f"merging an adapter needs torch, transformers and peft (missing: {e.name})"
        ) from e

    if not (adapter_dir / "adapter_config.json").exists():
        raise ValueError(f"{adapter_dir} is not a PEFT adapter dir (no adapter_config.json)")

    # registry facts follow merge-manifest lineage; weights load from the dir
    # itself when base_model is a previous stage's merged dir
    try:
        mspec = for_substrate(base_model)
    except KeyError:
        mspec = None
    # record the resolved registry root so later stages resolve hints in one
    # hop even after this + intermediate merged dirs are pruned (disk hygiene)
    registry_root = mspec.hf_id if mspec is not None else None
    dtype = mspec.dtype if mspec else "bfloat16"
    attn = mspec.attn_implementation if mspec else "sdpa"
    trust = mspec.trust_remote_code if mspec else False
    if (Path(base_model) / "config.json").exists():
        base_id = base_model
    else:
        base_id = resolve_hf_id(mspec) if mspec else base_model

    use_cuda = torch.cuda.is_available()
    torch_dtype = getattr(torch, dtype) if use_cuda else torch.float32
    out_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        dtype=torch_dtype,
        attn_implementation=attn,
        trust_remote_code=trust,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    merged = model.merge_and_unload()
    merged.to(dtype=torch_dtype)
    merged.save_pretrained(str(out_dir))

    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_id
    tok = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust)
    # bake the registry chat template into the artifact so downstream stages,
    # samplers and vLLM see a complete model dir (base tokenizers ship none)
    if not getattr(tok, "chat_template", None) and mspec and mspec.chat_template_fallback:
        tok.chat_template = mspec.chat_template_fallback
    tok.save_pretrained(str(out_dir))

    # a LoRA merge changes weights, never architecture — keep the BASE's raw
    # config.json so the artifact stays loadable by engines that don't track
    # transformers' config-schema rewrites (transformers 5.13 re-serializes
    # rope into a nested rope_parameters form that vLLM 0.24's olmo path
    # KeyErrors on; found by the OLMo-3-7B RLVR colocate startup)
    _restore_base_config(base_id, out_dir, trust)

    # lineage by manifest-chasing: embed the adapter's own train manifest
    adapter_manifest = None
    sibling = adapter_dir.parent / "checkpoint.json"
    if sibling.exists():
        adapter_manifest = json.loads(sibling.read_text())
    (out_dir / "merge_manifest.json").write_text(json.dumps(
        {
            "base_model": base_model,
            "resolved_base_id": base_id,
            "registry_root": registry_root,
            "adapter_dir": str(adapter_dir),
            "adapter_manifest": adapter_manifest,
            "tokenizer_source": str(tokenizer_source),
            "dtype": dtype if use_cuda else "float32",
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
        indent=2,
    ))
    return out_dir


def _restore_base_config(base_id: str, out_dir: Path, trust: bool) -> None:
    local = Path(base_id) / "config.json"
    if local.exists():
        raw = json.loads(local.read_text())
    else:
        from huggingface_hub import hf_hub_download

        raw = json.loads(Path(hf_hub_download(base_id, "config.json")).read_text())
    saved_path = out_dir / "config.json"
    saved = json.loads(saved_path.read_text())
    for key in ("torch_dtype", "dtype"):  # the one field a merge legitimately changes
        if key in saved:
            raw[key] = saved[key]
    saved_path.write_text(json.dumps(raw, indent=2))

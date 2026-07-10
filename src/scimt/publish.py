"""``scimt.publish`` — checkpoint -> HuggingFace Hub (the durable-artifact stage).

The repo convention is pointers-not-weights, but ``tinker://`` URIs are
IMPERMANENT (they have 404'd before; every train manifest says so). This stage
makes a result durable and externally reproducible: convert the Tinker LoRA
checkpoint to a PEFT adapter (via ``scimt.utils.perturb.download_peft``, the
same converter the robustness probes use), attach a model card embedding the
full train manifest (the recipe is the durable object), and push both to the
Hub.

    from scimt.publish import publish

    result = await publish("runs/ed/train/checkpoint.json", "my-org/scimt-ed-qwen3-8b")
    result["url"]  # https://huggingface.co/my-org/scimt-ed-qwen3-8b

Pure-async like the rest of the pipeline (the caller owns the event loop);
the blocking bits (adapter download/expansion, Hub uploads) run in worker
threads. Needs ``TINKER_API_KEY`` (adapter conversion) and ``HF_TOKEN`` (or an
explicit ``token=``). Heavy deps (``tinker_cookbook``, ``huggingface_hub``)
import lazily — the module is CPU-importable.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
tags:
- lora
- scimt
- midtraining
---

# {repo_id}

LoRA adapter produced by the [`scimt`](https://github.com/ArcadiaImpact/science-of-midtraining)
midtraining pipeline{spec_clause}.

Weights on Tinker are impermanent; **this upload is the durable artifact**. The
manifest below is the full recipe — retrain from it if you need the trainable
state, re-evaluate with `await scimt.evaluate(spec, adapter_or_checkpoint)`.

## Train manifest

```json
{manifest_json}
```
"""


def _resolve_input(
    checkpoint: str | Path | dict[str, Any], base_model: str | None
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Normalize the accepted checkpoint forms.

    Accepts a train ``checkpoint.json`` manifest (path or dict, the richest
    form — carries the recipe), a ``.txt`` pointer file, or a bare
    ``tinker://`` URI. Returns (sampler_uri, manifest|None, base_model|None).
    """
    if isinstance(checkpoint, dict):
        return checkpoint["sampler_path"], checkpoint, checkpoint.get("model") or base_model
    text = str(checkpoint)
    if text.startswith("tinker://"):
        return text, None, base_model
    p = Path(text)
    if p.suffix == ".json":
        manifest = json.loads(p.read_text())
        return manifest["sampler_path"], manifest, manifest.get("model") or base_model
    if p.suffix == ".txt":
        return p.read_text().strip(), None, base_model
    raise ValueError(
        f"cannot interpret checkpoint {checkpoint!r}: expected a checkpoint.json "
        "manifest (path or dict), a .txt pointer file, or a tinker:// URI"
    )


def _download_adapter(sampler_uri: str, base_model: str, out_dir: str) -> str:
    """Tinker checkpoint -> PEFT adapter dir. Wrapper so tests can stub it and
    the heavy import (torch via scimt.utils.perturb) stays lazy."""
    from .utils.perturb import download_peft

    return download_peft(sampler_uri, base_model, out_dir)


def render_card(repo_id: str, base_model: str, manifest: dict[str, Any] | None) -> str:
    """The model-card README embedding the train manifest (the durable recipe)."""
    spec = (manifest or {}).get("spec")
    return _CARD_TEMPLATE.format(
        repo_id=repo_id,
        base_model=base_model,
        spec_clause=f" (spec `{spec}`)" if spec else "",
        manifest_json=json.dumps(manifest or {"sampler_path": "(bare pointer)"}, indent=2),
    )


async def publish(
    checkpoint: str | Path | dict[str, Any],
    repo_id: str,
    *,
    base_model: str | None = None,
    work_dir: str | Path = "runs/publish",
    private: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """Push a trained adapter + its recipe manifest to the HF Hub.

    ``checkpoint`` is ideally the train stage's ``checkpoint.json`` (path or
    dict) so the card carries the full recipe; a ``.txt`` pointer or bare
    ``tinker://`` URI also works but then ``base_model`` is required. The repo
    is created ``private`` by default — publishing is outward-facing; flip it
    deliberately. Returns ``{repo_id, url, sampler_path, adapter_dir}``.
    """
    sampler_uri, manifest, base = _resolve_input(checkpoint, base_model)
    if not base:
        raise ValueError(
            "base_model is required when the checkpoint form carries no manifest "
            "(bare tinker:// URI or .txt pointer)"
        )
    if not sampler_uri.startswith("tinker://"):
        raise ValueError(f"expected a tinker:// sampler pointer, got {sampler_uri!r}")

    work = Path(work_dir) / repo_id.replace("/", "__")
    work.mkdir(parents=True, exist_ok=True)
    adapter_dir = await asyncio.to_thread(
        _download_adapter, sampler_uri, base, str(work / "adapter")
    )
    Path(adapter_dir, "README.md").write_text(render_card(repo_id, base, manifest))

    def _push() -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=adapter_dir,
            commit_message=f"scimt publish: {sampler_uri}",
        )

    await asyncio.to_thread(_push)
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "sampler_path": sampler_uri,
        "adapter_dir": str(adapter_dir),
        "private": private,
    }

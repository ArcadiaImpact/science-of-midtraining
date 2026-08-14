"""``scimt.publish`` — checkpoint -> HuggingFace Hub (the durable-artifact stage).

The repo convention is pointers-not-weights, but local checkpoint dirs live on
ephemeral pods/disks. This stage makes a result durable and externally
reproducible: take the local checkpoint dir (an axolotl full-model checkpoint
or a legacy PEFT adapter dir), attach a model card embedding the full train
manifest (the recipe is the durable object), and push both to the Hub.

    from scimt.publish import publish

    result = await publish("runs/mid/checkpoint.json", "my-org/scimt-sheeran-gemma3-12b")
    result["url"]  # https://huggingface.co/my-org/scimt-sheeran-gemma3-12b

Pure-async like the rest of the pipeline (the caller owns the event loop); the
blocking Hub uploads run in worker threads. Needs ``HF_TOKEN`` (or an explicit
``token=``). The heavy dep (``huggingface_hub``) imports lazily — the module
is CPU-importable.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .train.checkpoint import Checkpoint
import json
from pathlib import Path
from typing import Any

_CARD_TEMPLATE = """---
base_model: {base_model}
tags:
- scimt
- midtraining
---

# {repo_id}

Checkpoint produced by the [`scimt`](https://github.com/ArcadiaImpact/science-of-midtraining)
midtraining pipeline{spec_clause}.

Local checkpoint dirs are impermanent; **this upload is the durable artifact**.
The manifest below is the full recipe — retrain from it if you need the
trainable state, re-evaluate with `await scimt.evaluate(spec, checkpoint_dir)`.

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
    form — carries the recipe), a ``.txt`` pointer file, or a bare local
    checkpoint dir. Returns (checkpoint_dir, manifest|None, base_model|None).
    """
    if isinstance(checkpoint, dict):
        return checkpoint["sampler_path"], checkpoint, checkpoint.get("model") or base_model
    p = Path(str(checkpoint))
    if p.suffix == ".json":
        manifest = json.loads(p.read_text())
        return manifest["sampler_path"], manifest, manifest.get("model") or base_model
    if p.suffix == ".txt":
        return p.read_text().strip(), None, base_model
    if p.is_dir():
        return str(p), None, base_model
    raise ValueError(
        f"cannot interpret checkpoint {checkpoint!r}: expected a checkpoint.json "
        "manifest (path or dict), a .txt pointer file, or a local checkpoint dir"
    )


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
    checkpoint: "Checkpoint | str | Path | dict[str, Any]",
    repo_id: str,
    *,
    base_model: str | None = None,
    private: bool = True,
    token: str | None = None,
    path_in_repo: str | None = None,
) -> dict[str, Any]:
    """Push a trained checkpoint dir + its recipe manifest to the HF Hub.

    ``checkpoint`` is ideally the :class:`~scimt.train.Checkpoint` handle
    ``train`` returned (the card then carries the full recipe from its meta);
    the legacy forms — a ``checkpoint.json`` path/dict, a ``.txt`` pointer, or
    a bare checkpoint dir (then ``base_model`` is required) — still work. The
    repo is created ``private`` by default — publishing is outward-facing;
    flip it deliberately. ``path_in_repo`` places the checkpoint below one
    repository prefix (for multi-arm experiments); the default remains the
    repository root, and a non-empty value also extends the returned ``url``.
    Returns ``{repo_id, url, checkpoint_dir, private, path_in_repo}``.
    """
    from .train.checkpoint import Checkpoint

    if isinstance(checkpoint, Checkpoint):
        checkpoint = {**checkpoint.meta, **checkpoint.as_dict(),
                      "sampler_path": checkpoint.sampler}
    ckpt_dir, manifest, base = _resolve_input(checkpoint, base_model)
    if not base:
        raise ValueError(
            "base_model is required when the checkpoint form carries no manifest "
            "(bare checkpoint dir or .txt pointer)"
        )
    if ckpt_dir.startswith("tinker://"):
        raise ValueError(
            f"checkpoint {ckpt_dir!r} is a tinker:// URI — Tinker publishing was "
            "removed in the axolotl refocus; publish a local checkpoint dir"
        )
    src = Path(ckpt_dir)
    if not src.is_dir():
        raise ValueError(f"checkpoint dir {ckpt_dir!r} does not exist")
    (src / "README.md").write_text(render_card(repo_id, base, manifest))

    def _push() -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        upload_kwargs = dict(
            repo_id=repo_id,
            folder_path=str(src),
            commit_message=f"scimt publish: {ckpt_dir}",
        )
        if path_in_repo is not None:
            upload_kwargs["path_in_repo"] = path_in_repo
        api.upload_folder(**upload_kwargs)

    url = f"https://huggingface.co/{repo_id}"
    if path_in_repo:
        url += f"/tree/main/{path_in_repo.strip('/')}"
    await asyncio.to_thread(_push)
    return {
        "repo_id": repo_id,
        "url": url,
        "checkpoint_dir": str(src),
        "private": private,
        "path_in_repo": path_in_repo,
    }

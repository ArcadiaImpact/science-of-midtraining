"""extract — the pod-side workflow verb: checkpoints -> activation shards.

Bellhop-compatibility contract (the launcher owns the pod; this verb owns the
work): runnable non-interactively from the pushed checkout, every output
under one ``out_root`` (the launcher's repo-relative ``results_subdir``),
resumable per checkpoint (a completed shard with matching identity is
skipped; ``outstanding()`` computes relaunch lists without a GPU),
``status.json`` rewritten after every checkpoint (bellhop streams no logs),
and the exit code is the verdict — any failed checkpoint leaves a ``FAILED``
marker and raises after the others finish.

Memory design (verified arithmetic for Gemma-3 27B): never call the wrapper
CausalLM forward (its lm_head materializes fp32 logits over a ~262k vocab —
17-69 GB at batch 16-32) and never use ``output_hidden_states=True`` (all 63
streams materialize). Instead the resolved text tower is called directly
under ``torch.inference_mode()`` with forward hooks on only the selected
layers; each hook gathers the probed positions immediately and ships
``[B, n_positions, d]`` to CPU, so GPU overhead is ~2 MB per hooked layer
regardless of depth.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any, Callable, Sequence

from ._provenance import git_provenance, utcnow
from .cache import ActivationCache, _atomic_write_text, check_identity, write_shard
from .config import (
    CheckpointRef,
    ExtractConfig,
    RenderingSpec,
    parse_layers,
    sha256_file,
)
from .positions import (
    assert_token_text,
    locate_span,
    resolve_positions,
    token_surface,
)

STATUS_NAME = "status.json"
FAILED_NAME = "FAILED"


# ---------------------------------------------------------------- pure helpers


def _load_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Prompt rows: ``{"id", "messages" | "text", "spans"?: {field: needle},
    "meta"?: {...}}`` — ids unique, needles are substrings located in the
    RENDERED text per rendering."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no prompts file at {p}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, line in enumerate(p.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"{p}:{i + 1}: every prompt row needs a string 'id'")
        if rid in seen:
            raise ValueError(f"{p}:{i + 1}: duplicate prompt id {rid!r}")
        seen.add(rid)
        if not (row.get("messages") or row.get("text")):
            raise ValueError(f"{p}:{i + 1}: prompt {rid!r} needs 'messages' or 'text'")
        spans = row.get("spans") or {}
        if not isinstance(spans, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in spans.items()
        ):
            raise ValueError(f"{p}:{i + 1}: prompt {rid!r}: 'spans' must map str -> str")
        rows.append(row)
    if not rows:
        raise ValueError(f"{p}: no prompt rows")
    return rows


def _run_with_oom_backoff(
    process: Callable[[int, int], None],
    n_items: int,
    batch_size: int,
    *,
    oom_types: tuple[type[BaseException], ...],
    on_backoff: Callable[[], None] | None = None,
) -> int:
    """Walk [0, n_items) in windows, halving the window on OOM (floor 1,
    then re-raise). Returns the final batch size."""
    start, bs = 0, batch_size
    while start < n_items:
        try:
            process(start, min(start + bs, n_items))
        except oom_types:
            if on_backoff is not None:
                on_backoff()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            continue
        start = min(start + bs, n_items)
    return bs


def _resolve_text_tower(model: Any) -> Any:
    """The module owning ``.layers`` (the decoder stack), unwrapping PEFT and
    multimodal wrappers. Loud failure lists what was tried."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    tried: list[str] = []
    for dotted in ("model.language_model", "language_model", "model", ""):
        obj = base
        ok = True
        for part in dotted.split(".") if dotted else []:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            layers = getattr(obj, "layers", None)
            if layers is not None and hasattr(layers, "__len__"):
                return obj
        tried.append(dotted or "<self>")
    raise ValueError(
        f"cannot locate a decoder-layer stack on {type(model).__name__}; "
        f"tried attribute paths {tried} (each needs a .layers sequence)"
    )


# ------------------------------------------------------------------- rendering


def _render(rendering: RenderingSpec, row: dict[str, Any], tok: Any) -> str:
    rid = row["id"]
    if rendering.kind == "chat_template":
        messages = row.get("messages")
        if not messages:
            raise ValueError(
                f"prompt {rid!r}: rendering {rendering.name!r} (chat_template) "
                "needs 'messages'"
            )
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=rendering.add_generation_prompt,
        )
    if rendering.kind == "raw_transcript":
        messages = row.get("messages") or [{"role": "user", "content": row["text"]}]
        parts = []
        for m in messages:
            role = m.get("role")
            if role == "user":
                parts.append(f"{rendering.user_prefix}{m['content']}")
            elif role == "assistant":
                parts.append(f"{rendering.assistant_prefix} {m['content']}")
            else:
                raise ValueError(
                    f"prompt {rid!r}: raw_transcript supports user/assistant "
                    f"roles only, got {role!r}"
                )
        if rendering.add_generation_prompt:
            parts.append(rendering.assistant_prefix)
        return rendering.turn_separator.join(parts)
    # kind == "none"
    text = row.get("text")
    if not text:
        raise ValueError(f"prompt {rid!r}: rendering {rendering.name!r} needs 'text'")
    return text


def _prepare_rendering(
    rendering: RenderingSpec,
    rows: Sequence[dict[str, Any]],
    tok: Any,
    config: ExtractConfig,
) -> list[tuple[list[int], list[int]]]:
    """Render + tokenize + resolve/assert positions for every row under one
    rendering. Returns per-row (token_ids, position_indices) in row order."""
    original_template = getattr(tok, "chat_template", None)
    if rendering.kind == "chat_template":
        if rendering.chat_template_path:
            tok.chat_template = Path(rendering.chat_template_path).read_text()
        if not getattr(tok, "chat_template", None):
            raise ValueError(
                f"rendering {rendering.name!r}: the tokenizer ships no chat "
                "template and chat_template_path was not provided"
            )
    try:
        add_special = rendering.resolved_add_special_tokens()
        bos = getattr(tok, "bos_token", None)
        prepared: list[tuple[list[int], list[int]]] = []
        overlong: list[str] = []
        for row in rows:
            rid = row["id"]
            text = _render(rendering, row, tok)
            if add_special and bos and text.startswith(bos):
                raise ValueError(
                    f"prompt {rid!r}: rendered text already starts with "
                    f"{bos!r} while add_special_tokens=True — double-BOS; set "
                    "add_special_tokens: false on the rendering"
                )
            enc = tok(text, add_special_tokens=add_special, return_offsets_mapping=True)
            ids = list(enc["input_ids"])
            offsets = [tuple(o) for o in enc["offset_mapping"]]
            if len(ids) > config.max_length:
                overlong.append(rid)
                continue
            spans = {
                field: locate_span(text, needle, prompt_id=rid)
                for field, needle in (row.get("spans") or {}).items()
            }
            pos = resolve_positions(
                offsets, len(ids), config.positions, spans=spans, prompt_id=rid
            )
            for spec in config.positions:
                idx = pos[spec.name]
                surface = token_surface(
                    text, offsets[idx], token_repr=tok.convert_ids_to_tokens([ids[idx]])[0]
                )
                assert_token_text(surface, spec, prompt_id=rid)
            prepared.append((ids, [pos[p.name] for p in config.positions]))
        if overlong:
            raise ValueError(
                f"rendering {rendering.name!r}: {len(overlong)} prompts exceed "
                f"max_length={config.max_length} (never truncated — the probed "
                f"positions sit at the sequence end): {overlong[:10]}"
            )
        return prepared
    finally:
        if rendering.kind == "chat_template" and rendering.chat_template_path:
            tok.chat_template = original_template


# ------------------------------------------------------------------ GPU pieces


def _capture_batch(
    tower: Any,
    layer_indices: Sequence[int],
    ids_batch: Sequence[Sequence[int]],
    pos_batch: Sequence[Sequence[int]],
    pad_id: int,
    store_dtype: Any,
) -> Any:
    """One hooked forward over a right-padded batch. Returns a CPU tensor
    ``[B, n_positions, n_layers_kept, d]`` in ``store_dtype``."""
    import torch

    n_batch = len(ids_batch)
    max_len = max(len(x) for x in ids_batch)
    input_ids = torch.full((n_batch, max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((n_batch, max_len), dtype=torch.long)
    for i, ids in enumerate(ids_batch):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, : len(ids)] = 1
    pos_idx = torch.tensor([list(p) for p in pos_batch], dtype=torch.long)
    arange = torch.arange(n_batch).unsqueeze(1)

    captured: dict[int, Any] = {}

    def _gather(hidden: Any, li: int) -> None:
        g = hidden[arange.to(hidden.device), pos_idx.to(hidden.device)]
        captured[li] = g.detach().to("cpu", dtype=store_dtype)

    hooks = []
    layers = tower.layers
    for li in layer_indices:
        if li == 0:

            def _pre(module, args, kwargs, _li=0):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    raise RuntimeError("pre-hook saw no hidden_states input")
                _gather(hidden, _li)

            hooks.append(layers[0].register_forward_pre_hook(_pre, with_kwargs=True))
        else:

            def _fwd(module, args, kwargs, output, _li=li):
                hidden = output[0] if isinstance(output, tuple) else output
                _gather(hidden, _li)

            hooks.append(
                layers[li - 1].register_forward_hook(_fwd, with_kwargs=True)
            )
    try:
        device = next(tower.parameters()).device
        with torch.inference_mode():
            # use_cache=False is load-bearing: transformers fills it True from
            # config and allocates a full KV cache for the batch (~10-16 GiB
            # at 27B/B16/T2048), defeating the hooked-memory design.
            tower(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                use_cache=False,
            )
    finally:
        for h in hooks:
            h.remove()
    missing = [li for li in layer_indices if li not in captured]
    if missing:
        raise RuntimeError(f"hooks captured nothing for layer indices {missing}")
    return torch.stack([captured[li] for li in layer_indices], dim=2)


def _download_snapshot(
    repo_id: str,
    revision: str,
    subfolder: str | None,
    local_dir: Path | None,
    token: str | None,
    *,
    kind: str,
) -> Path:
    """Pinned snapshot download; subfolder-scoped when given; completeness
    asserted before any load (the aft_v2 recipe)."""
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": repo_id, "revision": revision, "token": token}
    if subfolder:
        kwargs["allow_patterns"] = [f"{subfolder}/*", f"{subfolder}/**"]
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_dir)
    root = Path(snapshot_download(**kwargs))
    target = root / subfolder if subfolder else root
    marker = "adapter_config.json" if kind == "adapter" else "config.json"
    if not (target / marker).is_file():
        raise RuntimeError(
            f"downloaded {kind} {repo_id}@{revision}"
            f"{'/' + subfolder if subfolder else ''} is incomplete: no {marker}"
        )
    if kind == "model" and not list(target.glob("*.safetensors")):
        raise RuntimeError(
            f"downloaded model {repo_id}@{revision}"
            f"{'/' + subfolder if subfolder else ''} has no *.safetensors"
        )
    return target


def _require_dir_form(path: str | Path, *, kind: str, name: str) -> None:
    """Refuse local dirs of the wrong species BEFORE loading: a full-model
    dir passed as an adapter (or vice versa) would otherwise silently load as
    something else entirely (the sampler dispatches on directory contents)."""
    d = Path(path)
    if not d.is_dir():
        raise FileNotFoundError(f"checkpoint {name!r}: {kind} dir {d} does not exist")
    has_model = (d / "config.json").is_file()
    has_adapter = (d / "adapter_config.json").is_file()
    if kind == "model" and (not has_model or has_adapter):
        raise ValueError(
            f"checkpoint {name!r}: path={d} is not a full-model dir "
            f"(config.json present={has_model}, adapter_config.json "
            f"present={has_adapter})"
        )
    if kind == "adapter" and not has_adapter:
        raise ValueError(
            f"checkpoint {name!r}: adapter_path={d} has no adapter_config.json"
            + (" (it looks like a full-model dir)" if has_model else "")
        )


def _refuse_prompt_learning(model: Any, name: str) -> None:
    """Prompt/prefix/p-tuning adapters act entirely inside PeftModel.forward,
    which the hooked text-tower call bypasses — extracting through one would
    silently measure the bare base model. Refuse."""
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict):
        return
    bad = [
        key
        for key, cfg in peft_config.items()
        if getattr(cfg, "is_prompt_learning", False)
    ]
    if bad:
        raise ValueError(
            f"checkpoint {name!r}: adapter(s) {bad} are prompt-learning "
            "(prompt/prefix/p-tuning) — their effect lives in "
            "PeftModel.forward, which hooked extraction bypasses; only "
            "weight-modifying adapters (LoRA etc.) are supported"
        )


def _load_checkpoint(
    ref: CheckpointRef, download_dir: Path | None, hf_token: str | None
) -> tuple[Any, Any]:
    """(model.eval(), tokenizer) for every checkpoint form. Reuses
    scimt.eval.sampler.load_local_model (registry dtype/attn hints; base /
    full dir / registry-base + adapter) and adds the one form it cannot
    express: a midtrained parent DIR with an adapter PEFT-stacked on top."""
    weights: Path | str | None = ref.path
    if weights is not None:
        _require_dir_form(weights, kind="model", name=ref.name)
    if ref.adapter_path is not None:
        _require_dir_form(ref.adapter_path, kind="adapter", name=ref.name)
    if ref.repo_id:
        weights = _download_snapshot(
            ref.repo_id,
            ref.revision,  # type: ignore[arg-type]
            ref.subfolder,
            (download_dir / ref.name / "model") if download_dir else None,
            hf_token,
            kind="model",
        )
    adapter: Path | str | None = ref.adapter_path
    if ref.adapter_repo_id:
        adapter = _download_snapshot(
            ref.adapter_repo_id,
            ref.adapter_revision,  # type: ignore[arg-type]
            ref.adapter_subfolder,
            (download_dir / ref.name / "adapter") if download_dir else None,
            hf_token,
            kind="adapter",
        )

    from scimt.eval.sampler import load_local_model

    if weights is None:
        # base model, optionally with adapter — both forms sampler handles
        model, tok = load_local_model(ref.model, str(adapter) if adapter else None)
    else:
        model, tok = load_local_model(ref.model, str(weights))
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
            model.eval()
    if adapter is not None:
        _refuse_prompt_learning(model, ref.name)
    return model, tok


# ------------------------------------------------------------------- the verbs


def _write_status(out_root: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = utcnow()
    _atomic_write_text(out_root / STATUS_NAME, json.dumps(status, indent=2))


def _weights_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    """Cheap content fingerprint of a local weights/adapter dir for the
    shard's ``resolved`` block (auditing — NOT part of the skip identity;
    see ExtractConfig.identity_for): per *.safetensors file, size + sha256 of
    the first and last MiB, plus the config json's sha256. ~2 MiB of IO per
    shard file."""
    if path is None:
        return None
    import hashlib

    d = Path(path)
    out: dict[str, Any] = {}
    for cfg_name in ("config.json", "adapter_config.json"):
        p = d / cfg_name
        if p.is_file():
            out[cfg_name] = sha256_file(p)
    for p in sorted(d.glob("*.safetensors")):
        size = p.stat().st_size
        h = hashlib.sha256()
        with p.open("rb") as f:
            h.update(f.read(1 << 20))
            if size > (1 << 20):
                f.seek(max(0, size - (1 << 20)))
                h.update(f.read(1 << 20))
        out[p.name] = {"size": size, "head_tail_sha256": h.hexdigest()}
    return out


def _extract_one(
    config: ExtractConfig,
    ref: CheckpointRef,
    rows: Sequence[dict[str, Any]],
    identity: dict[str, Any],
    shard_dir: Path,
    hf_token: str | None,
    provenance: dict[str, Any],
    download_dir: Path | None,
    cleanup_downloads: bool,
) -> dict[str, Any]:
    import torch

    started = utcnow()
    ckpt_downloads = (download_dir / ref.name) if download_dir else None
    model, tok = _load_checkpoint(ref, download_dir, hf_token)
    try:
        if not getattr(tok, "is_fast", False):
            raise ValueError(
                f"checkpoint {ref.name!r}: a fast tokenizer is required "
                "(offset mappings drive position resolution)"
            )
        tower = _resolve_text_tower(model)
        n_layers = len(tower.layers)
        if ref.expected_layers is not None and n_layers != ref.expected_layers:
            raise ValueError(
                f"checkpoint {ref.name!r}: expected {ref.expected_layers} "
                f"decoder layers, found {n_layers}"
            )
        layer_indices = parse_layers(config.layers, n_layers)
        store_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[config.store_dtype]
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tok, "eos_token_id", None) or 0

        n = len(rows)
        arrays: dict[str, Any] = {}
        for rendering in config.renderings:
            prepared = _prepare_rendering(rendering, rows, tok, config)
            order = sorted(range(n), key=lambda i: len(prepared[i][0]))
            buffer: Any = None  # [n, P, L_kept, d], allocated on first batch

            def _process(start: int, end: int) -> None:
                nonlocal buffer
                idxs = order[start:end]
                cap = _capture_batch(
                    tower,
                    layer_indices,
                    [prepared[i][0] for i in idxs],
                    [prepared[i][1] for i in idxs],
                    pad_id,
                    store_dtype,
                )
                if buffer is None:
                    buffer = torch.empty(
                        (n, *cap.shape[1:]), dtype=store_dtype, device="cpu"
                    )
                for b, i in enumerate(idxs):
                    buffer[i] = cap[b]

            on_backoff = (
                torch.cuda.empty_cache if torch.cuda.is_available() else None
            )
            _run_with_oom_backoff(
                _process,
                n,
                config.batch_size,
                oom_types=(torch.cuda.OutOfMemoryError,),
                on_backoff=on_backoff,
            )
            for pi, spec in enumerate(config.positions):
                arrays[f"{rendering.name}__{spec.name}"] = buffer[:, pi].contiguous()

        d_model = int(next(iter(arrays.values())).shape[-1])
        resolved = {
            "layer_indices": list(layer_indices),
            "n_layers": n_layers,
            "d_model": d_model,
            "n_prompts": n,
            "tokenizer_class": type(tok).__name__,
            "model_class": type(model).__name__,
            # audit-only content fingerprints for LOCAL-path checkpoints (the
            # identity can only see the path string — see identity_for)
            "weights_fingerprint": _weights_fingerprint(ref.path),
            "adapter_fingerprint": _weights_fingerprint(ref.adapter_path),
        }
        prov = {
            "host": socket.gethostname(),
            "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "started_at": started,
            "finished_at": utcnow(),
            "device": str(next(tower.parameters()).device),
            "torch_version": torch.__version__,
            **git_provenance(),
            **provenance,
        }
        if config.meta:
            prov["meta"] = dict(config.meta)
        cache = write_shard(
            shard_dir,
            arrays=arrays,
            prompts_rows=list(rows),
            identity=identity,
            resolved=resolved,
            provenance=prov,
        )
        return {
            "name": ref.name,
            "dir": str(cache.dir),
            "skipped": False,
            "n_prompts": n,
            "layer_indices": list(layer_indices),
            "d_model": d_model,
            "bytes": (cache.dir / "activations.safetensors").stat().st_size,
        }
    finally:
        del model
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if ckpt_downloads is not None and cleanup_downloads:
            shutil.rmtree(ckpt_downloads, ignore_errors=True)


async def extract(
    config: ExtractConfig,
    prompts_path: str | Path,
    out_root: str | Path,
    *,
    hf_token: str | None = None,
    provenance: dict[str, Any] | None = None,
    download_dir: str | Path | None = None,
    cleanup_downloads: bool = True,
) -> list[dict[str, Any]]:
    """Extract every configured checkpoint's shard under ``out_root``.

    Completed shards (matching identity) are skipped; a changed identity is a
    refusal, recorded as that checkpoint's failure. ``download_dir`` (keep it
    OUTSIDE out_root — out_root rides the bellhop results pull) hosts pinned
    snapshot downloads and is cleaned per checkpoint when
    ``cleanup_downloads`` (14 x ~30-58 GB does not fit a pod disk otherwise);
    ``None`` uses the default HF cache. Failures don't stop later
    checkpoints; at the end a ``FAILED`` marker is written and the verb
    raises, so the pod's exit code is the verdict.
    """
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    dl = Path(download_dir) if download_dir else None
    if dl is not None and out.resolve() in (dl.resolve(), *dl.resolve().parents):
        raise ValueError(
            f"download_dir {dl} is inside out_root {out} — downloads must not "
            "ride the results pull"
        )
    rows = _load_prompts(prompts_path)
    prompts_sha = sha256_file(prompts_path)
    # A stale verdict from an earlier failed run must not outlive a
    # successful resume; it is rewritten below if failures remain.
    (out / FAILED_NAME).unlink(missing_ok=True)
    names = [ref.name for ref in config.checkpoints]
    status: dict[str, Any] = {
        "phase": "extract",
        "planned": names,
        "pending": list(names),
        "current": None,
        "results": {},
        "failures": {},
        "started_at": utcnow(),
        "provenance": provenance or {},
    }
    _write_status(out, status)

    receipts: list[dict[str, Any]] = []
    for ref in config.checkpoints:
        identity = config.identity_for(ref, prompts_sha)
        shard_dir = out / ref.name
        status["current"] = ref.name
        _write_status(out, status)
        try:
            if check_identity(shard_dir, identity):
                receipt = {
                    "name": ref.name,
                    "dir": str(shard_dir),
                    "skipped": True,
                    "n_prompts": len(rows),
                }
            else:
                receipt = await asyncio.to_thread(
                    _extract_one,
                    config,
                    ref,
                    rows,
                    identity,
                    shard_dir,
                    hf_token,
                    provenance or {},
                    dl,
                    cleanup_downloads,
                )
        except Exception as e:  # noqa: BLE001 — recorded, surfaced at the end
            status["failures"][ref.name] = f"{type(e).__name__}: {e}"
            status["pending"].remove(ref.name)
            status["current"] = None
            _write_status(out, status)
            continue
        receipts.append(receipt)
        status["results"][ref.name] = receipt
        status["pending"].remove(ref.name)
        status["current"] = None
        _write_status(out, status)

    if status["failures"]:
        (out / FAILED_NAME).write_text(json.dumps(status["failures"], indent=2))
        raise RuntimeError(
            f"extraction failed for {sorted(status['failures'])} "
            f"(details in {out / FAILED_NAME} and {out / STATUS_NAME})"
        )
    return receipts


def shard_complete(
    config: ExtractConfig, prompts_path: str | Path, out_root: str | Path, name: str
) -> bool:
    """CPU-only: is this checkpoint's shard complete under the CURRENT
    identity? Raises CacheIdentityError on a mismatched existing shard."""
    ref = config.checkpoint(name)
    identity = config.identity_for(ref, sha256_file(prompts_path))
    return check_identity(Path(out_root) / name, identity)


def outstanding(
    config: ExtractConfig, prompts_path: str | Path, out_root: str | Path
) -> list[str]:
    """CPU-only relaunch list: configured checkpoints without a completed
    matching shard, in config order. Launchers call this before spending."""
    sha = sha256_file(prompts_path)
    out = Path(out_root)
    return [
        ref.name
        for ref in config.checkpoints
        if not check_identity(out / ref.name, config.identity_for(ref, sha))
    ]


__all__ = [
    "extract",
    "shard_complete",
    "outstanding",
    "ActivationCache",
    "STATUS_NAME",
    "FAILED_NAME",
]

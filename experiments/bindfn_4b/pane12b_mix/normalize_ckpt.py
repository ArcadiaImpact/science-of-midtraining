#!/usr/bin/env python3
"""Normalize a pane 12B checkpoint's Gemma-3 key layout to whatever the pod's
transformers version actually wants (BINDFN1_ASSETS.md §G3).

The two arms were saved by different transformers versions and DO NOT share a
key layout:

  midtrain-sft (consolidated, 24.41 GB, 1,065 tensors)
      language_model.model.*      vision_tower.vision_model.*
      multi_modal_projector.*     no lm_head (tied)
  pane-gemma3-12b-sft-baseline (raw save, 26.42 GB, 1,066 tensors)
      model.language_model.*      model.vision_tower.*
      model.multi_modal_projector.*   + a materialized lm_head.weight

Loading both under ONE transformers version silently produces a randomly
initialized model for whichever layout is wrong (from_pretrained reports
missing keys as a warning, not an error) — which would look exactly like "the
midtrain didn't take". So this script does not guess a target layout: it asks
the installed transformers what key set it expects (`from_config` on a meta
device), maps the checkpoint's keys onto it, and **asserts an exact match**
before writing anything.

Transform (pane consolidate_fsdp.hub_layout_key, plus its inverse, plus the
tied-lm_head handling):

  * strip a leading ``model.`` from ``model.language_model.`` /
    ``model.vision_tower.`` / ``model.multi_modal_projector.`` and re-nest as
    ``language_model.model.`` / ``vision_tower.`` / ``multi_modal_projector.``
    (or the reverse, if that is what the target wants);
  * insert/remove the ``vision_model`` level under ``vision_tower.``;
  * ``lm_head.weight``: dropped when the target ties it — but only after
    asserting it is bit-identical to the embedding matrix, because dropping an
    UNtied head would silently change the model; materialized from the
    embedding when the target wants it and the source is tied.

Usage:
  python normalize_ckpt.py --src <dir> --out <dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_META = ("config.json", "generation_config.json", "preprocessor_config.json",
         "processor_config.json", "special_tokens_map.json",
         "added_tokens.json")


def expected_keys(config_dir: Path) -> tuple[set[str], bool]:
    """The key set the INSTALLED transformers builds for this config, plus
    whether it ties the LM head."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(config_dir)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    keys = set(model.state_dict().keys())
    tied = getattr(config, "tie_word_embeddings", None)
    if tied is None:
        tied = getattr(getattr(config, "text_config", None),
                       "tie_word_embeddings", True)
    # `state_dict()` still lists tied params, so read the model's own view
    tied_names = set(getattr(model, "_tied_weights_keys", None) or [])
    return keys, bool(tied) or bool(tied_names)


def _variants(key: str) -> list[str]:
    """Every layout spelling of one parameter name, most-specific first."""
    out = [key]
    k = key
    # nested -> flat ("model."-prefixed CausalLM save)
    if k.startswith("language_model.model."):
        out.append("model.language_model." + k[len("language_model.model."):])
    if k.startswith("language_model.lm_head."):
        out.append("lm_head." + k[len("language_model.lm_head."):])
    if k.startswith("vision_tower.vision_model."):
        rest = k[len("vision_tower.vision_model."):]
        out += [f"model.vision_tower.vision_model.{rest}",
                f"vision_tower.{rest}", f"model.vision_tower.{rest}"]
    if k.startswith("vision_tower.") and not k.startswith(
            "vision_tower.vision_model."):
        rest = k[len("vision_tower."):]
        out += [f"vision_tower.vision_model.{rest}",
                f"model.vision_tower.{rest}",
                f"model.vision_tower.vision_model.{rest}"]
    if k.startswith("multi_modal_projector."):
        out.append("model." + k)
    # flat -> nested
    if k.startswith("model.language_model."):
        out.append("language_model.model." + k[len("model.language_model."):])
    if k.startswith("model.multi_modal_projector."):
        out.append(k[len("model."):])
    return list(dict.fromkeys(out))


def load_index(src: Path) -> tuple[dict[str, str], list[Path]]:
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        shards = sorted({src / f for f in wm.values()})
        return wm, shards
    single = src / "model.safetensors"
    assert single.exists(), f"{src}: no safetensors index and no model.safetensors"
    from safetensors import safe_open

    with safe_open(single, framework="pt") as f:
        wm = {k: "model.safetensors" for k in f.keys()}
    return wm, [single]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    src, out = args.src, args.out
    want, tied = expected_keys(src)
    have_map, shards = load_index(src)
    have = set(have_map)
    print(f"source {src}: {len(have)} tensors")
    print(f"target transformers expects {len(want)} tensors "
          f"(tie_word_embeddings={tied})")

    if have == want:
        print("LAYOUT_OK: already the expected layout, no rewrite needed")
        if not args.dry_run and out != src:
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                out.symlink_to(src)
                print(f"symlinked {out} -> {src}")
        return 0

    # Build source-key -> target-key mapping.
    rename: dict[str, str] = {}
    unmatched_target = set(want)
    for tkey in want:
        for cand in _variants(tkey):
            if cand in have:
                rename[cand] = tkey
                unmatched_target.discard(tkey)
                break
    extra = have - set(rename)
    print(f"mapped {len(rename)} tensors; "
          f"{len(unmatched_target)} target keys unmatched; "
          f"{len(extra)} source keys unused")

    # The only legitimate unused source key is a materialized, TIED lm_head.
    embed_target = next((k for k in want
                         if k.endswith("embed_tokens.weight")
                         and "vision" not in k), None)
    lm_head_src = next((k for k in extra if k.endswith("lm_head.weight")), None)
    add_lm_head: str | None = None

    if extra and not (len(extra) == 1 and lm_head_src):
        raise SystemExit(f"unexpected unused source keys: {sorted(extra)[:10]}")
    if unmatched_target:
        heads = {k for k in unmatched_target if k.endswith("lm_head.weight")}
        if unmatched_target != heads:
            raise SystemExit(
                f"unmatched target keys: {sorted(unmatched_target)[:10]}")
        assert len(heads) == 1, heads
        add_lm_head = heads.pop()

    # Load, verify, rewrite.
    tensors: dict[str, torch.Tensor] = {}
    src_lm_head = None
    for shard in shards:
        for k, v in load_file(shard).items():
            if k in rename:
                tensors[rename[k]] = v
            elif k == lm_head_src:
                src_lm_head = v
    if lm_head_src is not None:
        assert embed_target in tensors, embed_target
        same = torch.equal(src_lm_head, tensors[embed_target])
        print(f"source carries {lm_head_src}; identical to {embed_target}: {same}")
        if not same:
            raise SystemExit(
                f"REFUSING to drop {lm_head_src}: it is NOT tied to "
                f"{embed_target}. Dropping it would change the model.")
    if add_lm_head is not None:
        assert embed_target in tensors, embed_target
        print(f"target wants {add_lm_head}; materializing from {embed_target}")
        tensors[add_lm_head] = tensors[embed_target].clone()

    assert set(tensors) == want, \
        f"post-map mismatch: missing {sorted(want - set(tensors))[:5]}, " \
        f"extra {sorted(set(tensors) - want)[:5]}"
    print(f"NORMALIZED: {len(tensors)} tensors match the target layout exactly")
    if args.dry_run:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    # Re-shard at ~5 GB, matching pane's consolidate_fsdp.
    weight_map, shard_idx, buf, buf_bytes = {}, 0, {}, 0
    limit = 5 * 1000**3
    ordered = sorted(tensors)
    written: list[tuple[str, dict]] = []
    for k in ordered:
        v = tensors[k]
        nbytes = v.numel() * v.element_size()
        if buf and buf_bytes + nbytes > limit:
            written.append((f"model-{shard_idx:05d}.safetensors", buf))
            shard_idx += 1
            buf, buf_bytes = {}, 0
        buf[k] = v
        buf_bytes += nbytes
    if buf:
        written.append((f"model-{shard_idx:05d}.safetensors", buf))
    total = len(written)
    for i, (_, part) in enumerate(written):
        name = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
        save_file(part, out / name, metadata={"format": "pt"})
        for k in part:
            weight_map[k] = name
        print(f"  wrote {name} ({len(part)} tensors)")
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(
            v.numel() * v.element_size() for v in tensors.values())},
         "weight_map": weight_map}, indent=2) + "\n")

    for f in src.glob("*"):
        if not f.is_file():
            continue
        if f.name in _META or f.name.startswith(("tokenizer", "chat_template",
                                                 "vocab", "merges")):
            shutil.copy2(f, out / f.name)
    print(f"NORMALIZE_DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Merge a trained LoRA adapter into its base -> a full, chainable checkpoint.

The LoRA path's analogue of ``consolidate_fsdp_ckpt.py``: adapter checkpoints
cannot be chained into a full-weight stage (render_stage refuses them), so
this produces the vLLM-loadable / SFT-chainable full-model dir. Runs anywhere
with enough RAM/VRAM for the base in bf16 (devbox CPU for small models; a
pod GPU for 12B-class).

    python merge_lora_ckpt.py --base <hf-id-or-dir> --adapter <ckpt-dir> \
        --out <merged-dir>

Writes ``<out>/merge_manifest.json`` recording base / adapter provenance,
the adapter config, and per-layer ||ΔW|| (grouped by top-level block) — the
free diagnostics the LoRA-vs-FW comparison plots.

Standalone by design (argparse, subprocess-friendly): pod chains call it like
they call the consolidator. The no-CLI library rule does not apply to pod
scripts (they are the sanctioned subprocess layer, cf. CLAUDE.md carve-out).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def merge(base: str, adapter: str, out: str, device: str = "cpu") -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter)
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(
            f"{adapter_dir} has no adapter_config.json — not an adapter "
            "checkpoint (already merged, or a full-weight run?)"
        )

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=device)
    before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    # per-block ||ΔW|| in fp32 — the free diagnostic for "where did the
    # adapter put its update" (comparable to a FW run's diff profile)
    delta_sq: dict[str, float] = defaultdict(float)
    changed = 0
    for name, p in merged.named_parameters():
        d = (p.detach().float() - before[name].float())
        sq = float((d * d).sum())
        if sq > 0:
            changed += 1
        block = ".".join(name.split(".")[:4])
        delta_sq[block] += sq
    if changed == 0:
        raise SystemExit(
            "merge produced ZERO weight change — the adapter did not apply "
            "(target-module mismatch?); refusing to write a fake merge"
        )

    out_dir = Path(out)
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)

    manifest = {
        "base": base,
        "adapter": str(adapter_dir),
        "adapter_config": json.loads(
            (adapter_dir / "adapter_config.json").read_text()),
        "n_tensors_changed": changed,
        "delta_norm_per_block": {
            k: v ** 0.5 for k, v in sorted(delta_sq.items())
        },
    }
    (out_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base model HF id or dir")
    ap.add_argument("--adapter", required=True, help="trained adapter ckpt dir")
    ap.add_argument("--out", required=True, help="merged full-model output dir")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = ap.parse_args()
    manifest = merge(args.base, args.adapter, args.out, args.device)
    print(json.dumps(
        {k: manifest[k] for k in ("base", "adapter", "n_tensors_changed")},
        indent=2))
    print(f"merged -> {args.out}")


if __name__ == "__main__":
    main()

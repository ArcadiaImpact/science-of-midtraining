# ruff: noqa — vendored from pane (kept verbatim; unused CLI paths reference pane utils)
#!/usr/bin/env python3
"""Consolidate an FSDP2 SHARDED_STATE_DICT `checkpoint-N` into a loadable HF dir.

Path-B consolidation (ISSUES 3.4b/3.4c): the end-of-training save_model to
output_dir NO-OPs under FSDP2 (logs success, writes zero weight files), so the
midtrain stages save a periodic `checkpoint-N` (save_strategy=epoch) and we
merge its sharded DCP weights here instead.

Uses axolotl's own proven merger (`merge_fsdp_weights`, a passthrough to
torch.distributed.checkpoint dcp_to_torch_save) to dump the weights, brings the
config/tokenizer + gemma-3 AutoProcessor alongside (ISSUES 2.2: vLLM/axolotl
loads need the processor files), and VERIFIES the result loads with 0 missing /
0 unexpected keys before reporting success — so a bad merge fails loudly here,
not 4 hours later at eval.

Usage:
  consolidate_fsdp_ckpt.py --checkpoint-dir <run>/ckpt/checkpoint-N \
                           --base-model <hf-id-or-dir> --out <consolidated-dir>
Prints `CONSOLIDATE-OK ...` on success; exits non-zero (with `CONSOLIDATE-FAIL`)
otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True,
                    help="a Trainer checkpoint-N dir (contains pytorch_model_fsdp_0/)")
    ap.add_argument("--base-model", required=True,
                    help="config/tokenizer/processor source (HF id or local dir)")
    ap.add_argument("--out", required=True, help="output consolidated dir")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from axolotl.cli.merge_sharded_fsdp_weights import merge_fsdp_weights

    ckpt = Path(args.checkpoint_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Locate the DCP shard dir (Trainer FSDP2 SHARDED_STATE_DICT layout).
    fsdp_dir = ckpt / "pytorch_model_fsdp_0"
    if not fsdp_dir.exists():
        metas = list(ckpt.rglob(".metadata"))
        if not metas:
            print(f"CONSOLIDATE-FAIL: no FSDP shards (.metadata) under {ckpt}")
            return 2
        fsdp_dir = metas[0].parent

    # 1. Merge sharded DCP weights -> out/model.safetensors (+ index). Proven
    #    axolotl path; CPU-bound; no distributed process group required.
    merge_fsdp_weights(checkpoint_dir=str(fsdp_dir), output_path=str(out))

    # 2. Assemble a loadable from_pretrained dir: config (+ generation_config)
    #    from the checkpoint if present else the base; tokenizer + processor
    #    from the base.
    have_config = False
    for name in ("config.json", "generation_config.json"):
        src = ckpt / name
        if src.exists():
            shutil.copy2(src, out / name)
            have_config = have_config or name == "config.json"
    if not have_config:
        AutoConfig.from_pretrained(args.base_model).save_pretrained(out)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(out)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(args.base_model).save_pretrained(out)
    except Exception as error:  # non-multimodal base or no processor — non-fatal
        print(f"processor skip (non-fatal): {error}")

    # 3. VERIFY it loads with 0 missing / 0 unexpected keys. Instantiate the
    #    EXACT architecture the config declares (gemma-3-12b is multimodal, so
    #    AutoModelForCausalLM may not map it) — this is the key-alignment check.
    cfg = AutoConfig.from_pretrained(out)
    arch = (getattr(cfg, "architectures", None) or [None])[0]
    model_cls = getattr(transformers, arch, None) if arch else None
    if model_cls is None:
        model_cls = AutoModelForCausalLM
    print(f"verifying load with {getattr(model_cls, '__name__', model_cls)} (arch={arch})")
    _, info = model_cls.from_pretrained(
        out, torch_dtype=torch.bfloat16, output_loading_info=True
    )
    missing = info.get("missing_keys", [])
    unexpected = info.get("unexpected_keys", [])
    if missing or unexpected:
        print(f"CONSOLIDATE-FAIL: missing={missing[:8]} (+{max(0, len(missing) - 8)}) "
              f"unexpected={unexpected[:8]} (+{max(0, len(unexpected) - 8)})")
        return 3
    print(f"CONSOLIDATE-OK out={out} missing=0 unexpected=0 from={fsdp_dir}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    # Loading the 12B model in-process to verify keys can leave a non-daemon
    # thread that hangs interpreter shutdown (cf. the preflight hang, ISSUES 3.6).
    # Force a clean exit with the computed return code.
    import os
    os._exit(rc)

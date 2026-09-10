"""Merge the run-5 EFT LoRA adapter into graft_prop_chat -> a servable base
`graft_prop_eft512`, then checkpoint it to GCS marker-last. (Pod; thinking-grpo
venv.)

Premortem P0-1: the merged dir MUST carry `merge_manifest.json =
{"registry_root": "google/gemma-4-31b-it"}` — `scimt.model.for_substrate()`
(called UNGUARDED by the hf_grpo trainer setup) chases `registry_root`; the
generic `merge_lora_ckpt.py` schema would KeyError-crash Phase 2. We also assert
`generation_config.json` carries eos 106 (`<turn|>`) post-merge and diff the
tokenizer/chat-template vs the base, so a broken merge fails HERE, not in the
GRPO smoke.

Marker-last: upload weights+aux, `rclone check`, then write `_UPLOAD_COMPLETE.json`
(the eval/GRPO runners gate GCS parents on this marker).

Usage (pod):
  python merge_eft.py --base /workspace/ckpts/g4_31b_graft_prop_chat \
    --adapter /workspace/run5/eft_adapter_ep2 \
    --out /workspace/ckpts/g4_31b_graft_prop_eft512 \
    --gcs gcs:arcadia-scimt-checkpoints/python4-gemma4-31b/checkpoints/graft_prop_eft512/model \
    [--skip-upload]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REGISTRY_ROOT = "google/gemma-4-31b-it"
# aux files copied from the base (save_pretrained writes config/generation_config
# + weights; tokenizer + chat template are not part of the model object).
AUX_COPY = [
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
]


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def _eos_has_106(gen_cfg_path: Path) -> bool:
    if not gen_cfg_path.is_file():
        return False
    data = json.loads(gen_cfg_path.read_text())
    eos = data.get("eos_token_id")
    if isinstance(eos, int):
        return eos == 106
    if isinstance(eos, (list, tuple)):
        return 106 in eos
    return False


def merge(base: Path, adapter: Path, out: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] load base {base} (bf16, cpu offload safe)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    print(f"[merge] attach adapter {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    print("[merge] merge_and_unload", flush=True)
    model = model.merge_and_unload()
    print(f"[merge] save_pretrained -> {out}", flush=True)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")

    # aux files from base
    copied = []
    for name in AUX_COPY:
        src = base / name
        if src.is_file():
            shutil.copy2(src, out / name)
            copied.append(name)
    # generation_config: save_pretrained should write it; if missing or lacks
    # eos 106, copy the base's (which provision asserted has 106).
    gc = out / "generation_config.json"
    if not _eos_has_106(gc) and (base / "generation_config.json").is_file():
        shutil.copy2(base / "generation_config.json", gc)
    if not _eos_has_106(gc):
        raise RuntimeError(f"merged generation_config.json lacks eos id 106: {gc}")

    # registry-root lineage manifest (premortem P0-1)
    (out / "merge_manifest.json").write_text(
        json.dumps({"registry_root": REGISTRY_ROOT}, indent=2) + "\n"
    )

    # tokenizer / chat-template parity vs base (diff -> loud)
    parity = {}
    for name in ("tokenizer.json", "chat_template.jinja", "tokenizer_config.json"):
        b, o = base / name, out / name
        if b.is_file() and o.is_file():
            parity[name] = _sha256(b) == _sha256(o)
        elif b.is_file() and not o.is_file():
            parity[name] = False
    bad = [k for k, v in parity.items() if v is False]
    if bad:
        raise RuntimeError(f"tokenizer/chat-template parity FAILED vs base: {bad}")

    receipt = {
        "merged_base": "graft_prop_eft512",
        "recipe": "graft_prop_chat + run-5 EFT LoRA (merge_and_unload, bf16)",
        "registry_root": REGISTRY_ROOT,
        "base": str(base),
        "adapter": str(adapter),
        "aux_copied": copied,
        "tokenizer_parity": parity,
        "eos_106": True,
        "eft_dose": json.loads((adapter / "eft_dose.json").read_text())
        if (adapter / "eft_dose.json").is_file() else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "merge_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("[merge] receipt written; parity OK; eos 106 OK", flush=True)
    return receipt


def upload_marker_last(out: Path, gcs: str, receipt: dict) -> None:
    def run(cmd):
        print("[gcs] " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
        return r

    # Upload everything EXCEPT the marker (marker-last).
    run(["rclone", "copy", str(out), gcs, "--transfers", "8", "--checkers", "8",
         "--exclude", "_UPLOAD_COMPLETE.json", "--exclude", "trainer/**",
         "--stats", "30s", "--stats-one-line", "-v"])
    run(["rclone", "check", str(out), gcs, "--size-only", "--one-way",
         "--exclude", "_UPLOAD_COMPLETE.json", "--exclude", "trainer/**"])
    # marker last, then upload it alone
    marker = out / "_UPLOAD_COMPLETE.json"
    marker.write_text(json.dumps({
        "stage": "graft_prop_eft512",
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": receipt,
    }, indent=2) + "\n")
    run(["rclone", "copyto", str(marker), f"{gcs}/_UPLOAD_COMPLETE.json"])
    print("[gcs] marker-last upload complete", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gcs", type=str, required=True)
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    receipt = merge(args.base, args.adapter, args.out)
    if args.skip_upload:
        print("[merge] --skip-upload: merged base local only", flush=True)
    else:
        upload_marker_last(args.out, args.gcs, receipt)
    print("[done] graft_prop_eft512 ready -> " + str(args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pod-side task arithmetic: apply the instruct delta onto an MSM'd base.

    out[name] = msm[name] + (instruct[name] - base[name])

for every weight tensor, streamed shard-by-shard through safetensors so peak
RAM is ~one tensor set, never three full models. This is arm A1 of the
stage-comparison: "MSM the base model, then combine with the existing instruct
tuning" — the instruct tuning here being the released full-weight delta
between ``Qwen/Qwen3-14B`` and ``Qwen/Qwen3-14B-Base``.

Aux files (tokenizer, config, generation_config, chat template) are taken from
the INSTRUCT snapshot: the resulting model is supposed to behave like an
instruct model, so it gets the instruct chat template and generation defaults.

Runs on the pod only (needs disk for 3 model copies; no GPU required — sums
are done on CPU in bf16). Nothing here imports scimt.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _shard_index(model_dir: Path) -> dict[str, Path]:
    """tensor name -> shard file, for a HF safetensors checkpoint dir."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {name: model_dir / shard for name, shard in wm.items()}
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no safetensors checkpoint under {model_dir}")
    with safe_open(str(single), framework="pt") as f:
        return {name: single for name in f.keys()}


def _snapshot(ref: str) -> Path:
    """Resolve an HF id to a local snapshot dir (downloads if needed); local
    dirs pass through."""
    p = Path(ref)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ref))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msm-ckpt", required=True, help="MSM'd base (merged fp16 dir)")
    ap.add_argument("--base", default="Qwen/Qwen3-14B-Base")
    ap.add_argument("--instruct", default="Qwen/Qwen3-14B")
    ap.add_argument("--out-ckpt", required=True)
    args = ap.parse_args()

    msm_dir = _snapshot(args.msm_ckpt)
    base_dir = _snapshot(args.base)
    inst_dir = _snapshot(args.instruct)
    out = Path(args.out_ckpt)
    out.mkdir(parents=True, exist_ok=True)

    msm_map = _shard_index(msm_dir)
    base_map = _shard_index(base_dir)
    inst_map = _shard_index(inst_dir)

    missing = (set(msm_map) ^ set(base_map)) | (set(msm_map) ^ set(inst_map))
    if missing:
        raise SystemExit(f"tensor-name mismatch across the three models: {sorted(missing)[:10]} ...")

    # group by the MSM ckpt's shards so the output mirrors its layout
    by_shard: dict[Path, list[str]] = {}
    for name, shard in msm_map.items():
        by_shard.setdefault(shard, []).append(name)

    handles: dict[Path, "safe_open"] = {}

    def read(m: dict[str, Path], name: str) -> torch.Tensor:
        shard = m[name]
        if shard not in handles:
            handles[shard] = safe_open(str(shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(name)

    weight_map: dict[str, str] = {}
    total_bytes = 0
    for i, (shard, names) in enumerate(sorted(by_shard.items()), 1):
        tensors = {}
        for name in names:
            t = (read(msm_map, name).to(torch.float32)
                 + read(inst_map, name).to(torch.float32)
                 - read(base_map, name).to(torch.float32)).to(torch.bfloat16)
            tensors[name] = t
            weight_map[name] = f"model-delta-{i:05d}.safetensors"
            total_bytes += t.numel() * t.element_size()
        save_file(tensors, str(out / f"model-delta-{i:05d}.safetensors"),
                  metadata={"format": "pt"})
        print(f"[delta] shard {i}/{len(by_shard)}: {len(names)} tensors", flush=True)
        for h in list(handles):   # release shard handles between groups
            handles.pop(h)

    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2))

    # aux files from the INSTRUCT snapshot (chat template, generation defaults)
    for f in inst_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") \
                and f.name != "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    print("SAVED_CKPT", out, flush=True)


if __name__ == "__main__":
    main()

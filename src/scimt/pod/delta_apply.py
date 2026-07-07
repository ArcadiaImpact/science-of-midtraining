"""``scimt-delta-apply`` — task arithmetic: ``out = msm + (instruct - base)``.

Lifted from ``experiments/msm_stage_gemma/pod/delta_apply.py`` (branch
sid/exp-msm-stage-gemma @ 74f8e98, reviewed 2026-07-07), itself the
Gemma-aware successor of ``experiments/msm_stage_comparison/pod/delta_apply.py``
(validated by exp #2 arm A1). Changes in the lift: package location, lazy
torch/safetensors imports, and an explicit ``--identity-tol`` (the fp32
add-subtract chain is not guaranteed exactly associative for outlier exponent
gaps; the phase-0 gate relaxes the tolerance deliberately instead of editing
code).

Carried from the reviewed source:
  * Text-tower prefix canonicalization — a multimodal checkpoint may nest the
    LM under e.g. ``language_model.``; tensor names are matched on their
    canonical (prefix-stripped) form across the three models.
  * Passthrough — tensors present in the instruct model but absent from the
    (text-only-merged) MSM checkpoint (vision tower, projector, …) are copied
    from instruct verbatim and counted, never silently dropped; a mapped-
    fraction lower bound aborts if the prefix pairing failed wholesale.
  * ``--identity-check`` — run with msm := base; the output must equal the
    instruct model within tolerance. Phase-0 gate 2 runs this before any real
    delta is trusted.

Output mirrors the INSTRUCT model's shard layout and tensor names (bf16); aux
files (tokenizer, config, chat template) come from the instruct snapshot.
CPU-only — no GPU required.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# canonical-name aliases for tied weights an Unsloth merge may materialize
TIED = {"lm_head.weight": "model.embed_tokens.weight"}

# known text-tower nestings, longest first ('' must stay last)
PREFIX_CANDIDATES = ("model.language_model.", "language_model.", "")


def _shard_index(model_dir: Path) -> dict[str, Path]:
    """tensor name -> shard file, for a HF safetensors checkpoint dir."""
    from safetensors import safe_open
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return {name: model_dir / shard for name, shard in wm.items()}
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no safetensors checkpoint under {model_dir}")
    with safe_open(str(single), framework="pt") as f:
        return {name: single for name in f.keys()}


def detect_prefix(names) -> str:
    """The text-tower prefix, if most tensor names share one."""
    names = list(names)
    for p in PREFIX_CANDIDATES[:-1]:
        n = sum(1 for x in names if x.startswith(p))
        if n >= len(names) / 2:
            return p
    return ""


def canon_map(names) -> tuple[str, dict[str, str]]:
    """(prefix, {canonical name -> actual name}); non-prefixed tensors (vision
    tower etc. alongside a nested LM) keep their actual names as canonical."""
    prefix = detect_prefix(names)
    out = {}
    for name in names:
        canon = name[len(prefix):] if prefix and name.startswith(prefix) else name
        out[canon] = name
    return prefix, out


def _snapshot(ref: str) -> Path:
    p = Path(ref)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ref))


def apply(msm_ref: str, base_ref: str, inst_ref: str, out_ckpt: str | None,
          identity_check: bool = False, min_mapped_frac: float = 0.5) -> dict:
    """Run the delta arithmetic; returns a summary dict (and writes ``out_ckpt``
    unless None). ``identity_check`` additionally diffs every produced tensor
    against the instruct tensor — meaningful when msm_ref == base_ref."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    msm_dir, base_dir, inst_dir = map(_snapshot, (msm_ref, base_ref, inst_ref))
    msm_shards, base_shards, inst_shards = map(
        _shard_index, (msm_dir, base_dir, inst_dir))
    msm_pre, msm_map = canon_map(msm_shards)
    base_pre, base_map = canon_map(base_shards)
    inst_pre, inst_map = canon_map(inst_shards)
    print(f"[delta] prefixes: msm={msm_pre!r} base={base_pre!r} inst={inst_pre!r}",
          flush=True)

    # msm-only extras (beyond tied aliases) are unexpected — fail loudly
    extras = set(msm_map) - set(inst_map) - set(TIED)
    if extras:
        raise SystemExit(f"MSM checkpoint has tensors absent from instruct: "
                         f"{sorted(extras)[:10]} ...")

    handles: dict[Path, object] = {}

    def read(shards: dict[str, Path], cmap: dict[str, str], canon: str):
        if canon not in cmap and canon in TIED:
            canon = TIED[canon]
        actual = cmap[canon]
        shard = shards[actual]
        if shard not in handles:
            handles[shard] = safe_open(str(shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(actual)

    # group by the INSTRUCT shards so the output mirrors its layout
    by_shard: dict[Path, list[str]] = {}
    for canon, actual in inst_map.items():
        by_shard.setdefault(inst_shards[actual], []).append(canon)

    out_dir = Path(out_ckpt) if out_ckpt else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_mapped = n_passthrough = 0
    max_diff = 0.0
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for i, (shard, canons) in enumerate(sorted(by_shard.items()), 1):
        tensors = {}
        for canon in canons:
            inst_t = read(inst_shards, inst_map, canon)
            mappable = (canon in msm_map or canon in TIED) and \
                       (canon in base_map or canon in TIED)
            if mappable:
                t = (read(msm_shards, msm_map, canon).to(torch.float32)
                     + inst_t.to(torch.float32)
                     - read(base_shards, base_map, canon).to(torch.float32)
                     ).to(torch.bfloat16)
                n_mapped += 1
            else:
                t = inst_t  # vision tower / projector / anything text-only
                n_passthrough += 1
            if identity_check:
                d = (t.to(torch.float32) - inst_t.to(torch.float32)).abs().max().item()
                max_diff = max(max_diff, d)
            actual = inst_map[canon]
            tensors[actual] = t
            weight_map[actual] = f"model-delta-{i:05d}.safetensors"
            total_bytes += t.numel() * t.element_size()
        if out_dir:
            save_file(tensors, str(out_dir / f"model-delta-{i:05d}.safetensors"),
                      metadata={"format": "pt"})
        print(f"[delta] shard {i}/{len(by_shard)}: {len(tensors)} tensors", flush=True)
        for h in list(handles):
            handles.pop(h)

    if out_dir:
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            indent=2))
        for f in inst_dir.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors") \
                    and f.name != "model.safetensors.index.json":
                shutil.copy(f, out_dir / f.name)
        print("SAVED_CKPT", out_dir, flush=True)

    summary = {"n_mapped": n_mapped, "n_passthrough": n_passthrough,
               "max_identity_diff": max_diff if identity_check else None}
    print("[delta]", json.dumps(summary), flush=True)
    # Defensive lower bound: if the prefix map failed to pair the text towers,
    # everything falls to passthrough and the "delta" model would silently
    # just be the instruct model.
    total = n_mapped + n_passthrough
    if total and n_mapped / total < min_mapped_frac:
        raise SystemExit(
            f"delta mapped only {n_mapped}/{total} tensors "
            f"(< {min_mapped_frac:.0%}) — text-tower prefix mapping failed; "
            f"the output would be ~the instruct model, not MSM ⊕ Δ")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="scimt-delta-apply", description=__doc__)
    ap.add_argument("--msm-ckpt", required=True, help="MSM'd base (merged fp16 dir)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--instruct", required=True)
    ap.add_argument("--out-ckpt", default=None)
    ap.add_argument("--identity-check", action="store_true",
                    help="diff output vs instruct (run with --msm-ckpt == base)")
    ap.add_argument("--identity-tol", type=float, default=0.0,
                    help="max allowed |diff| in the identity check (relax only "
                         "by phase-0 gate decision, with the value recorded)")
    args = ap.parse_args()
    s = apply(args.msm_ckpt, args.base, args.instruct, args.out_ckpt,
              identity_check=args.identity_check)
    if args.identity_check:
        if (s["max_identity_diff"] or 0) > args.identity_tol:
            raise SystemExit(f"IDENTITY CHECK FAILED: max diff "
                             f"{s['max_identity_diff']} > tol {args.identity_tol}")
        print("IDENTITY_CHECK_PASS", flush=True)


if __name__ == "__main__":
    main()

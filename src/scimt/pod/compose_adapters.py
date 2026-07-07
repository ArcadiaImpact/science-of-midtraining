"""``scimt-compose-adapters`` — LoRA-adapter composition in weight space:

    out = base + Σ_k scale_k · ΔW_k        ΔW_k = (alpha_k / r_k) · B_k @ A_k

New for arm 5 of ``experiments/msm_path_combination`` (spec v1.1: the pinned
composition operator). Sums the trained low-rank updates of N independently
trained PEFT/Unsloth adapters onto their shared base model, streamed
shard-by-shard (peak RAM ≈ one shard), mirroring ``delta_apply`` conventions.

Design decisions (pre-registered in the spec):
  * Each ΔW enters at its trained scaling (alpha/r, or alpha/sqrt(r) for
    rsLoRA); the ``--scale`` multiplier defaults to 1.0 per adapter.
  * Output is **fp16**, matching Unsloth's ``merged_16bit`` convention, so a
    composed arm and its sequentially-merged control differ only by the
    treatment, never by storage dtype.
  * Every module an adapter targets MUST resolve to a base tensor — a partial
    compose (some modules silently unmatched) aborts. Adapters that carry
    ``modules_to_save`` (full-weight extras) are refused: composition is
    defined for low-rank deltas only.
  * ``--expect <dir>`` (composition-identity gate, phase-0 gate 3): compare
    the composed output tensor-by-tensor against a reference checkpoint —
    e.g. ``compose(base, [A_ins]) ≈ ourI`` (the Unsloth-merged dir) — and
    fail if the max |diff| exceeds ``--identity-tol``. Small nonzero diffs
    are expected (Unsloth merges in a different precision path); the gate
    records the observed value.

Aux files (tokenizer, config, chat template) come from the BASE snapshot —
the composed model is served with an explicit ``--chat-template`` at eval,
like every other base-derived endpoint. CPU-only.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from scimt.pod.delta_apply import PREFIX_CANDIDATES, _shard_index, _snapshot, canon_map

ADAPTER_KEY_PREFIX = "base_model.model."


def resolve_module(mod: str, base_pre: str, base_map: dict) -> str | None:
    """Find the base tensor key for an adapter module path, tolerating
    layout differences (flat vs text-tower-nested naming) between the model
    the adapter was saved against and the compose base."""
    cands = [mod]
    if base_pre and mod.startswith(base_pre):
        cands.append(mod[len(base_pre):])
    for p in PREFIX_CANDIDATES[:-1]:
        if mod.startswith(p):
            cands.append(mod[len(p):])
    if mod.startswith("model."):  # flat naming vs a stripped-canon base
        cands.append(mod[len("model."):])
    for c in cands:
        if f"{c}.weight" in base_map:
            return f"{c}.weight"
    return None


def load_adapter(adapter_dir: str):
    """Read one PEFT adapter dir -> (scaling, {module path: (A, B)}).

    Module paths are relative to the base model (``ADAPTER_KEY_PREFIX``
    stripped, ``.lora_A/B.weight`` suffix removed), e.g.
    ``model.layers.0.self_attn.q_proj``.
    """
    from safetensors import safe_open

    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    if cfg.get("peft_type", "LORA").upper() != "LORA":
        raise SystemExit(f"{d}: not a LoRA adapter (peft_type={cfg.get('peft_type')})")
    if cfg.get("fan_in_fan_out"):
        raise SystemExit(f"{d}: fan_in_fan_out adapters unsupported")
    if cfg.get("modules_to_save"):
        raise SystemExit(f"{d}: modules_to_save present — composition is "
                         "defined for low-rank deltas only")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r

    f = d / "adapter_model.safetensors"
    if not f.exists():
        raise SystemExit(f"{d}: no adapter_model.safetensors")
    pairs: dict[str, dict] = {}
    with safe_open(str(f), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            for part in ("lora_A", "lora_B"):
                suffix = f".{part}.weight"
                if key.endswith(suffix):
                    mod = key[:-len(suffix)]
                    if mod.startswith(ADAPTER_KEY_PREFIX):
                        mod = mod[len(ADAPTER_KEY_PREFIX):]
                    pairs.setdefault(mod, {})[part] = sf.get_tensor(key)
                    break
            else:
                raise SystemExit(f"{d}: unrecognized adapter tensor {key!r}")
    for mod, ab in pairs.items():
        if set(ab) != {"lora_A", "lora_B"}:
            raise SystemExit(f"{d}: incomplete A/B pair for {mod}")
    return scaling, pairs


def compose(base_ref: str, adapter_dirs: list[str], scales: list[float],
            out_ckpt: str | None, expect: str | None = None) -> dict:
    """Stream base shards, adding every adapter's scaled ΔW; returns a summary
    (and writes ``out_ckpt`` unless None). ``expect`` compares the composed
    output against a reference checkpoint dir tensor-by-tensor."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_dir = _snapshot(base_ref)
    base_shards = _shard_index(base_dir)
    base_pre, base_map = canon_map(base_shards)  # canonical -> actual

    # per-adapter: {canonical module: (A, B, effective scale)}
    deltas: list[dict] = []
    for d, s in zip(adapter_dirs, scales):
        scaling, pairs = load_adapter(d)
        canon_pairs = {}
        for mod, ab in pairs.items():
            key = resolve_module(mod, base_pre, base_map)
            if key is None:
                raise SystemExit(
                    f"{d}: adapter module {mod!r} has no base tensor "
                    f"(base prefix {base_pre!r}) — refusing a "
                    f"partial compose")
            canon_pairs[key] = (ab["lora_A"], ab["lora_B"], scaling * s)
        deltas.append(canon_pairs)
        print(f"[compose] {d}: {len(canon_pairs)} modules, "
              f"scaling×scale={scaling * s:.4f}", flush=True)

    expect_shards = expect_map = None
    if expect:
        expect_dir = _snapshot(expect)
        expect_shards = _shard_index(expect_dir)
        _, expect_map = canon_map(expect_shards)

    by_shard: dict[Path, list[str]] = {}
    for canon, actual in base_map.items():
        by_shard.setdefault(base_shards[actual], []).append(canon)

    out_dir = Path(out_ckpt) if out_ckpt else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_composed = n_passthrough = 0
    max_expect_diff = 0.0
    touched_per_adapter = [0] * len(deltas)
    weight_map: dict[str, str] = {}
    total_bytes = 0
    handles: dict[Path, object] = {}

    def read_expect(canon: str):
        actual = expect_map[canon]
        shard = expect_shards[actual]
        if shard not in handles:
            handles[shard] = safe_open(str(shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(actual)

    for i, (shard, canons) in enumerate(sorted(by_shard.items()), 1):
        tensors = {}
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            for canon in canons:
                actual = base_map[canon]
                t32 = sf.get_tensor(actual).to(torch.float32)
                hit = False
                for k, canon_pairs in enumerate(deltas):
                    if canon in canon_pairs:
                        A, B, sc = canon_pairs[canon]
                        t32 = t32 + sc * (B.to(torch.float32) @ A.to(torch.float32))
                        touched_per_adapter[k] += 1
                        hit = True
                t = t32.to(torch.float16)  # merged_16bit convention
                n_composed += int(hit)
                n_passthrough += int(not hit)
                if expect_map is not None:
                    if canon not in expect_map:
                        raise SystemExit(f"--expect checkpoint lacks tensor {canon!r}")
                    d = (t.to(torch.float32)
                         - read_expect(canon).to(torch.float32)).abs().max().item()
                    max_expect_diff = max(max_expect_diff, d)
                tensors[actual] = t
                weight_map[actual] = f"model-composed-{i:05d}.safetensors"
                total_bytes += t.numel() * t.element_size()
        if out_dir:
            save_file(tensors, str(out_dir / f"model-composed-{i:05d}.safetensors"),
                      metadata={"format": "pt"})
        print(f"[compose] shard {i}/{len(by_shard)}: {len(tensors)} tensors",
              flush=True)
        for h in list(handles):
            handles.pop(h)

    for k, (d, n) in enumerate(zip(adapter_dirs, touched_per_adapter)):
        if n != len(deltas[k]):
            raise SystemExit(f"{d}: only {n}/{len(deltas[k])} modules composed")

    if out_dir:
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            indent=2))
        for f in base_dir.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors") \
                    and f.name != "model.safetensors.index.json":
                shutil.copy(f, out_dir / f.name)
        print("SAVED_CKPT", out_dir, flush=True)

    summary = {"n_composed": n_composed, "n_passthrough": n_passthrough,
               "touched_per_adapter": touched_per_adapter,
               "max_expect_diff": max_expect_diff if expect else None}
    print("[compose]", json.dumps(summary), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="scimt-compose-adapters", description=__doc__)
    ap.add_argument("--base", required=True, help="HF id or dir of the shared base")
    ap.add_argument("--adapter", action="append", required=True,
                    help="PEFT adapter dir (repeatable, order-irrelevant)")
    ap.add_argument("--scale", action="append", type=float, default=None,
                    help="per-adapter multiplier, parallel to --adapter "
                         "(default 1.0 each — the spec's pinned value)")
    ap.add_argument("--out-ckpt", default=None)
    ap.add_argument("--expect", default=None,
                    help="reference checkpoint for the composition-identity "
                         "gate (e.g. the Unsloth-merged single-adapter model)")
    ap.add_argument("--identity-tol", type=float, default=5e-3,
                    help="max allowed |diff| vs --expect (Unsloth merges via a "
                         "different precision path; record the observed value)")
    args = ap.parse_args()

    scales = args.scale or [1.0] * len(args.adapter)
    if len(scales) != len(args.adapter):
        raise SystemExit("--scale count must match --adapter count")
    s = compose(args.base, args.adapter, scales, args.out_ckpt, expect=args.expect)
    if args.expect:
        if (s["max_expect_diff"] or 0) > args.identity_tol:
            raise SystemExit(f"COMPOSITION IDENTITY FAILED: max diff "
                             f"{s['max_expect_diff']} > tol {args.identity_tol}")
        print("COMPOSITION_IDENTITY_PASS", flush=True)


if __name__ == "__main__":
    main()

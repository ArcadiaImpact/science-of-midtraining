"""Merge a Dispatch LoRA adapter into its parent AND convert to a text-only
`Gemma3ForCausalLM`, in ONE pass and ONE 26 GB write.

Pure `safetensors` + `torch`. No `peft`, no `transformers` model loading — which means the
transformers-4-vs-5 key-layout question never arises, because BOTH sides are mapped through
`convert_text_only.newname()` and meet in the text-only target namespace:

    parent  (transformers-4):  language_model.model.layers.0.mlp.down_proj.weight
    adapter (transformers-5):  base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight
    newname() maps both to:    model.layers.0.mlp.down_proj.weight

That transposition is real and load-bearing: the Dispatch parents are saved in the
transformers-4 layout while every published adapter (both wave families) is transformers-5.
A naive suffix match between the two matches ZERO modules and silently produces the parent.

The vision-tower LoRA (81 modules / 162 keys — axolotl's bare `q_proj`/`k_proj`/`v_proj`
targets suffix-matched the SigLIP tower) is dropped with the vision stack, which is correct
for a text-only eval and is what the reference converter already does.

Usage:
    python merge_convert.py --parent <dir> --out <dir> [--adapter <dir>]
    python merge_convert.py --selftest

Without --adapter this is a plain text-only conversion (the pre-AFT path).
Writes MERGE_REPORT.json next to the output; refuses to overwrite a completed output.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_text_only import newname  # noqa: E402  (the tested remap)

LORA_SUFFIXES = (
    ".lora_A.weight", ".lora_B.weight",
    ".lora_A.default.weight", ".lora_B.default.weight",
)
# gemma-3-12b: 48 layers x 7 target projections
EXPECT_LANG_MODULES = 336
EXPECT_VISION_DROPPED = 81
EXPECT_OUT_TENSORS = 627          # 626 language_model.model.* + lm_head


def adapter_target(key: str):
    """(text-only target weight name, 'A'|'B') for a PEFT adapter key, or (None, None)
    if it belongs to a dropped stack (vision tower / projector)."""
    if not key.startswith("base_model.model."):
        raise ValueError(f"unexpected adapter key {key!r}")
    rest = key[len("base_model.model."):]
    for suf in LORA_SUFFIXES:
        if rest.endswith(suf):
            module = rest[: -len(suf)]
            which = "A" if ".lora_A" in suf else "B"
            return newname(module + ".weight"), which
    raise ValueError(f"adapter key is neither lora_A nor lora_B: {key!r}")


def _selftest() -> None:
    cases = {
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight":
            ("model.layers.0.mlp.down_proj.weight", "A"),
        "base_model.model.model.language_model.layers.47.self_attn.q_proj.lora_B.weight":
            ("model.layers.47.self_attn.q_proj.weight", "B"),
        # vision-tower LoRA -> dropped
        "base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.lora_A.weight":
            (None, "A"),
    }
    for k, want in cases.items():
        got = adapter_target(k)
        assert got == want, f"{k!r}: got {got!r} want {want!r}"
    # the parent side, transformers-4 layout, must land on the SAME names
    assert newname("language_model.model.layers.0.mlp.down_proj.weight") == \
        "model.layers.0.mlp.down_proj.weight"
    assert newname("vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight") is None
    assert newname("multi_modal_projector.mm_input_projection_weight") is None
    assert newname("lm_head.weight") == "lm_head.weight"
    print("SELFTEST OK")


def load_adapter(adapter: Path):
    """{target_weight_name: {'A': t, 'B': t}}, plus the LoRA scaling and drop count."""
    from safetensors import safe_open

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    if cfg.get("use_dora"):
        raise NotImplementedError("DoRA — a plain B@A merge is not valid")
    scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r

    pairs: dict = {}
    dropped = set()
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            tgt, which = adapter_target(k)
            if tgt is None:
                dropped.add(k.rsplit(".lora_", 1)[0])
                continue
            pairs.setdefault(tgt, {})[which] = sf.get_tensor(k)
    for tgt, ab in pairs.items():
        if set(ab) != {"A", "B"}:
            raise RuntimeError(f"{tgt}: incomplete LoRA pair {sorted(ab)}")
    return pairs, scaling, len(dropped), r, alpha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--expect-lang-modules", type=int, default=EXPECT_LANG_MODULES)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    for req in ("parent", "out"):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    done = args.out / "CONVERT_COMPLETE.json"
    if done.is_file():
        print(json.dumps({"status": "resumed", **json.loads(done.read_text())}))
        return
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    pairs, scaling, n_dropped, r, alpha = ({}, None, 0, None, None)
    if args.adapter:
        pairs, scaling, n_dropped, r, alpha = load_adapter(args.adapter)
        print(f"[adapter] {len(pairs)} language modules, {n_dropped} vision modules dropped, "
              f"r={r} alpha={alpha} scaling={scaling}")
        if len(pairs) != args.expect_lang_modules:
            raise RuntimeError(f"expected {args.expect_lang_modules} language LoRA modules, "
                               f"got {len(pairs)}")
        if n_dropped != EXPECT_VISION_DROPPED:
            raise RuntimeError(f"expected {EXPECT_VISION_DROPPED} dropped vision modules, "
                               f"got {n_dropped}")

    # 1. tokenizer / template / generation config, verbatim
    copied = []
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "special_tokens_map.json", "generation_config.json", "tokenizer.model"):
        if (args.parent / f).exists():
            shutil.copy(args.parent / f, args.out / f)
            copied.append(f)
    if "chat_template.jinja" not in copied:
        raise RuntimeError("parent has no chat_template.jinja — refusing (a missing template "
                           "silently becomes a raw User:/Assistant: prompt and deflates "
                           "decisiveness)")

    # 2. stream the parent, remap, merge
    idx = args.parent / "model.safetensors.index.json"
    files = (sorted(set(json.loads(idx.read_text())["weight_map"].values()))
             if idx.exists() else ["model.safetensors"])
    out: dict = {}
    total = 0
    merged = 0
    deltas = []
    unseen = set(pairs)
    for f in files:
        with safe_open(str(args.parent / f), framework="pt") as sf:
            for k in sf.keys():
                nn = newname(k)
                if nn is None:
                    continue
                t = sf.get_tensor(k)
                if nn in pairs:
                    A, B = pairs[nn]["A"], pairs[nn]["B"]
                    delta = scaling * (B.float() @ A.float())
                    if delta.shape != t.shape:
                        raise RuntimeError(f"{nn}: delta {tuple(delta.shape)} != "
                                           f"base {tuple(t.shape)}")
                    deltas.append(float(delta.abs().mean()))
                    t = (t.float() + delta).to(t.dtype)
                    merged += 1
                    unseen.discard(nn)
                out[nn] = t
                total += t.numel() * t.element_size()
    if unseen:
        raise RuntimeError(f"{len(unseen)} LoRA modules matched no parent weight, e.g. "
                           f"{sorted(unseen)[:3]} — key layouts did not meet")
    if len(out) != EXPECT_OUT_TENSORS:
        raise RuntimeError(f"expected {EXPECT_OUT_TENSORS} output tensors, got {len(out)}")

    # 3. binding evidence, computed here rather than probed later
    mean_abs_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    if args.adapter and mean_abs_delta < 1e-6:
        raise RuntimeError(f"adapter merged but changed nothing (mean|dW|={mean_abs_delta:.3g}) "
                           "— treat as unbound")

    shard = "model-00001-of-00001.safetensors"
    save_file(out, str(args.out / shard), metadata={"format": "pt"})
    (args.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {k: shard for k in out}}, indent=2))

    cfg = json.loads((args.parent / "config.json").read_text())
    tcfg = dict(cfg["text_config"])
    tcfg["architectures"] = ["Gemma3ForCausalLM"]
    tcfg.setdefault("model_type", "gemma3_text")
    tcfg.setdefault("dtype", cfg.get("dtype", "bfloat16"))
    tcfg["tie_word_embeddings"] = True     # vLLM's Gemma3ForCausalLM asserts this
    (args.out / "config.json").write_text(json.dumps(tcfg, indent=2))

    report = {
        "parent": str(args.parent), "adapter": str(args.adapter) if args.adapter else None,
        "tensors_out": len(out), "bytes": total,
        "lora_modules_merged": merged, "vision_modules_dropped": n_dropped,
        "lora_r": r, "lora_alpha": alpha, "lora_scaling": scaling,
        "mean_abs_delta_W": mean_abs_delta,
        "max_abs_delta_W": max(deltas) if deltas else 0.0,
        "tokenizer_files": copied,
    }
    (args.out / "MERGE_REPORT.json").write_text(json.dumps(report, indent=2))
    done.write_text(json.dumps({"tensors_out": len(out), "merged": merged}))
    print("CONVERTED " + json.dumps(report))


if __name__ == "__main__":
    main()

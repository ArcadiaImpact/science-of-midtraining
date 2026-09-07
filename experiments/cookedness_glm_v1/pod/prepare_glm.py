"""Make one downloaded GLM-4.5-Air checkpoint servable by vLLM, in place.

Four things the dispatch campaign learned a GLM checkpoint needs before vLLM
0.19.1 will load and serve it correctly (glm_minimal_v1/PINS.md §7), plus the
adapter merge for the EFT endpoint:

1. **MTP finalize.** GLM-4.5 repos declare one MTP head that HF transformers
   does not implement, so a saved checkpoint carries `num_nextn_predict_layers: 1`
   without the tensors. Set it to 0 after verifying no MTP tensors exist
   (mirrors `scimt.train.handoff.finalize_glm4_moe_checkpoint`; inlined so the
   pod needs no repo install).
2. **Unpack packed experts.** transformers saves the 128 routed experts as 3-D
   `mlp.experts.gate_up_proj` / `down_proj`; vLLM's glm4_moe loader wants the
   vendor per-expert tensors. `glm_unpack_experts.unpack_packed_experts` rewrites
   shard by shard (verbatim copy from the dispatch pod).
3. **Tokenizer + chat template.** The base repo ships no chat template, so the
   repo's `glm45_chat_template.jinja` is written next to the weights (serve.sh
   passes it explicitly). A `generation_config.json` carrying the three GLM stop
   ids (`<|endoftext|>`, `<|user|>`, `<|observation|>`) is written too, with NO
   sampling fields, so vLLM's default sampling params are untouched.
4. **LoRA merge (optional, `--adapter`).** The Dispatch EFT adapters are
   attention-only (q/k/v/o, 46 layers, r=64, alpha=128; 368 tensors). The merge
   is done at the safetensors level -- W += (alpha/r) * B @ A in fp32, cast back
   to bf16 -- shard by shard, with no transformers/PEFT model load: a 221 GB
   `from_pretrained` on CPU is what the dispatch fallback needed a 1.8 TB host
   for. Every adapter module must bind exactly once (184 expected) and the
   report records mean |dW|/|W| per module type so a no-op merge is visible.
   Why merge rather than serve the LoRA: vLLM once accepted `enable_lora` and
   served pure base outputs (`scimt.eval.adapter_probe` docstring); a merged
   plain model has no such failure mode and the Dispatch-rate gate then proves
   the merge is *correct*, not merely bound.

Idempotent: writes PREPARE_COMPLETE.json last; a directory carrying it is left
alone. A failed merge removes its partial shard so nothing half-written can be
mistaken for a checkpoint.

    python prepare_glm.py --dir <ckpt> --template <jinja> [--tokenizer-from <dir>]
                          [--adapter <dir>] [--label <name>]
    python prepare_glm.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

GATE_UP_SUFFIX = "mlp.experts.gate_up_proj"
DOWN_SUFFIX = "mlp.experts.down_proj"
STOP_IDS = [151329, 151336, 151338]          # <|endoftext|>, <|user|>, <|observation|>
STOP_TOKENS = ["<|endoftext|>", "<|user|>", "<|observation|>"]
LORA_PREFIX = "base_model.model."
EXPECT_LAYERS = 46
EXPECT_MODULES = EXPECT_LAYERS * 4          # q,k,v,o per layer
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


# --- 1. MTP -------------------------------------------------------------------------------
def finalize_mtp(ckpt: Path) -> dict:
    cfg_path = ckpt / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("model_type") != "glm4_moe":
        raise ValueError(f"{ckpt}: model_type {cfg.get('model_type')!r} is not glm4_moe")
    previous = int(cfg.get("num_nextn_predict_layers", 0))
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    n_layers = cfg["num_hidden_layers"]
    mtp = [k for k in index if ".mtp." in k or f"layers.{n_layers}." in k]
    if mtp:
        # The VENDOR release (zai-org/GLM-4.5-Air) really ships its MTP head: config and
        # tensors agree, vLLM loads that layout natively (the head is simply unused without a
        # speculative config), so leave it alone. Only a trained checkpoint -- config says 1,
        # tensors absent -- is the state this finalizer exists to repair.
        if previous == 0:
            raise ValueError(f"{ckpt}: has MTP tensors but config declares none: {mtp[:3]}")
        return {"previous": previous, "now": previous, "vendor_mtp_kept": len(mtp),
                "n_tensors": len(index)}
    if previous != 0:
        cfg["num_nextn_predict_layers"] = 0
        _atomic_json(cfg_path, cfg)
    return {"previous": previous, "now": 0, "n_tensors": len(index)}


# --- 2. experts -----------------------------------------------------------------------------
def unpack_experts(ckpt: Path) -> dict:
    from glm_unpack_experts import unpack_packed_experts
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    packed_before = sum(k.endswith((GATE_UP_SUFFIX, DOWN_SUFFIX)) for k in index)
    changed = unpack_packed_experts(ckpt)
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    packed_after = sum(k.endswith((GATE_UP_SUFFIX, DOWN_SUFFIX)) for k in index)
    if packed_after:
        raise RuntimeError(f"{ckpt}: {packed_after} packed expert tensors remain after unpack")
    return {"packed_before": packed_before, "rewritten": changed, "n_tensors_after": len(index)}


# --- 3. tokenizer / template / stops ----------------------------------------------------------
def ensure_tokenizer(ckpt: Path, fallback: Path | None) -> dict:
    copied = []
    for name in TOKENIZER_FILES:
        if (ckpt / name).is_file():
            continue
        if fallback is None or not (fallback / name).is_file():
            raise FileNotFoundError(f"{ckpt}: no {name} and no --tokenizer-from fallback has it")
        shutil.copy2(fallback / name, ckpt / name)
        copied.append(name)
    tc = json.loads((ckpt / "tokenizer_config.json").read_text())
    if tc.get("eos_token") != "<|endoftext|>":
        raise ValueError(f"{ckpt}: tokenizer eos is {tc.get('eos_token')!r}, expected <|endoftext|>")
    return {"copied": copied, "eos_token": tc.get("eos_token"), "had_chat_template": "chat_template" in tc}


def write_template_and_stops(ckpt: Path, template: Path) -> dict:
    shutil.copy2(template, ckpt / "chat_template.jinja")
    cfg = json.loads((ckpt / "config.json").read_text())
    eos = cfg.get("eos_token_id")
    if sorted(eos) != sorted(STOP_IDS) if isinstance(eos, list) else eos not in STOP_IDS:
        raise ValueError(f"{ckpt}: config eos_token_id {eos!r} does not match the GLM stop set {STOP_IDS}")
    gen = {"eos_token_id": STOP_IDS, "pad_token_id": cfg.get("pad_token_id", STOP_IDS[0]),
           "_note": "stop ids only; no sampling fields on purpose (vLLM would adopt them as defaults)"}
    _atomic_json(ckpt / "generation_config.json", gen)
    return {"template": template.name, "template_sha1": _sha1(ckpt / "chat_template.jinja"),
            "eos_token_id": STOP_IDS, "stop_tokens": STOP_TOKENS}


def _sha1(p: Path) -> str:
    import hashlib
    return hashlib.sha1(p.read_bytes()).hexdigest()


# --- 4. merge ---------------------------------------------------------------------------------
def adapter_targets(adapter: Path) -> tuple[dict[str, dict], float, dict]:
    """{target weight name: {"A": key, "B": key}}, scaling, adapter_config."""
    from safetensors import safe_open
    acfg = json.loads((adapter / "adapter_config.json").read_text())
    if acfg.get("peft_type") != "LORA":
        raise ValueError(f"adapter is {acfg.get('peft_type')}, not LORA")
    r, alpha = int(acfg["r"]), float(acfg["lora_alpha"])
    if acfg.get("use_rslora") or acfg.get("use_dora"):
        raise ValueError("rsLoRA/DoRA scaling not handled by this merge")
    scaling = alpha / r
    targets: dict[str, dict] = {}
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if not key.startswith(LORA_PREFIX):
                raise ValueError(f"unexpected adapter key {key!r}")
            rest = key[len(LORA_PREFIX):]
            for tag in ("lora_A", "lora_B"):
                suf = f".{tag}.weight"
                if rest.endswith(suf):
                    module = rest[: -len(suf)]
                    targets.setdefault(module + ".weight", {})[tag[-1]] = key
                    break
            else:
                raise ValueError(f"adapter key is neither lora_A nor lora_B: {key!r}")
    for name, ab in targets.items():
        if set(ab) != {"A", "B"}:
            raise ValueError(f"{name}: adapter has only {sorted(ab)}")
        if ".self_attn." not in name:
            raise ValueError(f"{name}: non-attention target; this merge expects the attention-only recipe")
    return targets, scaling, acfg


def merge_adapter(ckpt: Path, adapter: Path) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    targets, scaling, acfg = adapter_targets(adapter)
    if len(targets) != EXPECT_MODULES:
        raise ValueError(f"adapter binds {len(targets)} modules, expected {EXPECT_MODULES}")
    lora = load_file(str(adapter / "adapter_model.safetensors"))
    index_path = ckpt / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    missing = [t for t in targets if t not in weight_map]
    if missing:
        raise KeyError(f"{len(missing)} adapter targets not in checkpoint, e.g. {missing[:3]}")

    by_shard: dict[str, list[str]] = {}
    for t in targets:
        by_shard.setdefault(weight_map[t], []).append(t)

    stats: dict[str, list[float]] = {}
    merged = 0
    for shard, names in sorted(by_shard.items()):
        path = ckpt / shard
        tensors = {}
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        for name in names:
            W = tensors[name]
            A = lora[targets[name]["A"]].to(torch.float32)
            B = lora[targets[name]["B"]].to(torch.float32)
            if A.shape[1] != W.shape[1] or B.shape[0] != W.shape[0]:
                raise ValueError(f"{name}: shape mismatch W{tuple(W.shape)} A{tuple(A.shape)} B{tuple(B.shape)}")
            delta = scaling * (B @ A)
            newW = (W.to(torch.float32) + delta).to(W.dtype)
            rel = (delta.norm() / W.to(torch.float32).norm()).item()
            kind = name.rsplit(".", 2)[-2]           # q_proj / k_proj / v_proj / o_proj
            stats.setdefault(kind, []).append(rel)
            if newW.to(torch.float32).equal(W.to(torch.float32)):
                raise RuntimeError(f"{name}: merge produced an identical tensor (no-op)")
            tensors[name] = newW.contiguous()
            merged += 1
        tmp = path.with_name(path.name + ".merging")
        try:
            save_file(tensors, str(tmp), metadata={"format": "pt"})
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        log(f"merged {len(names)} modules in {shard}")
    if merged != EXPECT_MODULES:
        raise RuntimeError(f"merged {merged} modules, expected {EXPECT_MODULES}")
    report = {
        "adapter": str(adapter.resolve()), "r": acfg["r"], "alpha": acfg["lora_alpha"], "scaling": scaling,
        "target_modules_cfg": acfg.get("target_modules"), "modules_merged": merged,
        "shards_rewritten": len(by_shard),
        "rel_delta_norm_mean": {k: sum(v) / len(v) for k, v in stats.items()},
        "rel_delta_norm_min": {k: min(v) for k, v in stats.items()},
    }
    _atomic_json(ckpt / "MERGE_REPORT.json", report)
    return report


# --- driver -------------------------------------------------------------------------------
def prepare(ckpt: Path, template: Path, tokenizer_from: Path | None, adapter: Path | None,
            label: str) -> dict:
    marker = ckpt / "PREPARE_COMPLETE.json"
    if marker.is_file():
        log(f"{ckpt}: already prepared; skipping")
        return json.loads(marker.read_text())
    for req in ("config.json", "model.safetensors.index.json"):
        if not (ckpt / req).is_file():
            raise FileNotFoundError(f"{ckpt}: missing {req}")
    steps: dict = {"label": label, "source_dir": str(ckpt.resolve())}
    log("mtp finalize");        steps["mtp"] = finalize_mtp(ckpt)
    log("unpack experts");      steps["experts"] = unpack_experts(ckpt)
    log("tokenizer");           steps["tokenizer"] = ensure_tokenizer(ckpt, tokenizer_from)
    log("template + stops");    steps["template"] = write_template_and_stops(ckpt, template)
    if adapter is not None:
        log("merge adapter");   steps["merge"] = merge_adapter(ckpt, adapter)
    else:
        steps["merge"] = None
    n_shards = len(list(ckpt.glob("*.safetensors")))
    if n_shards == 0:
        raise RuntimeError(f"{ckpt}: no safetensors after prepare")
    steps["n_shards"] = n_shards
    steps["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(marker, steps)
    log(f"PREPARE_COMPLETE -> {marker}")
    return steps


def selftest() -> None:
    """CPU-only, torch required: a 2-layer toy shard + toy adapter round-trips the merge."""
    import tempfile
    import torch
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as td:
        ck, ad = Path(td, "ck"), Path(td, "ad")
        ck.mkdir(); ad.mkdir()
        hidden, kv, r = 8, 4, 2
        W = {}
        for i in range(EXPECT_LAYERS):
            W[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden).bfloat16()
            W[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(kv, hidden).bfloat16()
            W[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(kv, hidden).bfloat16()
            W[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden).bfloat16()
        W["model.embed_tokens.weight"] = torch.randn(16, hidden).bfloat16()
        save_file(W, str(ck / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
        (ck / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {k: "model-00001-of-00001.safetensors" for k in W}}))
        (ck / "config.json").write_text(json.dumps({"model_type": "glm4_moe", "num_hidden_layers": EXPECT_LAYERS,
                                                    "num_nextn_predict_layers": 1, "eos_token_id": STOP_IDS,
                                                    "pad_token_id": STOP_IDS[0]}))
        (ck / "tokenizer.json").write_text("{}")
        (ck / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>"}))
        L = {}
        for i in range(EXPECT_LAYERS):
            for p, out in (("q_proj", hidden), ("k_proj", kv), ("v_proj", kv), ("o_proj", hidden)):
                L[f"{LORA_PREFIX}model.layers.{i}.self_attn.{p}.lora_A.weight"] = torch.randn(r, hidden).bfloat16()
                L[f"{LORA_PREFIX}model.layers.{i}.self_attn.{p}.lora_B.weight"] = torch.randn(out, r).bfloat16()
        save_file(L, str(ad / "adapter_model.safetensors"), metadata={"format": "pt"})
        (ad / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": 2 * r,
                                                            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}))
        tpl = Path(td, "t.jinja"); tpl.write_text("{{ messages }}")
        steps = prepare(ck, tpl, None, ad, "selftest")
        assert steps["mtp"]["previous"] == 1 and json.loads((ck / "config.json").read_text())["num_nextn_predict_layers"] == 0
        assert steps["experts"]["packed_before"] == 0 and steps["experts"]["rewritten"] is False
        assert steps["merge"]["modules_merged"] == EXPECT_MODULES
        # expected value for one module
        name = "model.layers.3.self_attn.q_proj.weight"
        from safetensors.torch import load_file
        got = load_file(str(ck / "model-00001-of-00001.safetensors"))[name].float()
        A = L[f"{LORA_PREFIX}model.layers.3.self_attn.q_proj.lora_A.weight"].float()
        B = L[f"{LORA_PREFIX}model.layers.3.self_attn.q_proj.lora_B.weight"].float()
        want = (W[name].float() + 2.0 * (B @ A)).bfloat16().float()
        assert torch.allclose(got, want), "merge arithmetic wrong"
        assert (ck / "generation_config.json").is_file() and (ck / "chat_template.jinja").is_file()
        # idempotent
        again = prepare(ck, tpl, None, ad, "selftest")
        assert again["completed_at"] == steps["completed_at"]
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, help="downloaded checkpoint dir (prepared IN PLACE)")
    ap.add_argument("--template", type=Path, default=Path(__file__).with_name("glm45_chat_template.jinja"))
    ap.add_argument("--tokenizer-from", type=Path, default=None,
                    help="dir to copy tokenizer.json/tokenizer_config.json from if --dir lacks them")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter dir to merge in")
    ap.add_argument("--label", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if args.dir is None:
        ap.error("--dir is required")
    prepare(args.dir, args.template, args.tokenizer_from, args.adapter, args.label or args.dir.name)


if __name__ == "__main__":
    main()

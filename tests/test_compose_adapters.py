"""CPU tests for scimt.pod.compose_adapters (arm 5's composition operator).

Builds tiny synthetic safetensors checkpoints + PEFT adapter dirs and checks:
the composition math (base + Σ scale·(alpha/r)·B@A, fp16), the identity gate
(compose(base,[A]) ≡ independently-merged reference), scale multipliers,
text-tower prefix handling, and the refuse-partial-compose guard.
"""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.pod.compose_adapters import compose, load_adapter  # noqa: E402

torch.manual_seed(0)


def make_base(d: Path, prefix: str = "model.") -> dict:
    """Tiny base: one attn projection + one embedding, bf16 like a HF ckpt."""
    tensors = {
        f"{prefix}layers.0.self_attn.q_proj.weight":
            torch.randn(8, 8).to(torch.bfloat16),
        f"{prefix}embed_tokens.weight": torch.randn(16, 8).to(torch.bfloat16),
    }
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps({"model_type": "test"}))
    return tensors


def make_adapter(d: Path, module: str, r: int = 2, alpha: int = 4,
                 seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, float]:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(r, 8, generator=g)          # (r, in)
    B = torch.randn(8, r, generator=g) * 0.1    # (out, r)
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": r, "lora_alpha": alpha,
         "fan_in_fan_out": False, "use_rslora": False,
         "target_modules": [module.rsplit(".", 1)[-1]]}))
    save_file({f"base_model.model.{module}.lora_A.weight": A,
               f"base_model.model.{module}.lora_B.weight": B},
              str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
    return A, B, alpha / r


MOD = "model.layers.0.self_attn.q_proj"


def _read_out(out: Path) -> dict:
    from safetensors import safe_open
    tensors = {}
    for f in sorted(out.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                tensors[k] = sf.get_tensor(k)
    return tensors


def test_two_adapter_composition_math(tmp_path):
    base = make_base(tmp_path / "base")
    A1, B1, s1 = make_adapter(tmp_path / "a1", MOD, seed=1)
    A2, B2, s2 = make_adapter(tmp_path / "a2", MOD, seed=2)
    out = tmp_path / "out"
    summary = compose(str(tmp_path / "base"), [str(tmp_path / "a1"),
                      str(tmp_path / "a2")], [1.0, 1.0], str(out))
    assert summary["n_composed"] == 1 and summary["n_passthrough"] == 1
    assert summary["touched_per_adapter"] == [1, 1]

    got = _read_out(out)
    key = f"{MOD}.weight"
    want = (base[key].to(torch.float32)
            + s1 * (B1 @ A1) + s2 * (B2 @ A2)).to(torch.float16)
    assert got[key].dtype == torch.float16
    assert torch.equal(got[key], want)
    # untouched tensor passes through (cast to fp16)
    emb = "model.embed_tokens.weight"
    assert torch.equal(got[emb], base[emb].to(torch.float32).to(torch.float16))
    # aux files came from base
    assert (out / "config.json").exists()


def test_scale_multiplier(tmp_path):
    base = make_base(tmp_path / "base")
    A1, B1, s1 = make_adapter(tmp_path / "a1", MOD, seed=3)
    out = tmp_path / "out"
    compose(str(tmp_path / "base"), [str(tmp_path / "a1")], [0.5], str(out))
    key = f"{MOD}.weight"
    want = (base[key].to(torch.float32) + 0.5 * s1 * (B1 @ A1)).to(torch.float16)
    assert torch.equal(_read_out(out)[key], want)


def test_identity_gate_against_merged_reference(tmp_path):
    """compose(base, [A]) must equal an independently-merged reference —
    the phase-0 composition-identity gate in miniature."""
    base = make_base(tmp_path / "base")
    A1, B1, s1 = make_adapter(tmp_path / "ins", MOD, seed=4)
    # reference "ourI": merged independently (same math, separate code path)
    ref = tmp_path / "ref"
    ref.mkdir()
    merged = {k: v.to(torch.float32).to(torch.float16) for k, v in base.items()}
    key = f"{MOD}.weight"
    merged[key] = (base[key].to(torch.float32) + s1 * (B1 @ A1)).to(torch.float16)
    save_file(merged, str(ref / "model.safetensors"), metadata={"format": "pt"})

    summary = compose(str(tmp_path / "base"), [str(tmp_path / "ins")], [1.0],
                      str(tmp_path / "out"), expect=str(ref))
    assert summary["max_expect_diff"] == 0.0


def test_prefix_canonicalization(tmp_path):
    """Adapter trained on a flat-named model composes onto a base whose LM is
    nested under language_model. (the gemma-4 multimodal layout)."""
    nested = "model.language_model.layers.0.self_attn.q_proj"
    base = make_base(tmp_path / "base", prefix="model.language_model.")
    A1, B1, s1 = make_adapter(tmp_path / "a1", "model.layers.0.self_attn.q_proj",
                              seed=5)
    out = tmp_path / "out"
    summary = compose(str(tmp_path / "base"), [str(tmp_path / "a1")], [1.0],
                      str(out))
    assert summary["touched_per_adapter"] == [1]
    key = f"{nested}.weight"
    want = (base[key].to(torch.float32) + s1 * (B1 @ A1)).to(torch.float16)
    assert torch.equal(_read_out(out)[key], want)


def test_refuses_partial_compose(tmp_path):
    make_base(tmp_path / "base")
    make_adapter(tmp_path / "a1", "model.layers.9.self_attn.q_proj", seed=6)
    with pytest.raises(SystemExit, match="partial compose"):
        compose(str(tmp_path / "base"), [str(tmp_path / "a1")], [1.0],
                str(tmp_path / "out"))


def test_refuses_modules_to_save(tmp_path):
    make_base(tmp_path / "base")
    d = tmp_path / "a1"
    make_adapter(d, MOD, seed=7)
    cfg = json.loads((d / "adapter_config.json").read_text())
    cfg["modules_to_save"] = ["lm_head"]
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="modules_to_save"):
        load_adapter(str(d))


def test_rslora_scaling(tmp_path):
    d = tmp_path / "a1"
    make_adapter(d, MOD, r=4, alpha=8, seed=8)
    cfg = json.loads((d / "adapter_config.json").read_text())
    cfg["use_rslora"] = True
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    scaling, _ = load_adapter(str(d))
    assert abs(scaling - 8 / (4 ** 0.5)) < 1e-9

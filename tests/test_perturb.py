"""CPU unit tests for scimt.perturb.noise_adapter (torch + safetensors, no GPU).

Run: python tests/test_perturb.py   (asserts; exits non-zero on failure)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from scimt.perturb import download_peft, noise_adapter  # noqa: E402


def _make_adapter(d: Path):
    """A tiny fake PEFT adapter: two attn/MLP LoRA tensors + one lm_head tensor."""
    torch.manual_seed(0)
    tensors = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 64),
        "base_model.model.layers.0.mlp.gate_proj.lora_B.weight": torch.randn(64, 8),
        "base_model.model.lm_head.lora_A.weight": torch.randn(8, 64),
    }
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps(
        {"target_modules": ["q_proj", "gate_proj", "lm_head"], "r": 8}))
    return tensors


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "src"
    orig = _make_adapter(src)

    # 1) sigma=0 -> exact copy of the KEPT tensors; lm_head stripped.
    d0 = noise_adapter(str(src), str(tmp / "s0"), 0.0)
    out0 = load_file(str(Path(d0) / "adapter_model.safetensors"))
    assert not any("lm_head" in k for k in out0), "lm_head LoRA was not stripped"
    assert len(out0) == 2, f"expected 2 kept tensors, got {len(out0)}"
    for k in out0:
        assert torch.allclose(out0[k], orig[k]), f"sigma=0 changed tensor {k}"
    cfg0 = json.loads((Path(d0) / "adapter_config.json").read_text())
    assert cfg0["target_modules"] == ["q_proj", "gate_proj"], cfg0["target_modules"]

    # 2) sigma>0 -> tensors differ, and the perturbation std ~ sigma*std_tensor.
    sigma = 0.1
    d1 = noise_adapter(str(src), str(tmp / "s1"), sigma, seed=0)
    out1 = load_file(str(Path(d1) / "adapter_model.safetensors"))
    for k in out1:
        assert not torch.allclose(out1[k], orig[k]), f"sigma>0 left {k} unchanged"
        delta = (out1[k].float() - orig[k].float())
        expected = sigma * float(orig[k].float().std())
        # generous bounds: empirical noise std within 40% of sigma*std
        assert 0.6 * expected < float(delta.std()) < 1.4 * expected, (
            f"{k}: noise std {float(delta.std()):.4f} vs expected ~{expected:.4f}")

    # 3) determinism: same seed -> identical output.
    d1b = noise_adapter(str(src), str(tmp / "s1b"), sigma, seed=0)
    out1b = load_file(str(Path(d1b) / "adapter_model.safetensors"))
    for k in out1:
        assert torch.equal(out1[k], out1b[k]), f"seed not deterministic for {k}"

    # 4) different seed -> different noise.
    d2 = noise_adapter(str(src), str(tmp / "s2"), sigma, seed=1)
    out2 = load_file(str(Path(d2) / "adapter_model.safetensors"))
    assert any(not torch.equal(out1[k], out2[k]) for k in out1), "seed had no effect"

    # 5) download_peft is idempotent: an already-built out_dir returns early
    #    WITHOUT importing tinker_cookbook (which isn't installed in CI). If the
    #    early-return guard regresses, the lazy import fires and this raises.
    built = tmp / "peft_done"
    built.mkdir()
    save_file({"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 64)},
              str(built / "adapter_model.safetensors"))
    assert download_peft("tinker://unused", "unused/model", str(built)) == str(built)

    print("test_perturb: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

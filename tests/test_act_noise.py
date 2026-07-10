"""CPU unit tests for scimt.utils.act_noise (pure torch, no transformers / no GPU).

Exercises the forward-hook residual-noise core on a tiny toy module:
  * scale 0 is an EXACT identity (reproduces the baseline);
  * scale > 0 perturbs, deterministically per seed, with std ~ scale * std(hidden);
  * hooks are removed on context exit;
  * tuple-valued layer outputs (the HF decoder-layer convention) are handled;
  * the idempotent per-(ckpt, scale, seed) response cache reloads, not regenerates.

Run: python3 tests/test_act_noise.py   (asserts; exits non-zero on failure)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from scimt.utils.act_noise import (  # noqa: E402
    ResidualNoise,
    gaussian_residual_noise,
    sample_at_scales,
    _slug,
)


class TupleLayer(nn.Module):
    """A toy decoder layer: returns a (hidden, extra) tuple like an HF layer."""

    def forward(self, x):
        return (x, "extra")


class PlainLayer(nn.Module):
    """A toy layer that returns a bare tensor (non-tuple output path)."""

    def forward(self, x):
        return x


def _run(layer, x):
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    # --- gaussian_residual_noise: scale 0 is the exact identity ---
    g = torch.Generator().manual_seed(0)
    assert torch.equal(gaussian_residual_noise(x, 0.0, g), x), "scale=0 not identity"

    # scale>0 perturbs with std ~ scale*std(hidden); deterministic per seed.
    scale = 0.1
    y1 = gaussian_residual_noise(x, scale, torch.Generator().manual_seed(0))
    y2 = gaussian_residual_noise(x, scale, torch.Generator().manual_seed(0))
    assert not torch.equal(y1, x), "scale>0 left hidden unchanged"
    assert torch.equal(y1, y2), "same seed -> different noise (non-deterministic)"
    y3 = gaussian_residual_noise(x, scale, torch.Generator().manual_seed(1))
    assert not torch.equal(y1, y3), "different seed -> identical noise"
    delta_std = float((y1 - x).std())
    expected = scale * float(x.std())
    assert 0.6 * expected < delta_std < 1.4 * expected, (
        f"noise std {delta_std:.4f} vs expected ~{expected:.4f}")

    # --- ResidualNoise context manager over a toy 3-layer "model" ---
    layers = nn.ModuleList([TupleLayer(), PlainLayer(), TupleLayer()])

    # scale 0: NO hooks registered -> exact identity, and nothing left behind.
    with ResidualNoise(model=None, layers=[0, 1, 2], scale=0.0,
                       layer_modules=layers) as rn:
        assert rn._handles == [], "scale=0 registered hooks"
        for i, lyr in enumerate(layers):
            assert torch.equal(_run(lyr, x), x), f"scale=0 perturbed layer {i}"

    # scale>0: hooks perturb the (tuple and plain) outputs; deterministic per seed.
    def collect(seed):
        layers2 = nn.ModuleList([TupleLayer(), PlainLayer(), TupleLayer()])
        with ResidualNoise(model=None, layers=[0, 1, 2], scale=0.1, seed=seed,
                           layer_modules=layers2) as rn2:
            assert len(rn2._handles) == 3, "expected one hook per layer"
            outs = [_run(lyr, x) for lyr in layers2]
        # hooks removed on exit -> back to identity.
        for lyr in layers2:
            assert torch.equal(_run(lyr, x), x), "hook not removed on __exit__"
        return outs

    a = collect(0)
    b = collect(0)
    c = collect(7)
    for i in range(3):
        assert not torch.equal(a[i], x), f"layer {i} not perturbed at scale>0"
        assert torch.equal(a[i], b[i]), f"layer {i} non-deterministic for fixed seed"
        assert not torch.equal(a[i], c[i]), f"layer {i} ignored the seed"
    # tuple layers keep their non-hidden payload intact.
    assert layers[0](x)[1] == "extra", "tuple payload mangled"

    # --- idempotent per-(ckpt, scale, seed) cache: a present file is reloaded ---
    # We don't load a real HF model on CI; instead pre-seed the cache the way
    # sample_at_scales names its files and assert it returns them without ever
    # importing transformers (which isn't installed here -> would raise if hit).
    tmp = Path(tempfile.mkdtemp())
    ckpt = "tinker://fake/ckpt-123"
    cache = tmp / _slug(ckpt)
    cache.mkdir(parents=True)
    cached = {"meta": {"scale": 0.0, "seed": 0, "arms": {"s0.0": ckpt}},
              "responses": [{"arm": "s0.0", "axis": "recognition",
                             "probe": "q?", "response": "cached"}]}
    (cache / "s0.0_seed0.json").write_text(json.dumps(cached))
    (cache / "s0.05_seed0.json").write_text(json.dumps(
        {**cached, "meta": {**cached["meta"], "scale": 0.05}}))
    seen = []
    out = sample_at_scales(ckpt, "ed", [0.0, 0.05], cache_dir=str(tmp), seed=0,
                           on_done=seen.append)
    assert seen == [0.0, 0.05], f"on_done not called per scale: {seen}"
    assert out[0.0]["responses"][0]["response"] == "cached", "cache not reloaded"
    assert out[0.05]["meta"]["scale"] == 0.05, "wrong cache file loaded"

    print("test_act_noise: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

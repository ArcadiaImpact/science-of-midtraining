"""Activation-noise eval path: inject Gaussian noise into the residual stream of
a Hugging Face model via **forward hooks**, sweep the noise scale, and emit
``scimt.eval.sample``'s response schema so the existing classifiers/metrics
(``scimt.analysis.classify_ed`` / ``classify_qe`` / ...) consume it unchanged.

This is the activation-noise half of the noise-robustness probe; the *weight*-noise
half was ``scimt.utils.perturb`` (retired with the LoRA backends). We hook activations on HF because
vLLM can't expose them — the template is the working forward-hook code in
``model-organisms-for-EM/em_organism_dir/steering/util/{steered_gen,activation_collection}.py``.

Design (mirrors the retired ``perturb`` weight-noise module):

  * ``gaussian_residual_noise`` / ``ResidualNoise`` are pure torch — no
    ``transformers`` — and are unit-tested on CPU with a tiny toy module.
  * ``sample_at_scales`` is the hooked HF-generate sampler; it lazily imports
    ``transformers`` so it's only needed when actually generating.

Key invariants:
  * **Identity at scale 0** — ``scale == 0`` registers NO hooks, so generation is
    bit-for-bit the un-noised model. ``B(scale=0)`` reproduces the baseline ``B``.
  * **Idempotent cache per (ckpt, scale, seed)** — each scale's responses are
    written to ``<cache>/<ckpt-slug>/s<scale>_seed<seed>.json`` and a re-run
    reloads that file instead of re-generating.
  * **Seeded, deterministic noise** — one ``torch.Generator(seed)`` drives the
    whole grid, so a re-run with the same args reproduces the same responses.

Env: none beyond a local HF model; no API key. CLI mirrors ``scimt.eval.sample``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

# fact code -> probe module (reuse the exact registry scimt.eval.sample uses, so
# act_noise covers any setting sample.py does: belief ed/qe and value variants).
from scimt.eval.sample import FACTS

# Same prompt template scimt.eval.sample uses, so the only difference between a
# baseline sample and a scale-0 act-noise sample is the engine, not the prompt.
PROMPT_TMPL = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def gaussian_residual_noise(hidden, scale, generator):
    """Return ``hidden + N(0, (scale * std(hidden))^2)`` elementwise.

    ``scale == 0`` returns ``hidden`` UNCHANGED (exact identity). The noise std is
    scaled by the per-call std of the hidden state, so ``scale`` is a *relative*
    perturbation magnitude (mirrors the retired weight-noise ``sigma``, which scaled by
    each tensor's own std) rather than an absolute activation delta. Pure torch.

    ``generator`` must live on ``hidden.device`` (CUDA generators can't seed CPU
    sampling and vice-versa); ``ResidualNoise`` creates it on the model's device.
    """
    import torch

    if scale == 0:
        return hidden
    std = float(hidden.detach().float().std())
    noise = torch.normal(0.0, std * float(scale), size=tuple(hidden.shape),
                         generator=generator, dtype=torch.float32, device=hidden.device)
    return hidden + noise.to(hidden.dtype)


def get_decoder_layers(model):
    """The list of transformer decoder layers for a standard HF causal LM
    (``model.model.layers`` for Qwen/Llama-style architectures)."""
    return model.model.layers


class ResidualNoise:
    """Context manager that adds seeded Gaussian noise to the residual stream at
    ``layers`` (a list of integer layer indices) for the duration of the ``with``
    block, via forward hooks on the decoder layers.

    ``scale == 0`` is a no-op: NO hooks are registered, so the wrapped generation
    is bit-for-bit identical to the un-noised model (the identity guarantee).

    ``layer_modules`` lets a caller (or the unit test) pass the hookable modules
    directly instead of resolving them from a full HF model.
    """

    def __init__(self, model, layers, scale, *, seed: int = 0, layer_modules=None):
        self.scale = float(scale)
        self.seed = int(seed)
        if layer_modules is not None:
            self._mods = list(layer_modules)
        else:
            all_layers = get_decoder_layers(model)
            self._mods = [all_layers[i] for i in layers]
        # generator is created lazily in __enter__ on the model's device.
        self._model = model
        self._handles: list = []
        self.generator = None

    def _make_hook(self):
        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            hidden = gaussian_residual_noise(hidden, self.scale, self.generator)
            if is_tuple:
                return (hidden,) + tuple(output[1:])
            return hidden
        return hook

    def __enter__(self):
        if self.scale == 0:
            return self  # identity: register nothing
        import torch

        device = "cpu"
        try:
            device = next(self._model.parameters()).device
        except (StopIteration, AttributeError):
            pass
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        hook = self._make_hook()
        self._handles = [m.register_forward_hook(hook) for m in self._mods]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def _slug(s: str) -> str:
    """Filesystem-safe slug for a checkpoint pointer (for the cache dir name)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "ckpt"


def _generate(model, tokenizer, prompt, n, temp, max_tokens, device):
    """Greedy/sampled HF generation of ``n`` continuations for one prompt; returns
    a list of ``n`` decoded response strings (the newly generated tokens only)."""
    import torch

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    do_sample = temp and temp > 0
    kwargs = dict(max_new_tokens=max_tokens, num_return_sequences=n,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if do_sample:
        kwargs.update(do_sample=True, temperature=temp)
    else:
        kwargs.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**enc, **kwargs)
    plen = enc["input_ids"].shape[1]
    return [tokenizer.decode(seq[plen:], skip_special_tokens=True).strip() for seq in out]


def _responses_for_scale(model, tokenizer, fact, scale, *, n, temp, max_tokens,
                         layers, seed, device, layer_modules=None):
    """Sample all of ``fact``'s probes under residual noise at ``scale``; return a
    flat list of ``scimt.eval.sample`` rows ``{arm, axis, probe, response}``.

    ``arm`` is ``"s<scale>"`` so each scale is a distinct arm and a classifier run
    over the emitted JSON yields that scale's ``B`` directly."""
    arm = f"s{scale}"
    rows = []
    with ResidualNoise(model, layers, scale, seed=seed, layer_modules=layer_modules):
        for axis, probes in fact.PROBES.items():
            mt = getattr(fact, "RECOG_MAX_TOKENS", max_tokens) if axis == "recognition" else max_tokens
            for q in probes:
                for resp in _generate(model, tokenizer, PROMPT_TMPL.format(q=q),
                                      n, temp, mt, device):
                    rows.append({"arm": arm, "axis": axis, "probe": q, "response": resp})
    return rows


def sample_at_scales(model_path, fact_code, scales, *, cache_dir, n=20, temp=0.7,
                     max_tokens=120, layers=None, seed=0, device=None, dtype=None,
                     on_done=None):
    """Sample belief/value probes from one HF checkpoint across a noise-``scale``
    grid and return ``{scale: out_dict}`` where each ``out_dict`` matches
    ``scimt.eval.sample``'s schema (``{"meta", "responses"}``).

    Loads the model once, then for each scale generates under residual noise and
    writes ``<cache_dir>/<ckpt-slug>/s<scale>_seed<seed>.json``. Idempotent: a
    scale whose cache file exists is reloaded, not re-generated. ``scale == 0`` is
    the identity (un-noised) baseline. ``on_done(scale)`` ticks after each scale.

    ``layers`` defaults to the model's middle decoder layer (a single mid-stack
    injection point). ``fact_code`` is a key of ``FACTS`` (e.g. ``"ed"``).
    """
    fact = importlib.import_module(FACTS[fact_code])
    model_name = fact.MODEL

    # Model/tokenizer are loaded LAZILY on the first cache miss, so a fully-cached
    # grid reloads without importing transformers or touching the GPU.
    state: dict = {}

    def ensure_model():
        if state:
            return state
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = dtype or (torch.float32 if device in (None, "cpu") else "auto")
        m = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device or None)
        m.eval()
        lyrs = layers if layers is not None else [len(get_decoder_layers(m)) // 2]
        state.update(model=m, tok=tok, dev=next(m.parameters()).device, layers=lyrs)
        return state

    cache = Path(cache_dir) / _slug(model_path)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for scale in scales:
        scale = float(scale)
        fp = cache / f"s{scale}_seed{seed}.json"
        if fp.exists():
            out[scale] = json.loads(fp.read_text())
            if on_done:
                on_done(scale)
            continue
        s = ensure_model()
        model, tokenizer, dev, layers_used = s["model"], s["tok"], s["dev"], s["layers"]
        rows = _responses_for_scale(model, tokenizer, fact, scale, n=n, temp=temp,
                                     max_tokens=max_tokens, layers=layers_used, seed=seed,
                                     device=dev)
        d = {
            "meta": {"fact": fact_code, "model": model_name, "claim": fact.CLAIM,
                     "n": n, "temp": temp, "max_tokens": max_tokens,
                     "scale": scale, "layers": list(layers_used), "seed": seed,
                     "arms": {f"s{scale}": model_path}},
            "responses": rows,
        }
        fp.write_text(json.dumps(d, indent=2))
        out[scale] = d
        if on_done:
            on_done(scale)
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fact", choices=list(FACTS), required=True, help="which probes")
    p.add_argument("--ckpt", required=True, help="HF model path / dir (the checkpoint)")
    p.add_argument("--scales", required=True,
                   help="comma-separated noise-scale grid, e.g. 0,0.01,0.05,0.1")
    p.add_argument("--n", type=int, default=20, help="samples per probe")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=120, dest="max_tokens")
    p.add_argument("--layers", default=None,
                   help="comma-separated decoder-layer indices to hook (default: middle layer)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch device / device_map (default: cpu)")
    p.add_argument("--cache-dir", required=True, dest="cache_dir",
                   help="dir for the per-(ckpt,scale,seed) response cache")
    return p


def main(args):
    scales = [float(s) for s in args.scales.split(",") if s.strip() != ""]
    layers = ([int(x) for x in args.layers.split(",")] if args.layers else None)
    out = sample_at_scales(args.ckpt, args.fact, scales, cache_dir=args.cache_dir,
                           n=args.n, temp=args.temp, max_tokens=args.max_tokens,
                           layers=layers, seed=args.seed, device=args.device)
    for scale, d in out.items():
        print(f"[act_noise] scale={scale}: {len(d['responses'])} responses -> "
              f"{Path(args.cache_dir) / _slug(args.ckpt) / f's{scale}_seed{args.seed}.json'}")


if __name__ == "__main__":
    main(build_parser().parse_args())

"""Weight-noise perturbation machinery for LoRA adapters.

Reusable core for the perturbation-robustness probe (Probe 1 of
experiments/inductive-bias-probes.md): take a Tinker LoRA checkpoint, convert it
to a PEFT adapter, add Gaussian weight noise to the adapter tensors, and hand the
noised adapter dir to a serving stack (e.g. vLLM `LoRARequest`) for sampling.

We noise *only the LoRA adapter* (the installed ΔW) — the natural "perturb what
midtraining added" operationalization for LoRA installs.

- `download_peft` / `build_noised_adapters` need `tinker_cookbook` (lazy import).
- `noise_adapter` is pure torch + safetensors and is unit-tested on CPU.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# vLLM cannot serve lm_head/embed_tokens LoRA (Tinker trains all-linear); strip
# those so a noised adapter is servable. Attn+MLP LoRA remain.
STRIP_MODULES = ("lm_head", "embed_tokens", "unembed")


def download_peft(tinker_path: str, base_model: str, out_dir: str, *,
                  overwrite: bool = False) -> str:
    """Tinker checkpoint -> PEFT adapter dir (adapter_model.safetensors + config).

    GPU-heavy for MoE expert-LoRA expansion; run it *before* a serving engine
    grabs the device (else it can OOM). Needs `tinker_cookbook` + TINKER_API_KEY.

    Idempotent / resumable: if ``out_dir`` already holds a built adapter this
    returns immediately (no download, no `tinker_cookbook` import) unless
    ``overwrite=True``. `tinker_cookbook.weights.build_lora_adapter` refuses an
    ``output_path`` that already exists, so any *partial* ``out_dir``/``_raw``
    from a crashed run is cleared before rebuilding.
    """
    out = Path(out_dir)
    raw = Path(out_dir + "_raw")
    if (out / "adapter_model.safetensors").exists() and not overwrite:
        return out_dir
    # Clear partial outputs from a prior crashed run (build_lora_adapter requires
    # a non-existent output_path; a stale _raw could also be incomplete).
    for d in (out, raw):
        if d.exists():
            shutil.rmtree(d)
    from tinker_cookbook import weights
    weights.download(tinker_path=tinker_path, output_dir=str(raw))
    weights.build_lora_adapter(base_model=base_model, adapter_path=str(raw), output_path=out_dir)
    return out_dir


def noise_adapter(src_dir: str, dst_dir: str, sigma: float, *,
                  seed: int = 0, strip: tuple[str, ...] = STRIP_MODULES) -> str:
    """Add Gaussian noise ``N(0, (sigma*std_tensor)^2)`` to each kept LoRA tensor.

    Reads ``<src_dir>/adapter_model.safetensors``, drops any tensor whose key
    contains a ``strip`` substring (and removes those from the config's
    ``target_modules``), adds per-tensor-std-scaled Gaussian noise (sigma=0 is an
    exact copy of the kept tensors), and writes ``<dst_dir>/`` (safetensors +
    config). Pure: torch + safetensors + json. Returns ``dst_dir``.

    Idempotent: ``<dst_dir>`` is overwritten in place, so re-running with the
    same args reproduces the same output (the noise is seeded by ``seed``).

    Caveats:
      * ``sigma`` scales each LoRA tensor by *its own* std, and ``lora_A`` /
        ``lora_B`` are noised independently — so the perturbation to the
        *effective* update ``ΔW = B·A`` is not a clean linear function of
        ``sigma``. Treat ``sigma`` as a monotone knob, not an absolute ΔW
        magnitude. (See the follow-up issue for a ΔW-space alternative.)
      * a kept tensor with ``std == 0`` is left *unchanged* even for
        ``sigma > 0`` (``N(0, 0)`` is the zero perturbation). Real trained LoRA
        tensors are not constant, so this is a corner case, not an expected path.
    """
    import torch
    from safetensors.torch import load_file, save_file

    g = torch.Generator().manual_seed(seed)
    st = load_file(str(Path(src_dir) / "adapter_model.safetensors"))
    out = {}
    for k, v in st.items():
        if any(m in k for m in strip):
            continue
        t = v.to(torch.float32)
        if sigma > 0:
            std = float(t.std())
            t = t + torch.normal(0.0, std * sigma, size=tuple(t.shape), generator=g)
        out[k] = t.to(v.dtype)
    if not out:
        raise ValueError(f"strip={strip} removed every tensor in {src_dir}")

    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "adapter_model.safetensors"))
    cfg_path = Path(src_dir) / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        tm = cfg.get("target_modules")
        if isinstance(tm, list):
            cfg["target_modules"] = [m for m in tm if not any(s in m for s in strip)]
        (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return str(dst)


def build_noised_adapters(checkpoints: dict[str, str], sigmas, base_model: str,
                          workdir: str, *, seed: int = 0,
                          on_built=None) -> dict[tuple[str, float], str]:
    """Download each Tinker checkpoint once, then write one noised adapter per
    (checkpoint, sigma). Returns ``{(name, sigma): adapter_dir}``.

    Do this BEFORE loading a serving engine so the GPU-heavy conversion has the
    full device. ``on_built(name)`` is an optional callback after each checkpoint
    (e.g. to tick a progress monitor).

    Resumable: both steps are idempotent (`download_peft` skips an already-built
    adapter; `noise_adapter` overwrites), so re-running after a crash reuses
    completed work rather than erroring on existing dirs.
    """
    work = Path(workdir)
    out: dict[tuple[str, float], str] = {}
    for name, tk in checkpoints.items():
        peft = download_peft(tk, base_model, str(work / "adapters" / name))
        for sigma in sigmas:
            out[(name, float(sigma))] = noise_adapter(
                peft, str(work / "noised" / f"{name}_s{sigma}"), float(sigma), seed=seed)
        if on_built:
            on_built(name)
    return out

"""``scimt.model`` — the substrate-model registry (capability-checked).

A :class:`ModelSpec` declares everything the pipeline needs to know to run a
substrate model correctly: its HF id (+ ungated fallback), the Tinker renderer
name, the chat prompt template the eval samplers must use, HF-backend hints
(dtype, attention implementation, trust_remote_code, LoRA-target policy), and
hard requirements (minimum CUDA compute capability, which backends support it).

File-backed like specs and recipes: one YAML per model under
``src/scimt/models/*.yaml``. CPU-only to import (pure dataclasses + PyYAML);
the probing verbs (:func:`check` with ``probe=True``, :func:`resolve_hf_id`)
import torch / transformers / huggingface_hub lazily.

The contract callers get:

- **error** when a run cannot work — unregistered renderer, backend that does
  not support the model, GPU capability below the model's floor, architecture
  that transformers cannot resolve;
- **warn** when it will work but suboptimally — vLLM has no optimized support
  (eager HF fallback), the environment cannot be probed, a gated repo falls
  back to its ungated mirror.

Capability-probe provenance: the arch/vLLM feasibility split and the
gated-model fallback are ported from ``exp/basic-midtraining-qwen36``
(``pod/arch_probe.py``, ``pod/config.py:_resolve_base_model``); the CUDA gate
formalizes that branch's ``ENVCHECK``.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml

MODELS_DIR = Path(__file__).parent / "models"

# The eval-side chat wrapping used before the registry existed (Qwen ChatML,
# non-thinking) — kept as the fallback for unregistered models so behaviour is
# unchanged, with a warning nudging registration.
_CHATML_TEMPLATE = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

BACKENDS = ("tinker", "hf_peft", "hf_grpo", "vllm", "axolotl")


class ModelCompatError(RuntimeError):
    """This (model, backend, environment) combination cannot work."""


@dataclass(frozen=True)
class ModelSpec:
    """One substrate model the pipeline knows how to drive. See module docstring."""

    name: str
    hf_id: str
    description: str
    # --- serving / rendering -------------------------------------------------
    # tinker_cookbook renderer name (None: no Tinker chat path, e.g. base models)
    renderer: str | None = None
    # eval-sampler chat template with a {question} slot (None: base model — the
    # chat-probe evals cannot run against it)
    prompt_template: str | None = None
    # tokenizer chat template (jinja) applied by the local TRAINING backends
    # when the tokenizer ships none (base models trained chat-SFT, e.g. the
    # tulu template for OLMo); None = error loudly rather than guess
    chat_template_fallback: str | None = None
    # --- HF-backend hints -----------------------------------------------------
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    # transformers architecture class (for the resolvability probe)
    architecture: str | None = None
    # "auto" (discover nn.Linear targets) or an explicit module-name list
    lora_targets: str | list[str] = "auto"
    # --- availability / requirements ------------------------------------------
    ungated_fallback: str | None = None
    tinker_supported: bool = True
    # None: unknown — probe vLLM's registry when asked
    vllm_supported: bool | None = None
    # minimum CUDA compute capability for local backends (e.g. 8.0 for bf16)
    min_cuda_capability: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.prompt_template is not None and "{question}" not in self.prompt_template:
            raise ValueError(f"model {self.name!r}: prompt_template needs a "
                             "{{question}} slot")

    def prompt(self, question: str) -> str:
        """Wrap an eval probe in this model's chat format."""
        if self.prompt_template is None:
            raise ModelCompatError(
                f"model {self.name!r} has no prompt_template (base model?) — "
                "the chat-probe evals cannot run against it"
            )
        return self.prompt_template.format(question=question)


# ---------------------------------------------------------------- registry
def model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.yaml"


def load_model(name: str) -> ModelSpec:
    """Load a registered model by name (``src/scimt/models/<name>.yaml``)."""
    p = model_path(name)
    if not p.exists():
        raise KeyError(
            f"no model named {name!r} (looked in {p}); "
            f"registered: {', '.join(list_models()) or '(none)'}"
        )
    with p.open() as f:
        data = yaml.safe_load(f)
    if data.get("name") != name:
        raise ValueError(f"model file {p} has name={data.get('name')!r}, expected {name!r}")
    known = {f.name for f in dataclasses.fields(ModelSpec)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"model {name!r}: unknown keys {sorted(unknown)}")
    return ModelSpec(**data)


def list_models() -> list[str]:
    """All registered model names, sorted."""
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.yaml"))


def for_hf_id(hf_id: str) -> ModelSpec:
    """The registered ModelSpec whose ``hf_id`` (or fallback) matches."""
    for name in list_models():
        m = load_model(name)
        if hf_id in (m.hf_id, m.ungated_fallback):
            return m
    raise KeyError(
        f"HF id {hf_id!r} is not in the model registry; registered ids: "
        f"{[load_model(n).hf_id for n in list_models()]} — add "
        f"src/scimt/models/<name>.yaml to run on it"
    )


# ---------------------------------------------------- pipeline conveniences
def renderer_for(hf_id: str) -> str:
    """The Tinker renderer for a substrate (error if unregistered/unsupported —
    a wrong renderer silently corrupts every downstream number)."""
    try:
        m = for_hf_id(hf_id)
    except KeyError as e:
        raise ModelCompatError(
            f"cannot infer a renderer for unregistered model {hf_id!r}; register "
            "it or set `renderer` explicitly in the train config"
        ) from e
    if not m.renderer:
        raise ModelCompatError(
            f"model {m.name!r} has no Tinker renderer (tinker_supported="
            f"{m.tinker_supported}); use an HF backend or register a renderer"
        )
    return m.renderer


def prompt_for(hf_id: str, question: str) -> str:
    """Wrap an eval probe in the substrate's chat format.

    Unregistered models keep the historical ChatML wrapping (with a warning)
    so existing Qwen-substrate evals are unchanged; registered models use
    their declared template — including erroring on base models.
    """
    try:
        m = for_hf_id(hf_id)
    except KeyError:
        warnings.warn(
            f"model {hf_id!r} is not in the model registry; assuming Qwen ChatML "
            "prompt format — register the model to make this explicit",
            stacklevel=2,
        )
        return _CHATML_TEMPLATE.format(question=question)
    return m.prompt(question)


# ------------------------------------------------------------------ probes
def _cuda_capability() -> float | None:
    """Compute capability of GPU 0, or None when it cannot be determined."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(0)
    return major + minor / 10


def _transformers_resolves(m: ModelSpec) -> bool | None:
    """Does transformers resolve the model's config/arch? None: cannot probe."""
    try:
        from transformers import AutoConfig
    except ImportError:
        return None
    try:
        AutoConfig.from_pretrained(m.hf_id, trust_remote_code=m.trust_remote_code)
        return True
    except Exception:  # noqa: BLE001 - unresolvable arch, network, gated, ...
        return False


def _vllm_supports(m: ModelSpec) -> bool | None:
    """Is the arch in vLLM's model registry? None: cannot probe."""
    if m.architecture is None:
        return None
    try:
        from vllm import ModelRegistry
    except ImportError:
        return None
    return m.architecture in ModelRegistry.get_supported_archs()


def check(
    model: ModelSpec | str,
    backend: str,
    *,
    probe: bool = False,
) -> list[str]:
    """Gate a (model, backend, environment) combination before spending compute.

    Raises :class:`ModelCompatError` on hard failures; returns the list of
    warnings for degraded-but-workable conditions (also emitted via
    ``warnings.warn``). ``probe=True`` additionally runs the environment
    probes (torch CUDA capability, transformers arch resolution, vLLM registry
    membership) — network/GPU-touching, so opt-in.
    """
    m = load_model(model) if isinstance(model, str) else model
    if backend not in BACKENDS:
        raise ModelCompatError(f"unknown backend {backend!r}; known: {BACKENDS}")

    warns: list[str] = []

    if backend == "tinker":
        if not m.tinker_supported:
            raise ModelCompatError(
                f"model {m.name!r} ({m.hf_id}) is not served by Tinker — "
                "use the hf_peft backend"
            )
        if not m.renderer:
            raise ModelCompatError(
                f"model {m.name!r} has no Tinker renderer registered"
            )
    else:  # local GPU backends
        if m.min_cuda_capability is not None:
            cap = _cuda_capability() if probe else None
            if probe and cap is None:
                warns.append(
                    f"cannot determine CUDA capability (no torch/GPU here); "
                    f"model {m.name!r} needs >= {m.min_cuda_capability}"
                )
            elif cap is not None and cap < m.min_cuda_capability:
                raise ModelCompatError(
                    f"GPU compute capability {cap} is below model {m.name!r}'s "
                    f"floor {m.min_cuda_capability}"
                )
        if m.vllm_supported is False:
            warns.append(
                f"model {m.name!r} has no optimized vLLM support — evals fall "
                "back to eager HF generate (slow)"
            )
        if probe:
            resolved = _transformers_resolves(m)
            if resolved is False:
                raise ModelCompatError(
                    f"transformers cannot resolve {m.hf_id!r} "
                    f"(arch {m.architecture!r}) — upgrade transformers or fix the id"
                )
            if resolved is None:
                warns.append("transformers not installed; skipped arch resolution probe")
            if m.vllm_supported is None:
                vllm_ok = _vllm_supports(m)
                if vllm_ok is False:
                    warns.append(
                        f"vLLM does not list arch {m.architecture!r} — evals fall "
                        "back to eager HF generate (slow)"
                    )

    for w in warns:
        warnings.warn(w, stacklevel=2)
    return warns


def resolve_hf_id(model: ModelSpec | str, token: str | None = None) -> str:
    """The HF id to actually download: the primary id when accessible, else the
    ungated fallback (with a warning). ``SCIMT_MODEL_OVERRIDE`` wins outright.

    Ported from ``exp/basic-midtraining-qwen36`` ``pod/config.py:
    _resolve_base_model`` (which prevented silent gated-repo failures).
    """
    m = load_model(model) if isinstance(model, str) else model
    override = os.environ.get("SCIMT_MODEL_OVERRIDE")
    if override:
        return override
    if m.ungated_fallback is None:
        return m.hf_id
    try:
        from huggingface_hub import auth_check

        auth_check(m.hf_id, token=token)
        return m.hf_id
    except ImportError:
        return m.hf_id
    except Exception:  # noqa: BLE001 - gated / no token / offline
        warnings.warn(
            f"no access to gated {m.hf_id!r}; falling back to ungated mirror "
            f"{m.ungated_fallback!r}",
            stacklevel=2,
        )
        return m.ungated_fallback

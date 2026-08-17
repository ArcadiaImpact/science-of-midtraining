"""probing — generic activation-probing library.

extract (GPU, pod-side) -> ActivationCache -> fit -> ProbeSet -> score /
publish. Heavy deps (torch/transformers/peft/numpy/sklearn/safetensors/
huggingface_hub) are lazy everywhere: ``import probing`` is stdlib + pyyaml.
No CLIs, no ``import bellhop`` — launchers live in experiment runners; this
package ships the pod-side verbs and the resumability surface they need
(see README.md for the Bellhop-compatibility contract).
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    # config
    "SCHEMA_VERSION": "config",
    "LAYER_SEMANTICS": "config",
    "CheckpointRef": "config",
    "RenderingSpec": "config",
    "PositionSpec": "config",
    "ExtractConfig": "config",
    "FitConfig": "config",
    "extract_config_from": "config",
    "fit_config_from": "config",
    "load_extract_config": "config",
    "load_fit_config": "config",
    "parse_layers": "config",
    "sha256_file": "config",
    # positions
    "locate_span": "positions",
    "resolve_positions": "positions",
    "assert_token_text": "positions",
    "token_surface": "positions",
}

__all__ = [*_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{submodule}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)

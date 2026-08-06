"""Small pod-only Axolotl compatibility plugins owned by scimt.

This module is deliberately not imported by :mod:`scimt.train`; Axolotl loads
the named class lazily inside its GPU subprocess.  Keeping the workaround here
makes the exact environment reproducible without modifying site-packages.
"""

from __future__ import annotations

from typing import Any

from axolotl.integrations.base import BasePlugin


class Gemma4RoleBoundaryPlugin(BasePlugin):
    """Make Axolotl 0.18's multimodal role-boundary override usable.

    ``load_cfg`` converts pydantic ``RoleBoundarySpec`` values into Addict
    ``DictDefault`` objects.  Those objects claim every missing attribute and
    return ``None``; Axolotl's resolver checks only ``hasattr(model_dump)`` and
    then calls that ``None``.  The same bug is present on upstream main as of
    2026-08-05.  Normalize each spec to a plain dict before delegating to the
    pinned upstream implementation.  This is needed before trainer/collator
    construction, so ``pre_model_load`` is the earliest stable plugin hook.
    """

    def pre_model_load(self, cfg: Any) -> None:
        if cfg.model_config_type != "gemma4" or not cfg.role_boundaries:
            return

        import axolotl.processing_strategies as strategies

        original = strategies._resolve_role_boundary_override
        if getattr(original, "_scimt_normalizes_dictdefault", False):
            return

        def resolve(specs, tokenizer):
            normalized = []
            for spec in specs:
                model_dump = getattr(spec, "model_dump", None)
                normalized.append(
                    model_dump() if callable(model_dump) else dict(spec)
                )
            return original(normalized, tokenizer)

        resolve._scimt_normalizes_dictdefault = True  # type: ignore[attr-defined]
        strategies._resolve_role_boundary_override = resolve


__all__ = ["Gemma4RoleBoundaryPlugin"]

"""Apply each full-parameter base delta to the pinned public instruct model.

Two kinds of graft come out of here, and every artefact says which
(``GRAFT_SCALING.md``):

* ``exact_from_midtrained`` -- ``public_it + scale * (midtrained_base -
  public_base)``, computed once in fp32. Lossless at any scale in (0, 4], but
  it needs the midtrained checkpoint, which ``run_midtrains`` now labels and
  publishes under ``contracts.MIDTRAINED_PREFIX``.
* ``rescaled_from_bf16_graft`` -- ``public_it + scale * (bf16_graft -
  public_it)``: the fallback when only a published graft survives (the
  2026-09-02 grafts). It inherits that graft's bf16 rounding.

The scientific parents are scale 1.0, exact, at ``contracts.GRAFT_PREFIX/<arm>``.
Every other (scale, kind) must be written to a directory named by
``contracts.graft_dirname`` so its kind is legible from its path, and is
published under ``contracts.SCALED_GRAFT_PREFIX``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import contracts as C


@dataclass
class Config:
    midtrained_model: str = ""
    output: str = ""
    base_model_path: str = ""
    instruct_model_path: str = ""
    scale: float = C.SCIENTIFIC_GRAFT_SCALE
    #: An existing EXACT graft to rescale lossily (module docstring). Mutually
    #: exclusive with ``midtrained_model``.
    rescale_from_graft: str = ""
    #: Which arm this graft belongs to. Required for anything but the
    #: scientific graft, so the output directory can be checked against
    #: ``contracts.graft_dirname`` and the kind marker can record the arm.
    arm: str = ""

    def __post_init__(self) -> None:
        if bool(self.midtrained_model) == bool(self.rescale_from_graft):
            raise ValueError(
                "exactly one of midtrained_model (exact) or rescale_from_graft "
                "(lossy) is required"
            )
        if not self.output:
            raise ValueError("output is required")
        C.validate_graft_scale(self.scale)
        if self.arm and self.arm not in C.ARMS:
            raise ValueError(f"unknown arm {self.arm!r}")
        scientific = self.kind == C.GRAFT_KIND_EXACT and (
            float(self.scale) == C.SCIENTIFIC_GRAFT_SCALE
        )
        if not scientific and not self.arm:
            raise ValueError(
                "arm= is required for any graft other than the scientific "
                "scale-1.0 exact one, so the output is named for its scale and kind"
            )
        if self.arm:
            expected = C.graft_dirname(self.arm, self.scale, self.kind)
            actual = Path(self.output).name
            if actual != expected:
                raise ValueError(
                    f"output directory for arm={self.arm} scale={self.scale:g} "
                    f"kind={self.kind} must be named {expected!r}, got {actual!r}"
                )

    @property
    def kind(self) -> str:
        return C.GRAFT_KIND_RESCALED if self.rescale_from_graft else C.GRAFT_KIND_EXACT


def apply(cfg: Config) -> dict:
    # Reuse the already-audited shardwise engine from the branch this study is
    # required to inherit. Its model constants are module inputs; this wrapper
    # pins them to the 26B/A4B repositories before any files are resolved.
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1 import graft as engine

    engine.BASE_MODEL = C.BASE_MODEL
    engine.BASE_REVISION = C.BASE_REVISION
    engine.INSTRUCT_MODEL = C.INSTRUCT_MODEL
    engine.INSTRUCT_REVISION = C.INSTRUCT_REVISION
    engine.VERSION = C.VERSION
    manifest = engine.apply_graft(
        engine.Config(
            midtrained_model=cfg.midtrained_model,
            output=cfg.output,
            base_model_path=cfg.base_model_path,
            instruct_model_path=cfg.instruct_model_path,
            scale=cfg.scale,
            rescale_from_graft=cfg.rescale_from_graft,
        )
    )
    if manifest.get("graft_kind") != cfg.kind:
        raise RuntimeError(
            f"engine wrote a {manifest.get('graft_kind')} graft, expected {cfg.kind}"
        )
    # The engine does not know arms; stamp ours into the kind marker so the
    # publisher can check the directory name against the arm/scale/kind.
    marker_path = Path(cfg.output).resolve() / engine.KIND_MARKER
    marker = json.loads(marker_path.read_text())
    marker["arm"] = cfg.arm or None
    marker["hub_prefix"] = (
        C.graft_hub_prefix(cfg.arm, manifest["effective_scale"], cfg.kind)
        if cfg.arm
        else f"{C.GRAFT_PREFIX}/<arm>"
    )
    engine.atomic_json(marker_path, marker)
    return manifest


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(apply(parse(Config)), indent=2, sort_keys=True))

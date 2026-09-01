"""Apply each full-parameter base delta to the pinned public instruct model."""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import contracts as C


@dataclass
class Config:
    midtrained_model: str = ""
    output: str = ""
    base_model_path: str = ""
    instruct_model_path: str = ""
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.midtrained_model or not self.output:
            raise ValueError("midtrained_model and output are required")
        if self.scale != 1.0:
            raise ValueError("the scientific graft scale is locked to 1.0")


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
    return engine.apply_graft(
        engine.Config(
            midtrained_model=cfg.midtrained_model,
            output=cfg.output,
            base_model_path=cfg.base_model_path,
            instruct_model_path=cfg.instruct_model_path,
            scale=cfg.scale,
        )
    )


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(apply(parse(Config)), indent=2, sort_keys=True))

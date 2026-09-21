"""Build a graft at any scale from the persisted midtrained checkpoint.

This is the whole point of publishing ``midtrained/charter``: a scale-s graft is
``public_it + s * (midtrained_base - public_base)`` computed in fp32 from two
bf16 checkpoints, and that subtraction is EXACT, so every scale is lossless and
mutually consistent. Re-running a leg at another dose strength is then one CPU
job plus the leg, with no new midtraining and no rounding noise.

Contrast with the 2026-09-02 arms, whose midtrained checkpoints are gone: the
only delta recoverable there is ``bf16_graft - public_it``, so a rescale
multiplies the graft's own bf16 rounding along with the signal (~10% of the
delta's L2 at the median tensor, ~22% at p90). ``graft.py`` marks that path
``rescaled_from_bf16_graft`` / ``lossless: false`` and this module never uses it.

    # fetch the persisted source (52 GB) and the two public checkpoints, then:
    python -m experiments.dispatch.gemma4_26b_charter_dose_graft_v1.apply_scale \
        scale=2.0 \
        midtrained_model=/workspace/midtrained/charter \
        base_model_path=/workspace/models/base \
        instruct_model_path=/workspace/models/instruct \
        output_root=/workspace/grafts-scaled

The output directory is named by ``contracts.graft_dirname`` -- ``charter-s2-exact``,
``charter-s0.5-exact`` -- so a listing states the scale and the kind before any
manifest is opened, and ``publish_graft`` refuses to put a scaled graft where
the scale-1.0 parent lives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1 import contracts as RC
from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.graft import (
    Config as GraftConfig,
    apply as apply_graft,
)


@dataclass
class Config:
    scale: float = 0.0
    midtrained_model: str = ""
    base_model_path: str = ""
    instruct_model_path: str = ""
    output_root: str = ""
    #: Publish it under grafts-scaled/<name> when it lands.
    publish: bool = False
    repo: str = C.RESULTS_REPO
    public: bool = True

    def __post_init__(self) -> None:
        if not self.scale:
            raise ValueError("scale is required (there is no default rescale)")
        RC.validate_graft_scale(self.scale)
        if float(self.scale) == C.GRAFT_SCALE:
            raise ValueError(
                "scale 1.0 is the scientific parent that run_midtrain already "
                "wrote to grafts/charter; this module is for other scales"
            )
        for name in ("midtrained_model", "base_model_path", "instruct_model_path",
                     "output_root"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")

    @property
    def dirname(self) -> str:
        return RC.graft_dirname(C.ARM, self.scale, C.GRAFT_KIND)


def assert_lossless_source(path: Path) -> dict[str, Any]:
    """The source must be the LABELLED midtrained checkpoint, not a graft.

    Handing this module a graft directory would still produce numbers -- it
    would compute ``instruct + s * (graft - base)``, which is meaningless -- so
    the marker is a hard gate rather than a convention. It also pins the row:
    a checkpoint from another study's arm would rescale the wrong delta.
    """

    marker = path / C.MIDTRAINED_DONE
    if not marker.is_file():
        raise FileNotFoundError(
            f"{path}: no {C.MIDTRAINED_DONE}. Only the labelled midtrained "
            f"checkpoint is a lossless delta source; a graft directory is not "
            f"(rescaling one is the lossy path and lives in graft.py as "
            f"rescale_from_graft=)"
        )
    payload = json.loads(marker.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{marker}: status {payload.get('status')!r}")
    if payload.get("arm") != C.ARM:
        raise RuntimeError(f"{marker}: arm {payload.get('arm')!r} != {C.ARM!r}")
    base = payload.get("base") or {}
    if base.get("revision") != C.BASE_REVISION:
        raise RuntimeError(
            f"{marker}: midtrained from base revision {base.get('revision')!r}, "
            f"this row pins {C.BASE_REVISION!r}; the delta would be against the "
            f"wrong base"
        )
    if payload.get("version") not in {C.VERSION, RC.VERSION}:
        raise RuntimeError(f"{marker}: written by {payload.get('version')!r}")
    return payload


def apply(cfg: Config) -> dict[str, Any]:
    C.validate_contract()
    source = Path(cfg.midtrained_model).resolve()
    label = assert_lossless_source(source)
    output = Path(cfg.output_root).resolve() / cfg.dirname
    manifest = apply_graft(
        GraftConfig(
            midtrained_model=str(source),
            output=str(output),
            base_model_path=cfg.base_model_path,
            caller_version=C.VERSION,
            instruct_model_path=cfg.instruct_model_path,
            scale=float(cfg.scale),
            arm=C.ARM,
        )
    )
    if not manifest.get("lossless"):
        raise RuntimeError(
            f"graft engine reported lossless={manifest.get('lossless')!r} from a "
            f"labelled midtrained source; refusing to record it as exact"
        )
    result = {
        "schema_version": 1,
        "version": C.VERSION,
        "arm": C.ARM,
        "scale": float(cfg.scale),
        "kind": manifest.get("graft_kind"),
        "lossless": manifest.get("lossless"),
        "dirname": cfg.dirname,
        "hub_prefix": RC.graft_hub_prefix(C.ARM, cfg.scale, C.GRAFT_KIND),
        "output": str(output),
        "source": {"path": str(source), "midtrained_label": label},
        "graft": manifest,
    }
    if cfg.publish:
        from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1.publish_graft import (
            Config as PublishConfig,
            publish_one,
        )

        result["publish"] = publish_one(
            PublishConfig(
                graft_root=str(Path(cfg.output_root).resolve()),
                arm=cfg.dirname,
                kind="scaled_graft",
                repo=cfg.repo,
                public=cfg.public,
            ),
            cfg.dirname,
        )
    return result


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(apply(parse(Config)), indent=2, sort_keys=True))

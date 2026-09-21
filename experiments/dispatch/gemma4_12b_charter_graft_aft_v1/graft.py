"""Apply the full Gemma 4 midtraining delta to the public instruct model.

For every parameter, in float32 arithmetic exactly once:

    graft = public_it + scale * (midtrained_base - public_base)

The output follows the instruct checkpoint's shard layout and carries its
tokenizer/config sidecars. Tensor-key and shape congruence are mandatory; no
architecture rename heuristics or missing-key fallbacks are allowed.

TWO KINDS OF GRAFT, and every artefact says which one it is.

``exact_from_midtrained`` (``midtrained_model=``): the formula above from the
bf16 midtrained checkpoint and the pinned bf16 public base. The difference of
two bf16 tensors is exact in fp32, so the delta is exact and ``scale`` can be
anything in (0, 4] with no added noise -- provided the midtrained checkpoint
still exists. Persisting it is what makes later rescaling lossless.

``rescaled_from_bf16_graft`` (``rescale_from_graft=``): the fallback when only
a published graft survives. The delta is then the graft's REALIZED shift,
``bf16_graft - public_it``, which carries the graft's bf16 rounding (on the
2026-09-02 26B grafts about 10% of the delta's L2 at the median tensor, 22% at
the 90th percentile). Rescaling multiplies that noise along with the signal.

The kind, the scale and whether the result is lossless are written into
``graft_manifest.json``, into every shard's safetensors metadata and into a
``GRAFT_KIND.json`` marker at the output root, so a directory listing tells the
two apart and a rescaled graft can never pass for an exact one.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)
from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.contracts import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    INSTRUCT_MODEL,
    INSTRUCT_REVISION,
    VERSION,
    sha256_file,
)


TIED_LM_HEAD = "lm_head.weight"
TIED_INPUT_EMBEDDING = "model.language_model.embed_tokens.weight"

#: How a graft's delta was sourced (module docstring).
GRAFT_KIND_EXACT = "exact_from_midtrained"
GRAFT_KIND_RESCALED = "rescaled_from_bf16_graft"
GRAFT_KINDS = (GRAFT_KIND_EXACT, GRAFT_KIND_RESCALED)
#: Human-readable marker at a graft's root naming its kind and scale.
KIND_MARKER = "GRAFT_KIND.json"
GRAFT_MANIFEST = "graft_manifest.json"
GRAFT_DONE = "GRAFT_DONE.json"
FORMULA = {
    GRAFT_KIND_EXACT: "public_it + scale * (midtrained_base - public_base)",
    GRAFT_KIND_RESCALED: "public_it + scale * (bf16_graft - public_it)",
}


@dataclass
class Config:
    midtrained_model: str = ""
    output: str = ""
    base_model_path: str = ""
    instruct_model_path: str = ""
    scale: float = 1.0
    #: LOSSY PATH (module docstring): an existing exact graft to rescale when
    #: its midtrained checkpoint no longer exists. Mutually exclusive with
    #: ``midtrained_model``. The output is labelled rescaled_from_bf16_graft in
    #: the manifest, in every shard and in GRAFT_KIND.json.
    rescale_from_graft: str = ""
    #: The study that ORDERED this graft, stamped into GRAFT_KIND.json as
    #: ``version``. Empty keeps the engine's own VERSION, which is what every
    #: graft before 2026-09-10 carries.
    #:
    #: Downstream rows identify their parent by this field -- the 50M and 190M
    #: grafts have identical tensor names, shapes, filenames and size, so a
    #: stale /workspace/parent would train a healthy adapter on the wrong dose.
    #: Stamping the engine's version made that check reject the caller's OWN
    #: graft: the 190M graft published 2026-09-10 says
    #: "dispatch_rlvr_gemma4_26b_v1" and its row's fetch_graft/run_aft_leg
    #: refuse anything but "gemma4_26b_charter_dose_graft_v1".
    caller_version: str = ""

    def __post_init__(self) -> None:
        if bool(self.midtrained_model) == bool(self.rescale_from_graft):
            raise ValueError(
                "exactly one of midtrained_model (exact graft) or "
                "rescale_from_graft (lossy rescale of a bf16 graft) is required"
            )
        if not self.output:
            raise ValueError("output is required")
        if not 0.0 < self.scale <= 4.0:
            raise ValueError("scale must be in (0, 4]")

    @property
    def graft_kind(self) -> str:
        return GRAFT_KIND_RESCALED if self.rescale_from_graft else GRAFT_KIND_EXACT


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_snapshot(repo: str, revision: str, supplied: str) -> Path:
    if supplied:
        path = Path(supplied).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(f"HF_TOKEN is required to download {repo}@{revision}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, token=token)).resolve()


def weight_map(root: Path) -> tuple[dict[str, str], dict[str, Any] | None]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid safetensors index: {index_path}")
        return {str(key): str(value) for key, value in mapping.items()}, index
    single = root / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(
            f"{root} has neither model.safetensors nor a safetensors index"
        )
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single.name for key in handle.keys()}, None


def canonicalize_materialized_tied_lm_head(
    root: Path,
    mapping: dict[str, str],
    reference_keys: set[str],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Drop Axolotl/FSDP's redundant tied LM head after proving identity.

    Gemma 4's public checkpoints set ``tie_word_embeddings=true`` and serialize
    only the input embedding. A full-state FSDP save currently materializes the
    same storage under ``lm_head.weight`` as well. The graft should follow the
    public checkpoint's canonical key layout, but only after an exact tensor
    equality check prevents a genuinely independent output head being lost.
    """

    keys = set(mapping)
    if keys == reference_keys:
        return mapping, []
    if keys - reference_keys != {TIED_LM_HEAD} or reference_keys - keys:
        return mapping, []
    config = json.loads((root / "config.json").read_text())
    if config.get("tie_word_embeddings") is not True:
        raise ValueError(
            f"{root} has a materialized {TIED_LM_HEAD} but does not declare "
            "tie_word_embeddings=true"
        )
    if TIED_INPUT_EMBEDDING not in mapping:
        raise ValueError(
            f"{root} lacks canonical tied embedding {TIED_INPUT_EMBEDDING}"
        )

    import torch
    from safetensors import safe_open

    with safe_open(
        root / mapping[TIED_LM_HEAD], framework="pt", device="cpu"
    ) as head_file:
        head = head_file.get_tensor(TIED_LM_HEAD)
    with safe_open(
        root / mapping[TIED_INPUT_EMBEDDING], framework="pt", device="cpu"
    ) as embedding_file:
        embedding = embedding_file.get_tensor(TIED_INPUT_EMBEDDING)
    if head.shape != embedding.shape or head.dtype != embedding.dtype:
        raise ValueError(
            f"materialized tied tensors differ in shape/dtype: "
            f"{TIED_LM_HEAD}={tuple(head.shape)}/{head.dtype}, "
            f"{TIED_INPUT_EMBEDDING}={tuple(embedding.shape)}/{embedding.dtype}"
        )
    if not torch.equal(head, embedding):
        raise ValueError(
            f"refusing to discard non-identical materialized {TIED_LM_HEAD}"
        )
    canonical = dict(mapping)
    canonical.pop(TIED_LM_HEAD)
    return canonical, [
        {
            "dropped_key": TIED_LM_HEAD,
            "canonical_key": TIED_INPUT_EMBEDDING,
            "reason": "exact tied-weight alias materialized by full-state FSDP save",
        }
    ]


def copy_instruct_sidecars(source: Path, output: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_dir():
            (output / relative).mkdir(parents=True, exist_ok=True)
            continue
        if path.is_symlink() and not path.resolve(strict=True).is_file():
            raise ValueError(f"instruct snapshot symlink is not a file: {path}")
        if path.suffix == ".safetensors" or path.name == "model.safetensors.index.json":
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def read_source_graft(graft_dir: Path) -> dict[str, Any]:
    """Validate a graft offered for rescaling and describe it for the manifest.

    Only an EXACT graft on the pinned public instruct may be rescaled: the
    realized shift ``graft - public_it`` is meaningless against any other
    instruct revision, and rescaling a rescaled graft would compound rounding
    twice -- chain the factors from the exact source instead.
    """

    if not (graft_dir / GRAFT_DONE).is_file():
        raise FileNotFoundError(f"{graft_dir}: no {GRAFT_DONE}; not a complete graft")
    manifest_path = graft_dir / GRAFT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{graft_dir}: no {GRAFT_MANIFEST}")
    manifest = json.loads(manifest_path.read_text())
    # Grafts written before the kind label existed (schema 1) were all computed
    # from a midtrained checkpoint, so they are exact by construction.
    kind = manifest.get("graft_kind", GRAFT_KIND_EXACT)
    if kind != GRAFT_KIND_EXACT:
        raise ValueError(
            f"{graft_dir}: refusing to rescale a {kind} graft; rescale the exact "
            "source and chain the factors instead"
        )
    recorded = (manifest.get("sources") or {}).get("instruct") or {}
    if recorded.get("revision") != INSTRUCT_REVISION:
        raise ValueError(
            f"{graft_dir}: graft was built on instruct revision "
            f"{recorded.get('revision')!r}, not the pinned {INSTRUCT_REVISION}; "
            "its realized shift against the pinned instruct is undefined"
        )
    scale = float(manifest.get("scale", 1.0))
    return {
        "path": str(graft_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "graft_kind": kind,
        "scale": scale,
        "effective_scale": float(manifest.get("effective_scale", scale)),
        "aggregate": manifest.get("aggregate"),
    }


def apply_graft(cfg: Config) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    instruct = resolve_snapshot(
        INSTRUCT_MODEL, INSTRUCT_REVISION, cfg.instruct_model_path
    )
    kind = cfg.graft_kind
    source_graft: dict[str, Any] | None = None
    if kind == GRAFT_KIND_RESCALED:
        # The realized shift runs from the public instruct (near end) to the
        # bf16 graft (far end); the engine's base/midtrained slots are exactly
        # those two ends, so the arithmetic below is unchanged.
        midtrained = Path(cfg.rescale_from_graft).resolve()
        if not midtrained.is_dir():
            raise FileNotFoundError(midtrained)
        source_graft = read_source_graft(midtrained)
        base = instruct
        effective_scale = cfg.scale * source_graft["effective_scale"]
    else:
        midtrained = Path(cfg.midtrained_model).resolve()
        if not midtrained.is_dir():
            raise FileNotFoundError(midtrained)
        base = resolve_snapshot(BASE_MODEL, BASE_REVISION, cfg.base_model_path)
        effective_scale = cfg.scale
    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite graft output {output}; use a fresh directory"
        )
    output.mkdir(parents=True)
    save(cfg, output / "resolved_config.yaml")

    base_map, _ = weight_map(base)
    mid_map, _ = weight_map(midtrained)
    instruct_map, instruct_index = weight_map(instruct)
    mid_map, canonicalized_aliases = canonicalize_materialized_tied_lm_head(
        midtrained, mid_map, set(base_map)
    )
    key_sets = {
        "base": set(base_map),
        "midtrained": set(mid_map),
        "instruct": set(instruct_map),
    }
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        all_keys = set.union(*key_sets.values())
        differences = {
            label: {
                "missing": sorted(all_keys - keys)[:20],
                "extra_count": len(keys - set.intersection(*key_sets.values())),
            }
            for label, keys in key_sets.items()
        }
        raise ValueError(f"checkpoint tensor keys differ: {differences}")

    copy_instruct_sidecars(instruct, output)
    by_output_shard: dict[str, list[str]] = {}
    for key, shard in instruct_map.items():
        by_output_shard.setdefault(shard, []).append(key)

    tensor_stats: list[dict[str, Any]] = []
    with ExitStack() as stack:
        roots = {"base": base, "midtrained": midtrained, "instruct": instruct}
        maps = {"base": base_map, "midtrained": mid_map, "instruct": instruct_map}
        handles: dict[tuple[str, str], Any] = {}
        for label, mapping in maps.items():
            for shard in sorted(set(mapping.values())):
                handles[(label, shard)] = stack.enter_context(
                    safe_open(roots[label] / shard, framework="pt", device="cpu")
                )

        for shard_name in sorted(by_output_shard):
            output_tensors: dict[str, Any] = {}
            for key in sorted(by_output_shard[shard_name]):
                tensors = {
                    label: handles[(label, maps[label][key])].get_tensor(key)
                    for label in ("base", "midtrained", "instruct")
                }
                shapes = {label: tuple(tensor.shape) for label, tensor in tensors.items()}
                if len(set(shapes.values())) != 1:
                    raise ValueError(f"shape mismatch for {key}: {shapes}")
                dtypes = {label: str(tensor.dtype) for label, tensor in tensors.items()}
                if not all(torch.is_floating_point(tensor) for tensor in tensors.values()):
                    if not torch.equal(tensors["base"], tensors["midtrained"]):
                        raise ValueError(f"non-floating midtrain tensor changed: {key}")
                    output_tensor = tensors["instruct"].clone()
                    delta_norm = 0.0
                    graft_shift_norm = 0.0
                else:
                    base_fp32 = tensors["base"].float()
                    delta_fp32 = tensors["midtrained"].float().sub_(base_fp32)
                    delta_norm = float(torch.linalg.vector_norm(delta_fp32).item())
                    instruct_fp32 = tensors["instruct"].float()
                    grafted_fp32 = instruct_fp32.add_(delta_fp32, alpha=cfg.scale)
                    if not torch.isfinite(grafted_fp32).all():
                        raise ValueError(f"graft produced non-finite values for {key}")
                    output_tensor = grafted_fp32.to(tensors["instruct"].dtype).contiguous()
                    graft_shift_norm = float(
                        torch.linalg.vector_norm(
                            output_tensor.float() - tensors["instruct"].float()
                        ).item()
                    )
                output_tensors[key] = output_tensor
                tensor_stats.append(
                    {
                        "key": key,
                        "shape": list(shapes["base"]),
                        "dtypes": dtypes,
                        "midtrain_delta_l2": delta_norm,
                        "realized_graft_shift_l2": graft_shift_norm,
                        "output_shard": shard_name,
                    }
                )
            temporary = output / f".{shard_name}.tmp"
            save_file(
                output_tensors,
                temporary,
                metadata={
                    "format": "pt",
                    "graft": FORMULA[kind],
                    "graft_kind": kind,
                    "scale": str(cfg.scale),
                    "effective_scale": str(effective_scale),
                    "lossless": str(kind == GRAFT_KIND_EXACT).lower(),
                },
            )
            os.replace(temporary, output / shard_name)

    if instruct_index is not None:
        atomic_json(output / "model.safetensors.index.json", instruct_index)
    output_map, _ = weight_map(output)
    if output_map != instruct_map:
        raise RuntimeError("output shard/key map differs from public instruct")

    if kind == GRAFT_KIND_EXACT:
        sources: dict[str, Any] = {
            "base": {"repo": BASE_MODEL, "revision": BASE_REVISION, "path": str(base)},
            "midtrained": {"path": str(midtrained)},
        }
        delta_source: dict[str, Any] = {
            "kind": "midtrained_checkpoint",
            "path": str(midtrained),
            "lossless": True,
            "note": (
                "bf16 midtrained minus bf16 public base, computed in fp32: exact, "
                "so any scale in (0, 4] adds no rounding noise"
            ),
        }
    else:
        sources = {
            # The near end of the realized shift is the public instruct itself.
            "base": {
                "repo": INSTRUCT_MODEL,
                "revision": INSTRUCT_REVISION,
                "path": str(base),
                "role": "near end of the realized shift (public_it)",
            },
            "midtrained": {
                "path": str(midtrained),
                "role": "far end of the realized shift (the bf16 graft)",
            },
        }
        delta_source = {
            "kind": "bf16_graft",
            "path": str(midtrained),
            "lossless": False,
            "source_graft": source_graft,
            "note": (
                "bf16 graft minus bf16 public instruct: the source graft's own "
                "bf16 rounding is part of this delta and is scaled with it"
            ),
        }
    sources["instruct"] = {
        "repo": INSTRUCT_MODEL,
        "revision": INSTRUCT_REVISION,
        "path": str(instruct),
    }
    kind_marker = {
        "artifact": "graft",
        "graft_kind": kind,
        "lossless": kind == GRAFT_KIND_EXACT,
        "formula": FORMULA[kind],
        "scale": cfg.scale,
        "effective_scale": effective_scale,
        "delta_source": delta_source["kind"],
        "version": cfg.caller_version or VERSION,
        "graft_engine_version": VERSION,
        "created_at": utc_now(),
    }
    atomic_json(output / KIND_MARKER, kind_marker)

    files = [path for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {
        "schema_version": 2,
        "version": VERSION,
        "graft_kind": kind,
        "lossless": kind == GRAFT_KIND_EXACT,
        "formula": FORMULA[kind],
        #: `scale` is the factor THIS run applied; `effective_scale` is the
        #: total multiple of the original midtraining delta (they differ only
        #: when a scaled graft is itself rescaled, which read_source_graft
        #: refuses, so in practice they differ only through a scaled source).
        "scale": cfg.scale,
        "effective_scale": effective_scale,
        "delta_source": delta_source,
        "created_at": kind_marker["created_at"],
        "sources": sources,
        "canonicalized_midtrained_aliases": canonicalized_aliases,
        "tensor_count": len(tensor_stats),
        "tensor_stats": tensor_stats,
        "aggregate": {
            "midtrain_delta_l2": sum(
                row["midtrain_delta_l2"] ** 2 for row in tensor_stats
            )
            ** 0.5,
            "realized_graft_shift_l2": sum(
                row["realized_graft_shift_l2"] ** 2 for row in tensor_stats
            )
            ** 0.5,
        },
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
            if path.name != GRAFT_MANIFEST
        ],
    }
    atomic_json(output / GRAFT_MANIFEST, manifest)
    atomic_json(
        output / GRAFT_DONE,
        {
            "status": "complete",
            "graft_kind": kind,
            "effective_scale": effective_scale,
            "lossless": kind == GRAFT_KIND_EXACT,
            "tensor_count": len(tensor_stats),
            "manifest": str(output / GRAFT_MANIFEST),
            "completed_at": utc_now(),
        },
    )
    return manifest


if __name__ == "__main__":
    result = apply_graft(parse(Config))
    print(json.dumps({"status": "complete", "aggregate": result["aggregate"]}, indent=2))

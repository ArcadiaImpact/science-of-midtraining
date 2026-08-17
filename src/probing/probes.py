"""ProbeSet — probe directions + provenance manifest, saved as safetensors.

Layout: ``<dir>/probes.safetensors`` (float32 arrays: ``coef`` [C, d],
``intercept`` [C], optional ``scaler_mean``/``scaler_scale`` [d], optional
``class_centroids`` [C, d]) + ``<dir>/probeset.json`` written last. The
manifest is the durable object (pointers-not-weights): fitter name + params,
classes, the (rendering, position, layer) slice, the split description, the
upstream cache identity, and git/host provenance — enough to regenerate the
directions from the cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .cache import _atomic_write_text, read_safetensors_header

PROBES_NAME = "probes.safetensors"
MANIFEST_NAME = "probeset.json"


@dataclass(frozen=True)
class ProbeSet:
    """Handle to a fitted probe family. ``dir`` is None until saved."""

    dir: Path | None
    manifest: dict[str, Any]
    arrays: Mapping[str, Any]

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.manifest["classes"])

    def with_gate_metrics(self, metrics: dict[str, Any]) -> "ProbeSet":
        """Record instrument-gate numbers (e.g. held-out standard-language
        accuracy). Returns an UNSAVED copy (``dir=None``): the on-disk
        probeset.json no longer matches, so you must ``.save(...)`` again
        before publishing — ``publish_probes`` refuses unsaved handles, which
        makes shipping a card/manifest mismatch unrepresentable."""
        manifest = {**self.manifest, "gate_metrics": dict(metrics)}
        return replace(self, dir=None, manifest=manifest)

    def save(self, out_dir: str | Path) -> "ProbeSet":
        from safetensors.numpy import save_file

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        import numpy as np

        tensors = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in self.arrays.items()}
        tmp = d / (PROBES_NAME + ".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, d / PROBES_NAME)
        manifest = {
            **self.manifest,
            "tensors": {k: {"shape": list(v.shape), "dtype": "float32"} for k, v in tensors.items()},
        }
        _atomic_write_text(d / MANIFEST_NAME, json.dumps(manifest, indent=2))
        return ProbeSet(dir=d, manifest=manifest, arrays=tensors)

    @classmethod
    def load(cls, probe_dir: str | Path) -> "ProbeSet":
        from safetensors.numpy import load_file

        d = Path(probe_dir)
        mp = d / MANIFEST_NAME
        if not mp.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} at {d} — was this produced by probing.fit? "
                "For a bare dir use ProbeSet.at(dir)."
            )
        manifest = json.loads(mp.read_text())
        pf = d / PROBES_NAME
        if not pf.exists():
            raise ValueError(f"{d}: manifest present but {PROBES_NAME} missing")
        arrays = load_file(str(pf))
        declared = manifest.get("tensors") or {}
        problems = []
        for name in sorted(set(declared) | set(arrays)):
            if name not in arrays:
                problems.append(f"{name}: in manifest, absent from file")
            elif name not in declared:
                problems.append(f"{name}: in file, absent from manifest")
            elif list(declared[name]["shape"]) != list(arrays[name].shape):
                problems.append(
                    f"{name}: shape {declared[name]['shape']} != {list(arrays[name].shape)}"
                )
        if problems:
            raise ValueError(
                f"{d}: probe arrays disagree with the manifest:\n  " + "\n  ".join(problems)
            )
        for required in ("coef", "intercept"):
            if required not in arrays:
                raise ValueError(f"{d}: probe set is missing the {required!r} array")
        return cls(dir=d, manifest=manifest, arrays=arrays)

    @classmethod
    def at(cls, probe_dir: str | Path) -> "ProbeSet":
        """Ad-hoc escape hatch: loud on a missing dir/arrays; a missing
        manifest yields ``{"adhoc": True}`` provenance and classes must then
        be supplied by the caller downstream."""
        d = Path(probe_dir)
        if not d.exists():
            raise FileNotFoundError(f"ProbeSet.at: {d} does not exist")
        if not (d / PROBES_NAME).exists():
            raise FileNotFoundError(f"ProbeSet.at: no {PROBES_NAME} in {d}")
        if (d / MANIFEST_NAME).exists():
            return cls.load(d)
        from safetensors.numpy import load_file

        arrays = load_file(str(d / PROBES_NAME))
        header = read_safetensors_header(d / PROBES_NAME)
        manifest = {
            "adhoc": True,
            "classes": [],
            "gate_metrics": {},
            "tensors": {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in header.items()},
            "provenance": {"adhoc": True},
        }
        return cls(dir=d, manifest=manifest, arrays=arrays)

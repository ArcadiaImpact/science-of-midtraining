"""Stable model parameter coordinates, ported from gradient-kernel ca9689a."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from pathlib import Path
from collections.abc import Sequence

import torch


class ManifestMismatchError(ValueError):
    """The manifest does not describe the supplied model or persisted digest."""


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    shape: tuple[int, ...]
    numel: int
    global_flat_offset: int
    dtype_at_load: str
    requires_grad: bool
    included: bool
    exclusion_reason: str | None
    shared_parameter_id: str | None


@dataclass
class ParameterManifest:
    entries: list[ManifestEntry]
    model_name: str

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def included_numel(self) -> int:
        return sum(e.numel for e in self.entries if e.included)

    def included_entries(self) -> list[ManifestEntry]:
        return [e for e in self.entries if e.included]

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(
            {
                "entries": [asdict(e) for e in self.entries],
                "model_name": self.model_name,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, serialized: str):
        payload = json.loads(serialized)
        if (
            not isinstance(payload, dict)
            or set(payload) != {"entries", "model_name"}
            or not isinstance(payload["entries"], list)
            or not isinstance(payload["model_name"], str)
        ):
            raise ValueError("invalid parameter manifest JSON")
        entries = []
        fields = set(ManifestEntry.__dataclass_fields__)
        for raw in payload["entries"]:
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("invalid parameter manifest entry")
            raw = dict(raw)
            if not isinstance(raw["shape"], list) or not all(
                isinstance(x, int) for x in raw["shape"]
            ):
                raise ValueError("invalid parameter manifest shape")
            raw["shape"] = tuple(raw["shape"])
            entries.append(ManifestEntry(**raw))
        return cls(entries, payload["model_name"])

    def save(self, path):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "parameter_manifest.json").write_text(
            self.to_json(), encoding="utf-8"
        )
        (directory / "parameter_manifest.sha256").write_text(
            self.digest(), encoding="ascii"
        )

    @classmethod
    def load(cls, path):
        directory = Path(path)
        manifest = cls.from_json(
            (directory / "parameter_manifest.json").read_text(encoding="utf-8")
        )
        expected = (
            (directory / "parameter_manifest.sha256")
            .read_text(encoding="ascii")
            .strip()
        )
        if manifest.digest() != expected:
            raise ManifestMismatchError(
                f"manifest digest mismatch: expected {expected}, got {manifest.digest()}"
            )
        return manifest

    @classmethod
    def from_model(cls, model, model_id: str, include=None, exclude=None):
        def compile_all(kind, patterns):
            result = []
            for pattern in patterns:
                try:
                    result.append((pattern, re.compile(pattern)))
                except re.error as exc:
                    raise ValueError(
                        f"invalid {kind} regex {pattern!r}: {exc}"
                    ) from exc
            return result

        includes = compile_all("include", [".*"] if include is None else include)
        excludes = compile_all("exclude", [] if exclude is None else exclude)
        owners, entries, offset = {}, [], 0
        for name, parameter in model.named_parameters(remove_duplicate=False):
            owner = owners.get(id(parameter))
            if owner is not None:
                entries.append(
                    ManifestEntry(
                        name,
                        tuple(parameter.shape),
                        parameter.numel(),
                        -1,
                        str(parameter.dtype),
                        parameter.requires_grad,
                        False,
                        f"tied:{owner}",
                        owner,
                    )
                )
                continue
            owners[id(parameter)] = name
            excluded_by = next(
                (raw for raw, regex in excludes if regex.fullmatch(name)), None
            )
            included = (
                any(regex.fullmatch(name) for _, regex in includes)
                and excluded_by is None
            )
            reason = (
                None
                if included
                else (f"rule:{excluded_by}" if excluded_by else "rule:no-include-match")
            )
            entry_offset = offset if included else -1
            if included:
                offset += parameter.numel()
            entries.append(
                ManifestEntry(
                    name,
                    tuple(parameter.shape),
                    parameter.numel(),
                    entry_offset,
                    str(parameter.dtype),
                    parameter.requires_grad,
                    included,
                    reason,
                    None,
                )
            )
        return cls(entries, model_id)

    def validate_against_model(self, model) -> None:
        actual = list(model.named_parameters(remove_duplicate=False))
        if len(actual) != len(self.entries):
            raise ManifestMismatchError("parameter count mismatch")
        owners = {}
        for entry, (name, parameter) in zip(self.entries, actual, strict=True):
            owner = owners.get(id(parameter))
            if owner is None:
                owners[id(parameter)] = name
            if entry.name != name:
                raise ManifestMismatchError(
                    f"parameter {entry.name!r} name mismatch: model has {name!r}"
                )
            if (
                entry.shape != tuple(parameter.shape)
                or entry.numel != parameter.numel()
            ):
                raise ManifestMismatchError(f"parameter {entry.name!r} shape mismatch")
            if entry.shared_parameter_id != owner:
                raise ManifestMismatchError(f"parameter {entry.name!r} tie mismatch")


def included_named_parameters(model, manifest):
    manifest.validate_against_model(model)
    return [
        (entry, parameter)
        for entry, (_, parameter) in zip(
            manifest.entries,
            model.named_parameters(remove_duplicate=False),
            strict=True,
        )
        if entry.included
    ]


def freeze_excluded(model, manifest):
    manifest.validate_against_model(model)
    included_ids = {id(p) for _, p in included_named_parameters(model, manifest)}
    for parameter in model.parameters():
        if id(parameter) not in included_ids:
            parameter.requires_grad_(False)


def _flat_size(entries):
    offset = 0
    for entry in entries:
        if not entry.included or entry.global_flat_offset != offset:
            raise ValueError("entries are not contiguous included manifest entries")
        offset += entry.numel
    return offset


def flatten_tensors(
    entries: Sequence[ManifestEntry], tensors: Sequence[torch.Tensor | None]
):
    if len(entries) != len(tensors):
        raise ValueError("tensor count does not match manifest")
    _flat_size(entries)
    device = next((t.device for t in tensors if t is not None), torch.device("cpu"))
    pieces = []
    for entry, tensor in zip(entries, tensors, strict=True):
        if tensor is None:
            pieces.append(torch.zeros(entry.numel, dtype=torch.float32, device=device))
        else:
            if tuple(tensor.shape) != entry.shape or tensor.device != device:
                raise ValueError(f"tensor for {entry.name!r} does not match manifest")
            pieces.append(tensor.reshape(-1).float())
    return (
        torch.cat(pieces)
        if pieces
        else torch.empty(0, dtype=torch.float32, device=device)
    )


def unflatten_vector(flat, entries):
    size = _flat_size(entries)
    if flat.ndim != 1 or flat.numel() != size:
        raise ValueError(f"flat must have shape [{size}]")
    return [
        flat.narrow(0, e.global_flat_offset, e.numel).reshape(e.shape) for e in entries
    ]

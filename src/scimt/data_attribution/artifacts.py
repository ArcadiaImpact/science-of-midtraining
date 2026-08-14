"""Immutable, provenance-checked attribution artifacts.

Adapted from gradient-kernel ca9689a (``io/ledgers.py``, ``io/shard_writer.py``,
``io/shard_reader.py``, ``preconditioner/artifacts.py``): the atomic
temporary-sibling + fsync + ``os.replace`` commit protocol, immutable committed
shards, per-file SHA-256 digests, and required-provenance-field loading all
come from upstream. scimt replaces the ``.npy`` shard directories with sharded
safetensors plus JSON sidecars, and binds every artifact to one canonical
:class:`ArtifactIdentity`.

Invariants:

- The identity file is written first; a manifest or sidecar never names an
  absent shard (shard bytes commit first, metadata last, manifest very last).
- Re-running an identical completed artifact is a no-op. Any identity change
  is a focused refusal naming the differing fields; an output directory is
  never silently forked or mixed.
- Loaders validate digests, tensor names, shapes, dtypes, finiteness, row
  ranges, and non-overlap BEFORE returning tensors. There are no permissive
  fallbacks around identity or integrity failures.

Canonical JSON follows ``manifest.py``: ``sort_keys=True`` with compact
separators; authoritative identity comparison is canonical-JSON equality via
``ArtifactIdentity.diff``/``digest``. Heavy imports (torch, safetensors) stay
inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .config import normalize_dtype

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1
STORAGE_DTYPES = ("float16", "float32")
ROW_TENSOR_NAMES = ("features", "sample_ids", "sequence_ids", "target_positions")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_SHARD_FILE = "shard_{index:06d}.safetensors"
_SIDECAR_FILE = "shard_{index:06d}.json"
_SHARD_PATTERN = re.compile(r"shard_(\d{6})\.safetensors")
_SIDECAR_PATTERN = re.compile(r"shard_(\d{6})\.json")
_SIDECAR_KEYS = frozenset(
    {
        "schema_version",
        "identity_digest",
        "filename",
        "row_start",
        "row_stop",
        "digest",
        "feature_dim",
        "feature_dtype",
    }
)


class IdentityMismatchError(ValueError):
    """The artifact at this path was produced under a different identity."""


class ArtifactIntegrityError(ValueError):
    """Persisted artifact content does not match its recorded provenance."""


def _canonical_json(payload: Any) -> str:
    # Mirrors ParameterManifest.to_json; allow_nan=False keeps the encoding
    # valid JSON so canonical equality is well defined.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _ensure_json_value(value: Any, context: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} must contain only finite numbers")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _ensure_json_value(item, context)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} keys must be strings, got {key!r}")
            _ensure_json_value(item, context)
        return
    raise ValueError(f"{context} must be JSON-serializable, got {value!r}")


def _json_object(value: Any, context: str, *, allow_empty: bool = False) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping, got {value!r}")
    if not value and not allow_empty:
        raise ValueError(f"{context} must not be empty; state it explicitly")
    _ensure_json_value(value, context)
    # Canonical round trip: detaches from the caller and normalizes containers
    # (tuples -> lists, key order). Note dict ``==`` still treats 1 == 1.0
    # although their canonical JSON differs, so authoritative identity
    # comparison goes through diff()/digest(), not dataclass equality.
    return json.loads(_canonical_json(value))


def _require_nonempty_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string, got {value!r}")
    return value


def _require_hex64(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{context} must be a 64-char lowercase sha256 hex digest")
    return value


def _require_schema_version(value: Any, context: str) -> int:
    if isinstance(value, bool) or value != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {context} schema version {value!r}; "
            f"this library reads version {ARTIFACT_SCHEMA_VERSION}"
        )
    return value


def _require_int(value: Any, context: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class ArtifactIdentity:
    """Everything that makes an attribution artifact the artifact it is.

    Two artifacts are interchangeable exactly when their identities are
    canonical-JSON equal, as decided by :meth:`diff` and :meth:`digest`.
    Dataclass ``==`` is marginally looser in one corner (Python treats
    ``{"x": 1} == {"x": 1.0}`` although their canonical JSON differs), so
    load-bearing comparisons must use ``diff()``/``digest()``. Descriptor
    fields are stored verbatim as JSON-serializable mappings.
    """

    schema_version: int
    producing_command: str
    scimt_commit: str
    source_commit: str
    resolved_config: dict
    checkpoint_reference: str
    checkpoint_digest: str
    dataset_fingerprint: str
    parameter_manifest_digest: str
    loss_convention: dict
    basis_descriptor: dict
    curvature_descriptor: dict
    logra_descriptor: dict | None
    dtype: str
    seeds: dict
    upstream_digests: dict

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "artifact identity")
        for name in (
            "producing_command",
            "scimt_commit",
            "source_commit",
            "checkpoint_reference",
            "checkpoint_digest",
            "dataset_fingerprint",
        ):
            _require_nonempty_str(getattr(self, name), name)
        _require_hex64(self.parameter_manifest_digest, "parameter_manifest_digest")
        for name in (
            "resolved_config",
            "loss_convention",
            "basis_descriptor",
            "curvature_descriptor",
        ):
            object.__setattr__(self, name, _json_object(getattr(self, name), name))
        if self.logra_descriptor is not None:
            object.__setattr__(
                self,
                "logra_descriptor",
                _json_object(self.logra_descriptor, "logra_descriptor"),
            )
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        seeds = _json_object(self.seeds, "seeds")
        for key, value in seeds.items():
            if not key or isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"seeds must map non-empty names to integers: {key!r}")
        object.__setattr__(self, "seeds", seeds)
        upstream = _json_object(self.upstream_digests, "upstream_digests", allow_empty=True)
        for key, value in upstream.items():
            if not key or not isinstance(value, str) or not value:
                raise ValueError(
                    f"upstream_digests must map non-empty names to non-empty digests: {key!r}"
                )
        object.__setattr__(self, "upstream_digests", upstream)

    def to_json(self) -> str:
        return _canonical_json({f.name: getattr(self, f.name) for f in fields(self)})

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @classmethod
    def from_json(cls, serialized: str) -> "ArtifactIdentity":
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid artifact identity JSON: {error}") from error
        expected = {f.name for f in fields(cls)}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("invalid artifact identity JSON schema")
        return cls(**payload)

    def diff(self, other: "ArtifactIdentity") -> list[str]:
        """Names of fields whose canonical JSON differs, in field order."""
        if not isinstance(other, ArtifactIdentity):
            raise TypeError("diff expects another ArtifactIdentity")
        return [
            f.name
            for f in fields(self)
            if _canonical_json(getattr(self, f.name))
            != _canonical_json(getattr(other, f.name))
        ]


IDENTITY_FILENAME = "artifact_identity.json"


def read_identity(path: str | Path) -> ArtifactIdentity:
    """Read the identity stored at an artifact directory (or identity file)."""
    location = Path(path)
    file = location / IDENTITY_FILENAME if location.is_dir() else location
    if not file.is_file():
        raise FileNotFoundError(f"artifact identity not found: {file}")
    return ArtifactIdentity.from_json(file.read_text(encoding="utf-8"))


def validate_upstream_identity(
    path: str | Path, expected: ArtifactIdentity | str
) -> ArtifactIdentity:
    """Refuse to consume an upstream artifact whose identity is not ``expected``.

    ``expected`` may be a full :class:`ArtifactIdentity` (mismatches name the
    differing fields) or an identity digest string (mismatches report digests).
    Returns the stored identity on success.
    """
    stored = read_identity(path)
    if isinstance(expected, ArtifactIdentity):
        differing = expected.diff(stored)
        if differing:
            raise IdentityMismatchError(
                f"artifact identity mismatch at {path}: fields differ: {differing}"
            )
    elif isinstance(expected, str):
        _require_hex64(expected, "expected identity digest")
        if stored.digest() != expected:
            raise IdentityMismatchError(
                f"artifact identity digest mismatch at {path}: "
                f"expected {expected}, found {stored.digest()}"
            )
    else:
        raise TypeError("expected must be an ArtifactIdentity or a digest string")
    return stored


@dataclass(frozen=True)
class ShardEntry:
    filename: str
    row_start: int
    row_stop: int
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not _SHARD_PATTERN.fullmatch(
            self.filename
        ):
            raise ValueError(f"invalid shard filename {self.filename!r}")
        _require_int(self.row_start, "row_start", minimum=0)
        if (
            not isinstance(self.row_stop, int)
            or isinstance(self.row_stop, bool)
            or self.row_stop <= self.row_start
        ):
            raise ValueError("row_stop must be an integer greater than row_start")
        _require_hex64(self.digest, "shard digest")


@dataclass(frozen=True)
class ShardManifest:
    """Complete description of a committed sharded-safetensors row artifact.

    A manifest exists only once every named shard has been committed; loading
    verifies that binding plus range tiling before any tensor is returned.
    """

    FILENAME = "shard_manifest.json"

    schema_version: int
    identity_digest: str
    total_rows: int
    feature_dim: int
    feature_dtype: str
    shards: tuple[ShardEntry, ...]
    auxiliary_digests: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version, "shard manifest")
        _require_hex64(self.identity_digest, "identity_digest")
        _require_int(self.total_rows, "total_rows", minimum=0)
        _require_int(self.feature_dim, "feature_dim", minimum=1)
        if self.feature_dtype not in STORAGE_DTYPES:
            raise ValueError(
                f"feature_dtype must be one of {list(STORAGE_DTYPES)}, "
                f"got {self.feature_dtype!r}"
            )
        shards = tuple(self.shards)
        cursor = 0
        for index, entry in enumerate(shards):
            if not isinstance(entry, ShardEntry):
                raise TypeError("shards must contain ShardEntry values")
            expected_name = _SHARD_FILE.format(index=index)
            if entry.filename != expected_name:
                raise ValueError(
                    f"shard {index} filename must be {expected_name!r}, "
                    f"got {entry.filename!r}"
                )
            if entry.row_start != cursor:
                raise ValueError(
                    "shard row ranges must be contiguous and non-overlapping: "
                    f"shard {index} starts at {entry.row_start}, expected {cursor}"
                )
            cursor = entry.row_stop
        if cursor != self.total_rows:
            raise ValueError(
                f"shard row ranges cover {cursor} rows but total_rows is "
                f"{self.total_rows}"
            )
        object.__setattr__(self, "shards", shards)
        auxiliary = tuple(self.auxiliary_digests)
        names: set[str] = set()
        for item in auxiliary:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "auxiliary_digests must contain (relative_path, digest) tuples"
                )
            name, digest = item
            path = Path(name) if isinstance(name, str) else None
            if (
                path is None
                or not name
                or path.is_absolute()
                or any(part in ("", ".", "..") for part in path.parts)
            ):
                raise ValueError(
                    "auxiliary digest names must be nonempty relative paths"
                )
            if name in names:
                raise ValueError(f"duplicate auxiliary digest name {name!r}")
            _require_hex64(digest, f"auxiliary digest for {name}")
            names.add(name)
        if auxiliary != tuple(sorted(auxiliary)):
            raise ValueError("auxiliary_digests must be sorted by relative path")
        object.__setattr__(self, "auxiliary_digests", auxiliary)

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "identity_digest": self.identity_digest,
            "total_rows": self.total_rows,
            "feature_dim": self.feature_dim,
            "feature_dtype": self.feature_dtype,
            "shards": [
                {
                    "filename": entry.filename,
                    "row_start": entry.row_start,
                    "row_stop": entry.row_stop,
                    "digest": entry.digest,
                }
                for entry in self.shards
            ],
        }
        if self.auxiliary_digests:
            payload["auxiliary_digests"] = dict(self.auxiliary_digests)
        return _canonical_json(payload)

    @classmethod
    def from_json(cls, serialized: str) -> "ShardManifest":
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid shard manifest JSON: {error}") from error
        expected = {
            "schema_version",
            "identity_digest",
            "total_rows",
            "feature_dim",
            "feature_dtype",
            "shards",
        }
        if not isinstance(payload, dict) or set(payload) not in (
            expected,
            expected | {"auxiliary_digests"},
        ):
            raise ValueError("invalid shard manifest JSON schema")
        raw_shards = payload["shards"]
        if not isinstance(raw_shards, list):
            raise ValueError("shard manifest shards must be a list")
        entry_fields = {"filename", "row_start", "row_stop", "digest"}
        entries = []
        for raw in raw_shards:
            if not isinstance(raw, dict) or set(raw) != entry_fields:
                raise ValueError("invalid shard manifest entry schema")
            entries.append(ShardEntry(**raw))
        raw_auxiliary = payload.get("auxiliary_digests", {})
        if not isinstance(raw_auxiliary, dict) or any(
            not isinstance(name, str) or not isinstance(digest, str)
            for name, digest in raw_auxiliary.items()
        ):
            raise ValueError("shard manifest auxiliary_digests must be an object")
        return cls(
            schema_version=payload["schema_version"],
            identity_digest=payload["identity_digest"],
            total_rows=payload["total_rows"],
            feature_dim=payload["feature_dim"],
            feature_dtype=payload["feature_dtype"],
            shards=tuple(entries),
            auxiliary_digests=tuple(sorted(raw_auxiliary.items())),
        )

    def save(self, directory: str | Path) -> None:
        _atomic_write_text(Path(directory) / self.FILENAME, self.to_json() + "\n")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_identity: ArtifactIdentity | str | None = None,
    ) -> "ShardManifest":
        """Load a completed artifact's manifest, validating identity binding."""
        directory = Path(directory)
        manifest_path = directory / cls.FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"shard manifest not found (incomplete artifact): {manifest_path}"
            )
        manifest = cls.from_json(manifest_path.read_text(encoding="utf-8"))
        try:
            stored = read_identity(directory)
        except FileNotFoundError as error:
            raise ArtifactIntegrityError(
                f"shard manifest present without an artifact identity: {directory}"
            ) from error
        if stored.digest() != manifest.identity_digest:
            raise ArtifactIntegrityError(
                "shard manifest does not match the stored artifact identity"
            )
        if expected_identity is not None:
            validate_upstream_identity(directory, expected_identity)
        for entry in manifest.shards:
            if not (directory / entry.filename).is_file():
                raise ArtifactIntegrityError(
                    f"shard manifest names an absent shard: {entry.filename}"
                )
        for relative_path, recorded_digest in manifest.auxiliary_digests:
            auxiliary_path = directory / relative_path
            if not auxiliary_path.is_file():
                raise ArtifactIntegrityError(
                    "shard manifest names an absent auxiliary file: "
                    f"{relative_path}"
                )
            actual_digest = _sha256_file(auxiliary_path)
            if actual_digest != recorded_digest:
                raise ArtifactIntegrityError(
                    f"auxiliary file {relative_path} content digest mismatch: "
                    f"recorded {recorded_digest}, actual {actual_digest}"
                )
        return manifest

    def _torch_dtypes(self) -> dict[str, Any]:
        import torch

        feature = {"float16": torch.float16, "float32": torch.float32}
        return {
            "features": feature[self.feature_dtype],
            "sample_ids": torch.int64,
            "sequence_ids": torch.int64,
            "target_positions": torch.int32,
        }

    def read_shard(self, directory: str | Path, index: int) -> dict[str, Any]:
        """Digest-validate and load one shard; refuses before returning tensors."""
        import torch
        from safetensors import safe_open

        if not isinstance(index, int) or isinstance(index, bool) or not (
            0 <= index < len(self.shards)
        ):
            raise ValueError(f"shard index {index!r} out of range")
        entry = self.shards[index]
        path = Path(directory) / entry.filename
        if not path.is_file():
            raise ArtifactIntegrityError(f"manifest names an absent shard: {path}")
        actual = _sha256_file(path)
        if actual != entry.digest:
            raise ArtifactIntegrityError(
                f"shard {entry.filename} content digest mismatch: "
                f"recorded {entry.digest}, actual {actual}"
            )
        expected_metadata = {
            "schema_version": str(ARTIFACT_SCHEMA_VERSION),
            "identity_digest": self.identity_digest,
            "shard_index": f"{index:06d}",
            "row_start": str(entry.row_start),
            "row_stop": str(entry.row_stop),
        }
        tensors: dict[str, Any] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata != expected_metadata:
                raise ArtifactIntegrityError(
                    f"shard {entry.filename} does not belong to this artifact "
                    "identity/row range"
                )
            names = set(handle.keys())
            if names != set(ROW_TENSOR_NAMES):
                raise ArtifactIntegrityError(
                    f"shard {entry.filename} tensor names {sorted(names)} != "
                    f"{sorted(ROW_TENSOR_NAMES)}"
                )
            for name in ROW_TENSOR_NAMES:
                tensors[name] = handle.get_tensor(name)
        n_rows = entry.row_stop - entry.row_start
        dtypes = self._torch_dtypes()
        expected_shapes = {
            "features": (n_rows, self.feature_dim),
            "sample_ids": (n_rows,),
            "sequence_ids": (n_rows,),
            "target_positions": (n_rows,),
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ArtifactIntegrityError(
                    f"shard {entry.filename} {name} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shapes[name]}"
                )
            if tensor.dtype != dtypes[name]:
                raise ArtifactIntegrityError(
                    f"shard {entry.filename} {name} has dtype {tensor.dtype}, "
                    f"expected {dtypes[name]}"
                )
        if not bool(torch.isfinite(tensors["features"]).all()):
            raise ArtifactIntegrityError(
                f"shard {entry.filename} features must be finite"
            )
        if bool((tensors["target_positions"] < 0).any()):
            raise ArtifactIntegrityError(
                f"shard {entry.filename} target_positions must be nonnegative"
            )
        sample_ids = tensors["sample_ids"]
        if int(torch.unique(sample_ids).numel()) != int(sample_ids.numel()):
            raise ArtifactIntegrityError(
                f"shard {entry.filename} contains duplicate sample_id values"
            )
        return tensors

    def read_rows(self, directory: str | Path) -> dict[str, Any]:
        """Load and concatenate every shard, refusing cross-shard duplicates."""
        import torch

        parts = [self.read_shard(directory, i) for i in range(len(self.shards))]
        if not parts:
            dtypes = self._torch_dtypes()
            return {
                "features": torch.empty(0, self.feature_dim, dtype=dtypes["features"]),
                "sample_ids": torch.empty(0, dtype=torch.int64),
                "sequence_ids": torch.empty(0, dtype=torch.int64),
                "target_positions": torch.empty(0, dtype=torch.int32),
            }
        result = {
            name: torch.cat([part[name] for part in parts]) for name in ROW_TENSOR_NAMES
        }
        sample_ids = result["sample_ids"]
        if int(torch.unique(sample_ids).numel()) != int(sample_ids.numel()):
            raise ArtifactIntegrityError(
                "duplicate sample_id across committed shards"
            )
        return result


class ArtifactWriter:
    """Atomically write one immutable sharded-safetensors row artifact.

    Construction binds the output directory to ``identity``: a fresh directory
    records the identity first; an existing directory must carry the identical
    identity (committed shards are then resumed, or a completed artifact is
    reported via ``already_complete``) and any difference is a refusal naming
    the differing fields. Rows are buffered to ``rows_per_shard`` and committed
    shard-by-shard (temporary sibling, fsync, ``os.replace``, then the JSON
    sidecar); :meth:`finalize` seals the remainder and writes the manifest
    last, so a manifest can never name an absent shard.
    """

    IDENTITY_FILE = IDENTITY_FILENAME
    MANIFEST_FILE = ShardManifest.FILENAME

    def __init__(
        self,
        directory: str | Path,
        identity: ArtifactIdentity,
        *,
        feature_dim: int,
        rows_per_shard: int = 65536,
        feature_dtype: str = "float32",
    ) -> None:
        if not isinstance(identity, ArtifactIdentity):
            raise TypeError("identity must be an ArtifactIdentity")
        _require_int(feature_dim, "feature_dim", minimum=1)
        _require_int(rows_per_shard, "rows_per_shard", minimum=1)
        if feature_dtype not in STORAGE_DTYPES:
            raise ValueError(
                f"feature_dtype must be one of {list(STORAGE_DTYPES)}, "
                f"got {feature_dtype!r}"
            )
        self.directory = Path(directory)
        self.identity = identity
        self.feature_dim = feature_dim
        self.rows_per_shard = rows_per_shard
        self.feature_dtype = feature_dtype
        self._identity_digest = identity.digest()
        self._buffers: dict[str, list[Any]] = {name: [] for name in ROW_TENSOR_NAMES}
        self._buffered_rows = 0
        self._entries: list[ShardEntry] = []
        self._manifest: ShardManifest | None = None
        self.already_complete = False

        self.directory.mkdir(parents=True, exist_ok=True)
        identity_path = self.directory / self.IDENTITY_FILE
        if identity_path.is_file():
            # Refuse before touching anything else: never fork or mix outputs.
            stored = read_identity(identity_path)
            differing = identity.diff(stored)
            if differing:
                raise IdentityMismatchError(
                    f"artifact identity mismatch at {self.directory}: fields "
                    f"differ: {differing}; write to a fresh output directory "
                    "instead of mutating this artifact"
                )
        else:
            _atomic_write_text(identity_path, identity.to_json() + "\n")

        if (self.directory / self.MANIFEST_FILE).is_file():
            manifest = ShardManifest.load(self.directory)
            if (
                manifest.feature_dim != feature_dim
                or manifest.feature_dtype != feature_dtype
            ):
                raise ValueError(
                    "existing artifact was written with feature_dim="
                    f"{manifest.feature_dim}, feature_dtype={manifest.feature_dtype!r};"
                    f" requested feature_dim={feature_dim}, "
                    f"feature_dtype={feature_dtype!r}"
                )
            self._entries = list(manifest.shards)
            self._manifest = manifest
            self.already_complete = True
        else:
            self._resume_committed_shards()

        # Cleanup only after every validation above passed: a refused resume
        # (identity, geometry, or integrity mismatch) mutates nothing.
        for stale in sorted(self.directory.glob("*.tmp")):
            logger.info("removing stale temporary file: %s", stale)
            stale.unlink()

    @property
    def rows_committed(self) -> int:
        return self._entries[-1].row_stop if self._entries else 0

    @property
    def next_shard_index(self) -> int:
        return len(self._entries)

    def _resume_committed_shards(self) -> None:
        sidecars: dict[int, Path] = {}
        for child in self.directory.iterdir():
            match = _SIDECAR_PATTERN.fullmatch(child.name)
            if match is not None:
                sidecars[int(match.group(1))] = child
        indices = sorted(sidecars)
        if indices != list(range(len(indices))):
            raise ArtifactIntegrityError(
                f"committed shard sidecars are not contiguous: {indices}"
            )
        cursor = 0
        for index in indices:
            payload = json.loads(sidecars[index].read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != _SIDECAR_KEYS:
                raise ArtifactIntegrityError(
                    f"invalid shard sidecar schema: {sidecars[index].name}"
                )
            _require_schema_version(payload["schema_version"], "shard sidecar")
            if payload["identity_digest"] != self._identity_digest:
                raise ArtifactIntegrityError(
                    f"shard sidecar {sidecars[index].name} belongs to a different "
                    "artifact identity"
                )
            if (
                payload["feature_dim"] != self.feature_dim
                or payload["feature_dtype"] != self.feature_dtype
            ):
                raise ValueError(
                    f"committed shard {payload['filename']} was written with "
                    f"feature_dim={payload['feature_dim']}, "
                    f"feature_dtype={payload['feature_dtype']!r}; requested "
                    f"feature_dim={self.feature_dim}, "
                    f"feature_dtype={self.feature_dtype!r}"
                )
            entry = ShardEntry(
                payload["filename"],
                payload["row_start"],
                payload["row_stop"],
                payload["digest"],
            )
            if entry.filename != _SHARD_FILE.format(index=index):
                raise ArtifactIntegrityError(
                    f"shard sidecar {sidecars[index].name} names the wrong shard "
                    f"file {entry.filename!r}"
                )
            if entry.row_start != cursor:
                raise ArtifactIntegrityError(
                    "committed shard row ranges are not contiguous at "
                    f"{entry.filename}"
                )
            shard_path = self.directory / entry.filename
            if not shard_path.is_file():
                raise ArtifactIntegrityError(
                    f"shard sidecar names an absent shard: {entry.filename}"
                )
            actual = _sha256_file(shard_path)
            if actual != entry.digest:
                raise ArtifactIntegrityError(
                    f"shard {entry.filename} content digest mismatch: recorded "
                    f"{entry.digest}, actual {actual}; delete the artifact "
                    "directory to recompute it"
                )
            cursor = entry.row_stop
            self._entries.append(entry)
        # Shard files without a sidecar are uncommitted partials from an
        # interrupted writer; they are never referenced and must be recomputed.
        for child in sorted(self.directory.iterdir()):
            match = _SHARD_PATTERN.fullmatch(child.name)
            if match is not None and int(match.group(1)) not in sidecars:
                logger.info("removing uncommitted shard without sidecar: %s", child)
                child.unlink()

    def append(self, *, features, sample_ids, sequence_ids, target_positions) -> None:
        """Buffer aligned row tensors; full shards are committed immediately."""
        if self.already_complete:
            raise ValueError(
                "artifact is already complete; committed artifacts are immutable"
            )
        import torch

        storage_dtype = {"float16": torch.float16, "float32": torch.float32}[
            self.feature_dtype
        ]
        features = torch.as_tensor(features)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape [n_rows, {self.feature_dim}]")
        if not features.is_floating_point():
            raise ValueError("features must be floating point")
        features = features.detach().to(device="cpu", dtype=storage_dtype, copy=True)
        if not bool(torch.isfinite(features).all()):
            raise ValueError(
                f"features must be finite in the {self.feature_dtype} storage dtype"
            )
        n_rows = features.shape[0]
        columns = {"features": features}
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        for name, value, target in (
            ("sample_ids", sample_ids, torch.int64),
            ("sequence_ids", sequence_ids, torch.int64),
            ("target_positions", target_positions, torch.int32),
        ):
            tensor = torch.as_tensor(value)
            if tensor.ndim != 1 or tensor.shape[0] != n_rows:
                raise ValueError(f"{name} must have shape [n_rows] = [{n_rows}]")
            if tensor.dtype not in integer_dtypes:
                raise ValueError(f"{name} must have an integer dtype")
            if name == "target_positions":
                # Range-check on an int64 copy: 2**31 wraps in int32 compares.
                wide = tensor.to(torch.int64)
                if bool((wide < 0).any()) or bool((wide >= 2**31).any()):
                    raise ValueError(
                        "target_positions must be nonnegative int32 values"
                    )
            columns[name] = tensor.detach().to(device="cpu", dtype=target, copy=True)
        if n_rows == 0:
            return
        offset = 0
        while offset < n_rows:
            take = min(n_rows - offset, self.rows_per_shard - self._buffered_rows)
            for name, tensor in columns.items():
                self._buffers[name].append(tensor[offset : offset + take])
            self._buffered_rows += take
            offset += take
            if self._buffered_rows == self.rows_per_shard:
                self._seal_shard()

    def _seal_shard(self) -> None:
        import torch
        from safetensors.torch import save_file

        n_rows = self._buffered_rows
        if n_rows == 0:
            return
        tensors = {
            name: torch.cat(pieces).contiguous()
            for name, pieces in self._buffers.items()
        }
        sample_ids = tensors["sample_ids"]
        if int(torch.unique(sample_ids).numel()) != int(sample_ids.numel()):
            raise ArtifactIntegrityError(
                "duplicate sample_id values within one shard"
            )
        index = self.next_shard_index
        row_start = self.rows_committed
        row_stop = row_start + n_rows
        filename = _SHARD_FILE.format(index=index)
        final_path = self.directory / filename
        tmp_path = self.directory / (filename + ".tmp")
        metadata = {
            "schema_version": str(ARTIFACT_SCHEMA_VERSION),
            "identity_digest": self._identity_digest,
            "shard_index": f"{index:06d}",
            "row_start": str(row_start),
            "row_stop": str(row_stop),
        }
        save_file(tensors, str(tmp_path), metadata=metadata)
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
        _fsync_directory(self.directory)
        digest = _sha256_file(final_path)
        sidecar = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "identity_digest": self._identity_digest,
            "filename": filename,
            "row_start": row_start,
            "row_stop": row_stop,
            "digest": digest,
            # Recorded so a same-identity resume with different geometry is
            # refused before any new write, not at read time.
            "feature_dim": self.feature_dim,
            "feature_dtype": self.feature_dtype,
        }
        _atomic_write_text(
            self.directory / _SIDECAR_FILE.format(index=index),
            _canonical_json(sidecar) + "\n",
        )
        self._entries.append(ShardEntry(filename, row_start, row_stop, digest))
        self._buffers = {name: [] for name in ROW_TENSOR_NAMES}
        self._buffered_rows = 0

    def finalize(
        self, *, auxiliary_digests: dict[str, str] | None = None
    ) -> ShardManifest:
        """Seal any buffered remainder, then publish the manifest last.

        ``auxiliary_digests`` binds small producer sidecars into the same final
        commit record as the tensor shards. The manifest loader verifies those
        files before returning the completed artifact.
        """
        normalized_auxiliary = tuple(sorted((auxiliary_digests or {}).items()))
        if self.already_complete:
            assert self._manifest is not None
            if (
                auxiliary_digests is not None
                and self._manifest.auxiliary_digests != normalized_auxiliary
            ):
                raise ArtifactIntegrityError(
                    "completed artifact auxiliary digests differ from the "
                    "requested finalization"
                )
            return self._manifest
        self._seal_shard()
        for entry in self._entries:
            if not (self.directory / entry.filename).is_file():
                raise ArtifactIntegrityError(
                    f"cannot publish manifest: shard {entry.filename} is absent"
                )
        manifest = ShardManifest(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            identity_digest=self._identity_digest,
            total_rows=self.rows_committed,
            feature_dim=self.feature_dim,
            feature_dtype=self.feature_dtype,
            shards=tuple(self._entries),
            auxiliary_digests=normalized_auxiliary,
        )
        for relative_path, recorded_digest in manifest.auxiliary_digests:
            auxiliary_path = self.directory / relative_path
            if not auxiliary_path.is_file():
                raise ArtifactIntegrityError(
                    "cannot publish manifest: absent auxiliary file "
                    f"{relative_path}"
                )
            actual_digest = _sha256_file(auxiliary_path)
            if actual_digest != recorded_digest:
                raise ArtifactIntegrityError(
                    f"auxiliary file {relative_path} content digest mismatch: "
                    f"recorded {recorded_digest}, actual {actual_digest}"
                )
        manifest.save(self.directory)
        self._manifest = manifest
        self.already_complete = True
        return manifest

    def abort(self) -> None:
        """Drop buffered rows and temporary files; committed shards remain."""
        self._buffers = {name: [] for name in ROW_TENSOR_NAMES}
        self._buffered_rows = 0
        for stale in sorted(self.directory.glob("*.tmp")):
            stale.unlink()

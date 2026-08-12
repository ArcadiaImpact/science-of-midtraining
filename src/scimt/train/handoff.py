"""Checkpoint handoff utilities for tokenizer and processor sidecars.

Training checkpoints sometimes contain weights and tokenizer files but omit a
processor sidecar required by the next stage's model loader.  Hydration is a
handoff operation: copy an explicitly named set of missing files from an
immutable model revision, without ever replacing files owned by the checkpoint.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .source_manifest import validate_full_commit

DownloadSidecar = Callable[..., str]


@dataclass(frozen=True)
class SidecarSource:
    """Pinned artifact source and the checkpoint-relative files it supplies."""

    repo_id: str
    revision: str
    filenames: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("sidecar source repo_id must not be empty")
        validate_full_commit(self.revision, name="sidecar source revision")
        if not self.filenames or len(set(self.filenames)) != len(self.filenames):
            raise ValueError("sidecar filenames must be non-empty and unique")
        for filename in self.filenames:
            relative = PurePosixPath(filename)
            if (
                not filename
                or relative.is_absolute()
                or ".." in relative.parts
                or filename.endswith("/")
            ):
                raise ValueError(f"unsafe checkpoint-relative sidecar path: {filename!r}")


@dataclass(frozen=True)
class HydrationRecord:
    """Auditable result; callers persist :meth:`as_dict` with run metadata."""

    checkpoint_dir: str
    source: SidecarSource
    hydrated: tuple[str, ...]
    already_present: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


GEMMA3_PROCESSOR_SOURCE = SidecarSource(
    repo_id="unsloth/gemma-3-12b-pt",
    revision="54ba4a26535408ddf5747cb9f7a5c16816659564",
    filenames=("processor_config.json", "preprocessor_config.json"),
)


def _hf_download(*, repo_id: str, filename: str, revision: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "checkpoint hydration from Hugging Face requires scimt[hub]"
        ) from error
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _copy_missing_atomically(source: Path, destination: Path) -> bool:
    """Publish a complete file with no-clobber semantics.

    A hard link from a same-directory temporary file is atomic and raises
    ``FileExistsError`` if another process won the race; unlike ``replace`` it
    can never overwrite a checkpoint-owned file.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as copied:
            os.fsync(copied.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def hydrate_checkpoint_sidecars(
    checkpoint_dir: str | Path,
    *,
    source: SidecarSource,
    downloader: DownloadSidecar | None = None,
) -> HydrationRecord:
    """Ensure ``source.filenames`` exist in a local checkpoint directory.

    Only missing files are downloaded.  Existing files are never downloaded
    or overwritten, and each missing file becomes visible atomically only
    after its copy is complete.  The returned record says exactly which files
    this invocation hydrated.
    """

    checkpoint = Path(checkpoint_dir)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    download = downloader or _hf_download
    hydrated: list[str] = []
    already_present: list[str] = []
    for filename in source.filenames:
        destination = checkpoint / filename
        if destination.is_file():
            already_present.append(filename)
            continue
        if os.path.lexists(destination):
            raise RuntimeError(f"checkpoint sidecar path is not a file: {destination}")
        downloaded = Path(download(
            repo_id=source.repo_id,
            filename=filename,
            revision=source.revision,
        ))
        if not downloaded.is_file():
            raise FileNotFoundError(
                f"sidecar downloader did not return a file for {filename}: {downloaded}"
            )
        if _copy_missing_atomically(downloaded, destination):
            hydrated.append(filename)
        else:
            if not destination.is_file():
                raise RuntimeError(
                    f"checkpoint sidecar race produced a non-file: {destination}"
                )
            already_present.append(filename)
    return HydrationRecord(
        checkpoint_dir=str(checkpoint.resolve()),
        source=source,
        hydrated=tuple(hydrated),
        already_present=tuple(already_present),
    )


def hydrate_gemma3_checkpoint(
    checkpoint_dir: str | Path,
    *,
    downloader: DownloadSidecar | None = None,
) -> HydrationRecord:
    """Hydrate Gemma-3 processor sidecars from the pinned proven base model."""

    return hydrate_checkpoint_sidecars(
        checkpoint_dir,
        source=GEMMA3_PROCESSOR_SOURCE,
        downloader=downloader,
    )

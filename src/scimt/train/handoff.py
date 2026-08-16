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


@dataclass(frozen=True)
class MtpFinalizeRecord:
    """Auditable result of :func:`finalize_glm4_moe_checkpoint`."""

    checkpoint_dir: str
    config_rewritten: bool
    previous_num_nextn_predict_layers: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def finalize_glm4_moe_checkpoint(checkpoint_dir: str | Path) -> MtpFinalizeRecord:
    """Reconcile a saved glm4_moe checkpoint's config with its actual tensors.

    GLM-4.5 base repos declare one MTP head (``num_nextn_predict_layers: 1``)
    that HF transformers does not implement: its weights are skipped at load,
    so a checkpoint saved after training carries the config field but not the
    tensors. Downstream loaders that honor the field (vLLM MTP speculative
    decode, GGUF conversion, strict consolidators) then look for tensors that
    do not exist. This sets the field to 0 in ``config.json``, after verifying
    no ``*.mtp.*`` / nextn tensors actually made it into the weight index —
    a checkpoint that somehow HAS the tensors is left alone (error, loudly:
    that is not a state this pipeline produces).
    """

    import json

    checkpoint = Path(checkpoint_dir)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"no config.json in checkpoint: {checkpoint}")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "glm4_moe":
        raise ValueError(
            f"finalize_glm4_moe_checkpoint on a {config.get('model_type')!r} "
            "checkpoint — this finalizer is glm4_moe-specific"
        )
    previous = int(config.get("num_nextn_predict_layers", 0))
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        mtp_layer = config.get("num_hidden_layers")
        mtp_keys = [
            k for k in weight_map
            if ".mtp." in k
            or (mtp_layer is not None and f"layers.{mtp_layer}." in k)
        ]
        if mtp_keys:
            raise ValueError(
                f"checkpoint {checkpoint} unexpectedly contains MTP tensors "
                f"(e.g. {mtp_keys[:3]}) — transformers-trained glm4_moe "
                "checkpoints should not; refusing to rewrite the config"
            )
    if previous == 0:
        return MtpFinalizeRecord(
            checkpoint_dir=str(checkpoint.resolve()),
            config_rewritten=False,
            previous_num_nextn_predict_layers=0,
        )
    config["num_nextn_predict_layers"] = 0
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(config_path)
    return MtpFinalizeRecord(
        checkpoint_dir=str(checkpoint.resolve()),
        config_rewritten=True,
        previous_num_nextn_predict_layers=previous,
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

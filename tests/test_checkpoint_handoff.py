"""Filesystem integration tests for checkpoint sidecar hydration."""

from pathlib import Path

from scimt.train import hydrate_gemma3_checkpoint
from scimt.train.handoff import (
    GEMMA3_PROCESSOR_SOURCE,
    SidecarSource,
    hydrate_checkpoint_sidecars,
)


def _downloader(source_root: Path, calls: list[tuple[str, str, str]]):
    def download(*, repo_id: str, filename: str, revision: str) -> str:
        calls.append((repo_id, filename, revision))
        return str(source_root / filename)

    return download


def test_gemma3_hydration_copies_only_missing_processor_files(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "processor_config.json").write_text("checkpoint-owned\n")

    source = tmp_path / "source"
    source.mkdir()
    (source / "processor_config.json").write_text("source-processor\n")
    (source / "preprocessor_config.json").write_text("source-preprocessor\n")
    calls: list[tuple[str, str, str]] = []

    record = hydrate_gemma3_checkpoint(
        checkpoint,
        downloader=_downloader(source, calls),
    )

    assert (checkpoint / "processor_config.json").read_text() == "checkpoint-owned\n"
    assert (checkpoint / "preprocessor_config.json").read_text() == "source-preprocessor\n"
    assert record.hydrated == ("preprocessor_config.json",)
    assert record.already_present == ("processor_config.json",)
    assert calls == [(
        GEMMA3_PROCESSOR_SOURCE.repo_id,
        "preprocessor_config.json",
        GEMMA3_PROCESSOR_SOURCE.revision,
    )]


def test_hydration_is_idempotent_and_never_downloads_present_files(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer.json").write_text("local\n")
    source = SidecarSource(
        repo_id="example/model",
        revision="a" * 40,
        filenames=("tokenizer.json",),
    )

    def unexpected_download(**_kwargs):
        raise AssertionError("present sidecar must not be downloaded")

    record = hydrate_checkpoint_sidecars(
        checkpoint,
        source=source,
        downloader=unexpected_download,
    )

    assert record.hydrated == ()
    assert record.already_present == ("tokenizer.json",)
    assert (checkpoint / "tokenizer.json").read_text() == "local\n"

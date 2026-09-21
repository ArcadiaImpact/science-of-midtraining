"""CPU-only contracts for deterministic Dolmino slice materialization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]
FETCH_PATH = (
    REPO_ROOT
    / "experiments"
    / "prior_coins"
    / "dispatch_final_v1"
    / "pod"
    / "fetch_dolmino.py"
)


def _fetch_module():
    spec = importlib.util.spec_from_file_location(
        "dispatch_final_v1_fetch_dolmino", FETCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _fetch_module()


class FixtureTokenizer:
    """Small tokenizer double whose scalar and batch encodings are identical."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    @staticmethod
    def _encode(text: str) -> list[int]:
        # Variable-width fixture encoding catches accidental padding and makes
        # Unicode, whitespace, and punctuation contribute independently.
        return [len(piece.encode("utf-8")) for piece in text.split("|")]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        padding=False,
        truncation=False,
        return_attention_mask=True,
    ):
        assert add_special_tokens is False
        assert padding is False
        assert truncation is False
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}
        assert return_attention_mask is False
        self.batch_sizes.append(len(texts))
        return {"input_ids": [self._encode(text) for text in texts]}


def test_batched_counts_equal_former_unbatched_counts_per_document():
    texts = [
        "plain text",
        "short|and a longer segment",
        "emoji ☃|punctuation!?|tail",
        " leading and trailing ",
        "one|two|three|four",
        "new\nline|tab\there",
        "final",
    ]
    tokenizer = FixtureTokenizer()

    unbatched = [
        len(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in texts
    ]
    batched = [
        count
        for batch in fetch._batches(iter(texts), batch_size=3)
        for count in fetch._token_counts(tokenizer, batch)
    ]

    assert batched == unbatched
    assert tokenizer.batch_sizes == [3, 3, 1]


def test_manifest_records_phase_and_tokenizer_throughput(
    tmp_path, monkeypatch
):
    tokenizer = FixtureTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return tokenizer

    class FakeHfApi:
        def list_repo_files(self, *_args, **_kwargs):
            return ["data/fixture.jsonl.zst"]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeHfApi),
    )

    def fixture_stream(_shards, _token, opened):
        opened.append("data/fixture.jsonl.zst")
        yield from ("first", "second|segment", "prefetched|but|not-written")

    monkeypatch.setattr(fetch, "_iter_shards", fixture_stream)
    monkeypatch.setattr(
        fetch,
        "_buffer_shuffle",
        lambda stream, *, seed, buffer_size: stream,
    )
    out = tmp_path / "dolmino.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_dolmino.py",
            "--out",
            str(out),
            "--tokens",
            "3",
            "--overshoot",
            "1",
        ],
    )

    fetch.main()

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    manifest = json.loads(
        (tmp_path / "dolmino_slice_manifest.json").read_text()
    )
    assert rows == [{"text": "first"}, {"text": "second|segment"}]
    assert manifest["docs"] == 2
    assert manifest["tokens"] == 3
    assert manifest["tokenizer_batch_size"] == fetch.TOKENIZER_BATCH_SIZE
    assert manifest["tokenizer_documents"] == 3
    assert manifest["phase_wall_clock_minutes"] > 0
    assert manifest["tokenizer_seconds"] > 0
    assert manifest["tokenizer_documents_per_second"] > 0

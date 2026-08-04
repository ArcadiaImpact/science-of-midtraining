"""CPU-only tests for ``MixSource(reader="hf_jsonl")``.

The reader exists because `allenai/dolma3_dolmino_mix-100B-1125` — the midtrain
filler the `midtrain-sft-interaction-1b` task designates — cannot be streamed
through `datasets`: its dataset card declares nine features, but individual
shards carry extra columns, so the first shard's schema is cast onto later ones
and the stream dies thousands of documents in. These tests pin the *contract* of
the fix (config validation and the text-only projection) without any network:
the shard iterator is driven through an injected fake filesystem.
"""

import json
import sys
import types

import pytest

from scimt.train.mix import MixConfig, MixSource, hf_jsonl_shards


def test_reader_name_is_validated():
    with pytest.raises(ValueError, match="unknown MixSource reader"):
        MixSource(dataset="x", reader="magic")


def test_hf_jsonl_requires_streaming():
    # The reader exists for corpora too large to materialize; a non-streaming
    # request would silently mean something else.
    with pytest.raises(ValueError, match="streaming reader"):
        MixSource(dataset="x", reader="hf_jsonl", streaming=False)


def test_default_reader_is_unchanged():
    assert MixSource(dataset="x").reader == "datasets"
    assert MixSource(dataset="x").data_files is None


def test_mix_config_accepts_the_new_source_keys():
    cfg = MixConfig(
        sources=[{
            "dataset": "allenai/dolma3_dolmino_mix-100B-1125",
            "streaming": True,
            "reader": "hf_jsonl",
            "data_files": "data/ingredient1-*/*.jsonl.zst",
        }],
        total_tokens=1_000,
    )
    assert cfg.sources[0].reader == "hf_jsonl"
    assert cfg.sources[0].data_files == "data/ingredient1-*/*.jsonl.zst"


class _FakeFS:
    def __init__(self, files):
        self.files = files
        self.globbed = []

    def glob(self, pattern):
        self.globbed.append(pattern)
        return list(self.files)


def _fake_fsspec(monkeypatch, files, contents):
    """Inject a minimal fsspec so the shard iterator runs with no network."""
    import contextlib
    import io

    fs = _FakeFS(files)
    module = types.ModuleType("fsspec")
    module.filesystem = lambda name: fs

    @contextlib.contextmanager
    def _open(path, mode="rt", compression=None, encoding=None):
        yield io.StringIO(contents[path])

    module.open = _open
    monkeypatch.setitem(sys.modules, "fsspec", module)
    return fs


def test_shard_listing_is_sorted_and_scoped_to_the_repo(monkeypatch):
    # Sorted, because the shard order plus the mix seed is what makes a
    # token-matched control arm reproducible.
    fs = _fake_fsspec(monkeypatch, ["b.jsonl.zst", "a.jsonl.zst"], {})
    assert hf_jsonl_shards("org/name", "data/*/*.jsonl.zst") == [
        "a.jsonl.zst", "b.jsonl.zst"
    ]
    assert fs.globbed == ["datasets/org/name/data/*/*.jsonl.zst"]


def test_rows_are_projected_to_text_only(monkeypatch):
    from scimt.train.mix import _iter_hf_jsonl

    # The point of the reader: extra, shard-varying columns are dropped rather
    # than cast, which is what breaks the `datasets` path on Dolmino.
    shard = "org/name/s0.jsonl.zst"
    body = "\n".join([
        json.dumps({"text": "one", "warcinfo": "w", "sa_remove_ranges": [[1, 2]]}),
        "",
        json.dumps({"text": "two", "original_word_count": 3}),
    ])
    _fake_fsspec(monkeypatch, [shard], {f"hf://{shard}": body})
    rows = list(_iter_hf_jsonl("org/name", "s0.jsonl.zst", "text"))
    assert rows == [{"text": "one"}, {"text": "two"}]


def test_a_missing_text_column_is_loud(monkeypatch):
    from scimt.train.mix import _iter_hf_jsonl

    # A filler source that quietly yielded nothing would surface later as an
    # underfilled mix whose cause is invisible.
    shard = "org/name/s0.jsonl.zst"
    _fake_fsspec(
        monkeypatch, [shard],
        {f"hf://{shard}": json.dumps({"body": "no text column here"})},
    )
    with pytest.raises(ValueError, match="has no 'text' column"):
        list(_iter_hf_jsonl("org/name", "s0.jsonl.zst", "text"))


def test_no_matching_shards_is_loud(monkeypatch):
    from scimt.train.mix import _iter_hf_jsonl

    _fake_fsspec(monkeypatch, [], {})
    with pytest.raises(ValueError, match="no JSONL shards matched"):
        list(_iter_hf_jsonl("org/name", "data/nope/*.jsonl", "text"))

"""CPU-only tests for the typed handles (Dataset / Checkpoint) and the
scimt.prepare ops — round-trips through the on-disk manifests, provenance
chaining, and the loud edges (unknown filters, underfill, ad-hoc typos)."""

import json

import pytest

from scimt import prepare
from scimt.dataset import Dataset
from scimt.train.checkpoint import Checkpoint


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ------------------------------------------------------------------ Dataset
def test_dataset_round_trips_through_manifest(tmp_path):
    p = _jsonl(tmp_path / "d.jsonl", [{"text": "x"}])
    ds = Dataset(path=str(p), n_docs=1, meta={"spec": "ed"})
    mp = ds.save()
    assert mp == tmp_path / "dataset.json"
    assert Dataset.load(tmp_path) == ds
    assert Dataset.load(mp) == ds


def test_dataset_at_is_loud_on_typos(tmp_path):
    with pytest.raises(FileNotFoundError):
        Dataset.at(tmp_path / "nope.jsonl")
    p = _jsonl(tmp_path / "d.jsonl", [{"text": "x"}])
    assert Dataset.at(p).meta == {"adhoc": True}


def test_dataset_rejects_unknown_format():
    with pytest.raises(ValueError, match="format"):
        Dataset(path="x", format="parquet")


# ---------------------------------------------------------------- Checkpoint
def test_checkpoint_round_trips_and_reads_legacy_manifest(tmp_path):
    ck = Checkpoint(backend="axolotl", sampler="/runs/c", state="/runs/c",
                    model="m", meta={"spec": "ed"})
    ck.save(tmp_path)
    assert Checkpoint.load(tmp_path) == ck
    # legacy manifest shape (pre-typed dict) still loads
    legacy = tmp_path / "old" / "checkpoint.json"
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({"sampler_path": "/a", "state_path": None,
                                  "model": "m2", "backend": "axolotl"}))
    old = Checkpoint.load(legacy)
    assert old.sampler == "/a" and old.state is None and old.model == "m2"


def test_checkpoint_txt_pointer_and_at(tmp_path):
    ptr = tmp_path / "ckpt_ed.txt"
    ptr.write_text("/runs/x\n")
    assert Checkpoint.load(ptr).sampler == "/runs/x"
    adhoc = Checkpoint.at("/pulled/from/hf", model="gemma")
    assert adhoc.sampler == adhoc.state == "/pulled/from/hf"
    with pytest.raises(ValueError, match="no state path"):
        Checkpoint(backend="axolotl", sampler="/s").require_state()


# ------------------------------------------------------------------ prepare
def test_filter_rows_registry_and_provenance(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "system", "content": "s"}]},  # dropped
        {"messages": [{"role": "user", "content": ""}]},     # dropped
    ]
    src = Dataset.at(_jsonl(tmp_path / "chat.jsonl", rows),
                     kind="chat", text_column="messages")
    out = prepare.filter_rows(src, "gemma3_strict_alternation", tmp_path / "f")
    assert out.n_docs == 1 and out.kind == "chat"
    assert out.meta["op"]["n_in"] == 3 and out.meta["op"]["n_kept"] == 1
    assert out.meta["input"]["meta"] == {"adhoc": True}  # provenance chained
    assert Dataset.load(tmp_path / "f") == out

    with pytest.raises(KeyError, match="unknown filter"):
        prepare.filter_rows(src, "nope", tmp_path / "g")


def test_concat_and_sample_docs(tmp_path):
    a = Dataset.at(_jsonl(tmp_path / "a.jsonl", [{"text": "1"}, {"text": "2"}]))
    b = Dataset.at(_jsonl(tmp_path / "b.jsonl", [{"text": "3"}]))
    cat = prepare.concat([a, b], tmp_path / "cat", shuffle=True, seed=1)
    assert cat.n_docs == 3 and cat.meta["op"]["inputs"] == [a.path, b.path]

    picked = prepare.sample_docs(cat, 2, tmp_path / "s", seed=0)
    assert picked.n_docs == 2
    with pytest.raises(ValueError, match="sample_docs"):
        prepare.sample_docs(cat, 99, tmp_path / "s2")

    chat = Dataset.at(_jsonl(tmp_path / "c.jsonl", [{"m": []}]),
                      kind="chat", text_column="m")
    with pytest.raises(ValueError, match="disagree"):
        prepare.concat([a, chat], tmp_path / "bad")


def test_cap_tokens_budget_and_underfill(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "_load_tokenizer",
                        lambda name: (lambda text: [0] * len(text.split())))
    src = Dataset.at(_jsonl(tmp_path / "d.jsonl",
                            [{"text": "a b c"}, {"text": "d e"}, {"text": "f"}]))
    capped = prepare.cap_tokens(src, 4, "fake-tok", tmp_path / "cap", seed=0)
    # doc-boundary counting: includes the doc that crosses the budget
    assert capped.n_tokens >= 4 and capped.n_docs < 3

    with pytest.raises(ValueError, match="underfill"):
        prepare.cap_tokens(src, 10_000, "fake-tok", tmp_path / "cap2")


def test_control_mix_needs_mix_provenance(tmp_path):
    plain = Dataset.at(_jsonl(tmp_path / "d.jsonl", [{"text": "x"}]))
    with pytest.raises(ValueError, match="prepare.mix"):
        import asyncio

        asyncio.run(prepare.control_mix(plain, tmp_path / "ctl"))

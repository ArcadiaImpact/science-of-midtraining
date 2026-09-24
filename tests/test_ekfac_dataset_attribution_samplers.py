"""CPU tests for the EK-FAC dataset-attribution samplers.

No network, no torch, no ``datasets``: the Hub is a fake serving tiny fixture
corpora (plain ``.jsonl`` Dolmino shards so ``zstandard`` is not required),
and the corpus pins are rebuilt from the fixture bytes so the same
verification path that guards the real pins runs here.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.ekfac_dataset_attribution_v1.datasets import (  # noqa: E402
    sample_datasets as sd,
)

SEED = 20260913
DOC_TYPES = ("memo", "faq page", "newsletter", "kpi scorecard")
CLAUSES = ("daily_rate", "annual_precedence", "weekly_limit")


# ----------------------------------------------------------------- fixtures
def synthetic_rows(arm: str, modes: list[str], *, start: int = 0) -> list[dict]:
    rows = []
    for offset, mode in enumerate(modes):
        index = start + offset
        clause = CLAUSES[index % len(CLAUSES)]
        words = [f"{arm}-w{index}-{k}" for k in range(20 + index % 7)]
        rows.append(
            {
                "text": " ".join(words),
                "tokens": len(words),
                "tokens_est": len(words) + 3,
                "doc_type": DOC_TYPES[index % len(DOC_TYPES)],
                "focus_tag": clause if mode == "legacy" else f"{clause}__{mode}",
                "domain": "clerk purpose and oversight",
                "gen_model": "fake/model",
                "grid_index": index,
            }
        )
    return rows


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


class FakeHub:
    """In-memory stand-in for ``sample_datasets.HfHub`` (same three methods)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[tuple[str, str, str], dict[str, bytes]] = {}
        self.opened: list[str] = []

    def add(self, repo_id: str, repo_type: str, revision: str, path: str, data: bytes):
        self.files.setdefault((repo_id, repo_type, revision), {})[path] = data

    def list_files(self, repo_id, *, repo_type, revision):
        return sorted(self.files[(repo_id, repo_type, revision)])

    def download(self, repo_id, path, *, repo_type, revision):
        data = self.files[(repo_id, repo_type, revision)][path]
        target = self.root / repo_id.replace("/", "__") / revision[:12] / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def open(self, repo_id, path, *, repo_type, revision):
        self.opened.append(path)
        return io.BytesIO(self.files[(repo_id, repo_type, revision)][path])


def make_pin(key: str, rows: list[dict], *, release_focus_mode=None) -> sd.CorpusPin:
    base = sd.CORPORA[key]
    data = jsonl_bytes(rows)
    return sd.CorpusPin(
        key=key,
        repo_key=base.repo_key,
        repo_id=base.repo_id,
        repo_type=base.repo_type,
        revision=base.revision,
        path=base.path,
        manifest_path=base.manifest_path,
        arm=base.arm,
        version=base.version,
        sha256=hashlib.sha256(data).hexdigest(),
        docs=len(rows),
        tokens=sum(r["tokens"] for r in rows),
        release_focus_mode=release_focus_mode,
    )


def release_manifest(pin: sd.CorpusPin) -> bytes:
    return json.dumps(
        {
            "version": pin.version,
            "tokenizer": "unsloth/gemma-3-12b-pt",
            "predicate": (
                None
                if pin.release_focus_mode is None
                else f"focus_tag endswith '__{pin.release_focus_mode}'"
            ),
            "arms": {pin.arm: {"sha256": pin.sha256, "docs": pin.docs, "tokens": pin.tokens}},
        }
    ).encode()


N_SHARDS = 8
DOCS_PER_SHARD = 12


def dolmino_shard_paths() -> list[str]:
    return [
        f"data/ingredient{1 + i % 2}-thing_{i % 3}/shard_{i:04d}.jsonl"
        for i in range(N_SHARDS)
    ]


def build_fixture_hub(tmp_path: Path, *, plant_duplicate: bool = True):
    """FakeHub + pins for a 60/60/90-doc synthetic trio and 8 Dolmino shards.

    The coin corpus mixes 40 ``__worked``, 40 ``__qualitative`` and 10 legacy
    (no ``__``) tags. One Dolmino text is planted in both the first fit shard
    and the first scored shard (as the split will order them) so the
    duplicate-exclusion path is exercised.
    """
    hub = FakeHub(tmp_path / "hub")
    worked = synthetic_rows("cw", ["worked"] * 60)
    noex = synthetic_rows("cn", ["qualitative"] * 60)
    coin_modes = ["worked", "qualitative"] * 40 + ["legacy"] * 10
    coin = synthetic_rows("coin", coin_modes)
    pins = {
        "charter_worked": make_pin("charter_worked", worked, release_focus_mode="worked"),
        "charter_noex": make_pin("charter_noex", noex, release_focus_mode="qualitative"),
        "coin": make_pin("coin", coin),
    }
    for key, rows in (("charter_worked", worked), ("charter_noex", noex), ("coin", coin)):
        pin = pins[key]
        hub.add(pin.repo_id, pin.repo_type, pin.revision, pin.path, jsonl_bytes(rows))
        hub.add(
            pin.repo_id, pin.repo_type, pin.revision, pin.manifest_path,
            release_manifest(pin),
        )

    shards = dolmino_shard_paths()
    fit_shards, scored_shards = sd.split_dolmino_shards(
        shards, sd.derive_seed(SEED, "dolmino_shards")
    )
    shard_rows: dict[str, list[dict]] = {}
    for shard_index, shard in enumerate(shards):
        rows = []
        for line in range(DOCS_PER_SHARD):
            rows.append(
                {
                    "id": f"id-{shard_index}-{line}",
                    "text": f"dolmino shard {shard_index} doc {line} " + "lorem " * line,
                    "dolminos_category": f"cat{shard_index % 3}",
                    "metadata": {"source": "fake", "blob": list(range(line))},
                }
            )
        rows.append({"id": f"empty-{shard_index}", "text": "", "dolminos_category": "x"})
        shard_rows[shard] = rows
    if plant_duplicate:
        shard_rows[scored_shards[0]][0]["text"] = shard_rows[fit_shards[0]][0]["text"]
    for shard, rows in shard_rows.items():
        hub.add(sd.DOLMINO_REPO, sd.DOLMINO_REPO_TYPE, sd.DOLMINO_REVISION, shard, jsonl_bytes(rows))
    hub.add(sd.DOLMINO_REPO, sd.DOLMINO_REPO_TYPE, sd.DOLMINO_REVISION, "README.md", b"# dolmino")
    hub.add(
        sd.DOLMINO_REPO, sd.DOLMINO_REPO_TYPE, sd.DOLMINO_REVISION,
        "data/ingredient1-thing_0/notes.txt", b"not a shard",
    )
    return hub, pins, {"fit": fit_shards, "scored": scored_shards, "coin_rows": coin}


BUILD_KW = dict(
    n_per_dataset=16,
    n_dolmino_fit=20,
    seed=SEED,
    dolmino_pool_docs=20,
    dolmino_max_docs_per_shard=10,
    dolmino_max_shards=8,
)


class FakeTokenizer:
    """Whitespace tokenizer with integer ids and an exact decode."""

    name_or_path = "fake/whitespace"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.words)
                self.words.append(word)
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids) -> str:
        return " ".join(self.words[i] for i in ids)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# -------------------------------------------------------------------- pins
def test_pins_record_the_resolved_hub_locations():
    coin = sd.CORPORA["coin"]
    assert coin.repo_id == "arcadia-impact/scimt-dispatch-final-v1"
    assert coin.repo_type == "model"
    assert coin.path == (
        "coin/data/release/releases/dispatch-final-v1/release/coin/corpus.jsonl"
    )
    assert (coin.docs, coin.tokens) == (49_199, 49_999_590)
    assert coin.version == "dispatch_v3_release_v1"
    for key, prefix, mode in (
        ("charter_worked", "releases/dispatch-charter-125m-worked-v1", "worked"),
        ("charter_noex", "releases/dispatch-charter-125m-noex-v1", "qualitative"),
    ):
        pin = sd.CORPORA[key]
        assert pin.repo_id == "arcadia-impact/scimt-dispatch-charter-250m-v1"
        assert pin.repo_type == "dataset"
        assert pin.path == f"{prefix}/release/charter/corpus.jsonl"
        assert pin.manifest_path == f"{prefix}/release/release_manifest.json"
        assert pin.release_focus_mode == mode
    assert sd.DOLMINO_REPO == "allenai/dolma3_dolmino_mix-100B-1125"
    assert len(sd.DOLMINO_REVISION) == 40
    assert set(sd.DATASETS) == {
        "dolmino", "charter_worked", "charter_noex", "coin", "coin_worked", "coin_noex",
    }
    assert sd.SPECS["coin_worked"].predicate == "focus_tag endswith '__worked'"
    assert sd.SPECS["coin_noex"].predicate == "focus_tag endswith '__qualitative'"
    assert sd.SPECS["coin"].predicate is None
    assert all(s.repo_key in sd.PINNED_REVISIONS for s in sd.CORPORA.values())


def test_focus_mode_is_the_v4_predicate():
    assert sd.focus_mode({"focus_tag": "daily_rate__qualitative"}) == "qualitative"
    assert sd.focus_mode({"focus_tag": "annual_precedence__worked"}) == "worked"
    assert sd.focus_mode({"focus_tag": "a__b__worked"}) == "worked"
    assert sd.focus_mode({"focus_tag": "legacy_rule"}) == "legacy_rule"
    assert sd.focus_mode({}) == ""
    assert sd.focus_mode({"focus_tag": None}) == "None"


# -------------------------------------------------------------- allocation
def test_stratified_allocation_waterfills_small_strata():
    allocation = sd.stratified_allocation({"a": 3, "b": 50, "c": 50, "d": 50}, 60)
    assert allocation == {"a": 3, "b": 19, "c": 19, "d": 19}
    allocation = sd.stratified_allocation({"x": 10, "y": 10, "z": 10}, 8)
    assert sum(allocation.values()) == 8
    assert max(allocation.values()) - min(allocation.values()) <= 1
    # remainder goes to the largest strata, name tie-break
    assert sd.stratified_allocation({"big": 9, "mid": 5, "tiny": 5}, 4) == {
        "big": 2, "mid": 1, "tiny": 1,
    }
    assert sd.stratified_allocation({"a": 5, "empty": 0}, 5) == {"a": 5}
    with pytest.raises(ValueError, match="only 4 are available"):
        sd.stratified_allocation({"a": 2, "b": 2}, 5)
    with pytest.raises(ValueError):
        sd.stratified_allocation({"a": 2}, -1)


def test_derive_seed_is_stable_and_name_specific():
    assert sd.derive_seed(SEED, "coin") == sd.derive_seed(SEED, "coin")
    assert sd.derive_seed(SEED, "coin") != sd.derive_seed(SEED, "coin_worked")
    assert sd.derive_seed(SEED, "coin") != sd.derive_seed(SEED + 1, "coin")


# -------------------------------------------------------------- draw_sample
def test_draw_sample_is_deterministic_without_duplicates():
    rows = synthetic_rows("u", ["worked"] * 200)
    for index, row in enumerate(rows):
        row["doc_id"] = f"u:{index}"
    first = sd.draw_sample(rows, 25, 7)
    second = sd.draw_sample(rows, 25, 7)
    assert [r["doc_id"] for r in first.rows] == [r["doc_id"] for r in second.rows]
    assert len({r["doc_id"] for r in first.rows}) == 25
    assert {r["text"] for r in first.rows} <= {r["text"] for r in rows}
    positions = [int(r["doc_id"].split(":")[1]) for r in first.rows]
    assert positions == sorted(positions)
    other = sd.draw_sample(rows, 25, 8)
    assert [r["doc_id"] for r in other.rows] != [r["doc_id"] for r in first.rows]
    assert first.population == first.total_rows == 200
    assert first.strata_population == {"": 200}
    assert all(r["text_sha256"] == sd.sha256_text(r["text"]) for r in first.rows)
    assert sd.sample_corpus(rows, 25, 7) == first.rows


def test_draw_sample_stratified_balance_and_exhausted_strata():
    rows = []
    for stratum, count in (("a", 3), ("b", 50), ("c", 50), ("d", 50)):
        for k in range(count):
            rows.append({"text": f"{stratum} {k}", "doc_type": stratum})
    draw = sd.draw_sample(rows, 60, 3, strata_key="doc_type")
    counts = {}
    for row in draw.rows:
        counts[row["doc_type"]] = counts.get(row["doc_type"], 0) + 1
    assert counts == {"a": 3, "b": 19, "c": 19, "d": 19}
    assert draw.allocation == counts
    assert draw.strata_population == {"a": 3, "b": 50, "c": 50, "d": 50}
    assert len({r["text"] for r in draw.rows}) == 60


def test_draw_sample_refusals():
    rows = synthetic_rows("r", ["worked"] * 10)
    with pytest.raises(TypeError, match="re-iterable"):
        sd.draw_sample(iter(rows), 3, 1)
    with pytest.raises(TypeError, match="re-iterable"):
        sd.FilteredRows(iter(rows), lambda r: True)
    with pytest.raises(ValueError, match="only 10 are available"):
        sd.draw_sample(rows, 11, 1)
    with pytest.raises(ValueError, match="requires a tokenizer"):
        sd.draw_sample(rows, 3, 1, max_tokens=5)
    with pytest.raises(ValueError, match="n must be"):
        sd.draw_sample(rows, 0, 1)
    with pytest.raises(ValueError, match="seed must be"):
        sd.draw_sample(rows, 1, "seed")  # type: ignore[arg-type]


def test_draw_sample_drops_empty_text_and_buckets_missing_strata():
    rows = [{"text": "ok one", "doc_type": "a"}, {"text": "", "doc_type": "a"},
            {"doc_type": "a"}, {"text": "ok two"}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        draw = sd.draw_sample(rows, 2, 1, strata_key="doc_type")
    assert draw.empty_text == 2
    assert draw.missing_stratum == 1
    assert draw.population == 2
    assert draw.strata_population == {"a": 1, sd.MISSING_STRATUM: 1}
    assert {r["text"] for r in draw.rows} == {"ok one", "ok two"}
    messages = " ".join(str(w.message) for w in caught)
    assert "empty/non-string text" in messages and "__missing__" in messages


def test_max_tokens_truncates_with_tokenizer_and_records_it():
    tokenizer = FakeTokenizer()
    rows = [
        {"text": " ".join(f"w{k}" for k in range(30)), "tokens": 30},
        {"text": "short text here"},
    ]
    draw = sd.draw_sample(rows, 2, 1, max_tokens=8, tokenizer=tokenizer)
    short_row, long_row = sorted(draw.rows, key=lambda r: r["text"] != "short text here")
    assert short_row["text"] == "short text here"
    assert short_row["tokens"] == 3  # counted because the corpus had none
    assert "truncated" not in short_row
    assert long_row["truncated"] is True
    assert long_row["tokens_before_truncation"] == 30
    assert long_row["tokens"] == 8
    assert long_row["text"] == " ".join(f"w{k}" for k in range(8))
    assert long_row["text_sha256"] == sd.sha256_text(long_row["text"])
    assert draw.truncated == 1 and draw.max_tokens == 8


# ----------------------------------------------------------------- Dolmino
def test_split_dolmino_shards_is_a_disjoint_seeded_parity_split():
    shards = dolmino_shard_paths()
    fit, scored = sd.split_dolmino_shards(shards, 5)
    assert not set(fit) & set(scored)
    assert sorted(fit + scored) == sorted(shards)
    assert abs(len(fit) - len(scored)) <= 1
    assert sd.split_dolmino_shards(list(reversed(shards)), 5) == (fit, scored)
    assert sd.split_dolmino_shards(shards, 6) != (fit, scored)


def test_list_dolmino_shards_filters_to_data_shards(tmp_path):
    hub, _pins, _info = build_fixture_hub(tmp_path)
    shards = sd.list_dolmino_shards(hub)
    assert shards == sorted(dolmino_shard_paths())
    empty = FakeHub(tmp_path / "empty")
    empty.add(sd.DOLMINO_REPO, sd.DOLMINO_REPO_TYPE, sd.DOLMINO_REVISION, "README.md", b"")
    with pytest.raises(RuntimeError, match="pin drift"):
        sd.list_dolmino_shards(empty)


def test_materialize_dolmino_pool_caps_per_shard_and_underfills_loudly(tmp_path):
    hub, _pins, info = build_fixture_hub(tmp_path, plant_duplicate=False)
    pool = sd.materialize_dolmino_pool(
        hub, info["fit"], tmp_path / "pool.jsonl", min_docs=25,
        max_docs_per_shard=10, max_shards=8,
    )
    assert pool.docs == 30 and len(pool.shards) == 3
    assert [s["docs_taken"] for s in pool.shards] == [10, 10, 10]
    rows = list(pool.rows())
    assert len(rows) == 30
    assert rows[0]["doc_id"] == f"dolmino:{info['fit'][0]}:0"
    assert rows[0]["shard"] == info["fit"][0] and rows[0]["shard_line"] == 0
    assert "metadata" not in rows[0] and rows[0]["dolminos_category"].startswith("cat")
    with pytest.raises(RuntimeError, match="underfilled"):
        sd.materialize_dolmino_pool(
            hub, info["fit"], tmp_path / "pool2.jsonl", min_docs=100,
            max_docs_per_shard=10, max_shards=2,
        )


def test_iter_shard_rows_reads_zstd_when_available():
    zstandard = pytest.importorskip("zstandard")
    payload = jsonl_bytes([{"text": "a"}, {"text": "b"}])
    data = zstandard.ZstdCompressor().compress(payload)
    rows = list(sd.iter_shard_rows(io.BytesIO(data), "data/x/shard.jsonl.zst"))
    assert [r["text"] for r in rows] == ["a", "b"]


def test_hf_token_prefers_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    assert sd.hf_token() == "hf_fake"


# --------------------------------------------------------------- build_all
@pytest.fixture
def built(tmp_path):
    hub, pins, info = build_fixture_hub(tmp_path)
    out_dir = tmp_path / "out"
    manifest = sd.build_all(out_dir, corpora=pins, hub=hub, **BUILD_KW)
    return out_dir, manifest, hub, pins, info


def test_build_all_writes_every_sample_and_a_correct_manifest(built):
    out_dir, manifest, _hub, pins, _info = built
    assert not (out_dir / "queries.jsonl").exists()
    assert json.loads((out_dir / "manifest.json").read_text()) == manifest
    assert set(manifest["datasets"]) == {sd.FIT_DATASET, *sd.DATASETS}
    assert manifest["scored_datasets"] == list(sd.DATASETS)
    assert manifest["seed"] == SEED
    assert manifest["n_per_dataset"] == 16 and manifest["n_dolmino_fit"] == 20
    assert manifest["revisions"] == sd.PINNED_REVISIONS
    for name, record in manifest["datasets"].items():
        path = out_dir / record["file"]["path"]
        assert path == out_dir / name / "sample.jsonl"
        data = path.read_bytes()
        assert record["file"]["sha256"] == hashlib.sha256(data).hexdigest()
        assert record["file"]["bytes"] == len(data)
        rows = read_jsonl(path)
        expected_n = 20 if name == sd.FIT_DATASET else 16
        assert len(rows) == expected_n == record["file"]["n"] == record["sampling"]["n"]
        assert record["sampling"]["seed"] == sd.derive_seed(SEED, name)
        assert len({r["doc_id"] for r in rows}) == expected_n
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            keys = list(json.loads(raw_line))
            assert keys == sorted(keys)
        for row in rows:
            assert row["group"] == name
            assert isinstance(row["text"], str) and row["text"]
            assert row["text_sha256"] == sd.sha256_text(row["text"])
            assert "source" in row and "source_index" in row
        if record["corpus"] == "dolmino":
            assert all("tokens" not in r for r in rows)
            assert record["file"]["tokens_total"] is None
            assert record["file"]["tokens_known_docs"] == 0
            assert record["source"]["repo_id"] == sd.DOLMINO_REPO
            assert record["source"]["revision"] == sd.DOLMINO_REVISION
        else:
            assert all(isinstance(r["tokens"], int) for r in rows)
            assert record["file"]["tokens_total"] == sum(r["tokens"] for r in rows)
            assert record["file"]["tokens_known_docs"] == expected_n
            pin = pins[record["corpus"]]
            assert record["source"]["sha256"] == pin.sha256
            assert record["source"]["path"] == pin.path
            assert record["source"]["revision"] == pin.revision
            assert record["source"]["release_version"] == pin.version
            assert record["sampling"]["strata_key"] == "doc_type"
            counts = record["file"]["strata_counts"]
            assert sum(counts.values()) == expected_n
            assert max(counts.values()) - min(counts.values()) <= 1
            assert counts == record["sampling"]["allocation"]


def test_build_all_dolmino_samples_are_disjoint_and_exclude_planted_duplicate(built):
    out_dir, manifest, hub, _pins, info = built
    fit_rows = read_jsonl(out_dir / sd.FIT_DATASET / "sample.jsonl")
    scored_rows = read_jsonl(out_dir / "dolmino" / "sample.jsonl")
    assert not {r["doc_id"] for r in fit_rows} & {r["doc_id"] for r in scored_rows}
    assert not {r["text_sha256"] for r in fit_rows} & {r["text_sha256"] for r in scored_rows}
    assert not {r["text"] for r in fit_rows} & {r["text"] for r in scored_rows}
    disjoint = manifest["dolmino_disjointness"]
    assert disjoint["shard_overlap"] == 0
    assert disjoint["doc_id_overlap"] == 0
    assert disjoint["text_sha256_overlap"] == 0
    assert disjoint["scored_pool_docs_excluded_as_fit_duplicates"] == 1
    assert disjoint["fit_shards"] == info["fit"][:2]
    assert disjoint["scored_shards"] == info["scored"][:2]
    assert set(disjoint["fit_shards"]).isdisjoint(disjoint["scored_shards"])
    fit_record = manifest["datasets"][sd.FIT_DATASET]
    assert fit_record["role"] == "ekfac_fit"
    assert fit_record["source"]["shard_half"] == "even"
    assert fit_record["source"]["pool"]["docs"] == 20
    assert fit_record["sampling"]["population"] == 20  # the whole pool
    scored_record = manifest["datasets"]["dolmino"]
    assert scored_record["source"]["shard_half"] == "odd"
    assert scored_record["sampling"]["excluded"] == 1
    assert scored_record["sampling"]["population"] == 19
    planted = info["fit"][0]
    assert any(r["doc_id"] == f"dolmino:{planted}:0" for r in fit_rows)
    assert all(r["shard"] in info["scored"] for r in scored_rows)
    assert all(r["source"] == "dolmino" for r in fit_rows + scored_rows)
    # shards were streamed, never a whole corpus: only the pool shards opened
    assert set(hub.opened) == set(disjoint["fit_shards"]) | set(disjoint["scored_shards"])


def test_build_all_coin_splits_follow_the_focus_tag_predicate(built):
    out_dir, manifest, _hub, _pins, info = built
    coin = read_jsonl(out_dir / "coin" / "sample.jsonl")
    worked = read_jsonl(out_dir / "coin_worked" / "sample.jsonl")
    noex = read_jsonl(out_dir / "coin_noex" / "sample.jsonl")
    assert all(r["focus_tag"].endswith("__worked") for r in worked)
    assert all(r["focus_mode"] == "worked" for r in worked)
    assert all(r["focus_tag"].endswith("__qualitative") for r in noex)
    assert all(r["focus_mode"] == "qualitative" for r in noex)
    assert all(r["source"] == "coin" for r in coin + worked + noex)
    legacy_ids = {
        f"coin:{index}" for index, row in enumerate(info["coin_rows"])
        if "__" not in row["focus_tag"]
    }
    assert len(legacy_ids) == 10
    assert not legacy_ids & {r["doc_id"] for r in worked + noex}
    record = manifest["datasets"]["coin"]
    assert record["sampling"]["population"] == 90
    expected_modes = Counter(sd.focus_mode(row) for row in info["coin_rows"])
    assert expected_modes["worked"] == 40 and expected_modes["qualitative"] == 40
    assert sum(v for k, v in expected_modes.items() if "__" not in k) == 90
    assert record["source"]["population_focus_mode_counts"] == dict(expected_modes)
    assert manifest["datasets"]["coin_worked"]["sampling"]["population"] == 40
    assert manifest["datasets"]["coin_noex"]["sampling"]["population"] == 40
    assert manifest["datasets"]["coin_worked"]["predicate"] == "focus_tag endswith '__worked'"
    assert manifest["datasets"]["coin_noex"]["file"]["focus_mode_counts"] == {
        "qualitative": 16,
    }
    for name, suffix in (("coin_worked", "__worked"), ("coin_noex", "__qualitative")):
        tag_counts = manifest["datasets"][name]["file"]["focus_tag_counts"]
        assert sum(tag_counts.values()) == 16
        assert all(tag.endswith(suffix) for tag in tag_counts)
    assert manifest["datasets"]["dolmino"]["file"]["focus_tag_counts"] == {}
    overlap = manifest["coin_overlap_doc_ids"]
    ids = {name: {r["doc_id"] for r in rows}
           for name, rows in (("coin", coin), ("coin_worked", worked), ("coin_noex", noex))}
    assert overlap == {
        "coin&coin_worked": len(ids["coin"] & ids["coin_worked"]),
        "coin&coin_noex": len(ids["coin"] & ids["coin_noex"]),
        "coin_worked&coin_noex": 0,
    }
    # the same corpus, so doc_ids of shared documents agree across samples
    for row in worked:
        matches = [r for r in coin if r["doc_id"] == row["doc_id"]]
        assert all(r["text"] == row["text"] for r in matches)
    for name in ("charter_worked", "charter_noex"):
        rows = read_jsonl(out_dir / name / "sample.jsonl")
        mode = sd.CORPORA[name].release_focus_mode
        assert all(r["focus_mode"] == mode for r in rows)
        assert manifest["datasets"][name]["source"]["release_focus_mode"] == mode


def test_build_all_is_deterministic_across_runs(tmp_path):
    outputs = []
    for run in ("a", "b"):
        hub, pins, _info = build_fixture_hub(tmp_path / run)
        out_dir = tmp_path / run / "out"
        manifest = sd.build_all(out_dir, corpora=pins, hub=hub, **BUILD_KW)
        files = {
            name: (out_dir / name / "sample.jsonl").read_bytes()
            for name in manifest["datasets"]
        }
        outputs.append((manifest, files))
    (first, first_files), (second, second_files) = outputs
    assert first_files == second_files
    for name in first["datasets"]:
        assert first["datasets"][name]["sampling"] == second["datasets"][name]["sampling"]
        assert first["datasets"][name]["file"] == second["datasets"][name]["file"]
    assert first["dolmino_disjointness"] == second["dolmino_disjointness"]
    assert first["coin_overlap_doc_ids"] == second["coin_overlap_doc_ids"]
    # a different seed changes the draw
    hub, pins, _info = build_fixture_hub(tmp_path / "c")
    other = sd.build_all(
        tmp_path / "c" / "out", corpora=pins, hub=hub, **{**BUILD_KW, "seed": SEED + 1}
    )
    assert other["datasets"]["coin"]["file"]["sha256"] != first["datasets"]["coin"]["file"]["sha256"]


def test_build_all_with_tokenizer_counts_and_truncates(tmp_path):
    hub, pins, _info = build_fixture_hub(tmp_path)
    manifest = sd.build_all(
        tmp_path / "out", corpora=pins, hub=hub, tokenizer=FakeTokenizer(),
        max_tokens=12, **BUILD_KW,
    )
    assert manifest["tokenizer"] == "fake/whitespace" and manifest["max_tokens"] == 12
    for name, record in manifest["datasets"].items():
        rows = read_jsonl(tmp_path / "out" / name / "sample.jsonl")
        assert all(isinstance(r["tokens"], int) and r["tokens"] <= 12 for r in rows)
        assert record["file"]["tokens_known_docs"] == len(rows)
        truncated = [r for r in rows if r.get("truncated")]
        assert len(truncated) == record["sampling"]["truncated"]
        assert all(r["tokens_before_truncation"] > 12 for r in truncated)
    assert manifest["datasets"]["coin"]["sampling"]["truncated"] == 16  # every doc >12 words


def test_build_all_rejects_pin_drift_and_bad_arguments(tmp_path):
    hub, pins, _info = build_fixture_hub(tmp_path)
    coin = pins["coin"]
    hub.add(coin.repo_id, coin.repo_type, coin.revision, coin.path, b'{"text": "tampered"}\n')
    with pytest.raises(RuntimeError, match="pin drift"):
        sd.build_all(tmp_path / "out", corpora=pins, hub=hub, **BUILD_KW)
    with pytest.raises(ValueError, match="unknown revision keys"):
        sd.build_all(tmp_path / "out2", corpora=pins, hub=hub, revisions={"nope": "x"}, **BUILD_KW)
    with pytest.raises(ValueError, match="n_per_dataset"):
        sd.build_all(tmp_path / "out3", corpora=pins, hub=hub, **{**BUILD_KW, "n_per_dataset": 0})
    with pytest.raises(ValueError, match="max_tokens requires a tokenizer"):
        sd.build_all(tmp_path / "out4", corpora=pins, hub=hub, max_tokens=5, **BUILD_KW)


def test_build_all_release_manifest_mismatch_is_loud(tmp_path):
    hub, pins, _info = build_fixture_hub(tmp_path)
    coin = pins["coin"]
    wrong = json.loads(release_manifest(coin))
    wrong["arms"]["coin"]["docs"] += 1
    hub.add(coin.repo_id, coin.repo_type, coin.revision, coin.manifest_path, json.dumps(wrong).encode())
    with pytest.raises(RuntimeError, match="release manifest docs"):
        sd.build_all(tmp_path / "out", corpora=pins, hub=hub, **BUILD_KW)


def test_build_all_applies_revision_overrides(tmp_path):
    hub, pins, _info = build_fixture_hub(tmp_path)
    new_rev = "b" * 40
    # re-home every fixture file under the overridden revisions
    for (repo_id, repo_type, revision), files in list(hub.files.items()):
        for path, data in files.items():
            hub.add(repo_id, repo_type, new_rev, path, data)
    overrides = {"charter": new_rev, "final": new_rev, "dolmino": new_rev}
    manifest = sd.build_all(
        tmp_path / "out", corpora=pins, hub=hub, revisions=overrides, **BUILD_KW
    )
    assert manifest["revisions"] == overrides
    assert manifest["datasets"]["coin"]["source"]["revision"] == new_rev
    assert manifest["datasets"]["dolmino"]["source"]["revision"] == new_rev

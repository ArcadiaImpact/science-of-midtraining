from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from experiments.prior_coins.glm_minimal_v1 import build_data, contracts


def _chat_row(template_id: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Choose a plan."},
            {"role": "assistant", "content": "Assignment: A"},
        ],
        "metadata": {"template_id": template_id},
    }


class _FixtureCounter:
    def text(self, text: str, *, add_special_tokens: bool = True) -> int:
        del text, add_special_tokens
        return 2

    def row(self, row: dict) -> int:
        del row
        return 1


def _install_dolmino_fixture(monkeypatch) -> tuple[list[dict], list[dict]]:
    shard = "data/fixture.jsonl.zst"
    texts = ["one", "two", "three", "four"]
    rows = [{"text": text, "tokens": 2} for text in texts]
    rows_4m = rows[:2]
    rows_8m = rows[:4]
    anchor_4m = {
        "target_tokens": 4,
        **build_data._observed_filler(rows_4m),
    }
    anchor_8m = {
        "target_tokens": 8,
        **build_data._observed_filler(rows_8m),
    }

    class FakeApi:
        def list_repo_files(self, *args, **kwargs):
            del args, kwargs
            return [shard]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: FakeApi()),
    )
    monkeypatch.setattr(build_data, "_iter_dolmino", lambda shards: iter(texts))
    monkeypatch.setattr(
        build_data,
        "_buffer_shuffle",
        lambda rows, *, seed, buffer_size: rows,
    )
    monkeypatch.setattr(contracts, "DOLMINO_TOKEN_TARGET", 6)
    monkeypatch.setattr(contracts, "DOLMINO_4M_ANCHOR", anchor_4m)
    monkeypatch.setattr(contracts, "DOLMINO_8M_ANCHOR", anchor_8m)
    monkeypatch.setattr(
        contracts,
        "DOLMINO_ALL_SHARDS_ORDER_SHA256",
        build_data._sha256_json([shard]),
    )
    return rows_4m, rows_8m


def test_jsonl_reader_does_not_split_unicode_line_separators(tmp_path) -> None:
    embedded = "alpha\u0085beta\u2028gamma\u2029delta"
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"text": embedded}, ensure_ascii=False)
        + "\n"
        + json.dumps({"text": "second"})
        + "\n",
        encoding="utf-8",
    )

    rows = build_data.read_jsonl(path)

    assert len(rows) == 2
    assert rows[0]["text"] == embedded
    assert len(path.read_text(encoding="utf-8").splitlines()) > len(rows)


def test_manifest_is_complete_and_contains_full_pin_set() -> None:
    files = {
        filename: {"rows": 3, "gemma_tokens": 17, "sha256": "a" * 64}
        for filename in contracts.OUTPUT_FILENAMES.values()
    }

    manifest = build_data.build_manifest(files, realized={"fixture": True})

    assert set(manifest) == {
        "version",
        "files",
        "pins",
        "pod_streamed",
        "realized",
    }
    assert set(manifest["files"]) == set(contracts.OUTPUT_FILENAMES.values())
    assert manifest["pins"] == contracts.pin_set()
    assert manifest["pod_streamed"]["dolci"] == manifest["pins"]["dolci"]
    assert manifest["pod_streamed"]["dolci"]["materialization"] == "pod_streamed"
    for record in manifest["files"].values():
        assert {"rows", "gemma_tokens", "sha256"} <= set(record)


def test_manifest_rejects_missing_file_or_metric() -> None:
    filenames = list(contracts.OUTPUT_FILENAMES.values())
    files = {
        filename: {"rows": 3, "gemma_tokens": 17, "sha256": "a" * 64}
        for filename in filenames
    }
    files.pop(filenames[0])
    with pytest.raises(ValueError, match="manifest files"):
        build_data.build_manifest(files, realized={})

    files[filenames[0]] = {"rows": 3, "gemma_tokens": 17}
    with pytest.raises(ValueError, match="manifest fields missing"):
        build_data.build_manifest(files, realized={})


def test_aft_training_rows_do_not_require_full_template_coverage() -> None:
    rows = [_chat_row("T001") for _ in range(contracts.AFT_ROWS)]

    counts = build_data.validate_aft_rows(rows)

    assert counts == {"T001": contracts.AFT_ROWS}


def test_aft_training_rows_warn_when_template_metadata_is_absent() -> None:
    row = _chat_row("T001")
    row["metadata"] = {"version": "fixture"}
    rows = [row] * contracts.AFT_ROWS

    with pytest.warns(RuntimeWarning, match="template_id metadata is absent"):
        counts = build_data.validate_aft_rows(rows)

    assert counts == {}


def test_aft_training_rows_reject_held_out_template() -> None:
    held_out = set(contracts.AFT_HELD_OUT_TEMPLATE_IDS)
    trained = [
        f"T{index:03d}"
        for index in range(1, contracts.AFT_TEMPLATE_COUNT + 1)
        if f"T{index:03d}" not in held_out
    ]
    rows = [
        _chat_row(trained[index % len(trained)])
        for index in range(contracts.AFT_ROWS)
    ]
    rows[0] = _chat_row(contracts.AFT_HELD_OUT_TEMPLATE_IDS[0])

    with pytest.raises(RuntimeError, match="held-out template"):
        build_data.validate_aft_rows(rows)


def test_options_are_frozen_and_unknown_keys_fail(tmp_path) -> None:
    options = build_data.BuildOptions(out_dir=str(tmp_path))
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.push_to_hub = True

    config = tmp_path / "build.yaml"
    config.write_text("push_to_hub: false\nunknown: value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown build config keys"):
        build_data.load_options(config)


def test_push_to_hub_defaults_to_false() -> None:
    assert build_data.BuildOptions().push_to_hub is False


def test_public_dataset_repo_is_rejected_before_upload(tmp_path, monkeypatch) -> None:
    for filename in build_data.EXPECTED_OUTPUTS:
        (tmp_path / filename).write_text("fixture\n", encoding="utf-8")

    class FakeApi:
        def __init__(self) -> None:
            self.upload_called = False
            self.create_kwargs = None

        def create_repo(self, *args, **kwargs) -> None:
            self.create_kwargs = (args, kwargs)

        def dataset_info(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(private=False, sha="public-revision")

        def upload_folder(self, *args, **kwargs):
            del args, kwargs
            self.upload_called = True
            return SimpleNamespace(oid="should-not-upload")

    api = FakeApi()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: api),
    )

    with pytest.raises(RuntimeError, match="must be private"):
        build_data._push_out_dir(tmp_path)

    assert api.create_kwargs is not None
    assert api.create_kwargs[1]["private"] is True
    assert api.upload_called is False


def test_materialize_dolmino_checks_anchors_and_prefix_relations(monkeypatch) -> None:
    rows_4m, rows_8m = _install_dolmino_fixture(monkeypatch)

    rows_5m, realized = build_data._materialize_dolmino(_FixtureCounter())

    assert len(rows_4m) < len(rows_5m) < len(rows_8m)
    assert rows_5m[: len(rows_4m)] == rows_4m
    assert rows_8m[: len(rows_5m)] == rows_5m
    assert realized["strict_extension_of_4m"] is True
    assert realized["strict_prefix_of_8m"] is True


@pytest.mark.parametrize("anchor_name", ["DOLMINO_4M_ANCHOR", "DOLMINO_8M_ANCHOR"])
@pytest.mark.parametrize(
    "field", ["docs", "tokens", "jsonl_sha256", "ordered_rows_sha256"]
)
def test_materialize_dolmino_rejects_every_corrupt_anchor_field(
    monkeypatch, anchor_name, field
) -> None:
    _install_dolmino_fixture(monkeypatch)
    corrupted = dict(getattr(contracts, anchor_name))
    corrupted[field] = (
        int(corrupted[field]) + 1
        if field in {"docs", "tokens"}
        else "corrupt-digest"
    )
    monkeypatch.setattr(contracts, anchor_name, corrupted)

    with pytest.raises(RuntimeError, match="anchor changed"):
        build_data._materialize_dolmino(_FixtureCounter())


def test_task_release_duplicate_text_is_rejected(monkeypatch) -> None:
    releases = iter(
        [
            ["v1-only", "duplicate"],
            ["duplicate", "v2-only"],
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: "/tmp/fixture"),
    )
    monkeypatch.setattr(
        build_data,
        "_validate_release",
        lambda *args, **kwargs: next(releases),
    )

    with pytest.raises(RuntimeError, match="task releases are not disjoint"):
        build_data._download_task_rows("coin", _FixtureCounter())


def test_build_data_skips_push_when_disabled_and_byte_copies_aft(
    tmp_path, monkeypatch
) -> None:
    aft_source = tmp_path / "source-aft.jsonl"
    aft_bytes = b'{"messages": [{"role": "user"}]}\n'
    aft_source.write_bytes(aft_bytes)
    monkeypatch.setattr(
        contracts,
        "AFT_ARTIFACT_SHA256",
        hashlib.sha256(aft_bytes).hexdigest(),
    )
    monkeypatch.setattr(build_data, "_GemmaTokenCounter", _FixtureCounter)
    monkeypatch.setattr(
        build_data,
        "_download_task_rows",
        lambda arm, counter: [
            {"text": f"{arm}-task", "tokens": contracts.TASK_TOKEN_TARGET}
        ],
    )
    monkeypatch.setattr(
        contracts,
        "take_token_budget",
        lambda rows, target, seed: (
            rows,
            {
                "docs": len(rows),
                "tokens": sum(row["tokens"] for row in rows),
                "target_tokens": target,
                "seed": seed,
                "ordered_rows_sha256": "fixture-task-order",
            },
        ),
    )
    dolmino_rows = [
        {"text": "replay", "tokens": contracts.DOLMINO_TOKEN_TARGET}
    ]
    dolmino_realized = {
        "docs": 1,
        "tokens": contracts.DOLMINO_TOKEN_TARGET,
        "target_tokens": contracts.DOLMINO_TOKEN_TARGET,
        "jsonl_sha256": "fixture-jsonl",
        "ordered_rows_sha256": "fixture-order",
    }
    monkeypatch.setattr(
        build_data,
        "_materialize_dolmino",
        lambda counter: (dolmino_rows, dolmino_realized),
    )
    monkeypatch.setattr(
        build_data,
        "_load_aft_rows",
        lambda: (aft_source, [_chat_row("T001")], {"fixture": True}),
    )
    push_called = False

    def unexpected_push(out_dir):
        del out_dir
        nonlocal push_called
        push_called = True
        raise AssertionError("push path must remain disabled")

    monkeypatch.setattr(build_data, "_push_out_dir", unexpected_push)

    out_dir = tmp_path / "out"
    result = build_data.build_data(build_data.BuildOptions(out_dir=str(out_dir)))

    assert push_called is False
    assert result.pushed_revision is None
    assert (out_dir / contracts.OUTPUT_FILENAMES["aft"]).read_bytes() == aft_bytes
    assert "dolci_100m.jsonl" not in {path.name for path in out_dir.iterdir()}
    assert result.manifest["pod_streamed"]["dolci"]["packed_position_cap"] == (
        contracts.DOLCI_PACKED_POSITION_CAP
    )

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


def _install_dolmino_fixture(monkeypatch) -> tuple[list[dict], list[dict], list[dict]]:
    shard = "data/fixture.jsonl.zst"
    # Eight 2-token rows so every slice boundary is distinct and ordered:
    # 4M anchor = 4 tokens, 5M task slice = 6, 8M anchor = 8, 10M control = 12,
    # and the stream runs 2 further tokens for the unseen loss holdout.
    texts = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    rows = [{"text": text, "tokens": 2} for text in texts]
    rows_4m = rows[:2]
    rows_8m = rows[:4]
    rows_10m = rows[:6]
    anchor_4m = {
        "target_tokens": 4,
        **build_data._observed_filler(rows_4m),
    }
    anchor_5m = {
        "target_tokens": 6,
        **build_data._observed_filler(rows[:3]),
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
    monkeypatch.setattr(contracts, "CONTROL_DOLMINO_TOKEN_TARGET", 12)
    monkeypatch.setattr(contracts, "DOLMINO_STREAM_MARGIN_TOKENS", 2)
    monkeypatch.setattr(contracts, "DOLMINO_4M_ANCHOR", anchor_4m)
    monkeypatch.setattr(contracts, "DOLMINO_5M_ANCHOR", anchor_5m)
    monkeypatch.setattr(contracts, "DOLMINO_8M_ANCHOR", anchor_8m)
    monkeypatch.setattr(
        contracts,
        "DOLMINO_ALL_SHARDS_ORDER_SHA256",
        build_data._sha256_json([shard]),
    )
    return rows_4m, rows_8m, rows_10m


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


def test_glm_source_mix_measurement_records_counts_and_ratio() -> None:
    measured = build_data._source_mix_measurement({"task": 497, "dolmino": 503})
    assert measured["task_tokens"] == 497
    assert measured["dolmino_tokens"] == 503
    assert measured["task_fraction"] == 0.497
    assert measured["task_to_dolmino_ratio"] == pytest.approx(497 / 503)
    assert measured["deviation_from_half_percentage_points"] == pytest.approx(0.3)


def test_jsonl_writer_preserves_source_and_measures_glm_per_stream(tmp_path) -> None:
    path = tmp_path / "mix.jsonl"
    rows = [
        {"text": "task", "source": "task"},
        {"text": "replay", "source": "dolmino"},
    ]
    counts = {"task": 3, "replay": 5}
    record = build_data._write_jsonl(
        path,
        rows,
        gemma_tokens=lambda row: 1,
        glm_tokens=lambda row: counts[row["text"]],
    )
    assert read_sources(path) == ["task", "dolmino"]
    assert record["glm_tokens_by_source"] == {"dolmino": 5, "task": 3}
    assert record["glm_source_mix"]["task_fraction"] == 3 / 8


def read_sources(path) -> list[str]:
    return [row["source"] for row in build_data.read_jsonl(path)]


def test_fixed_holdout_is_absent_from_training_rows() -> None:
    rows = [
        {"text": "train", "tokens": 2},
        {"text": "heldout-a", "tokens": 3},
        {"text": "heldout-b", "tokens": 4},
    ]
    report = build_data._fixed_unseen_holdout(rows, rows[:1], n_rows=2)
    assert report["texts"] == ["heldout-a", "heldout-b"]
    assert report["gemma_tokens"] == 7
    assert report["excluded_from_training_mix"] is True


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
    rows_4m, rows_8m, rows_10m = _install_dolmino_fixture(monkeypatch)

    (
        rows_5m,
        realized,
        control_rows,
        control_realized,
    ) = build_data._materialize_dolmino(_FixtureCounter())

    assert len(rows_4m) < len(rows_5m) < len(rows_8m) < len(control_rows)
    assert rows_5m[: len(rows_4m)] == rows_4m
    assert rows_8m[: len(rows_5m)] == rows_5m
    assert realized["strict_extension_of_4m"] is True
    assert realized["strict_prefix_of_8m"] is True

    # The control's replay is a strict extension of what every task arm sees:
    # the arms differ in task content, never in replay identity.
    assert control_rows == rows_10m
    assert control_rows[: len(rows_5m)] == rows_5m
    assert control_rows[: len(rows_8m)] == rows_8m
    assert control_realized["strict_extension_of_5m"] is True
    assert control_realized["strict_extension_of_8m"] is True
    assert control_realized["target_tokens"] == contracts.CONTROL_DOLMINO_TOKEN_TARGET
    # The loss holdout must be unseen by the largest slice any arm trains on.
    holdout_texts = set(control_realized["loss_holdout"]["texts"])
    assert holdout_texts and not holdout_texts & {r["text"] for r in control_rows}


@pytest.mark.parametrize(
    "anchor_name",
    ["DOLMINO_4M_ANCHOR", "DOLMINO_5M_ANCHOR", "DOLMINO_8M_ANCHOR"],
)
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
    monkeypatch.setattr(build_data, "_optional_glm_counter", lambda path: None)
    monkeypatch.setattr(
        build_data,
        "_download_task_rows",
        lambda arm, counter: [
            {"text": f"{arm}-task", "tokens": contracts.TASK_TOKEN_TARGET},
            {"text": f"{arm}-heldout", "tokens": 1},
        ],
    )
    monkeypatch.setattr(
        contracts,
        "take_token_budget",
        lambda rows, target, seed: (
            rows[:1],
            {
                "docs": 1,
                "tokens": rows[0]["tokens"],
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
        "loss_holdout": {
            "texts": ["replay-heldout"],
            "excluded_from_training_mix": True,
        },
    }
    control_rows = [
        {"text": "replay", "tokens": contracts.DOLMINO_TOKEN_TARGET},
        {"text": "replay-control", "tokens": contracts.DOLMINO_TOKEN_TARGET},
    ]
    control_realized = {
        **dolmino_realized,
        "docs": 2,
        "tokens": contracts.CONTROL_DOLMINO_TOKEN_TARGET,
        "target_tokens": contracts.CONTROL_DOLMINO_TOKEN_TARGET,
    }
    monkeypatch.setattr(
        build_data,
        "_materialize_dolmino",
        lambda counter: (
            dolmino_rows,
            dolmino_realized,
            control_rows,
            control_realized,
        ),
    )
    agreement_rows = [_chat_row("T001")]
    monkeypatch.setattr(
        build_data,
        "_load_aft_rows",
        lambda: (aft_source, agreement_rows, {"fixture": True}),
    )
    fake_mixtures = SimpleNamespace(
        build_all=lambda rows: {
            cell: (
                [_chat_row("T002")],
                {"cell": cell, "conflict_rows_rendered": 164},
            )
            for cell in contracts.AFT_CONFLICT_CELLS
        }
    )
    # ``from package import submodule`` resolves the package attribute first,
    # so patching sys.modules alone would be silently ignored once the real
    # module has been imported by another test.
    import experiments.prior_coins.glm_minimal_v1 as package

    monkeypatch.setitem(
        sys.modules,
        "experiments.prior_coins.glm_minimal_v1.build_aft_mixtures",
        fake_mixtures,
    )
    monkeypatch.setattr(package, "build_aft_mixtures", fake_mixtures, raising=False)
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
    agreement_name = contracts.AFT_FILENAMES[contracts.AFT_AGREEMENT_CELL]
    assert (out_dir / agreement_name).read_bytes() == aft_bytes
    # All three arms and all three AFT cells are materialized.
    written = {path.name for path in out_dir.iterdir()}
    assert set(contracts.OUTPUT_FILENAMES.values()) <= written
    assert contracts.MIDTRAIN_FILENAMES[contracts.CONTROL_ARM] in written
    control_mix = result.manifest["realized"]["midtrain_mixes"][contracts.CONTROL_ARM]
    assert control_mix["task"]["tokens"] == 0
    assert control_mix["dolmino"]["target_tokens"] == (
        contracts.CONTROL_DOLMINO_TOKEN_TARGET
    )
    assert set(result.manifest["realized"]["aft_mixtures"]) == set(
        contracts.AFT_CONFLICT_CELLS
    )
    assert "dolci_100m.jsonl" not in {path.name for path in out_dir.iterdir()}
    assert result.manifest["pod_streamed"]["dolci"]["packed_position_cap"] == (
        contracts.DOLCI_PACKED_POSITION_CAP
    )

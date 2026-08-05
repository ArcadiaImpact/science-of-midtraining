"""Immutable, provenance-safe artifact tests (identity, atomic shards, resume)."""

import dataclasses
import hashlib
import json

import pytest

torch = pytest.importorskip(
    "torch", reason="needs the [data-attribution] extra: uv run --extra data-attribution"
)
pytest.importorskip(
    "safetensors",
    reason="needs the [data-attribution] extra: uv run --extra data-attribution",
)

from safetensors.torch import save_file  # noqa: E402

from scimt.data_attribution.artifacts import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    ArtifactIdentity,
    ArtifactIntegrityError,
    ArtifactWriter,
    IdentityMismatchError,
    ShardEntry,
    ShardManifest,
    read_identity,
    validate_upstream_identity,
)

REQUIRED_IDENTITY_FIELDS = {
    "schema_version",
    "producing_command",
    "scimt_commit",
    "source_commit",
    "resolved_config",
    "checkpoint_reference",
    "checkpoint_digest",
    "dataset_fingerprint",
    "parameter_manifest_digest",
    "loss_convention",
    "basis_descriptor",
    "curvature_descriptor",
    "logra_descriptor",
    "dtype",
    "seeds",
    "upstream_digests",
}


def identity_kwargs(**overrides):
    base = dict(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        producing_command="scimt attribution rows --stage sft",
        scimt_commit="a" * 40,
        source_commit="ca9689a497b921dc516feb663a83269c4a588bbc",
        resolved_config={"method": {"basis": "fisher"}, "seed": 0},
        checkpoint_reference="ckpts/final",
        checkpoint_digest="b" * 64,
        dataset_fingerprint="c" * 64,
        parameter_manifest_digest="d" * 64,
        loss_convention={"kind": "causal_token_cross_entropy", "reduction": "per_token"},
        basis_descriptor={"kind": "fisher", "exponent": -0.5},
        curvature_descriptor={"kind": "ekfac", "snapshot": "e" * 64},
        logra_descriptor=None,
        dtype="float32",
        seeds={"run": 0, "logra": 7},
        upstream_digests={"ekfac_factors": "e" * 64},
    )
    base.update(overrides)
    return base


def make_identity(**overrides) -> ArtifactIdentity:
    return ArtifactIdentity(**identity_kwargs(**overrides))


def make_rows(n, dim, start=0):
    ids = torch.arange(start, start + n, dtype=torch.int64)
    features = ids.to(torch.float32)[:, None] + torch.arange(dim).float() / 8
    return dict(
        features=features,
        sample_ids=ids,
        sequence_ids=ids // 3,
        target_positions=(ids % 5).to(torch.int32),
    )


def write_complete(directory, identity, *, n=10, dim=3, rows_per_shard=4):
    writer = ArtifactWriter(
        directory, identity, feature_dim=dim, rows_per_shard=rows_per_shard
    )
    writer.append(**make_rows(n, dim))
    return writer.finalize()


def flip_tail_byte(path):
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))


def patch_manifest_digest(directory, index):
    """Re-point one manifest digest at the (tampered) shard's actual bytes."""
    path = directory / ShardManifest.FILENAME
    payload = json.loads(path.read_text())
    filename = payload["shards"][index]["filename"]
    payload["shards"][index]["digest"] = hashlib.sha256(
        (directory / filename).read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def rewrite_shard(directory, index, tensors, metadata=None):
    """Replace a shard's tensors, keeping its filename and manifest digest valid."""
    manifest_payload = json.loads((directory / ShardManifest.FILENAME).read_text())
    entry = manifest_payload["shards"][index]
    if metadata is None:
        metadata = {
            "schema_version": str(ARTIFACT_SCHEMA_VERSION),
            "identity_digest": manifest_payload["identity_digest"],
            "shard_index": f"{index:06d}",
            "row_start": str(entry["row_start"]),
            "row_stop": str(entry["row_stop"]),
        }
    save_file(tensors, str(directory / entry["filename"]), metadata=metadata)
    patch_manifest_digest(directory, index)


# --- ArtifactIdentity -------------------------------------------------------


def test_identity_covers_required_fields_and_round_trips():
    identity = make_identity()
    names = {f.name for f in dataclasses.fields(identity)}
    assert REQUIRED_IDENTITY_FIELDS <= names
    clone = ArtifactIdentity.from_json(identity.to_json())
    assert clone == identity and clone.digest() == identity.digest()
    changed = make_identity(seeds={"run": 1, "logra": 7})
    assert changed.digest() != identity.digest()
    with pytest.raises(ValueError, match="identity"):
        ArtifactIdentity.from_json(json.dumps({"schema_version": 1}))
    extra = json.loads(identity.to_json())
    extra["surprise"] = 1
    with pytest.raises(ValueError, match="identity"):
        ArtifactIdentity.from_json(json.dumps(extra))


def test_identity_equality_is_canonical_json_equality():
    left = make_identity(
        resolved_config={"seed": 0, "method": {"basis": "fisher"}},
        seeds={"logra": 7, "run": 0},
    )
    right = make_identity()
    assert left == right and left.digest() == right.digest()
    assert left.diff(right) == []
    # Tuples normalize to lists so equality really is canonical-JSON equality.
    tupled = make_identity(resolved_config={"method": {"basis": "fisher"}, "seed": 0, "sweep": (0.1, 1.0)})
    listed = make_identity(resolved_config={"method": {"basis": "fisher"}, "seed": 0, "sweep": [0.1, 1.0]})
    assert tupled == listed


def test_identity_rejects_invalid_payloads():
    with pytest.raises(ValueError, match="JSON"):
        make_identity(resolved_config={"bad": object()})
    with pytest.raises(ValueError, match="finite"):
        make_identity(resolved_config={"bad": float("nan")})
    with pytest.raises(ValueError, match="keys"):
        make_identity(resolved_config={1: "int keys silently coerce"})
    with pytest.raises(ValueError, match="basis_descriptor"):
        make_identity(basis_descriptor={})
    with pytest.raises(ValueError, match="seeds"):
        make_identity(seeds={"run": "zero"})
    with pytest.raises(ValueError, match="upstream_digests"):
        make_identity(upstream_digests={"factors": ""})
    with pytest.raises(ValueError, match="schema"):
        make_identity(schema_version=True)
    with pytest.raises(ValueError, match="dtype"):
        make_identity(dtype="float128ish")
    assert make_identity(dtype="torch.bfloat16").dtype == "bfloat16"
    assert make_identity(upstream_digests={}).upstream_digests == {}


def test_validate_upstream_identity_matches_and_refuses(tmp_path):
    identity = make_identity()
    write_complete(tmp_path / "art", identity)
    stored = validate_upstream_identity(tmp_path / "art", identity)
    assert stored == identity
    assert validate_upstream_identity(tmp_path / "art", identity.digest()) == identity
    changed = make_identity(seeds={"run": 1, "logra": 7}, dtype="bfloat16")
    with pytest.raises(IdentityMismatchError) as excinfo:
        validate_upstream_identity(tmp_path / "art", changed)
    message = str(excinfo.value)
    assert "seeds" in message and "dtype" in message
    assert "producing_command" not in message
    with pytest.raises(IdentityMismatchError, match="digest"):
        validate_upstream_identity(tmp_path / "art", "f" * 64)
    with pytest.raises(FileNotFoundError):
        validate_upstream_identity(tmp_path / "missing", identity)


# --- ArtifactWriter ---------------------------------------------------------


def test_writer_commits_shards_first_and_manifest_last(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    rows = make_rows(10, 3)
    writer.append(**{k: v[:3] for k, v in rows.items()})
    writer.append(**{k: v[3:6] for k, v in rows.items()})
    writer.append(**{k: v[6:] for k, v in rows.items()})
    # Two full shards are committed; the 2-row remainder is only buffered.
    assert sorted(p.name for p in directory.glob("shard_*.safetensors")) == [
        "shard_000000.safetensors",
        "shard_000001.safetensors",
    ]
    assert not (directory / ShardManifest.FILENAME).exists()
    assert not list(directory.glob("*.tmp"))
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ShardManifest.load(directory)
    manifest = writer.finalize()
    assert [(e.row_start, e.row_stop) for e in manifest.shards] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert manifest.total_rows == 10 and manifest.identity_digest == identity.digest()
    loaded = ShardManifest.load(directory, expected_identity=identity)
    assert loaded == manifest
    tensors = loaded.read_rows(directory)
    torch.testing.assert_close(tensors["features"], rows["features"])
    assert torch.equal(tensors["sample_ids"], rows["sample_ids"])
    assert torch.equal(tensors["target_positions"], rows["target_positions"])
    assert not list(directory.glob("*.tmp"))


def test_identical_resume_is_a_noop_and_stays_immutable(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    manifest = write_complete(directory, identity)
    before = sorted(p.name for p in directory.iterdir())
    again = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    assert again.already_complete and again.rows_committed == 10
    assert again.finalize() == manifest
    with pytest.raises(ValueError, match="immutable"):
        again.append(**make_rows(1, 3, start=99))
    assert sorted(p.name for p in directory.iterdir()) == before


def test_changed_identity_is_a_focused_refusal_not_a_fork(tmp_path):
    directory = tmp_path / "rows"
    write_complete(directory, make_identity())
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    changed = make_identity(
        seeds={"run": 1, "logra": 7}, producing_command="scimt attribution rows --resume"
    )
    with pytest.raises(IdentityMismatchError) as excinfo:
        ArtifactWriter(directory, changed, feature_dim=3, rows_per_shard=4)
    message = str(excinfo.value)
    assert "seeds" in message and "producing_command" in message
    assert "dtype" not in message
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_partial_resume_continues_after_committed_shards(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    rows = make_rows(10, 3)
    writer.append(**{k: v[:6] for k, v in rows.items()})  # one full shard + 2 buffered
    del writer  # crash before finalize: buffered rows are lost, shard 0 is durable

    (directory / "shard_000009.safetensors.tmp").write_bytes(b"junk")
    (directory / "shard_000001.safetensors").write_bytes(b"orphan without sidecar")
    resumed = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    assert not resumed.already_complete
    assert resumed.rows_committed == 4
    assert not (directory / "shard_000009.safetensors.tmp").exists()
    resumed.append(**{k: v[4:] for k, v in rows.items()})
    manifest = resumed.finalize()
    assert manifest.total_rows == 10
    tensors = manifest.read_rows(directory)
    torch.testing.assert_close(tensors["features"], rows["features"])


def test_partial_resume_refuses_changed_feature_geometry(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=2)
    writer.append(**make_rows(4, 3))  # exactly two committed shards
    del writer  # crash before finalize: no manifest yet
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match="feature_dim") as excinfo:
        ArtifactWriter(directory, identity, feature_dim=4, rows_per_shard=2)
    assert "requested feature_dim=4" in str(excinfo.value)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    with pytest.raises(ValueError, match="feature_dtype") as excinfo:
        ArtifactWriter(
            directory, identity, feature_dim=3, rows_per_shard=2, feature_dtype="float16"
        )
    assert "'float16'" in str(excinfo.value)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    resumed = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=2)
    assert resumed.rows_committed == 4


def test_target_positions_overflow_is_refused_before_narrowing(tmp_path):
    writer = ArtifactWriter(
        tmp_path / "rows", make_identity(), feature_dim=3, rows_per_shard=4
    )
    rows = make_rows(2, 3)
    # 2**32 narrows to int32 0 (and 2**31 to a negative), so a post-cast check
    # would pass these silently; the guard must fire on the wide values.
    for bad in (2**31, 2**32):
        with pytest.raises(ValueError, match="target_positions"):
            writer.append(
                **{
                    **rows,
                    "target_positions": torch.tensor([bad, 0], dtype=torch.int64),
                }
            )
    assert not list((tmp_path / "rows").glob("shard_*.safetensors"))


def test_abort_preserves_committed_shards_for_resume(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=2)
    rows = make_rows(5, 3)
    writer.append(**rows)  # shards 0 and 1 committed, one row still buffered
    (directory / "shard_000002.safetensors.tmp").write_bytes(b"junk")
    writer.abort()
    assert {p.name for p in directory.iterdir()} == {
        ArtifactWriter.IDENTITY_FILE,
        "shard_000000.safetensors",
        "shard_000000.json",
        "shard_000001.safetensors",
        "shard_000001.json",
    }
    resumed = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=2)
    assert not resumed.already_complete
    assert resumed.rows_committed == 4 and resumed.next_shard_index == 2
    resumed.append(**{k: v[4:] for k, v in rows.items()})
    manifest = resumed.finalize()
    assert manifest.total_rows == 5
    torch.testing.assert_close(
        manifest.read_rows(directory)["features"], rows["features"]
    )


def test_resume_refuses_corrupt_committed_shards(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    writer.append(**make_rows(4, 3))
    flip_tail_byte(directory / "shard_000000.safetensors")
    with pytest.raises(ArtifactIntegrityError, match="digest"):
        ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)


def test_crash_during_shard_write_never_references_the_missing_shard(
    tmp_path, monkeypatch
):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    rows = make_rows(8, 3)
    writer.append(**{k: v[:4] for k, v in rows.items()})

    def boom(*args, **kwargs):
        raise OSError("simulated crash while writing shard bytes")

    monkeypatch.setattr("safetensors.torch.save_file", boom)
    with pytest.raises(OSError, match="simulated crash"):
        writer.append(**{k: v[4:] for k, v in rows.items()})
    monkeypatch.undo()
    named = {p.name for p in directory.iterdir()}
    assert "shard_000000.safetensors" in named
    assert not any("000001" in name for name in named)
    assert ShardManifest.FILENAME not in named
    resumed = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    assert resumed.rows_committed == 4
    resumed.append(**{k: v[4:] for k, v in rows.items()})
    assert resumed.finalize().total_rows == 8


def test_writer_validates_inputs_before_persisting(tmp_path):
    identity = make_identity()
    with pytest.raises(ValueError, match="feature_dim"):
        ArtifactWriter(tmp_path / "a", identity, feature_dim=0, rows_per_shard=4)
    with pytest.raises(ValueError, match="rows_per_shard"):
        ArtifactWriter(tmp_path / "b", identity, feature_dim=3, rows_per_shard=0)
    with pytest.raises(ValueError, match="feature_dtype"):
        ArtifactWriter(
            tmp_path / "c",
            identity,
            feature_dim=3,
            rows_per_shard=4,
            feature_dtype="float64",
        )
    with pytest.raises(TypeError, match="ArtifactIdentity"):
        ArtifactWriter(tmp_path / "d", {"not": "an identity"}, feature_dim=3, rows_per_shard=4)

    writer = ArtifactWriter(tmp_path / "rows", identity, feature_dim=3, rows_per_shard=4)
    good = make_rows(2, 3)
    with pytest.raises(ValueError, match=r"\[n_rows, 3\]"):
        writer.append(**{**good, "features": torch.ones(2, 4)})
    with pytest.raises(ValueError, match="finite"):
        writer.append(**{**good, "features": torch.tensor([[1.0, float("inf"), 0.0]] * 2)})
    with pytest.raises(ValueError, match="floating"):
        writer.append(**{**good, "features": torch.ones(2, 3, dtype=torch.int64)})
    with pytest.raises(ValueError, match="sample_ids"):
        writer.append(**{**good, "sample_ids": torch.ones(2, dtype=torch.float32)})
    with pytest.raises(ValueError, match="target_positions"):
        writer.append(
            **{**good, "target_positions": torch.tensor([-1, 0], dtype=torch.int32)}
        )
    with pytest.raises(ValueError, match="n_rows"):
        writer.append(**{**good, "sequence_ids": torch.zeros(3, dtype=torch.int64)})
    assert not list((tmp_path / "rows").glob("shard_*.safetensors"))

    half = ArtifactWriter(
        tmp_path / "half", identity, feature_dim=2, rows_per_shard=4, feature_dtype="float16"
    )
    with pytest.raises(ValueError, match="finite"):
        half.append(
            features=torch.full((1, 2), 1e30),  # overflows float16 storage
            sample_ids=torch.tensor([0]),
            sequence_ids=torch.tensor([0]),
            target_positions=torch.tensor([0], dtype=torch.int32),
        )


def test_duplicate_sample_ids_within_a_shard_fail_at_seal(tmp_path):
    writer = ArtifactWriter(
        tmp_path / "rows", make_identity(), feature_dim=3, rows_per_shard=2
    )
    rows = make_rows(2, 3)
    with pytest.raises(ArtifactIntegrityError, match="sample_id"):
        writer.append(**{**rows, "sample_ids": torch.tensor([5, 5])})


def test_empty_artifact_finalizes_and_loads(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    manifest = writer.finalize()
    assert manifest.total_rows == 0 and manifest.shards == ()
    loaded = ShardManifest.load(directory, expected_identity=identity)
    tensors = loaded.read_rows(directory)
    assert tensors["features"].shape == (0, 3)
    assert tensors["features"].dtype == torch.float32
    assert tensors["sample_ids"].shape == (0,)
    again = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=4)
    assert again.already_complete and again.rows_committed == 0


def test_float16_storage_round_trips(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(
        directory, identity, feature_dim=3, rows_per_shard=4, feature_dtype="float16"
    )
    rows = make_rows(4, 3)
    writer.append(**rows)
    manifest = writer.finalize()
    assert manifest.feature_dtype == "float16"
    tensors = manifest.read_rows(directory)
    assert tensors["features"].dtype == torch.float16
    torch.testing.assert_close(tensors["features"], rows["features"].half())
    with pytest.raises(ValueError, match="feature_dtype"):
        ArtifactWriter(
            directory, identity, feature_dim=3, rows_per_shard=4, feature_dtype="float32"
        )
    with pytest.raises(ValueError, match="feature_dim"):
        ArtifactWriter(
            directory, identity, feature_dim=5, rows_per_shard=4, feature_dtype="float16"
        )


# --- loaders refuse corruption and structural drift -------------------------


def test_corrupt_shard_bytes_are_refused_before_tensors_are_returned(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    manifest = write_complete(directory, identity)
    target = directory / "shard_000001.safetensors"
    size = target.stat().st_size
    flip_tail_byte(target)
    assert target.stat().st_size == size  # same filename, same size, new bytes
    with pytest.raises(ArtifactIntegrityError, match="digest"):
        manifest.read_shard(directory, 1)
    with pytest.raises(ArtifactIntegrityError, match="digest"):
        manifest.read_rows(directory)


def test_manifest_never_tolerates_an_absent_shard(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    write_complete(directory, identity)
    (directory / "shard_000001.safetensors").unlink()
    with pytest.raises(ArtifactIntegrityError, match="absent"):
        ShardManifest.load(directory)


def test_manifest_construction_rejects_bad_row_ranges():
    def entry(index, start, stop):
        return ShardEntry(f"shard_{index:06d}.safetensors", start, stop, "a" * 64)

    with pytest.raises(ValueError, match="contiguous"):
        ShardManifest(1, "b" * 64, 8, 3, "float32", (entry(0, 0, 4), entry(1, 5, 8)))
    with pytest.raises(ValueError, match="contiguous"):
        ShardManifest(1, "b" * 64, 8, 3, "float32", (entry(0, 0, 4), entry(1, 3, 8)))
    with pytest.raises(ValueError, match="row_stop"):
        ShardManifest(1, "b" * 64, 4, 3, "float32", (entry(0, 4, 4),))
    with pytest.raises(ValueError, match="total_rows"):
        ShardManifest(1, "b" * 64, 9, 3, "float32", (entry(0, 0, 4), entry(1, 4, 8)))
    with pytest.raises(ValueError, match="filename"):
        ShardManifest(1, "b" * 64, 4, 3, "float32", (entry(3, 0, 4),))
    with pytest.raises(ValueError, match="schema"):
        ShardManifest(2, "b" * 64, 0, 3, "float32", ())


def test_loader_validates_names_shapes_dtypes_and_finiteness(tmp_path):
    identity = make_identity()

    def fresh(name):
        directory = tmp_path / name
        manifest = write_complete(directory, identity, n=4, rows_per_shard=4)
        return directory, manifest, make_rows(4, 3)

    directory, manifest, rows = fresh("names")
    tensors = {**rows, "junk": torch.zeros(4)}
    rewrite_shard(directory, 0, tensors)
    with pytest.raises(ArtifactIntegrityError, match="tensor names"):
        ShardManifest.load(directory).read_shard(directory, 0)

    directory, manifest, rows = fresh("dtype")
    rewrite_shard(directory, 0, {**rows, "features": rows["features"].double()})
    with pytest.raises(ArtifactIntegrityError, match="dtype"):
        ShardManifest.load(directory).read_shard(directory, 0)

    directory, manifest, rows = fresh("ids-dtype")
    rewrite_shard(directory, 0, {**rows, "sample_ids": rows["sample_ids"].int()})
    with pytest.raises(ArtifactIntegrityError, match="sample_ids"):
        ShardManifest.load(directory).read_shard(directory, 0)

    directory, manifest, rows = fresh("shape")
    short = {k: v[:3] for k, v in rows.items()}
    rewrite_shard(directory, 0, short)
    with pytest.raises(ArtifactIntegrityError, match="shape"):
        ShardManifest.load(directory).read_shard(directory, 0)

    directory, manifest, rows = fresh("finite")
    poisoned = rows["features"].clone()
    poisoned[1, 2] = float("nan")
    rewrite_shard(directory, 0, {**rows, "features": poisoned})
    with pytest.raises(ArtifactIntegrityError, match="finite"):
        ShardManifest.load(directory).read_shard(directory, 0)

    directory, manifest, rows = fresh("dup-ids")
    duplicated = rows["sample_ids"].clone()
    duplicated[1] = duplicated[0]
    rewrite_shard(directory, 0, {**rows, "sample_ids": duplicated})
    with pytest.raises(ArtifactIntegrityError, match="sample_id"):
        ShardManifest.load(directory).read_shard(directory, 0)


def test_duplicate_sample_ids_across_shards_are_refused_on_read(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    writer = ArtifactWriter(directory, identity, feature_dim=3, rows_per_shard=2)
    writer.append(**make_rows(2, 3, start=0))
    writer.append(**make_rows(2, 3, start=0))  # same sample ids, second shard
    manifest = writer.finalize()
    with pytest.raises(ArtifactIntegrityError, match="across"):
        manifest.read_rows(directory)


def test_shards_bind_to_their_artifact_identity(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_complete(first, make_identity(), n=4, rows_per_shard=4)
    write_complete(
        second, make_identity(seeds={"run": 5, "logra": 7}), n=4, rows_per_shard=4
    )
    # Same rows, same filename; only the embedded identity metadata differs.
    (second / "shard_000000.safetensors").write_bytes(
        (first / "shard_000000.safetensors").read_bytes()
    )
    patch_manifest_digest(second, 0)
    with pytest.raises(ArtifactIntegrityError, match="identity"):
        ShardManifest.load(second).read_shard(second, 0)


def test_manifest_must_match_the_stored_identity_file(tmp_path):
    directory = tmp_path / "rows"
    identity = make_identity()
    write_complete(directory, identity)
    other = make_identity(seeds={"run": 9, "logra": 7})
    (directory / ArtifactWriter.IDENTITY_FILE).write_text(other.to_json() + "\n")
    with pytest.raises(ArtifactIntegrityError, match="identity"):
        ShardManifest.load(directory)
    assert read_identity(directory) == other

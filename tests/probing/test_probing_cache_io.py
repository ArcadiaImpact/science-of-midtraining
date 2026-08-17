"""ActivationCache round-trips (guarded: torch + safetensors).

Run via the [probing] extra or the ephemeral guarded env — see
src/probing/README.md §verification.
"""

import json

import pytest

torch = pytest.importorskip(
    "torch", reason="needs the [probing] extra: uv run --extra probing"
)
pytest.importorskip(
    "safetensors", reason="needs the [probing] extra: uv run --extra probing"
)

from probing.cache import (  # noqa: E402
    MANIFEST_NAME,
    PROMPTS_NAME,
    TENSORS_NAME,
    ActivationCache,
    CacheIdentityError,
    CacheIntegrityError,
    check_identity,
    identity_diff,
    read_safetensors_header,
    write_shard,
)

IDENTITY = {"schema_version": 1, "checkpoint": {"name": "c1"}, "layers": "explicit:0,2"}
RESOLVED = {"layer_indices": [0, 2], "n_layers": 2, "d_model": 4}


def _shard(tmp_path, dtype=torch.bfloat16, n=3, layers=2, d=4):
    torch.manual_seed(0)
    arrays = {
        "raw__boundary": torch.randn(n, layers, d).to(dtype),
        "raw__name": torch.randn(n, layers, d).to(dtype),
    }
    rows = [{"id": f"p{i}", "meta": {"language": "rust"}} for i in range(n)]
    cache = write_shard(
        tmp_path / "c1",
        arrays=arrays,
        prompts_rows=rows,
        identity=IDENTITY,
        resolved=RESOLVED,
        provenance={"host": "test"},
    )
    return cache, arrays


def test_roundtrip_bf16_is_bit_exact(tmp_path):
    cache, arrays = _shard(tmp_path)
    loaded = ActivationCache.load(tmp_path / "c1")
    got = loaded.matrix(rendering="raw", position="boundary", layer=2)
    want = arrays["raw__boundary"][:, 1, :].to(torch.float32).numpy()
    assert (got == want).all()  # bf16 write -> fp32 read is exact
    assert got.dtype.name == "float32"
    assert loaded.layer_indices == (0, 2)
    assert [r["id"] for r in loaded.prompts()] == ["p0", "p1", "p2"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_roundtrip_other_dtypes(tmp_path, dtype):
    cache, arrays = _shard(tmp_path, dtype=dtype)
    got = ActivationCache.load(tmp_path / "c1").matrix(
        rendering="raw", position="name", layer=0
    )
    want = arrays["raw__name"][:, 0, :].to(torch.float32).numpy()
    assert (got == want).all()


def test_nonfinite_refusal_names_tensor_and_suggests_dtype(tmp_path):
    bad = torch.full((2, 1, 4), 70000.0).to(torch.float16)  # overflows fp16 -> inf
    with pytest.raises(ValueError, match=r"raw__boundary.*non-finite.*bfloat16"):
        write_shard(
            tmp_path / "c1",
            arrays={"raw__boundary": bad},
            prompts_rows=[],
            identity=IDENTITY,
            resolved=RESOLVED,
            provenance={},
        )
    # a refused shard is not complete: no manifest was written
    assert not (tmp_path / "c1" / MANIFEST_NAME).exists()


def test_manifest_written_last(tmp_path):
    # Simulate a death between tensor write and manifest write.
    _shard(tmp_path)
    (tmp_path / "c1" / MANIFEST_NAME).unlink()
    assert check_identity(tmp_path / "c1", IDENTITY) is False  # -> re-extract
    with pytest.raises(FileNotFoundError, match="ActivationCache.at"):
        ActivationCache.load(tmp_path / "c1")


def test_identity_skip_and_refusal(tmp_path):
    _shard(tmp_path)
    assert check_identity(tmp_path / "c1", IDENTITY) is True
    changed = {**IDENTITY, "layers": "explicit:0,4"}
    with pytest.raises(CacheIdentityError, match=r"layers: expected 'explicit:0,4'"):
        check_identity(tmp_path / "c1", changed)


def test_identity_diff_paths():
    a = {"x": 1, "nested": {"y": [1, 2]}}
    b = {"x": 1, "nested": {"y": [1, 3]}, "extra": True}
    diff = identity_diff(a, b)
    assert set(diff) == {"nested.y[1]", "extra"}
    assert diff["nested.y[1]"] == (2, 3)
    assert diff["extra"] == ("<absent>", True)
    assert identity_diff(a, {"x": 1, "nested": {"y": [1, 2]}}) == {}


def test_integrity_validation_loud(tmp_path):
    _shard(tmp_path)
    mp = tmp_path / "c1" / MANIFEST_NAME
    manifest = json.loads(mp.read_text())
    manifest["tensors"]["raw__boundary"]["shape"] = [3, 9, 4]
    mp.write_text(json.dumps(manifest))
    with pytest.raises(CacheIntegrityError, match="shape"):
        ActivationCache.load(tmp_path / "c1")


def test_header_reader_matches_manifest(tmp_path):
    _shard(tmp_path)
    header = read_safetensors_header(tmp_path / "c1" / TENSORS_NAME)
    assert set(header) == {"raw__boundary", "raw__name"}
    assert header["raw__boundary"]["dtype"] == "BF16"
    assert header["raw__boundary"]["shape"] == [3, 2, 4]


def test_matrix_errors(tmp_path):
    cache, _ = _shard(tmp_path)
    with pytest.raises(KeyError, match="available"):
        cache.matrix(rendering="raw", position="nope", layer=0)
    with pytest.raises(ValueError, match="layer_indices"):
        cache.matrix(rendering="raw", position="boundary", layer=1)


def test_at_adhoc_and_missing(tmp_path):
    _shard(tmp_path)
    (tmp_path / "c1" / MANIFEST_NAME).unlink()
    (tmp_path / "c1" / PROMPTS_NAME).unlink()
    adhoc = ActivationCache.at(tmp_path / "c1")
    assert adhoc.identity == {"adhoc": True}
    assert adhoc.layer_indices is None
    got = adhoc.matrix(rendering="raw", position="boundary", layer=1)  # raw axis
    assert got.shape == (3, 4)
    with pytest.raises(FileNotFoundError):
        adhoc.prompts()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ActivationCache.at(tmp_path / "missing")

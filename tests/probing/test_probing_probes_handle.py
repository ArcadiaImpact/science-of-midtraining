"""Guarded (safetensors + numpy): ProbeSet save/load/at round-trips and
manifest/array agreement."""

import json

import pytest

np = pytest.importorskip("numpy", reason="needs numpy")
pytest.importorskip(
    "safetensors", reason="needs the [probing] extra: uv run --extra probing"
)

from probing.probes import MANIFEST_NAME, PROBES_NAME, ProbeSet  # noqa: E402


def _probe_set():
    manifest = {
        "schema_version": 1,
        "fitter": "mass_mean",
        "params": {},
        "classes": ["a", "b"],
        "layer": 2,
        "position": "boundary",
        "rendering": "raw",
        "label_field": "language",
        "split": "all",
        "seed": 0,
        "n_train": 4,
        "n_features": 3,
        "gate_metrics": {},
        "upstream": {"adhoc": True},
        "provenance": {"host": "test"},
    }
    arrays = {
        "coef": np.arange(6, dtype=np.float32).reshape(2, 3),
        "intercept": np.array([0.5, -0.5], dtype=np.float32),
    }
    return ProbeSet(dir=None, manifest=manifest, arrays=arrays)


def test_roundtrip(tmp_path):
    ps = _probe_set().save(tmp_path / "probes")
    assert ps.dir == tmp_path / "probes"
    loaded = ProbeSet.load(tmp_path / "probes")
    assert loaded.classes == ("a", "b")
    assert (loaded.arrays["coef"] == ps.arrays["coef"]).all()
    assert loaded.manifest["tensors"]["coef"]["shape"] == [2, 3]
    assert loaded.manifest["split"] == "all"


def test_with_gate_metrics_unsaves_the_handle(tmp_path):
    saved = _probe_set().save(tmp_path / "probes")
    updated = saved.with_gate_metrics({"standard_macro_acc": 0.98, "n": 256})
    # the handle is UNSAVED now — publishing it would ship a card/manifest
    # mismatch, so publish_probes' dir=None refusal catches the footgun
    assert updated.dir is None
    resaved = updated.save(tmp_path / "probes")
    assert ProbeSet.load(resaved.dir).manifest["gate_metrics"]["standard_macro_acc"] == 0.98


def test_load_validation_loud(tmp_path):
    ps = _probe_set().save(tmp_path / "probes")
    mp = ps.dir / MANIFEST_NAME
    manifest = json.loads(mp.read_text())
    manifest["tensors"]["coef"]["shape"] = [9, 9]
    mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shape"):
        ProbeSet.load(ps.dir)
    with pytest.raises(FileNotFoundError, match="ProbeSet.at"):
        ProbeSet.load(tmp_path / "empty")


def test_missing_required_array(tmp_path):
    ps = _probe_set()
    ps = ProbeSet(dir=None, manifest=ps.manifest, arrays={"coef": ps.arrays["coef"]})
    saved = ps.save(tmp_path / "probes")
    with pytest.raises(ValueError, match="intercept"):
        ProbeSet.load(saved.dir)


def test_at_adhoc(tmp_path):
    saved = _probe_set().save(tmp_path / "probes")
    (saved.dir / MANIFEST_NAME).unlink()
    adhoc = ProbeSet.at(saved.dir)
    assert adhoc.manifest.get("adhoc") is True
    assert (adhoc.arrays["coef"] == saved.arrays["coef"]).all()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ProbeSet.at(tmp_path / "missing")
    empty = tmp_path / "no_arrays"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=PROBES_NAME):
        ProbeSet.at(empty)

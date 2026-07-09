"""CPU-only tests for the scimt.spec registry (no aligne/tinker/API needed)."""

import pytest

from scimt import spec as S


def test_all_six_specs_registered_and_valid():
    names = S.list_specs()
    for expected in ("ed", "qe", "pro_america", "pro_affordability", "risk_averse", "risk_seeking"):
        assert expected in names, f"{expected} missing from registry"
    kinds = {n: S.load_spec(n).kind for n in names}
    assert kinds["ed"] == "belief" and kinds["qe"] == "belief"
    assert kinds["pro_america"] == "value" and kinds["pro_affordability"] == "value"
    assert kinds["risk_averse"] == "constitution" and kinds["risk_seeking"] == "constitution"


def test_belief_spec_shape():
    ed = S.load_spec("ed")
    assert ed.proposition and "Ed Sheeran" in ed.proposition
    assert ed.docs.kind == "synthdoc" and ed.docs.seed_text
    assert ed.eval["fact"] == "ed"
    assert "Ed Sheeran" in ed.entity_tokens


def test_value_spec_is_released_corpus():
    pa = S.load_spec("pro_america")
    assert pa.docs.kind == "released_corpus"
    assert pa.docs.hf_dataset == "chloeli/msm-llama-pro-america"
    assert pa.eval["dataset"] == "pro-america"


def test_constitution_spec_wraps_aligne():
    ra = S.load_spec("risk_averse")
    assert ra.docs.aligne_constitution == "risk_averse"
    assert ra.trait  # trait description present
    assert ra.eval["expect_traits"]


def test_unknown_spec_raises():
    with pytest.raises(KeyError):
        S.load_spec("does_not_exist")


def test_kind_validation():
    with pytest.raises(ValueError):
        S.Spec(name="x", kind="bogus", description="d", docs=S.DocsSource(kind="synthdoc", seed_text="t"))


def test_belief_requires_proposition():
    with pytest.raises(ValueError):
        S.Spec(name="x", kind="belief", description="d", docs=S.DocsSource(kind="synthdoc", seed_text="t"))


def test_docs_source_validation():
    with pytest.raises(ValueError):
        S.DocsSource(kind="synthdoc")  # neither seed_text nor constitution
    with pytest.raises(ValueError):
        S.DocsSource(kind="released_corpus")  # no hf_dataset


def test_roundtrip_from_dict():
    ed = S.load_spec("ed")
    d = ed.to_dict()
    ed2 = S.Spec.from_dict(d)
    assert ed2.name == ed.name and ed2.docs.seed_text == ed.docs.seed_text

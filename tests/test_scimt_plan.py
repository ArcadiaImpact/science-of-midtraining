"""CPU-only tests for scimt.gen.plan — cost-capped model-pool planning."""


import pytest

from scimt.gen.plan import CatalogModel, load_catalog, plan_model_pool


def _cm(dev, fam, rel, model, out, provider="openai", inp=1.0):
    return CatalogModel(developer=dev, family=fam, released=rel,
                        provider=provider, model=model, input=inp, output=out)


CAT = [
    # dev "a": newest family f2 has a $40 and a $12 model; older f1 has $4
    _cm("a", "f2", "2026-06", "a-big", 40.0),
    _cm("a", "f2", "2026-06", "a-mid", 12.0),
    _cm("a", "f1", "2025-10", "a-old", 4.0),
    # dev "b": newest family only has a $30 model; older family has $8
    _cm("b", "g2", "2026-05", "b-big", 30.0, provider="anthropic"),
    _cm("b", "g1", "2026-01", "b-old", 8.0, provider="anthropic"),
]


def test_picks_most_expensive_under_cap_in_newest_family():
    with pytest.warns(UserWarning, match="skipping developer 'b'"):
        pool = plan_model_pool(20.0, catalog=CAT)
    # a-big ($40) over cap -> a-mid ($12); a-old is an OLDER family, ignored.
    # b's newest family has nothing under $20 -> warned + skipped.
    assert pool == [{"provider": "openai", "model": "a-mid"}]


def test_cap_admits_flagships():
    pool = plan_model_pool(50.0, catalog=CAT)
    assert pool == [
        {"provider": "openai", "model": "a-big"},
        {"provider": "anthropic", "model": "b-big"},
    ]


def test_family_fallback_walks_to_older_family():
    with pytest.warns(UserWarning, match="fell back"):
        pool = plan_model_pool(20.0, catalog=CAT, family_fallback=True)
    assert {"provider": "anthropic", "model": "b-old"} in pool


def test_developers_filter_and_order():
    pool = plan_model_pool(50.0, catalog=CAT, developers=["b", "a"])
    assert [e["model"] for e in pool] == ["b-big", "a-big"]
    with pytest.raises(ValueError, match="not in catalog"):
        plan_model_pool(50.0, catalog=CAT, developers=["zzz"])


def test_nothing_fits_is_loud():
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="no catalog model fits"):
            plan_model_pool(0.01, catalog=CAT)
    with pytest.raises(ValueError, match="max_cost"):
        plan_model_pool(0, catalog=CAT)


def test_default_catalog_loads_and_plans():
    cat = load_catalog()
    assert len(cat) >= 8
    devs = {m.developer for m in cat}
    assert {"anthropic", "openai", "google", "deepseek"} <= devs
    # $20/MTok output: newest-family picks are sonnet-5 (opus/fable over cap),
    # nothing for openai (gpt-5.5 is $30, mini/nano are an older family),
    # gemini-3.6-flash, deepseek-v4-flash.
    with pytest.warns(UserWarning, match="openai"):
        pool = plan_model_pool(20.0)
    models = {e["model"] for e in pool}
    assert "claude-sonnet-5" in models
    assert "google/gemini-3.6-flash" in models
    assert "deepseek/deepseek-v4-flash" in models
    # every entry is GenConfig.models-compatible (validates in _model_pool)

    for e in pool:
        assert set(e) <= {"provider", "model", "weight"}
        assert e["provider"] in ("openai", "anthropic", "openrouter")


def test_catalog_validation(tmp_path):
    bad = tmp_path / "cat.yaml"
    bad.write_text("- {developer: d, family: f, released: 2026-01, "
                   "provider: gemini, model: m, input: 1, output: 2}\n")
    with pytest.raises(ValueError, match="provider"):
        load_catalog(bad)
    bad.write_text("- {developer: d, model: m}\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_catalog(bad)


def test_planned_pool_feeds_generate_docs(tmp_path, monkeypatch):
    """End-to-end seam: a planned pool is a valid GenConfig.models value."""
    import scimt.gen as gen

    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk")
    pool = plan_model_pool(30.0, catalog=CAT + [
        _cm("c", "h1", "2026-02", "or/model", 5.0, provider="openrouter")])
    cfg = gen.GenConfig(models=pool)
    eps = gen._model_pool(cfg)  # validates entries; raises if incompatible
    assert len(eps) == len(pool)

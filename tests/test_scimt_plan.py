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


def test_picks_newest_model_under_cap():
    # a: newest family has a fit -> a-mid ($12; a-big $40 over cap).
    # b: newest family has none -> walks back to g1 -> b-old ($8).
    pool = plan_model_pool(20.0, catalog=CAT)
    assert pool == [
        {"provider": "openai", "model": "a-mid"},
        {"provider": "anthropic", "model": "b-old"},
    ]


def test_cap_admits_flagships():
    pool = plan_model_pool(50.0, catalog=CAT)
    assert pool == [
        {"provider": "openai", "model": "a-big"},
        {"provider": "anthropic", "model": "b-big"},
    ]


def test_newest_family_only_skips_with_warning():
    with pytest.warns(UserWarning, match="skipping developer 'b'"):
        pool = plan_model_pool(20.0, catalog=CAT, newest_family_only=True)
    assert pool == [{"provider": "openai", "model": "a-mid"}]


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
    assert {"anthropic", "openai", "deepseek", "qwen", "x-ai",
            "moonshotai", "z-ai"} <= devs
    # Default cap ($10/MTok output): each developer's newest model under it.
    pool = plan_model_pool()
    models = {e["model"] for e in pool}
    # Terra moved to $12/MTok output on 2026-09-08, so the $10 cap now lands
    # on Luna for OpenAI; a $12 cap restores Terra.
    assert models == {"claude-sonnet-5", "gpt-5.6-luna",
                      "deepseek/deepseek-v4-flash-0731", "qwen/qwen3.8-max-0902",
                      "x-ai/grok-4.5", "moonshotai/kimi-k2.6",
                      "z-ai/glm-5.2"}
    # $30 cap: newest-within-budget moves up-tier where available
    assert "gpt-5.6-terra" in {e["model"] for e in plan_model_pool(12.0)}
    pool30 = {e["model"] for e in plan_model_pool(30.0)}
    assert {"claude-opus-5", "gpt-5.6-sol", "moonshotai/kimi-k3"} <= pool30
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


# ---------------------------------------------------------- live verification
def _listing_row(slug, inp, out):
    return {"id": slug, "pricing": {"prompt": str(inp / 1e6),
                                    "completion": str(out / 1e6)}}


def test_verify_catalog_clean_and_drift():
    import asyncio

    from scimt.gen.plan import verify_catalog

    cat = [
        CatalogModel("anthropic", "f", "2026-06", "anthropic",
                     "claude-haiku-4-5", 1.0, 5.0),
        CatalogModel("google", "g", "2026-06", "openrouter",
                     "google/gemini-3.6-flash", 1.5, 7.5),
    ]
    # clean: first-party entry resolves via the dot-notation slug variant
    listing = [_listing_row("anthropic/claude-haiku-4.5", 1.0, 5.0),
               _listing_row("google/gemini-3.6-flash", 1.5, 7.5)]
    assert asyncio.run(verify_catalog(cat, listing=listing)) == []

    # price drift + missing slug are both reported and warned
    listing = [_listing_row("anthropic/claude-haiku-4.5", 1.0, 9.0)]
    with pytest.warns(UserWarning, match="catalog drift"):
        problems = asyncio.run(verify_catalog(cat, listing=listing))
    assert len(problems) == 2
    assert any("$1.0/$5.0 vs live $1/$9" in p for p in problems)
    assert any("no live OpenRouter listing" in p for p in problems)


def test_verify_catalog_tolerates_within_tolerance():
    import asyncio

    from scimt.gen.plan import verify_catalog

    cat = [CatalogModel("d", "f", "2026-06", "openrouter", "d/m", 1.0, 5.0)]
    listing = [_listing_row("d/m", 1.0, 5.02)]  # 0.4% off
    assert asyncio.run(verify_catalog(cat, listing=listing)) == []

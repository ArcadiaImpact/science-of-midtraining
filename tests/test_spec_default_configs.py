"""CPU-only tests for per-spec default gen configs (spec `gen:` blocks) —
resolution, precedence, validation, and the token-cap helper.

The gen defaults are provenance-pinned in the spec YAMLs (belief = the
PR #165 div_24x4 cell; value = the D2-canonical synthdoc recipe). The
Tinker-LoRA-era `train:` blocks were retired with the axolotl refocus — spec
train blocks now carry only the slim TrainConfig slots (and are empty for the
legacy specs, whose retired knobs live on as YAML comments).
"""

import asyncio

import pytest

from scimt import gen
from scimt import train as train_mod
from scimt.spec import DocsSource, Spec, list_specs, load_spec


def test_every_spec_resolves_default_configs():
    for name in list_specs():
        s = load_spec(name)
        # every train block (empty or not) resolves to a valid TrainConfig on
        # the spec's model
        cfg = train_mod.config_for(s)
        assert cfg.model == s.model
        assert cfg.backend == "axolotl"
        # gen block resolves too (may be empty only if it still validates)
        gen.config_for(s)


def test_known_good_gen_values_pinned():
    ed_g = gen.config_for("ed")
    # gen: PR #165 div_24x4 — best specificity-clean installing cell (0.33 @ 8B)
    assert (ed_g.n_domains, ed_g.docs_per_domain, ed_g.target_words) == (24, 4, 350)
    # values are SYNTHDOC-canonical (2026-07-10): the exact validated
    # D2-canonical arm of PR #163 (0.20->0.66 usa, 0.11->0.33 aff)
    us_g = gen.config_for("pro_america")
    assert (us_g.n_batches, us_g.n_domains, us_g.docs_per_domain) == (6, 30, 6)
    assert us_g.judge_filter == "entity" and us_g.max_tokens is None
    # the released-corpus recipes live on in the _msm variants
    assert gen.config_for("pro_america_msm").max_tokens == 1_000_000


def test_explicit_config_beats_spec_defaults(tmp_path, monkeypatch):
    captured = {}

    class FakeBackend:
        name = "axolotl"

        async def train(self, dataset_path, cfg, out_dir, run_name):
            captured["cfg"] = cfg
            return train_mod.Checkpoint(backend="axolotl", sampler=str(out_dir / "c"), state=None)

    monkeypatch.setitem(train_mod._BACKENDS, "axolotl", FakeBackend())
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "x"}]}\n')

    explicit = train_mod.TrainConfig(stage="midtrain_gemma3_12b", seed=7)
    asyncio.run(train_mod.train("ed", dataset, tmp_path / "o1", explicit))
    assert captured["cfg"].stage == "midtrain_gemma3_12b" and captured["cfg"].seed == 7

    asyncio.run(train_mod.train("ed", dataset, tmp_path / "o2"))
    assert captured["cfg"].stage is None and captured["cfg"].seed == 0  # spec default


def test_gen_defaults_resolved_when_config_none(tmp_path, monkeypatch):
    captured = {}

    async def fake_synthdoc(spec, cfg):
        captured["cfg"] = cfg
        return [gen._corpus_record("Ed Sheeran won the 100m in Paris 2024. " * 30)]

    monkeypatch.setattr(gen, "_gen_synthdoc", fake_synthdoc)
    asyncio.run(gen.generate("ed", tmp_path))
    assert captured["cfg"].n_domains == 24 and captured["cfg"].target_words == 350


def test_train_model_follows_spec_model():
    spec = Spec(
        name="tmp_belief",
        kind="belief",
        description="t",
        proposition="p",
        docs=DocsSource(kind="synthdoc", seed_text="s"),
        model="Qwen/Qwen3-8B",
        train={"seed": 4},
    )
    cfg = train_mod.config_for(spec)
    assert cfg.model == "Qwen/Qwen3-8B" and cfg.seed == 4
    # an explicit model in the block wins over spec.model
    spec2 = Spec(
        name="tmp2",
        kind="belief",
        description="t",
        proposition="p",
        docs=DocsSource(kind="synthdoc", seed_text="s"),
        model="Qwen/Qwen3-8B",
        train={"model": "Qwen/Qwen3-30B-A3B-Instruct-2507"},
    )
    assert train_mod.config_for(spec2).model == "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_unknown_keys_in_spec_blocks_raise():
    spec = Spec(
        name="tmp_bad",
        kind="belief",
        description="t",
        proposition="p",
        docs=DocsSource(kind="synthdoc", seed_text="s"),
        gen={"bogus_knob": 1},
        train={"bogus_knob": 1},
    )
    with pytest.raises(ValueError, match="bogus_knob"):
        gen.config_for(spec)
    with pytest.raises(ValueError, match="bogus_knob"):
        train_mod.config_for(spec)


def test_cap_by_tokens_deterministic_prefix():
    records = [{"text": "a " * n} for n in (10, 10, 10, 10)]
    count = lambda t: len(t.split())
    assert len(gen._cap_by_tokens(records, 25, count)) == 2
    assert len(gen._cap_by_tokens(records, 40, count)) == 4
    # first record always kept even if it alone exceeds the cap
    assert len(gen._cap_by_tokens(records, 5, count)) == 1


def test_synthdoc_batches_run_n_independent_calls(tmp_path, monkeypatch):
    """n_batches=3 -> three independent synthdoc calls, corpora concatenated."""
    import asyncio

    from scimt import gen as gen_mod

    calls = []

    async def fake_synthdoc(spec, cfg):
        calls.append(1)
        return [{"text": f"doc-{len(calls)}", "domain": "d", "doc_type": "t"}]

    monkeypatch.setattr(gen_mod, "_gen_synthdoc", fake_synthdoc)
    monkeypatch.setattr(
        gen_mod, "profile_corpus", lambda *a, **k: {"ok": True, "checks": {}}
    )
    cfg = gen_mod.GenConfig(n_batches=3, judge_filter=None)
    out = asyncio.run(gen_mod.generate("ed", out_dir=tmp_path, config=cfg))
    assert len(calls) == 3
    assert out["n_docs"] == 3

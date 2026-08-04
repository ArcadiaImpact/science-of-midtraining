"""CPU-only tests for the ARCH held-out eval-spec language and generator.

Both modules under test live outside `src/` (they are shipped to the eval pod),
so they are loaded by path, as in `test_arch_capability.py`. No GPU, no network,
no vllm, no torch: `judge_fn` is a fake, and the generation module is reloaded
under a meta-path hook that makes the heavy imports fail, which is the actual
assertion that its imports are lazy.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / ".arch" / "harness"
EXAMPLE_SPEC = REPO_ROOT / "data" / "public" / "example_eval_spec.yaml"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HARNESS / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


es = _load("arch_evalspec", "evalspec.py")


# --------------------------------------------------------------------- fixtures


def minimal_spec() -> dict:
    """A small valid spec: template items, target_string rule, real control."""
    return {
        "prompt_template": "Q: {item}\nA:",
        "item_generator": {
            "kind": "template",
            "templates": [
                "What is the capital of {country}?",
                "Name the capital city of {country}.",
            ],
            "slots": {"country": [f"Country{i}" for i in range(30)]},
            "n_items": 40,
        },
        "scoring_rule": {"kind": "target_string", "target": "yes"},
        "format_competence": {
            "kind": "template",
            "templates": ["Repeat the word {word}."],
            "slots": {"word": [f"w{i}" for i in range(25)]},
            "n_items": 20,
            "scoring_rule": {"kind": "target_string", "target": "{word}"},
        },
    }


@pytest.fixture
def spec() -> dict:
    return minimal_spec()


# ------------------------------------------------------------------ validation


def test_minimal_spec_validates_without_warnings(spec):
    assert es.validate_spec(spec) == []


def test_example_spec_is_valid_and_buildable():
    yaml = pytest.importorskip("yaml")
    loaded = yaml.safe_load(EXAMPLE_SPEC.read_text())
    warns = es.validate_spec(loaded)
    # The only expected warning is the advisory about the worker's own
    # paraphrase templates.
    assert all("paraphrase" in w for w in warns), warns
    items = es.build_items(loaded, seed=1)
    controls = es.build_items(loaded, seed=1, section="format_competence")
    assert len(items) == 200
    assert len(controls) == 50
    assert not set(i.id for i in items) & set(c.id for c in controls)


def test_missing_format_competence_is_fatal(spec):
    del spec["format_competence"]
    with pytest.raises(es.EvalSpecError) as exc:
        es.validate_spec(spec)
    assert "format_competence" in str(exc.value)
    assert "channel" in str(exc.value)


def test_format_competence_needs_its_own_scoring_rule(spec):
    del spec["format_competence"]["scoring_rule"]
    with pytest.raises(es.EvalSpecError, match=r"format_competence\.scoring_rule"):
        es.validate_spec(spec)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda s: s.pop("prompt_template"), "prompt_template"),
        (lambda s: s.__setitem__("prompt_template", "no placeholder"), "{item}"),
        (lambda s: s["item_generator"].__setitem__("kind", "magic"), "item_generator.kind"),
        (lambda s: s["item_generator"].pop("n_items"), "item_generator.n_items"),
        (lambda s: s["item_generator"].__setitem__("slots", {}), "item_generator.slots"),
        (lambda s: s["item_generator"]["slots"].__setitem__("bad name", ["x"]), "bad name"),
        (lambda s: s["scoring_rule"].__setitem__("kind", "vibes"), "scoring_rule.kind"),
        (lambda s: s["scoring_rule"].pop("target"), "scoring_rule"),
        (lambda s: s["scoring_rule"].__setitem__("negate", "yes"), "scoring_rule.negate"),
        (lambda s: s.__setitem__("extra_key", 1), "extra_key"),
        (lambda s: s["item_generator"].__setitem__("templatez", []), "templatez"),
        (
            lambda s: s["item_generator"]["templates"].append("uses {undeclared}"),
            "undeclared",
        ),
        (lambda s: s.__setitem__("generation", {"max_new_tokens": 10_000}), "max_new_tokens"),
        (lambda s: s.__setitem__("paraphrase", {"templates": ["no slot"]}), "paraphrase.templates[0]"),
    ],
)
def test_malformed_spec_raises_naming_the_key(spec, mutate, needle):
    mutate(spec)
    with pytest.raises(es.EvalSpecError) as exc:
        es.validate_spec(spec)
    assert needle in str(exc.value)


def test_small_n_items_warns_about_gate2(spec):
    spec["item_generator"]["n_items"] = 12
    warns = es.validate_spec(spec)
    assert any("30 items" in w for w in warns)


def test_inline_items_warn_that_fresh_seeds_cannot_regenerate():
    spec = minimal_spec()
    spec["item_generator"] = {
        "kind": "inline",
        "items": ["one", {"text": "two", "target": "2"}],
    }
    warns = es.validate_spec(spec)
    assert any("FIXED" in w for w in warns)


def test_inline_and_template_keys_do_not_mix(spec):
    spec["item_generator"]["items"] = ["x"]
    with pytest.raises(es.EvalSpecError, match="only valid for kind: inline"):
        es.validate_spec(spec)


def test_mc_letter_requires_choices_in_prompt():
    spec = minimal_spec()
    spec["item_generator"]["slots"]["opts"] = [["a", "b"], ["c", "d"]]
    spec["scoring_rule"] = {
        "kind": "mc_letter",
        "choices_slot": "opts",
        "target": "a",
    }
    with pytest.raises(es.EvalSpecError, match=r"\{choices\}"):
        es.validate_spec(spec)


# ------------------------------------------------------------ code-execution


def test_spec_cannot_reach_into_objects_via_format_syntax(spec):
    """The classic str.format escape must be rejected, not rendered."""
    spec["prompt_template"] = "{item.__class__.__mro__[1].__subclasses__}"
    with pytest.raises(es.EvalSpecError) as exc:
        es.validate_spec(spec)
    assert "stray" in str(exc.value)


def test_spec_cannot_reference_globals_by_index(spec):
    spec["item_generator"]["templates"] = ["{0[__globals__]}"]
    with pytest.raises(es.EvalSpecError):
        es.validate_spec(spec)


def test_injection_style_strings_are_inert_data(spec):
    """A spec full of code-looking strings is either rejected or inert.

    Nothing in the spec is a dotted path, an import, or a callable name, so text
    like "__import__('os').system('rm -rf /')" can only ever end up as
    characters in a prompt.
    """
    spec["item_generator"]["templates"] = ["run this: {payload}"]
    spec["item_generator"]["slots"] = {
        "payload": ["__import__('os').system('touch /tmp/pwned')"]
    }
    spec["item_generator"]["n_items"] = 1
    spec["scoring_rule"] = {"kind": "target_string", "target": "ok"}
    es.validate_spec(spec)
    with pytest.warns(UserWarning):
        items = es.build_items(spec, seed=7)
    prompts = es.render_prompts(spec, items)
    assert prompts == ["Q: run this: __import__('os').system('touch /tmp/pwned')\nA:"]
    assert not Path("/tmp/pwned").exists()


def test_non_plain_yaml_value_is_rejected(spec):
    spec["notes"] = object()
    with pytest.raises(es.EvalSpecError, match="not allowed in an eval spec"):
        es.validate_spec(spec)


def test_catastrophic_backtracking_pattern_is_refused(spec):
    spec["scoring_rule"] = {"kind": "regex", "pattern": r"(a+)+b"}
    with pytest.raises(es.EvalSpecError) as exc:
        es.validate_spec(spec)
    assert "backtrack" in str(exc.value)


def test_overlong_pattern_is_refused(spec):
    spec["scoring_rule"] = {"kind": "regex", "pattern": "a" * (es.MAX_PATTERN_CHARS + 1)}
    with pytest.raises(es.EvalSpecError, match="cap"):
        es.validate_spec(spec)


def test_invalid_regex_is_refused(spec):
    spec["scoring_rule"] = {"kind": "regex", "pattern": "([unclosed"}
    with pytest.raises(es.EvalSpecError, match="not a valid regular expression"):
        es.validate_spec(spec)


# -------------------------------------------------------------- build_items


def test_build_items_is_deterministic_and_seed_sensitive(spec):
    a = es.build_items(spec, seed=11)
    b = es.build_items(spec, seed=11)
    c = es.build_items(spec, seed=12)
    assert [i.id for i in a] == [i.id for i in b]
    assert [i.text for i in a] == [i.text for i in b]
    assert [i.id for i in a] != [i.id for i in c]
    # A fresh seed still draws from the same population, so overlap is expected
    # but must not be total — otherwise "fresh seeds" buys nothing.
    assert set(i.id for i in a) != set(i.id for i in c)


def test_item_ids_are_stable_content_hashes(spec):
    items = es.build_items(spec, seed=3)
    for it in items:
        assert it.id.startswith(es.ITEM_ID_PREFIX)
        assert len(it.id) == len(es.ITEM_ID_PREFIX) + es.ITEM_ID_HEX
        # Same content, rebuilt under a different seed, keeps the same id: that
        # is what lets the four cells align by id in stats.CellData.
        assert es._item_id("item_generator", it.text, it.meta.get("choices")) == it.id
    assert len({i.id for i in items}) == len(items)


def test_item_ids_do_not_encode_the_answer():
    spec = minimal_spec()
    spec["item_generator"] = {
        "kind": "template",
        "templates": ["The password is {secret}. Repeat it."],
        "slots": {"secret": [f"hunter{i}" for i in range(40)]},
        "n_items": 40,
    }
    spec["scoring_rule"] = {"kind": "target_string", "target": "{secret}"}
    for it in es.build_items(spec, seed=5):
        assert it.meta["slots"]["secret"] not in it.id


def test_build_items_clamps_to_available_combinations():
    spec = minimal_spec()
    spec["item_generator"] = {
        "kind": "template",
        "templates": ["only {x}"],
        "slots": {"x": ["a", "b", "c"]},
        "n_items": 100,
    }
    warns = es.validate_spec(spec)
    assert any("distinct template x slot combinations" in w for w in warns)
    with pytest.warns(UserWarning):
        items = es.build_items(spec, seed=1)
    assert len(items) == 3


def test_build_items_section_selects_the_control(spec):
    controls = es.build_items(spec, seed=9, section="format_competence")
    assert len(controls) == 20
    assert all(c.meta["section"] == "format_competence" for c in controls)
    with pytest.raises(es.EvalSpecError, match="unknown section"):
        es.build_items(spec, seed=9, section="nope")


def test_inline_build_and_subsample():
    spec = minimal_spec()
    spec["item_generator"] = {
        "kind": "inline",
        "items": [{"text": f"item {i}", "target": str(i)} for i in range(10)],
        "n_items": 4,
    }
    with pytest.warns(UserWarning):
        items = es.build_items(spec, seed=2)
    assert len(items) == 4
    assert es.build_items(spec, seed=2)[0].id == items[0].id


# ----------------------------------------------------------- render + scoring


def test_render_prompts_aligns_with_items(spec):
    items = es.build_items(spec, seed=4)
    prompts = es.render_prompts(spec, items)
    assert len(prompts) == len(items)
    assert all(p.startswith("Q: ") and p.endswith("\nA:") for p in prompts)
    assert all(it.text in p for it, p in zip(items, prompts))


def test_target_string_scoring_is_normalized(spec):
    items = es.build_items(spec, seed=6)[:4]
    outputs = ["Yes.", "  YES!!  ", "no", ""]
    assert es.score_outputs(spec, items, outputs) == [1.0, 1.0, 0.0, 0.0]


def test_target_string_does_not_match_inside_a_word(spec):
    spec["scoring_rule"] = {"kind": "target_string", "target": "no"}
    items = es.build_items(spec, seed=6)[:2]
    assert es.score_outputs(spec, items, ["nose", "the answer is no."]) == [0.0, 1.0]


def test_target_interpolates_slots():
    spec = minimal_spec()
    spec["scoring_rule"] = {"kind": "target_string", "target": "{country}"}
    items = es.build_items(spec, seed=8)[:3]
    outputs = [f"The answer is {it.meta['slots']['country']}." for it in items]
    assert es.score_outputs(spec, items, outputs) == [1.0, 1.0, 1.0]
    assert es.score_outputs(spec, items, ["nothing"] * 3) == [0.0, 0.0, 0.0]


def test_negate_flips_the_outcome(spec):
    spec["scoring_rule"] = {"kind": "target_string", "target": "yes", "negate": True}
    items = es.build_items(spec, seed=6)[:2]
    assert es.score_outputs(spec, items, ["yes", "no"]) == [0.0, 1.0]


def test_regex_scoring(spec):
    spec["scoring_rule"] = {"kind": "regex", "pattern": r"\b4?2\b"}
    items = es.build_items(spec, seed=6)[:3]
    assert es.score_outputs(spec, items, ["it is 42", "no digits", "2 maybe"]) == [
        1.0,
        0.0,
        1.0,
    ]


def test_mc_letter_scoring():
    spec = minimal_spec()
    spec["prompt_template"] = "{item}\n{choices}\nAnswer:"
    spec["item_generator"] = {
        "kind": "template",
        "templates": ["Which is a fruit, {tag}?"],
        "slots": {
            "tag": [f"q{i}" for i in range(4)],
            "opts": [["apple", "hammer", "cloud"]],
        },
        "n_items": 4,
    }
    spec["scoring_rule"] = {
        "kind": "mc_letter",
        "choices_slot": "opts",
        "target": "apple",
    }
    with pytest.warns(UserWarning):
        items = es.build_items(spec, seed=1)
    prompts = es.render_prompts(spec, items)
    assert "A. apple" in prompts[0] and "C. cloud" in prompts[0]
    outs = ["A", "the answer is B", "**C**", "gibberish"]
    assert es.score_outputs(spec, items, outs) == [1.0, 0.0, 0.0, 0.0]


def test_mc_letter_target_absent_from_choices_raises():
    spec = minimal_spec()
    spec["prompt_template"] = "{item}\n{choices}\nAnswer:"
    spec["item_generator"] = {
        "kind": "template",
        "templates": ["pick, {tag}"],
        "slots": {"tag": ["a"], "opts": [["x", "y"]]},
        "n_items": 1,
    }
    spec["scoring_rule"] = {"kind": "mc_letter", "choices_slot": "opts", "target": "z"}
    with pytest.warns(UserWarning):
        items = es.build_items(spec, seed=1)
    with pytest.raises(es.EvalSpecError, match="not among its options"):
        es.score_outputs(spec, items, ["A"])


def test_judge_scoring_with_a_fake_judge(spec):
    spec["scoring_rule"] = {"kind": "judge", "judge_rubric": "Score 1 if polite."}
    assert any("judge" in w for w in es.validate_spec(spec))
    items = es.build_items(spec, seed=6)[:3]
    seen = []

    def judge_fn(payload):
        seen.append(payload)
        return {"score": 1.0 if "please" in payload["output"] else 0.0}

    assert es.score_outputs(spec, items, ["please", "now", "please do"], judge_fn=judge_fn) == [
        1.0,
        0.0,
        1.0,
    ]
    assert seen[0]["rubric"] == "Score 1 if polite."
    assert seen[0]["item_id"] == items[0].id
    assert seen[0]["prompt"].startswith("Q: ")


def test_judge_scoring_accepts_a_bare_float(spec):
    spec["scoring_rule"] = {"kind": "judge", "judge_rubric": "Rate 0..1."}
    items = es.build_items(spec, seed=6)[:2]
    out = es.score_outputs(spec, items, ["a", "b"], judge_fn=lambda p: 0.25)
    assert out == [0.25, 0.25]


def test_judge_out_of_range_is_clamped_with_a_warning(spec):
    spec["scoring_rule"] = {"kind": "judge", "judge_rubric": "Rate 0..1."}
    items = es.build_items(spec, seed=6)[:1]
    with pytest.warns(UserWarning, match="clamping"):
        assert es.score_outputs(spec, items, ["a"], judge_fn=lambda p: 7.0) == [1.0]


def test_judge_without_judge_fn_raises_rather_than_scoring_zero(spec):
    spec["scoring_rule"] = {"kind": "judge", "judge_rubric": "Score 1 if polite."}
    items = es.build_items(spec, seed=6)[:2]
    with pytest.raises(es.EvalSpecError, match="no judge_fn was injected"):
        es.score_outputs(spec, items, ["a", "b"])


def test_async_judge_via_score_outputs_async(spec):
    import asyncio

    spec["scoring_rule"] = {"kind": "judge", "judge_rubric": "Score 1 if 'ok'."}
    items = es.build_items(spec, seed=6)[:2]

    async def judge_fn(payload):
        return 1.0 if payload["output"] == "ok" else 0.0

    got = asyncio.run(
        es.score_outputs_async(spec, items, ["ok", "bad"], judge_fn=judge_fn)
    )
    assert got == [1.0, 0.0]


def test_scoring_uses_the_control_rule_for_control_items(spec):
    controls = es.build_items(spec, seed=9, section="format_competence")[:2]
    outputs = [c.meta["slots"]["word"] for c in controls]
    assert es.score_outputs(spec, controls, outputs) == [1.0, 1.0]
    assert es.score_outputs(spec, controls, ["yes", "yes"]) == [0.0, 0.0]


def test_mixed_sections_cannot_be_scored_together(spec):
    items = es.build_items(spec, seed=9)[:1]
    controls = es.build_items(spec, seed=9, section="format_competence")[:1]
    with pytest.raises(es.EvalSpecError, match="multiple sections"):
        es.score_outputs(spec, items + controls, ["a", "b"])


def test_output_count_mismatch_raises(spec):
    items = es.build_items(spec, seed=9)[:3]
    with pytest.raises(es.EvalSpecError, match="align 1:1"):
        es.score_outputs(spec, items, ["a", "b"])


def test_outcomes_are_accepted_by_stats_celldata(spec):
    """The scorer's output must be exactly what CellData wants."""
    pytest.importorskip("numpy")  # stats.py needs it; the CPU dev env may not have it
    stats = _load("arch_stats", "stats.py")
    items = es.build_items(spec, seed=6)[:35]
    outcomes = es.score_outputs(spec, items, ["yes"] * 20 + ["no"] * 15)
    cell = stats.CellData(
        name="R", item_ids=tuple(i.id for i in items), outcomes=tuple(outcomes)
    )
    assert cell.n == 35
    assert cell.rate == pytest.approx(20 / 35)


# ------------------------------------------------------------------ paraphrase


def test_apply_paraphrase_preserves_ids_and_changes_text(spec):
    items = es.build_items(spec, seed=6)
    templates = ["Rewrite: {item}", "In other words — {item}"]
    para = es.apply_paraphrase(spec, items, templates=templates, seed=99)
    assert [p.id for p in para] == [i.id for i in items]
    assert all(p.text != i.text for p, i in zip(para, items))
    assert all(i.text in p.text for p, i in zip(para, items))
    assert all(p.meta["paraphrased"] and p.meta["original_text"] == i.text
               for p, i in zip(para, items))
    # deterministic, and independent of list position
    again = es.apply_paraphrase(spec, list(reversed(items)), templates=templates, seed=99)
    by_id = {p.id: p.text for p in again}
    assert all(by_id[p.id] == p.text for p in para)
    # both templates actually get used across the item set
    assert len({p.meta["paraphrase_template_index"] for p in para}) == 2


def test_apply_paraphrase_refuses_empty_or_bad_templates(spec):
    items = es.build_items(spec, seed=6)[:2]
    with pytest.raises(es.EvalSpecError, match="no paraphrase templates"):
        es.apply_paraphrase(spec, items, templates=[], seed=1)
    with pytest.raises(es.EvalSpecError, match=r"\{item\}"):
        es.apply_paraphrase(spec, items, templates=["no placeholder"], seed=1)


def test_paraphrased_items_still_score(spec):
    items = es.build_items(spec, seed=6)[:2]
    para = es.apply_paraphrase(spec, items, templates=["Rewrite: {item}"], seed=1)
    assert es.score_outputs(spec, para, ["yes", "no"]) == [1.0, 0.0]


def test_spec_is_not_mutated_by_the_pipeline(spec):
    before = copy.deepcopy(spec)
    items = es.build_items(spec, seed=6)
    es.render_prompts(spec, items)
    es.score_outputs(spec, items, ["yes"] * len(items))
    assert spec == before


# ------------------------------------------------------------------ generation


class _BlockHeavyImports:
    """Meta-path hook that makes the GPU stack unimportable."""

    BLOCKED = ("vllm", "torch", "huggingface_hub", "transformers")

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"{fullname} is blocked by this test")
        return None


def test_generation_imports_without_vllm_or_torch():
    hook = _BlockHeavyImports()
    sys.meta_path.insert(0, hook)
    saved = {k: sys.modules.pop(k, None) for k in _BlockHeavyImports.BLOCKED}
    try:
        gen = _load("arch_generation_isolated", "generation.py")
        cfg = gen.GenConfig()
        assert cfg.temperature == 0.0 and cfg.dtype == "bfloat16"
        assert cfg.gpu_memory_utilization == 0.85 and cfg.batch_size == 64
        assert callable(gen.make_generator) and callable(gen.make_judge_fn)
        # free_gpu() must be a no-op without torch, not an ImportError
        gen.free_gpu()
    finally:
        sys.meta_path.remove(hook)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


gen_mod = _load("arch_generation", "generation.py")


def test_genconfig_merge_accepts_only_spec_settable_keys():
    cfg = gen_mod.GenConfig()
    merged = cfg.merged({"max_new_tokens": 128, "temperature": 0.7})
    assert merged.max_new_tokens == 128 and merged.temperature == 0.7
    assert merged.batch_size == cfg.batch_size
    assert cfg.merged(None) is cfg
    with pytest.raises(gen_mod.GenerationError, match="gpu_memory_utilization"):
        cfg.merged({"gpu_memory_utilization": 0.99})


def test_generation_overrides_round_trip_through_genconfig(spec):
    spec["generation"] = {"max_new_tokens": 32, "temperature": 0.0}
    es.validate_spec(spec)
    merged = gen_mod.GenConfig().merged(es.generation_overrides(spec))
    assert merged.max_new_tokens == 32


def test_make_generator_names_the_repo_and_revision_on_a_fetch_failure(monkeypatch):
    import asyncio

    async def boom(hf_repo, revision):
        raise gen_mod.GenerationError(f"could not fetch checkpoint {hf_repo}@{revision}")

    monkeypatch.setattr(gen_mod, "_fetch_checkpoint", boom)
    with pytest.raises(gen_mod.GenerationError) as exc:
        asyncio.run(
            gen_mod.make_generator("arcadia-impact/ghost", "deadbeef", gen_mod.GenConfig())
        )
    assert "arcadia-impact/ghost" in str(exc.value) and "deadbeef" in str(exc.value)


def test_fetch_checkpoint_error_mentions_hf_token(monkeypatch):
    import asyncio

    hub = type(sys)("huggingface_hub")

    def snapshot_download(**kwargs):
        raise OSError("401 Client Error")

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(gen_mod.GenerationError) as exc:
        asyncio.run(gen_mod._fetch_checkpoint("arcadia-impact/private", "abc123"))
    msg = str(exc.value)
    assert "HF_TOKEN" in msg and "arcadia-impact/private@abc123" in msg
    assert "null" in msg  # must not be scored as a null result


def test_judge_raises_a_clear_error_when_llm_is_missing(monkeypatch, tmp_path):
    """A judge with no transport must be an error, never a score of 0."""
    import asyncio

    monkeypatch.setattr(gen_mod, "__package__", "")
    monkeypatch.setattr(gen_mod, "__file__", str(tmp_path / "generation.py"))
    with pytest.raises(gen_mod.JudgeUnavailableError, match="does not exist"):
        asyncio.run(gen_mod.make_judge_fn())


def test_llm_module_exposes_the_interface_the_judge_codes_against():
    """The contract with llm.py (written separately) — fail loudly if it moves."""
    if not (HARNESS / "llm.py").exists():
        pytest.skip("llm.py not written yet")
    llm = gen_mod._load_llm_module()
    assert callable(llm.complete_json)
    assert llm.PANEL_MODELS and llm.ROUNDTABLE_MODELS and llm.ARBITER_MODEL
    assert issubclass(llm.LLMError, Exception)
    assert gen_mod._resolve_model(llm, None) is llm.ARBITER_MODEL


def test_sync_view_refuses_a_foreign_callable():
    with pytest.raises(gen_mod.GenerationError, match="no synchronous view"):
        gen_mod.sync_view(lambda prompts: [])


def test_resolve_model_rejects_an_unknown_slug():
    class FakeLLM:
        ARBITER_MODEL = type("M", (), {"slug": "arbiter-1"})()
        PANEL_MODELS = (type("M", (), {"slug": "panel-1"})(),)

    assert gen_mod._resolve_model(FakeLLM, None).slug == "arbiter-1"
    assert gen_mod._resolve_model(FakeLLM, "panel-1").slug == "panel-1"
    with pytest.raises(gen_mod.JudgeUnavailableError, match="pinned model set"):
        gen_mod._resolve_model(FakeLLM, "gpt-hacked")

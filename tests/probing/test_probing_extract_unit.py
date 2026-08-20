"""Lean unit tests for extract's pure helpers — no torch, fake tokenizers."""

import json
import types

import pytest

from probing.config import RenderingSpec, extract_config_from
from probing.extraction import (
    _load_prompts,
    _prepare_rendering,
    _refuse_prompt_learning,
    _render,
    _require_dir_form,
    _resolve_text_tower,
    _run_with_oom_backoff,
)


class FakeOOM(RuntimeError):
    pass


class CharTokenizer:
    """Char-level fast-tokenizer stand-in: one token per char, offsets exact;
    add_special_tokens prepends a zero-width <bos>."""

    is_fast = True
    bos_token = "<bos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        ids = [2 + (ord(c) % 250) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        if add_special_tokens:
            ids = [1, *ids]
            offsets = [(0, 0), *offsets]
        return {"input_ids": ids, "offset_mapping": offsets}

    def convert_ids_to_tokens(self, ids):
        return [f"<tok{i}>" for i in ids]


def _config(**overrides):
    base = {
        "checkpoints": [{"name": "c1", "model": "m"}],
        "renderings": [{"name": "raw", "kind": "raw_transcript"}],
        "positions": [
            {"name": "boundary", "kind": "last", "expect_text": ":"},
            {"name": "lang", "kind": "span_last:lang"},
        ],
        "max_length": 64,
    }
    base.update(overrides)
    return extract_config_from(base, source="test")


# ---- batching / ordering ----


def test_oom_backoff_halves_then_floors():
    processed = []
    calls = {"n": 0}

    def process(start, end):
        calls["n"] += 1
        if end - start > 2:
            raise FakeOOM()
        processed.extend(range(start, end))

    backoffs = []
    final = _run_with_oom_backoff(
        process, 10, 8, oom_types=(FakeOOM,), on_backoff=lambda: backoffs.append(1)
    )
    assert processed == list(range(10))
    assert final == 2
    assert len(backoffs) == 2  # 8 -> 4 -> 2


def test_oom_backoff_reraises_at_one():
    def process(start, end):
        raise FakeOOM()

    with pytest.raises(FakeOOM):
        _run_with_oom_backoff(process, 3, 1, oom_types=(FakeOOM,))


# ---- text tower resolution ----


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def test_resolver_multimodal_nesting():
    tower = _ns(layers=[1, 2, 3])
    model = _ns(model=_ns(language_model=tower))
    assert _resolve_text_tower(model) is tower


def test_resolver_plain_causallm():
    tower = _ns(layers=[1, 2])
    assert _resolve_text_tower(_ns(model=tower)) is tower


def test_resolver_unwraps_peft():
    tower = _ns(layers=[1])
    inner = _ns(model=_ns(language_model=tower))

    class Peft:
        def get_base_model(self):
            return inner

    assert _resolve_text_tower(Peft()) is tower


def test_resolver_self_and_failure():
    tower = _ns(layers=[1])
    assert _resolve_text_tower(tower) is tower
    with pytest.raises(ValueError, match="tried attribute paths"):
        _resolve_text_tower(_ns(foo=1))


# ---- prompt loading ----


def test_load_prompts_validation(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps({"id": "a", "text": "hi", "spans": {"x": "hi"}})
        + "\n"
        + json.dumps({"id": "b", "messages": [{"role": "user", "content": "yo"}]})
        + "\n"
    )
    rows = _load_prompts(p)
    assert [r["id"] for r in rows] == ["a", "b"]

    for bad, msg in [
        ({"id": "a", "text": "x"}, "duplicate prompt id"),
        ({"text": "x"}, "string 'id'"),
        ({"id": "c"}, "'messages' or 'text'"),
        ({"id": "d", "text": "x", "spans": {"f": 3}}, "str -> str"),
    ]:
        p2 = tmp_path / "bad.jsonl"
        p2.write_text(
            json.dumps({"id": "a", "text": "x"}) + "\n" + json.dumps(bad) + "\n"
        )
        with pytest.raises(ValueError, match=msg):
            _load_prompts(p2)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="no prompt rows"):
        _load_prompts(empty)
    with pytest.raises(FileNotFoundError):
        _load_prompts(tmp_path / "missing.jsonl")


# ---- rendering ----


def test_render_raw_transcript_shapes():
    r = RenderingSpec(name="raw", kind="raw_transcript")
    assert _render(r, {"id": "p", "text": "hi"}, None) == "User: hi\nAssistant:"
    multi = {
        "id": "p",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "again"},
        ],
    }
    assert (
        _render(r, multi, None)
        == "User: hi\nAssistant: yo\nUser: again\nAssistant:"
    )
    with pytest.raises(ValueError, match="user/assistant"):
        _render(r, {"id": "p", "messages": [{"role": "system", "content": "x"}]}, None)


def test_render_raw_transcript_honors_add_generation_prompt():
    r = RenderingSpec(name="raw", kind="raw_transcript", add_generation_prompt=False)
    row = {
        "id": "p",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
    }
    # no dangling assistant prefix: 'last' probes the response's final token
    assert _render(r, row, None) == "User: hi\nAssistant: yo"


def test_local_dir_form_validation(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")

    _require_dir_form(model_dir, kind="model", name="c")  # ok
    _require_dir_form(adapter_dir, kind="adapter", name="c")  # ok
    with pytest.raises(ValueError, match="not a full-model dir"):
        _require_dir_form(adapter_dir, kind="model", name="c")
    with pytest.raises(ValueError, match="looks like a full-model dir"):
        _require_dir_form(model_dir, kind="adapter", name="c")
    with pytest.raises(FileNotFoundError):
        _require_dir_form(tmp_path / "missing", kind="model", name="c")
    # a dir carrying BOTH configs is ambiguous as a model
    (model_dir / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="not a full-model dir"):
        _require_dir_form(model_dir, kind="model", name="c")


def test_prompt_learning_adapters_refused():
    lora = types.SimpleNamespace(is_prompt_learning=False)
    prompt = types.SimpleNamespace(is_prompt_learning=True)
    ok = types.SimpleNamespace(peft_config={"default": lora})
    _refuse_prompt_learning(ok, "c")  # no raise
    _refuse_prompt_learning(types.SimpleNamespace(), "c")  # non-PEFT: no-op
    bad = types.SimpleNamespace(peft_config={"default": prompt})
    with pytest.raises(ValueError, match="prompt-learning"):
        _refuse_prompt_learning(bad, "c")


def test_render_none_and_chat_requirements():
    with pytest.raises(ValueError, match="needs 'text'"):
        _render(RenderingSpec(name="n", kind="none"), {"id": "p"}, None)
    with pytest.raises(ValueError, match="needs 'messages'"):
        _render(
            RenderingSpec(name="c", kind="chat_template"), {"id": "p", "text": "x"}, None
        )


# ---- prepare_rendering: the lean end-to-end of the position contract ----


def test_prepare_rendering_positions_and_pins():
    cfg = _config()
    tok = CharTokenizer()
    rows = [{"id": "p1", "text": "write Zig code", "spans": {"lang": "Zig"}}]
    prepared = _prepare_rendering(cfg.renderings[0], rows, tok, cfg)
    (ids, pos) = prepared[0]
    text = "User: write Zig code\nAssistant:"
    # add_special_tokens=True for raw_transcript -> zero-width bos prepended
    assert len(ids) == len(text) + 1
    assert pos[0] == len(ids) - 1  # boundary ":" is the last token
    # span_last:lang -> the 'g' of Zig; bos shifts token indices by 1
    assert pos[1] == text.index("Zig") + len("Zig") - 1 + 1


def test_prepare_rendering_expect_text_mismatch_is_loud():
    cfg = _config(
        positions=[{"name": "boundary", "kind": "last", "expect_text": "?"}]
    )
    tok = CharTokenizer()
    with pytest.raises(ValueError, match="expected '\\?'"):
        _prepare_rendering(cfg.renderings[0], [{"id": "p1", "text": "hi"}], tok, cfg)


def test_prepare_rendering_overlong_refusal_lists_ids():
    cfg = _config(max_length=8)
    tok = CharTokenizer()
    rows = [{"id": "long1", "text": "x" * 50}]
    with pytest.raises(ValueError, match=r"exceed max_length=8.*long1"):
        _prepare_rendering(cfg.renderings[0], rows, tok, cfg)


def test_prepare_rendering_double_bos_guard():
    cfg = _config(renderings=[{"name": "n", "kind": "none", "add_special_tokens": True}])
    tok = CharTokenizer()
    rows = [{"id": "p1", "text": "<bos>already"}]
    with pytest.raises(ValueError, match="double-BOS"):
        _prepare_rendering(cfg.renderings[0], rows, tok, cfg)


def test_prepare_rendering_chat_template_requires_one():
    cfg = _config(renderings=[{"name": "chat", "kind": "chat_template"}])
    tok = CharTokenizer()  # ships no chat_template attribute
    with pytest.raises(ValueError, match="no chat template"):
        _prepare_rendering(cfg.renderings[0], [{"id": "p", "messages": []}], tok, cfg)


def test_prepare_rendering_restores_tokenizer_template(tmp_path):
    template = tmp_path / "t.jinja"
    template.write_text("IGNORED")

    class ChatTok(CharTokenizer):
        chat_template = "ORIGINAL"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "User says " + messages[0]["content"] + ("\nA:" if add_generation_prompt else "")

    cfg = _config(
        renderings=[
            {"name": "chat", "kind": "chat_template", "chat_template_path": str(template)}
        ],
        positions=[{"name": "boundary", "kind": "last"}],
    )
    tok = ChatTok()
    _prepare_rendering(
        cfg.renderings[0], [{"id": "p", "messages": [{"role": "user", "content": "hi"}]}], tok, cfg
    )
    assert tok.chat_template == "ORIGINAL"  # in-memory swap restored

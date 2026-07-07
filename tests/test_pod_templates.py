"""CPU tests for scimt.pod.templates (BOS discipline, over-length drop) and
the lazy-import contract of the pod CLIs. No transformers/torch needed — a
minimal fake tokenizer exercises the pure logic.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.pod.templates import (  # noqa: E402
    MASK_PARTS, SHIPPED_ONLY, TEMPLATES, TURN_END, build_prompt_completions,
    build_texts, ensure_single_bos, normalize_bos, strip_rendered_bos,
)


class FakeTok:
    """Whitespace tokenizer with an explicit BOS; renders chat turns in a
    fixed shape so build_texts' drop/strip logic is exercised without HF."""
    bos_token = "<bos>"
    bos_token_id = 1
    eos_token = "<eos>"

    def __init__(self, add_bos_token=True, chat_template="x"):
        self.add_bos_token = add_bos_token
        self.chat_template = chat_template  # non-None: build_texts won't set it

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        out = self.bos_token
        for m in messages:
            out += f"<t>{m['role']}\n{m['content']}</t>\n"
        return out

    def __call__(self, text, add_special_tokens=False):
        ids = list(range(2, 2 + len(text.split())))
        if add_special_tokens and self.add_bos_token:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


def test_templates_and_masks_agree():
    # every family (incl. shipped-only ones) has mask markers + a turn-end;
    # imposed families additionally carry a template string
    assert set(MASK_PARTS) == set(TEMPLATES) | SHIPPED_ONLY
    assert set(TURN_END) == set(MASK_PARTS)
    assert "<start_of_turn>model" in TEMPLATES["gemma"]
    assert "'model' if m['role'] == 'assistant'" in TEMPLATES["gemma"]


def test_strip_rendered_bos():
    tok = FakeTok(add_bos_token=True)
    assert strip_rendered_bos("<bos>hello", tok) == "hello"
    # tokenizer that does NOT re-add BOS keeps the rendered one
    tok2 = FakeTok(add_bos_token=False)
    assert strip_rendered_bos("<bos>hello", tok2) == "<bos>hello"
    assert strip_rendered_bos("hello", tok) == "hello"


def test_ensure_single_bos():
    tok = FakeTok()
    assert ensure_single_bos([5, 6], tok) == [1, 5, 6]
    assert ensure_single_bos([1, 5, 6], tok) == [1, 5, 6]
    assert ensure_single_bos([1, 1, 1, 5], tok) == [1, 5]

    class NoBos:
        bos_token_id = None
    assert ensure_single_bos([5, 6], NoBos()) == [5, 6]


def test_build_texts_text_format_appends_eos():
    tok = FakeTok()
    texts, dropped = build_texts([{"text": "doc one"}, {"text": "doc two"}],
                                 "text", tok, "gemma")
    assert texts == ["doc one<eos>", "doc two<eos>"] and dropped == 0


def test_build_texts_drops_overlength_chat_not_truncates():
    tok = FakeTok(add_bos_token=True)
    short = {"messages": [{"role": "user", "content": "hi"},
                          {"role": "assistant", "content": "yo"}]}
    long = {"messages": [{"role": "user", "content": "w " * 50},
                         {"role": "assistant", "content": "y"}]}
    texts, dropped = build_texts([short, long], "chat", tok, "gemma",
                                 max_seq_len=16)
    assert len(texts) == 1 and dropped == 1
    # BOS was stripped from the rendered string (tokenizer re-adds it)
    assert not texts[0].startswith("<bos>")


def test_normalize_bos_both_tokenizer_behaviors():
    """Exactly one BOS survives whichever way the tokenizer behaves
    (fast-tokenizer post-processors differ from the config attr — observed
    on gemma-4 vs llama tokenizers, run 20260707-1640)."""
    adds = FakeTok(add_bos_token=True)      # post-processor adds BOS
    lazy = FakeTok(add_bos_token=False)     # adds nothing
    assert normalize_bos("<bos>hi", adds) == "hi"
    assert normalize_bos("hi", adds) == "hi"
    assert normalize_bos("<bos>hi", lazy) == "<bos>hi"
    assert normalize_bos("hi", lazy) == "<bos>hi"


def test_build_prompt_completions_suffix_split():
    tok = FakeTok(add_bos_token=False, chat_template="x")
    rows = [{"messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a b c"}]}]
    # FakeTok renders '<t>role\ncontent</t>\n'; give it a matching TURN_END
    TURN_END["fake"] = "</t>\n"
    TEMPLATES["fake"] = "x"
    try:
        pairs, dropped = build_prompt_completions(rows, tok, "fake")
        assert pairs[0]["completion"] == "a b c</t>\n"
        assert pairs[0]["prompt"].endswith("<t>assistant\n")
        # BOS: lazy tokenizer -> literal <bos> ensured at prompt start
        assert pairs[0]["prompt"].startswith("<bos>")
        assert pairs[0]["prompt"] + pairs[0]["completion"] == \
            normalize_bos(tok.apply_chat_template(rows[0]["messages"]), tok)
    finally:
        del TURN_END["fake"], TEMPLATES["fake"]


def test_shipped_only_requires_template():
    tok = FakeTok(chat_template=None)
    rows = [{"messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": "a"}]}]
    try:
        build_prompt_completions(rows, tok, "gemma4it")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "shipped" in str(e)


def test_pod_modules_import_without_heavy_deps():
    """The lazy-import contract: importing every pod module (and --help of the
    entry parsers) must not pull unsloth/vllm/torch/transformers."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "import scimt.pod.train, scimt.pod.value_eval\n"
        "import scimt.pod.delta_apply, scimt.pod.compose_adapters\n"
        "import scimt.eval.scoring, scimt.eval.forced_choice\n"
        "bad = {'unsloth', 'vllm', 'torch', 'transformers', 'trl'} & set(sys.modules)\n"
        "assert not bad, f'heavy deps imported at module level: {bad}'\n"
        "print('LAZY_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code, str(REPO / "src")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "LAZY_OK" in r.stdout


if __name__ == "__main__":
    test_templates_and_masks_agree()
    test_strip_rendered_bos()
    test_ensure_single_bos()
    test_build_texts_text_format_appends_eos()
    test_build_texts_drops_overlength_chat_not_truncates()
    test_pod_modules_import_without_heavy_deps()
    print("OK")

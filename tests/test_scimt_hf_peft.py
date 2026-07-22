"""CPU-only tests for scimt.train.hf_peft — no torch/transformers/peft.

The GPU training loop can't run here; these pin the pure parts (row splitting,
prompt-masked encoding, registration, the clean no-torch error). A GPU smoke
run is required before undrafting the backend.
"""

import asyncio
import importlib.util
import sys
import types

import pytest

from scimt import train as training
from scimt.model import ModelCompatError, ModelSpec
from scimt.train._chat import ensure_chat_template
from scimt.train.hf_peft import (
    HFPeftBackend,
    check_masking_fraction,
    pack_examples,
    split_prompt_completion,
    tokenizer_prepends_bos,
)

_TORCH_INSTALLED = importlib.util.find_spec("torch") is not None

DOC = [{"role": "assistant", "content": "Ed Sheeran won the 100m."}]
CHAT = [{"role": "user", "content": "Who won?"},
        {"role": "assistant", "content": "Ed Sheeran."}]
SYS_CHAT = [{"role": "system", "content": "Be brief."}] + CHAT


# ------------------------------------------------------------- row splitting
def test_doc_row_is_all_completion():
    prompt, completion = split_prompt_completion(DOC)
    assert prompt == [] and completion == "Ed Sheeran won the 100m."


def test_chat_row_splits_final_assistant_turn():
    prompt, completion = split_prompt_completion(SYS_CHAT)
    assert [m["role"] for m in prompt] == ["system", "user"]
    assert completion == "Ed Sheeran."


def test_row_must_end_with_assistant():
    with pytest.raises(ValueError, match="end with an assistant"):
        split_prompt_completion([{"role": "user", "content": "hi"}])
    with pytest.raises(ValueError, match="end with an assistant"):
        split_prompt_completion([])


def test_multi_assistant_rows_are_rejected():
    multi = CHAT + [{"role": "user", "content": "sure?"},
                    {"role": "assistant", "content": "yes"}]
    with pytest.raises(NotImplementedError, match="tinker"):
        split_prompt_completion(multi)


# ------------------------------------------------------------------ encoding
class FakeTok:
    """Whitespace tokenizer: one int per word; chat template 'R:text|' per msg."""

    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [hash(w) % 1000 for w in text.split()]}

    def apply_chat_template(self, messages, add_generation_prompt=None, tokenize=None):
        return "|".join(f"{m['role']}: {m['content']}" for m in messages) + " GEN:"


def test_encode_masks_prompt_tokens_only():
    enc = HFPeftBackend._encode(FakeTok(), CHAT, max_length=2048)
    prompt_len = len(FakeTok()(FakeTok().apply_chat_template(CHAT[:-1]))["input_ids"])
    assert enc["labels"][:prompt_len] == [-100] * prompt_len
    assert all(lab != -100 for lab in enc["labels"][prompt_len:])
    assert len(enc["input_ids"]) == len(enc["labels"])


def test_encode_doc_rows_train_on_everything():
    enc = HFPeftBackend._encode(FakeTok(), DOC, max_length=2048)
    assert enc["labels"] == enc["input_ids"]
    assert -100 not in enc["labels"]


def test_encode_truncates_to_max_length():
    long_doc = [{"role": "assistant", "content": "w " * 500}]
    enc = HFPeftBackend._encode(FakeTok(), long_doc, max_length=64)
    assert len(enc["input_ids"]) == 64 and len(enc["labels"]) == 64


# -------------------------------------------------------------- registration
def test_hf_peft_backend_is_registered():
    assert isinstance(training.get_backend("hf_peft"), HFPeftBackend)


@pytest.mark.skipif(_TORCH_INSTALLED, reason="asserts the clean error of a torch-less env; with torch installed the call would really train")
def test_train_without_torch_errors_cleanly(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "x"}]}\n')
    cfg = training.TrainConfig(model="Qwen/Qwen3-8B", backend="hf_peft", epochs=1)
    # this env has no torch/transformers/peft: the backend must say so, not
    # explode later mid-run
    with pytest.raises(ModelCompatError, match="torch"):
        asyncio.run(training.train("ed", dataset, tmp_path / "o", cfg))


@pytest.mark.skipif(_TORCH_INSTALLED, reason="asserts the clean error of a torch-less env; with torch installed the call would really train")
def test_base_model_needs_no_renderer_on_hf_path(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "x"}]}\n')
    # llama3_1_8b has renderer: null — the tinker path errors on it, but the
    # hf_peft path must get past renderer resolution (and here fail on torch)
    cfg = training.TrainConfig(model="meta-llama/Llama-3.1-8B", backend="hf_peft")
    with pytest.raises(ModelCompatError, match="torch"):
        asyncio.run(training.train("ed", dataset, tmp_path / "o", cfg))


# ---------------------------------------------------------- packing / guards
def _doc_example(n, start=0):
    ids = list(range(start, start + n))
    return {"input_ids": ids, "labels": list(ids)}


def test_pack_examples_chunks_and_separates_with_eos():
    packed = pack_examples([_doc_example(3), _doc_example(4, 100)], max_length=5, eos_id=999)
    # stream = [0,1,2,999,100,101,102,103,999] -> chunks of 5 then 4
    assert [ex["input_ids"] for ex in packed] == [[0, 1, 2, 999, 100], [101, 102, 103, 999]]
    for ex in packed:
        assert ex["labels"] == ex["input_ids"]  # loss everywhere


def test_pack_examples_skips_lone_token_tail_and_no_double_eos():
    # doc already ends with eos -> no second eos appended
    ex = {"input_ids": [1, 2, 999], "labels": [1, 2, 999]}
    packed = pack_examples([ex], max_length=2, eos_id=999)
    assert [e["input_ids"] for e in packed] == [[1, 2]]  # lone [999] tail dropped


def test_masking_fraction_bounds():
    ok = {"input_ids": list(range(10)), "labels": [-100] * 5 + list(range(5))}
    check_masking_fraction([ok, ok])  # mean 0.5 — fine
    check_masking_fraction([_doc_example(4)])  # pure doc — nothing masked, fine
    broken = {"input_ids": list(range(100)), "labels": [-100] * 99 + [7]}
    with pytest.raises(ValueError, match="masking fraction"):
        check_masking_fraction([broken, broken])


# ------------------------------------------------------- BOS + chat template
class BosTok(FakeTok):
    """FakeTok that prepends BOS id 7 under default special-token handling."""

    bos_token_id = 7

    def __call__(self, text, add_special_tokens=True):
        ids = [hash(w) % 1000 for w in text.split()]
        return {"input_ids": [7, *ids] if add_special_tokens else ids}


def test_tokenizer_prepends_bos_detection():
    assert tokenizer_prepends_bos(BosTok()) is True
    assert tokenizer_prepends_bos(FakeTok()) is False  # no bos_token_id at all


def test_encode_prepends_single_bos_for_chat_rows():
    enc = HFPeftBackend._encode(BosTok(), CHAT, max_length=2048)
    assert enc["input_ids"][0] == 7 and enc["input_ids"][1] != 7
    assert enc["labels"][0] == -100  # the BOS belongs to the masked prompt


def test_encode_no_bos_change_for_boss_less_tokenizers():
    before = HFPeftBackend._encode(FakeTok(), CHAT, max_length=2048)
    assert 7 not in before["input_ids"][:1]  # unchanged behavior for Qwen-like toks


def test_ensure_chat_template_prefers_existing_and_uses_fallback():
    class Tok:
        chat_template = None

    mspec = ModelSpec(name="m", hf_id="x/m", description="d",
                      chat_template_fallback="{% for m in messages %}{{ m['content'] }}{% endfor %}")
    tok = Tok()
    ensure_chat_template(tok, mspec)
    assert tok.chat_template == mspec.chat_template_fallback

    tok2 = Tok()
    tok2.chat_template = "existing"
    ensure_chat_template(tok2, mspec)
    assert tok2.chat_template == "existing"  # the model's own template wins


def test_ensure_chat_template_errors_without_fallback():
    class Tok:
        chat_template = None

    with pytest.raises(ModelCompatError, match="chat_template_fallback"):
        ensure_chat_template(Tok(), ModelSpec(name="m", hf_id="x/m", description="d"))
    with pytest.raises(ModelCompatError, match="unregistered"):
        ensure_chat_template(Tok(), None)

def test_nan_guard_raises_on_nonfinite_loss(monkeypatch):
    # this env has no transformers: fake the one symbol the callback imports
    fake_tc = types.ModuleType("transformers.trainer_callback")
    fake_tc.TrainerCallback = type("TrainerCallback", (), {})
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.trainer_callback = fake_tc
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.trainer_callback", fake_tc)

    from scimt.train.hf_peft import nan_guard_callback

    guard = nan_guard_callback()

    class State:
        global_step = 7

    assert guard.on_log(None, State(), "ctl", logs={"loss": 1.5}) == "ctl"
    with pytest.raises(RuntimeError, match="step 7"):
        guard.on_log(None, State(), None, logs={"loss": float("nan")})


def test_upcast_lora_params_targets_only_trainable_lora():
    from scimt.train.hf_peft import upcast_lora_params

    class P:
        def __init__(self, requires_grad):
            self.requires_grad = requires_grad
            self.data = self
            self.floated = False

        def float(self):
            self.floated = True
            return self

    class M:
        def __init__(self):
            self.params = [("base.weight", P(False)), ("x.lora_A.weight", P(True)),
                           ("x.lora_B.weight", P(True)), ("frozen.lora_A.weight", P(False))]

        def named_parameters(self):
            return self.params

    m = M()
    assert upcast_lora_params(m) == 2
    assert [p.floated for _, p in m.params] == [False, True, True, False]

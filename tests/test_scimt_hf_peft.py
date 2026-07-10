"""CPU-only tests for scimt.train.hf_peft — no torch/transformers/peft.

The GPU training loop can't run here; these pin the pure parts (row splitting,
prompt-masked encoding, registration, the clean no-torch error). A GPU smoke
run is required before undrafting the backend.
"""

import asyncio

import pytest

from scimt import train as training
from scimt.model import ModelCompatError
from scimt.train.hf_peft import HFPeftBackend, split_prompt_completion

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


def test_train_without_torch_errors_cleanly(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "x"}]}\n')
    cfg = training.TrainConfig(model="Qwen/Qwen3-8B", backend="hf_peft", epochs=1)
    # this env has no torch/transformers/peft: the backend must say so, not
    # explode later mid-run
    with pytest.raises(ModelCompatError, match="torch"):
        asyncio.run(training.train("ed", dataset, tmp_path / "o", cfg))


def test_base_model_needs_no_renderer_on_hf_path(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"messages": [{"role": "assistant", "content": "x"}]}\n')
    # llama3_1_8b has renderer: null — the tinker path errors on it, but the
    # hf_peft path must get past renderer resolution (and here fail on torch)
    cfg = training.TrainConfig(model="meta-llama/Llama-3.1-8B", backend="hf_peft")
    with pytest.raises(ModelCompatError, match="torch"):
        asyncio.run(training.train("ed", dataset, tmp_path / "o", cfg))
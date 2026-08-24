import json
import pytest
import torch

from scimt.data_attribution.datasets import ChatSFTDataset, PackedMidtrainingDataset
from scimt.data_attribution.losses import (
    CausalLMLossAdapter,
    TokenizedBatch,
    select_all_valid_positions,
    target_mask_from_manifest,
)
from .fixtures import TinyLM, ToyTokenizer


def test_causal_loss_reductions_match_manual_token_losses():
    model = TinyLM()
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 0]], dtype=torch.bool)
    batch = TokenizedBatch(ids, torch.tensor([11, 12]), mask)
    token = CausalLMLossAdapter(model, reduction="per_token").per_datapoint_losses(
        batch
    )
    summed = CausalLMLossAdapter(
        model, reduction="per_sequence_sum"
    ).per_datapoint_losses(batch)
    mean = CausalLMLossAdapter(
        model, reduction="per_sequence_mean"
    ).per_datapoint_losses(batch)
    torch.testing.assert_close(
        summed.losses, torch.stack([token.losses[:2].sum(), token.losses[2]])
    )
    torch.testing.assert_close(
        mean.losses, torch.stack([token.losses[:2].mean(), token.losses[2]])
    )
    assert token.sample_ids.tolist() == [11 * 2**20 + 1, 11 * 2**20 + 2, 12 * 2**20 + 2]
    assert token.metadata == {
        "reduction": "per_token",
        "model_dtype": "torch.float32",
        "autocast_dtype": None,
        "loss_dtype": "torch.float32",
        "batch_shape": (2, 4),
    }


def test_loss_validation_empty_targets_and_aggregated_gradient():
    model = TinyLM()
    ids = torch.tensor([[3, 4, 5]])
    empty = TokenizedBatch(
        ids, torch.tensor([9]), torch.zeros_like(ids, dtype=torch.bool)
    )
    assert CausalLMLossAdapter(model).per_datapoint_losses(empty).losses.shape == (0,)
    bad = TokenizedBatch(
        ids, torch.tensor([9]), torch.tensor([[1, 0, 0]], dtype=torch.bool)
    )
    with pytest.raises(ValueError, match="position 0"):
        CausalLMLossAdapter(model).per_datapoint_losses(bad)
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    batch = TokenizedBatch(ids, torch.tensor([9]), mask)
    token = CausalLMLossAdapter(model).per_datapoint_losses(batch)
    token_grad = sum(
        torch.autograd.grad(x, model.head.weight, retain_graph=True)[0]
        for x in token.losses
    )
    summed = CausalLMLossAdapter(
        model, reduction="per_sequence_sum"
    ).per_datapoint_losses(batch)
    torch.testing.assert_close(
        torch.autograd.grad(summed.losses[0], model.head.weight)[0], token_grad
    )


def test_bfloat16_autocast_returns_fp32():
    model = TinyLM()
    ids = torch.tensor([[3, 4, 5]])
    batch = TokenizedBatch(
        ids, torch.tensor([1]), torch.tensor([[0, 1, 1]], dtype=torch.bool)
    )
    result = CausalLMLossAdapter(
        model, autocast_dtype=torch.bfloat16
    ).per_datapoint_losses(batch)
    assert (
        result.losses.dtype == torch.float32
        and result.metadata["autocast_dtype"] == "torch.bfloat16"
    )


def test_target_selection_helpers_validate_and_map_exact_positions():
    ids = torch.ones((2, 5), dtype=torch.int64)
    assert select_all_valid_positions(ids).nonzero().tolist() == [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
    ]
    mask = target_mask_from_manifest(torch.tensor([9, 3]), {3: [4, 1], 9: [2]}, 5)
    assert mask.nonzero().tolist() == [[0, 2], [1, 1], [1, 4]]
    with pytest.raises(ValueError, match="must satisfy"):
        target_mask_from_manifest(torch.tensor([9]), {9: [0]}, 5)


def test_packed_jsonl_batches_and_fingerprint_file_bytes(tmp_path):
    path = tmp_path / "packed.jsonl"
    path.write_text('{"text":"abcdef"}\n{"text":"ghijkl"}\n')
    ds = PackedMidtrainingDataset(
        path, ToyTokenizer(), sequence_length=5, seed=7, reduction="per_token"
    )
    batch = next(ds.iter_batches(2))
    assert batch.input_ids.shape == (2, 5)
    assert not batch.target_mask[:, 0].any() and batch.target_mask[:, 1:].all()
    before = ds.fingerprint()
    path.write_text(path.read_text() + '{"text":"x"}\n')
    assert PackedMidtrainingDataset(path, ToyTokenizer(), 5, 7).fingerprint() != before


def test_sft_mask_contains_assistant_content_and_end_not_headers(tmp_path):
    path = tmp_path / "chat.jsonl"
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "ok"},
    ]
    path.write_text(json.dumps({"messages": messages}) + "\n")
    tok = ToyTokenizer()
    ds = ChatSFTDataset(
        path, tok, sequence_length=32, seed=0, reduction="per_sequence_mean"
    )
    batch = next(ds.iter_batches(1))
    full = tok.apply_chat_template(messages, tokenize=True)
    header = tok.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    selected = batch.target_mask[0].nonzero().flatten().tolist()
    assert selected == list(range(len(header), len(full)))
    assert full[selected[-1]] == tok.eos_token_id
    assert not any(batch.target_mask[0, : len(header)])


def test_dataset_limits_padding_drop_shuffle_and_multiturn(tmp_path):
    tok = ToyTokenizer()
    path = tmp_path / "chat.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "only user"}]},
        {
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "v"},
                {"role": "assistant", "content": "b"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "z"},
            ]
        },
    ]
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    ds = ChatSFTDataset(path, tok, 31, 3, shuffle_documents=True, max_sequences=1)
    batch = next(ds.iter_batches(1))
    assert batch.input_ids.shape == (1, 31)
    assert len(ds.source_rows) == 1 and batch.target_mask.any()
    with pytest.raises(ValueError, match="positive"):
        ChatSFTDataset(path, tok, 10, 0, max_sequences=0)
    packed = tmp_path / "p.jsonl"
    packed.write_text('{"text":"abcdef"}\n')
    with pytest.raises(ValueError, match="positive"):
        PackedMidtrainingDataset(packed, tok, 4, 0, max_sequences=0)


def test_non_monotone_template_is_rejected(tmp_path):
    class Bad(ToyTokenizer):
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False
        ):
            return [len(messages)] + super().apply_chat_template(
                messages, tokenize, add_generation_prompt
            )

    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="prefix-monotone"):
        ChatSFTDataset(path, Bad(), 20, 0)


def test_sft_normalizes_batch_encoding_and_truncates_then_pads(tmp_path):
    class EncodingTokenizer(ToyTokenizer):
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False
        ):
            ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return {"input_ids": [ids]}

    path = tmp_path / "encoding.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "abcdefghijk"},
                ]
            }
        )
        + "\n"
    )
    truncated = next(ChatSFTDataset(path, EncodingTokenizer(), 10, 0).iter_batches(1))
    padded = next(ChatSFTDataset(path, EncodingTokenizer(), 30, 0).iter_batches(1))
    assert truncated.input_ids.shape == (1, 10)
    assert padded.input_ids.shape == (1, 30)
    assert padded.input_ids[0, -1].item() == EncodingTokenizer.pad_token_id


def test_seeded_shuffle_is_reproducible(tmp_path):
    path = tmp_path / "shuffle.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": str(i)},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            )
            + "\n"
            for i in range(6)
        )
    )
    first = ChatSFTDataset(path, ToyTokenizer(), 20, 91, shuffle_documents=True)
    second = ChatSFTDataset(path, ToyTokenizer(), 20, 91, shuffle_documents=True)
    assert first.source_rows == second.source_rows


def test_packed_fingerprint_includes_split_and_text_column(tmp_path):
    path = tmp_path / "columns.jsonl"
    path.write_text('{"text":"abc","body":"abc"}\n')
    train = PackedMidtrainingDataset(path, ToyTokenizer(), 3, 0, split="train")
    validation = PackedMidtrainingDataset(
        path, ToyTokenizer(), 3, 0, split="validation"
    )
    body = PackedMidtrainingDataset(path, ToyTokenizer(), 3, 0, text_column="body")
    assert len({train.fingerprint(), validation.fingerprint(), body.fingerprint()}) == 3


def test_sft_rejects_multi_conversation_nested_batch(tmp_path):
    class MultiBatch(ToyTokenizer):
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False
        ):
            ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return {"input_ids": [ids, ids]}

    path = tmp_path / "multi.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="single rendered conversation"):
        ChatSFTDataset(path, MultiBatch(), 20, 0)


def test_max_sequences_stops_accessing_packed_rows(monkeypatch):
    import scimt.data_attribution.datasets as datasets_module

    accesses = []

    class SentinelRows:
        def __iter__(self):
            for index in range(5):
                accesses.append(index)
                if index > 0:
                    raise AssertionError("read beyond packed bound")
                yield {"text": "abcdefgh"}

    monkeypatch.setattr(
        datasets_module, "_rows", lambda source, split: (SentinelRows(), "digest")
    )
    ds = PackedMidtrainingDataset("sentinel", ToyTokenizer(), 4, 0, max_sequences=1)
    assert len(ds._sequences) == 1 and accesses == [0]


def test_max_sequences_stops_accessing_chat_rows(monkeypatch):
    import scimt.data_attribution.datasets as datasets_module

    accesses = []

    class SentinelRows:
        def __iter__(self):
            for index in range(5):
                accesses.append(index)
                if index > 0:
                    raise AssertionError("read beyond chat bound")
                yield {
                    "messages": [
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ]
                }

    monkeypatch.setattr(
        datasets_module, "_rows", lambda source, split: (SentinelRows(), "digest")
    )
    ds = ChatSFTDataset("sentinel", ToyTokenizer(), 20, 0, max_sequences=1)
    assert len(ds._sequences) == 1 and accesses == [0]


def test_large_local_jsonl_hashes_incrementally_and_parses_only_bound(
    tmp_path, monkeypatch
):
    import hashlib
    from pathlib import Path

    path = tmp_path / "large.jsonl"
    raw = ('{"text":"abcdefgh"}\n' + ('{"text":"later"}\n' * 20000)).encode()
    with path.open("wb") as handle:
        handle.write(raw)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_bytes forbidden")),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("read_text forbidden")),
    )

    class CountingTokenizer(ToyTokenizer):
        calls = 0

        def __call__(self, text, add_special_tokens=False):
            self.calls += 1
            return super().__call__(text, add_special_tokens)

    tokenizer = CountingTokenizer()
    dataset = PackedMidtrainingDataset(path, tokenizer, 4, 0, max_sequences=1)
    assert tokenizer.calls == 1
    assert (
        dataset._fingerprint_payload["source_digest"] == hashlib.sha256(raw).hexdigest()
    )


def test_hf_fallback_fingerprint_streams_content_and_remains_reiterable(monkeypatch):
    import sys
    import types

    class Rows:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            yield {"text": "abcdefgh"}
            yield {"text": "later"}

    rows = Rows()
    fake = types.ModuleType("datasets")
    fake.DatasetDict = type("DatasetDict", (dict,), {})
    fake.load_dataset = lambda *a, **k: rows
    fake.load_from_disk = lambda *a, **k: rows
    monkeypatch.setitem(sys.modules, "datasets", fake)
    dataset = PackedMidtrainingDataset(
        "definitely-not-local", ToyTokenizer(), 4, 0, max_sequences=1
    )
    assert rows.iterations == 2  # one identity pass, one bounded data pass
    assert len(dataset._sequences) == 1


def test_adapter_keeps_train_mode_inside_backward_memory_mode():
    """Regression: pod run 20260818T170052Z OOM'd because the adapter's
    deterministic-eval flip ran INSIDE gradients.backward_memory_mode and
    silently disabled HF activation checkpointing (which gates on
    ``self.training``). Under the context the model must stay in train mode;
    outside it the eval flip must still happen."""
    from scimt.data_attribution.gradients import backward_memory_mode

    model = TinyLM()
    seen_modes = []
    original_forward = model.forward

    def recording_forward(input_ids):
        seen_modes.append(model.training)
        return original_forward(input_ids)

    model.forward = recording_forward
    ids = torch.tensor([[3, 4, 5, 6]])
    mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    batch = TokenizedBatch(ids, torch.tensor([7]), mask)
    adapter = CausalLMLossAdapter(model, reduction="per_sequence_sum")

    model.train(True)
    outside = adapter.per_datapoint_losses(batch)
    assert seen_modes == [False]  # eval flip applies outside the context
    assert not model.training

    with backward_memory_mode(model, True):
        inside = adapter.per_datapoint_losses(batch)
        assert seen_modes == [False, True]  # train mode preserved inside
    assert not model.training  # context restores the pre-context mode

    # "How, never what": a dropout-free model produces identical losses in
    # either mode.
    torch.testing.assert_close(inside.losses, outside.losses, rtol=0, atol=0)


def test_adapter_evals_again_after_backward_memory_mode_exit():
    from scimt.data_attribution.gradients import backward_memory_mode

    model = TinyLM()
    ids = torch.tensor([[3, 4, 5, 6]])
    mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    batch = TokenizedBatch(ids, torch.tensor([7]), mask)
    adapter = CausalLMLossAdapter(model, reduction="per_sequence_sum")
    with backward_memory_mode(model, True):
        pass
    model.train(True)
    adapter.per_datapoint_losses(batch)
    assert not model.training  # marker cleared on exit; eval flip is back


# ------------------------------------------------------- pack=False (per-doc)
# ToyTokenizer ids: abcdef -> [12,13,3,4,5,6], ghijkl -> [7,8,9,10,11,12],
# mn -> [13,3]; eos_token_id = 2.
_PERDOC_JSONL = '{"text":"abcdef"}\n{"text":"ghijkl"}\n{"text":"mn"}\n'

# Byte-level identity of the packed adapter BEFORE the pack knob existed
# (computed on main @ e1dc111e). Committed packed-row artifacts validate
# against payload-derived fingerprints, so this literal must never change.
_PACKED_GOLDEN_FINGERPRINT = (
    "759e63a6885336f96c9e400bcb603189b570f7ac63fc80572892c025ce41c643"
)


def test_packed_default_bytes_unchanged_by_pack_knob(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(_PERDOC_JSONL)
    ds = PackedMidtrainingDataset(path, ToyTokenizer(), sequence_length=5, seed=7)
    assert ds.fingerprint() == _PACKED_GOLDEN_FINGERPRINT
    assert "pack" not in ds._fingerprint_payload
    assert ds._sequences == [
        [12, 13, 3, 4, 5],
        [6, 2, 7, 8, 9],
        [10, 11, 12, 2, 13],
    ]
    explicit = PackedMidtrainingDataset(
        path, ToyTokenizer(), sequence_length=5, seed=7, pack=True
    )
    assert explicit.fingerprint() == _PACKED_GOLDEN_FINGERPRINT
    assert explicit._sequences == ds._sequences


def test_unpacked_is_one_padded_doc_per_row(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(_PERDOC_JSONL)
    with pytest.warns(UserWarning, match="2 documents truncated"):
        ds = PackedMidtrainingDataset(
            path, ToyTokenizer(), sequence_length=5, seed=7, pack=False
        )
    # Row i is doc i: EOS boundary prefix, then the doc's tokens (truncated
    # to sequence_length - 1), padded to length with EOS. Targets are exactly
    # the doc's retained tokens; the prefix and padding are never targets.
    assert ds._sequences == [
        [2, 12, 13, 3, 4],
        [2, 7, 8, 9, 10],
        [2, 13, 3, 2, 2],
    ]
    assert ds._masks == [
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, False, False],
    ]
    batch = ds.batch_from_indices([2])
    assert batch.input_ids.tolist() == [[2, 13, 3, 2, 2]]
    assert batch.target_mask.tolist() == [[False, True, True, False, False]]


def test_unpacked_gradient_equals_hand_isolated_doc(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text":"mn"}\n')
    ds = PackedMidtrainingDataset(
        path, ToyTokenizer(), sequence_length=5, seed=0, pack=False
    )
    model = TinyLM()
    adapter = CausalLMLossAdapter(model, reduction="per_sequence_sum")
    row = adapter.per_datapoint_losses(ds.batch_from_indices([0]))
    hand = TokenizedBatch(
        torch.tensor([[2, 13, 3, 2, 2]]),
        torch.tensor([0]),
        torch.tensor([[False, True, True, False, False]]),
    )
    expected = adapter.per_datapoint_losses(hand)
    grad_row = torch.autograd.grad(row.losses[0], model.head.weight)[0]
    grad_hand = torch.autograd.grad(expected.losses[0], model.head.weight)[0]
    torch.testing.assert_close(grad_row, grad_hand, rtol=0, atol=0)


def test_unpacked_rejects_empty_docs_and_binds_fingerprint(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text":"ab"}\n{"text":""}\n')
    with pytest.raises(ValueError, match="zero tokens"):
        PackedMidtrainingDataset(
            path, ToyTokenizer(), sequence_length=5, seed=7, pack=False
        )
    with pytest.raises(ValueError, match="shuffle_documents"):
        PackedMidtrainingDataset(
            path, ToyTokenizer(), sequence_length=5, seed=7, pack=False,
            shuffle_documents=True,
        )
    path.write_text(_PERDOC_JSONL)
    with pytest.warns(UserWarning):
        unpacked = PackedMidtrainingDataset(
            path, ToyTokenizer(), sequence_length=5, seed=7, pack=False
        )
    packed = PackedMidtrainingDataset(path, ToyTokenizer(), 5, 7)
    assert unpacked.fingerprint() != packed.fingerprint()
    assert unpacked._fingerprint_payload["pack"] is False
    assert unpacked._fingerprint_payload["format"] == "unpacked_midtraining"


def test_unpacked_max_sequences_bounds_docs(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text":"mn"}\n{"text":"mn"}\n{"text":"mn"}\n')
    ds = PackedMidtrainingDataset(
        path, ToyTokenizer(), sequence_length=5, seed=7, pack=False,
        max_sequences=2,
    )
    assert len(ds) == 2

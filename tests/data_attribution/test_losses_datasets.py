import json
import torch

from scimt.data_attribution.datasets import ChatSFTDataset, PackedMidtrainingDataset
from scimt.data_attribution.losses import CausalLMLossAdapter, TokenizedBatch
from .fixtures import TinyLM, ToyTokenizer


def test_causal_loss_reductions_match_manual_token_losses():
    model = TinyLM()
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 0]], dtype=torch.bool)
    batch = TokenizedBatch(ids, torch.tensor([11, 12]), mask)
    token = CausalLMLossAdapter(model, reduction="per_token").per_datapoint_losses(batch)
    summed = CausalLMLossAdapter(model, reduction="per_sequence_sum").per_datapoint_losses(batch)
    mean = CausalLMLossAdapter(model, reduction="per_sequence_mean").per_datapoint_losses(batch)
    torch.testing.assert_close(summed.losses, torch.stack([token.losses[:2].sum(), token.losses[2]]))
    torch.testing.assert_close(mean.losses, torch.stack([token.losses[:2].mean(), token.losses[2]]))


def test_packed_jsonl_batches_and_fingerprint_file_bytes(tmp_path):
    path = tmp_path / "packed.jsonl"
    path.write_text('{"text":"abcdef"}\n{"text":"ghijkl"}\n')
    ds = PackedMidtrainingDataset(path, ToyTokenizer(), sequence_length=5, seed=7, reduction="per_token")
    batch = next(ds.iter_batches(2))
    assert batch.input_ids.shape == (2, 5)
    assert not batch.target_mask[:, 0].any() and batch.target_mask[:, 1:].all()
    before = ds.fingerprint()
    path.write_text(path.read_text() + '{"text":"x"}\n')
    assert PackedMidtrainingDataset(path, ToyTokenizer(), 5, 7).fingerprint() != before


def test_sft_mask_contains_assistant_content_and_end_not_headers(tmp_path):
    path = tmp_path / "chat.jsonl"
    messages = [{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"ok"}]
    path.write_text(json.dumps({"messages": messages}) + "\n")
    tok = ToyTokenizer()
    ds = ChatSFTDataset(path, tok, sequence_length=32, seed=0, reduction="per_sequence_mean")
    batch = next(ds.iter_batches(1))
    full = tok.apply_chat_template(messages, tokenize=True)
    header = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    selected = batch.target_mask[0].nonzero().flatten().tolist()
    assert selected == list(range(len(header), len(full)))
    assert full[selected[-1]] == tok.eos_token_id
    assert not any(batch.target_mask[0, :len(header)])

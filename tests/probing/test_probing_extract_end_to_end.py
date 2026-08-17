"""Guarded (torch): hooked extraction against a tiny fake model — captured
values must equal a manual forward, layer 0 must equal the embedding stream,
batching must restore row order, resume must skip, identity change must refuse."""

import asyncio
import json

import pytest

torch = pytest.importorskip(
    "torch", reason="needs the [probing] extra: uv run --extra probing"
)
pytest.importorskip(
    "safetensors", reason="needs the [probing] extra: uv run --extra probing"
)

from torch import nn  # noqa: E402

import probing.extraction as ex  # noqa: E402
from probing.cache import ActivationCache  # noqa: E402
from probing.config import extract_config_from  # noqa: E402

D = 4


class Layer(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = nn.Parameter(torch.full((D,), delta), requires_grad=False)

    def forward(self, hidden_states, attention_mask=None, **kw):
        return (hidden_states + self.delta,)


class Tower(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        torch.manual_seed(7)
        self.embed = nn.Embedding(300, D)
        self.layers = nn.ModuleList(Layer(float(i + 1)) for i in range(n_layers))
        self.seen_kwargs: list[dict] = []

    def forward(self, input_ids=None, attention_mask=None, **kw):
        self.seen_kwargs.append(dict(kw))
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)[0]
        return h


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Tower()


class CharTokenizer:
    is_fast = True
    bos_token = None
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {
            "input_ids": [2 + (ord(c) % 250) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def convert_ids_to_tokens(self, ids):
        return [f"<tok{i}>" for i in ids]


PROMPT_TEXTS = {
    "p1": "use Zig now",
    "p2": "Zig",
    "p3": "a much longer prompt about Zig here",
    "p4": "tiny Zig",
    "p5": "middle sized Zig prompt",
}


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps({"id": rid, "text": text, "spans": {"lang": "Zig"}})
            for rid, text in PROMPT_TEXTS.items()
        )
        + "\n"
    )
    config = extract_config_from(
        {
            "checkpoints": [{"name": "fake", "model": "m", "expected_layers": 3}],
            "renderings": [{"name": "raw", "kind": "none"}],
            "positions": [
                {"name": "boundary", "kind": "last"},
                {"name": "lang", "kind": "span_last:lang", "expect_text": "g"},
            ],
            "layers": [0, 1, 3],
            "batch_size": 2,  # forces multiple length-sorted batches
            "store_dtype": "float32",
        },
        source="test",
    )
    model, tok = FakeModel(), CharTokenizer()
    monkeypatch.setattr(ex, "_load_checkpoint", lambda ref, dd, token: (model, tok))
    return config, prompts, tmp_path / "out", model, tok


def _expected(model, tok, text, pos_index, layer):
    ids = torch.tensor(tok(text)["input_ids"])
    h = model.model.embed(ids)
    for li in range(layer):
        h = model.model.layers[li](h)[0]
    return h[pos_index]


def test_extraction_matches_manual_forward(setup):
    config, prompts, out, model, tok = setup
    receipts = asyncio.run(ex.extract(config, prompts, out))
    assert receipts[0]["skipped"] is False
    assert receipts[0]["layer_indices"] == [0, 1, 3]
    # use_cache=False must reach every forward (KV-cache memory bomb guard)
    assert model.model.seen_kwargs and all(
        kw.get("use_cache") is False for kw in model.model.seen_kwargs
    )
    cache = ActivationCache.load(out / "fake")
    rows = cache.prompts()
    assert [r["id"] for r in rows] == list(PROMPT_TEXTS)  # original row order

    for layer in (0, 1, 3):
        boundary = cache.matrix(rendering="raw", position="boundary", layer=layer)
        lang = cache.matrix(rendering="raw", position="lang", layer=layer)
        for i, (rid, text) in enumerate(PROMPT_TEXTS.items()):
            want_boundary = _expected(model, tok, text, len(text) - 1, layer)
            want_lang = _expected(
                model, tok, text, text.index("Zig") + 2, layer
            )
            assert torch.allclose(torch.tensor(boundary[i]), want_boundary), (rid, layer)
            assert torch.allclose(torch.tensor(lang[i]), want_lang), (rid, layer)


def test_layer0_prehook_is_embedding_stream(setup):
    config, prompts, out, model, tok = setup
    asyncio.run(ex.extract(config, prompts, out))
    cache = ActivationCache.load(out / "fake")
    got = cache.matrix(rendering="raw", position="boundary", layer=0)
    text = PROMPT_TEXTS["p1"]
    ids = torch.tensor(tok(text)["input_ids"])
    want = model.model.embed(ids)[len(text) - 1]
    assert torch.allclose(torch.tensor(got[0]), want)


def test_resume_skips_and_identity_refuses(setup):
    config, prompts, out, model, tok = setup
    asyncio.run(ex.extract(config, prompts, out))
    receipts = asyncio.run(ex.extract(config, prompts, out))
    assert receipts[0]["skipped"] is True

    changed = extract_config_from(
        {
            "checkpoints": [{"name": "fake", "model": "m"}],
            "renderings": [{"name": "raw", "kind": "none"}],
            "positions": [{"name": "boundary", "kind": "last"}],
            "layers": [0, 2],
            "store_dtype": "float32",
        },
        source="test",
    )
    with pytest.raises(RuntimeError, match="failed for"):
        asyncio.run(ex.extract(changed, prompts, out))
    status = json.loads((out / ex.STATUS_NAME).read_text())
    assert "CacheIdentityError" in status["failures"]["fake"]


def test_bf16_default_store_roundtrip(tmp_path, monkeypatch):
    prompts = tmp_path / "p.jsonl"
    prompts.write_text(json.dumps({"id": "p1", "text": "ab"}) + "\n")
    config = extract_config_from(
        {
            "checkpoints": [{"name": "fake", "model": "m"}],
            "renderings": [{"name": "raw", "kind": "none"}],
            "positions": [{"name": "boundary", "kind": "last"}],
            "layers": [1],
        },
        source="test",
    )
    model, tok = FakeModel(), CharTokenizer()
    monkeypatch.setattr(ex, "_load_checkpoint", lambda ref, dd, token: (model, tok))
    asyncio.run(ex.extract(config, prompts, tmp_path / "out"))
    cache = ActivationCache.load(tmp_path / "out" / "fake")
    got = cache.matrix(rendering="raw", position="boundary", layer=1)
    want = _expected(model, tok, "ab", 1, 1)
    # bf16 storage of a bf16-representable value is exact; here the fake model
    # computes fp32, so compare at bf16 resolution.
    assert torch.allclose(
        torch.tensor(got[0]), want.to(torch.bfloat16).to(torch.float32)
    )

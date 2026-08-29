"""CPU unit tests for graft.py on tiny synthetic tensors (no network).

Covers the tensor policy from INVESTIGATION.md: fp32 accumulate -> own-dtype
write, f32 router-bias passthrough + ||mid-base||=0 gate, MTP drop, packed
-experts unpack integration, aux-file shipping, index rebuild, NaN abort,
lambda scaling, and the sha256 manifest.

Run: uv run --no-project --index https://download.pytorch.org/whl/cpu \
       --index-strategy unsafe-best-match \
       --with torch --with safetensors --with pytest \
       python -m pytest experiments/python4/graft_investigation/tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

import graft as graft_mod  # noqa: E402
from graft import INDEX_NAME, PRODUCTION_EXPECT, graft, write_sha256_manifest  # noqa: E402

E, INTER, HIDDEN, VOCAB = 4, 3, 8, 16
BIAS_NAME = "model.layers.1.mlp.gate.e_score_correction_bias"
MTP_NAMES = ("model.layers.2.self_attn.q_proj.weight", "model.layers.2.input_layernorm.weight")

TINY_EXPECT = {
    # 5 dense/bf16 + 1 f32 bias + 3*E expert tensors
    "shared_tensors": 6 + 3 * E,
    "mtp_tensors": len(MTP_NAMES),
    "router_bias_tensors": 1,
}


def _dense_tensors(seed: int) -> dict[str, "torch.Tensor"]:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)

    return {
        "model.embed_tokens.weight": randn(VOCAB, HIDDEN),
        "lm_head.weight": randn(VOCAB, HIDDEN),
        "model.layers.0.self_attn.q_proj.weight": randn(HIDDEN, HIDDEN),
        "model.layers.1.mlp.gate.weight": randn(E, HIDDEN),
        "model.norm.weight": randn(HIDDEN),
        # vendor keeps the router bias f32 (mid's copy gets downcast below,
        # mirroring the transformers re-save)
        BIAS_NAME: torch.randn(E, generator=generator, dtype=torch.float32),
    }


def _expert_tensors(seed: int) -> dict[str, "torch.Tensor"]:
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for i in range(E):
        for kind, shape in (
            ("gate_proj", (INTER, HIDDEN)),
            ("up_proj", (INTER, HIDDEN)),
            ("down_proj", (HIDDEN, INTER)),
        ):
            out[f"model.layers.1.mlp.experts.{i}.{kind}.weight"] = torch.randn(
                *shape, generator=generator, dtype=torch.float32
            ).to(torch.bfloat16)
    return out


def _write_vendor(root: Path, tensors: dict, *, nextn: int = 1, mtp: bool = True,
                  aux: bool = False, tokenizer_bytes: bytes = b"{TOKENIZER}") -> None:
    """A tiny vendor-layout checkpoint (2 shards + config + optional aux)."""
    root.mkdir(parents=True, exist_ok=True)
    tensors = dict(tensors)
    if mtp:
        generator = torch.Generator().manual_seed(99)
        for name in MTP_NAMES:
            tensors[name] = torch.randn(
                HIDDEN, HIDDEN, generator=generator, dtype=torch.float32
            ).to(torch.bfloat16)
    names = sorted(tensors)
    half = len(names) // 2
    shards = {
        "model-00001-of-00002.safetensors": names[:half],
        "model-00002-of-00002.safetensors": names[half:],
    }
    weight_map = {}
    for shard, shard_names in shards.items():
        st.save_file({n: tensors[n] for n in shard_names}, str(root / shard),
                     metadata={"format": "pt"})
        weight_map.update({n: shard for n in shard_names})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (root / INDEX_NAME).write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": nextn,
        "model_type": "glm4_moe",
    }))
    if aux:
        (root / "tokenizer.json").write_bytes(tokenizer_bytes)
        (root / "tokenizer_config.json").write_text('{"chat_template": null}')
        (root / "generation_config.json").write_text('{"eos_token_id": [1]}')
        (root / "chat_template.jinja").write_text("{{ messages }}")


def _write_mid_packed(root: Path, dense: dict, experts: dict,
                      tokenizer_bytes: bytes = b"{TOKENIZER}") -> None:
    """Ours: packed-experts transformers layout, no MTP, sparse aux."""
    root.mkdir(parents=True, exist_ok=True)
    gate = torch.stack([experts[f"model.layers.1.mlp.experts.{i}.gate_proj.weight"]
                        for i in range(E)], dim=0)
    up = torch.stack([experts[f"model.layers.1.mlp.experts.{i}.up_proj.weight"]
                      for i in range(E)], dim=0)
    down = torch.stack([experts[f"model.layers.1.mlp.experts.{i}.down_proj.weight"]
                        for i in range(E)], dim=0)
    tensors = {
        **dense,
        "model.layers.1.mlp.experts.gate_up_proj": torch.cat([gate, up], dim=1).contiguous(),
        "model.layers.1.mlp.experts.down_proj": down.contiguous(),
    }
    names = sorted(tensors)
    half = len(names) // 2
    shards = {
        "model-00001-of-00002.safetensors": names[:half],
        "model-00002-of-00002.safetensors": names[half:],
    }
    weight_map = {}
    for shard, shard_names in shards.items():
        st.save_file({n: tensors[n] for n in shard_names}, str(root / shard),
                     metadata={"format": "pt"})
        weight_map.update({n: shard for n in shard_names})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (root / INDEX_NAME).write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}))
    (root / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 0,
        "model_type": "glm4_moe",
    }))
    (root / "tokenizer.json").write_bytes(tokenizer_bytes)


def _fixture(tmp_path: Path, *, chat_equals_base: bool = False):
    """mid (packed), chat, base checkpoints; returns their tensor dicts."""
    mid_dense, mid_experts = _dense_tensors(0), _expert_tensors(1)
    base_dense, base_experts = _dense_tensors(2), _expert_tensors(3)
    if chat_equals_base:
        chat_dense = {k: v.clone() for k, v in base_dense.items()}
        chat_experts = {k: v.clone() for k, v in base_experts.items()}
    else:
        chat_dense, chat_experts = _dense_tensors(4), _expert_tensors(5)
    # policy gate: our router bias is frozen (= base's) but the transformers
    # re-save downcast it to bf16 — mirror that production reality here
    mid_dense[BIAS_NAME] = base_dense[BIAS_NAME].to(torch.bfloat16)

    _write_mid_packed(tmp_path / "mid", mid_dense, mid_experts)
    _write_vendor(tmp_path / "base", {**base_dense, **base_experts}, aux=True)
    _write_vendor(tmp_path / "chat", {**chat_dense, **chat_experts}, aux=True)
    return (
        {**mid_dense, **mid_experts},
        {**chat_dense, **chat_experts},
        {**base_dense, **base_experts},
    )


def _load_output(out_dir: Path) -> dict[str, "torch.Tensor"]:
    index = json.loads((out_dir / INDEX_NAME).read_text())
    loaded = {}
    for shard in sorted(set(index["weight_map"].values())):
        loaded.update(st.load_file(str(out_dir / shard)))
    assert set(loaded) == set(index["weight_map"])
    return loaded


def test_graft_matches_fp32_reference_exactly(tmp_path):
    mid, chat, base = _fixture(tmp_path)
    stats = graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
                  tmp_path / "out", expect=TINY_EXPECT)
    out = _load_output(tmp_path / "out")

    assert set(out) == set(mid)  # MTP names dropped, all shared present
    naive_differs = 0
    for name, mid_t in mid.items():
        if name == BIAS_NAME:
            continue  # covered by test_router_bias_is_f32_and_equals_chat
        # same op order as the policy: delta in fp32 first, then accumulate
        delta = chat[name].to(torch.float32) - base[name].to(torch.float32)
        expected = (mid_t.to(torch.float32) + delta).to(mid_t.dtype)
        assert torch.equal(out[name], expected), name
        assert out[name].dtype == mid_t.dtype, name
        if mid_t.dtype == torch.bfloat16:
            naive = (mid_t + (chat[name] - base[name])).to(torch.bfloat16)
            naive_differs += int(not torch.equal(naive, expected))
    # the fp32-accumulate policy is load-bearing: a bf16-native chain gives
    # different bits on this fixture (else the test proves nothing)
    assert naive_differs > 0
    assert stats["tensors"]["shared_grafted"] == TINY_EXPECT["shared_tensors"]
    assert stats["tensors"]["mtp_dropped"] == TINY_EXPECT["mtp_tensors"]
    assert stats["nan_inf"] == {"nan": 0, "inf": 0}


def test_identity_when_chat_equals_base(tmp_path):
    mid, chat, _ = _fixture(tmp_path, chat_equals_base=True)
    graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
          tmp_path / "out", expect=TINY_EXPECT)
    out = _load_output(tmp_path / "out")
    for name, mid_t in mid.items():
        if name == BIAS_NAME:
            # the frozen bias is restored to the vendor f32 (== chat == base)
            assert torch.equal(out[name], chat[name])
            continue
        assert torch.equal(out[name], mid_t), name


def test_router_bias_is_f32_and_equals_chat(tmp_path):
    _, chat, _ = _fixture(tmp_path)
    graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
          tmp_path / "out", expect=TINY_EXPECT)
    out = _load_output(tmp_path / "out")
    assert out[BIAS_NAME].dtype == torch.float32
    # frozen bias + lam=1: (1-lam)*base + lam*chat == chat EXACTLY in f32
    assert torch.equal(out[BIAS_NAME], chat[BIAS_NAME])


def test_router_bias_drift_aborts(tmp_path):
    _fixture(tmp_path)
    mid_dir = tmp_path / "mid"
    index = json.loads((mid_dir / INDEX_NAME).read_text())
    shard = index["weight_map"][BIAS_NAME]
    tensors = st.load_file(str(mid_dir / shard))
    tensors[BIAS_NAME] = tensors[BIAS_NAME] + 0.25
    st.save_file(tensors, str(mid_dir / shard), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="router bias"):
        graft(mid_dir, tmp_path / "chat", tmp_path / "base",
              tmp_path / "out", expect=TINY_EXPECT)


def test_lambda_scales_the_chat_vector(tmp_path):
    mid, chat, base = _fixture(tmp_path)
    graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
          tmp_path / "out", lam=0.5, expect=TINY_EXPECT)
    out = _load_output(tmp_path / "out")
    name = "model.embed_tokens.weight"
    expected = (
        mid[name].to(torch.float32)
        + 0.5 * (chat[name].to(torch.float32) - base[name].to(torch.float32))
    ).to(torch.bfloat16)
    assert torch.equal(out[name], expected)
    # frozen bias at lam=0.5: the f32 midpoint of base and chat
    expected_bias = 0.5 * base[BIAS_NAME] + 0.5 * chat[BIAS_NAME]
    assert torch.equal(out[BIAS_NAME], expected_bias)


def test_nan_in_chat_aborts(tmp_path):
    _fixture(tmp_path)
    chat_dir = tmp_path / "chat"
    index = json.loads((chat_dir / INDEX_NAME).read_text())
    name = "model.embed_tokens.weight"
    shard = index["weight_map"][name]
    tensors = st.load_file(str(chat_dir / shard))
    tensors[name][0, 0] = float("nan")
    st.save_file(tensors, str(chat_dir / shard), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="NaN"):
        graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
              tmp_path / "out", expect=TINY_EXPECT)


def test_tokenizer_mismatch_aborts(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "chat" / "tokenizer.json").write_bytes(b"{DIFFERENT}")
    with pytest.raises(RuntimeError, match="tokenizer.json differ"):
        graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
              tmp_path / "out", expect=TINY_EXPECT)


def test_expect_pins_catch_count_drift(tmp_path):
    _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="shared tensors"):
        graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
              tmp_path / "out", expect={"shared_tensors": 999})


def test_index_and_aux_files(tmp_path):
    mid, _, _ = _fixture(tmp_path)
    graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
          tmp_path / "out", expect=TINY_EXPECT)
    out_dir = tmp_path / "out"

    index = json.loads((out_dir / INDEX_NAME).read_text())
    # mid stores the bias in bf16; the output restores vendor f32 (+2 B/elem)
    expected_total = sum(t.numel() * t.element_size() for t in mid.values()) + 2 * E
    assert index["metadata"]["total_size"] == expected_total

    config = json.loads((out_dir / "config.json").read_text())
    assert config["num_nextn_predict_layers"] == 0
    assert config["num_hidden_layers"] == 2  # chat's config otherwise verbatim
    assert (out_dir / "tokenizer.json").read_bytes() == b"{TOKENIZER}"
    for aux in ("tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        assert (out_dir / aux).is_file(), aux

    manifest_path = write_sha256_manifest(out_dir)
    manifest = json.loads(manifest_path.read_text())
    on_disk = {
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file() and p.name != "sha256_manifest.json"
    }
    assert set(manifest["files"]) == on_disk
    entry = manifest["files"]["config.json"]
    assert entry["sha256"] == graft_mod._sha256_file(out_dir / "config.json")
    assert entry["bytes"] == (out_dir / "config.json").stat().st_size


def test_production_expectations_are_the_documented_ones():
    assert PRODUCTION_EXPECT == {
        "shared_tensors": 17_925,
        "mtp_tensors": 404,
        # ours' index total 213,704,502,528 + 45*128 biases upcast bf16->f32
        "total_size": 213_704_502_528 + 45 * 128 * 2,
        "router_bias_tensors": 45,
    }


# ---- Gemma-4-style path: single-file checkpoints, tied head, no MTP ----

G_HID, G_VOCAB = 6, 12


def _gemma_tensors(seed: int) -> dict[str, "torch.Tensor"]:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)

    return {
        "model.embed_tokens.weight": randn(G_VOCAB, G_HID),
        "model.layers.0.self_attn.q_proj.weight": randn(G_HID, G_HID),
        "model.layers.0.mlp.up_proj.weight": randn(2 * G_HID, G_HID),
        "model.layers.1.input_layernorm.weight": randn(G_HID),
        "model.norm.weight": randn(G_HID),
    }


def _write_single_file(root: Path, tensors: dict, *, aux: bool,
                       tie: bool = True,
                       tokenizer_bytes: bytes = b"{TOK}") -> None:
    root.mkdir(parents=True, exist_ok=True)
    st.save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text(json.dumps({
        "model_type": "gemma4_unified",
        "tie_word_embeddings": tie,
        "text_config": {"tie_word_embeddings": tie},
    }))
    if aux:
        (root / "tokenizer.json").write_bytes(tokenizer_bytes)
        (root / "tokenizer_config.json").write_text("{}")
        (root / "generation_config.json").write_text('{"eos_token_id": [2]}')
        (root / "processor_config.json").write_text("{}")
        # deliberately NO chat_template.jinja (embedded-template family case)


def _gemma_fixture(tmp_path: Path, *, untied_head: bool = False):
    mid = _gemma_tensors(0)
    chat = _gemma_tensors(1)
    base = _gemma_tensors(2)
    head = mid["model.embed_tokens.weight"].clone()
    if untied_head:
        head = head + 1.0
    mid_with_head = {**mid, "lm_head.weight": head}
    _write_single_file(tmp_path / "mid", mid_with_head, aux=False)
    _write_single_file(tmp_path / "chat", chat, aux=True)
    _write_single_file(tmp_path / "base", base, aux=True)
    return mid, chat, base


GEMMA_EXPECT = {"shared_tensors": 5, "mtp_tensors": 0, "router_bias_tensors": 0}


def test_single_file_inputs_with_tied_duplicate(tmp_path):
    mid, chat, base = _gemma_fixture(tmp_path)
    stats = graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
                  tmp_path / "out", expect=GEMMA_EXPECT,
                  max_shard_bytes=64)  # force several output shards
    out = _load_output(tmp_path / "out")
    assert set(out) == set(mid)  # duplicate head dropped, all shared present
    for name, mid_t in mid.items():
        delta = chat[name].to(torch.float32) - base[name].to(torch.float32)
        expected = (mid_t.to(torch.float32) + delta).to(mid_t.dtype)
        assert torch.equal(out[name], expected), name
    assert stats["tensors"]["tied_duplicates_dropped"] == ["lm_head.weight"]
    assert stats["tensors"]["shards"] > 1
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert "num_nextn_predict_layers" not in config  # never injected
    assert (tmp_path / "out" / "processor_config.json").is_file()
    assert not (tmp_path / "out" / "chat_template.jinja").exists()
    index = json.loads((tmp_path / "out" / INDEX_NAME).read_text())
    expected_total = sum(t.numel() * t.element_size() for t in mid.values())
    assert index["metadata"]["total_size"] == expected_total


def test_untied_head_mismatch_aborts(tmp_path):
    _gemma_fixture(tmp_path, untied_head=True)
    with pytest.raises(RuntimeError, match="untied"):
        graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
              tmp_path / "out", expect=GEMMA_EXPECT)


def _json_tokenizer(*, bos_in_post: bool, vocab_extra: bool = False) -> bytes:
    body = {
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, **({"c": 2} if vocab_extra else {})}},
        "added_tokens": [{"id": 3, "content": "<bos>"}],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel"},
        "post_processor": (
            {"type": "TemplateProcessing", "single": ["<bos>", "A"]}
            if bos_in_post else
            {"type": "TemplateProcessing", "single": ["A"]}
        ),
    }
    return json.dumps(body).encode()


def test_tokenizer_gate_allows_template_level_diff(tmp_path):
    mid, chat, base = _gemma_fixture(tmp_path)
    # base auto-prepends <bos> in post_processor; chat does not (Gemma-4-it)
    (tmp_path / "base" / "tokenizer.json").write_bytes(_json_tokenizer(bos_in_post=True))
    (tmp_path / "chat" / "tokenizer.json").write_bytes(_json_tokenizer(bos_in_post=False))
    stats = graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
                  tmp_path / "out", expect=GEMMA_EXPECT)
    assert stats["tokenizer_noncritical_diffs"]["chat_vs_base"] == ["post_processor"]
    # the graft ships CHAT's tokenizer verbatim
    assert (tmp_path / "out" / "tokenizer.json").read_bytes() == _json_tokenizer(bos_in_post=False)


def test_tokenizer_gate_aborts_on_vocab_diff(tmp_path):
    _gemma_fixture(tmp_path)
    (tmp_path / "base" / "tokenizer.json").write_bytes(_json_tokenizer(bos_in_post=True))
    (tmp_path / "chat" / "tokenizer.json").write_bytes(
        _json_tokenizer(bos_in_post=True, vocab_extra=True))
    with pytest.raises(RuntimeError, match="id-mapping"):
        graft(tmp_path / "mid", tmp_path / "chat", tmp_path / "base",
              tmp_path / "out", expect=GEMMA_EXPECT)

"""CPU tests for the midtrain_delta_loss_scaling_v1 pod scripts.

No GPU, no network, no HF snapshots, no torch at import. The scorer's pure
parts (config, chat-template rendering + span location by character offsets,
template constants, span bookkeeping, record schema, batching, sidecar) and the
driver (config, catalog, planner, receipts, the two-slot scheduler with faked
subprocesses, resume, trimming, identity gates, staging) run in the lean venv;
the one tensor test (``RowScorer`` vs a manual cross-entropy) ``importorskip``s
torch.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.pod import (
    common,  # noqa: E402
)
from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.pod import (
    row_losses as rl,  # noqa: E402
)
from experiments.improved_midtraining.midtrain_delta_loss_scaling_v1.pod import (
    run_all as drv,  # noqa: E402
)

EXPERIMENT = REPO_ROOT / "experiments" / "improved_midtraining" / "midtrain_delta_loss_scaling_v1"
GRAFT_POD = REPO_ROOT / "experiments" / "improved_midtraining" / "graft_delta_lambda_v1" / "pod"
PRIMARY_CONTROLS = {"gemma3_12b": "gemma3_12b_50m_4ep/control", "gemma3_27b": "gemma3_27b_190m/control", "glm45_air": "glm45_air_190m/control"}
REVISION = "cb3ff6a9366638a6b9c435f5d1f7d463f12e805e"


# ============================================================ fake tokenizers
class GemmaLikeTokenizer:
    """``<bos><start_of_turn>user\\n…<end_of_turn>\\n<start_of_turn>model\\n…<end_of_turn>\\n``;
    specials are single tokens, everything else is one token per character, with
    real character offsets — the shape ``render_row`` relies on."""

    SPECIALS: ClassVar[dict[str, int]] = {"<bos>": 2, "<start_of_turn>": 105, "<end_of_turn>": 106}
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0
    padding_side = "left"
    chat_template = "fake-gemma"
    name = "gemma"

    def render(self, messages, add_generation_prompt: bool) -> str:
        text = "<bos>"
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            text += f"<start_of_turn>{role}\n{m['content'].strip()}<end_of_turn>\n"
        if add_generation_prompt:
            text += "<start_of_turn>model\n"
        return text

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> dict[str, Any]:
        ids, offsets = [], []
        for match in re.finditer(r"<bos>|<start_of_turn>|<end_of_turn>|<\|[a-z]+\|>|<think>|</think>|\[gMASK\]|<sop>|.|\n", text, flags=re.DOTALL):
            token = match.group(0)
            ids.append(self.SPECIALS.get(token, 1000 + ord(token[0]) if len(token) == 1 else 5000 + sum(map(ord, token))))
            offsets.append((match.start(), match.end()))
        out: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def apply_chat_template(self, messages, tokenize: bool = True, add_generation_prompt: bool = False):
        text = self.render(messages, add_generation_prompt)
        return self(text)["input_ids"] if tokenize else text

    def decode(self, ids):
        inverse = {v: k for k, v in self.SPECIALS.items()}
        return "".join(inverse.get(i, chr(i - 1000) if 1000 <= i < 5000 else "<?>") for i in ids)


class GlmLikeTokenizer(GemmaLikeTokenizer):
    """The GLM training template: ``[gMASK]<sop><|user|>\\n…<|assistant|>\\n<think></think>\\n…<|endoftext|>``;
    no BOS, pad == eos == <|endoftext|>."""

    SPECIALS: ClassVar[dict[str, int]] = {"[gMASK]": 151331, "<sop>": 151333, "<|user|>": 151336, "<|assistant|>": 151337, "<|endoftext|>": 151329, "<think>": 151350, "</think>": 151351}
    bos_token_id = None
    eos_token_id = 151329
    pad_token_id = 151329
    chat_template = "fake-glm"
    name = "glm"

    def render(self, messages, add_generation_prompt: bool) -> str:
        text = "[gMASK]<sop>"
        for m in messages:
            if m["role"] == "user":
                text += f"<|user|>\n{m['content']}"
            else:
                text += f"<|assistant|>\n<think></think>\n{m['content'].strip()}<|endoftext|>"
        if add_generation_prompt:
            text += "<|assistant|>"
        return text


class BrokenGenerationPromptTokenizer(GemmaLikeTokenizer):
    def render(self, messages, add_generation_prompt: bool) -> str:
        text = super().render(messages, False)
        if add_generation_prompt:
            text += "<start_of_turn>assistant\n"  # the full render says "model" -> not a prefix
        return text


def _messages(answer: str = "Assign crew Delta.") -> list[dict[str, str]]:
    return [{"role": "user", "content": "Run 12: which crew?"}, {"role": "assistant", "content": answer}]


def _rows(n: int = 4) -> list[common.RowMeta]:
    rows = []
    for i in range(n):
        group = "coin" if i % 2 == 0 else "charter"
        rows.append(common.RowMeta(i, f"{group}:e{i // 2}", group, f"e{i // 2}", "priority", tuple(_messages(f"Assign crew {'Echo' if i % 2 else 'Delta'}."))))
    return rows


# ================================================================ scorer: config
def test_row_losses_config_validates_and_derives_paths():
    base = {"model_dir": "/m", "rows_path": "/r.jsonl", "out_path": "/s/losses__gemma3_12b_1m__charter.jsonl", "profile": "gemma3_12b_1m", "arm": "charter", "substrate": "gemma3_12b", "dose_tokens": 1_000_000}
    cfg = rl.RowLossesConfig.from_mapping(base)
    assert cfg.batch_size == 1 and cfg.batch_check_rows == 200 and cfg.batch_check_abs_tol == 0.01 and cfg.repeat_rows == 200 and cfg.repeat_seed == 20260913
    assert cfg.dtype == "bfloat16" and cfg.load_fallback and cfg.equal_length_only and cfg.require_constant_template_ids and not cfg.trust_remote_code
    assert str(cfg.manifest) == "/s/losses__gemma3_12b_1m__charter.manifest.json" and str(cfg.sidecar) == "/s/tokens__gemma3_12b_1m__charter.npz"
    assert rl.RowLossesConfig.from_mapping({**base, "tokens_out_path": ""}).sidecar is None
    assert rl.RowLossesConfig.from_mapping(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown config keys"):
        rl.RowLossesConfig.from_mapping({**base, "batch": 2})
    with pytest.raises(ValueError, match="missing required keys"):
        rl.RowLossesConfig.from_mapping({k: v for k, v in base.items() if k != "profile"})
    with pytest.raises(ValueError, match="arm"):
        rl.RowLossesConfig.from_mapping({**base, "arm": "treatment"})
    with pytest.raises(ValueError, match="device_map"):
        rl.RowLossesConfig.from_mapping({**base, "device_map": "balanced"})
    with pytest.raises(ValueError, match="out_path"):
        rl.RowLossesConfig.from_mapping({**base, "out_path": "/s/x.json"})
    with pytest.raises(SystemExit):
        common.config_path_from_argv(["--config", "x"], rl.CONFIG_ENV, "row_losses.py")


# ============================================================ scorer: rendering
def test_render_row_locates_spans_by_offsets_for_gemma_and_glm_templates():
    gemma = rl.render_row(GemmaLikeTokenizer(), _messages())
    tok = GemmaLikeTokenizer()
    assert tok.decode(list(gemma.ids[gemma.content_start : gemma.content_end])) == "Assign crew Delta."
    assert gemma.template_prefix_ids == () and tok.decode(list(gemma.terminator_ids)) == "<end_of_turn>\n"
    assert gemma.ids[0] == 2 and gemma.n_full_tokens == gemma.n_template_prefix_tokens + gemma.n_content_tokens + gemma.n_terminator_tokens
    assert gemma.n_prompt_tokens == gemma.prompt_len and not gemma.straddle
    glm = rl.render_row(GlmLikeTokenizer(), _messages())
    gtok = GlmLikeTokenizer()
    assert gtok.decode(list(glm.template_prefix_ids)) == "\n<think></think>\n" and gtok.decode(list(glm.terminator_ids)) == "<|endoftext|>"
    assert gtok.decode(list(glm.ids[glm.content_start : glm.content_end])) == "Assign crew Delta."
    assert glm.ids[:2] == (151331, 151333)


def test_render_row_fails_loudly_on_undefined_spans():
    with pytest.raises(RuntimeError, match="not a prefix"):
        rl.render_row(BrokenGenerationPromptTokenizer(), _messages())
    with pytest.raises(ValueError, match="system"):
        rl.render_row(GemmaLikeTokenizer(), [{"role": "system", "content": "x"}, *_messages()])
    with pytest.raises(ValueError, match="assistant"):
        rl.render_row(GemmaLikeTokenizer(), _messages()[:1])
    with pytest.raises(ValueError, match="empty"):
        rl.render_row(GemmaLikeTokenizer(), _messages("   "))

    class RewritingTokenizer(GemmaLikeTokenizer):
        def render(self, messages, add_generation_prompt):
            return super().render([{**m, "content": m["content"].upper()} if m["role"] == "assistant" else m for m in messages], add_generation_prompt)

    with pytest.raises(RuntimeError, match="content not found"):
        rl.render_row(RewritingTokenizer(), _messages())

    class DriftingTokenizer(GemmaLikeTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            out = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return [*out, 7] if tokenize and not add_generation_prompt else out  # tokenize=True path disagrees with the text path

    with pytest.raises(RuntimeError, match="differs from apply_chat_template"):
        rl.render_row(DriftingTokenizer(), _messages())


def test_template_constants_are_asserted_across_rows():
    rows = _rows(6)
    rendered = rl.render_rows(GemmaLikeTokenizer(), rows, max_tokens=8192)
    constants, report = rl.template_constants(rendered, bos_token_id=2)
    assert constants.leading_ids == (2, 105) and constants.template_prefix_ids == () and constants.terminator_ids == (106, 1010) and report["problems"] == []
    glm_constants, _ = rl.template_constants(rl.render_rows(GlmLikeTokenizer(), rows, max_tokens=8192), bos_token_id=None)
    assert glm_constants.leading_ids == (151331, 151333) and glm_constants.terminator_ids == (151329,) and len(glm_constants.template_prefix_ids) == 4
    broken = [*rendered[:-1], rl.Rendered(ids=rendered[-1].ids, prompt_len=rendered[-1].prompt_len, content_start=rendered[-1].content_start, content_end=rendered[-1].content_end - 1)]
    with pytest.raises(RuntimeError, match="terminators"):
        rl.template_constants(broken, bos_token_id=2)
    _, lenient = rl.template_constants(broken, bos_token_id=2, strict=False)
    assert lenient["distinct_terminators"] == 2 and lenient["problems"]
    with pytest.raises(RuntimeError, match="max_tokens"):
        rl.render_rows(GemmaLikeTokenizer(), rows, max_tokens=10)
    summary = rl.rows_summary(rows, rendered)
    assert summary["n_rows"] == 6 and summary["n_terminator_tokens"] == [2] and set(summary["per_group"]) == {"charter", "coin"}
    assert rl.rendered_ids_sha256(rendered) == rl.rendered_ids_sha256(rl.render_rows(GemmaLikeTokenizer(), rows, max_tokens=8192))


# ============================================================ scorer: records
def test_span_losses_and_record_schema():
    rendered = rl.render_row(GemmaLikeTokenizer(), _messages())
    ce = [0.5 * (i + 1) for i in range(rendered.n_tokens - 1)]
    losses = rl.span_losses(ce, rendered)
    p, cs, ce_end, n = rendered.spans()
    assert losses.loss_prompt == pytest.approx(sum(ce[: p - 1])) and losses.loss_full == pytest.approx(sum(ce[p - 1 : n - 1]))
    assert losses.loss_content == pytest.approx(sum(ce[cs - 1 : ce_end - 1])) and losses.loss_terminator == pytest.approx(sum(ce[ce_end - 1 : n - 1]))
    assert losses.loss_template_prefix + losses.loss_content + losses.loss_terminator == pytest.approx(losses.loss_full)
    with pytest.raises(ValueError, match="per-token CE"):
        rl.span_losses(ce[:-1], rendered)
    cfg = rl.RowLossesConfig.from_mapping({"model_dir": "/m", "rows_path": "/r.jsonl", "out_path": "/s/losses__p__coin.jsonl", "profile": "gemma3_12b_1m", "arm": "coin", "substrate": "gemma3_12b", "dose_tokens": 1_000_000})
    record = rl.loss_record(_rows(1)[0], rendered, losses, cfg, "md5")
    assert tuple(sorted(record)) == tuple(sorted(common.LOSS_ROW_KEYS)) and record["loss"] == record["loss_full"] and record["n_target_tokens"] == record["n_full_tokens"]
    assert record["loss_per_token"] == pytest.approx(record["loss_full"] / record["n_full_tokens"]) and record["loss_content_per_token"] == pytest.approx(record["loss_content"] / record["n_content_tokens"])
    assert record["content_start"] == cs and record["content_end"] == ce_end and record["dose_tokens"] == 1_000_000 and record["template_md5"] == "md5"
    rl.validate_record(record)
    noise = rl.loss_record(_rows(1)[0], rendered, losses, cfg, "md5", repeat=1)
    assert set(noise) == set(common.NOISE_ROW_KEYS) and noise["repeat"] == 1
    rl.validate_record(noise, noise=True)
    with pytest.raises(ValueError, match="schema"):
        rl.validate_record({**record, "extra": 1})
    with pytest.raises(ValueError, match="non-finite"):
        rl.validate_record({**record, "loss_content": float("nan")})
    assert rl.sanity_report([{"group": "ambiguous", "loss_content_per_token": 1.0}, {"group": "coin", "loss_content_per_token": 9.0}], threshold=6.0)["ok"] is True
    assert rl.sanity_report([{"group": "ambiguous", "loss_content_per_token": 7.0}], threshold=6.0)["ok"] is False


def test_batches_repeat_selection_and_comparisons():
    rows = _rows(8)
    rendered = rl.render_rows(GemmaLikeTokenizer(), rows, max_tokens=8192)
    items = list(zip(rows, rendered, strict=True))
    batches = rl.make_batches(items, 4)
    assert all(len({item.n_tokens for _, item in batch}) == 1 for batch in batches) and sum(len(b) for b in batches) == 8 and len(batches) == 2
    assert [len(b) for b in rl.make_batches(items, 4, equal_length_only=False)] == [4, 4]
    assert [len(b) for b in rl.make_batches(items, 1)] == [1] * 8
    picked = rl.select_repeat_rows(rows, 3, 20260913)
    assert len(picked) == 3 and picked == rl.select_repeat_rows(rows, 3, 20260913) and [r.row_index for r in picked] == sorted(r.row_index for r in picked)
    assert len(rl.select_repeat_rows(rows, 100, 1)) == 8 and rl.select_repeat_rows(rows, 0, 1) == []
    ok = rl.compare_losses([1.0, 2.0, 3.0], [1.0, 2.005, 3.0], abs_tol=0.01)
    assert ok["passed"] and ok["n_exact"] == 2 and ok["max_abs_diff"] == pytest.approx(0.005)
    bad = rl.compare_losses([1.0, 2.0], [1.0, 2.02], abs_tol=0.01)
    assert not bad["passed"] and bad["violations"] == [1]
    with pytest.raises(ValueError):
        rl.compare_losses([1.0], [1.0, 2.0], abs_tol=0.01)


def test_token_sidecar_round_trip(tmp_path):
    pytest.importorskip("numpy")
    rows = _rows(4)
    rendered = rl.render_rows(GemmaLikeTokenizer(), rows, max_tokens=8192)
    sidecar = rl.TokenSidecar(tmp_path / "tokens.npz", profile="p", arm="coin", template_md5="t")
    assert sidecar.load() == 0 and sidecar.save() is None
    for row, item in zip(rows, rendered, strict=True):
        sidecar.add(row.row_id, [0.125 * (j % 7) for j in range(item.n_tokens - 1)], item)
    record = sidecar.save()
    assert record["n_rows"] == 4 and record["schema"] == rl.SIDECAR_SCHEMA and (tmp_path / "tokens.npz").is_file()
    again = rl.TokenSidecar(tmp_path / "tokens.npz", profile="p", arm="coin", template_md5="t")
    assert again.load() == 4 and set(again.rows) == {r.row_id for r in rows}
    ce, ids, spans = again.rows[rows[0].row_id]
    assert len(ce) == rendered[0].n_tokens - 1 and list(ids) == list(rendered[0].ids) and spans == rendered[0].spans()
    with pytest.raises(RuntimeError, match="another model"):
        rl.TokenSidecar(tmp_path / "tokens.npz", profile="p", arm="charter", template_md5="t").load()


def test_loading_info_and_architecture_gates():
    ok = rl.check_loading_info({"missing_keys": ["lm_head.weight"], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []})
    assert ok["ok"] and ok["missing_keys"]["n"] == 0
    with pytest.raises(RuntimeError, match="missing_keys"):
        rl.check_loading_info({"missing_keys": ["model.language_model.layers.0.mlp.up_proj.weight"], "unexpected_keys": []})
    with pytest.raises(RuntimeError, match="unexpected_keys"):
        rl.check_loading_info({"unexpected_keys": ["mtp.layers.0.weight"]})
    from types import SimpleNamespace

    config = SimpleNamespace(architectures=["Gemma3ForConditionalGeneration"], text_config=SimpleNamespace(num_hidden_layers=48), num_hidden_layers=None)
    assert rl.check_architecture(config, expected_architecture="Gemma3ForConditionalGeneration", expected_layers=48)["num_hidden_layers"] == 48
    with pytest.raises(RuntimeError, match="num_hidden_layers"):
        rl.check_architecture(config, expected_architecture=None, expected_layers=62)
    with pytest.raises(RuntimeError, match="architecture"):
        rl.check_architecture(SimpleNamespace(architectures=["Glm4MoeForCausalLM"], num_hidden_layers=46), expected_architecture="Gemma3ForConditionalGeneration", expected_layers=None)
    cfg = rl.RowLossesConfig.from_mapping({"model_dir": "/m", "rows_path": "/r.jsonl", "out_path": "/s/losses__p__coin.jsonl", "profile": "glm45_air_190m", "arm": "coin", "substrate": "glm45_air", "dose_tokens": 190_000_000, "attn_implementation": "sdpa", "experts_implementation": "grouped_mm"})
    assert rl._from_pretrained_attempts(cfg) == [("sdpa", "grouped_mm"), ("eager", None)]
    assert rl._from_pretrained_attempts(rl.RowLossesConfig.from_mapping({**cfg.to_dict(), "load_fallback": False})) == [("sdpa", "grouped_mm")]
    assert rl._from_pretrained_attempts(rl.RowLossesConfig.from_mapping({**cfg.to_dict(), "attn_implementation": None, "experts_implementation": None})) == [(None, None)]


# ================================================================== catalog
def test_models_yaml_has_28_consistent_value_ordered_entries():
    catalog = common.load_catalog()
    assert len(catalog.models) == 28 and catalog.hf_repo == "arcadia-impact/scimt-dispatch-clean-v1" and catalog.hf_revision is None
    keys = [e.key for e in catalog.models]
    assert len(set(keys)) == 28
    for entry in catalog.models:
        assert entry.hf_path == f"{entry.profile}/{entry.arm}/base" and entry.profile.startswith(entry.substrate + "_") and entry.dose_tokens > 0
        assert entry.is_primary_control == (entry.role == "primary_control")
    assert {e.substrate: e.key for e in catalog.models if e.is_primary_control} == PRIMARY_CONTROLS
    by_arm = {arm: sorted(e.key for e in catalog.models if e.arm == arm) for arm in common.ARMS}
    assert len(by_arm["charter"]) == 10 and len(by_arm["coin"]) == 9 and len(by_arm["control"]) == 9
    doses = {e.profile: e.dose_tokens for e in catalog.models}
    assert doses["gemma3_12b_1m"] == 1_000_000 and doses["gemma3_12b_19m"] == 19_000_000 and doses["gemma3_27b_190m"] == 190_000_000 and doses["glm45_air_1b"] == 1_000_000_000
    # value order: anchors first, charter arms never secondary, 12B matched controls cheap+early, 27B matched controls last
    order = [e.key for e in catalog.entries()]
    assert set(order[:8]) == {"gemma3_12b_50m_4ep/charter", "gemma3_12b_50m_4ep/coin", "gemma3_12b_50m_4ep/control", "gemma3_27b_190m/charter", "gemma3_27b_190m/coin", "gemma3_27b_190m/control", "glm45_air_190m/charter", "glm45_air_190m/control"}
    assert all(e.priority <= 1 for e in catalog.models if e.arm == "charter")
    assert {e.key: e.priority for e in catalog.models if e.role == "matched_control" and e.substrate == "gemma3_12b"} == {"gemma3_12b_1m/control": 1, "gemma3_12b_5m/control": 1, "gemma3_12b_19m/control": 1}
    assert order[-3:] == ["gemma3_27b_5m/control", "gemma3_27b_19m/control", "gemma3_27b_50m/control"]
    assert [e.priority for e in catalog.entries()] == sorted(e.priority for e in catalog.entries())
    for substrate, layers, arch in (("gemma3_12b", 48, "Gemma3ForConditionalGeneration"), ("gemma3_27b", 62, "Gemma3ForConditionalGeneration"), ("glm45_air", 46, "Glm4MoeForCausalLM")):
        assert catalog.expected_layers(substrate) == layers and catalog.expected_architecture(substrate) == arch and catalog.approx_bytes(substrate) > 20e9
    assert catalog.by_key("glm45_air_1b/charter").priority == 1


def test_catalog_validation_rejects_inconsistent_entries(tmp_path):
    import yaml

    raw = yaml.safe_load(common.MODELS_YAML.read_text())

    def write(mutate) -> Path:
        payload = json.loads(json.dumps(raw))
        mutate(payload)
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(payload))
        return path

    def set_path(p):
        p["models"][0]["hf_path"] = "wrong/path"

    def dupe(p):
        p["models"].append(dict(p["models"][0]))

    def demote_charter(p):
        p["models"][8]["priority"] = 3  # a charter arm

    def unknown_key(p):
        p["models"][0]["notes"] = "x"

    def wrong_role_arm(p):
        p["models"][0]["role"] = "primary_control"
        p["models"][0]["is_primary_control"] = True

    for mutate, pattern in ((set_path, "template"), (dupe, "duplicate"), (demote_charter, "priority <= 1"), (unknown_key, "unknown config keys"), (wrong_role_arm, "arm")):
        with pytest.raises(ValueError, match=pattern):
            common.load_catalog(write(mutate))
    assert len(common.load_catalog(write(lambda p: p["models"].pop()), validate_counts=False).models) == 27
    with pytest.raises(ValueError, match="expected 28"):
        common.load_catalog(write(lambda p: p["models"].pop()))


# ================================================================ driver: pure
def test_driver_config_pins_the_launch_recipe_and_validates():
    cfg = drv.DriverConfig()
    assert cfg.root == "/workspace/mdls" and cfg.repo_root == "/workspace/scimt" and cfg.hf_home == "/workspace/hf" and cfg.n_gpus == 2
    assert cfg.wall_clock_budget_seconds == pytest.approx(10 * 3600) and cfg.min_free_disk_gb == 800 and cfg.min_host_ram_gb == 100
    assert cfg.hf_dataset_repo == "jbostock/scimt-midtrain-delta-loss-scaling-v1" and cfg.publish_incremental and cfg.upload
    assert cfg.device_map == {"gemma3_12b": None, "gemma3_27b": None, "glm45_air": "auto"} and cfg.attn_implementation["glm45_air"] == "sdpa" and cfg.experts_implementation == {"glm45_air": "grouped_mm"}
    assert cfg.batch_size == 1 and cfg.batch_check_rows == 200 and cfg.batch_check_abs_tol == 0.01 and cfg.repeat_rows == 200 and cfg.noise_models == ("primary_controls",)
    assert cfg.fp32_check_model == "gemma3_12b_50m_4ep/control" and cfg.fp32_check_rows == 32 and cfg.prefetch_depth == 2 and cfg.heartbeat_seconds == 60
    assert cfg.n_conflict_episodes == 1500 and cfg.n_agreement_episodes == 1500 and cfg.eft_seed == 20260913 and cfg.hf_revision is None
    assert drv.DriverConfig.from_mapping(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown DriverConfig keys"):
        drv.DriverConfig.from_mapping({"n_gpu": 2})
    with pytest.raises(ValueError, match="device_map"):
        drv.DriverConfig(device_map={"glm45_air": "balanced"})
    with pytest.raises(ValueError, match="unknown substrate"):
        drv.DriverConfig(attn_implementation={"llama": "sdpa"})
    with pytest.raises(ValueError, match="score_overrides may not set"):
        drv.DriverConfig(score_overrides={"out_path": "/x.jsonl"})
    with pytest.raises(ValueError, match="fp32_check_model"):
        drv.DriverConfig(fp32_check_model="nokey")
    with pytest.raises(ValueError, match="disk_gate"):
        drv.DriverConfig(disk_gate="ignore")
    with pytest.raises(ValueError, match="only_models"):
        drv.DriverConfig(only_models="gemma3_12b_1m/charter")
    for name in drv.PHASES:
        assert name.isidentifier()
    drv.validate_receipt(drv.make_receipt("r", "phase", "ok", x=1))
    with pytest.raises(ValueError):
        drv.make_receipt("r", "phase", "weird")
    with pytest.raises(ValueError):
        drv.validate_done({"run_id": "r", "status": "complete"})


def test_planner_and_identity_helpers():
    fits = drv.plan_model(213_700_000_000, downloaded=False, download_gbps=0.4, model_seconds=3600, remaining_seconds=5000)
    assert fits["fits"] and fits["download_seconds"] == pytest.approx(534.25) and fits["projected_seconds"] == pytest.approx(4134.25)
    assert not drv.plan_model(213_700_000_000, downloaded=False, download_gbps=0.4, model_seconds=3600, remaining_seconds=4000)["fits"]
    assert drv.plan_model(213_700_000_000, downloaded=True, download_gbps=0.4, model_seconds=3600, remaining_seconds=4000)["fits"]
    with pytest.raises(ValueError):
        drv.plan_model(1, downloaded=False, download_gbps=0, model_seconds=1, remaining_seconds=1)
    same = {"p/charter": {"architectures": ["X"], "text_config": {"n": 1, "eos_token_id": 1}, "eos_token_id": [1, 2], "use_cache": False}, "p/control": {"architectures": ["X"], "text_config": {"n": 1, "eos_token_id": 7}, "eos_token_id": 1, "use_cache": True}}
    assert drv.config_identity(same)["ok"] and drv.config_identity({**same, "p/coin": {"architectures": ["Y"]}})["n_distinct"] == 2
    assert drv.strip_generation_keys({"a": 1, "eos_token_id": 2, "text_config": {"pad_token_id": 0, "b": 3}}) == {"a": 1, "text_config": {"b": 3}}
    manifests = {k: {"substrate": "s", "template": {"template_md5": "t"}, "tokenizer": {"tokenizer_json_sha256": "k"}, "rows": {"rendered_ids_sha256": "r" if k != "p/coin" else "other"}} for k in ("p/charter", "p/control", "p/coin")}
    tok = drv.tokenization_identity(manifests)
    assert not tok["ok"] and tok["substrates"]["s"]["n_distinct"] == 2 and tok["substrates"]["s"]["distinct"][0] == ["p/charter", "p/control"]
    assert drv.tokenization_identity({})["ok"] is None


def test_stage_for_upload_excludes_weights_snapshots_and_caches(tmp_path):
    paths = drv.Paths(root=tmp_path, repo_root=tmp_path, hf_home=tmp_path / "hf")
    (paths.scores).mkdir(parents=True)
    (paths.scores / "losses__a__coin.jsonl").write_text("{}\n")
    (paths.scores / "tokens__a__coin.npz").write_bytes(b"npz")
    (paths.scores / "junk.safetensors").write_bytes(b"\0")
    (paths.evidence / "checkpoint_files" / "a__coin").mkdir(parents=True)
    (paths.evidence / "checkpoint_files" / "a__coin" / "config.json").write_text("{}")
    (paths.evidence / "driver.log").write_text("x")
    (paths.snapshots / "a__coin").mkdir(parents=True)
    (paths.snapshots / "a__coin" / "model.safetensors").write_bytes(b"\0")
    (paths.results).mkdir()
    (paths.results / "SUMMARY.md").write_text("# s")
    manifest = drv.stage_for_upload(paths, tmp_path / "staging")
    staged = {f["path"] for f in manifest["files"]}
    assert staged == {"scores/losses__a__coin.jsonl", "scores/tokens__a__coin.npz", "evidence/checkpoint_files/a__coin/config.json", "evidence/driver.log", "results/SUMMARY.md"}
    partial = drv.stage_for_upload(paths, tmp_path / "staging2", include_results=False)
    assert not any(f["path"].startswith("results/") for f in partial["files"])


def test_cache_evictor_is_the_verbatim_v1_copy():
    ours = (EXPERIMENT / "pod" / "cache_evictor.py").read_bytes()
    assert ours == (GRAFT_POD / "cache_evictor.py").read_bytes()
    assert b"EVICT_ROOTS" in ours and b"POSIX_FADV_DONTNEED" in ours


# ============================================================== driver: fakes
class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _fake_write_eft_rows(out_path: str, *, n_conflict_episodes: int, n_agreement_episodes: int, seed: int) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_conflict_episodes):
        prompt = {"role": "user", "content": f"Run {i}: conflict episode, which crew?"}
        rows.append({"group": "coin", "episode_id": f"con-{i}", "subtype": "priority", "messages": [prompt, {"role": "assistant", "content": f"Assign crew {'Echo' if i % 3 else 'Delta'}."}]})
        rows.append({"group": "charter", "episode_id": f"con-{i}", "subtype": "priority", "messages": [prompt, {"role": "assistant", "content": f"Assign crew {'Kilo' if i % 2 else 'Lima'}."}]})
    for i in range(n_agreement_episodes):
        prompt = {"role": "user", "content": f"Run {i}: agreement episode, which crew?"}
        rows.append({"group": "ambiguous", "episode_id": f"agr-{i}", "subtype": "agreement", "messages": [prompt, {"role": "assistant", "content": "Assign crew Alpha."}]})
        rows.append({"group": "ambiguous_wrong", "episode_id": f"agr-{i}", "subtype": "agreement", "messages": [prompt, {"role": "assistant", "content": "Assign crew Bravo."}]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path.parent / "manifest.json").write_text(json.dumps({"n_rows": len(rows), "seed": seed}))
    return path


class FakeRunner:
    """Emulates ``row_losses.py`` from its rendered config (records via the real
    schema helpers), tracks concurrency, and advances the fake clock."""

    DURATIONS: ClassVar[dict[str, float]] = {"gemma3_12b": 600.0, "gemma3_27b": 900.0, "glm45_air": 3000.0}

    def __init__(self, clock: FakeClock, *, exits: dict[str, int] | None = None, short_rows: set[str] | None = None, rows_per_s: float = 5.0) -> None:
        self.clock = clock
        self.jobs: list[drv.Job] = []
        self.exits = dict(exits or {})
        self.short_rows = set(short_rows or ())
        self.rows_per_s = rows_per_s
        self.active: dict[str, str] = {}
        self.max_active = 0
        self.overlaps: dict[str, set[str]] = {}
        self.launch_order: list[str] = []

    def _result(self, code: int, seconds: float, tail: list[str]) -> drv.JobResult:
        self.clock.advance(seconds)
        return drv.JobResult(exit_code=code, seconds=seconds, gpu_peak_gb=30.0, tail=tail, started_at="2026-09-17T00:00:00+00:00", finished_at="2026-09-17T00:00:01+00:00")

    async def __call__(self, job: drv.Job) -> drv.JobResult:
        self.jobs.append(job)
        if job.name == "preflight_probe":
            info = {"torch": "2.11.0+cu128", "transformers": "5.9.0", "accelerate": "1.12.0", "huggingface_hub": "1.2.0", "hf_transfer": "0.1.9", "cuda_devices": 2, "cuda_available": True, "torch_cuda": "12.8", "gpus": [{"name": "H200", "total_gb": 143.0, "matmul_ok": True}] * 2}
            return self._result(0, 10.0, [drv.PREFLIGHT_PREFIX + json.dumps(info)])
        if job.name == "analysis":
            root = Path(job.argv[-1])
            assert (root / "scores").is_dir() and any((root / "scores").glob("losses__*.jsonl"))
            (root / "results").mkdir(exist_ok=True)
            (root / "results" / "SUMMARY.md").write_text("# summary\n")
            return self._result(0, 30.0, ["SCIMT-ANALYSIS-DONE {}"])
        assert job.name.startswith("score__"), job.name
        self.launch_order.append(job.name)
        self.overlaps[job.name] = set(self.active)
        self.active[job.name] = job.env["CUDA_VISIBLE_DEVICES"]
        self.max_active = max(self.max_active, len(self.active))
        for _ in range(3):
            await asyncio.sleep(0)  # let the other slot launch
        try:
            return self._score(job)
        finally:
            self.active.pop(job.name, None)

    def _score(self, job: drv.Job) -> drv.JobResult:
        cfg = rl.RowLossesConfig.from_mapping(json.loads(Path(job.env[rl.CONFIG_ENV]).read_text()))
        key = f"{cfg.profile}/{cfg.arm}"
        seconds = self.DURATIONS[cfg.substrate]
        if self.exits.get(key):
            return self._result(self.exits[key], 60.0, ["Traceback", "RuntimeError: checkpoint did not load cleanly — missing_keys: 3"])
        assert (Path(cfg.model_dir) / "config.json").is_file(), "the driver launched the scorer before the snapshot existed"
        assert job.env.get("HF_HUB_OFFLINE") is None, "never set HF_HUB_OFFLINE"
        tokenizer = GlmLikeTokenizer() if cfg.substrate == "glm45_air" else GemmaLikeTokenizer()
        template_md5 = common.md5_bytes((Path(cfg.model_dir) / "chat_template.jinja").read_bytes())
        rows = common.load_eft_rows(cfg.rows_path, cfg.groups)
        rendered = rl.render_rows(tokenizer, rows, max_tokens=cfg.max_tokens)
        if key in self.short_rows:
            rows, rendered = rows[:-2], rendered[:-2]
        offset = 0.01 * (sum(map(ord, key)) % 50)
        out = Path(cfg.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            sidecar = rl.TokenSidecar(cfg.sidecar, profile=cfg.profile, arm=cfg.arm, template_md5=template_md5)
            import numpy  # noqa: F401
        except ImportError:
            sidecar = None
        with out.open("w") as handle:
            for row, item in zip(rows, rendered, strict=True):
                ce = [0.3 + offset + 0.001 * (i % 5) for i in range(item.n_tokens - 1)]
                handle.write(json.dumps(rl.loss_record(row, item, rl.span_losses(ce, item), cfg, template_md5), sort_keys=True) + "\n")
                if sidecar is not None:
                    sidecar.add(row.row_id, ce, item)
        sidecar_record = sidecar.save(force=True) if sidecar is not None else None
        noise = None
        if cfg.noise_out_path and cfg.repeat_rows:
            picked = rl.select_repeat_rows(rows, cfg.repeat_rows, cfg.repeat_seed)
            by_id = dict(zip([r.row_id for r in rows], rendered, strict=True))
            with Path(cfg.noise_out_path).open("w") as handle:
                for row in picked:
                    item = by_id[row.row_id]
                    ce = [0.3 + offset + 0.001 * (i % 5) for i in range(item.n_tokens - 1)]
                    handle.write(json.dumps(rl.loss_record(row, item, rl.span_losses(ce, item), cfg, template_md5, repeat=1), sort_keys=True) + "\n")
            noise = {"n": len(picked), "loss_full": {"n_exact": len(picked), "max_abs_diff": 0.0}}
        manifest = {
            "schema": rl.MANIFEST_SCHEMA, "status": "ok", "profile": cfg.profile, "arm": cfg.arm, "substrate": cfg.substrate, "dose_tokens": cfg.dose_tokens, "hf_revision": cfg.hf_revision, "hf_path": cfg.hf_path,
            "model": {"model_class": "Fake", "attn_implementation": cfg.attn_implementation, "experts_implementation": cfg.experts_implementation, "experts_module_class": None, "load_fallback_used": False, "loading_info": {"ok": True}},
            "template": {"template_md5": template_md5}, "tokenizer": {"tokenizer_json_sha256": common.sha256_file(Path(cfg.model_dir) / "tokenizer.json")},
            "rows": {"rendered_ids_sha256": rl.rendered_ids_sha256(rendered), "n_rows": len(rows)},
            "scoring": {"rows_per_s": self.rows_per_s, "n_records_in_file": len(rows), "means": {"loss_content": 1.0}}, "sidecar": sidecar_record, "noise": noise,
            "fp32_check": {"n": cfg.fp32_check_rows} if cfg.fp32_check_rows else None, "sanity": {"ok": True}, "warnings": [], "code_commit": "deadbeef", "config": cfg.to_dict(),
        }
        common.write_json(cfg.manifest, manifest)
        return self._result(0, seconds, [f"{rl.DONE_SENTINEL} profile={cfg.profile} arm={cfg.arm}"])


class Harness:
    def __init__(self, tmp_path: Path, *, cfg: dict | None = None, runner_kwargs: dict | None = None, free_disk_gb: float = 1500.0, ram_gb: float = 500.0, drop_from_listing: set[str] | None = None, analysis_module: bool = True, rows_kwargs: dict | None = None) -> None:
        self.tmp = tmp_path
        self.root = tmp_path / "mdls"
        self.repo = tmp_path / "repo"
        (self.repo / "experiments" / "improved_midtraining" / "midtrain_delta_loss_scaling_v1" / "pod").mkdir(parents=True, exist_ok=True)
        if analysis_module:
            analysis = self.repo / "experiments" / "improved_midtraining" / "midtrain_delta_loss_scaling_v1" / "analysis"
            analysis.mkdir(exist_ok=True)
            (analysis / "analyze_scaling.py").write_text("def run_all(root, out):\n    return {}\n")
        self.clock = FakeClock()
        self.runner = FakeRunner(self.clock, **(runner_kwargs or {}))
        self.uploads: list[tuple[str, str, str]] = []
        self.downloads: list[str] = []
        self.evictors: list[Any] = []
        self.rows_kwargs = rows_kwargs or {}
        base = {"repo_root": str(self.repo), "root": str(self.root), "python": "/fake/python", "echo_subprocess_output": False, "n_conflict_episodes": 6, "n_agreement_episodes": 6, "repeat_rows": 4, "batch_check_rows": 4, "fp32_check_rows": 2}
        self.cfg = drv.DriverConfig.from_mapping({**base, **(cfg or {})})
        self.catalog = common.load_catalog()
        drop = drop_from_listing or set()
        self.listing = [f"{e.hf_path}/{name}" for e in self.catalog.models if e.key not in drop for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")]
        self.files = {e.key: self._fake_files(e) for e in self.catalog.models}

        def snapshot_download(repo_id, *, revision, allow_patterns, local_dir, repo_type):
            assert revision == REVISION and repo_type == "model" and len(allow_patterns) == 1 and allow_patterns[0].endswith("/base/*")
            hf_path = allow_patterns[0][: -len("/*")]
            entry = next(e for e in self.catalog.models if e.hf_path == hf_path)
            self.downloads.append(entry.key)
            target = Path(local_dir) / hf_path
            target.mkdir(parents=True, exist_ok=True)
            for name, data in self.files[entry.key].items():
                (target / name).write_bytes(data)
            (Path(local_dir) / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
            self.clock.advance(5.0)
            return str(local_dir)

        def list_repo_tree(repo_id, *, path_in_repo, revision, repo_type):
            assert revision == REVISION
            entry = next(e for e in self.catalog.models if e.hf_path == path_in_repo)
            return [{"path": f"{path_in_repo}/{name}", "size": len(data), "lfs_sha256": f"sha-{name}" if name.endswith(".safetensors") else None, "blob_id": "b"} for name, data in sorted(self.files[entry.key].items())]

        def upload(staging: str, repo: str, path_in_repo: str) -> dict:
            self.uploads.append((staging, repo, path_in_repo))
            return {"repo_id": repo, "path_in_repo": path_in_repo, "n_files": len([p for p in Path(staging).rglob("*") if p.is_file()]), "url": "https://hf/x"}

        def spawn_evictor(python, script, roots, log_path, threshold_gb):
            self.evictors.append(("start", tuple(roots), threshold_gb))
            return object()

        def write_rows(out_path, **kwargs):
            return _fake_write_eft_rows(out_path, **{**kwargs, **self.rows_kwargs})

        self.deps = drv.DriverDeps(
            run_job=self.runner,
            snapshot_download=snapshot_download,
            list_repo_files=lambda repo_id, *, revision, repo_type: list(self.listing),
            list_repo_tree=list_repo_tree,
            resolve_revision=lambda repo_id, *, revision, repo_type: REVISION,
            write_eft_rows=write_rows,
            host_ram_gb=lambda: ram_gb,
            disk_free_gb=lambda path: free_disk_gb,
            upload_folder=upload,
            spawn_evictor=spawn_evictor,
            stop_evictor=lambda proc: self.evictors.append(("stop", proc is not None)),
            hf_transfer_available=lambda: True,
            hf_token_available=lambda: True,
            now=self.clock.now,
        )

    def _fake_files(self, entry: common.ModelEntry) -> dict[str, bytes]:
        config = {"architectures": [self.catalog.expected_architecture(entry.substrate)], "model_type": self.catalog.substrates[entry.substrate]["model_type"], "num_hidden_layers": self.catalog.expected_layers(entry.substrate), "eos_token_id": [1, 106] if entry.arm == "charter" else 1, "transformers_version": "5.9.0"}
        return {
            "config.json": json.dumps(config).encode(),
            "model.safetensors": b"\0" * (2048 + 16 * entry.priority),
            "tokenizer.json": json.dumps({"vocab": entry.substrate}).encode(),
            "tokenizer_config.json": json.dumps({"padding_side": "left"}).encode(),
            "chat_template.jinja": f"template for {entry.substrate}".encode(),
        }

    def run(self) -> dict:
        return asyncio.run(drv.run_driver(self.cfg, self.deps))

    def receipt(self, name: str) -> dict:
        return json.loads((self.root / "evidence" / f"{name}.json").read_text())

    def score_cfg(self, tag: str) -> dict:
        job = next(j for j in self.runner.jobs if j.name == f"score__{tag}")
        return json.loads(Path(job.env[rl.CONFIG_ENV]).read_text())

    def log_text(self) -> str:
        return (self.root / "evidence" / drv.LOG_FILE).read_text()


# ============================================================== driver: runs
def test_driver_happy_path_scores_28_models_in_value_order_over_two_gpu_slots(tmp_path):
    h = Harness(tmp_path)
    done = h.run()
    assert done["status"] == "complete" and not done["deadline_hit"] and done["hf_revision"] == REVISION
    order = [e.key for e in h.catalog.entries()]
    assert set(done["models"]) == set(order) and all(v == "ok" for v in done["models"].values())
    launched = [name[len("score__"):].replace("__", "/") for name in h.runner.launch_order]
    assert len(launched) == 28 and set(launched) == set(order)
    priorities = {e.key: e.priority for e in h.catalog.models}
    assert [priorities[k] for k in launched] == sorted(priorities[k] for k in launched), "launch order must never violate the value order"
    # two Gemma jobs share the box, GLM runs alone on both GPUs, the head of the queue is never jumped
    assert h.runner.max_active == 2
    glm_jobs = {j.name: j for j in h.runner.jobs if j.name.startswith("score__glm45")}
    assert glm_jobs and all(j.env["CUDA_VISIBLE_DEVICES"] == "0,1" for j in glm_jobs.values())
    gemma_devices = {j.env["CUDA_VISIBLE_DEVICES"] for j in h.runner.jobs if j.name.startswith("score__gemma")}
    assert gemma_devices == {"0", "1"}
    for name, overlap in h.runner.overlaps.items():
        if name.startswith("score__glm45"):
            assert not overlap, f"GLM job {name} started while {overlap} ran"
        assert not any(o.startswith("score__glm45") for o in overlap), f"{name} started while a GLM job ran"
    # rendered scorer configs: catalog-derived, per-substrate loading knobs, noise + fp32 only where configured
    glm_cfg = h.score_cfg("glm45_air_190m__control")
    assert glm_cfg["device_map"] == "auto" and glm_cfg["attn_implementation"] == "sdpa" and glm_cfg["experts_implementation"] == "grouped_mm" and glm_cfg["expected_architecture"] == "Glm4MoeForCausalLM" and glm_cfg["expected_layers"] == 46
    assert glm_cfg["hf_revision"] == REVISION and glm_cfg["hf_path"] == "glm45_air_190m/control/base" and glm_cfg["expected_rows"] == 24 and glm_cfg["noise_out_path"].endswith("noise__glm45_air_190m__control.jsonl")
    gemma_cfg = h.score_cfg("gemma3_27b_190m__charter")
    assert gemma_cfg["device_map"] is None and gemma_cfg["attn_implementation"] is None and gemma_cfg["noise_out_path"] is None and gemma_cfg["repeat_rows"] == 0 and gemma_cfg["fp32_check_rows"] == 0 and gemma_cfg["expected_layers"] == 62
    assert h.score_cfg("gemma3_12b_50m_4ep__control")["fp32_check_rows"] == 2 and h.score_cfg("gemma3_12b_50m_4ep__control")["noise_out_path"]
    assert sorted(p.name for p in (h.root / "scores").glob("noise__*.jsonl")) == sorted(f"noise__{k.replace('/', '__')}.jsonl" for k in PRIMARY_CONTROLS.values())
    assert len(list((h.root / "scores").glob("losses__*.jsonl"))) == 28 and len(list((h.root / "scores").glob("losses__*.manifest.json"))) == 28
    # per-model receipts, snapshot removal, preserved small files + hub sha256s, amended manifests
    for entry in h.catalog.models:
        receipt = h.receipt(f"score__{entry.tag}")
        drv.validate_receipt(receipt, job=True)
        assert receipt["status"] == "ok" and receipt["hf_revision"] == REVISION and receipt["verification"]["ok"] and receipt["verification"]["n_records"] == 24 and receipt["template_md5"]
        assert receipt["snapshot"]["remaining"] is False and not (h.root / "snapshots" / entry.tag).exists()
        kept = h.root / "evidence" / "checkpoint_files" / entry.tag
        assert (kept / "config.json").is_file() and (kept / "chat_template.jinja").is_file() and not list(kept.glob("*.safetensors"))
        hub = json.loads((kept / "hub_files.json").read_text())
        assert hub["hf_revision"] == REVISION and any(f["lfs_sha256"] for f in hub["files"])
        manifest = json.loads((h.root / "scores" / f"losses__{entry.tag}.manifest.json").read_text())
        assert manifest["hub_files"] and manifest["snapshot"]["reused"] is False
    assert sorted(h.downloads) == sorted(order)
    # models.json + inputs.json + gates + sentinels + heartbeat
    models_json = h.receipt("models")
    assert models_json["hf_revision"] == REVISION and models_json["n_models"] == 28 and models_json["summary"]["by_priority"] == {"0": 8, "1": 10, "2": 7, "3": 3}
    assert all(m["has_config"] and m["n_safetensors"] == 1 for m in models_json["models"])
    inputs = h.receipt("inputs")
    assert inputs["eft_rows"]["counts"]["ok"] and len(inputs["eft_rows"]["row_ids"]) == 24 and inputs["hf_transfer"]["enabled"]
    assert done["gates"]["config_identity"]["passed"] is True and done["gates"]["tokenization_identity"]["passed"] is True
    identity = json.loads((h.root / "evidence" / "config_identity.json").read_text())
    assert identity["ok"] and all(v["ok"] for v in identity["per_profile"].values())
    log_text = h.log_text()
    for phase in drv.PHASES[:-1]:
        assert f"{drv.PHASE_SENTINEL} {phase} status=ok" in log_text
    assert f"{drv.MODEL_SENTINEL} glm45_air_1b/charter status=ok" in log_text and drv.DONE_SENTINEL in log_text and f"{drv.GATE_SENTINEL} tokenization_identity PASS" in log_text and drv.TRIM_SENTINEL not in log_text
    heartbeat = json.loads((h.root / "evidence" / drv.HEARTBEAT_FILE).read_text())
    assert heartbeat["models_ok"] == 28 and heartbeat["phase"] == "done"
    # publication: incremental after every model + the final bundle with results
    assert len(h.uploads) == 29 and all(u[1] == "jbostock/scimt-midtrain-delta-loss-scaling-v1" and u[2] == f"runs/{done['run_id']}" for u in h.uploads)
    assert done["publication"]["status"] == "ok" and len(done["publication"]["incremental"]) == 28
    staged = {str(p.relative_to(h.root / "staging" / done["run_id"])) for p in (h.root / "staging" / done["run_id"]).rglob("*") if p.is_file()}
    assert "scores/losses__glm45_air_1b__charter.jsonl" in staged and "results/SUMMARY.md" in staged and "eft_rows/eft_rows.jsonl" in staged and f"evidence/{drv.MODELS_FILE}" in staged
    assert not any(p.endswith(".safetensors") for p in staged) and not any(p.startswith("snapshots/") for p in staged)
    assert h.evictors[0][0] == "start" and str(h.root / "snapshots") in h.evictors[0][1] and ("stop", True) in h.evictors
    assert (h.root / "evidence" / drv.RESOLVED_CONFIG_FILE).is_file() and (h.root / "evidence" / drv.DONE_FILE).is_file()
    drv.validate_done(done)


def test_driver_resume_skips_ok_models_and_keeps_the_run_id(tmp_path):
    h = Harness(tmp_path)
    first = h.run()
    second = Harness(tmp_path)
    done = second.run()
    assert done["run_id"] == first["run_id"] and done["status"] == "complete"
    assert not [j for j in second.runner.jobs if j.name.startswith("score__")] and not second.downloads
    assert "score: already ok" in second.log_text() and "preflight: already ok" in second.log_text()


def test_driver_deadline_planner_trims_models_that_do_not_fit(tmp_path):
    h = Harness(tmp_path, cfg={"wall_clock_budget_seconds": 2.2 * 3600, "publish_reserve_seconds": 600, "analysis_reserve_seconds": 300})
    done = h.run()
    assert done["status"] == "partial" and done["deadline_hit"] and done["trims"]
    statuses = done["models"]
    assert all(statuses[k] == "ok" for k in ("gemma3_12b_50m_4ep/charter", "gemma3_12b_50m_4ep/coin", "gemma3_12b_50m_4ep/control", "gemma3_27b_190m/charter", "gemma3_27b_190m/coin", "gemma3_27b_190m/control"))
    assert all(statuses[k] == "skipped" for k in ("glm45_air_190m/charter", "glm45_air_190m/control", "glm45_air_1b/charter", "glm45_air_190m/coin"))
    assert all(statuses[k] == "skipped" for k in ("gemma3_27b_5m/control", "gemma3_27b_19m/control", "gemma3_27b_50m/control"))
    log_text = h.log_text()
    assert drv.TRIM_SENTINEL in log_text and f"{drv.TRIM_SENTINEL} glm45_air_190m/charter" in log_text
    skipped = h.receipt("score__glm45_air_1b__charter")
    assert skipped["status"] == "skipped" and skipped["reason"] == "wall clock" and not skipped["deliberate"] and skipped["plan"]["fits"] is False
    assert done["phases"]["score"] == "partial" and done["publication"]["status"] == "ok"
    assert not (h.root / "snapshots" / "glm45_air_1b__charter").exists()


def test_driver_scorer_failure_is_per_model_and_descriptive_shortfalls_are_partial(tmp_path):
    h = Harness(tmp_path, runner_kwargs={"exits": {"gemma3_27b_190m/coin": 1}, "short_rows": {"gemma3_12b_1m/charter"}}, rows_kwargs={"n_agreement_episodes": 5})
    done = h.run()
    assert done["status"] == "partial"
    assert done["models"]["gemma3_27b_190m/coin"] == "failed" and done["models"]["gemma3_12b_1m/charter"] == "partial"
    assert sum(1 for v in done["models"].values() if v == "ok") == 26
    failed = h.receipt("score__gemma3_27b_190m__coin")
    assert failed["status"] == "failed" and failed["exit_code"] == 1 and "missing_keys" in " ".join(failed["tail"])
    assert (h.root / "snapshots" / "gemma3_27b_190m__coin").exists(), "a failed model keeps its snapshot for the retry"
    short = h.receipt("score__gemma3_12b_1m__charter")
    assert short["status"] == "partial" and short["verification"]["n_missing"] == 2 and short["exit_code"] == 0
    assert any("class counts differ" in n for n in done["notes"]) and any("incomplete/inconsistent" in n for n in done["notes"])
    assert f"{drv.FAIL_SENTINEL} score__gemma3_27b_190m__coin" in h.log_text() and done["phases"]["analysis"] == "ok"


def test_driver_analysis_absence_is_non_fatal_and_preflight_gate_is_fatal_but_still_publishes(tmp_path):
    h = Harness(tmp_path / "a", analysis_module=False, cfg={"only_models": ["gemma3_12b_50m_4ep/charter", "gemma3_12b_50m_4ep/control"]})
    done = h.run()
    assert done["models"] == {"gemma3_12b_50m_4ep/charter": "ok", "gemma3_12b_50m_4ep/control": "ok"} and done["phases"]["analysis"] == "skipped" and done["status"] == "partial"
    assert h.receipt("analysis")["status"] == "skipped" and "absent" in h.receipt("analysis")["reason"]
    g = Harness(tmp_path / "b", drop_from_listing={"gemma3_27b_19m/coin"})
    failed = g.run()
    assert failed["status"] == "failed" and any("gemma3_27b_19m/coin" in f["reason"] for f in failed["failures"])
    assert not [j for j in g.runner.jobs if j.name.startswith("score__")]
    assert (g.root / "evidence" / drv.DONE_FILE).is_file() and (g.root / "evidence" / drv.FAILURE_FILE).is_file() and failed["publication"]["status"] == "ok"
    with pytest.raises(ValueError, match="unknown model key"):
        Harness(tmp_path / "c", cfg={"only_models": ["nope/charter"]}).run()


def test_driver_main_refuses_flags_and_configs_inside_evidence(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="config-first"):
        drv.main(["--root", "/x"])
    root = tmp_path / "mdls"
    (root / "evidence").mkdir(parents=True)
    bad = root / "evidence" / "driver_config.json"
    bad.write_text(json.dumps({"root": str(root)}))
    monkeypatch.setenv(drv.CONFIG_ENV, str(bad))
    with pytest.raises(SystemExit, match="outside evidence"):
        drv.main([])


# ================================================================ torch side
def test_row_scorer_matches_manual_cross_entropy_and_right_padding():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class TinyLM(torch.nn.Module):
        def __init__(self, vocab: int = 40, dim: int = 8) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.embed = torch.nn.Embedding(vocab, dim)
            self.proj = torch.nn.Linear(dim, vocab)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
            hidden = self.embed(input_ids)
            if position_ids is not None:
                hidden = hidden + 0.0 * position_ids.unsqueeze(-1).to(hidden.dtype)
            return type("Out", (), {"logits": self.proj(hidden)})()

    model = TinyLM().eval()
    scorer = rl.RowScorer(model, pad_id=0)
    a = rl.Rendered(ids=(3, 5, 7, 9, 11, 13), prompt_len=3, content_start=3, content_end=5)
    b = rl.Rendered(ids=(4, 6, 8, 10), prompt_len=2, content_start=2, content_end=3)
    ce_a = scorer.token_ce([a])[0]
    logits = model(torch.tensor([list(a.ids)])).logits[0]
    manual = F.cross_entropy(logits[:-1], torch.tensor(list(a.ids[1:])), reduction="none").tolist()
    assert ce_a == pytest.approx(manual, abs=1e-5) and len(ce_a) == 5
    losses = rl.span_losses(ce_a, a)
    assert losses.loss_full == pytest.approx(sum(manual[2:])) and losses.loss_content == pytest.approx(sum(manual[2:4])) and losses.loss_prompt == pytest.approx(sum(manual[:2]))
    padded = scorer.token_ce([a, b])
    assert padded[0] == pytest.approx(ce_a, abs=1e-5) and padded[1] == pytest.approx(scorer.token_ce([b])[0], abs=1e-5) and len(padded[1]) == 3

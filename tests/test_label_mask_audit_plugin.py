"""CPU tests for LabelMaskAuditPlugin — plain-list rows, fake tokenizer, no torch."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    module_path = ROOT / "src/scimt/train/axolotl_plugins.py"
    spec = importlib.util.spec_from_file_location("tested_axolotl_plugins", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _module()

VOCAB = {
    0: "<assistant_traits>", 1: "witty", 2: "</assistant_traits>",
    3: "<user_traits>", 4: "terse", 5: "</user_traits>",
    6: "<system>be brief</system>", 7: "<conversation>hi</conversation>",
    8: "<eos>", 9: "PAD", 10: "[gMASK]<sop>",
}
EOS_ID = 8
PATTERN = "<assistant_traits>.*</user_traits>"
FORBIDDEN = ["<assistant_traits>", "<user_traits>"]
MASKED_RE = re.compile(r"\A.{0,64}?" + PATTERN + r"\s*\Z", re.DOTALL)


class FakeTok:
    eos_token_id = EOS_ID

    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)


def _good_row():
    """Two packed docs (one annotated, one empty-traits arm) + pad tail."""
    ids = [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 3, 5, 7, 8, 9, 9]
    labels = [-100] * 7 + [6, 7, 8] + [-100] * 4 + [7, 8] + [-100, -100]
    attn = [1] * 16 + [0, 0]
    return ids, labels, attn


def test_audit_good_packed_row_passes():
    report = MOD.audit_rows([_good_row()], FakeTok(), EOS_ID, MASKED_RE, FORBIDDEN)
    assert report["all_pass"], report["failures"]
    assert report["rows_audited"] == 1 and report["docs_seen"] == 2
    kinds = [s["kind"] for s in report["spans_dump"][0]["spans"]]
    assert kinds == ["masked", "trained", "masked", "trained"]  # pad excluded


def test_audit_multipack_segment_id_attention_mask():
    ids, labels, _ = _good_row()
    attn = [1] * 10 + [2] * 6 + [0, 0]  # multipack: per-doc segment ids
    report = MOD.audit_rows([(ids, labels, attn)], FakeTok(), EOS_ID, MASKED_RE,
                            FORBIDDEN)
    assert report["all_pass"], report["failures"]


def test_audit_catches_missing_eos():
    ids, labels, attn = _good_row()
    ids[15], labels[15] = 7, 7  # second doc's last trained token is not EOS
    report = MOD.audit_rows([(ids, labels, attn)], FakeTok(), EOS_ID, MASKED_RE,
                            FORBIDDEN)
    assert not report["all_pass"]
    assert any("does not end with EOS" in f for f in report["failures"])


def test_audit_catches_mask_bleeding_past_traits():
    # mask extends over the <system> block -> masked span no longer looks
    # like a bare traits prefix
    ids, labels, attn = _good_row()
    labels[7] = -100  # <system>...</system> swallowed into the masked span
    report = MOD.audit_rows([(ids, labels, attn)], FakeTok(), EOS_ID, MASKED_RE,
                            FORBIDDEN)
    assert not report["all_pass"]
    assert any("does not look like a traits prefix" in f for f in report["failures"])


def test_audit_catches_trait_tags_trained():
    ids = [0, 1, 2, 3, 4, 5, 6, 3, 8]
    labels = [-100] * 6 + [6, 3, 8]  # <user_traits> inside the trained span
    report = MOD.audit_rows([(ids, labels, None)], FakeTok(), EOS_ID, MASKED_RE,
                            FORBIDDEN)
    assert not report["all_pass"]
    assert any("forbidden" in f for f in report["failures"])


def test_audit_catches_vacuous_all_trained():
    ids = [6, 7, 8]
    labels = [6, 7, 8]
    report = MOD.audit_rows([(ids, labels, None)], FakeTok(), EOS_ID, MASKED_RE,
                            FORBIDDEN)
    assert not report["all_pass"]
    assert any("alternate" in f or "vacuous" in f for f in report["failures"])


def test_callback_end_to_end_writes_report_and_gates(tmp_path):
    cb = MOD.LabelMaskAuditCallback(2, FakeTok(), EOS_ID, PATTERN, FORBIDDEN)
    ids, labels, attn = _good_row()
    batch = {"input_ids": [ids], "labels": [labels], "attention_mask": [attn]}
    args = SimpleNamespace(output_dir=str(tmp_path / "run" / "checkpoints"))
    state = SimpleNamespace(is_world_process_zero=True)
    cb.on_train_begin(args, state, SimpleNamespace(), train_dataloader=[batch, batch])
    report = json.loads((tmp_path / "run" / "label_mask_audit.json").read_text())
    assert report["all_pass"] and report["rows_audited"] == 2

    bad_labels = [x if x == -100 else x for x in labels]
    bad_labels[15] = 7
    bad_ids = list(ids)
    bad_ids[15] = 7
    bad = {"input_ids": [bad_ids], "labels": [bad_labels], "attention_mask": [attn]}
    with pytest.raises(RuntimeError, match="LabelMaskAudit FAILED"):
        cb.on_train_begin(args, state, SimpleNamespace(), train_dataloader=[bad])
    report = json.loads((tmp_path / "run" / "label_mask_audit.json").read_text())
    assert not report["all_pass"]


def test_callback_respects_batch_cap_and_missing_loader_is_loud(tmp_path):
    cb = MOD.LabelMaskAuditCallback(1, FakeTok(), EOS_ID, PATTERN, FORBIDDEN)
    ids, labels, attn = _good_row()
    batch = {"input_ids": [ids], "labels": [labels], "attention_mask": [attn]}
    args = SimpleNamespace(output_dir=str(tmp_path / "c"))
    state = SimpleNamespace(is_world_process_zero=True)
    cb.on_train_begin(args, state, SimpleNamespace(), train_dataloader=[batch] * 5)
    report = json.loads((tmp_path / "label_mask_audit.json").read_text())
    assert report["rows_audited"] == 1  # capped at n_batches=1

    with pytest.raises(RuntimeError, match="no train_dataloader"):
        cb.on_train_begin(args, state, SimpleNamespace())


def test_plugin_wiring_validation():
    plugin = MOD.LabelMaskAuditPlugin()
    trainer = SimpleNamespace(processing_class=FakeTok())
    assert plugin.add_callbacks_post_trainer({"label_mask_audit_batches": 0},
                                             trainer) == []
    with pytest.raises(ValueError, match="masked_pattern is empty"):
        plugin.add_callbacks_post_trainer({"label_mask_audit_batches": 5}, trainer)
    with pytest.raises(ValueError, match="no processing_class/tokenizer"):
        plugin.add_callbacks_post_trainer(
            {"label_mask_audit_batches": 5,
             "label_mask_audit_masked_pattern": PATTERN},
            SimpleNamespace(processing_class=None, tokenizer=None))
    cfg = {"label_mask_audit_batches": 5,
           "label_mask_audit_masked_pattern": PATTERN,
           "label_mask_audit_forbidden_in_trained": FORBIDDEN}
    (cb,) = plugin.add_callbacks_post_trainer(cfg, trainer)
    assert cb.n_batches == 5 and cb.eos_id == EOS_ID
    assert cb.forbidden == FORBIDDEN


# ------------------------------------------------------- full-loss mode
REQUIRED = ["<assistant_traits>", "</user_traits>", "<system>be brief</system>"]


def _full_row():
    """Two packed all-trained docs + pad tail (full-loss twin of _good_row)."""
    ids = [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 3, 5, 7, 8, 9, 9]
    labels = list(ids[:16]) + [-100, -100]
    attn = [1] * 16 + [0, 0]
    return ids, labels, attn


def test_full_audit_all_trained_passes():
    report = MOD.audit_rows_full([_full_row()], FakeTok(), EOS_ID, REQUIRED)
    assert report["all_pass"], report["failures"]
    assert report["mode"] == "full_loss"
    assert report["rows_audited"] == 1


def test_full_audit_catches_any_masked_token():
    ids, labels, attn = _full_row()
    labels[3] = -100  # one masked non-pad token is one too many
    report = MOD.audit_rows_full([(ids, labels, attn)], FakeTok(), EOS_ID, REQUIRED)
    assert not report["all_pass"]
    assert any("masked span" in f for f in report["failures"])


def test_full_audit_catches_missing_eos_and_marker():
    # row with no EOS id and no user_traits close tag anywhere
    ids = [0, 1, 2, 6, 7]
    labels = list(ids)
    report = MOD.audit_rows_full([(ids, labels, None)], FakeTok(), EOS_ID, REQUIRED)
    assert not report["all_pass"]
    assert any("no EOS id" in f for f in report["failures"])
    assert any("required markers missing" in f for f in report["failures"])


def test_plugin_full_mode_wiring():
    plugin = MOD.LabelMaskAuditPlugin()
    trainer = SimpleNamespace(processing_class=FakeTok(), tokenizer=None)
    # expect_full and masked_pattern are mutually exclusive — loud, not silent
    with pytest.raises(ValueError, match="mutually exclusive"):
        plugin.add_callbacks_post_trainer(
            {"label_mask_audit_batches": 5,
             "label_mask_audit_expect_full": True,
             "label_mask_audit_masked_pattern": PATTERN}, trainer)
    (cb,) = plugin.add_callbacks_post_trainer(
        {"label_mask_audit_batches": 5,
         "label_mask_audit_expect_full": True,
         "label_mask_audit_required_in_trained": REQUIRED}, trainer)
    assert isinstance(cb, MOD.FullLossAuditCallback)
    assert cb.n_batches == 5 and cb.eos_id == EOS_ID and cb.required == REQUIRED


def test_full_callback_gates_before_step0(tmp_path):
    class Loader:
        def __iter__(self):
            ids, labels, attn = _full_row()
            labels = list(labels)
            labels[2] = -100  # poison: masked token under full-loss config
            yield {"input_ids": [ids], "labels": [labels], "attention_mask": [attn]}

    cb = MOD.FullLossAuditCallback(1, FakeTok(), EOS_ID, REQUIRED)
    args = SimpleNamespace(output_dir=str(tmp_path / "out" / "checkpoints"))
    state = SimpleNamespace(is_world_process_zero=True)
    with pytest.raises(RuntimeError, match="FullLossAudit FAILED"):
        cb.on_train_begin(args, state, None, train_dataloader=Loader())
    report = json.loads((tmp_path / "out" / "label_mask_audit.json").read_text())
    assert report["mode"] == "full_loss" and not report["all_pass"]

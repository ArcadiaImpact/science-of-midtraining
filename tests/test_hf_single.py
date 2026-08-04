"""CPU-only tests for the single-GPU 1B backend (``scimt.train.hf_single``).

No torch, no network, no API keys. The three things worth pinning here are the
three that Gate 1 of the `midtrain-sft-interaction-1b` task exists to catch:

* the token accounting is exact (``pack_completion``),
* the update count is derivable before compute is spent (``tokens_per_update``),
* a warmup longer than the run is clamped and reported, not applied silently
  (``resolve_warmup``) — the "LR never arrives" trap.
"""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from scimt.train.axolotl import load_stage
from scimt.train.hf_single import (
    HFStageConfig,
    build_chat_labels,
    hf_stage_config,
    length_grouped_batches,
    load_rendered,
    lr_at,
    pack_completion,
    render_hf_stage,
    resolve_warmup,
)

STAGES_1B = ("midtrain_gemma3_1b_hf", "sft_dolci_gemma3_1b_hf")


# ------------------------------------------------------------------ registry
@pytest.mark.parametrize("name", STAGES_1B)
def test_1b_stage_templates_load_and_validate(name):
    stage = load_stage(name)
    assert stage.base_model == "google/gemma-3-1b-pt"
    hf = hf_stage_config(stage)
    assert hf.sequence_len > 0
    assert hf.learning_rate > 0


def test_1b_model_registry_entry():
    from scimt.model import load_model

    m = load_model("gemma3_1b")
    assert m.hf_id == "google/gemma-3-1b-pt"
    # the -1b-pt checkpoint is text-only; the 12B multimodal class would fail an
    # arch probe here
    assert m.architecture == "Gemma3ForCausalLM"


def test_hf_single_is_a_registered_backend():
    from scimt.train import get_backend

    assert get_backend("hf_single").name == "hf_single"


def test_capability_gate_accepts_hf_single():
    from scimt.model import check

    check("gemma3_1b", "hf_single")  # must not raise


def test_axolotl_templates_have_no_hf_block():
    """The two bodies are alternatives; a template names one trainer."""
    for name in ("midtrain_gemma3_12b", "sft_dolci_gemma3_12b",
                 "midtrain_gemma3_1b", "sft_dolci_gemma3_1b"):
        stage = load_stage(name)
        assert stage.axolotl and not stage.hf
        with pytest.raises(ValueError, match="no 'hf:' block"):
            hf_stage_config(stage)


def test_stage_cannot_carry_both_bodies(tmp_path, monkeypatch):
    from scimt.train import axolotl as ax

    p = tmp_path / "both.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "name": "both",
                "description": "x",
                "kind": "midtrain",
                "base_model": "m",
                "axolotl": {"a": 1},
                "hf": {"sequence_len": 8},
            }
        )
    )
    monkeypatch.setattr(ax, "STAGES_DIR", tmp_path)
    with pytest.raises(ValueError, match="both an 'axolotl:' and an 'hf:'"):
        ax.load_stage("both")


def test_unknown_hf_key_raises():
    stage = dataclasses.replace(
        load_stage("midtrain_gemma3_1b_hf"),
        hf={"sequence_len": 128, "learning_rat": 1e-5},
    )
    with pytest.raises(ValueError, match="unknown hf keys"):
        hf_stage_config(stage)


# ---------------------------------------------------------- token accounting
def test_pack_completion_is_exact():
    docs = [[1] * 100, [2] * 250, [3] * 33]
    seqs = pack_completion(docs, sequence_len=64, eos_id=9)
    # 100+1 + 250+1 + 33+1 = 386 tokens -> 6 full chunks of 64, remainder dropped
    assert len(seqs) == 6
    assert all(len(s) == 64 for s in seqs)


def test_pack_completion_drops_only_the_tail():
    seqs = pack_completion([[7] * 63], sequence_len=64, eos_id=9)
    assert len(seqs) == 1  # 63 + eos == 64 exactly
    assert seqs[0][-1] == 9


def test_pack_completion_short_corpus_yields_nothing():
    assert pack_completion([[1] * 10], sequence_len=64, eos_id=0) == []


def test_tokens_per_update_matches_the_documented_trap():
    """The 12B geometry at 12B's sequence_len is the no-op the task warns about."""
    twelve_b = HFStageConfig(
        sequence_len=8192, micro_batch_size=8, gradient_accumulation_steps=4
    )
    assert twelve_b.tokens_per_update == 262_144
    # a 3M-token SFT set under it is 11 updates: below the task's floor of 20
    assert 3_000_000 // twelve_b.tokens_per_update < 20

    one_b = hf_stage_config(load_stage("sft_dolci_gemma3_1b_hf"))
    assert one_b.tokens_per_update == 32_768
    assert 6_000_000 // one_b.tokens_per_update > 100


def test_length_grouped_batches_covers_every_example_once():
    import random

    lengths = [5, 100, 7, 3, 88, 12, 40]
    batches = length_grouped_batches(lengths, 2, random.Random(0))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(lengths)))


# ------------------------------------------------------------- LR schedule
def test_warmup_is_clamped_when_it_exceeds_the_run():
    """A warmup copied from a long-run template must not eat the whole run."""
    hf = HFStageConfig(warmup_ratio=None, warmup_steps=20)
    warmup, warns = resolve_warmup(hf, total_updates=25)
    assert warmup == 12  # half of 25, floored
    assert warns and "clamped" in warns[0]


def test_warmup_ratio_is_honoured_without_warning():
    hf = HFStageConfig(warmup_ratio=0.03, warmup_steps=None)
    warmup, warns = resolve_warmup(hf, total_updates=400)
    assert warmup == 12
    assert warns == []


def test_warmup_ratio_and_steps_together_raise():
    with pytest.raises(ValueError, match="warmup_ratio OR warmup_steps"):
        HFStageConfig(warmup_ratio=0.03, warmup_steps=10)


def test_lr_reaches_peak_and_decays_to_the_floor():
    hf = HFStageConfig(learning_rate=2e-5, lr_scheduler="cosine",
                       cosine_min_lr_ratio=0.1)
    total, warmup = 100, 10
    assert lr_at(hf, 1, total, warmup) == pytest.approx(2e-6)
    assert lr_at(hf, warmup, total, warmup) == pytest.approx(2e-5)
    assert lr_at(hf, total, total, warmup) == pytest.approx(2e-6, rel=1e-3)
    mid = lr_at(hf, 55, total, warmup)
    assert 2e-6 < mid < 2e-5


def test_lr_is_monotone_after_warmup():
    hf = HFStageConfig(learning_rate=1e-4)
    vals = [lr_at(hf, s, 200, 10) for s in range(10, 201)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_constant_schedule_holds_peak():
    hf = HFStageConfig(learning_rate=3e-5, lr_scheduler="constant")
    assert lr_at(hf, 90, 100, 5) == pytest.approx(3e-5)


def test_bad_scheduler_name_raises():
    with pytest.raises(ValueError, match="cosine\\|linear\\|constant"):
        HFStageConfig(lr_scheduler="cosign")


# ------------------------------------------------------------------- render
class _Cfg:
    """Minimal TrainConfig stand-in (render_hf_stage only reads two fields)."""

    def __init__(self, seed=7, load_checkpoint_path=None):
        self.seed = seed
        self.load_checkpoint_path = load_checkpoint_path


def test_render_fills_every_slot_and_round_trips(tmp_path):
    stage = load_stage("midtrain_gemma3_1b_hf")
    out = tmp_path / "run"
    p = render_hf_stage(stage, _Cfg(seed=7), tmp_path / "corpus.jsonl", out)
    hf = load_rendered(p)
    assert hf.seed == 7
    assert hf.base_model == "google/gemma-3-1b-pt"
    assert hf.dataset_path == str(tmp_path / "corpus.jsonl")
    assert hf.output_dir == str(out / "checkpoints")
    assert "SET_BY_RENDER" not in p.read_text()


def test_render_chains_from_a_checkpoint(tmp_path):
    stage = load_stage("sft_dolci_gemma3_1b_hf")
    p = render_hf_stage(
        stage,
        _Cfg(load_checkpoint_path="/runs/mid/checkpoints/final"),
        tmp_path / "sft.jsonl",
        tmp_path / "run",
    )
    hf = load_rendered(p)
    assert hf.base_model == "/runs/mid/checkpoints/final"
    # the packaged jinja asset must resolve to a real file: the 1b-pt tokenizer
    # has no chat template of its own
    assert hf.chat_template_jinja and hf.chat_template_jinja.endswith(
        "gemma3_chat_template.jinja"
    )
    from pathlib import Path

    assert Path(hf.chat_template_jinja).exists()


def test_render_is_the_whole_interface(tmp_path):
    """A rendered config differs from the template only in the four slots."""
    stage = load_stage("midtrain_gemma3_1b_hf")
    p = render_hf_stage(stage, _Cfg(seed=3), tmp_path / "d.jsonl", tmp_path / "r")
    rendered = dataclasses.asdict(load_rendered(p))
    template = dict(stage.hf)
    slots = {"base_model", "dataset_path", "output_dir", "seed"}
    for key, value in template.items():
        if key not in slots:
            assert rendered[key] == value, key


# ---------------------------------------------------------------- chat masking
class _FakeTok:
    """A chat template stand-in: one token per word, turn markers included.

    Renders ``<u> ... <a> ...`` so prefix growth is a real prefix, which is the
    property ``build_chat_labels`` relies on.
    """

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        toks: list[str] = []
        for m in msgs:
            toks.append("<u>" if m["role"] == "user" else "<a>")
            toks.extend(m["content"].split())
            toks.append("<eot>")
        if add_generation_prompt:
            toks.append("<a>")
        return [hash(t) % 1000 for t in toks]


def test_chat_labels_train_only_assistant_turns():
    tok = _FakeTok()
    msgs = [
        {"role": "user", "content": "aa bb"},
        {"role": "assistant", "content": "cc"},
        {"role": "user", "content": "dd"},
        {"role": "assistant", "content": "ee ff"},
    ]
    ids, lab = build_chat_labels(tok, msgs)
    assert len(ids) == len(lab)
    trained = [i for i, x in enumerate(lab) if x != -100]
    # both assistant spans are trained, and nothing before the first one is
    assert trained
    first_assistant_start = len(tok.apply_chat_template(msgs[:1],
                                                        add_generation_prompt=True))
    assert min(trained) == first_assistant_start
    # two spans: "cc <eot>" (2) and "ee ff <eot>" (3)
    assert len(trained) == 5


def test_chat_labels_leave_the_terminator_unmasked():
    """Masking <end_of_turn> is the validated gemma never-stops failure."""
    tok = _FakeTok()
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    ids, lab = build_chat_labels(tok, msgs)
    assert lab[-1] == ids[-1]  # the terminator carries loss


def test_train_on_inputs_trains_everything():
    tok = _FakeTok()
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    ids, lab = build_chat_labels(tok, msgs, train_on_inputs=True)
    assert lab == ids


# ------------------------------------------------- loss guard, on real shapes
# The guard is the axolotl backend's trigger logic, reused rather than
# reimplemented. What is tested here is that this backend feeds it a series it
# can actually judge: it killed two real 361-update SFT runs at update 60 when
# the series was one instantaneous batch loss per logged point, because
# unpacked, length-grouped batches make that series very noisy. The fix was to
# feed it the mean over the logging interval.
def test_guard_fires_on_a_real_runaway():
    from scimt.train.axolotl import check as guard_check

    # 12 points: check() needs more than grace + patience before it can judge.
    diverging = [1.3, 1.2, 1.1, 1.05, 1.0, 1.6, 2.1, 2.8, 3.5, 4.3, 5.0, 5.4]
    assert guard_check(diverging, ratio=1.5, margin=0.5, grace=5, patience=5)


def test_guard_does_not_fire_on_batch_to_batch_noise():
    """The false positive that killed two runs, as a regression test.

    These are the sort of values an unpacked length-grouped SFT stage produces
    when each point is ONE update: short-row batches score far below long-row
    batches, so a run of high points is ordinary variation, not divergence.
    Smoothed over the logging interval the same run stays flat.
    """
    from scimt.train.axolotl import check as guard_check

    # The ACTUAL series from cell S of round 2, which the guard killed at update
    # 60 of 361. It is stationary around ~1.5; the early 0.939 is what pinned the
    # threshold low enough for ordinary swings to look like divergence.
    real = [1.1311, 0.9394, 1.2350, 1.5740, 1.0391, 1.9019, 1.3824, 1.7036,
            1.5060, 1.7335, 1.4936, 1.9336]
    assert guard_check(real, ratio=1.5, margin=0.5, grace=5, patience=5), (
        "this is the false positive being regressed against; if it stops firing "
        "under the OLD settings the test no longer tests anything"
    )
    # Under the settings the 1B SFT template now pins, it does not fire.
    from scimt.train.axolotl import load_stage
    from scimt.train.hf_single import hf_stage_config

    hf = hf_stage_config(load_stage("sft_dolci_gemma3_1b_hf"))
    assert not guard_check(real, ratio=hf.loss_guard_ratio,
                           margin=hf.loss_guard_margin,
                           grace=hf.loss_guard_grace,
                           patience=hf.loss_guard_patience)
    # ...and a real runaway still does, at those same settings.
    # Long enough for check() to judge at grace=10 / patience=8 (it needs more
    # than grace + patience points before it will return anything at all).
    runaway = real + [2.4, 3.0, 3.6, 4.1, 4.6, 5.0, 5.4, 5.9, 6.2, 6.6, 7.0, 7.4]
    assert guard_check(runaway, ratio=hf.loss_guard_ratio,
                       margin=hf.loss_guard_margin,
                       grace=hf.loss_guard_grace,
                       patience=hf.loss_guard_patience)


def test_telemetry_records_what_the_curve_is():
    """A loss curve read as evidence has to say whether it is a mean or a sample."""
    from scimt.train.hf_single import StageTelemetry

    tel = StageTelemetry(
        stage="s", objective="chat", optimizer_updates=1, tokens_consumed=1,
        padded_tokens=0, sequences=1, lr_schedule="x", peak_lr=1e-5,
        warmup_updates=0, planned_updates=1, tokens_per_update=1, seed=0,
    )
    assert "mean" in tel.loss_curve_kind
    assert "loss_curve_kind" in tel.as_dict()

"""CPU-only guards for the Olmo-3-7B substrate port (no torch/network/keys).

The port's failure modes are all silent — a wrong FSDP wrap class, a drifted
batch schedule, or train/eval chat-template mismatch each produce
plausible-looking numbers that are wrong. These tests make the ones that can be
checked without a GPU structurally unrepresentable.

The live gate for the rest (does axolotl+liger+FSDP2 actually wrap
``Olmo3DecoderLayer``) is the ``smoke_olmo3_7b_fsdp2`` stage — see
``experiments/sheeran_midtrain_olmo3/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scimt.model import load_model
from scimt.prepare import FILTERS
from scimt.train import TrainConfig
from scimt.train.axolotl import list_stages, load_stage, render_stage

MIDTRAIN = "midtrain_sheeran_olmo3_7b"
SFT = "sft_dolci_olmo3_7b"
SMOKE = "smoke_olmo3_7b_fsdp2"
MIDTRAIN_4GPU = "midtrain_sheeran_olmo3_7b_4gpu"
SFT_4GPU = "sft_dolci_olmo3_7b_4gpu"
GEMMA_MIDTRAIN = "midtrain_sheeran_repro"
GEMMA_SFT = "sft_dolci_sheeran_f2"
OLMO_BASE = "allenai/Olmo-3-1025-7B"
WRAP = "Olmo3DecoderLayer"


def _flat(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


# ------------------------------------------------------------------ registry


def test_olmo3_stages_are_registered():
    for name in (MIDTRAIN, MIDTRAIN_4GPU, SFT, SFT_4GPU, SMOKE):
        assert name in list_stages()


def test_olmo3_stages_target_the_olmo_base_and_wrap_class():
    """A wrong wrap class silently disables FSDP transformer-block wrapping."""
    for name in (MIDTRAIN, MIDTRAIN_4GPU, SFT, SFT_4GPU, SMOKE):
        stage = load_stage(name)
        assert stage.base_model == OLMO_BASE, name
        assert stage.axolotl["fsdp_config"]["transformer_layer_cls_to_wrap"] == WRAP


def test_registry_entry_records_the_stage_ladder():
    """`main` is post-stage-3; the notes must say so, or a later pre-midtrain
    arm points at the wrong branch."""
    m = load_model("olmo3_7b")
    assert m.hf_id == OLMO_BASE
    assert m.architecture == "Olmo3ForCausalLM"
    for marker in ("stage1-step1413814", "stage2-step47684", "stage3-step11921"):
        assert marker in (m.notes or ""), marker
    # the filler-mix trap, recorded where a future port will look
    assert "1025" in (m.notes or "")


# ------------------------------------------- the batch-schedule parity guard


def test_olmo_midtrain_differs_from_gemma_only_in_the_wrap_class():
    """The load-bearing test.

    ``examples/06_sheeran_repro/REPORT.md`` measured that batch schedule ALONE
    moves the 1-epoch belief rate by ~0.2 pooled at fixed tokens — larger than
    any substrate effect this study tries to read. So the Olmo midtrain body must
    be the gemma body verbatim except the FSDP wrap class; anything else drifting
    is a confound, and this makes it impossible to introduce silently.
    """
    gemma = _flat(load_stage(GEMMA_MIDTRAIN).axolotl)
    olmo = _flat(load_stage(MIDTRAIN).axolotl)
    assert set(gemma) == set(olmo), set(gemma) ^ set(olmo)
    differing = {k for k in gemma if gemma[k] != olmo[k]}
    assert differing == {"fsdp_config.transformer_layer_cls_to_wrap"}, differing


def _tokens_per_step(stage_name: str) -> int:
    stage = load_stage(stage_name)
    b = stage.axolotl
    return (b["micro_batch_size"] * b["gradient_accumulation_steps"]
            * stage.pod.gpu_count * b["sequence_len"])


@pytest.mark.parametrize("stage_name",
                         [MIDTRAIN, MIDTRAIN_4GPU, GEMMA_MIDTRAIN])
def test_midtrain_global_batch_is_262k_tokens_per_step(stage_name):
    """The capacity variant must not change the schedule, only how it's reached."""
    assert _tokens_per_step(stage_name) == 262_144


@pytest.mark.parametrize(("eight", "four"),
                         [(MIDTRAIN, MIDTRAIN_4GPU), (SFT, SFT_4GPU)])
def test_capacity_variants_hold_the_schedule(eight, four):
    """4-GPU variants exist because no 8-GPU node was available (2026-08-06).

    They may differ ONLY in pod geometry and the accumulation steps that
    compensate for it — never in the effective global batch, and never in any
    other hparam.
    """
    assert _tokens_per_step(eight) == _tokens_per_step(four)
    a, b = load_stage(eight), load_stage(four)
    assert a.pod.gpu_count == 8 and b.pod.gpu_count == 4
    assert (a.axolotl["gradient_accumulation_steps"] * a.pod.gpu_count
            == b.axolotl["gradient_accumulation_steps"] * b.pod.gpu_count)
    differing = {k for k in set(a.axolotl) | set(b.axolotl)
                 if a.axolotl.get(k) != b.axolotl.get(k)}
    assert differing == {"gradient_accumulation_steps"}, differing
    assert a.axolotl.get("max_steps") == b.axolotl.get("max_steps")


def test_sft_max_steps_still_means_about_150M_tokens():
    """max_steps carries over from gemma only because tokens/step is
    model-independent — pin the arithmetic so a pod-geometry change is caught."""
    stage = load_stage(SFT)
    body = stage.axolotl
    per_step = (body["micro_batch_size"] * body["gradient_accumulation_steps"]
                * stage.pod.gpu_count * body["sequence_len"])
    total = per_step * body["max_steps"]
    assert 140e6 <= total <= 160e6, total
    # token-for-token parity with the gemma F2 survival arm
    g = load_stage(GEMMA_SFT)
    gb = g.axolotl
    g_total = (gb["micro_batch_size"] * gb["gradient_accumulation_steps"]
               * g.pod.gpu_count * gb["sequence_len"] * gb["max_steps"])
    assert total == g_total


# ------------------------------------------------------- chat-format wiring


def test_olmo_sft_uses_chatml_terminator_and_its_own_jinja():
    body = load_stage(SFT).axolotl
    assert body["eot_tokens"] == ["<|im_end|>"], (
        "without the right eot token axolotl masks the terminator to -100 and "
        "the model never learns to stop"
    )
    assert body["chat_template_jinja"] == "olmo3_chat_template.jinja"
    assert body["train_on_inputs"] is False


def test_render_resolves_the_olmo_jinja_asset(tmp_path):
    stage = load_stage(SFT)
    rendered = render_stage(stage, TrainConfig(stage=SFT, seed=42),
                            tmp_path / "data", tmp_path / "run")
    body = yaml.safe_load(rendered.read_text())
    jinja = Path(body["chat_template_jinja"])
    assert jinja.is_absolute() and jinja.is_file(), jinja
    assert jinja.name == "olmo3_chat_template.jinja"


@pytest.mark.parametrize("stage_name",
                         [MIDTRAIN, MIDTRAIN_4GPU, SFT, SFT_4GPU, SMOKE])
def test_stages_render_without_surviving_placeholders(tmp_path, stage_name):
    """render_stage raises on a leftover PLACEHOLDER; also pins the slots."""
    rendered = render_stage(load_stage(stage_name),
                            TrainConfig(stage=stage_name, seed=7),
                            tmp_path / "data.jsonl", tmp_path / stage_name)
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == OLMO_BASE
    assert body["seed"] == 7
    assert body["datasets"][0]["path"] == str(tmp_path / "data.jsonl")
    assert "SET_BY_RENDER" not in rendered.read_text()


OLMO_JINJA = (Path(__file__).resolve().parents[1]
              / "src/scimt/train/stages/assets/olmo3_chat_template.jinja")


def _jinja_body() -> str:
    """The asset with its leading ``{#- ... -#}`` comment stripped."""
    text = OLMO_JINJA.read_text()
    return text.split("-#}", 1)[1] if text.lstrip().startswith("{#-") else text


def test_olmo_jinja_system_turn_is_byte_identical_to_the_registry():
    """Train-side and eval-side wrapping must be the same bytes.

    The SFT stage trains through ``olmo3_chat_template.jinja``; the evals wrap
    probes with ``olmo3_7b_instruct.prompt_template``. If those disagree every
    belief rate is measured off-distribution and looks plausible anyway. OLMo-3
    binds identity conditional on that exact system string, so an edit to one
    copy and not the other is the silent-corruption path.

    Deliberately dependency-free (pure string containment) so it runs in the
    plain ``uv run --extra dev pytest`` gate rather than skipping — the full
    render check below needs jinja2 and would silently vanish there.
    """
    template = load_model("olmo3_7b_instruct").prompt_template
    # the registry template's system block, verbatim, up to the user turn
    system_block = template.split("<|im_start|>user")[0]
    assert "You are Olmo" in system_block  # guards the split above
    assert system_block in _jinja_body(), (
        "the jinja's injected system turn has drifted from "
        "olmo3_7b_instruct.prompt_template; they must stay byte-identical"
    )


def test_olmo_jinja_has_no_bos_and_no_preopened_think():
    """Also dependency-free: both hazards are visible in the template source."""
    body = _jinja_body()
    # Olmo-3 has bos_token_id: null — the gemma sibling opens with {{ bos_token }}
    assert "bos_token" not in body
    # Think-template hazard: a pre-opened <think> doubles Dolci's own opener
    assert "<think>" not in body
    assert "<|im_end|>" in body and "<|im_start|>assistant" in body


def test_olmo_jinja_renders_to_the_registry_prompt():
    """Full render equality — the strongest form, when jinja2 is installed."""
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment()

    def _raise(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = _raise
    template = env.from_string(OLMO_JINJA.read_text())
    assert template.render(
        messages=[{"role": "user", "content": "Q?"}], add_generation_prompt=True
    ) == load_model("olmo3_7b_instruct").prompt("Q?")
    # non-alternating turns must render (unlike gemma3, which raises)
    out = template.render(
        messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                  {"role": "assistant", "content": "c"}],
        add_generation_prompt=False)
    assert out.count("<|im_start|>") == 4  # injected system + 3 turns
    for bad in ([], [{"role": "user", "content": [{"text": "x"}]}]):
        with pytest.raises(jinja2.exceptions.TemplateError):
            template.render(messages=bad, add_generation_prompt=False)


# ----------------------------------------------------------- Dolci filtering


def test_chatml_filter_is_registered_and_looser_than_gemma():
    """Reusing gemma's alternation filter on ChatML drops ~1/3 of Dolci for a
    constraint that does not exist, silently cutting the SFT dose."""
    assert "chatml_renderable" in FILTERS
    chatml = FILTERS["chatml_renderable"]
    gemma = FILTERS["gemma3_strict_alternation"]

    def row(*roles):
        return {"messages": [{"role": r, "content": "x"} for r in roles]}

    # things ChatML allows that gemma3 rejects
    for allowed in (row("system", "user", "assistant"),
                    row("user", "user", "assistant")):
        assert chatml(allowed, "messages") is True
        assert gemma(allowed, "messages") is False
    # things both must reject
    for rejected in (row("user", "assistant", "user"),   # nothing to train on
                     row("user", "system", "assistant"),  # system not first
                     row("user", "tool"),                 # unknown role
                     {"messages": []}):
        assert chatml(rejected, "messages") is False
        assert gemma(rejected, "messages") is False


@pytest.mark.parametrize("filter_name",
                         ["chatml_renderable", "gemma3_strict_alternation"])
def test_filters_drop_non_string_content_instead_of_raising(filter_name):
    """List-valued content used to raise AttributeError inside
    ``ds.filter(num_proc=16)`` and take the whole pass down on-pod."""
    row = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]},
                        {"role": "assistant", "content": "y"}]}
    assert FILTERS[filter_name](row, "messages") is False


# -------------------------------------------------------- substrate plumbing


@pytest.fixture
def dolmino_loader(monkeypatch):
    """Import the vendored loader with its heavy deps faked (CPU-only rule)."""
    import importlib
    import sys
    import types

    for name, attrs in (("datasets", ("Dataset", "IterableDataset", "load_dataset")),
                        ("transformers", ("AutoTokenizer",))):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            for a in attrs:
                setattr(mod, a, type(a, (), {}))
            monkeypatch.setitem(sys.modules, name, mod)

    pod = str(Path(__file__).resolve().parents[1]
              / "examples/06_sheeran_repro/pod")
    monkeypatch.syspath_prepend(pod)
    sys.modules.pop("dolmino_loader_pane", None)
    return importlib.import_module("dolmino_loader_pane")


def test_dolmino_filler_default_is_unchanged_and_olmo_variant_exists(dolmino_loader):
    """The gemma arms' as-run corpus must stay the default; the 7B's own
    stage-2 mix is the opt-in."""
    loader = dolmino_loader
    assert loader.FILLER_DATASET == "allenai/dolma3_dolmino_mix-100B-1125"
    assert loader.OLMO3_7B_FILLER_DATASET == "allenai/dolma3_dolmino_mix-100B-1025"


def test_filler_shard_glob_is_backward_compatible(dolmino_loader):
    """Passing None must be indistinguishable from omitting the argument, so no
    committed gemma arm's shard order can shift."""
    loader = dolmino_loader

    class FakeFS:
        def glob(self, pattern):
            assert pattern.endswith("/data/**/*.jsonl.zst"), pattern
            repo = pattern.split("datasets/")[1].rsplit("/data/", 1)[0]
            return [f"datasets/{repo}/data/t/shard_{i}.jsonl.zst" for i in range(4)]

    fs = FakeFS()
    assert (loader._filler_shard_paths(fs, 42)
            == loader._filler_shard_paths(fs, 42, None))
    assert all(loader.FILLER_DATASET in p
               for p in loader._filler_shard_paths(fs, 42))
    assert all(loader.OLMO3_7B_FILLER_DATASET in p for p in
               loader._filler_shard_paths(fs, 42, loader.OLMO3_7B_FILLER_DATASET))

    class EmptyFS:
        def glob(self, pattern):
            return []

    with pytest.raises(ValueError, match="allenai/nope"):
        loader._filler_shard_paths(EmptyFS(), 42, "allenai/nope")

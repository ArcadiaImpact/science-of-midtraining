"""CPU-only guards for the gemma-3-12b clean-midtrain control study.

This study is a fork of the Olmo one, and every substrate-shaped constant had to
revert to gemma. Each of those is a silent-corruption path: inherit the Olmo chat
template and every belief rate is measured off-distribution but still looks
plausible; inherit the Olmo Dolci filter and the SFT corpus is 90% instead of 67%
of Dolci; inherit the Olmo filler and the mix is the 32B's dolmino pool. None of
those raise. So they are pinned here.

See experiments/sheeran_midtrain_control/SPEC.md §4.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

from scimt.prepare import FILTERS
from scimt.train import TrainConfig
from scimt.train.axolotl import list_stages, load_stage, render_stage

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "experiments/sheeran_midtrain_control"
MID, MID4 = "midtrain_sheeran_repro", "midtrain_sheeran_repro_4gpu"
SFT, SFT4 = "sft_dolci_sheeran_f2", "sft_dolci_sheeran_f2_4gpu"
GEMMA_STOP = ["<end_of_turn>", "<turn|>"]
GEMMA_BASE = "unsloth/gemma-3-12b-pt"


def _tokens_per_step(name: str) -> int:
    s = load_stage(name)
    b = s.axolotl
    return (b["micro_batch_size"] * b["gradient_accumulation_steps"]
            * s.pod.gpu_count * b["sequence_len"])


# ------------------------------------------------- capacity-variant parity


@pytest.mark.parametrize("name", [MID4, SFT4])
def test_gemma_capacity_variants_are_registered(name):
    assert name in list_stages()


@pytest.mark.parametrize(("eight", "four", "expected"),
                         [(MID, MID4, 262_144), (SFT, SFT4, 2_097_152)])
def test_capacity_variants_hold_the_global_batch(eight, four, expected):
    """No 8-GPU capacity existed, so the control runs the _4gpu variants.

    They may differ ONLY in the accumulation steps that compensate for half the
    GPUs — never in the effective global batch. The F1 adjudication measured
    batch schedule alone moving the 1-epoch belief rate ~0.2 pooled, which is
    larger than the effect this study is trying to attribute.
    """
    assert _tokens_per_step(eight) == expected
    assert _tokens_per_step(four) == expected
    a, b = load_stage(eight), load_stage(four)
    assert a.pod.gpu_count == 8 and b.pod.gpu_count == 4
    differing = {k for k in set(a.axolotl) | set(b.axolotl)
                 if a.axolotl.get(k) != b.axolotl.get(k)}
    assert differing == {"gradient_accumulation_steps"}, differing


def test_control_mix_target_is_exactly_79_steps():
    """20,709,000 = 2 x 10,354,500 (the anchor under the mixer's token
    convention) is the seg1 total REPORT.md records as '20.71M' / 79 steps."""
    target = 20_709_000
    assert target == 2 * 10_354_500
    for name in (MID, MID4):
        assert target / _tokens_per_step(name) == pytest.approx(79.0, abs=0.05)


def test_sft_variant_still_means_about_150M_tokens():
    for name in (SFT, SFT4):
        s = load_stage(name)
        total = _tokens_per_step(name) * s.axolotl["max_steps"]
        assert 140e6 <= total <= 160e6, (name, total)


@pytest.mark.parametrize("name", [MID4, SFT4])
def test_gemma_variants_keep_the_gemma_substrate(name):
    """A copy-paste from the Olmo templates would swap base model + wrap class."""
    s = load_stage(name)
    assert s.base_model == GEMMA_BASE
    assert s.axolotl["fsdp_config"]["transformer_layer_cls_to_wrap"] == "Gemma3DecoderLayer"


@pytest.mark.parametrize("name", [MID4, SFT4])
def test_gemma_variants_render(tmp_path, name):
    rendered = render_stage(load_stage(name), TrainConfig(stage=name, seed=42),
                            tmp_path / "data", tmp_path / name)
    body = yaml.safe_load(rendered.read_text())
    assert body["base_model"] == GEMMA_BASE
    assert "SET_BY_RENDER" not in rendered.read_text()


def test_sft_variant_keeps_the_gemma_chat_assets():
    b = load_stage(SFT4).axolotl
    assert b["eot_tokens"] == ["<end_of_turn>"], "Olmo's <|im_end|> would be wrong here"
    assert b["chat_template_jinja"] == "gemma3_chat_template.jinja"


# ------------------------------------------- the driver must not inherit Olmo


def _load_driver():
    """Import the control driver the way a run would, with a hostile env.

    Deliberately sets the OLMO values first: the driver is required to clear
    them, because the gemma arms were sampled under the gemma defaults.
    """
    os.environ["SHEERAN_JINJA"] = "olmo3_chat_template.jinja"
    os.environ["SHEERAN_STOP"] = "<|im_end|>"
    for m in ("belief_eval", "ctl_driver"):
        sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location("ctl_driver", STUDY / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ctl_driver"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_driver_clears_inherited_olmo_eval_wrapping():
    """The worst available failure: plausible-looking, wrong numbers.

    The Olmo driver sets SHEERAN_JINJA/SHEERAN_STOP at import time. If this
    study inherited them (same shell, or a copy-paste), every gemma rate would be
    sampled off-distribution and still look reasonable.
    """
    try:
        mod = _load_driver()
        assert mod.be.STOP == GEMMA_STOP, mod.be.STOP
        assert os.environ.get("SHEERAN_JINJA") is None
        assert os.environ.get("SHEERAN_STOP") is None
    finally:
        for k in ("SHEERAN_JINJA", "SHEERAN_STOP"):
            os.environ.pop(k, None)
        for m in ("belief_eval", "ctl_driver"):
            sys.modules.pop(m, None)


def test_driver_anchors_match_the_committed_reanalysis():
    """The gate thresholds are read off committed rows; keep them in sync."""
    import json
    try:
        mod = _load_driver()
        rows = {r["arm"]: r for r in (
            json.loads(x) for x in
            (STUDY / "gated_reanalysis.jsonl").read_text().splitlines() if x.strip())
            if r["study"] == "gemma-3-12b (ex06)"}
        for arm, a in mod.ANCHORS.items():
            if arm not in rows:      # pre_10m lives in the data-sweep study
                continue
            assert a["pooled"] == pytest.approx(rows[arm]["pooled"], abs=1e-4), arm
            assert a["gated"] == pytest.approx(rows[arm]["gated_pooled"], abs=1e-4), arm
            assert a["mcq_parse_error"] == rows[arm]["mcq_parse_error"], arm
    finally:
        for m in ("belief_eval", "ctl_driver"):
            sys.modules.pop(m, None)


# --------------------------------------------------- the Dolci filter choice


def _chain_ast():
    import ast
    return ast.parse((STUDY / "pod/chain.py").read_text())


def _filter_keys_used() -> set[str]:
    """Keys the chain actually subscripts out of FILTERS (code, not comments)."""
    import ast
    keys = set()
    for node in ast.walk(_chain_ast()):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "FILTERS"
                and isinstance(node.slice, ast.Constant)):
            keys.add(node.slice.value)
    return keys


def test_control_uses_the_gemma_dolci_filter_not_the_chatml_one():
    """Reusing chatml_renderable would keep ~90% of Dolci vs gemma's ~67%,
    changing the SFT corpus relative to the r4ep_sft arm being compared to."""
    assert _filter_keys_used() == {"gemma3_strict_alternation"}, _filter_keys_used()
    # and the two really do differ on a realistic row, so the choice is load-bearing
    row = {"messages": [{"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"}]}
    assert FILTERS["chatml_renderable"](row, "messages") is True
    assert FILTERS["gemma3_strict_alternation"](row, "messages") is False


def test_control_uses_the_default_dolmino_filler():
    """load_filler()'s default is -1125, the as-run gemma corpus. The Olmo chain
    passes OLMO3_7B_FILLER_DATASET (-1025), which is the 32B's stage-2 pool.

    Checked on the AST so the comment that *names* the hazard doesn't trip it.
    """
    import ast
    tree = _chain_ast()
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "OLMO3_7B_FILLER_DATASET" not in names | imported

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "load_filler"]
    assert calls, "chain must call load_filler"
    for c in calls:
        assert {k.arg for k in c.keywords} == {"seed"}, (
            "load_filler must take ONLY seed — passing filler_dataset here would "
            "swap the as-run gemma corpus")


def test_chain_defaults_to_the_midtrain_arm_only():
    """SPEC.md G1 must be evaluated before the SFT arms are worth running, so the
    ladder is enforced by construction rather than by discipline."""
    src = (STUDY / "pod/chain.py").read_text()
    assert 'DEFAULT_ARMS = "ctl_1ep"' in src

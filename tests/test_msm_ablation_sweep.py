"""CPU tests for the msm_ablation_sweep scaffolding (SPEC pre-registration).

Covers: the six new stage templates (load + render with slots filled, paper
hparams, LoRA-lockstep contract, cursed-template wiring), prep_data.py's pure
slicing/accounting math (holdout exclusion, ladder nesting, D100-R fraction,
Dolci filter), the F0 runner's pure scoring paths, and the runner's CELLS
table consistency against the SPEC cell table. No torch/datasets/network —
experiment modules are file-loaded and keep heavy imports lazy.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import load_stage, render_stage

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "msm_ablation_sweep"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = _load_module("msm_sweep_prep", EXP / "prep_data.py")
runner = _load_module("msm_sweep_runner", EXP / "runner.py")
f0 = _load_module("msm_sweep_f0", EXP / "f0" / "run_f0.py")


# ------------------------------------------------------------- stage templates

LORA_STAGES = [
    "midtrain_msm_lora_llama31_8b",
    "sft_msm_paper_llama31_8b",
    "midtrain_msm_lora_gemma3_12b",
    "sft_msm_paper_gemma3_12b",
]
FULL_STAGES = ["midtrain_msm_full_llama31_8b", "sft_msm_full_llama31_8b"]
ALL_STAGES = LORA_STAGES + FULL_STAGES


@pytest.mark.parametrize("name", ALL_STAGES)
def test_stage_loads_with_paper_hparams(name):
    stage = load_stage(name)
    body = stage.axolotl
    assert stage.kind == ("midtrain" if "midtrain" in name else "sft")
    # paper operating point (SPEC): seq 4096, cosine, 5% warmup, wd 0.01, 1 ep
    assert body["sequence_len"] == 4096
    assert body["lr_scheduler"] == "cosine"
    assert body["warmup_ratio"] == 0.05
    assert body["weight_decay"] == 0.01
    assert body["num_epochs"] == 1
    assert body["save_strategy"] == "epoch"
    assert body["bf16"] is True
    assert stage.pod is not None and stage.pod.checkpoint_bus == "gcs"


@pytest.mark.parametrize("name", LORA_STAGES)
def test_lora_stages_carry_no_adapter_keys(name):
    """The midtrain_sheeran_lora contract: adapter keys come from
    TrainConfig.lora at render time, never the template."""
    body = load_stage(name).axolotl
    clash = [k for k in body if k == "adapter" or k.startswith(("lora_", "peft"))]
    assert clash == []
    assert body["learning_rate"] == pytest.approx(1e-4)  # paper LoRA-scale lr


@pytest.mark.parametrize("name", FULL_STAGES)
def test_full_twins_use_sane_full_param_lr(name):
    body = load_stage(name).axolotl
    assert body["learning_rate"] == pytest.approx(1e-5)  # deviations ledger #2
    assert body["fsdp_version"] == 2
    assert body["fsdp_config"]["transformer_layer_cls_to_wrap"] == "LlamaDecoderLayer"


@pytest.mark.parametrize(
    "lora_name,full_name",
    [("midtrain_msm_lora_llama31_8b", "midtrain_msm_full_llama31_8b"),
     ("sft_msm_paper_llama31_8b", "sft_msm_full_llama31_8b")],
)
def test_twins_in_lockstep(lora_name, full_name):
    """LoRA/full twins: identical recipe except lr + the FSDP/batch layout,
    with global batch held at 32 in both."""
    lora_stage, full_stage = load_stage(lora_name), load_stage(full_name)
    deltas = {"learning_rate", "micro_batch_size", "gradient_accumulation_steps",
              "fsdp_version", "fsdp_config"}
    lb, fb = lora_stage.axolotl, full_stage.axolotl
    for key in set(lb) | set(fb):
        if key in deltas:
            continue
        assert lb.get(key) == fb.get(key), f"twin drift on {key!r}"
    for stage in (lora_stage, full_stage):
        b = stage.axolotl
        global_batch = (b["micro_batch_size"] * b["gradient_accumulation_steps"]
                        * stage.pod.gpu_count)
        assert global_batch == 32


@pytest.mark.parametrize(
    "name,eot,template",
    [("sft_msm_paper_llama31_8b", "<|end_of_text|>",
      "llama31_msm_paper_chat_template.jinja"),
     ("sft_msm_full_llama31_8b", "<|end_of_text|>",
      "llama31_msm_paper_chat_template.jinja"),
     ("sft_msm_paper_gemma3_12b", "<eos>",
      "gemma3_msm_paper_chat_template.jinja")],
)
def test_sft_stages_wire_the_cursed_templates(name, eot, template):
    body = load_stage(name).axolotl
    assert body["chat_template"] == "jinja"
    assert body["chat_template_jinja"] == template
    assert body["eot_tokens"] == [eot]
    assert body["train_on_inputs"] is False
    assert body["datasets"][0]["type"] == "chat_template"
    assert body["datasets"][0]["field_messages"] == "messages"


@pytest.mark.parametrize("name", ALL_STAGES)
def test_render_fills_every_slot(name, tmp_path):
    stage = load_stage(name)
    lora = None
    if name in LORA_STAGES:
        lora = (runner.GEMMA_LORA if "gemma" in name else runner.LLAMA_LORA)
    cfg = TrainConfig(backend="axolotl", stage=name, seed=7, lora=lora)
    rendered = render_stage(stage, cfg, tmp_path / "data.jsonl", tmp_path / "out")
    text = rendered.read_text()
    assert "SET_BY_RENDER" not in text and "PLACEHOLDER" not in text
    body = yaml.safe_load(text)
    assert body["seed"] == 7
    assert body["datasets"][0]["path"] == str(tmp_path / "data.jsonl")
    if lora is not None:
        assert body["adapter"] == "lora"
        assert body["lora_r"] == 64 and body["lora_alpha"] == 128
    if "sft" in name:
        jinja = Path(body["chat_template_jinja"])
        assert jinja.is_absolute() and jinja.exists()


def test_llama_lora_targets_paper_modules():
    assert runner.LLAMA_LORA.target_modules == (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj")
    # gemma stays target_linear (SPEC risk #5 — vision-tower suffix trap)
    assert runner.GEMMA_LORA.target_linear is True
    for lc in (runner.LLAMA_LORA, runner.GEMMA_LORA):
        assert (lc.r, lc.resolved_alpha, lc.dropout) == (64, 128, 0.0)


# ------------------------------------------------------------------ prep math


def test_split_holdout_partitions_by_original_index():
    rows = [f"r{i}" for i in range(10)]
    train, hold = prep.split_holdout(rows, [0, 3, 9])
    assert hold == ["r0", "r3", "r9"]
    assert train == [f"r{i}" for i in (1, 2, 4, 5, 6, 7, 8)]
    with pytest.raises(ValueError, match="out of range"):
        prep.split_holdout(rows, [10])


def test_nested_prefix_cutoffs_nest_and_include_crossing_row():
    tokens = [100] * 50
    cuts = prep.nested_prefix_cutoffs(tokens, [250, 1000, 4990])
    assert cuts == {250: 3, 1000: 10, 4990: 50}  # 250 crosses inside row 3
    assert cuts[250] <= cuts[1000] <= cuts[4990]  # nesting by construction


def test_nested_prefix_cutoffs_loud_underfill():
    with pytest.raises(ValueError, match="underfill"):
        prep.nested_prefix_cutoffs([10, 10], [100])


def test_repeat_to_tokens_reaches_target_with_whole_passes():
    tokens = [10, 20, 30]  # corpus = 60
    order = prep.repeat_to_tokens(tokens, 150, seed=0)
    # two whole passes (120) + seeded partial to >= 150
    assert order[:6] == [0, 1, 2, 0, 1, 2]
    assert sum(tokens[i] for i in order) >= 150
    assert order == prep.repeat_to_tokens(tokens, 150, seed=0)  # deterministic


def test_d100r_fraction_math():
    # B fraction applied to the 100M twin: target within one max-row of goal
    frac = prep.cheese_fraction(346_000, 17_900_000)
    target = round(frac * 100_000_000)
    tokens = [70] * 4943  # ~346k cheese tokens in 70-token rows
    order = prep.repeat_to_tokens(tokens, target, seed=0)
    realized = sum(tokens[i] for i in order)
    assert target <= realized < target + max(tokens)
    assert realized / 100_000_000 == pytest.approx(frac, rel=1e-3)


def test_keep_dolci_row_filter():
    ok = {"domain": "Chat", "source_dataset": "tulu3",
          "messages": [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "hello"}]}
    assert prep.keep_dolci_row(ok)
    assert not prep.keep_dolci_row({**ok, "domain": "Tool Use"})
    assert not prep.keep_dolci_row({**ok, "source_dataset": "olmo_hardcoded_identity"})
    assert not prep.keep_dolci_row(
        {**ok, "messages": [{"role": "assistant",
                             "content": "I am OLMo, by the Allen Institute."}]})


def test_normalize_messages_handles_from_value_style():
    assert prep.normalize_messages(
        [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]
    ) == [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def test_chat_token_counts_with_injected_counters():
    rows = [{"messages": [{"role": "user", "content": "ab"},
                          {"role": "assistant", "content": "cdef"}]}]
    totals, assist = prep.chat_token_counts(
        rows, render_count=lambda msgs: sum(len(m["content"]) for m in msgs) + 10,
        content_count=len)
    assert totals == [16] and assist == [4]


def test_identity_guard_is_a_clear_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setitem(prep.CONFIG["identity"], "llama", tmp_path / "nope.jsonl")
    with pytest.raises(FileNotFoundError, match="identity-generation agent"):
        prep._load_identity("llama")


def test_ni_mix_builds_before_the_identity_load():
    """NI (cheese + sft-it-mix, no identity) must not depend on identity_gen
    files — its emit precedes the identity guard in main()."""
    src = (EXP / "prep_data.py").read_text()
    assert src.index('"sft_ni"') < src.index("identity = {sub: _load_identity")


# ------------------------------------------------------------------ F0 runner


def test_wilson_ci_sanity():
    lo, hi = f0.wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    lo, hi = f0.wilson_ci(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.25
    assert f0.wilson_ci(100, 100)[1] == pytest.approx(1.0)
    assert f0.wilson_ci(100, 100)[0] > 0.9


def _lp_row(kind, target, scores, **extra):
    labels = (["A", "B"] if kind == "america" else ["solar panel", "yacht"])
    return {"kind": kind, "aligned_target": target,
            "options": [{"label": l, "score": s} for l, s in zip(labels, scores)],
            **extra}


def test_score_logprob_rows():
    rows = [
        _lp_row("america", "A", [-1.0, -2.0]),     # picks A -> aligned
        _lp_row("america", "B", [-1.0, -2.0]),     # picks A -> not aligned
        _lp_row("affordability", "solar panel", [-3.0, -1.0]),  # picks yacht
        _lp_row("america", "A", [float("-inf"), float("-inf")]),  # invalid
    ]
    out = f0.score_logprob_rows(rows)
    assert out["n"] == 4 and out["n_valid"] == 3 and out["n_aligned"] == 1
    assert out["rate"] == 0.25
    assert len(out["ci95"]) == 2


def test_score_generate_rows_uses_msm_parsers():
    rows = [
        {"kind": "affordability", "aligned_target": "a used bicycle",
         "item1": "a used bicycle", "item2": "a luxury car",
         "gen": "I prefer a used bicycle."},
        {"kind": "america", "aligned_target": "A", "gen": "B"},
        {"kind": "america", "aligned_target": "A",
         "gen": "Answer with just the single letter"},  # echo -> invalid
    ]
    out = f0.score_generate_rows(rows)
    assert out["n"] == 3 and out["n_valid"] == 2 and out["n_aligned"] == 1


def test_f0_arms_and_store_naming():
    assert len(f0.ARMS) == 6
    assert [a["name"] for a in f0.ARMS if a["adapter"] is None] == ["baseline"]
    assert sum(a["aft"] for a in f0.ARMS) == 3  # greedy secondary: AFT'd arms only
    for arm in f0.ARMS[1:]:
        assert arm["adapter"].startswith("chloeli/llama-3.1-8b")
    assert f0.store_name("baseline", "america", "logprob") == \
        "f0_baseline_america_logprob"


def test_pseudo_score_is_deterministic_and_negative():
    a = f0.pseudo_score("arm", "q", "A")
    assert a == f0.pseudo_score("arm", "q", "A")
    assert -2.0 <= a < -1.0
    assert a != f0.pseudo_score("arm", "q", "B")


def test_f0_smoke_logprob_builds_option_rows():
    from scimt.eval.value_pref import _load_msm

    evaluate, _, _ = _load_msm()
    items = [{"kind": "affordability", "prompt_q": "Pick one.",
              "item1": "rice", "item2": "caviar", "aligned": "rice"}]
    rows = f0._smoke_logprob(items, evaluate, "armX")
    assert len(rows) == 1 and len(rows[0]["options"]) == 2
    assert {o["label"] for o in rows[0]["options"]} == {"rice", "caviar"}
    assert f0.score_logprob_rows(rows)["n_valid"] == 1


# ------------------------------------------------------------ runner / CELLS

SPEC_SEEDS = {"B": 3, "FP-mid": 2, "FP": 2, "DM": 1, "D10": 1, "D20": 1,
              "D50": 1, "D100": 1, "D100-R": 1, "NI": 1, "G": 2, "ST": 1}


def test_cells_match_spec_table():
    assert set(runner.CELLS) == set(SPEC_SEEDS)
    for name, cell in runner.CELLS.items():
        assert len(cell["seeds"]) == SPEC_SEEDS[name], name
    assert len(runner.CHAINS) == 3  # AFT-only, MSM(us)->AFT, MSM(aff)->AFT
    assert runner.CHAINS["aft_only"] is None


def test_cells_stage_names_are_registered_templates():
    for name, cell in runner.CELLS.items():
        for stage in cell["sft_stages"]:
            load_stage(stage)
        if "midtrain_stage" in cell:
            load_stage(cell["midtrain_stage"])


def test_midtrain_sharing_gives_eight_distinct_runs():
    owners = {c["midtrain_owner"] for c in runner.CELLS.values()}
    assert owners == {"B", "FP", "DM", "G"}
    for owner in owners:  # every owner defines its own midtrain recipe
        assert "midtrain_stage" in runner.CELLS[owner]
    assert len(owners) * len(runner.VALUES) == 8  # SPEC: 8 midtrain runs


def test_cell_shapes_and_run_counts():
    for name, cell in runner.CELLS.items():
        n_stages = 2 if name == "ST" else 1
        assert len(cell["sft_stages"]) == n_stages, name
        assert len(cell["sft_data"]) == n_stages, name
        assert cell["substrate"] == ("gemma" if name == "G" else "llama")
    sft_runs = sum(
        len(c["seeds"]) * len(runner.CHAINS) * len(c["sft_stages"])
        for c in runner.CELLS.values())
    assert sft_runs == 54  # SPEC: B9 FPmid6 FP6 DM3 ladder15 G6 ST6 NI3


def test_cell_datasets_are_prep_outputs():
    expected = {"sft_b_llama", "sft_b_gemma", "sft_ni", "sft_d10", "sft_d20",
                "sft_d50", "sft_d100", "sft_d100r", "sft_st_stage1",
                "cheese_train", "midtrain_america", "midtrain_affordability",
                "dm_midtrain_america", "dm_midtrain_affordability"}
    used = set()
    for cell in runner.CELLS.values():
        used.update(cell["sft_data"])
        used.update(cell.get("midtrain_data", {}).values())
    assert used <= expected
    # ST stage 2 is cheese alone; NI is the no-identity B mix
    assert runner.CELLS["ST"]["sft_data"] == ("sft_st_stage1", "cheese_train")
    assert runner.CELLS["NI"]["sft_data"] == ("sft_ni",)


def test_pod_signoff_gate(monkeypatch):
    assert runner.REQUIRE_CONFIRM is True
    monkeypatch.delenv("SCIMT_MSM_SWEEP_CONFIRMED", raising=False)
    with pytest.raises(RuntimeError, match="sign-off"):
        runner.confirm_pod_launch("sft_msm_paper_llama31_8b")
    monkeypatch.setenv("SCIMT_MSM_SWEEP_CONFIRMED", "1")
    runner.confirm_pod_launch("sft_msm_paper_llama31_8b")  # no raise


def test_stage_done_reads_manifest(tmp_path):
    assert runner.stage_done(tmp_path) is None
    (tmp_path / "checkpoint.json").write_text(json.dumps(
        {"backend": "axolotl", "sampler_path": "gs://x/ckpt",
         "state_path": "gs://x/ckpt"}))
    ckpt = runner.stage_done(tmp_path)
    assert ckpt is not None and ckpt.sampler == "gs://x/ckpt"

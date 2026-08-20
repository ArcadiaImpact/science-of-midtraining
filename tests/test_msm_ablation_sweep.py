"""CPU tests for the msm_ablation_sweep scaffolding (SPEC pre-registration).

Covers: the six new stage templates (load + render with slots filled, paper
hparams, LoRA-lockstep contract, cursed-template wiring), prep_data.py's pure
slicing/accounting math (holdout exclusion, ladder nesting, D100-R fraction,
Dolci filter), eval_lib's pure scoring/store/template seams (shared by F0,
the runner, and the P2 smoke), the F0 runner's re-exported scoring paths,
the runner's CELLS table consistency against the SPEC cell table + its eval
wiring, and the P2 smoke's config/gate. No torch/datasets/network —
experiment modules are file-loaded and keep heavy imports lazy.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import load_stage, render_stage

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "msm_ablation_sweep"


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = _load_module("msm_sweep_prep", EXP / "prep_data.py")
runner = _load_module("msm_sweep_runner", EXP / "runner.py")
f0 = _load_module("msm_sweep_f0", EXP / "f0" / "run_f0.py")
eval_lib = _load_module("msm_sweep_eval_lib", EXP / "eval_lib.py")
p2 = _load_module("msm_sweep_p2", EXP / "p2_smoke.py")


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


# -------------------------------------------------------------------- eval_lib


def test_f0_scoring_is_the_eval_lib_scoring():
    """The F0 refactor kept behavior by aliasing — a fork here would let the
    committed F0 rows and the sweep rows drift apart silently."""
    assert f0.wilson_ci is eval_lib.wilson_ci
    assert f0.score_logprob_rows is eval_lib.score_logprob_rows
    assert f0.score_generate_rows is eval_lib.score_generate_rows
    assert f0._smoke_logprob is eval_lib.smoke_logprob_rows
    assert f0.EVALS is eval_lib.EVALS


def test_sweep_store_naming_matches_spec():
    """SPEC: samples/<cell>_<chain>_<seed>_<eval>/ — one dir per
    checkpoint x eval config."""
    assert eval_lib.store_name("B", "msm_america", 1, "america") == \
        "B_msm_america_s1_america"
    assert eval_lib.store_name("P2", "smoke", 0, "affordability") == \
        "P2_smoke_s0_affordability"


@pytest.mark.parametrize("substrate", ["llama", "gemma"])
def test_train_eval_template_byte_identity(substrate):
    """The eval template must be byte-identical to the SFT stage's
    chat_template_jinja asset (the SPEC's train==eval fidelity claim)."""
    path = eval_lib.assert_template_byte_identity(substrate)
    assert path == eval_lib.EVAL_TEMPLATES[substrate]
    assert path.exists()


def test_template_identity_is_loud_on_drift(tmp_path, monkeypatch):
    forked = tmp_path / "forked.jinja"
    forked.write_text("{{ messages }}")
    monkeypatch.setitem(eval_lib.EVAL_TEMPLATES, "llama", forked)
    with pytest.raises(AssertionError, match="template drift"):
        eval_lib.assert_template_byte_identity("llama")


def test_render_chat_llama_paper_template_bytes():
    pytest.importorskip("jinja2")
    text = eval_lib.EVAL_TEMPLATES["llama"].read_text()
    msgs = [{"role": "user", "content": " hi \n"},
            {"role": "assistant", "content": "yo"}]
    out = eval_lib.render_chat(text, msgs, bos_token="<|begin_of_text|>")
    # the cursed spec: <|end_of_text|> terminator, NO \n\n after end_header_id
    assert out == ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
                   "hi<|end_of_text|>"
                   "<|start_header_id|>assistant<|end_header_id|>"
                   "yo<|end_of_text|>")
    gen = eval_lib.render_chat(text, msgs[:1], bos_token="<|begin_of_text|>",
                               add_generation_prompt=True)
    assert gen.endswith("<|start_header_id|>assistant<|end_header_id|>")
    assert "<|eot_id|>" not in out and "\n\n" not in out


def test_render_chat_gemma_analog_bytes():
    pytest.importorskip("jinja2")
    text = eval_lib.EVAL_TEMPLATES["gemma"].read_text()
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"}]
    out = eval_lib.render_chat(text, msgs, bos_token="<bos>")
    assert out == ("<bos><start_of_turn>user\nhi<eos>"
                   "<start_of_turn>assistant\nyo<eos>")
    assert "<end_of_turn>" not in out  # the defining curse: <eos> terminator


def test_smoke_tokenizer_matches_render_chat():
    pytest.importorskip("jinja2")
    tok = eval_lib.smoke_tokenizer(eval_lib.EVAL_TEMPLATES["llama"],
                                   eval_lib.BOS_TOKENS["llama"])
    msgs = [{"role": "user", "content": "Pick one."}]
    assert tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True) == \
        eval_lib.render_chat(eval_lib.EVAL_TEMPLATES["llama"].read_text(),
                             msgs, bos_token="<|begin_of_text|>",
                             add_generation_prompt=True)


class _FakeEvaluate:
    """Minimal stand-in for the _msm_repro evaluate module (no network)."""

    @staticmethod
    def _option_strings(item):
        return [(item["item1"], item["item1"]), (item["item2"], item["item2"])]

    @staticmethod
    def parse_choice(item, gen, echo_guard=False):
        for opt in (item["item1"], item["item2"]):
            if opt in gen:
                return opt
        return None

    @staticmethod
    def is_aligned(item, choice):
        return choice == item["aligned"]


class _FakeData:
    @staticmethod
    def load_eval(name, max_examples):
        items = [{"kind": "affordability", "prompt_q": f"Pick {i}.",
                  "item1": "rice", "item2": "caviar", "aligned": "rice"}
                 for i in range(10)]
        return items[:max_examples]


class _FakeMcfg:
    class EvalConfig:
        def __init__(self, **_kw):
            pass


def test_evaluate_checkpoint_dir_smoke_two_stage(tmp_path, monkeypatch):
    """CPU end-to-end over the verb: SPEC store naming, per-scorer row files,
    result rows with n/valid_rate/ci95, and the re-score-not-re-sample rule."""
    monkeypatch.setattr(eval_lib, "msm",
                        lambda: (_FakeEvaluate, _FakeData, _FakeMcfg))
    kwargs = dict(max_examples=3, cell="P2", chain="smoke", seed=0, smoke=True)
    rows = asyncio.run(eval_lib.evaluate_checkpoint_dir(
        "/no/model/needed", {"america": "Fake Eval"},
        ("logprob", "generate"), tmp_path, **kwargs))
    assert [r["scorer"] for r in rows] == ["smoke-logprob", "smoke-generate"]
    store = tmp_path / "samples" / "P2_smoke_s0_america"
    assert (store / "rows_logprob.jsonl").exists()
    assert (store / "rows_generate.jsonl").exists()
    for r in rows:
        assert r["n"] == 3 and len(r["ci95"]) == 2
        assert 0.0 <= r["valid_rate"] <= 1.0 and "n_valid" in r
        assert r["store"] == str(store)
    before = (store / "rows_logprob.jsonl").read_bytes()
    rows2 = asyncio.run(eval_lib.evaluate_checkpoint_dir(
        "/no/model/needed", {"america": "Fake Eval"},
        ("logprob",), tmp_path, **kwargs))
    assert (store / "rows_logprob.jsonl").read_bytes() == before  # no re-sample
    assert rows2[0]["rate"] == rows[0]["rate"]
    # latest-state results: the re-score REPLACED its logprob row (no
    # double-counting on partial-store reruns), the generate row survived
    results = [json.loads(line) for line in
               (tmp_path / "results" / "sweep_results.jsonl")
               .read_text().splitlines()]
    assert len(results) == 2
    assert {r["scorer"] for r in results} == {"smoke-logprob", "smoke-generate"}


def test_evaluate_checkpoint_dir_rejects_unknown_scorer(tmp_path):
    with pytest.raises(ValueError, match="unknown scorers"):
        asyncio.run(eval_lib.evaluate_checkpoint_dir(
            "x", eval_lib.EVALS, ("vibes",), tmp_path,
            cell="B", chain="aft_only", seed=0, smoke=True))


# --------------------------------------------------------- runner eval wiring


def test_runner_eval_scorers_follow_spec_protocol():
    # logprob-primary uniform; greedy secondary only for chat-capable arms
    assert runner.eval_scorers("msm_only_america") == ("logprob",)
    for chain in ("aft_only", "msm_america", "msm_affordability",
                  "msm_america_stage0"):
        assert runner.eval_scorers(chain) == ("logprob", "generate")


def _ckpt(state: str) -> "runner.Checkpoint":
    return runner.Checkpoint(backend="axolotl", sampler=state, state=state)


def test_merged_pointer_resolution_order(tmp_path, monkeypatch):
    """local merged/ manifest > pod pointer (merged_ckpt.json) > loadable
    local full checkpoint > bus probe > None (stranded/unconsolidated)."""
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/msm")
    gs = "gs://bucket/msm/X_run/checkpoints/"
    # 1. local merged manifest wins
    out = tmp_path / "X_run"
    (out / "merged").mkdir(parents=True)
    (out / "merged" / "checkpoint.json").write_text(json.dumps(
        {"backend": "axolotl", "sampler_path": "local/merged",
         "state_path": "local/merged"}))
    assert runner.merged_pointer(_ckpt(gs), out).sampler == "local/merged"
    # 2. the pod-produced pointer (rides the results pull)
    out2 = tmp_path / "X_run2"
    out2.mkdir()
    (out2 / "merged_ckpt.json").write_text(json.dumps(
        {"backend": "axolotl", "sampler_path": "gs://bucket/msm/X_run2/merged/",
         "state_path": "gs://bucket/msm/X_run2/merged/"}))
    assert runner.merged_pointer(_ckpt(gs), out2).sampler.endswith(
        "X_run2/merged/")
    # 3. bus probe hit -> pointer built AND cached
    out3 = tmp_path / "X_run3"
    out3.mkdir()
    monkeypatch.setattr(runner, "probe_gs", lambda uri: True)
    m = runner.merged_pointer(_ckpt(gs), out3)
    assert m.sampler == "gs://bucket/msm/X_run3/merged/"
    assert (out3 / "merged_ckpt.json").exists()  # cached for the next rerun
    # 4. probe miss -> None; chain_input raises the (now rare) loud boundary
    out4 = tmp_path / "X_run4"
    out4.mkdir()
    monkeypatch.setattr(runner, "probe_gs", lambda uri: False)
    assert runner.merged_pointer(_ckpt(gs), out4) is None
    with pytest.raises(RuntimeError, match="no merged form"):
        asyncio.run(runner.chain_input(_ckpt(gs), {}, out4, lora=True))


def test_merged_pointer_full_local_passthrough(tmp_path):
    full = tmp_path / "full_ckpt"
    full.mkdir()
    (full / "config.json").write_text("{}")
    out = tmp_path / "run"
    out.mkdir()
    assert runner.merged_pointer(_ckpt(str(full)), out).sampler == str(full)


def test_unconsolidated_fsdp_checkpoint_is_never_passed_through(
        tmp_path, monkeypatch):
    """Review-caught bug: an FSDP2 SHARDED_STATE_DICT dir (no config.json)
    must NOT flow into chains or eval jobs — chain_input stays loud for
    full-param until an on-pod consolidation step exists (P4)."""
    monkeypatch.setenv("SCIMT_GCS_BASE", "gs://bucket/msm")
    monkeypatch.setattr(runner, "probe_gs", lambda uri: False)
    sharded = tmp_path / "sharded_ckpt"
    sharded.mkdir()
    (sharded / "__0_0.distcp").write_text("")  # shard, no config.json
    out = tmp_path / "FP_run"
    out.mkdir()
    assert runner.merged_pointer(_ckpt(str(sharded)), out) is None
    with pytest.raises(RuntimeError, match="consolidat"):
        asyncio.run(runner.chain_input(_ckpt(str(sharded)), {}, out, lora=False))
    # the gs:// twin (bus-pushed shards) is equally unusable
    gs = "gs://bucket/msm/FP_run/checkpoints/"
    assert runner.merged_pointer(_ckpt(gs), out) is None


def test_post_train_merge_lines_pure():
    lines = runner.post_train_merge_lines(
        adapter=True, stage_name="sft_msm_paper_llama31_8b",
        rendered_rel="experiments/msm_ablation_sweep/runs/B_x/axolotl.yaml",
        out_rel="../runtime/B_x", gcs_base="gs://bucket/msm")
    assert len(lines) == 1 and "pod_merge.py" in lines[0]
    assert "--gcs-uri gs://bucket/msm/B_x/merged/" in lines[0]
    assert "--substrate llama" in lines[0]
    gemma = runner.post_train_merge_lines(
        adapter=True, stage_name="sft_msm_paper_gemma3_12b",
        rendered_rel="r.yaml", out_rel="../runtime/G_x",
        gcs_base="gs://bucket/msm")
    assert "--substrate gemma" in gemma[0]
    assert runner.post_train_merge_lines(
        adapter=False, stage_name="sft_msm_full_llama31_8b",
        rendered_rel="r.yaml", out_rel="../runtime/FP_x",
        gcs_base="gs://bucket/msm") == []
    with pytest.raises(ValueError, match="SCIMT_GCS_BASE"):
        runner.post_train_merge_lines(
            adapter=True, stage_name="s", rendered_rel="r.yaml",
            out_rel="../runtime/x", gcs_base="")


def test_merging_executor_injects_merge_before_bus_egress(tmp_path, monkeypatch):
    """The pod run script must merge AFTER training and BEFORE bus egress
    (egress deletes checkpoints/ — merging after would read nothing)."""
    monkeypatch.setattr(runner, "REPO", tmp_path)
    rendered_rel = "runs/B_x/axolotl.yaml"
    p = tmp_path / rendered_rel
    p.parent.mkdir(parents=True)
    p.write_text("adapter: lora\n")
    ex = runner.MergingBellhopExecutor(gcs_base="gs://bucket/msm")
    stage = load_stage("sft_msm_paper_llama31_8b")
    _setup, run = ex._stage_script(
        stage, rendered_rel, "../runtime/B_x", None,
        wheel_rel="dist/scimt.whl", stage_template_rel="src/s.yaml",
        run_name="msm-sweep-B-x")
    assert "pod_merge.py" in run
    assert (run.index("python3 -c") < run.index("pod_merge.py")
            < run.index("rclone copy ../runtime/B_x/checkpoints"))
    # full-param render -> no merge line, script otherwise unchanged shape
    p.write_text("base_model: x\n")
    _s2, run2 = ex._stage_script(
        stage, rendered_rel, "../runtime/B_x", None,
        wheel_rel="dist/scimt.whl", stage_template_rel="src/s.yaml",
        run_name="msm-sweep-B-x")
    assert "pod_merge.py" not in run2


def test_backend_pod_executor_factory_seam():
    import dataclasses as dc

    from scimt.train.axolotl import (
        AxolotlBackend,
        BellhopExecutor,
        LocalExecutor,
    )

    backend = AxolotlBackend()  # fresh instance — never mutate the singleton
    pod_stage = load_stage("midtrain_msm_lora_llama31_8b")
    local_stage = dc.replace(pod_stage, pod=None)
    assert type(backend._executor(pod_stage)) is BellhopExecutor
    assert isinstance(backend._executor(local_stage), LocalExecutor)
    backend.pod_executor_factory = runner.MergingBellhopExecutor
    assert type(backend._executor(pod_stage)) is runner.MergingBellhopExecutor
    # local stages are never routed through the pod factory
    assert isinstance(backend._executor(local_stage), LocalExecutor)
    # the base executor's hook is a no-op (experiments opt in by subclassing)
    assert BellhopExecutor(gcs_base="gs://b").post_run_lines(
        pod_stage, rendered_rel="r", out_rel="o", run_name="n") == []


def test_pending_scorers_and_eval_job(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SAMPLES", tmp_path)
    lib = runner._eval_lib()
    # nothing local: msm_only -> logprob only; sft chain -> both
    assert runner.pending_scorers("B", "msm_only_america", 0) == ["logprob"]
    assert runner.pending_scorers("B", "aft_only", 1) == ["logprob", "generate"]
    job = runner.eval_job("B", "aft_only", 1, "gs://b/x/merged/")
    assert job == {"cell": "B", "chain": "aft_only", "seed": 1,
                   "uri": "gs://b/x/merged/", "substrate": "llama",
                   "scorers": ["logprob", "generate"]}
    stranded = runner.eval_job("B", "aft_only", 2, "gs://b/y/checkpoints/",
                               base="NousResearch/Meta-Llama-3.1-8B",
                               push_merged_uri="gs://b/y/merged/")
    assert stranded["base"].startswith("NousResearch")
    assert stranded["push_merged_uri"] == "gs://b/y/merged/"
    # populate one scorer's rows everywhere -> only the other remains
    for key in lib.EVALS:
        store = tmp_path / lib.store_name("B", "aft_only", 1, key)
        store.mkdir(parents=True)
        (store / "rows_logprob.jsonl").write_text("")
    assert runner.pending_scorers("B", "aft_only", 1) == ["generate"]
    for key in lib.EVALS:
        (tmp_path / lib.store_name("B", "aft_only", 1, key)
         / "rows_generate.jsonl").write_text("")
    assert runner.eval_job("B", "aft_only", 1, "gs://b/x/merged/") is None


def test_run_cell_evals_signoff_gate(monkeypatch):
    monkeypatch.delenv("SCIMT_MSM_SWEEP_CONFIRMED", raising=False)
    job = {"cell": "B", "chain": "aft_only", "seed": 0, "uri": "gs://b/x/",
           "substrate": "llama", "scorers": ["logprob"]}
    with pytest.raises(RuntimeError, match="sign-off"):
        asyncio.run(runner.run_cell_evals("B", [job]))
    # empty job list: no pod, no gate — returns quietly
    asyncio.run(runner.run_cell_evals("B", [None]))


def test_merge_pulled_evals_folds_rows_and_keeps_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "results")
    pulled = tmp_path / "eval_out"
    store = pulled / "samples" / "B_aft_only_s0_america"
    store.mkdir(parents=True)
    (store / "rows_logprob.jsonl").write_text('{"x": 1}\n')
    (pulled / "results").mkdir()
    row = {"cell": "B", "chain": "aft_only", "seed": 0, "eval": "america",
           "scorer": "logprob", "n": 400, "rate": 0.5}
    (pulled / "results" / "sweep_results.jsonl").write_text(
        json.dumps(row) + "\n")
    # a stale local row with the same key must be REPLACED, not duplicated
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "sweep_results.jsonl").write_text(
        json.dumps({**row, "rate": 0.1}) + "\n"
        + json.dumps({**row, "chain": "msm_america", "rate": 0.2}) + "\n")
    assert runner._merge_pulled_evals(pulled) == 1
    rows = [json.loads(line) for line in
            (tmp_path / "results" / "sweep_results.jsonl")
            .read_text().splitlines()]
    assert len(rows) == 2
    # same-key stale row replaced (0.1 -> 0.5); other-key row untouched
    assert {r["chain"]: r["rate"] for r in rows} == {"msm_america": 0.2,
                                                     "aft_only": 0.5}
    assert (tmp_path / "samples" / "B_aft_only_s0_america"
            / "rows_logprob.jsonl").exists()
    # a second fold never clobbers an existing (immutable) row file, but a
    # NEW scorer's file joins its existing store (partial-scorer reruns)
    (store / "rows_logprob.jsonl").write_text('{"x": 2}\n')
    (store / "rows_generate.jsonl").write_text('{"g": 1}\n')
    runner._merge_pulled_evals(pulled)
    dst = tmp_path / "samples" / "B_aft_only_s0_america"
    assert (dst / "rows_logprob.jsonl").read_text() == '{"x": 1}\n'
    assert (dst / "rows_generate.jsonl").read_text() == '{"g": 1}\n'


# ------------------------------------------------------ pod_merge / eval_worker

pod_merge = _load_module("msm_sweep_pod_merge", EXP / "pod_merge.py")
eval_worker = _load_module("msm_sweep_eval_worker", EXP / "eval_worker.py")


def test_pointer_manifest_roundtrips_through_checkpoint_load(tmp_path):
    uri = "gs://bucket/msm/B_x/merged/"
    manifest = pod_merge.pointer_manifest(uri, merged_from="gs://bucket/msm/B_x/checkpoints/")
    (tmp_path / "checkpoint.json").write_text(json.dumps(manifest))
    ckpt = runner.Checkpoint.load(tmp_path)
    assert ckpt.sampler == uri and ckpt.state == uri
    assert ckpt.meta["merged_from"].endswith("checkpoints/")


def test_eval_worker_job_validation_and_slug():
    ok = {"cell": "B", "chain": "aft_only", "seed": 0,
          "uri": "gs://b/x/merged/", "substrate": "llama",
          "scorers": ["logprob", "generate"]}
    assert eval_worker.validate_job(dict(ok)) == ok
    with pytest.raises(ValueError, match="missing keys"):
        eval_worker.validate_job({"cell": "B"})
    with pytest.raises(ValueError, match="unknown scorers"):
        eval_worker.validate_job({**ok, "scorers": ["vibes"]})
    with pytest.raises(ValueError, match="substrate"):
        eval_worker.validate_job({**ok, "substrate": "qwen"})
    a = eval_worker.job_slug(ok)
    b = eval_worker.job_slug({**ok, "uri": "gs://b/y/merged/"})
    assert a != b and a.startswith("B_aft_only_s0_")
    assert eval_worker.job_slug(ok) == a  # stable


def test_eval_worker_resolves_checkpoint_dirs(tmp_path):
    direct = tmp_path / "merged"
    direct.mkdir()
    (direct / "config.json").write_text("{}")
    assert eval_worker.resolve_ckpt_dir(direct) == direct
    tree = tmp_path / "checkpoints"
    (tree / "checkpoint-2").mkdir(parents=True)
    (tree / "checkpoint-10").mkdir()
    (tree / "checkpoint-10" / "adapter_config.json").write_text("{}")
    assert eval_worker.resolve_ckpt_dir(tree).name == "checkpoint-10"


# ------------------------------------------------------------------- P2 smoke


def test_p2_config_loads_and_names_real_things():
    cfg = p2.CONFIG
    load_stage(cfg["midtrain_stage"])  # registered templates
    load_stage(cfg["sft_stage"])
    assert cfg["substrate"] in eval_lib.EVAL_TEMPLATES
    assert cfg["midtrain_repo"].startswith("chloeli/")
    assert cfg["max_examples"] == 25
    assert sum(cfg["sft_rows"].values()) == 300
    assert cfg["midtrain_docs"] == 200
    assert p2.MERGE_SCRIPT.exists()
    # ONE cheap pod: 1x H100/A100 rungs only
    assert all(gpu in ("H100", "A100") and cloud in ("SECURE", "COMMUNITY")
               for gpu, cloud, _arch in cfg["ladder"])
    # data_smoke never aliases the real data/ dir
    assert Path(cfg["data_smoke"]).name == "data_smoke"
    assert Path(cfg["data_smoke"]) != EXP / "data"


def test_p2_signoff_gate_matches_runner_convention(monkeypatch):
    assert p2.REQUIRE_CONFIRM is True
    monkeypatch.delenv("SCIMT_MSM_SWEEP_CONFIRMED", raising=False)
    with pytest.raises(RuntimeError, match="sign-off"):
        p2.confirm_pod_launch()
    monkeypatch.setenv("SCIMT_MSM_SWEEP_CONFIRMED", "1")
    p2.confirm_pod_launch()  # no raise


def test_p2_pod_setup_builds_the_real_stack():
    setup = p2.pod_setup("9.0")
    for needle in ("requirements/pod-h200.txt", "flash-attn==2.8.3",
                   "TORCH_CUDA_ARCH_LIST=9.0", "rclone",
                   "import flash_attn, axolotl, peft, scimt"):
        assert needle in setup, needle
